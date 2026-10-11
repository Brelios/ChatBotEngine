"""FastAPI Application Server exposing OpenAI-compatible endpoints & UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from engine.model_runner import BaseModelRunner, SimulatedRunner
from engine.scheduler import ContinuousBatchingScheduler, QueueFullException
from engine.types import ChatCompletionRequest, EngineStats

logger = logging.getLogger("api.app")

# Global scheduler holder
_scheduler: Optional[ContinuousBatchingScheduler] = None


def get_scheduler() -> ContinuousBatchingScheduler:
    if _scheduler is None:
        raise RuntimeError("Scheduler is not initialized.")
    return _scheduler


def set_scheduler(scheduler: ContinuousBatchingScheduler) -> None:
    global _scheduler
    _scheduler = scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure scheduler is running
    scheduler = get_scheduler()
    await scheduler.start()
    yield
    # Shutdown
    await scheduler.stop()


def create_app(scheduler: Optional[ContinuousBatchingScheduler] = None) -> FastAPI:
    if scheduler:
        set_scheduler(scheduler)

    app = FastAPI(
        title="Chat Bot Serving Engine",
        description="High-concurrency LLM inference engine with continuous dynamic batching and token streaming.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/v1/models")
    async def list_models():
        sched = get_scheduler()
        return {
            "object": "list",
            "data": [
                {
                    "id": sched.runner.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": sched.runner.name.lower(),
                }
            ],
        }

    @app.get("/v1/metrics", response_model=EngineStats)
    async def get_metrics():
        sched = get_scheduler()
        return sched.get_stats()

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, http_request: Request):
        sched = get_scheduler()

        try:
            state = sched.add_request(req)
        except QueueFullException as e:
            raise HTTPException(status_code=429, detail=str(e))

        # Stream mode: Server-Sent Events (SSE)
        if req.stream:
            async def sse_generator() -> AsyncGenerator[str, None]:
                created_timestamp = int(state.created_at)
                try:
                    while True:
                        # Check for client disconnect
                        if await http_request.is_disconnected():
                            logger.info("Client disconnected: aborting request %s", state.request_id)
                            sched.abort_request(state.request_id)
                            break

                        # Wait for next token from scheduler
                        try:
                            chunk = await asyncio.wait_for(state.output_queue.get(), timeout=0.2)
                        except asyncio.TimeoutError:
                            # Periodic loop to re-check client connection
                            continue

                        # End of generation
                        if chunk is None:
                            yield "data: [DONE]\n\n"
                            break

                        chunk_payload = {
                            "id": f"chatcmpl-{state.request_id}",
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": sched.runner.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.text} if chunk.text else {},
                                    "finish_reason": chunk.finish_reason if chunk.is_final else None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                except asyncio.CancelledError:
                    sched.abort_request(state.request_id)
                    raise

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming mode: wait for complete output
        try:
            while True:
                if await http_request.is_disconnected():
                    sched.abort_request(state.request_id)
                    raise HTTPException(status_code=499, detail="Client Closed Request")

                try:
                    chunk = await asyncio.wait_for(state.output_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    break

            return JSONResponse(
                content={
                    "id": f"chatcmpl-{state.request_id}",
                    "object": "chat.completion",
                    "created": int(state.created_at),
                    "model": sched.runner.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": state.generated_text,
                            },
                            "finish_reason": state.finish_reason or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(state.prompt_text.split()),
                        "completion_tokens": state.token_count,
                        "total_tokens": len(state.prompt_text.split()) + state.token_count,
                    },
                }
            )
        except asyncio.CancelledError:
            sched.abort_request(state.request_id)
            raise

    # UI Static Route
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    index_file = os.path.join(web_dir, "index.html")

    @app.get("/")
    async def serve_ui():
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Chat Bot Serving Engine is running. Visit /v1/metrics or /docs"}

    return app
