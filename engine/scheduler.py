"""Continuous Batching Scheduler for multi-tenant LLM serving."""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Deque, Dict, List, Optional

from engine.metrics import MetricsTracker
from engine.model_runner import BaseModelRunner
from engine.types import (
    ChatCompletionRequest,
    EngineStats,
    RequestState,
    TokenChunk,
)

logger = logging.getLogger("engine.scheduler")


class QueueFullException(Exception):
    """Raised when engine queue exceeds capacity."""
    pass


class ContinuousBatchingScheduler:
    """
    Manages multi-request queueing and iteration-level dynamic batching.
    
    New requests can enter the batch at every single decoding step, and
    completed requests are retired immediately, maximizing hardware utilization.
    """

    def __init__(
        self,
        runner: BaseModelRunner,
        max_batch_size: int = 16,
        max_waiting_queue_size: int = 256,
        metrics_window: int = 100,
    ):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_waiting_queue_size = max_waiting_queue_size
        
        # Request queues
        self.waiting_queue: Deque[RequestState] = collections.deque()
        self.active_batch: List[RequestState] = []
        self.all_requests: Dict[str, RequestState] = {}
        
        # Telemetry & coordination
        self.metrics = MetricsTracker(window_size=metrics_window)
        self._work_available = asyncio.Event()
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        logger.info("Loading model runner: %s (%s)...", self.runner.name, self.runner.model_id)
        await self.runner.load()
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop(), name="scheduler_loop")
        logger.info(
            "Scheduler started (max_batch_size=%d, max_queue=%d)",
            self.max_batch_size,
            self.max_waiting_queue_size,
        )

    async def stop(self) -> None:
        """Stop scheduler and clean up active tasks."""
        self._running = False
        self._work_available.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped.")

    def add_request(self, req: ChatCompletionRequest) -> RequestState:
        """Admit a new request into the waiting queue."""
        if len(self.waiting_queue) >= self.max_waiting_queue_size:
            raise QueueFullException("Engine is at peak capacity. Please retry shortly.")

        state = RequestState.from_request(req)
        self.waiting_queue.append(state)
        self.all_requests[state.request_id] = state
        self.metrics.record_request_arrival()
        
        # Signal scheduler loop that work is available
        self._work_available.set()
        return state

    def abort_request(self, request_id: str) -> None:
        """Cancel an in-flight or queued request (e.g. client disconnected)."""
        state = self.all_requests.get(request_id)
        if not state or state.is_finished:
            return

        state.is_cancelled = True
        state.finish_reason = "cancelled"
        
        # If in waiting queue, remove directly
        try:
            self.waiting_queue.remove(state)
            state.is_finished = True
            state.finished_at = time.perf_counter()
            self.metrics.record_request_finished(is_cancelled=True)
            # Notify any listeners
            state.output_queue.put_nowait(None)
            logger.info("Cancelled request %s from waiting queue.", request_id)
        except ValueError:
            # It is in active_batch; will be purged on next loop iteration
            pass

    async def _run_loop(self) -> None:
        """Continuous iteration loop executing forward steps across active batch."""
        while self._running:
            # 1. Admit queued requests up to max_batch_size
            while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
                candidate = self.waiting_queue.popleft()
                if not candidate.is_cancelled:
                    self.active_batch.append(candidate)

            # 2. If no active work, await notification
            if not self.active_batch:
                self._work_available.clear()
                # Double-check waiting queue before sleeping
                if not self.waiting_queue:
                    await self._work_available.wait()
                    continue

            # 3. Clean any cancelled requests before executing step
            active_clean = []
            for req in self.active_batch:
                if req.is_cancelled:
                    req.is_finished = True
                    req.finished_at = time.perf_counter()
                    self.metrics.record_request_finished(is_cancelled=True)
                    req.output_queue.put_nowait(None)
                else:
                    active_clean.append(req)
            self.active_batch = active_clean

            if not self.active_batch:
                continue

            # 4. Execute single model step for current batch
            try:
                step_results = await self.runner.step(self.active_batch)
            except Exception as e:
                logger.exception("Error during model runner step: %s", e)
                # Fail active batch gracefully
                for req in self.active_batch:
                    req.is_finished = True
                    chunk = TokenChunk(text=f"\n[Engine Error: {str(e)}]", is_final=True, finish_reason="error")
                    req.output_queue.put_nowait(chunk)
                    req.output_queue.put_nowait(None)
                self.active_batch.clear()
                continue

            now = time.perf_counter()
            tokens_in_step = 0

            # 5. Process generated tokens and dispatch to individual stream queues
            for req, chunk in step_results:
                if req.is_cancelled or req.is_finished:
                    continue

                # First token telemetry
                if req.first_token_at is None and chunk.text:
                    req.first_token_at = now
                    if req.ttft is not None:
                        self.metrics.record_first_token(req.ttft)

                # Inter-token latency telemetry
                itl = (now - req.last_token_at) if req.last_token_at is not None else None
                req.last_token_at = now

                if chunk.text:
                    req.generated_text += chunk.text
                    tokens_in_step += 1

                self.metrics.record_token_step(num_tokens=1 if chunk.text else 0, itl_seconds=itl)

                # Send chunk to client stream
                req.output_queue.put_nowait(chunk)

                if chunk.is_final:
                    req.is_finished = True
                    req.finish_reason = chunk.finish_reason or "stop"
                    req.finished_at = now
                    self.metrics.record_request_finished(is_cancelled=False)
                    # Send EOF sentinel
                    req.output_queue.put_nowait(None)

            # 6. Continuous Batching Retirement:
            # Sequences that finished leave the batch immediately, opening slots for waiting queue
            self.active_batch = [
                r for r in self.active_batch if not r.is_finished and not r.is_cancelled
            ]

            # Brief yield to allow event loop I/O and client responses to flush
            await asyncio.sleep(0)

    def get_stats(self) -> EngineStats:
        """Returns snapshot of engine health and telemetry."""
        return self.metrics.snapshot(
            active_requests=len(self.active_batch),
            queued_requests=len(self.waiting_queue),
            backend_name=self.runner.name,
            model_name=self.runner.model_id,
            max_batch_size=self.max_batch_size,
        )
