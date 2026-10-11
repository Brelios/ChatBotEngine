"""Integration tests for FastAPI endpoints."""

import json
import pytest
import httpx

from api.app import create_app
from engine.model_runner import SimulatedRunner
from engine.scheduler import ContinuousBatchingScheduler


@pytest.fixture
def test_app():
    runner = SimulatedRunner(step_delay_ms=2.0)
    scheduler = ContinuousBatchingScheduler(runner=runner, max_batch_size=4)
    app = create_app(scheduler=scheduler)
    return app


@pytest.mark.asyncio
async def test_health_and_models(test_app):
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Health
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # Models
        res = await client.get("/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

        # Metrics
        res = await client.get("/v1/metrics")
        assert res.status_code == 200
        metrics = res.json()
        assert "active_requests" in metrics
        assert "avg_ttft_ms" in metrics


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(test_app):
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "messages": [{"role": "user", "content": "Hello engine"}],
            "stream": False,
            "max_tokens": 15,
        }
        res = await client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "content" in data["choices"][0]["message"]
        assert len(data["choices"][0]["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_completions_streaming(test_app):
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "messages": [{"role": "user", "content": "Tell me about streaming"}],
            "stream": True,
            "max_tokens": 20,
        }
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            chunks = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    parsed = json.loads(data_str)
                    content = parsed["choices"][0]["delta"].get("content", "")
                    if content:
                        chunks.append(content)

            assert len(chunks) > 0
            full_text = "".join(chunks)
            assert len(full_text) > 0
