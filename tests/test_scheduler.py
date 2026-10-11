"""Unit tests for Continuous Batching Scheduler."""

import asyncio
import pytest

from engine.model_runner import SimulatedRunner
from engine.scheduler import ContinuousBatchingScheduler, QueueFullException
from engine.types import ChatCompletionRequest, ChatMessage


@pytest.mark.asyncio
async def test_scheduler_lifecycle():
    runner = SimulatedRunner(step_delay_ms=1.0)
    scheduler = ContinuousBatchingScheduler(runner=runner, max_batch_size=4)
    await scheduler.start()

    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello test")],
        max_tokens=10,
    )
    state = scheduler.add_request(req)
    assert state.request_id in scheduler.all_requests

    # Collect tokens
    tokens = []
    while True:
        chunk = await asyncio.wait_for(state.output_queue.get(), timeout=2.0)
        if chunk is None:
            break
        tokens.append(chunk.text)

    await scheduler.stop()
    assert len(tokens) > 0
    assert state.is_finished
    assert state.finish_reason in ["stop", "length"]
    assert state.token_count > 0


@pytest.mark.asyncio
async def test_scheduler_queue_capacity():
    runner = SimulatedRunner(step_delay_ms=50.0)
    scheduler = ContinuousBatchingScheduler(runner=runner, max_batch_size=1, max_waiting_queue_size=2)
    # Don't start scheduler to let queue fill up
    
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="msg")],
        max_tokens=5,
    )
    
    scheduler.add_request(req)
    scheduler.add_request(req)
    
    with pytest.raises(QueueFullException):
        scheduler.add_request(req)


@pytest.mark.asyncio
async def test_scheduler_abort_request():
    runner = SimulatedRunner(step_delay_ms=20.0)
    scheduler = ContinuousBatchingScheduler(runner=runner, max_batch_size=2)
    await scheduler.start()

    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Long prompt to cancel")],
        max_tokens=100,
    )
    state = scheduler.add_request(req)

    # Wait for first token
    first_chunk = await asyncio.wait_for(state.output_queue.get(), timeout=2.0)
    assert first_chunk is not None

    # Abort
    scheduler.abort_request(state.request_id)
    assert state.is_cancelled

    # Ensure queue receives EOF
    eof_reached = False
    for _ in range(5):
        chunk = await asyncio.wait_for(state.output_queue.get(), timeout=1.0)
        if chunk is None:
            eof_reached = True
            break

    await scheduler.stop()
    assert eof_reached
