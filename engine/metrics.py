"""Real-time telemetry and metrics tracking for the Chat Bot Engine."""

from __future__ import annotations

import collections
import time
from typing import Deque, List, Optional
from engine.types import EngineStats


class MetricsTracker:
    """Tracks latency, throughput, and queue health metrics with rolling windows."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Counters
        self.total_requests = 0
        self.completed_requests = 0
        self.cancelled_requests = 0
        self.total_tokens = 0
        
        # Latency records (in seconds)
        self.ttft_history: Deque[float] = collections.deque(maxlen=window_size)
        self.itl_history: Deque[float] = collections.deque(maxlen=window_size)
        
        # Token throughput timestamps: (timestamp, token_count)
        self.token_events: Deque[tuple[float, int]] = collections.deque(maxlen=1000)

    def record_request_arrival(self) -> None:
        self.total_requests += 1

    def record_first_token(self, ttft_seconds: float) -> None:
        self.ttft_history.append(ttft_seconds)

    def record_token_step(self, num_tokens: int = 1, itl_seconds: Optional[float] = None) -> None:
        now = time.perf_counter()
        self.total_tokens += num_tokens
        self.token_events.append((now, num_tokens))
        if itl_seconds is not None and itl_seconds > 0:
            self.itl_history.append(itl_seconds)

    def record_request_finished(self, is_cancelled: bool = False) -> None:
        if is_cancelled:
            self.cancelled_requests += 1
        else:
            self.completed_requests += 1

    def get_current_tps(self, interval_seconds: float = 3.0) -> float:
        """Calculates token generation rate over the last N seconds."""
        now = time.perf_counter()
        cutoff = now - interval_seconds
        
        # Clean older events
        while self.token_events and self.token_events[0][0] < cutoff:
            self.token_events.popleft()
            
        recent_tokens = sum(count for _, count in self.token_events)
        return recent_tokens / interval_seconds if interval_seconds > 0 else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        if not self.ttft_history:
            return 0.0
        return (sum(self.ttft_history) / len(self.ttft_history)) * 1000.0

    @property
    def avg_itl_ms(self) -> float:
        if not self.itl_history:
            return 0.0
        return (sum(self.itl_history) / len(self.itl_history)) * 1000.0

    def snapshot(
        self,
        active_requests: int,
        queued_requests: int,
        backend_name: str,
        model_name: str,
        max_batch_size: int,
    ) -> EngineStats:
        return EngineStats(
            active_requests=active_requests,
            queued_requests=queued_requests,
            total_requests_processed=self.completed_requests + self.cancelled_requests,
            total_tokens_generated=self.total_tokens,
            current_tokens_per_second=round(self.get_current_tps(), 2),
            avg_ttft_ms=round(self.avg_ttft_ms, 2),
            avg_itl_ms=round(self.avg_itl_ms, 2),
            backend_name=backend_name,
            model_name=model_name,
            max_batch_size=max_batch_size,
        )
