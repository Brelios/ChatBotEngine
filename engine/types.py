"""Data structures and types for the Chat Bot Engine."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role | str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stop: Optional[List[str]] = None


@dataclass
class TokenChunk:
    """Represents a single streamed token or finish signal."""
    text: str
    is_final: bool = False
    finish_reason: Optional[str] = None  # "stop", "length", "cancelled"


@dataclass
class RequestState:
    """Internal state maintained by the scheduler for each sequence."""
    request_id: str
    request: ChatCompletionRequest
    created_at: float = field(default_factory=time.perf_counter)
    
    # Execution state
    prompt_text: str = ""
    generated_text: str = ""
    token_count: int = 0
    max_tokens: int = 256
    
    # Streaming channel
    output_queue: asyncio.Queue[Optional[TokenChunk]] = field(default_factory=asyncio.Queue)
    is_cancelled: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None
    
    # Telemetry timestamps
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_token_at: Optional[float] = None
    
    # Generator iterator/state for step-by-step decoding
    generator_state: Any = None

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_at is not None:
            return self.first_token_at - self.created_at
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """Total sequence generation time in seconds."""
        if self.finished_at is not None:
            return self.finished_at - self.created_at
        return None

    @classmethod
    def from_request(cls, req: ChatCompletionRequest) -> "RequestState":
        # Format conversation messages into a prompt string
        prompt_parts = []
        for msg in req.messages:
            prompt_parts.append(f"{msg.role}: {msg.content}")
        prompt_str = "\n".join(prompt_parts) + "\nassistant: "
        
        return cls(
            request_id=str(uuid.uuid4()),
            request=req,
            prompt_text=prompt_str,
            max_tokens=req.max_tokens,
        )


class EngineStats(BaseModel):
    """Real-time engine statistics for telemetry and monitoring."""
    active_requests: int
    queued_requests: int
    total_requests_processed: int
    total_tokens_generated: int
    current_tokens_per_second: float
    avg_ttft_ms: float
    avg_itl_ms: float
    backend_name: str
    model_name: str
    max_batch_size: int
