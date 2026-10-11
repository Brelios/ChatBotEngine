"""Model runner abstractions and implementations for the Chat Bot Engine."""

from __future__ import annotations

import abc
import asyncio
import random
import time
from typing import List, Optional, Tuple
from engine.types import RequestState, TokenChunk


class BaseModelRunner(abc.ABC):
    """Abstract interface for model execution engines."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the runner backend."""
        pass

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Loaded model identifier."""
        pass

    @abc.abstractmethod
    async def load(self) -> None:
        """Load model weights and tokenizer into memory/GPU."""
        pass

    @abc.abstractmethod
    async def step(self, batch: List[RequestState]) -> List[Tuple[RequestState, TokenChunk]]:
        """
        Execute one token generation step for all active sequences in the batch.
        Returns a list of (request, token_chunk) pairs.
        """
        pass


class SimulatedRunner(BaseModelRunner):
    """
    High-throughput synthetic model runner for testing continuous batching,
    concurrency, streaming SSE, and client cancellation with zero model downloads.
    """

    RESPONSES = [
        "Continuous batching is an advanced inference optimization technique where sequences are batched at the iteration level rather than request level. This drastically minimizes idle GPU bubbles and allows new requests to immediately enter active execution slots.",
        "The Chat Bot Engine coordinates asynchronous I/O with model execution steps. Server-Sent Events (SSE) push tokens into the browser client the millisecond they are decoded, delivering the iconic ChatGPT real-time typing effect.",
        "Under high load with multiple concurrent users, requests are queued in admission control. As active requests complete their generations, queued requests are admitted into the batch dynamically without restarting the forward pass.",
        "In modern LLM inference architectures like vLLM and TensorRT-LLM, KV-caches and iteration-level scheduling achieve up to 10x higher throughput compared to static batching systems.",
        "By listening to client disconnect events, the engine immediately cancels generation for closed connections, reclaiming compute resources for other users in the queue.",
    ]

    def __init__(self, model_id: str = "simulated-chat-v1", step_delay_ms: float = 25.0):
        self._model_id = model_id
        self.step_delay = step_delay_ms / 1000.0

    @property
    def name(self) -> str:
        return "Simulated"

    @property
    def model_id(self) -> str:
        return self._model_id

    async def load(self) -> None:
        # Simulate instant weight initialization
        await asyncio.sleep(0.05)

    def _prepare_sequence(self, req_state: RequestState) -> None:
        """Prepares token words sequence for a newly admitted request."""
        if req_state.generator_state is not None:
            return

        # Select response based on prompt or random choice
        prompt_lower = req_state.prompt_text.lower()
        if "batch" in prompt_lower or "engine" in prompt_lower:
            text = self.RESPONSES[0]
        elif "stream" in prompt_lower or "sse" in prompt_lower:
            text = self.RESPONSES[1]
        elif "concurren" in prompt_lower or "load" in prompt_lower:
            text = self.RESPONSES[2]
        elif "kv" in prompt_lower or "vllm" in prompt_lower:
            text = self.RESPONSES[3]
        elif "cancel" in prompt_lower or "disconnect" in prompt_lower:
            text = self.RESPONSES[4]
        else:
            text = random.choice(self.RESPONSES)

        # Tokenize into word tokens with spaces
        words = text.split(" ")
        tokens = [words[0]] + [" " + w for w in words[1:]]
        
        req_state.generator_state = {
            "tokens": tokens,
            "cursor": 0,
        }

    async def step(self, batch: List[RequestState]) -> List[Tuple[RequestState, TokenChunk]]:
        if not batch:
            return []

        # Simulate batch forward pass latency on GPU/CPU
        if self.step_delay > 0:
            await asyncio.sleep(self.step_delay)

        results: List[Tuple[RequestState, TokenChunk]] = []

        for req in batch:
            self._prepare_sequence(req)
            state = req.generator_state
            cursor = state["cursor"]
            tokens = state["tokens"]

            if cursor < len(tokens) and req.token_count < req.max_tokens:
                token_text = tokens[cursor]
                state["cursor"] += 1
                req.token_count += 1
                
                is_final = (state["cursor"] >= len(tokens)) or (req.token_count >= req.max_tokens)
                finish_reason = "length" if req.token_count >= req.max_tokens else ("stop" if is_final else None)
                
                chunk = TokenChunk(
                    text=token_text,
                    is_final=is_final,
                    finish_reason=finish_reason,
                )
                results.append((req, chunk))
            else:
                chunk = TokenChunk(text="", is_final=True, finish_reason="stop")
                results.append((req, chunk))

        return results


class HuggingFaceRunner(BaseModelRunner):
    """
    Real open-weights model runner utilizing PyTorch and HuggingFace Transformers.
    Runs on CUDA (e.g. RTX 4060) or CPU.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto"):
        self._model_id = model_id
        self._device = device
        self.model = None
        self.tokenizer = None
        self.device_used = "cpu"

    @property
    def name(self) -> str:
        return "HuggingFace"

    @property
    def model_id(self) -> str:
        return self._model_id

    async def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device == "auto":
            self.device_used = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_used = self._device

        dtype = torch.float16 if self.device_used == "cuda" else torch.float32

        # Run model loading in threadpool to keep event loop responsive
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device_used == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device_used == "cpu":
                model.to("cpu")
            model.eval()
            return tokenizer, model

        self.tokenizer, self.model = await asyncio.to_thread(_load)

    async def step(self, batch: List[RequestState]) -> List[Tuple[RequestState, TokenChunk]]:
        if not batch:
            return []

        import torch

        def _forward_step():
            results: List[Tuple[RequestState, TokenChunk]] = []

            for req in batch:
                if req.generator_state is None:
                    # Initial prefill step
                    messages = [{"role": msg.role, "content": msg.content} for msg in req.request.messages]
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device_used)
                    
                    req.generator_state = {
                        "input_ids": inputs.input_ids,
                        "past_key_values": None,
                    }

                # Next token generation step
                state = req.generator_state
                input_ids = state["input_ids"]
                past_key_values = state["past_key_values"]

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Greedy or temperature sampling
                if req.request.temperature > 0.0:
                    probs = torch.softmax(logits / req.request.temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

                token_id_int = next_token_id.item()
                token_text = self.tokenizer.decode([token_id_int], skip_special_tokens=False)

                # Update sequence state
                req.token_count += 1
                state["input_ids"] = torch.cat([input_ids, next_token_id], dim=-1)
                state["past_key_values"] = past_key_values

                is_eos = (token_id_int == self.tokenizer.eos_token_id) or (token_text == "<|im_end|>")
                is_max = req.token_count >= req.max_tokens
                is_final = is_eos or is_max
                finish_reason = "stop" if is_eos else ("length" if is_max else None)

                clean_text = "" if is_eos else token_text
                chunk = TokenChunk(
                    text=clean_text,
                    is_final=is_final,
                    finish_reason=finish_reason,
                )
                results.append((req, chunk))

            return results

        return await asyncio.to_thread(_forward_step)
