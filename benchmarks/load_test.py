"""Asynchronous Concurrency & Throughput Benchmark Suite for Chat Bot Engine."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List, Optional
import httpx


PROMPTS = [
    "Explain how continuous batching improves LLM serving efficiency.",
    "Why does ChatGPT stream tokens using Server-Sent Events instead of waiting?",
    "How does an iteration-level scheduler dynamically admit incoming requests?",
    "Compare static batching with dynamic continuous batching.",
    "What is the impact of client disconnection detection on server capacity?",
    "Explain the difference between Time To First Token (TTFT) and Inter-Token Latency (ITL).",
    "Describe how KV cache memory management functions during autoregressive decoding.",
]


class BenchmarkStats:
    def __init__(self):
        self.ttfts: List[float] = []
        self.latencies: List[float] = []
        self.tokens: List[int] = []
        self.errors: int = 0
        self.successes: int = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens)

    @property
    def wall_clock_seconds(self) -> float:
        return max(self.end_time - self.start_time, 0.001)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.wall_clock_seconds

    @property
    def requests_per_second(self) -> float:
        return self.successes / self.wall_clock_seconds


async def worker(
    worker_id: int,
    url: str,
    queue: asyncio.Queue,
    stats: BenchmarkStats,
    stream: bool,
    max_tokens: int,
    client: httpx.AsyncClient,
) -> None:
    while not queue.empty():
        try:
            req_idx, prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": max_tokens,
        }

        req_start = time.perf_counter()
        first_token_time: Optional[float] = None
        tok_count = 0

        try:
            if stream:
                async with client.stream(
                    "POST", f"{url}/v1/chat/completions", json=payload, timeout=60.0
                ) as resp:
                    if resp.status_code != 200:
                        stats.errors += 1
                        continue

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tok_count += 1
            else:
                resp = await client.post(
                    f"{url}/v1/chat/completions", json=payload, timeout=60.0
                )
                if resp.status_code != 200:
                    stats.errors += 1
                    continue
                data = resp.json()
                tok_count = data.get("usage", {}).get("completion_tokens", max_tokens)

            req_end = time.perf_counter()
            total_time = req_end - req_start

            stats.successes += 1
            stats.latencies.append(total_time)
            stats.tokens.append(tok_count)
            if first_token_time is not None:
                stats.ttfts.append((first_token_time - req_start) * 1000.0)

        except Exception as err:
            stats.errors += 1
        finally:
            queue.task_done()


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int((len(sorted_d) - 1) * (p / 100.0))
    return sorted_d[idx]


async def run_benchmark(
    url: str,
    concurrency: int,
    total_requests: int,
    stream: bool,
    max_tokens: int,
) -> None:
    print("\n" + "=" * 60)
    print("🚀 Starting Chat Bot Engine Concurrency Benchmark")
    print(f"   Target URL       : {url}")
    print(f"   Concurrent Users : {concurrency}")
    print(f"   Total Requests   : {total_requests}")
    print(f"   Streaming (SSE)  : {stream}")
    print(f"   Max Tokens/Req   : {max_tokens}")
    print("=" * 60 + "\n")

    # Fill request queue
    q: asyncio.Queue = asyncio.Queue()
    for i in range(total_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        q.put_nowait((i, prompt))

    stats = BenchmarkStats()
    stats.start_time = time.perf_counter()

    limits = httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            asyncio.create_task(
                worker(i, url, q, stats, stream, max_tokens, client)
            )
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)

    stats.end_time = time.perf_counter()

    # Report
    print("\n" + "-" * 60)
    print("📊 BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Completed Requests   : {stats.successes}/{total_requests} (Errors: {stats.errors})")
    print(f"Wall Clock Time      : {stats.wall_clock_seconds:.2f} s")
    print(f"Total Tokens Emitted : {stats.total_tokens} tokens")
    print(f"Aggregate Throughput : {stats.tokens_per_second:.2f} tokens/sec")
    print(f"Request Throughput   : {stats.requests_per_second:.2f} reqs/sec")
    
    if stats.ttfts:
        print("\n--- Time To First Token (TTFT) Latencies ---")
        print(f"  Avg TTFT           : {statistics.mean(stats.ttfts):.1f} ms")
        print(f"  p50 TTFT           : {percentile(stats.ttfts, 50):.1f} ms")
        print(f"  p95 TTFT           : {percentile(stats.ttfts, 95):.1f} ms")
        print(f"  p99 TTFT           : {percentile(stats.ttfts, 99):.1f} ms")

    if stats.latencies:
        print("\n--- Total Sequence Latency ---")
        print(f"  Avg Request Latency: {statistics.mean(stats.latencies):.2f} s")
        print(f"  p50 Latency        : {percentile(stats.latencies, 50):.2f} s")
        print(f"  p95 Latency        : {percentile(stats.latencies, 95):.2f} s")
        print(f"  p99 Latency        : {percentile(stats.latencies, 99):.2f} s")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chat Bot Engine Concurrency Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Engine server URL")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent clients")
    parser.add_argument("--requests", type=int, default=32, help="Total number of requests")
    parser.add_argument("--stream", action="store_true", default=True, help="Use streaming SSE")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Use standard JSON response")
    parser.add_argument("--max-tokens", type=int, default=35, help="Tokens to generate per request")

    args = parser.parse_args()
    asyncio.run(
        run_benchmark(
            url=args.url,
            concurrency=args.concurrency,
            total_requests=args.requests,
            stream=args.stream,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
