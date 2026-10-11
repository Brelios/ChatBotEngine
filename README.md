# AI Chatbot Serving Engine ⚡

An ultra-efficient, custom inference and serving engine for AI chatbots built in Python. Designed to handle high-concurrency multi-user traffic with **continuous dynamic batching**, **iteration-level scheduling**, and **real-time word-by-word streaming** via Server-Sent Events (SSE).

---

## 🌟 Highlights & Architecture

- **Continuous Dynamic Batching**: Eliminates waiting bubbles! At every token generation step, new user requests join the batch and completed requests are immediately retired.
- **Word-by-Word Streaming (SSE)**: Streams tokens directly to clients via HTTP `text/event-stream` with standard OpenAI `/v1/chat/completions` compatibility.
- **Dual Execution Backends**:
  - **`SimulatedRunner` (Default)**: Zero-download synthetic LLM for instant stress testing with 10–500 concurrent users without gigabytes of weights or GPU limits.
  - **`HuggingFaceRunner`**: Real open-weights model (`Qwen/Qwen2.5-0.5B-Instruct`, `SmolLM2-135M`, etc.) running on NVIDIA RTX GPUs (CUDA) or CPU.
- **Client Disconnection Recovery**: Automatically detects aborted HTTP requests and purges in-flight tokens from the scheduler, freeing GPU memory and compute.
- **Interactive Web Interface**: Modern dark-mode ChatGPT-style UI with a built-in **Concurrency Studio** and live telemetry counters (active batch, queue depth, throughput, TTFT, ITL).
- **Benchmarking Suite**: Asynchronous multi-client load testing tool for benchmarking latency distributions (p50, p95, p99) under heavy traffic.

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Engine (Simulated Mode - Zero Download)
```powershell
python main.py --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to chat or test concurrent streams!

### 3. Launch with Real Open-Weights Model (PyTorch / GPU)
```powershell
python main.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

---

## 📊 Running Concurrency Benchmarks

Simulate multiple concurrent users generating tokens simultaneously:

```powershell
# Run benchmark with 10 concurrent users generating 40 requests total
python benchmarks/load_test.py --concurrency 10 --requests 40 --stream
```

Example Output:
```
============================================================
🚀 Starting Chat Bot Engine Concurrency Benchmark
   Target URL       : http://localhost:8000
   Concurrent Users : 10
   Total Requests   : 40
   Streaming (SSE)  : True
   Max Tokens/Req   : 35
============================================================

------------------------------------------------------------
📊 BENCHMARK RESULTS
------------------------------------------------------------
Completed Requests   : 40/40 (Errors: 0)
Wall Clock Time      : 3.12 s
Total Tokens Emitted : 1400 tokens
Aggregate Throughput : 448.72 tokens/sec
Request Throughput   : 12.82 reqs/sec

--- Time To First Token (TTFT) Latencies ---
  Avg TTFT           : 34.8 ms
  p50 TTFT           : 31.2 ms
  p95 TTFT           : 48.6 ms
  p99 TTFT           : 52.1 ms

--- Total Sequence Latency ---
  Avg Request Latency: 0.74 s
  p50 Latency        : 0.72 s
  p95 Latency        : 0.88 s
  p99 Latency        : 0.91 s
============================================================
```

---

## 🛠️ API Reference

### `POST /v1/chat/completions`
OpenAI-compatible chat completions endpoint.

**Streaming Request (SSE):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain continuous batching."}],
    "stream": true
  }'
```

**Non-Streaming Request:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### `GET /v1/metrics`
Returns live telemetry:
```json
{
  "active_requests": 3,
  "queued_requests": 0,
  "total_requests_processed": 142,
  "total_tokens_generated": 4820,
  "current_tokens_per_second": 284.5,
  "avg_ttft_ms": 32.4,
  "avg_itl_ms": 19.8,
  "backend_name": "Simulated",
  "model_name": "simulated-chat-v1",
  "max_batch_size": 16
}
```

---

## 🧪 Running Unit & Integration Tests

```powershell
python -m pytest tests/ -v
```
