# AI Chatbot Engine - Project Plan & Architecture Guide

## 1. Project Motivation
When using ChatGPT or other LLM chatbots, users experience instant word-by-word streaming responses even when thousands of users are interacting simultaneously. 

Behind the scenes, this is powered not by naive sequential execution, but by a specialized **LLM Serving Engine** (such as vLLM, TGI, or Triton). The engine solves several difficult distributed systems and scheduling problems:
1. **Concurrency Without Slowdown**: Efficiently batching requests across users so GPU/CPU compute is utilized at maximum capacity.
2. **Dynamic / Continuous Batching**: Unlike traditional deep learning where batch sizes are fixed and fast sequences wait for slow sequences, iteration-level continuous batching adds new requests and retires completed requests at every token step.
3. **Word-by-Word Streaming**: Token generation is dispatched directly to HTTP Server-Sent Events (SSE) streams in real time as each token is generated.
4. **Client Disconnection Management**: If a client drops their connection, generation is immediately aborted to reclaim precious memory and compute.

This project builds that engine from scratch.

---

## 2. Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │            Browser / API Clients             │
                      │  - Web Chat Interface (ChatGPT dark mode)    │
                      │  - Multi-user Concurrency Studio             │
                      │  - OpenAI /v1/chat/completions Client        │
                      └───────────────────────┬──────────────────────┘
                                              │ HTTP + SSE Stream
                                              ▼
                      ┌──────────────────────────────────────────────┐
                      │              FastAPI Gateway                 │
                      │  - Schema validation & Request parsing       │
                      │  - Client disconnect detection               │
                      │  - Real-time engine telemetry (/v1/metrics)  │
                      └───────────────────────┬──────────────────────┘
                                              │ async enqueue
                                              ▼
                      ┌──────────────────────────────────────────────┐
                      │        Continuous Batching Scheduler         │
                      │  - Admission Control & Waiting Queue         │
                      │  - Dynamic Active Batch                      │
                      │  - Iteration step coordination               │
                      └───────────────────────┬──────────────────────┘
                                              │ forward step
                                              ▼
                      ┌──────────────────────────────────────────────┐
                      │             Model Runner Layer               │
                      │  - SimulatedRunner: Zero-setup synthetic LLM │
                      │  - HuggingFaceRunner: Qwen 2.5 / SmolLM2     │
                      └──────────────────────────────────────────────┘
```

---

## 3. Directory Layout

```
Chat-Bot-Engine/
├── engine/
│   ├── __init__.py
│   ├── types.py            # Data structures (ChatRequest, TokenChunk, Metrics)
│   ├── metrics.py          # Real-time metrics tracker (TTFT, ITL, TPS)
│   ├── model_runner.py     # Pluggable runners (Simulated & HuggingFace)
│   └── scheduler.py        # Continuous batching engine
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI server & SSE streaming
├── web/
│   └── index.html          # Chat interface & multi-client stress test studio
├── benchmarks/
│   └── load_test.py        # Asynchronous multi-client load tester
├── tests/
│   ├── test_scheduler.py   # Unit tests for continuous batching
│   └── test_api.py         # Unit tests for API endpoints
├── requirements.txt
├── main.py                 # Engine CLI startup
├── description             # Original project description
└── PLAN.md                 # This plan file
```

---

## 4. Key Engineering Details

### Dynamic / Continuous Batching
In static batching:
- Request 1 produces 10 tokens.
- Request 2 produces 200 tokens.
- Request 1 is forced to wait (or idly padded) until Request 2 finishes all 200 tokens.

In **Continuous Batching**:
- The engine executes a single generation step for all active sequences in the current batch.
- Each generated token is immediately emitted to the user's stream.
- As soon as Request 1 finishes (at step 10), Request 1 is immediately retired and freed.
- A new request from the waiting queue (e.g. Request 3) is admitted at step 11 and joins the batch seamlessly.

### Telemetry & Metrics
- **TTFT (Time To First Token)**: The latency from the moment the user sends a message until the first token appears on screen.
- **ITL (Inter-Token Latency)**: Average time between successive tokens (determines reading smoothness).
- **Throughput (Tokens/sec)**: Total tokens generated across all concurrent users per second.
- **Queue Depth & Active Batch Size**: Current engine saturation indicators.
