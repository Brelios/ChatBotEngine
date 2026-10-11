"""Main CLI entrypoint for launching the AI Chatbot Serving Engine."""

from __future__ import annotations

import argparse
import sys
import uvicorn

from api.app import create_app
from engine.model_runner import BaseModelRunner, HuggingFaceRunner, SimulatedRunner
from engine.scheduler import ContinuousBatchingScheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the Custom AI Chatbot Serving Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["simulated", "hf"],
        default="simulated",
        help="Inference engine backend: 'simulated' (zero-download instant) or 'hf' (HuggingFace/PyTorch).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID or Hugging Face repo name (e.g. Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum concurrent sequences in active forward pass.",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=256,
        help="Capacity of waiting queue before rejecting with 429.",
    )
    parser.add_argument(
        "--step-delay-ms",
        type=float,
        default=20.0,
        help="Simulated forward step latency in milliseconds (for simulated backend).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Hardware device for HF backend ('auto', 'cuda', or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("⚡ AI Chatbot Serving Engine Initializing")
    print("=" * 65)
    print(f"  Backend        : {args.backend.upper()}")
    print(f"  Max Batch Size : {args.max_batch_size}")
    print(f"  Max Queue Size : {args.max_queue_size}")

    runner: BaseModelRunner
    if args.backend == "simulated":
        model_name = args.model or "simulated-chat-v1"
        runner = SimulatedRunner(model_id=model_name, step_delay_ms=args.step_delay_ms)
        print(f"  Model ID       : {model_name} (Simulated Step Delay: {args.step_delay_ms}ms)")
    elif args.backend == "hf":
        model_name = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"  Model ID       : {model_name} (Device: {args.device})")
        runner = HuggingFaceRunner(model_id=model_name, device=args.device)
    else:
        print(f"Error: Unknown backend '{args.backend}'", file=sys.stderr)
        sys.exit(1)

    # Initialize Continuous Batching Scheduler
    scheduler = ContinuousBatchingScheduler(
        runner=runner,
        max_batch_size=args.max_batch_size,
        max_waiting_queue_size=args.max_queue_size,
    )

    # Create FastAPI app
    app = create_app(scheduler=scheduler)

    print(f"  Serving at     : http://{args.host}:{args.port}")
    print(f"  Web Interface  : http://localhost:{args.port}")
    print(f"  API Docs       : http://localhost:{args.port}/docs")
    print(f"  Metrics        : http://localhost:{args.port}/v1/metrics")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
