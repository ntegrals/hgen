#!/usr/bin/env python3
"""Quick test script for HyperGen - Downloads Flux model and measures inference speed."""

import time
import torch
from hypergen import Model


def main():
    print("=" * 60)
    print("HyperGen Quick Test - Flux Model Inference")
    print("=" * 60)
    print()

    # Configuration
    model_id = "black-forest-labs/FLUX.1-schnell"  # Using schnell for faster download/inference
    test_prompt = "A cat holding a sign that says hello world"

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, using CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()
    print("-" * 60)
    print("Step 1: Loading model...")
    print(f"  Model: {model_id}")
    print("-" * 60)

    load_start = time.perf_counter()
    model = Model.load(
        model_id,
        device=device,
        enable_cpu_offload=False,
        compile_mode="none",  # Disable compilation for faster initial run
    )
    load_time = time.perf_counter() - load_start

    print(f"✓ Model loaded in {load_time:.2f} seconds")

    if device == "cuda":
        print(f"  Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print()
    print("-" * 60)
    print("Step 2: Running inference (warmup)...")
    print(f"  Prompt: '{test_prompt}'")
    print("-" * 60)

    # Warmup run
    warmup_start = time.perf_counter()
    _ = model.run(test_prompt, num_inference_steps=4)
    warmup_time = time.perf_counter() - warmup_start

    print(f"✓ Warmup completed in {warmup_time:.2f} seconds")
    model.clear_cache()

    print()
    print("-" * 60)
    print("Step 3: Running benchmark (5 runs)...")
    print("-" * 60)

    # Benchmark
    results = model.benchmark(
        prompt=test_prompt,
        num_runs=5,
        warmup_runs=0,  # Already warmed up
    )

    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"  Mean inference time:  {results['mean_time']:.2f} seconds")
    print(f"  Min inference time:   {results['min_time']:.2f} seconds")
    print(f"  Max inference time:   {results['max_time']:.2f} seconds")
    print(f"  Speed:                {1/results['mean_time']:.2f} images/second")

    if device == "cuda":
        print()
        print(f"  Mean memory usage:    {results['mean_memory_gb']:.2f} GB")
        print(f"  Peak memory usage:    {results['max_memory_gb']:.2f} GB")

    print("=" * 60)
    print()
    print("✓ Test completed successfully!")
    print()


if __name__ == "__main__":
    main()
