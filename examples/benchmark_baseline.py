#!/usr/bin/env python3
"""Baseline benchmark using raw diffusers (no optimizations)."""

import time
import torch
from diffusers import FluxPipeline


def benchmark_baseline(
    model_id: str = "black-forest-labs/FLUX.1-schnell",
    prompt: str = "A cat holding a sign that says hello world",
    num_inference_steps: int = 4,
    num_runs: int = 5,
    warmup_runs: int = 2,
):
    """Run baseline benchmark with raw diffusers."""
    print("=" * 80)
    print("BASELINE BENCHMARK - Raw Diffusers (No Optimizations)")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {model_id}")
    print(f"Steps: {num_inference_steps}")
    print()

    # Load pipeline in full precision (baseline)
    print("Loading pipeline (float32)...")
    load_start = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to("cuda")

    # Disable any optimizations
    pipe.unet.set_default_attn_processor()
    if hasattr(pipe, "vae"):
        pipe.vae.set_default_attn_processor()

    load_time = time.perf_counter() - load_start
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # Warmup
    print(f"Warmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        _ = pipe(prompt, num_inference_steps=num_inference_steps)
        torch.cuda.empty_cache()
    print("✓ Warmup complete")
    print()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    memory_peaks = []

    for i in range(num_runs):
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        _ = pipe(prompt, num_inference_steps=num_inference_steps)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        memory_peaks.append(torch.cuda.max_memory_allocated() / 1e9)

        print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f}s, {memory_peaks[-1]:.2f} GB")
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"Mean time:        {sum(times) / len(times):.2f}s")
    print(f"Min time:         {min(times):.2f}s")
    print(f"Max time:         {max(times):.2f}s")
    print(f"Mean memory:      {sum(memory_peaks) / len(memory_peaks):.2f} GB")
    print(f"Peak memory:      {max(memory_peaks):.2f} GB")
    print(f"Images/sec:       {1 / (sum(times) / len(times)):.3f}")
    print("=" * 80)
    print()

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "mean_memory": sum(memory_peaks) / len(memory_peaks),
        "peak_memory": max(memory_peaks),
    }


if __name__ == "__main__":
    benchmark_baseline()
