#!/usr/bin/env python3
"""Comprehensive benchmark comparing baseline diffusers vs optimized HyperGen."""

import argparse
import time
from typing import Dict, List

import torch
from diffusers import FluxPipeline

from hypergen import Model


def run_benchmark(
    pipe: any,
    prompt: str,
    num_steps: int,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> Dict[str, float]:
    """Run benchmark on a pipeline."""
    # Warmup
    for _ in range(warmup_runs):
        _ = pipe(prompt, num_inference_steps=num_steps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Benchmark
    times = []
    memory_peaks = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        _ = pipe(prompt, num_inference_steps=num_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        if torch.cuda.is_available():
            memory_peaks.append(torch.cuda.max_memory_allocated() / 1e9)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "mean_memory": sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
        "peak_memory": max(memory_peaks) if memory_peaks else 0,
    }


def benchmark_baseline(
    model_id: str,
    prompt: str,
    num_steps: int,
    use_bfloat16: bool = False,
) -> Dict[str, float]:
    """Benchmark using raw diffusers."""
    print("\n" + "=" * 80)
    print("BASELINE: Raw Diffusers")
    print("=" * 80)

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Loading model ({dtype})...")

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to("cuda")

    # Disable optimizations
    if hasattr(pipe, "transformer"):
        pipe.transformer.set_default_attn_processor()
    if hasattr(pipe, "unet"):
        pipe.unet.set_default_attn_processor()
    if hasattr(pipe, "vae"):
        pipe.vae.set_default_attn_processor()

    print("Running benchmark...")
    results = run_benchmark(pipe, prompt, num_steps)

    print(f"  Mean time:   {results['mean_time']:.2f}s")
    print(f"  Mean memory: {results['mean_memory']:.2f} GB")

    del pipe
    torch.cuda.empty_cache()

    return results


def benchmark_bfloat16(
    model_id: str,
    prompt: str,
    num_steps: int,
) -> Dict[str, float]:
    """Benchmark with bfloat16 precision."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 1: bfloat16 Precision")
    print("=" * 80)

    print("Loading model (bfloat16)...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print("Running benchmark...")
    results = run_benchmark(pipe, prompt, num_steps)

    print(f"  Mean time:   {results['mean_time']:.2f}s")
    print(f"  Mean memory: {results['mean_memory']:.2f} GB")

    del pipe
    torch.cuda.empty_cache()

    return results


def benchmark_sdpa(
    model_id: str,
    prompt: str,
    num_steps: int,
) -> Dict[str, float]:
    """Benchmark with SDPA (Scaled Dot Product Attention)."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 2: bfloat16 + SDPA")
    print("=" * 80)

    print("Loading model (bfloat16 + SDPA)...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # SDPA is enabled by default in Diffusers when using PyTorch 2+
    # Just ensure we don't set a custom processor

    print("Running benchmark...")
    results = run_benchmark(pipe, prompt, num_steps)

    print(f"  Mean time:   {results['mean_time']:.2f}s")
    print(f"  Mean memory: {results['mean_memory']:.2f} GB")

    del pipe
    torch.cuda.empty_cache()

    return results


def benchmark_compiled(
    model_id: str,
    prompt: str,
    num_steps: int,
    mode: str = "regional",
) -> Dict[str, float]:
    """Benchmark with torch.compile."""
    print("\n" + "=" * 80)
    print(f"OPTIMIZATION 3: bfloat16 + SDPA + torch.compile ({mode})")
    print("=" * 80)

    print(f"Loading HyperGen model with {mode} compilation...")
    load_start = time.perf_counter()

    model = Model.load(
        model_id,
        dtype=torch.bfloat16,
        compile_mode=mode,
    )

    load_time = time.perf_counter() - load_start
    print(f"  Loaded in {load_time:.2f}s")

    print("Running benchmark (includes first compile)...")
    compile_start = time.perf_counter()

    results = run_benchmark(model.pipeline, prompt, num_steps, warmup_runs=0)

    compile_time = time.perf_counter() - compile_start - results["mean_time"] * 5
    print(f"  Compilation time: ~{compile_time:.2f}s")
    print(f"  Mean time:        {results['mean_time']:.2f}s")
    print(f"  Mean memory:      {results['mean_memory']:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return results


def benchmark_optimized(
    model_id: str,
    prompt: str,
    num_steps: int,
) -> Dict[str, float]:
    """Benchmark with all HyperGen optimizations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 4: Full HyperGen (compile + QKV fusion + channels_last)")
    print("=" * 80)

    print("Loading HyperGen model with all optimizations...")
    model = Model.load(
        model_id,
        dtype=torch.bfloat16,
        compile_mode="regional",
    )

    # Enable fused QKV projections
    model.enable_fused_qkv_projections()

    print("Running benchmark...")
    results = run_benchmark(model.pipeline, prompt, num_steps)

    print(f"  Mean time:   {results['mean_time']:.2f}s")
    print(f"  Mean memory: {results['mean_memory']:.2f} GB")

    del model
    torch.cuda.empty_cache()

    return results


def print_summary(results: Dict[str, Dict[str, float]], baseline_key: str):
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    baseline = results[baseline_key]

    print(f"{'Configuration':<50} {'Time (s)':<12} {'Speedup':<10} {'Memory (GB)':<12}")
    print("-" * 84)

    for name, result in results.items():
        speedup = baseline["mean_time"] / result["mean_time"]
        print(
            f"{name:<50} {result['mean_time']:>8.2f}    "
            f"{speedup:>6.2f}x    {result['mean_memory']:>8.2f}"
        )

    print()
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark HyperGen optimizations")
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Model ID to benchmark",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Prompt to use for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark (saves time)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print("=" * 80)
    print("HyperGen Optimization Benchmark")
    print("=" * 80)
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model:  {args.model}")
    print(f"Steps:  {args.steps}")
    print(f"Prompt: {args.prompt}")

    results = {}

    # Run benchmarks
    if not args.skip_baseline:
        results["1. Baseline (float32, no SDPA)"] = benchmark_baseline(
            args.model, args.prompt, args.steps, use_bfloat16=False
        )

    results["2. bfloat16"] = benchmark_bfloat16(args.model, args.prompt, args.steps)

    results["3. bfloat16 + SDPA"] = benchmark_sdpa(args.model, args.prompt, args.steps)

    results["4. bfloat16 + SDPA + torch.compile (regional)"] = benchmark_compiled(
        args.model, args.prompt, args.steps, mode="regional"
    )

    results["5. Full HyperGen (all optimizations)"] = benchmark_optimized(
        args.model, args.prompt, args.steps
    )

    # Print summary
    baseline_key = "2. bfloat16" if args.skip_baseline else "1. Baseline (float32, no SDPA)"
    print_summary(results, baseline_key)


if __name__ == "__main__":
    main()
