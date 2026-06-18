#!/usr/bin/env python3
"""Portable GPT-2 Large GA timing profile for H100/A100 calibration.

This script intentionally avoids any SimAI imports and does not download model
weights. It constructs GPT-2 Large from config, uses dummy token IDs on GPU, and
measures one optimizer step for GA values such as 1, 2, and 4.
"""

import argparse
import json
import os
import platform
import socket
import statistics
import sys
from datetime import datetime, timezone

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def percentile(values, frac):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * frac))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def stdev_or_zero(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def profile_ga(model, optimizer, input_ids, labels, ga, warmup, steps):
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ga):
            outputs = model(input_ids, labels=labels)
            (outputs.loss / ga).backward()
        optimizer.step()
    torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ga):
            outputs = model(input_ids, labels=labels)
            (outputs.loss / ga).backward()
        optimizer.step()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def device_metadata(device_index):
    props = torch.cuda.get_device_properties(device_index)
    attrs = {
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
    }
    for optional_name in ("clock_rate", "memory_clock_rate", "memory_bus_width"):
        if hasattr(props, optional_name):
            attrs[optional_name] = getattr(props, optional_name)
    return attrs


def main():
    parser = argparse.ArgumentParser(description="Portable GPU GA timing profile")
    parser.add_argument("--ga", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}")
    gpu_name = torch.cuda.get_device_name(args.device)

    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Host: {socket.gethostname()}")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Python: {sys.version.split()[0]}")

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=1280,
        n_layer=36,
        n_head=20,
        attn_implementation="sdpa",
    )

    model = GPT2LMHeadModel(config).to(device).bfloat16()
    model.gradient_checkpointing_disable()
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT-2 Large ({param_count / 1e6:.0f}M params)")
    print("Attention: sdpa")
    print("Gradient checkpointing: OFF")
    print(f"Config: mbs={args.mbs}, seq_len={args.seq_len}, seed={args.seed}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (args.mbs, args.seq_len),
        device=device,
    )
    labels = input_ids.clone()

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_name,
        "device": device_metadata(args.device),
        "model": "gpt2-large",
        "params_M": param_count / 1e6,
        "mbs": args.mbs,
        "seq_len": args.seq_len,
        "attn": "sdpa",
        "grad_ckpt": False,
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "ga_results": {},
    }

    medians = {}
    for ga in sorted(args.ga):
        print(f"Profiling GA={ga} ({args.warmup} warmup + {args.steps} timed steps)...")
        times = profile_ga(
            model,
            optimizer,
            input_ids,
            labels,
            ga=ga,
            warmup=args.warmup,
            steps=args.steps,
        )
        med = statistics.median(times)
        medians[ga] = med
        p5 = percentile(times, 0.05)
        p95 = percentile(times, 0.95)
        print(f"  GA={ga}: median={med:.2f}ms, P5={p5:.2f}ms, P95={p95:.2f}ms")
        results["ga_results"][str(ga)] = {
            "median_ms": round(med, 2),
            "mean_ms": round(statistics.mean(times), 2),
            "stdev_ms": round(stdev_or_zero(times), 2),
            "p5_ms": round(p5, 2),
            "p95_ms": round(p95, 2),
            "all_ms": [round(t, 2) for t in times],
        }

    print()
    if len(medians) >= 2:
        slopes = []
        slope_details = []
        ga_list = sorted(medians.keys())
        for i, ga_a in enumerate(ga_list):
            for ga_b in ga_list[i + 1 :]:
                slope = (medians[ga_b] - medians[ga_a]) / (ga_b - ga_a)
                slopes.append(slope)
                slope_details.append(
                    {
                        "from_ga": ga_a,
                        "to_ga": ga_b,
                        "slope_ms_per_microbatch": round(slope, 4),
                    }
                )
                print(f"  Slope GA{ga_a}->GA{ga_b}: {slope:.2f} ms/micro-batch")

        avg_slope = statistics.mean(slopes)
        first_ga = ga_list[0]
        t_fixed = medians[first_ga] - avg_slope * first_ga
        results["slope_details"] = slope_details
        results["T_fwdbwd_ms"] = round(avg_slope, 2)
        results["T_fixed_ms"] = round(t_fixed, 2)

        print(f"\n  T_fwdbwd (per micro-batch) = {avg_slope:.2f} ms")
        print(f"  T_fixed (optimizer etc.)   = {t_fixed:.2f} ms")
    else:
        only_ga = next(iter(medians))
        results["T_step_ms"] = round(medians[only_ga], 2)
        print(f"  T_step(GA={only_ga}) = {medians[only_ga]:.2f} ms")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
