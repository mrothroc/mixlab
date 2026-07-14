#!/usr/bin/env python3
"""Opt-in performance gate for an exported TTT-MLP Hugging Face model."""

import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, input_ids, iterations):
    device = input_ids.device
    with torch.no_grad():
        model(input_ids=input_ids)
        sync(device)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(input_ids=input_ids).logits
            sync(device)
            samples.append(time.perf_counter() - start)
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{device} output contains non-finite logits")
    return statistics.median(samples)


def ttt_blocks(model):
    return [
        block
        for block in model.blocks
        if block.__class__.__name__ == "MixlabTTTMLPBlock"
    ]


def set_online_reference(blocks):
    dual_methods = [block._stateless_dual_scan for block in blocks]
    for block in blocks:
        block._stateless_dual_scan = block._stateless_online_scan
    return dual_methods


def restore_dual(blocks, dual_methods):
    for block, dual_method in zip(blocks, dual_methods):
        block._stateless_dual_scan = dual_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--seq-len", type=int, default=43)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-cpu-speedup", type=float, default=3.0)
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= 0 or args.batch <= 0 or args.iterations <= 0:
        raise ValueError("seq-len, batch, and iterations must be positive")

    torch.manual_seed(314159)
    cpu = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.dir, trust_remote_code=True
    ).to(cpu).eval()
    blocks = ttt_blocks(model)
    if not blocks:
        raise RuntimeError("exported model contains no TTT-MLP blocks")
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (args.batch, args.seq_len),
        device=cpu,
    )

    dual_cpu = measure(model, input_ids, args.iterations)
    dual_methods = set_online_reference(blocks)
    try:
        online_cpu = measure(model, input_ids, args.iterations)
    finally:
        restore_dual(blocks, dual_methods)
    speedup = online_cpu / dual_cpu
    result = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "ttt_blocks": len(blocks),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cpu_dual_ms": dual_cpu * 1000.0,
        "cpu_online_ms": online_cpu * 1000.0,
        "cpu_speedup": speedup,
    }

    if args.mps and torch.backends.mps.is_available():
        mps = torch.device("mps")
        mps_model = AutoModelForCausalLM.from_pretrained(
            args.dir, trust_remote_code=True
        ).to(mps).eval()
        mps_input = input_ids.to(mps)
        dual_mps = measure(mps_model, mps_input, args.iterations)
        result["mps_dual_ms"] = dual_mps * 1000.0
        result["mps_cpu_ratio"] = dual_mps / dual_cpu

    print(json.dumps(result, sort_keys=True))
    if speedup < args.min_cpu_speedup:
        raise RuntimeError(
            f"TTT dual CPU speedup {speedup:.2f}x < {args.min_cpu_speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
