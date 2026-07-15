#!/usr/bin/env python3
"""Opt-in inference and compiled-training gates for exported TTT-MLP models."""

import argparse
import gc
import json
import statistics
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference(model, input_ids, iterations):
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


def inference_gate(args, result):
    torch.manual_seed(args.seed)
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

    dual_cpu = measure_inference(model, input_ids, args.iterations)
    dual_methods = set_online_reference(blocks)
    try:
        online_cpu = measure_inference(model, input_ids, args.iterations)
    finally:
        restore_dual(blocks, dual_methods)
    speedup = online_cpu / dual_cpu
    result.update({
        "batch": args.batch,
        "seq_len": args.seq_len,
        "ttt_blocks": len(blocks),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cpu_dual_ms": dual_cpu * 1000.0,
        "cpu_online_ms": online_cpu * 1000.0,
        "cpu_speedup": speedup,
    })

    if args.mps and torch.backends.mps.is_available():
        mps = torch.device("mps")
        mps_model = AutoModelForCausalLM.from_pretrained(
            args.dir, trust_remote_code=True
        ).to(mps).eval()
        mps_input = input_ids.to(mps)
        dual_mps = measure_inference(mps_model, mps_input, args.iterations)
        result["mps_dual_ms"] = dual_mps * 1000.0
        result["mps_cpu_ratio"] = dual_mps / dual_cpu

    if speedup < args.min_cpu_speedup:
        raise RuntimeError(
            f"TTT dual CPU speedup {speedup:.2f}x < {args.min_cpu_speedup:.2f}x"
        )


def resolve_training_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training gate requested, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS training gate requested, but MPS is unavailable")
    return device


def training_inputs(model, device, batch, seq_len):
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (batch, seq_len),
        device=device,
    )
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    if batch > 1 and seq_len > 2:
        trailing = max(1, min(seq_len // 4, seq_len - 1))
        attention_mask[1, -trailing:] = 0
        input_ids[1, -trailing:] = 0
    labels = torch.arange(batch, device=device, dtype=torch.long) % int(model.config.num_labels)
    return input_ids, attention_mask, labels


def gradient_snapshot(model):
    snapshot = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            snapshot[name] = parameter.grad.detach().float().cpu().clone()
    return snapshot


def max_gradient_difference(want, model):
    got_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    if got_names != set(want):
        missing = sorted(set(want) - got_names)
        extra = sorted(got_names - set(want))
        raise RuntimeError(f"compiled gradient names differ: missing={missing}, extra={extra}")
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        difference = torch.max(
            torch.abs(parameter.grad.detach().float().cpu() - want[name])
        ).item()
        maximum = max(maximum, difference)
    return maximum


class CUDAUtilizationSampler:
    def __init__(self, device):
        self.device = device
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.device.type != "cuda":
            return
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                self.samples.append(float(torch.cuda.utilization(self.device)))
            except Exception as error:  # Best-effort diagnostic; throughput is the gate.
                self.error = str(error)
                return
            self.stop_event.wait(0.1)

    def stop(self):
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join()


def training_step(model, optimizer, input_ids, attention_mask, labels):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = output.loss
    loss.backward()
    if optimizer is not None:
        optimizer.step()
    return loss


def measure_training(model, input_ids, attention_mask, labels, warmups, iterations):
    device = input_ids.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(warmups):
        loss = training_step(model, optimizer, input_ids, attention_mask, labels)
        sync(device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{device} training warm-up produced non-finite loss")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    sampler = CUDAUtilizationSampler(device)
    sampler.start()
    try:
        for _ in range(iterations):
            start = time.perf_counter()
            loss = training_step(model, optimizer, input_ids, attention_mask, labels)
            sync(device)
            samples.append(time.perf_counter() - start)
    finally:
        sampler.stop()
    if not torch.isfinite(loss):
        raise RuntimeError(f"{device} training produced non-finite loss")
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return statistics.median(samples), peak_bytes, sampler


def load_classifier(args, device):
    torch.manual_seed(args.seed)
    return AutoModelForSequenceClassification.from_pretrained(
        args.dir,
        trust_remote_code=True,
        num_labels=2,
    ).to(device).train()


def compile_model(model, args):
    kwargs = {
        "backend": args.compile_backend,
        "fullgraph": True,
    }
    if args.compile_backend != "eager":
        kwargs["mode"] = args.compile_mode
    return torch.compile(model, **kwargs)


def release_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def training_gate(args, result):
    device = resolve_training_device(args.train_device)
    eager_model = load_classifier(args, device)
    if not ttt_blocks(eager_model):
        raise RuntimeError("exported classifier contains no TTT-MLP blocks")
    inputs = training_inputs(eager_model, device, args.batch, args.seq_len)

    eager_model.zero_grad(set_to_none=True)
    eager_loss = training_step(eager_model, None, *inputs)
    sync(device)
    eager_gradients = gradient_snapshot(eager_model) if args.gradient_parity else None
    eager_ms, eager_peak, eager_util = measure_training(
        eager_model, *inputs, args.warmup_iterations, args.iterations
    )
    eager_loss_value = float(eager_loss.detach().cpu())
    del eager_loss
    del eager_model
    release_device_memory(device)

    compiled_base = load_classifier(args, device)
    compiled_inputs = tuple(value.detach().clone() for value in inputs)
    del inputs
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    compiled_model = compile_model(compiled_base, args)
    compiled_base.zero_grad(set_to_none=True)
    compile_start = time.perf_counter()
    compiled_loss = training_step(compiled_model, None, *compiled_inputs)
    sync(device)
    compile_first_step = time.perf_counter() - compile_start
    loss_difference = abs(eager_loss_value - float(compiled_loss.detach().cpu()))
    gradient_difference = None
    if eager_gradients is not None:
        gradient_difference = max_gradient_difference(eager_gradients, compiled_base)
        del eager_gradients
    if loss_difference > args.max_loss_diff:
        raise RuntimeError(
            f"compiled loss difference {loss_difference:.6g} > {args.max_loss_diff:.6g}"
        )
    if gradient_difference is not None and gradient_difference > args.max_grad_diff:
        raise RuntimeError(
            f"compiled gradient difference {gradient_difference:.6g} > {args.max_grad_diff:.6g}"
        )

    if args.check_padding_guard:
        left_padded = compiled_inputs[1].clone()
        left_padded[0, 0] = 0
        left_padded[0, 1] = 1
        empty_row = compiled_inputs[1].clone()
        empty_row[0] = 0
        for bad_mask, want_error in (
            (left_padded, "right-padded"),
            (empty_row, "no real tokens"),
        ):
            try:
                compiled_model(
                    input_ids=compiled_inputs[0],
                    attention_mask=bad_mask,
                    labels=compiled_inputs[2],
                )
                sync(device)
            except RuntimeError as error:
                if want_error not in str(error):
                    raise RuntimeError(
                        f"compiled input guard returned the wrong error: {error}"
                    ) from error
            else:
                raise RuntimeError(
                    f"compiled TTT-MLP did not reject invalid mask ({want_error})"
                )

    compiled_ms, compiled_peak, compiled_util = measure_training(
        compiled_model, *compiled_inputs, args.warmup_iterations, args.iterations
    )
    speedup = eager_ms / compiled_ms
    tokens = args.batch * args.seq_len
    result.update({
        "train_device": str(device),
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode if args.compile_backend != "eager" else None,
        "compile_fullgraph": True,
        "compile_first_step_ms": compile_first_step * 1000.0,
        "eager_train_ms": eager_ms * 1000.0,
        "compiled_train_ms": compiled_ms * 1000.0,
        "eager_train_tokens_per_second": tokens / eager_ms,
        "compiled_train_tokens_per_second": tokens / compiled_ms,
        "compiled_train_speedup": speedup,
        "compiled_loss_difference": loss_difference,
    })
    if gradient_difference is not None:
        result["compiled_max_gradient_difference"] = gradient_difference
    if eager_peak is not None:
        result["eager_peak_memory_bytes"] = int(eager_peak)
    if compiled_peak is not None:
        result["compiled_peak_memory_bytes"] = int(compiled_peak)
    if eager_util.samples:
        result["eager_gpu_utilization_median"] = statistics.median(eager_util.samples)
        result["eager_gpu_utilization_max"] = max(eager_util.samples)
    if compiled_util.samples:
        result["compiled_gpu_utilization_median"] = statistics.median(compiled_util.samples)
        result["compiled_gpu_utilization_max"] = max(compiled_util.samples)
    if eager_util.error:
        result["eager_gpu_utilization_error"] = eager_util.error
    if compiled_util.error:
        result["compiled_gpu_utilization_error"] = compiled_util.error
    if hasattr(torch, "_dynamo"):
        stats = torch._dynamo.utils.counters.get("stats", {})
        result["compiled_unique_graphs"] = int(stats.get("unique_graphs", 0))

    del compiled_loss
    del compiled_model
    del compiled_base
    release_device_memory(device)
    if speedup < args.min_compile_speedup:
        raise RuntimeError(
            f"compiled TTT training speedup {speedup:.2f}x < {args.min_compile_speedup:.2f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--mode", choices=("inference", "train", "both"), default="inference")
    parser.add_argument("--seq-len", type=int, default=43)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--min-cpu-speedup", type=float, default=3.0)
    parser.add_argument("--min-compile-speedup", type=float, default=0.0)
    parser.add_argument("--train-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--max-loss-diff", type=float, default=1e-5)
    parser.add_argument("--max-grad-diff", type=float, default=1e-4)
    parser.add_argument("--gradient-parity", action="store_true")
    parser.add_argument("--check-padding-guard", action="store_true")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= 1 or args.batch <= 0 or args.iterations <= 0:
        raise ValueError("seq-len must exceed one; batch and iterations must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if args.min_compile_speedup < 0:
        raise ValueError("min-compile-speedup must be non-negative")

    result = {"mode": args.mode}
    if args.mode in ("inference", "both"):
        inference_gate(args, result)
    if args.mode in ("train", "both"):
        training_gate(args, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
