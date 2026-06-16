#!/usr/bin/env python3
"""Generate static gated_deltanet full-block reference fixture.

This is a lightweight scalar reference for Mixlab's public gated_deltanet block
semantics. It is separate from the optional FLA parity harness, which remains
useful when an upstream FLA fixture is available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "train" / "testdata" / "gated_deltanet_full_block_reference.json"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def rms_norm_rows(x: list[float], rows: int, cols: int, scale: list[float], eps: float) -> list[float]:
    out: list[float] = []
    for r in range(rows):
        base = r * cols
        mean_sq = sum(x[base + c] * x[base + c] for c in range(cols)) / cols
        inv = 1.0 / math.sqrt(mean_sq + eps)
        for c in range(cols):
            out.append(x[base + c] * inv * scale[c])
    return out


def matmul(a: list[float], b: list[float], rows: int, inner: int, cols: int) -> list[float]:
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            acc = 0.0
            for k in range(inner):
                acc += a[r * inner + k] * b[k * cols + c]
            out[r * cols + c] = acc
    return out


def short_conv_1d(seq: list[float], weight: list[float], batch: int, steps: int, channels: int) -> list[float]:
    out = [0.0] * (batch * steps * channels)
    for b in range(batch):
        for t in range(steps):
            for c in range(channels):
                acc = 0.0
                for tap in range(4):
                    src_t = t - tap
                    if src_t >= 0:
                        acc += seq[(b * steps + src_t) * channels + c] * weight[tap * channels + c]
                out[(b * steps + t) * channels + c] = silu(acc)
    return out


def gated_delta_scan(
    q: list[float],
    k: list[float],
    v: list[float],
    beta: list[float],
    gate: list[float],
    batch: int,
    steps: int,
    heads: int,
    d_key: int,
    d_value: int,
) -> list[float]:
    state = [0.0] * (batch * heads * d_key * d_value)
    out = [0.0] * (batch * steps * heads * d_value)
    for b in range(batch):
        for t in range(steps):
            for h in range(heads):
                gate_t = gate[(b * steps + t) * heads + h]
                beta_t = beta[(b * steps + t) * heads + h]
                for dk in range(d_key):
                    for dv in range(d_value):
                        sidx = ((b * heads + h) * d_key + dk) * d_value + dv
                        state[sidx] *= gate_t
                pred = [0.0] * d_value
                for dv in range(d_value):
                    for dk in range(d_key):
                        kval = k[((b * steps + t) * heads + h) * d_key + dk]
                        pred[dv] += kval * state[((b * heads + h) * d_key + dk) * d_value + dv]
                for dk in range(d_key):
                    kval = k[((b * steps + t) * heads + h) * d_key + dk]
                    for dv in range(d_value):
                        vval = v[((b * steps + t) * heads + h) * d_value + dv]
                        err = (vval - pred[dv]) * beta_t
                        state[((b * heads + h) * d_key + dk) * d_value + dv] += kval * err
                for dv in range(d_value):
                    acc = 0.0
                    for dk in range(d_key):
                        qval = q[((b * steps + t) * heads + h) * d_key + dk]
                        acc += qval * state[((b * heads + h) * d_key + dk) * d_value + dv]
                    out[((b * steps + t) * heads + h) * d_value + dv] = acc
    return out


def embed_weights(vocab: int, dim: int, x: list[float]) -> list[float]:
    out = [0.0] * (vocab * dim)
    out[: len(x)] = x
    return out


def fixture() -> dict:
    batch, steps, dim, vocab, heads, d_key, d_value = 1, 3, 4, 8, 1, 2, 3
    key_dim = heads * d_key
    val_dim = heads * d_value
    x = [
        0.20, -0.35, 0.15, 0.42,
        -0.18, 0.31, 0.27, -0.24,
        0.46, 0.08, -0.39, 0.12,
    ]
    norm = [1.05, 0.90, 1.15, 0.85]
    wq = [
        0.24, -0.18,
        -0.31, 0.22,
        0.16, 0.35,
        0.28, -0.11,
    ]
    w_kv = [
        0.30, -0.14, 0.21,
        -0.25, 0.33, 0.08,
        0.17, 0.26, -0.19,
        -0.09, 0.12, 0.37,
    ]
    q_conv = [
        0.45, -0.20,
        -0.15, 0.32,
        0.11, 0.27,
        -0.08, 0.19,
    ]
    k_conv = [
        0.36, 0.14,
        -0.21, 0.29,
        0.18, -0.12,
        0.07, 0.23,
    ]
    v_conv = [
        0.22, -0.31, 0.18,
        0.27, 0.09, -0.16,
        -0.13, 0.24, 0.35,
        0.19, -0.08, 0.11,
    ]
    w_a = [
        0.17,
        -0.23,
        0.31,
        0.09,
    ]
    a_log = [-0.35]
    dt_bias = [0.20]
    w_beta = [
        -0.19,
        0.28,
        0.07,
        0.34,
    ]
    o_norm = [1.10, 0.95, 0.80]
    w_out_gate = [
        0.13, -0.26, 0.31,
        0.22, 0.18, -0.15,
        -0.33, 0.09, 0.25,
        0.16, -0.11, 0.29,
    ]
    wo = [
        0.25, -0.17, 0.12, 0.31,
        -0.14, 0.28, 0.33, -0.06,
        0.19, 0.07, -0.22, 0.24,
    ]

    x_norm = rms_norm_rows(x, batch * steps, dim, norm, 1e-5)
    q_proj = matmul(x_norm, wq, batch * steps, dim, key_dim)
    q_seq = short_conv_1d(q_proj, q_conv, batch, steps, key_dim)
    kv_wide = matmul(x_norm, w_kv, batch * steps, dim, val_dim)
    k_proj = []
    for row in range(batch * steps):
        k_proj.extend(kv_wide[row * val_dim : row * val_dim + d_key])
    k_seq = short_conv_1d(k_proj, k_conv, batch, steps, key_dim)
    v_seq = short_conv_1d(kv_wide, v_conv, batch, steps, val_dim)

    q_unit = rms_norm_rows(q_seq, batch * steps * heads, d_key, [1.0] * d_key, 1e-6)
    k_unit = rms_norm_rows(k_seq, batch * steps * heads, d_key, [1.0] * d_key, 1e-6)
    inv_sqrt_dk = 1.0 / math.sqrt(d_key)
    q_scaled = [v * inv_sqrt_dk * inv_sqrt_dk for v in q_unit]
    k_scaled = [v * inv_sqrt_dk for v in k_unit]

    gate_raw = matmul(x_norm, w_a, batch * steps, dim, heads)
    gate = []
    for v in gate_raw:
        dt = softplus(v + dt_bias[0])
        gate.append(math.exp(-dt * math.exp(a_log[0])))
    beta = [sigmoid(v) for v in matmul(x_norm, w_beta, batch * steps, dim, heads)]

    scan = gated_delta_scan(q_scaled, k_scaled, v_seq, beta, gate, batch, steps, heads, d_key, d_value)
    y_norm = rms_norm_rows(scan, batch * steps * heads, d_value, o_norm, 1e-5)
    out_gate = [silu(v) for v in matmul(x_norm, w_out_gate, batch * steps, dim, val_dim)]
    y_gated = [a * b for a, b in zip(y_norm, out_gate)]
    projected = matmul(y_gated, wo, batch * steps, val_dim, dim)
    expected = [a + b for a, b in zip(x, projected)]

    return {
        "name": "gated_deltanet_full_block",
        "config": {
            "name": "gated_deltanet_reference_fixture",
            "model_dim": dim,
            "vocab_size": vocab,
            "seq_len": steps,
            "blocks": [{"type": "gated_deltanet", "heads": heads, "d_k": d_key, "d_v": d_value, "kv_share": True}],
            "training": {
                "steps": 1,
                "batch_tokens": batch * steps,
                "seed": 13,
                "objective": "mntp",
                "mlm_mask_token_id": vocab - 1,
                "data2vec": {"loss_weight": 1.0, "top_k_layers": 1},
            },
        },
        "output": "data2vec_layer_00_hidden_flat",
        "batch": batch,
        "seq_len": steps,
        "model_dim": dim,
        "tokens": [0, 1, 2],
        "targets": [1, 2, 3],
        "loss_mask": [1.0, 1.0, 1.0],
        "weights": [
            {"name": "embed", "shape": [vocab, dim], "values": embed_weights(vocab, dim, x)},
            {"name": "head", "shape": [dim, vocab], "values": [0.0] * (dim * vocab)},
            {"name": "final_norm", "shape": [dim], "values": [1.0] * dim},
            {"name": "norm_scale", "shape": [dim], "values": norm},
            {"name": "wq", "shape": [dim, key_dim], "values": wq},
            {"name": "q_conv", "shape": [4, key_dim], "values": q_conv},
            {"name": "w_kv", "shape": [dim, val_dim], "values": w_kv},
            {"name": "k_conv", "shape": [4, key_dim], "values": k_conv},
            {"name": "v_conv", "shape": [4, val_dim], "values": v_conv},
            {"name": "w_a", "shape": [dim, heads], "values": w_a},
            {"name": "A_log", "shape": [heads], "values": a_log},
            {"name": "dt_bias", "shape": [heads], "values": dt_bias},
            {"name": "w_beta", "shape": [dim, heads], "values": w_beta},
            {"name": "o_norm_scale", "shape": [d_value], "values": o_norm},
            {"name": "w_out_gate", "shape": [dim, val_dim], "values": w_out_gate},
            {"name": "wo", "shape": [val_dim, dim], "values": wo},
        ],
        "expected_hidden": expected,
    }


def main() -> None:
    payload = {
        "version": 1,
        "description": "Static full-block gated_deltanet reference fixture generated from independent scalar reference loops.",
        "fixtures": [fixture()],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
