#!/usr/bin/env python3
"""Generate static mamba3-canonical full-block reference fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "train" / "testdata" / "mamba3_full_block_reference.json"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log1p(math.exp(x))


def rms_norm_rows(x: list[float], rows: int, cols: int, scale: list[float], eps: float = 1e-5) -> list[float]:
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


def depthwise_conv_1d(x: list[float], w: list[float], batch: int, steps: int, dim: int, kernel: int) -> list[float]:
    out = [0.0] * (batch * steps * dim)
    for b in range(batch):
        for t in range(steps):
            for d in range(dim):
                acc = 0.0
                for k in range(kernel):
                    src_t = t - k
                    if src_t >= 0:
                        acc += x[(b * steps + src_t) * dim + d] * w[d * kernel + k]
                out[(b * steps + t) * dim + d] = acc
    return out


def mamba3_scan(
    x: list[float],
    dt_raw: list[float],
    lambda_raw: list[float],
    theta: list[float],
    a_log: list[float],
    b_proj: list[float],
    c_proj: list[float],
    batch: int,
    steps: int,
    dim: int,
    state: int,
    groups: int,
) -> list[float]:
    pairs = state // 2
    channels_per_group = dim // groups
    h = [0.0] * (batch * dim * state)
    phi_prev = [0.0] * (batch * dim * pairs)
    out = [0.0] * (batch * steps * dim)
    for b in range(batch):
        for t in range(steps):
            b_rot = [0.0] * (dim * state)
            c_rot = [0.0] * (dim * state)
            delta = [0.0] * dim
            lam = [0.0] * dim
            for d in range(dim):
                row = (b * steps + t) * dim + d
                delta[d] = softplus(dt_raw[row])
                lam[d] = sigmoid(lambda_raw[row])
                group = d // channels_per_group
                group_base = ((b * steps + t) * groups + group) * state
                for p in range(pairs):
                    theta_idx = ((b * steps + t) * dim + d) * pairs + p
                    prev_idx = (b * dim + d) * pairs + p
                    angle = phi_prev[prev_idx] + delta[d] * theta[theta_idx]
                    phi_prev[prev_idx] = angle
                    cosv = math.cos(angle)
                    sinv = math.sin(angle)
                    b0 = b_proj[group_base + 2 * p]
                    b1 = b_proj[group_base + 2 * p + 1]
                    c0 = c_proj[group_base + 2 * p]
                    c1 = c_proj[group_base + 2 * p + 1]
                    dst = d * state + 2 * p
                    b_rot[dst] = cosv * b0 + sinv * b1
                    b_rot[dst + 1] = -sinv * b0 + cosv * b1
                    c_rot[dst] = cosv * c0 + sinv * c1
                    c_rot[dst + 1] = -sinv * c0 + cosv * c1
            prev_b_rot = None
            prev_x = None
            if t > 0:
                # Reconstruct previous rotated B for the previous row from the
                # saved cumulative angles by replaying within this tiny fixture.
                prev_b_rot = mamba3_previous_b_rot(
                    dt_raw, theta, b_proj, batch, steps, dim, state, groups, b, t - 1, channels_per_group
                )
                prev_x = x[(b * steps + t - 1) * dim : (b * steps + t) * dim]
            for d in range(dim):
                x_t = x[(b * steps + t) * dim + d]
                for n in range(state):
                    h_idx = (b * dim + d) * state + n
                    a = math.exp(delta[d] * -math.exp(a_log[d * state + n]))
                    previous = 0.0
                    if prev_b_rot is not None and prev_x is not None:
                        previous = (1.0 - lam[d]) * delta[d] * a * prev_b_rot[d * state + n] * prev_x[d]
                    current = lam[d] * delta[d] * b_rot[d * state + n] * x_t
                    h[h_idx] = a * h[h_idx] + previous + current
            for d in range(dim):
                acc = 0.0
                for n in range(state):
                    acc += c_rot[d * state + n] * h[(b * dim + d) * state + n]
                out[(b * steps + t) * dim + d] = acc
    return out


def mamba3_previous_b_rot(
    dt_raw: list[float],
    theta: list[float],
    b_proj: list[float],
    batch: int,
    steps: int,
    dim: int,
    state: int,
    groups: int,
    b: int,
    t: int,
    channels_per_group: int,
) -> list[float]:
    del batch, steps
    pairs = state // 2
    out = [0.0] * (dim * state)
    for d in range(dim):
        group = d // channels_per_group
        group_base = ((b * (t + 1) + t) * groups + group) * state
        # The fixture has B=1; replay cumulative phi explicitly for clarity.
        for p in range(pairs):
            angle = 0.0
            for tt in range(t + 1):
                row = (b * (t + 1) + tt) * dim + d
                angle += softplus(dt_raw[row]) * theta[row * pairs + p]
            cosv = math.cos(angle)
            sinv = math.sin(angle)
            b0 = b_proj[group_base + 2 * p]
            b1 = b_proj[group_base + 2 * p + 1]
            dst = d * state + 2 * p
            out[dst] = cosv * b0 + sinv * b1
            out[dst + 1] = -sinv * b0 + cosv * b1
    return out


def embed_weights(vocab: int, dim: int, x: list[float]) -> list[float]:
    out = [0.0] * (vocab * dim)
    out[: len(x)] = x
    return out


def fixture() -> dict:
    batch, steps, dim, vocab = 1, 3, 4, 8
    inner, state, groups, rank, kernel = 4, 4, 2, 2, 3
    group_state = groups * state
    x = [
        0.18, -0.22, 0.35, 0.11,
        -0.31, 0.27, 0.09, -0.16,
        0.42, 0.13, -0.28, 0.24,
    ]
    pre_norm = [1.05, 0.95, 1.10, 0.90]
    w_x = [
        0.21, -0.17, 0.09, 0.28,
        -0.24, 0.31, 0.16, -0.12,
        0.14, 0.22, -0.27, 0.19,
        0.33, -0.08, 0.25, 0.11,
    ]
    conv_w = [
        0.30, -0.11, 0.18,
        -0.22, 0.27, 0.09,
        0.16, 0.21, -0.14,
        0.25, -0.19, 0.12,
    ]
    w_dt_low = [0.18, -0.21, 0.07, 0.26, -0.12, 0.31, 0.24, -0.08]
    w_dt_high = [0.22, -0.16, 0.11, 0.29, -0.18, 0.27, 0.20, -0.13]
    w_lambda_low = [-0.19, 0.25, 0.30, -0.07, 0.13, -0.28, 0.09, 0.21]
    w_lambda_high = [0.17, 0.23, -0.15, 0.31, -0.24, 0.10, 0.28, -0.09]
    w_theta_low = [0.12, -0.27, 0.19, 0.08, -0.16, 0.32, 0.24, -0.20]
    w_theta_high = [
        0.20, -0.14, 0.25, -0.09, 0.18, 0.11, -0.22, 0.30,
        -0.17, 0.26, -0.12, 0.21, 0.07, -0.24, 0.28, -0.15,
    ]
    w_b = [
        0.19, -0.23, 0.15, 0.27, -0.08, 0.31, 0.12, -0.18,
        -0.21, 0.16, 0.29, -0.10, 0.25, -0.14, 0.07, 0.22,
        0.11, 0.28, -0.19, 0.13, -0.26, 0.20, 0.32, -0.06,
        0.24, -0.12, 0.18, 0.09, -0.15, 0.30, -0.22, 0.17,
    ]
    w_c = [
        -0.16, 0.24, -0.11, 0.33, 0.20, -0.09, 0.27, -0.18,
        0.29, -0.13, 0.08, 0.21, -0.25, 0.15, -0.07, 0.30,
        -0.10, 0.19, 0.26, -0.14, 0.31, -0.22, 0.12, 0.17,
        0.23, 0.06, -0.28, 0.16, -0.19, 0.25, 0.09, -0.12,
    ]
    b_norm = [1.10, 0.90, 1.05, 0.95]
    c_norm = [0.92, 1.08, 0.88, 1.12]
    b_bias = [1.0, 0.8, 1.2, 0.9, 1.1, 0.85, 1.05, 0.95]
    c_bias = [0.9, 1.15, 0.85, 1.05, 1.2, 0.75, 1.1, 0.8]
    a_log = [
        -2.0, -1.6, -1.2, -0.9,
        -1.8, -1.4, -1.1, -0.7,
        -2.1, -1.5, -1.0, -0.6,
        -1.9, -1.3, -0.8, -0.5,
    ]
    dt_bias = [0.10, -0.05, 0.15, -0.12]
    post_norm = [1.05, 0.97, 1.12, 0.89]
    w_gate = [
        0.27, -0.18, 0.13, 0.22,
        -0.15, 0.29, 0.24, -0.11,
        0.08, 0.20, -0.26, 0.31,
        0.18, -0.09, 0.28, 0.16,
    ]
    w_out = [
        0.23, -0.14, 0.19, 0.27,
        -0.21, 0.30, 0.11, -0.08,
        0.16, 0.25, -0.18, 0.12,
        0.29, -0.10, 0.07, 0.22,
    ]

    x_norm = rms_norm_rows(x, batch * steps, dim, pre_norm)
    x_proj = matmul(x_norm, w_x, batch * steps, dim, inner)
    x_conv = depthwise_conv_1d(x_proj, conv_w, batch, steps, inner, kernel)
    dt = matmul(matmul(x_conv, w_dt_low, batch * steps, inner, rank), w_dt_high, batch * steps, rank, inner)
    dt = [v + dt_bias[i % inner] for i, v in enumerate(dt)]
    lam = matmul(matmul(x_conv, w_lambda_low, batch * steps, inner, rank), w_lambda_high, batch * steps, rank, inner)
    theta = matmul(matmul(x_conv, w_theta_low, batch * steps, inner, rank), w_theta_high, batch * steps, rank, inner * (state // 2))
    b_proj = matmul(x_conv, w_b, batch * steps, inner, group_state)
    b_proj = rms_norm_rows(b_proj, batch * steps * groups, state, b_norm)
    b_proj = [v + b_bias[i % group_state] for i, v in enumerate(b_proj)]
    c_proj = matmul(x_conv, w_c, batch * steps, inner, group_state)
    c_proj = rms_norm_rows(c_proj, batch * steps * groups, state, c_norm)
    c_proj = [v + c_bias[i % group_state] for i, v in enumerate(c_proj)]
    y = mamba3_scan(x_conv, dt, lam, theta, a_log, b_proj, c_proj, batch, steps, inner, state, groups)
    y_norm = rms_norm_rows(y, batch * steps, inner, post_norm)
    z = [silu(v) for v in matmul(x_norm, w_gate, batch * steps, dim, inner)]
    y_gated = [a * b for a, b in zip(y_norm, z)]
    out_proj = matmul(y_gated, w_out, batch * steps, inner, dim)
    expected = [a + b for a, b in zip(x, out_proj)]

    return {
        "name": "mamba3_full_block",
        "config": {
            "name": "mamba3_reference_fixture",
            "model_dim": dim,
            "vocab_size": vocab,
            "seq_len": steps,
            "blocks": [{
                "type": "mamba3-canonical",
                "inner_dim": inner,
                "state_size": state,
                "n_groups": groups,
                "dt_rank": rank,
                "conv_kernel": kernel,
                "use_conv": True,
                "scan_chunk_size": 0,
            }],
            "training": {
                "steps": 1,
                "batch_tokens": batch * steps,
                "seed": 17,
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
            {"name": "pre_norm_scale", "shape": [dim], "values": pre_norm},
            {"name": "w_x", "shape": [dim, inner], "values": w_x},
            {"name": "conv_w", "shape": [inner, kernel], "values": conv_w},
            {"name": "w_dt_low", "shape": [inner, rank], "values": w_dt_low},
            {"name": "w_dt_high", "shape": [rank, inner], "values": w_dt_high},
            {"name": "w_lambda_low", "shape": [inner, rank], "values": w_lambda_low},
            {"name": "w_lambda_high", "shape": [rank, inner], "values": w_lambda_high},
            {"name": "w_theta_low", "shape": [inner, rank], "values": w_theta_low},
            {"name": "w_theta_high", "shape": [rank, inner * (state // 2)], "values": w_theta_high},
            {"name": "w_B", "shape": [inner, group_state], "values": w_b},
            {"name": "w_C", "shape": [inner, group_state], "values": w_c},
            {"name": "B_norm_scale", "shape": [state], "values": b_norm},
            {"name": "C_norm_scale", "shape": [state], "values": c_norm},
            {"name": "B_bias", "shape": [group_state], "values": b_bias},
            {"name": "C_bias", "shape": [group_state], "values": c_bias},
            {"name": "A_log", "shape": [inner, state], "values": a_log},
            {"name": "dt_bias", "shape": [inner], "values": dt_bias},
            {"name": "post_norm_scale", "shape": [inner], "values": post_norm},
            {"name": "w_gate", "shape": [dim, inner], "values": w_gate},
            {"name": "w_out", "shape": [inner, dim], "values": w_out},
        ],
        "expected_hidden": expected,
    }


def main() -> None:
    payload = {
        "version": 1,
        "description": "Static full-block mamba3-canonical reference fixture generated from independent scalar reference loops.",
        "fixtures": [fixture()],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
