"""Validate optional native GPU TF32 probe dumps against PyTorch float64.

Run native TestGPUPrecisionSubprocess (on CUDA also MIXLAB_TEST_CUDA_PRECISION=1) with
MIXLAB_TF32_PROBE_OUT=/tmp/probes, then invoke this script on that directory.
PyTorch is an optional acceptance-test dependency, not a Mixlab runtime dependency.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def validate(path):
    data = json.loads(path.read_text())

    def tensor(values, shape):
        # JSON contains float32 payloads: restore that exact value before widening.
        return torch.tensor(values, dtype=torch.float32).double().reshape(shape).requires_grad_()

    conv = data["conv"]
    x = tensor(data["x"], (1, 8, 8, 32) if conv else (64, 32))
    w = tensor(data["w"], (32, 3, 3, 32) if conv else (32, 32))
    b = tensor(data["b"], (32,))
    y = x
    for _ in range(data["depth"]):
        y = F.conv2d(y.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2), b, padding=1).permute(0, 2, 3, 1) if conv else y @ w
    loss = y.square().mean()
    loss.backward()
    pairs = [("output", data["output"], y), ("loss", [data["loss"]], loss)]
    for i, v in enumerate((x, w, b)):
        pairs.append((f"gradient{i}", data["gradients"][i], v.grad if v.grad is not None else torch.zeros_like(v)))
    result = {"case": path.name, "reference": "torch.float64", "torch_version": torch.__version__}
    for name, actual, expected in pairs:
        a = torch.tensor(actual, dtype=torch.float32).double().flatten()
        e = expected.detach().flatten()
        error = float((a - e).abs().max() / e.abs().max().clamp(min=1e-20))
        result[name] = error
        limit = 0.02 if data["tf32_allowed"] else 2e-5
        if not torch.isfinite(a).all() or error > limit:
            raise AssertionError(f"{path.name} {name}: {error} > {limit}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.glob("conv-*-depth-*-tf32-*.json"))
    if not files:
        parser.error("no native precision probe dumps found")
    for file in files:
        print(json.dumps(validate(file), sort_keys=True))
