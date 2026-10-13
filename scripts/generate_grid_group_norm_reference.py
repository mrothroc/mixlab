"""Generate NHWC GroupNorm fixtures with PyTorch forward and all gradients.

Run from the repository root; existing spatial fixtures are not changed.
"""
import json
from pathlib import Path

import numpy as np
import torch

from generate_grid_reference import blob, nhwc


def fixture(batch, groups, mode, eps=1e-5):
    rng = np.random.default_rng(81 + batch + groups)
    values = rng.normal(size=(batch, 4, 3, 5)).astype("float32")
    if mode == "constant":
        values.fill(0.25)
    elif mode == "near_constant":
        values = 0.25 + values * 1e-3
    x = torch.tensor(values, requires_grad=True)
    model = torch.nn.GroupNorm(groups, 4, eps=eps)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([0.7, 1.2, -0.4, 1.5]))
        model.bias.copy_(torch.tensor([0.2, -0.1, 0.3, -0.4]))
    # Near-constant fp32 fused affine kernels themselves lose low bits when
    # subtracting the mean. Use float64 PyTorch as the mathematical oracle for
    # adversarial inputs, starting from exactly the same float32 input/weights.
    precision = "float32"
    if mode != "random":
        x = x.detach().double().requires_grad_()
        model = model.double()
        precision = "float64_from_float32_inputs"
    y = model(x)
    cot = torch.linspace(0.2, 1.7, y.numel()).reshape(y.shape).to(x.dtype)
    ((y * cot) ** 2).mean().backward()
    return {
        "name": f"group_norm_b{batch}g{groups}_{mode}_eps{eps}", "op": "group_norm",
        "groups": groups, "eps": model.eps, "oracle_precision": precision,
        "weights": [blob(nhwc(x)), blob(model.weight.detach()), blob(model.bias.detach())],
        "grads": [blob(nhwc(x.grad)), blob(model.weight.grad), blob(model.bias.grad)],
        "output": blob(nhwc(y)), "cotangent": blob(nhwc(cot)),
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    cases = [fixture(b, g, mode) for b in (1, 2) for g in (1, 2, 4)
             for mode in ("random", "constant", "near_constant")]
    cases += [fixture(1, 2, mode, eps=1e-3) for mode in ("random", "near_constant")]
    Path("gpu/testdata/grid_group_norm_reference.json").write_text(json.dumps({
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "loss": "mean((output * cotangent)^2)", "cases": cases,
    }, separators=(",", ":")) + "\n")
