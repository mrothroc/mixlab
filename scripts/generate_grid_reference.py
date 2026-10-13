"""Generate small fp32 PyTorch spatial forward/backward fixtures.

Run in an isolated environment with torch and numpy. Fixtures use explicit
weights, not assumptions about matching PRNG streams across frameworks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def nhwc(value):
    return value.detach().permute(0, 2, 3, 1).contiguous()


def blob(value):
    return {"shape": list(value.shape), "data": value.flatten().tolist()}


def fixture(kind, kernel, stride, padding, tied=False, dilation=1, bias=True):
    rng = np.random.default_rng(42 + kernel)
    x = torch.tensor(rng.normal(size=(1, 2, 5, 7)).astype("float32"), requires_grad=True)
    if tied:
        x = torch.ones_like(x).requires_grad_()
    weights = [x]
    if kind == "layout":
        y = torch.cat([F.relu(x), x.detach()], dim=1).transpose(2, 3)[:, :, 1:4, :]
    elif kind == "max_pool2d":
        y = F.max_pool2d(x, kernel, stride)
    else:
        shape = (3, 2, kernel, kernel) if kind == "conv2d" else (2, 3, kernel, kernel)
        w = torch.tensor((rng.normal(size=shape) * 0.1).astype("float32"), requires_grad=True)
        b = torch.tensor([0.2, -0.4, 0.6], requires_grad=True)
        weights.append(w)
        if bias:
            weights.append(b)
        op = F.conv2d if kind == "conv2d" else F.conv_transpose2d
        y = op(x, w, b if bias else None, stride=stride, padding=padding, dilation=dilation)
    cot = torch.linspace(0.2, 1.7, y.numel()).reshape(y.shape)
    loss = ((y * cot) ** 2).mean()
    loss.backward()
    canonical = [nhwc(x)]
    grads = [nhwc(x.grad)]
    if len(weights) > 1:
        canonical.append(weights[1].detach().permute(0, 2, 3, 1).contiguous())
        grads.append(weights[1].grad.permute(0, 2, 3, 1).contiguous())
        if bias:
            canonical.append(weights[2].detach())
            grads.append(weights[2].grad)
    return {
        "name": f"{kind}_k{kernel}s{stride}p{padding}" + ("_ties" if tied else "") + (f"_d{dilation}" if dilation != 1 else "") + ("_no_bias" if not bias else ""),
        "op": kind, "kernel": kernel, "stride": stride, "padding": padding, "dilation": dilation,
        "weights": [blob(w) for w in canonical], "grads": [blob(g) for g in grads],
        "output": blob(nhwc(y)), "cotangent": blob(nhwc(cot)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gpu/testdata/grid_reference.json"))
    parser.add_argument("--dilation-only", action="store_true", help="Generate the separate dilated-convolution fixture suite")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = [fixture("conv2d", k, 1, k // 2) for k in (3, 5)]
    cases += [fixture("conv_transpose2d", 4, 2, 1), fixture("conv_transpose2d", 6, 2, 2)]
    cases += [fixture("max_pool2d", k, k, 0, tied) for k, tied in ((1, False), (2, False), (2, True))]
    cases += [fixture("layout", 1, 1, 0)]
    if args.dilation_only:
        cases = [fixture("conv2d", 3, stride, d, dilation=d, bias=bias)
                 for d in (2, 4) for stride, bias in ((1, True), (2, False))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"torch_version": torch.__version__, "numpy_version": np.__version__,
                                   "loss": "mean((output * cotangent)^2)", "cases": cases}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
