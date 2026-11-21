"""Independent float64 PyTorch oracles for per-axis convolution and NHWC pad.

Inputs/weights/cotangents are exactly float32-representable, then promoted for
the oracle. Stored gradients match mean((output * cotangent)**2).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def blob(t):
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)
    return dict(shape=list(t.shape), data=t.detach().reshape(-1).tolist())


def tensor(rng, shape, scale=1):
    return torch.tensor((rng.normal(size=shape)*scale).astype(np.float32), dtype=torch.float64, requires_grad=True)


def case(name, kind, params, bias=True):
    rng = np.random.default_rng(915)
    x = tensor(rng, (2, 2, 11, 21) if kind == "conv2d" else (2, 2, 3, 4))
    weights = [x]
    if kind == "conv2d":
        kh, kw, sh, sw, ph, pw, dh, dw = params
        w, b = tensor(rng, (3, 2, kh, kw), .1), tensor(rng, (3,), .1)
        weights.append(w)
        if bias:
            weights.append(b)
        y = F.conv2d(x, w, b if bias else None, (sh, sw), (ph, pw), (dh, dw))
    else:
        before_h, after_h, before_w, after_w, mode = params
        y = F.pad(x, (before_w, after_w, before_h, after_h), mode=("constant", "replicate", "circular")[mode])
    cot = tensor(rng, tuple(y.shape)).detach()
    ((y*cot)**2).mean().backward()
    return dict(name=name, op=kind, params=params, weights=[blob(v) for v in weights],
                grads=[blob(v.grad) for v in weights], output=blob(y), cotangent=blob(cot))


def main():
    torch.set_num_threads(1)
    cases = [case(name, "conv2d", params, bias) for name, params, bias in [
        ("rect_1x9", [1,9,1,1,0,4,1,1], True),
        ("rect_9x1", [9,1,1,1,4,0,1,1], True),
        ("rect_1x3_d1x8", [1,3,1,1,0,8,1,8], True),
        ("rect_stride_padding", [3,5,2,3,1,4,2,1], True),
        ("rect_no_bias", [2,3,2,1,0,2,1,2], False),
    ]]
    for mode in range(3):
        for label, extents in [("identity", [0,0,0,0]), ("asymmetric", [1,2,0,3]), ("equal_axis", [3,3,4,4])]:
            cases.append(case(f"pad_{mode}_{label}", "pad", [*extents,mode]))
    cases.append(case("pad_edge_large", "pad", [5,1,6,0,1]))
    path = Path("gpu/testdata/grid_rectangular_reference.json")
    path.write_text(json.dumps(dict(producer="PyTorch float64", torch_version=torch.__version__, cases=cases), separators=(",", ":"))+"\n")
    print(path, len(cases), "cases")


if __name__ == "__main__":
    main()
