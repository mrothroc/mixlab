"""Independent float64 grid_sample forward/autograd fixtures for polar back-maps."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def blob(x):
    if x.ndim == 4:
        x = x.permute(0, 2, 3, 1)
    return {"shape": list(x.shape), "data": x.detach().reshape(-1).tolist()}


def case(k, dr):
    h, w = 4, 5
    j = int(np.ceil(np.hypot(h-1, w-1) / dr)) + 2
    rng = np.random.default_rng(412)
    x = torch.tensor(rng.normal(size=(2, 2, k, j)).astype(np.float32), dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[1.25, 2.5], [0., 4.]], dtype=torch.float64)
    dy = torch.arange(h, dtype=torch.float64)[None, :, None] - v[:, 0, None, None]
    dx = torch.arange(w, dtype=torch.float64)[None, None, :] - v[:, 1, None, None]
    a = torch.atan2(dy, dx).remainder(2 * torch.pi) * (k / (2 * torch.pi))
    rho = torch.sqrt(dy**2 + dx**2) / dr
    grid = torch.stack((rho / (j-1) * 2 - 1, a / k * 2 - 1), -1)
    y = F.grid_sample(torch.cat((x, x[:, :, :1]), 2), grid, align_corners=True)
    cot = torch.tensor(rng.normal(size=y.shape).astype(np.float32), dtype=torch.float64)
    ((y*cot)**2).mean().backward()
    return dict(name=f"polar_k{k}_dr{dr}", op="polar_to_cartesian", params=[k,j,h,w,0,1], dr=dr,
                vectors=blob(v), weights=[blob(x)], grads=[blob(x.grad)], output=blob(y), cotangent=blob(cot))


if __name__ == "__main__":
    torch.set_num_threads(1)
    path = Path("gpu/testdata/grid_polar_reference.json")
    path.write_text(json.dumps(dict(producer="PyTorch float64 grid_sample", torch_version=torch.__version__,
                                   cases=[case(8, 1.), case(16, .5), case(1, 1.)]), separators=(",", ":"))+"\n")
    print(path)
