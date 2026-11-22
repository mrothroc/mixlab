"""Independent PyTorch grid_sample twin for the generic polar example."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F


class PolarGridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radial = torch.nn.Conv2d(2, 2, (1, 3))
        self.angular = torch.nn.Conv2d(2, 2, (3, 1))
        self.fusion = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x, v):
        theta = torch.arange(16) * (2 * torch.pi / 16)
        radius = torch.arange(12)
        row = v[:, 0, None, None] + torch.sin(theta)[None, :, None] * radius
        col = v[:, 1, None, None] + torch.cos(theta)[None, :, None] * radius
        grid = torch.stack((col / 7 * 2 - 1, row / 7 * 2 - 1), -1)
        p = F.grid_sample(x, grid, align_corners=True, padding_mode="border")
        p = torch.relu(self.radial(F.pad(p, (1, 1, 0, 0), mode="replicate")))
        p = self.angular(F.pad(p, (0, 0, 1, 1), mode="circular"))
        dy = torch.arange(8)[None, :, None] - v[:, 0, None, None]
        dx = torch.arange(8)[None, None, :] - v[:, 1, None, None]
        angle = torch.atan2(dy, dx).remainder(2 * torch.pi) * (16 / (2 * torch.pi))
        angle = torch.where((dy == 0) & (dx == 0), 0., angle)
        rho = torch.sqrt(dx ** 2 + dy ** 2)
        # Duplicate ray zero after ray 15 so ordinary bilinear sampling wraps.
        p = torch.cat((p, p[:, :, :1, :]), 2)
        grid = torch.stack((rho / 11 * 2 - 1, angle / 16 * 2 - 1), -1)
        back = F.grid_sample(p, grid, align_corners=True)
        return self.fusion(torch.cat((x, back), 1))


def build_model():
    return PolarGridModel()


def main():
    package, reference = map(Path, sys.argv[1:])
    native = json.loads(reference.read_text())
    manifest = json.loads((package / "export.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        x = np.array(native["inputs"], np.float32).reshape(3, 8, 8, 2)
        np.save(root / "x.npy", x.transpose(0, 3, 1, 2))
        np.save(root / "v.npy", np.array(native["vectors"], np.float32).reshape(3, 2))
        cmd = [sys.executable, str(package / "infer.py"), "--model-file", __file__,
               "--factory", "build_model", "--native-output", manifest["selected_output"],
               "--input", str(root / "x.npy"), "--vectors", str(root / "v.npy"),
               "--output", str(root / "y.npy"), "--batch-size", "2"]
        subprocess.run(cmd, check=True)
        got = np.load(root / "y.npy").reshape(-1)
        want = np.array(native["predictions"], np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=2e-6)
        print(f"polar export: max_abs={np.max(np.abs(got-want)):.9g}")
        (root / "y.npy").unlink()
        np.save(root / "v.npy", np.array([[0, 0], [8, 0], [0, 0]], np.float32))
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0 and "polar transmitter" in result.stderr
        assert not (root / "y.npy").exists()


if __name__ == "__main__":
    main()
