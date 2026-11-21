"""Check rectangular kernels and spatial padding via the packaged infer.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F


class RectangularGridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.row = torch.nn.Conv2d(2, 2, (1, 9))
        self.col = torch.nn.Conv2d(2, 1, (9, 1))

    def forward(self, x):
        x = torch.relu(self.row(F.pad(x, (4, 4, 0, 0), mode="circular")))
        return self.col(F.pad(x, (0, 0, 4, 4), mode="replicate"))


def build_model():
    return RectangularGridModel()


def main():
    package, reference = map(Path, sys.argv[1:])
    native = json.loads(reference.read_text())
    manifest = json.loads((package / "export.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        x = np.array(native["inputs"], np.float32).reshape(2, 8, 8, 2)
        np.save(root / "x.npy", x.transpose(0, 3, 1, 2))
        subprocess.run([sys.executable, str(package / "infer.py"), "--model-file", __file__,
                        "--factory", "build_model", "--native-output", manifest["selected_output"],
                        "--input", str(root / "x.npy"), "--output", str(root / "y.npy"),
                        "--batch-size", "1"], check=True)
        got = np.load(root / "y.npy").reshape(-1)
        want = np.array(native["predictions"], np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=2e-6)
        print(f"rectangular/pad export: max_abs={np.max(np.abs(got-want)):.9g}")


if __name__ == "__main__":
    main()
