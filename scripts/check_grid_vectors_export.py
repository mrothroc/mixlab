"""Round-trip the tiny per-record-vector graph through the exported CLI helper."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch


class VectorGridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, v):
        return self.conv(x + v.reshape(-1, 1, 1, 1))


def build_model():
    return VectorGridModel()


def main():
    package, reference = map(Path, sys.argv[1:])
    native = json.loads(reference.read_text())
    manifest = json.loads((package / "export.json").read_text())
    assert (manifest["vector_dim"], manifest["vector_dtype"], manifest["vector_layout"]) == (1, "float32", "NS")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        np.save(root / "x.npy", np.array(native["inputs"], np.float32).reshape(-1, 1, 1, 1))
        np.save(root / "v.npy", np.array(native["vectors"], np.float32).reshape(-1, 1))
        subprocess.run([sys.executable, str(package / "infer.py"), "--model-file", __file__,
                        "--factory", "build_model", "--native-output", manifest["selected_output"],
                        "--input", str(root / "x.npy"), "--vectors", str(root / "v.npy"),
                        "--output", str(root / "y.npy"), "--batch-size", "2"], check=True)
        got = np.load(root / "y.npy").reshape(-1)
        want = np.array(native["predictions"], np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        print(f"vector export model(x,v): max_abs={np.max(np.abs(got-want)):.9g}")


if __name__ == "__main__":
    main()
