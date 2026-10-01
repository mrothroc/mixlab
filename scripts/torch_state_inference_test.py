import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from safetensors.torch import save_file
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy/PyTorch/safetensors required")
class TorchStateInferenceTest(unittest.TestCase):
    def test_batched_strict_loading_and_inverse_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = Path(__file__).resolve().parents[1] / "train/torch_state_inference.py"
            (root / "infer.py").write_bytes(helper.read_bytes())
            (root / "model.py").write_text("import torch\ndef model():\n    return torch.nn.Conv2d(2,1,1)\n")
            (root / "export.json").write_text(json.dumps(dict(format="mixlab.torch_state_export.v1", selected_output="network.output", input_channels=2, target_channels=1, height=2, width=2)))
            save_file({"weight": torch.ones(1,2,1,1), "bias": torch.tensor([-3.0])}, root / "model.safetensors")
            x = np.arange(24, dtype=np.float32).reshape(3,2,2,2)
            np.save(root / "input.npy", x)
            cmd = [sys.executable, str(root / "infer.py"), "--model-file", str(root / "model.py"), "--factory", "model", "--native-output", "network.output", "--input", str(root / "input.npy"), "--output", str(root / "out.npy"), "--batch-size", "2", "--target-scale", "2", "--target-offset", "-10"]
            subprocess.run(cmd, check=True, capture_output=True)
            np.testing.assert_array_equal(np.load(root / "out.npy"), (x.sum(axis=1,keepdims=True)-3)*2-10)
            self.assertLess(np.load(root / "out.npy").min(), 0)  # no exporter-added clamp
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(list(root.glob(".torch-infer-*")), [])
            (root / "out.npy").unlink()
            save_file({"weight": torch.ones(1,2,1,1)}, root / "model.safetensors")
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertFalse((root / "out.npy").exists())

    def test_normalization_validation(self):
        path = Path(__file__).resolve().parents[1] / "train/torch_state_inference.py"
        spec = importlib.util.spec_from_file_location("grid_infer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for offset, scale in (([0],[0]), ([0],[float("nan")]), ([0,1,2],[1])):
            with self.assertRaises(ValueError):
                module.inverse_normalize(np.zeros((1,2,2,2)), offset, scale)


if __name__ == "__main__":
    unittest.main()
