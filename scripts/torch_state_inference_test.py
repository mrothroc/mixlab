import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

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

    def test_vector_contract_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = Path(__file__).resolve().parents[1] / "train/torch_state_inference.py"
            (root / "infer.py").write_bytes(helper.read_bytes())
            spec = importlib.util.spec_from_file_location("vector_infer", root / "infer.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            (root / "model.py").write_text(
                "import torch\nclass Model(torch.nn.Module):\n"
                "    def __init__(self):\n        super().__init__()\n        self.scale = torch.nn.Parameter(torch.tensor(1.0))\n"
                "    def forward(self,x,v):\n        return self.scale*x + v[:,0,None,None,None]\n"
                "def model():\n    return Model()\n")
            manifest = dict(format="mixlab.torch_state_export.v1", selected_output="net.out",
                            input_channels=1, target_channels=1, height=2, width=2,
                            vector_dim=2, vector_dtype="float32", vector_layout="NS")
            (root / "export.json").write_text(json.dumps(manifest))
            save_file({"scale": torch.tensor(2.0)}, root / "model.safetensors")
            x = np.arange(12, dtype=np.float32).reshape(3,1,2,2)
            v = np.array([[.125, 2], [3.5, 4], [-1.25, 6]], np.float32)
            np.save(root / "x.npy", x)
            np.save(root / "v.npy", v)
            args = SimpleNamespace(native_output="net.out", batch_size=2, kwargs_json="{}", factory="model",
                                   model_file=str(root / "model.py"), input=root / "x.npy", vectors=root / "v.npy",
                                   output=root / "out.npy", output_index=None, target_offset=[0.0], target_scale=[1.0])
            module.run(args)
            np.testing.assert_array_equal(np.load(args.output), 2*x + v[:,0,None,None,None])
            args.output.unlink()
            args.vectors = None
            with self.assertRaisesRegex(ValueError, "--vectors is required"):
                module.run(args)
            args.vectors = root / "v.npy"
            for wrong in (v[:2], v[:,:1], v.astype(np.float16), v.astype(np.float64), v.reshape(3,1,2)):
                np.save(args.vectors, wrong)
                with self.assertRaisesRegex(ValueError, "float32 \\[N,S\\]"):
                    module.run(args)
            for nonfinite in (np.nan, np.inf):
                bad = v.copy()
                bad[2,1] = nonfinite  # fail after the first output chunk
                np.save(args.vectors, bad)
                with self.assertRaisesRegex(ValueError, "non-finite vectors in rows 2:3"):
                    module.run(args)
                self.assertFalse(args.output.exists())
                self.assertEqual(list(root.glob(".torch-infer-*")), [])
            manifest.pop("vector_dim")
            (root / "export.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "requires a package with vector_dim"):
                module.run(args)


if __name__ == "__main__":
    unittest.main()
