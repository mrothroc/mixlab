"""Gated export acceptance against the pinned, unmodified reference forward."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file

from generate_grid_model_reference import SOURCE_SHA256


def check(source, package, inputs, native, stage, report, native_output):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("reference source hash mismatch")
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "torchvision")]
    ns = {}
    exec(compile(tree, str(source), "exec"), ns)
    x = np.load(inputs, allow_pickle=False)
    expected = np.load(native, allow_pickle=False)
    assert x.shape[0] == 4 and x.shape[2:] == (256, 256)
    torch.set_num_threads(4)
    model = ns["RadioWNet"](inputs=x.shape[1], phase="firstU" if stage == 1 else "secondU").float().eval()
    model.load_state_dict(load_file(str(Path(package) / "model.safetensors")), strict=True)
    manifest = json.loads((Path(package) / "export.json").read_text())
    assert manifest["format"] == "mixlab.torch_state_export.v1"
    assert manifest["selected_output"] == native_output
    assert (manifest["input_channels"], manifest["height"], manifest["width"]) == (x.shape[1], 256, 256)
    with torch.inference_mode():
        predictions = []
        for row in x:
            outputs = model(torch.from_numpy(row[None].copy()))
            assert isinstance(outputs, (tuple, list)) and len(outputs) == 2
            predictions.append(outputs[stage-1].numpy())
        got = np.concatenate(predictions)
    assert got.shape == expected.shape
    assert np.isfinite(got).all() and np.isfinite(expected).all()
    assert got.min() >= 0 and expected.min() >= 0
    diff = float(np.max(np.abs(got-expected)))
    # Exercise the shipped entry point with explicitly trusted, hash-verified
    # reference code. AST printing changes formatting, not model semantics.
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        trusted = directory / "trusted_model.py"
        trusted.write_text(ast.unparse(tree) + "\n")
        helper_output = directory / "predictions.npy"
        subprocess.run([sys.executable, str(Path(package) / "infer.py"),
            "--model-file", str(trusted), "--factory", "RadioWNet",
            "--kwargs-json", json.dumps(dict(inputs=x.shape[1], phase="firstU" if stage == 1 else "secondU")),
            "--native-output", native_output, "--output-index", str(stage-1),
            "--input", str(inputs), "--output", str(helper_output), "--batch-size", "2",
            "--target-offset", "-10", "--target-scale", "2"], check=True)
        helper_diff = float(np.max(np.abs(np.load(helper_output) - (expected*2-10))))
        if helper_diff > 2e-4:
            raise AssertionError(f"packaged batch inference mismatch {helper_diff}")
    result = dict(max_abs=diff, helper_max_abs=helper_diff, threshold=1e-4, samples=4, stage=stage, channels=x.shape[1], source_sha256=SOURCE_SHA256,
                  torch=torch.__version__, numpy=np.__version__, strict_load=True)
    Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if diff > 1e-4:
        raise AssertionError(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "package", "inputs", "native", "report", "native-output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--stage", type=int, choices=(1,2), required=True)
    a = p.parse_args()
    check(a.source, a.package, a.inputs, a.native, a.stage, a.report, a.native_output)
