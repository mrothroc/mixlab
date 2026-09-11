"""Exercise the installed CLI's embedded prepare path without a GPU or checkout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def check(binary):
    binary = str(Path(binary).resolve())
    with tempfile.TemporaryDirectory(prefix="mixlab-container-prepare-") as tmp:
        root = Path(tmp)
        env = dict(os.environ)
        env.pop("MIXLAB_SCRIPTS", None)

        def prepare(name, *args):
            out = root / name
            subprocess.run([binary, "-mode", "prepare", "-prepare-output-dir", str(out), *args],
                           cwd=root, env=env, check=True, timeout=60)
            if not list(out.glob("train_*.bin")):
                raise AssertionError(f"{name}: no training shards")
            manifest = json.loads((out / "mixlab.dataset.json").read_text())
            return out, manifest

        corpus = root / "input.txt"
        corpus.write_text("red green blue yellow. " * 100)
        # Omitting val-split really reserves the documented default 10 percent.
        text_out, manifest = prepare("text", "-input", str(corpus), "-vocab-size", "64")
        assert manifest["modality"] == "text"
        assert (text_out / "tokenizer.json").is_file()
        assert list(text_out.glob("val_*.bin"))

        arrays = root / "input.npy"
        np.save(arrays, np.arange(32, dtype=np.float32).reshape(4, 8, 1))
        labels = root / "labels.tsv"
        labels.write_text("0\t0\n1\t0\n2\t1\n3\t1\n")
        continuous_out, manifest = prepare("continuous", "-input", str(arrays),
            "-input-format", "continuous", "-label-file", str(labels), "-val-split", "0")
        assert manifest["representation"] == "continuous_frames"
        assert manifest["feature_dim"] == 1
        assert not list(continuous_out.glob("val_*.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--stamp", type=Path)
    args = parser.parse_args()
    check(args.binary)
    if args.stamp:
        args.stamp.write_text("embedded text and continuous prepare passed\n")
