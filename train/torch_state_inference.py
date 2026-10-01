"""Explicit trusted-model loader for a Mixlab torch-state package.

Dependencies: numpy, torch, safetensors. --model-file is executable Python chosen
by the caller; only load code you trust. Export maps and checkpoint data never
select a module or execute code. Inputs are already normalized NCHW float arrays.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from safetensors.torch import load_file


def inverse_normalize(values, offset, scale):
    channels = values.shape[1]
    offset = np.asarray(offset, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if offset.size not in (1, channels) or scale.size not in (1, channels):
        raise ValueError("normalization must have one value or one per output channel")
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("normalization must be finite with positive scales")
    return values * scale.reshape(1, -1, 1, 1) + offset.reshape(1, -1, 1, 1)


def run(args):
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / "export.json").read_text())
    if manifest["format"] != "mixlab.torch_state_export.v1":
        raise ValueError("unsupported package")
    if args.native_output != manifest["selected_output"]:
        raise ValueError("--native-output must match the package selected_output")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    kwargs = json.loads(args.kwargs_json)
    if not isinstance(kwargs, dict) or not args.factory.isidentifier():
        raise ValueError("factory must be an identifier and kwargs-json an object")
    spec = importlib.util.spec_from_file_location("user_grid_model", args.model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, args.factory)(**kwargs).cpu().float().eval()
    model.load_state_dict(load_file(str(package / "model.safetensors"), device="cpu"), strict=True)
    x = np.load(args.input, mmap_mode="r", allow_pickle=False)
    expected = (manifest["input_channels"], manifest["height"], manifest["width"])
    if x.ndim != 4 or tuple(x.shape[1:]) != expected or len(x) == 0 or x.dtype.kind != "f":
        raise ValueError("input must be nonempty floating NCHW matching export.json")
    dest = Path(args.output)
    if dest.exists():
        raise ValueError("output must not already exist")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".torch-infer-", suffix=".npy", dir=dest.parent)
    os.close(fd)
    out = None
    try:
        shape = (len(x), manifest["target_channels"], manifest["height"], manifest["width"])
        out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        with torch.inference_mode():
            for start in range(0, len(x), args.batch_size):
                batch = np.array(x[start:start+args.batch_size], dtype=np.float32, copy=True)
                if not np.isfinite(batch).all():
                    raise ValueError("non-finite input")
                y = model(torch.from_numpy(batch))
                if isinstance(y, (tuple, list)):
                    if args.output_index is None or not 0 <= args.output_index < len(y):
                        raise ValueError("tuple output requires explicit in-range --output-index")
                    y = y[args.output_index]
                elif args.output_index is not None:
                    raise ValueError("output-index is only valid for tuple/list outputs")
                y = y.detach().cpu().float().numpy()
                if y.shape != (len(batch), *shape[1:]) or not np.isfinite(y).all():
                    raise ValueError("invalid model output shape or non-finite values")
                y = inverse_normalize(y, args.target_offset, args.target_scale)
                if not np.isfinite(y).all():
                    raise ValueError("non-finite denormalized output")
                out[start:start+len(batch)] = y
        out.flush()
        del out
        out = None
        with open(temporary, "rb") as completed:
            os.fsync(completed.fileno())
        # Same-filesystem hard link is an atomic no-overwrite publication.
        os.link(temporary, dest)
    finally:
        if out is not None:
            del out
        os.unlink(temporary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-file", required=True)
    p.add_argument("--factory", required=True)
    p.add_argument("--kwargs-json", default="{}")
    p.add_argument("--native-output", required=True)
    p.add_argument("--output-index", type=int)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--target-offset", type=float, nargs="+", default=[0.0])
    p.add_argument("--target-scale", type=float, nargs="+", default=[1.0])
    run(p.parse_args())


if __name__ == "__main__":
    main()
