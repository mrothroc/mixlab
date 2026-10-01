"""Streaming fixed-shape grid records. Layout is documented in docs/dense-grid.md."""
import json
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np


def _strict_keys(value, allowed, scope):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{scope}: invalid/unknown fields")


def _validate_record_size(c, h, w, ct, dtype):
    elements = h * w * (c + ct)
    size = elements * (4 if dtype == "float32" else 2)
    if ct > 0:
        size += (h * w + 7) // 8
    if elements > 2**31 - 1 or size > 512 << 20:
        raise ValueError("grid dimensions overflow")


def prepare_grid(source, output):
    source = Path(source)
    spec = json.loads(source.read_text())
    _strict_keys(spec, ("dtype", "records_per_shard", "splits", "normalization"), "grid source")
    dtype = spec.get("dtype", "float32")
    if dtype not in ("float32", "float16"):
        raise ValueError("grid dtype must be float32 or float16")
    per_shard = spec.get("records_per_shard", 64)
    if not isinstance(per_shard, int) or not 1 <= per_shard <= 100000:
        raise ValueError("invalid records_per_shard")
    splits = spec.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("grid source requires explicit splits")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("grid prepare output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "mixlab.dataset", "version": 1, "representation": "grid", "modality": "image",
                "feature_dtype": dtype, "shard_format": "mixlab_grid_shard_v1",
                "task": {"type": "dense_regression"}, "splits": {}}
    geometry = None
    provenance = {"groups": {}}
    created = []
    try:
        for split, ent in splits.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", split):
                raise ValueError("invalid grid split name")
            _strict_keys(ent, ("inputs", "targets", "masks", "ids", "groups"), f"split {split}")
            def load(key):
                return np.load(source.parent / ent[key], mmap_mode="r", allow_pickle=False)
            x = load("inputs")
            if x.ndim != 4 or min(x.shape) <= 0 or x.dtype not in (np.float32, np.float16):
                raise ValueError("grid inputs require finite floating [N,C,H,W]")
            n, c, h, w = x.shape
            y = load("targets") if "targets" in ent else None
            mask = load("masks") if "masks" in ent else None
            if (y is None) != (mask is None):
                raise ValueError("targets and masks must be supplied together")
            ct = 0 if y is None else y.shape[1] if y.ndim == 4 else 0
            if y is not None and (ct <= 0 or y.shape != (n, ct, h, w) or mask.shape != (n, 1, h, w)
                                  or y.dtype not in (np.float32, np.float16)):
                raise ValueError("grid targets/masks shape or dtype mismatch")
            current = dict(channels=c, height=h, width=w, target_channels=ct)
            if geometry is not None and current != geometry:
                raise ValueError("grid split geometry mismatch")
            geometry = current
            _validate_record_size(c, h, w, ct, dtype)
            if "normalization" in spec:
                norm = spec["normalization"]
                _strict_keys(norm, ("fit_split", "input_offset", "input_scale", "target_offset", "target_scale"), "normalization")
                if norm.get("fit_split") != "train":
                    raise ValueError("normalization must be fitted on train only")
                for key, size in (("input_offset", c), ("input_scale", c), ("target_offset", ct), ("target_scale", ct)):
                    values = np.asarray(norm.get(key, []), dtype=np.float64)
                    if values.shape != (size,) or not np.isfinite(values).all() or (key.endswith("scale") and not (values > 0).all()):
                        raise ValueError("normalization channel count or value invalid")
                provenance["normalization"] = norm
            if "groups" in ent:
                groups = json.loads((source.parent / ent["groups"]).read_text())
                if not isinstance(groups, list) or len(groups) != n or any(not isinstance(g, str) or not g.strip() for g in groups):
                    raise ValueError("groups require one nonempty string per record")
                provenance["groups"][split] = groups
            ids = json.loads((source.parent / ent["ids"]).read_text()) if "ids" in ent else [f"{split}_{j}" for j in range(n)]
            if (not isinstance(ids, list) or len(ids) != n or
                    any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != n):
                raise ValueError("grid ids must be unique nonempty strings, one per record")
            files = 0
            shard_digits = max(5, len(str((n - 1) // per_shard)))
            for start in range(0, n, per_shard):
                end = min(start+per_shard, n)
                meta = json.dumps(ids[start:end], ensure_ascii=True).encode()
                if len(meta) > 16 << 20:
                    raise ValueError("grid ID metadata exceeds limit")
                header = [20260930, 1, 1 if dtype == "float32" else 2, c, h, w, ct, end-start, len(meta)] + [0]*247
                path = output / f"{split}_{files:0{shard_digits}d}.grid"
                fd, temporary = tempfile.mkstemp(dir=output, prefix=".grid-")
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(struct.pack("<256I", *header))
                        f.write(meta)
                        for j in range(start, end):
                            for array in (x[j],) if y is None else (x[j], y[j]):
                                if not np.isfinite(array).all():
                                    raise ValueError(f"nonfinite grid values at {split}[{j}]")
                                with np.errstate(over="ignore"):
                                    stored = array.astype("<f4" if dtype == "float32" else "<f2")
                                if not np.isfinite(stored).all():
                                    raise ValueError("float16 grid conversion overflow")
                                f.write(stored.tobytes(order="C"))
                            if mask is not None:
                                m = mask[j].reshape(-1)
                                if not ((m == 0) | (m == 1)).all():
                                    raise ValueError("grid masks must contain only 0 and 1")
                                f.write(np.packbits(m.astype(np.uint8), bitorder="little").tobytes())
                    os.replace(temporary, path)
                    created.append(path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                files += 1
            pattern = f"{split}_" + "[0-9]" * shard_digits + ".grid"
            manifest["splits"][split] = {"pattern": pattern, "shards": files, "sequences": n, "tokens": 0}
        manifest["grid"] = geometry
        if provenance["groups"] or "normalization" in provenance:
            manifest["grid_provenance"] = provenance
        path = output / "mixlab.dataset.json"
        fd, temporary = tempfile.mkstemp(dir=output, prefix=".manifest-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return manifest
