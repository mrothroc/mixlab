"""Grid writer contract tests, included by the repository CI discovery pattern."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from prepare_grid import prepare_grid, _validate_record_size


class GridPrepareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.x = np.linspace(-1, 1, 36, dtype=np.float32).reshape(3, 2, 2, 3)
        self.x[0, 0, 0, 0] = -0.
        self.y = self.x.copy()
        self.mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8).reshape(1, 1, 2, 3).repeat(3, axis=0)
        self.save_arrays()
        self.spec = dict(records_per_shard=2, splits={"train":dict(inputs="x.npy", targets="y.npy", masks="m.npy"),
                                                    "val":dict(inputs="x.npy", targets="y.npy", masks="m.npy")})

    def save_arrays(self):
        for name,a in (("x",self.x),("y",self.y),("m",self.mask)): np.save(self.root/f"{name}.npy",a)

    def run_prepare(self):
        source=self.root/"source.json";source.write_text(json.dumps(self.spec))
        return prepare_grid(source,self.root/"out")

    def test_float32_and_half_roundtrip(self):
        for dtype,code in (("float32",1),("float16",2)):
            self.spec["dtype"]=dtype
            result=self.run_prepare()
            self.assertEqual(result["splits"]["train"]["sequences"],3)
            raw=(self.root/"out/train_00000.grid").read_bytes();header=struct.unpack("<256I",raw[:1024])
            self.assertEqual(header[:8],(20260930,1,code,2,2,3,2,2))
            arr=np.frombuffer(raw,dtype="<f4" if code==1 else "<f2",count=12,offset=1024+header[8])
            self.assertEqual(arr.tobytes(),self.x[0].astype(dtype).tobytes())
            self.assertTrue(np.signbit(arr[0]))
            for path in (self.root/"out").iterdir():path.unlink()

    def test_metadata(self):
        (self.root/"groups.json").write_text('["terrain0","terrain0","terrain1"]')
        self.spec["splits"]["train"]["groups"]="groups.json"
        self.spec["normalization"]=dict(fit_split="train",input_offset=[0,0],input_scale=[1,1],target_offset=[0,0],target_scale=[2,2])
        m=self.run_prepare()
        self.assertEqual(m["grid_provenance"]["normalization"],self.spec["normalization"])
        self.assertEqual(m["grid_provenance"]["groups"]["train"][2],"terrain1")

    def test_reject_validation_fitted_normalization(self):
        self.spec["normalization"]=dict(fit_split="val")
        with self.assertRaisesRegex(ValueError,"train only"):self.run_prepare()

    def test_reject_masks_and_clean_partial_outputs(self):
        self.mask[2,0,0,0]=2;self.save_arrays()
        with self.assertRaisesRegex(ValueError,"masks"):self.run_prepare()
        self.assertEqual(list((self.root/"out").iterdir()),[])

    def test_reject_conversion_overflow(self):
        self.spec["dtype"]="float16";self.x[0,0,0,0]=1e10;self.save_arrays()
        with self.assertRaisesRegex(ValueError,"overflow"):self.run_prepare()

    def test_reject_nonfinite(self):
        self.y[2,0,0,0]=np.nan;self.save_arrays()
        with self.assertRaisesRegex(ValueError,"nonfinite"):self.run_prepare()

    def test_input_only(self):
        self.spec["splits"]={"predict":{"inputs":"x.npy"}}
        self.assertEqual(self.run_prepare()["grid"]["target_channels"],0)

    def test_record_size_limits_without_allocating_payloads(self):
        for dtype, width in (("float32", 8192), ("float16", 16384)):
            with self.subTest(dtype=dtype):
                _validate_record_size(1, 16384, width, 0, dtype)
                with self.assertRaisesRegex(ValueError, "overflow"):
                    _validate_record_size(1, 16384, width + 1, 0, dtype)
                # Two planes fill the limit before adding the required mask.
                with self.assertRaisesRegex(ValueError, "overflow"):
                    _validate_record_size(1, 8192, width, 1, dtype)

    def test_reject_duplicates(self):
        (self.root/"ids.json").write_text('["same","same","other"]')
        self.spec["splits"]["train"]["ids"]="ids.json"
        with self.assertRaisesRegex(ValueError,"unique"):self.run_prepare()

    def test_refuse_overwrite(self):
        self.run_prepare()
        with self.assertRaisesRegex(ValueError,"empty"):self.run_prepare()

    def test_split_patterns_do_not_overlap(self):
        entry = dict(inputs="x.npy", targets="y.npy", masks="m.npy")
        self.spec["splits"] = {name: entry for name in ("train", "train_extra", "train_1", "val")}
        manifest = self.run_prepare()
        for name, split in manifest["splits"].items():
            paths = sorted((self.root / "out").glob(split["pattern"]))
            self.assertEqual(len(paths), split["shards"], name)
            for path in paths:
                raw = path.read_bytes()
                header = struct.unpack("<256I", raw[:1024])
                ids = json.loads(raw[1024:1024 + header[8]])
                self.assertTrue(all(value.startswith(name + "_") for value in ids))


if __name__ == "__main__": unittest.main()
