import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from replay import compare, load_tensors, outer_step, replay


def artifact(root, value, kind):
    payload = np.array([value], dtype="<f4").tobytes()
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    raw = struct.pack("<Q", len(header)) + header + payload
    digest = hashlib.sha256(raw).hexdigest()
    (root / digest).write_bytes(raw)
    return {"ref": {"sha256": digest, "bytes": len(raw)}, "kind": kind}


def fixture(root):
    members = [{"worker_id": f"w{i}", "participant_id": f"n{i}", "island_id": "single", "dataset_partition_id": f"p{i}"} for i in range(2)]
    base = artifact(root, 1, "model_f32")
    momentum = artifact(root, 0, "outer_state_f32")
    outer = {"kind": "nesterov", "aggregation": "uniform", "lr": .7, "momentum": .9, "inner_steps": 1, "total_steps": 2}
    plan = {"cohort": {"run_id": "run", "ordered_learners": members}, "slots": [{"member": m} for m in members], "base_model": base, "contract": {"config_hash": "config", "dataset_hash": "data"}, "outer": outer}
    binding = {"run_id": "run"}
    commit = {"binding": binding, "registrations": [{"slot": s, "contract": plan["contract"], "base_model_hash": base["ref"]["sha256"]} for s in plan["slots"]]}
    records = []
    # Hand-calculated two-round Nesterov example: g=.2 then .3,
    # m=.2 then .48; w=1-.7*(.2+.9*.2), then w-.7*(.3+.9*.48).
    for r, (values, model, moment) in enumerate([([.1, .3], .734, .2), ([.2, .4], .2216, .48)]):
        assignments, updates, accepted = [], [], []
        for i, member in enumerate(members):
            a = {"round": r, "base_round": r, "worker_id": member["worker_id"], "dataset_partition_id": member["dataset_partition_id"], "binding": binding, "contract": plan["contract"], "base_model": base, "outer_state": momentum, "inner_steps": 1}
            u = {"update_id": f"u{r}-{i}", "worker_id": member["worker_id"], "participant_id": member["participant_id"], "island_id": "single", "binding": binding, "contract": plan["contract"], "round": r, "base_round": r, "base_model_hash": base["ref"]["sha256"], "committed_steps": 1, "attempted_steps": 1, "skipped_steps": 0, "private_hash": "private", "precision_hash": "precision", "dtype": "float32", "delta": artifact(root, values[i], "delta_f32")}
            receipt = {"update_id": u["update_id"], "update_hash": f"hash-{r}-{i}", "private_hash": "private", "binding": binding}
            accepted.append(dict(receipt, worker_id=member["worker_id"], precision_hash="precision"))
            assignments.append(a)
            updates.append(u)
            records.append({"record": {"change": {"Next": {"plan": plan, "commit": commit, "phase": "ROUND_OPEN", "assignments": copy.deepcopy(assignments), "updates": copy.deepcopy(updates)}, "Receipt": receipt}}})
        base = artifact(root, model, "model_f32")
        momentum = artifact(root, moment, "outer_state_f32")
        published = {"round": r, "outer": outer, "binding": binding, "contract": plan["contract"], "model": base, "outer_state": momentum, "accepted_updates": accepted}
        records.append({"record": {"change": {"Next": {"plan": plan, "commit": commit, "round": r, "phase": "COMPLETE" if r == 1 else "EVALUATING", "base": base, "momentum": momentum}, "Published": published}}})
    return records


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = fixture(self.root)

    def test_hand_calculated_replay(self):
        result = replay(self.records, self.root, "run")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual([p["ordinal_rule"] for p in result["partitions"]], ["2*k+0", "2*k+1"])

    def test_zeroed_published_momentum_rejected(self):
        c = self.records[-1]["record"]["change"]
        c["Next"]["momentum"] = c["Published"]["outer_state"] = artifact(self.root, 0, "outer_state_f32")
        with self.assertRaisesRegex(ValueError, "outer replay mismatch"):
            replay(self.records, self.root, "run")

    def test_stale_delta_values_rejected_even_with_current_metadata(self):
        stale = self.records[0]["record"]["change"]["Next"]["updates"][0]["delta"]
        for e in self.records:
            for u in e["record"]["change"]["Next"].get("updates", []):
                if u["update_id"] == "u1-0":
                    u["delta"] = stale
        with self.assertRaisesRegex(ValueError, "outer replay mismatch"):
            replay(self.records, self.root, "run")

    def test_stale_round_and_broken_lineage(self):
        for field, value, message in [("round", 0, "stale delta"), ("base_model_hash", "stale", "stale delta")]:
            records = copy.deepcopy(self.records)
            for e in records:
                for u in e["record"]["change"]["Next"].get("updates", []):
                    if u["update_id"] == "u1-0":
                        u[field] = value
            with self.assertRaisesRegex(ValueError, message):
                replay(records, self.root, "run")
        for e in self.records:
            for a in e["record"]["change"]["Next"].get("assignments", []):
                if a["round"] == 1:
                    a["base_model"] = self.records[0]["record"]["change"]["Next"]["plan"]["base_model"]
        with self.assertRaisesRegex(ValueError, "previous published model"):
            replay(self.records, self.root, "run")

    def test_partition_and_worker_fences(self):
        for mutation in ("partition", "duplicate", "missing", "receipt"):
            records = copy.deepcopy(self.records)
            if mutation == "partition":
                for e in records:
                    for a in e["record"]["change"]["Next"].get("assignments", []):
                        a["dataset_partition_id"] = "p0"
            else:
                accepted = records[2]["record"]["change"]["Published"]["accepted_updates"]
                if mutation == "duplicate":
                    accepted[1] = accepted[0]
                elif mutation == "missing":
                    accepted.pop()
                else:
                    accepted[0]["update_hash"] = "wrong"
            with self.assertRaises(ValueError, msg=mutation):
                replay(records, self.root, "run")

    def test_hash_and_shape_validation(self):
        blob = artifact(self.root, 1, "model_f32")
        path = self.root / blob["ref"]["sha256"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_tensors(self.root, blob)
        with self.assertRaisesRegex(ValueError, "shape"):
            outer_step({"w": np.ones(2)}, {"w": np.ones(1)}, [{"w": np.ones(2)}], .7, .9)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare({"w": np.array([np.nan])}, {"w": np.array([1.0])})


if __name__ == "__main__":
    unittest.main()
