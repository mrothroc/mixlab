import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze import compare, first_target, load_evidence, summarize_arm
from measure import history, idle, network_counters
from prepare import partition


class ExperimentTests(unittest.TestCase):
    def test_network_and_idle_preflight(self):
        sample = "Name Mtu Network Address Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll\nen7 1500 <Link#17> aa:bb 10 0 123 5 0 456 0\nen7 1500 10/24 10.0.0.1 10 - 123 5 - 456 -\n"
        self.assertEqual(network_counters(sample), {"Ibytes": 123, "Obytes": 456})
        with self.assertRaises(ValueError):
            network_counters("")
        load = {"local_load": [3, 3, 3], "local_cpus": 10, "remote": {"load": [3, 3, 3], "cpus": 16}}
        self.assertTrue(idle(load))
        load["local_load"][0] = 63
        self.assertFalse(idle(load))
        load["local_load"][0] = 3
        load["remote_error"] = "unreachable"
        self.assertFalse(idle(load))

    def test_contaminated_session_cannot_pass(self):
        sample = {"local_load": [21, 3, 3], "local_cpus": 10, "local_network": {}, "remote": {"load": [3, 3, 3], "cpus": 16, "network": {}}}
        evidence = load_evidence([sample, sample])
        self.assertEqual(len(evidence["contamination_reasons"]), 1)
        reg, arms = self.fixture()
        for a in arms:
            if a["arm"] == "pair":
                a["time_to_target_seconds"] = 5
        arms[0]["load"] = evidence
        result = compare(reg, arms)
        self.assertTrue(result["contaminated"])
        self.assertFalse(result["performance_pass"])
        self.assertEqual(len(result["arms"]), 9)

    def test_partition_before_tokenizer(self):
        first = partition("a complete document")
        self.assertEqual(first, partition("a complete document"))
        self.assertEqual(first[0], hashlib.sha256(b"a complete document").hexdigest())
        self.assertEqual(first[1], "validation" if int(first[0][:8], 16) % 20 == 0 else "train")

    def fixture(self):
        reg = {"seeds": [17, 29, 43], "config_sha256": {str(s): str(s) for s in [17, 29, 43]}, "final_loss_tolerance": .1}
        arms = []
        for s in reg["seeds"]:
            for name, t in [("m1", 20), ("m4", 10), ("pair", 12)]:
                arms.append(dict(seed=s, arm=name, config_sha256=str(s), complete=True, initial_sha256="weights", train_hash="train", validation_hash="val", precision={"compute_dtype": "float32", "backend": "metal", "tf32_allowed": False}, validation_tokens=16384, time_to_target_seconds=t, final_loss=4.0))
        return reg, arms

    def test_stronger_baseline_not_m1(self):
        reg, arms = self.fixture()
        result = compare(reg, arms)
        self.assertFalse(result["performance_pass"])
        self.assertEqual(result["median_stronger_baseline_seconds"], 10)
        self.assertTrue(all(r["baseline"] == "m4" for r in result["seeds"]))

    def test_loss_tolerance_and_missing_target(self):
        reg, arms = self.fixture()
        for a in arms:
            if a["arm"] == "pair":
                a["time_to_target_seconds"] = 5
        self.assertTrue(compare(reg, arms)["performance_pass"])
        arms[2]["final_loss"] = 4.11
        self.assertFalse(compare(reg, arms)["performance_pass"])
        arms[2]["time_to_target_seconds"] = None
        self.assertFalse(compare(reg, arms)["performance_pass"])
        self.assertEqual(len(compare(reg, arms)["arms"]), 9)

    def test_registration_and_completeness(self):
        reg, arms = self.fixture()
        for altered in [arms[:-1], arms + [arms[0]]]:
            with self.assertRaises(ValueError):
                compare(reg, altered)
        for field in ["config_sha256", "initial_sha256", "train_hash", "validation_hash", "precision", "validation_tokens"]:
            altered = copy.deepcopy(arms)
            altered[0][field] = "mismatch"
            with self.assertRaises(ValueError):
                compare(reg, altered)

    def test_pair_committed_update_schema_and_deduplication(self):
        # Production updates use update_id, not id; later states repeat updates.
        ref = {"ref": {"bytes": 100, "sha256": "weights"}}
        precision = {"compute_dtype": "float32", "backend": "metal", "tf32_allowed": False}
        state = {"phase": "ROUND_OPEN", "round": 0, "plan": {"outer": {"inner_steps": 100}, "base_model": ref, "contract": {"dataset_hash": "train"}}, "base": ref, "evaluation_policy": {"validation_hash": "val"}, "assignments": [{"issued_at": 102}, {"issued_at": 102}], "updates": [{"update_id": "u1", "round": 0, "delta": ref, "started_at": 103, "finished_at": 110, "effective_tokens": 200}, {"update_id": "u2", "round": 0, "delta": ref, "started_at": 104, "finished_at": 112, "effective_tokens": 200}]}
        done = dict(state, phase="COMPLETE")
        evaluation = {"assignment": {"round": 0, "model": ref, "issued_at": 114, "precision": precision}, "loss": 4.0, "started_at": 115, "finished_at": 116, "effective_tokens": 64}
        records = [{"published_wall_ns": 102_000_000_000, "record": {"change": {"Next": state}}}, {"published_wall_ns": 117_000_000_000, "record": {"change": {"Next": done, "Evaluation": evaluation, "Published": {"round": 0, "created_at": 114}}}}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {"arm": "pair", "status": {"run_phase": "COMPLETE"}, "config_sha256": "config", "elapsed_seconds": 18, "started_wall_ns": 100_000_000_000, "config": {"training": {"seed": 42}}}
            (root / "result.json").write_text(json.dumps(result))
            (root / "history.json").write_text(json.dumps(records))
            actual = summarize_arm(root, 4.5)
        self.assertTrue(actual["complete"])
        self.assertEqual(actual["logical_artifact_bytes"], 500)
        self.assertEqual(actual["effective_tokens"], 400)
        self.assertEqual(actual["inner_and_private_publication_worker_seconds"], 15)
        self.assertEqual(actual["worker_finish_spread_seconds"], 2)
        self.assertEqual(actual["time_to_target_seconds"], 17)
        self.assertEqual(actual["target_step"], 100)
        self.assertEqual(actual["precision"], precision)
        self.assertEqual(actual["validation_tokens"], 64)
        self.assertEqual(actual["transfer_inclusive_envelopes_seconds"], {"download_and_setup_critical_path": 2, "last_upload_and_aggregation": 2, "evaluation_dispatch_and_download": 1})

    def test_baseline_execution_identity_and_launch_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {"arm": "m4", "status": "complete", "config_sha256": "config", "elapsed_seconds": 12, "process": {"seconds": 12}, "config": {"training": {"seed": 17}}}
            rows = [{"event": "ready", "config_sha256": "config", "initial_sha256": "weights", "train_hash": "train", "validation_hash": "val", "elapsed_seconds": 1}, {"event": "first_step", "seconds": 2}, {"event": "evaluation", "elapsed_seconds": 6, "step": 100, "loss": 4, "validation_tokens": 64, "effective_tokens": 200, "precision": {"compute_dtype": "float32"}}, {"event": "complete", "elapsed_seconds": 10, "training_seconds": 8}]
            (root / "result.json").write_text(json.dumps(result))
            log = root / "baseline.out"
            log.write_text("optimizer banner\n" + "\n".join(json.dumps(r) for r in rows))
            actual = summarize_arm(root, 4.5)
            self.assertEqual(actual["time_to_target_seconds"], 8)
            self.assertEqual(actual["setup_seconds"], 3)
            self.assertEqual(actual["target_effective_tokens"], 200)
            self.assertEqual(actual["target_goodput_tokens_per_second"], 25)
            rows[0]["config_sha256"] = "remote stale config"
            log.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "different config"):
                summarize_arm(root, 4.5)

    def test_first_observed_crossing_no_interpolation(self):
        rows = [{"loss": 6, "seconds": 10}, {"loss": 4, "seconds": 20}, {"loss": 3, "seconds": 30}]
        self.assertEqual(first_target(rows, 5), rows[1])
        self.assertIsNone(first_target(rows, 2))

    def test_history_follows_committed_hash_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = hashlib.sha256(b"run").hexdigest()
            record = {"previous": {"revision": 0, "hash": ""}, "change": {"Next": {"revision": 1}}}
            blob = json.dumps(record).encode()
            digest = hashlib.sha256(blob).hexdigest()
            path = root / (key + "-" + digest + ".record")
            path.write_bytes(blob)
            (root / (key + ".head")).write_text(json.dumps({"revision": 1, "hash": digest}))
            (root / (key + "-orphan.record")).write_text("invalid orphan ignored")
            self.assertEqual(history(root, "run")[0]["record"], record)
            path.write_bytes(blob + b" ")
            with self.assertRaises(ValueError):
                history(root, "run")


if __name__ == "__main__":
    unittest.main()
