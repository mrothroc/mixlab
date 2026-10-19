"""Read-only, independent float32 Nesterov replay of committed P8 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

from measure import history

ATOL = 1e-7
RTOL = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_tensors(root, blob):
    ref = blob["ref"]
    require(re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "invalid artifact digest")
    raw = (Path(root) / ref["sha256"]).read_bytes()
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "artifact size/hash mismatch")
    require(len(raw) >= 8, "truncated tensor header")
    size, = struct.unpack("<Q", raw[:8])
    require(0 < size <= len(raw) - 8, "invalid tensor header size")
    header = json.loads(raw[8:8 + size])
    result, intervals = {}, []
    for name, entry in header.items():
        require(name != "__metadata__" and entry["dtype"] == "F32", "expected named F32 tensors")
        shape = entry["shape"]
        require(shape and all(type(d) is int and d > 0 for d in shape), "invalid tensor shape")
        begin, end = entry["data_offsets"]
        require(0 <= begin < end <= len(raw) - 8 - size and end - begin == 4 * int(np.prod(shape)), "invalid tensor offsets")
        values = np.frombuffer(raw, dtype="<f4", count=(end - begin) // 4, offset=8 + size + begin).reshape(shape)
        require(np.isfinite(values).all(), "nonfinite artifact tensor")
        result[name] = values
        intervals.append((begin, end))
    intervals.sort()
    require(intervals and intervals[0][0] == 0 and intervals[-1][1] == len(raw) - 8 - size and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])), "tensor payload gap/overlap")
    return result


def outer_step(base, momentum, deltas, lr, mu):
    """Delta is base-minus-local. Round g, then m, then w to float32.

    Accumulation/update intermediates are float64, matching the declared outer
    numerical contract, not an imported production optimizer implementation.
    """
    require(deltas, "missing deltas")
    for tensors in [momentum, *deltas]:
        require(tensors.keys() == base.keys(), "tensor names differ")
        for name in base:
            require(tensors[name].shape == base[name].shape, "tensor shape differs")
    weights, moments = {}, {}
    for name, w in base.items():
        total = np.zeros(w.shape, dtype=np.float64)
        for delta in deltas:
            total += delta[name].astype(np.float64) * (1.0 / len(deltas))
        g = total.astype(np.float32)
        m = (mu * momentum[name].astype(np.float64) + g.astype(np.float64)).astype(np.float32)
        weights[name] = (w.astype(np.float64) - lr * (g.astype(np.float64) + mu * m.astype(np.float64))).astype(np.float32)
        moments[name] = m
    return weights, moments


def compare(actual, expected):
    require(actual.keys() == expected.keys(), "published tensor names differ")
    absolute, relative, matches = 0.0, 0.0, True
    for name, want in expected.items():
        got = actual[name]
        require(got.shape == want.shape, "published shape differs")
        require(np.isfinite(got).all() and np.isfinite(want).all(), "nonfinite replay")
        error = np.abs(got.astype(np.float64) - want.astype(np.float64))
        absolute = max(absolute, float(error.max()))
        relative = max(relative, float((error / np.maximum(np.abs(want), np.finfo(np.float32).tiny)).max()))
        matches = matches and bool(np.all(error <= ATOL + RTOL * np.abs(want)))
    return {"max_absolute_error": absolute, "max_relative_error": relative, "matches": matches}


def partitions(plan, commit):
    members = plan["cohort"]["ordered_learners"]
    require(len(members) == 2, "P8 requires exactly two learners")
    require(len({m["worker_id"] for m in members}) == 2 and len({m["dataset_partition_id"] for m in members}) == 2, "workers/partitions not distinct")
    require([s["member"] for s in plan["slots"]] == members, "slot/cohort order differs")
    registrations = commit["registrations"]
    require([r["slot"] for r in registrations] == plan["slots"], "registration/slot order differs")
    for reg in registrations:
        require(reg["contract"] == plan["contract"] and reg["base_model_hash"] == plan["base_model"]["ref"]["sha256"], "registration contract/base differs")
    # PrepareLocalLearner derives Index from immutable cohort order and Count
    # from cohort length. NewLearnerLoader uses a common config seed, with
    # pre-permutation ordinals cursor + Index and cursor increments of Count.
    # Distinct residues stay disjoint through the common epoch bijection.
    return [{"worker_id": m["worker_id"], "partition_id": m["dataset_partition_id"], "index": i, "count": len(members), "ordinal_rule": f"{len(members)}*k+{i}"} for i, m in enumerate(members)]


def replay(records, root, run_id):
    require(records, "empty committed history")
    states = [e["record"]["change"]["Next"] for e in records]
    plan = states[0]["plan"]
    require(plan["cohort"]["run_id"] == run_id and states[-1]["phase"] == "COMPLETE", "wrong or incomplete run")
    commit = next(s["commit"] for s in states if s.get("commit"))
    partition_report = partitions(plan, commit)
    outer = plan["outer"]
    require(outer["kind"] == "nesterov" and outer["aggregation"] == "uniform" and outer["lr"] == .7 and outer["momentum"] == .9, "not the registered P8 optimizer")
    base_blob, momentum_blob = plan["base_model"], None
    assignments, updates, receipts, reports = {}, {}, {}, []
    for entry in records:
        change = entry["record"]["change"]
        state = change["Next"]
        require(state["plan"] == plan and state.get("commit") in (None, commit), "immutable run identity changed")
        for assignment in state.get("assignments") or []:
            key = (assignment["round"], assignment["worker_id"])
            if key in assignments:
                require(assignments[key] == assignment, "assignment changed within round")
            assignments[key] = assignment
        for update in state.get("updates") or []:
            uid = update["update_id"]
            if uid in updates:
                require(updates[uid] == update, "update identity reused")
            updates[uid] = update
        receipt = change.get("Receipt")
        if receipt:
            receipts[receipt["update_id"]] = receipt
        published = change.get("Published")
        if not published:
            continue
        round_id = len(reports)
        require(published["round"] == round_id and state["round"] == round_id, "published round gap/duplicate")
        require(published["outer"] == outer and published["binding"] == commit["binding"] and published["contract"] == plan["contract"], "published contract differs")
        require(state["base"] == published["model"] and state["momentum"] == published["outer_state"], "published state differs")
        accepted = published["accepted_updates"]
        require([u["worker_id"] for u in accepted] == [p["worker_id"] for p in partition_report], "accepted worker order/coverage differs")
        deltas = []
        for member, accepted_update in zip(plan["cohort"]["ordered_learners"], accepted):
            worker = member["worker_id"]
            a = assignments[(round_id, worker)]
            u = updates[accepted_update["update_id"]]
            require(a["base_model"] == base_blob, "base is not previous published model")
            if momentum_blob is None:
                momentum_blob = a["outer_state"]
            require(a["outer_state"] == momentum_blob, "momentum is not previous published momentum")
            require(a["dataset_partition_id"] == member["dataset_partition_id"] and a["binding"] == commit["binding"] and a["contract"] == plan["contract"], "assignment partition/contract differs")
            require(u["round"] == round_id and u["base_round"] == a["base_round"] and u["base_model_hash"] == base_blob["ref"]["sha256"], "stale delta round/base")
            require(u["worker_id"] == worker and u["participant_id"] == member["participant_id"] and u["island_id"] == member["island_id"] and u["binding"] == commit["binding"] and u["contract"] == plan["contract"], "delta ownership/contract differs")
            require(u["committed_steps"] == u["attempted_steps"] == a["inner_steps"] == outer["inner_steps"] and u["skipped_steps"] == 0, "inner-step contract differs")
            r = receipts[u["update_id"]]
            require(all(r[k] == accepted_update[k] for k in ("update_id", "update_hash", "private_hash")) and r["binding"] == commit["binding"], "accepted receipt differs")
            require(u["private_hash"] == accepted_update["private_hash"] and u["precision_hash"] == accepted_update["precision_hash"], "accepted update differs")
            require(u["dtype"] == "float32" and u["delta"]["kind"] == "delta_f32", "non-float32 delta")
            deltas.append(load_tensors(root, u["delta"]))
        base, momentum = load_tensors(root, base_blob), load_tensors(root, momentum_blob)
        if round_id == 0:
            require(all(np.count_nonzero(v) == 0 for v in momentum.values()), "initial momentum is not zero")
        expected_model, expected_momentum = outer_step(base, momentum, deltas, outer["lr"], outer["momentum"])
        report = {"round": round_id, "base": base_blob["ref"]["sha256"], "momentum_input": momentum_blob["ref"]["sha256"], "model": compare(load_tensors(root, published["model"]), expected_model), "momentum": compare(load_tensors(root, published["outer_state"]), expected_momentum)}
        require(report["model"]["matches"] and report["momentum"]["matches"], "outer replay mismatch: " + json.dumps(report))
        reports.append(report)
        base_blob, momentum_blob = published["model"], published["outer_state"]
    require(len(reports) * outer["inner_steps"] == outer["total_steps"], "missing published rounds")
    return {"run_id": run_id, "passed": True, "atol": ATOL, "rtol": RTOL, "rounds": reports, "partitions": partition_report, "partition_evidence": "committed cohort/registrations/assignments imply disjoint per-epoch strided ordinals under the production common-seed bijection; not a historical per-token consumption trace"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args()
    try:
        for run_id in args.run:
            print(json.dumps(replay(history(args.state, run_id), args.artifacts, run_id)), flush=True)
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        parser.exit(1, f"replay failed: {exc}\n")


if __name__ == "__main__":
    main()
