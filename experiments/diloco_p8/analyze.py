"""Summarize pre-registered P8 arms, preserving misses and failed runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from measure import json_lines


def first_target(evaluations, target):
    return next((r for r in evaluations if math.isfinite(r["loss"]) and r["loss"] <= target), None)


def load_evidence(samples):
    reasons = []
    if len(samples) < 2:
        reasons.append("fewer than two host-load samples")
    overloaded = 0
    for s in samples:
        if "remote" not in s or "remote_error" in s or "local_network" not in s or "network" not in s.get("remote", {}):
            reasons.append("incomplete host/network telemetry")
        busy = s["local_load"][0] > 2 * s["local_cpus"] or ("remote" in s and s["remote"]["load"][0] > 2 * s["remote"]["cpus"])
        overloaded = overloaded + 1 if busy else 0
        if overloaded >= 2:
            reasons.append("two consecutive samples exceed twice host CPU count")
    return {"samples": len(samples), "max_local_load_1m": max((s["local_load"][0] for s in samples), default=None), "max_remote_load_1m": max((s["remote"]["load"][0] for s in samples if "remote" in s), default=None), "contamination_reasons": sorted(set(reasons))}


def summarize_arm(path, target):
    path = Path(path)
    result = json.loads((path / "result.json").read_text())
    out = {k: result[k] for k in ("arm", "status", "config_sha256", "elapsed_seconds")}
    out["seed"] = result["config"]["training"]["seed"]
    out["source"] = str(path)
    loads = path / "load.jsonl"
    out["load"] = load_evidence(json_lines(loads) if loads.exists() else [])
    if result["arm"] != "pair":
        rows = json_lines(path / "baseline.out")
        complete = next((r for r in rows if r.get("event") == "complete"), None)
        ready = next((r for r in rows if r.get("event") == "ready"), None)
        # Conservatively charge all process/SSH launch and teardown overhead at
        # every crossing, avoiding comparisons of clocks on different hosts.
        overhead = max(0, result["process"]["seconds"] - complete["elapsed_seconds"]) if complete else 0
        evaluations = [dict(r, seconds=r["elapsed_seconds"] + overhead) for r in rows if r.get("event") == "evaluation"]
        out["complete"] = result["status"] == "complete"
        out["initial_sha256"] = ready["initial_sha256"] if ready else None
        if ready:
            if ready["config_sha256"] != out["config_sha256"]:
                raise ValueError("baseline executed a different config")
            out["train_hash"], out["validation_hash"] = ready["train_hash"], ready["validation_hash"]
            out["setup_seconds"] = ready["elapsed_seconds"] + overhead
        out["first_step_seconds"] = next((r["seconds"] for r in rows if r.get("event") == "first_step"), None)
        out["training_seconds"] = complete["training_seconds"] if complete else None
        out["logical_artifact_bytes"] = 0
    else:
        history_path = path / "history.json"
        records = json.loads(history_path.read_text()) if history_path.exists() else []
        evaluations, updates, payload = [], {}, 0
        states = [r["record"]["change"]["Next"] for r in records]
        start = result["started_wall_ns"] / 1e9
        for record in records:
            change = record["record"]["change"]
            state = change["Next"]
            for update in state.get("updates") or []:
                updates[update["update_id"]] = update
            if change.get("Evaluation"):
                e = change["Evaluation"]
                evaluations.append({"step": (e["assignment"]["round"] + 1) * state["plan"]["outer"]["inner_steps"], "loss": e["loss"], "seconds": record["published_wall_ns"] / 1e9 - start, "evaluation_seconds": e["finished_at"] - e["started_at"], "validation_tokens": e["effective_tokens"], "precision": e["assignment"]["precision"]})
        for update in updates.values():
            payload += update["delta"]["ref"]["bytes"]
        for e in evaluations:
            e["effective_tokens"] = sum(u["effective_tokens"] for u in updates.values() if (u["round"] + 1) * states[0]["plan"]["outer"]["inner_steps"] <= e["step"])
        # Each learner downloads its base once; the designated evaluator reads
        # each candidate once. This is logical payload, not measured wire bytes.
        by_round = {}
        for state in states:
            if state["phase"] == "ROUND_OPEN":
                by_round[state["round"]] = state
        payload += sum(len(s["assignments"]) * s["base"]["ref"]["bytes"] for s in by_round.values())
        payload += sum(r["record"]["change"]["Evaluation"]["assignment"]["model"]["ref"]["bytes"] for r in records if r["record"]["change"].get("Evaluation"))
        out["logical_artifact_bytes"] = payload
        out["complete"] = bool(states and states[-1]["phase"] == "COMPLETE")
        if states:
            out["initial_sha256"] = states[0]["plan"]["base_model"]["ref"]["sha256"]
            out["train_hash"] = states[0]["plan"]["contract"]["dataset_hash"]
            out["validation_hash"] = states[0]["evaluation_policy"]["validation_hash"]
            out["registration_precision"] = [r["precision"] for r in (states[-1].get("commit") or {}).get("registrations", [])]
            registrations = (states[-1].get("commit") or {}).get("registrations", [])
            nodes = {r["slot"]["member"]["worker_id"]: r["slot"]["allocation"]["node_id"] for r in registrations}
            evaluators = {r["record"]["change"]["Evaluation"]["assignment"].get("worker_id") for r in records if r["record"]["change"].get("Evaluation")}
            out["evaluator_nodes"] = sorted(nodes[w] for w in evaluators if w in nodes)
        opens = [r for r in records if r["record"]["change"]["Next"]["phase"] == "ROUND_OPEN"]
        out["admission_to_first_round_seconds"] = opens[0]["published_wall_ns"] / 1e9 - start if opens else None
        out["inner_and_private_publication_worker_seconds"] = sum(u["finished_at"] - u["started_at"] for u in updates.values())
        out["effective_tokens"] = sum(u["effective_tokens"] for u in updates.values())
        straggler = sum(max(u["finished_at"] for u in updates.values() if u["round"] == r) - min(u["finished_at"] for u in updates.values() if u["round"] == r) for r in by_round if any(u["round"] == r for u in updates.values()))
        out["worker_finish_spread_seconds"] = straggler
        out["straggler_fraction_of_wall"] = straggler / result["elapsed_seconds"]
        download_setup, upload_reduce, eval_download = 0, 0, 0
        for number, state in by_round.items():
            work = [u for u in updates.values() if u["round"] == number]
            publication = next((r["record"]["change"]["Published"] for r in records if r["record"]["change"].get("Published") and r["record"]["change"]["Published"]["round"] == number), None)
            if work and publication:
                download_setup += max(0, max(u["started_at"] for u in work) - state["assignments"][0]["issued_at"])
                upload_reduce += max(0, publication["created_at"] - max(u["finished_at"] for u in work))
        for r in records:
            e = r["record"]["change"].get("Evaluation")
            if e:
                eval_download += max(0, e["started_at"] - e["assignment"]["issued_at"])
        out["transfer_inclusive_envelopes_seconds"] = {"download_and_setup_critical_path": download_setup, "last_upload_and_aggregation": upload_reduce, "evaluation_dispatch_and_download": eval_download}
        out["timing_resolution_seconds"] = 1
        out["exclusive_transfer_seconds"] = None
        out["timing_limitation"] = "inner window includes snapshot/delta persistence; finish spread is a clock-sensitive straggler proxy; exclusive transport and JIT are not instrumented"
    out["evaluations"] = evaluations
    out["validation_tokens"] = evaluations[-1]["validation_tokens"] if evaluations else None
    out["final_loss"] = evaluations[-1]["loss"] if evaluations else None
    if evaluations:
        out.setdefault("effective_tokens", evaluations[-1]["effective_tokens"] if "effective_tokens" in evaluations[-1] else None)
        out["precision"] = evaluations[-1]["precision"]
    crossing = first_target(evaluations, target)
    out["time_to_target_seconds"] = crossing["seconds"] if crossing and out["complete"] else None
    out["target_step"] = crossing["step"] if crossing else None
    out["target_effective_tokens"] = crossing.get("effective_tokens") if crossing else None
    out["target_goodput_tokens_per_second"] = out["target_effective_tokens"] / out["time_to_target_seconds"] if out["target_effective_tokens"] is not None and out["time_to_target_seconds"] else None
    out["effective_tokens_per_wall_second"] = out["effective_tokens"] / out["elapsed_seconds"] if out.get("effective_tokens") is not None else None
    return out


def compare(registration, arms):
    grouped = {}
    for arm in arms:
        seed, name = arm["seed"], arm["arm"]
        key = str(seed)
        if key not in registration["config_sha256"] or arm["config_sha256"] != registration["config_sha256"][key]:
            raise ValueError("run config differs from pre-registration")
        if (seed, name) in grouped:
            raise ValueError("duplicate arm; retain/review contaminated sessions explicitly")
        grouped[seed, name] = arm
    if len(arms) != len(registration["seeds"]) * 3:
        raise ValueError("incomplete experiment: all three arms for every seed required")
    rows = []
    for seed in registration["seeds"]:
        m1, m4, pair = [grouped[seed, x] for x in ("m1", "m4", "pair")]
        for field in ("initial_sha256", "train_hash", "validation_hash", "precision", "validation_tokens"):
            values = [a.get(field) for a in (m1, m4, pair)]
            if all(a["complete"] for a in (m1, m4, pair)) and (None in values or any(v != values[0] for v in values)):
                raise ValueError("mismatched " + field)
            if field in registration and any(a["complete"] and a.get(field) != registration[field] for a in (m1, m4, pair)):
                raise ValueError("run differs from registered " + field)
        reached = all(a["time_to_target_seconds"] is not None for a in (m1, m4, pair))
        best = min((m1, m4), key=lambda a: a["time_to_target_seconds"] if a["time_to_target_seconds"] is not None else math.inf)
        delta = pair["final_loss"] - best["final_loss"] if pair["final_loss"] is not None and best["final_loss"] is not None else None
        rows.append({"seed": seed, "all_reached": reached, "baseline": best["arm"], "baseline_seconds": best["time_to_target_seconds"], "pair_seconds": pair["time_to_target_seconds"], "speedup": best["time_to_target_seconds"] / pair["time_to_target_seconds"] if reached else None, "final_loss_delta": delta, "loss_within_tolerance": delta is not None and delta <= registration["final_loss_tolerance"]})
    contaminated = any(a.get("load", {}).get("contamination_reasons") for a in arms)
    success = all(r["all_reached"] for r in rows)
    median_pair = statistics.median(r["pair_seconds"] for r in rows) if success else None
    median_base = statistics.median(r["baseline_seconds"] for r in rows) if success else None
    numerical_gate = success and median_pair < median_base and all(r["loss_within_tolerance"] for r in rows)
    return {"seeds": rows, "contaminated": contaminated, "median_pair_seconds": median_pair, "median_stronger_baseline_seconds": median_base, "raw_numerical_gate": numerical_gate, "performance_pass": not contaminated and numerical_gate, "arms": arms}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    reg = json.loads(a.registration.read_text())
    result = compare(reg, [summarize_arm(path, reg["target_loss"]) for path in a.runs])
    result["registration_sha256"] = hashlib.sha256(a.registration.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))
