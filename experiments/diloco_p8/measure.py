"""Run one bounded P8 arm; retain failures and raw evidence without overwriting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


def history(root, run_id):
    root = Path(root)
    key = hashlib.sha256(run_id.encode()).hexdigest()
    head = json.loads((root / (key + ".head")).read_text())
    records = []
    while head["revision"]:
        path = root / (key + "-" + head["hash"] + ".record")
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != head["hash"]:
            raise ValueError("coordinator record hash mismatch")
        record = json.loads(blob)
        if record["change"]["Next"]["revision"] != head["revision"]:
            raise ValueError("coordinator revision mismatch")
        records.append({"record": record, "published_wall_ns": path.stat().st_mtime_ns})
        previous = record["previous"]
        if previous["revision"] != head["revision"] - 1:
            raise ValueError("coordinator revision gap")
        head = previous
    return list(reversed(records))


def json_lines(path):
    output = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("{"):
            output.append(json.loads(line))
    return output


def network_counters(text):
    lines = text.splitlines()
    if not lines:
        raise ValueError("missing interface counters")
    header = lines[0].split()
    for line in lines[1:]:
        row = line.split()
        if any(value.startswith("<Link#") for value in row):
            return {name: int(row[header.index(name)]) for name in ("Ibytes", "Obytes")}
    raise ValueError("missing physical interface row")


def sample_load(remote):
    # Interface counters include ALL traffic, not just Mixlab payloads. They
    # identify background traffic without mislabeling it as artifact transfer.
    code = "import os,json,time,subprocess; print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),wall=time.time(),network_raw=subprocess.check_output(['netstat','-ibn','-I','en7'],text=True))))"
    sample = {"wall": time.time(), "local_load": os.getloadavg(), "local_cpus": os.cpu_count()}
    try:
        sample["local_network"] = network_counters(subprocess.check_output(["netstat", "-ibn", "-I", "en7"], text=True, timeout=3))
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        sample["local_network_error"] = str(exc)
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, shlex.join(["python3", "-c", code])], capture_output=True, text=True, timeout=8, check=True)
        sample["remote"] = json.loads(result.stdout)
        sample["remote"]["network"] = network_counters(sample["remote"].pop("network_raw"))
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        sample["remote_error"] = str(exc)
    return sample


def idle(sample):
    return "remote" in sample and sample["local_load"][0] <= sample["local_cpus"] and sample["remote"]["load"][0] <= sample["remote"]["cpus"] and "remote_error" not in sample and "local_network_error" not in sample


def preflight(remote):
    samples = []
    for i in range(3):
        sample = sample_load(remote)
        samples.append(sample)
        if not idle(sample):
            raise ValueError("host preflight not idle: " + json.dumps(samples))
        if i < 2:
            time.sleep(10)
    return samples


def monitor(remote, path, stop):
    with path.open("x") as out:
        while not stop.is_set():
            sample = sample_load(remote)
            out.write(json.dumps(sample) + "\n")
            out.flush()
            stop.wait(5)


def command(argv, output, timeout):
    begin = time.monotonic()
    with output.with_suffix(".out").open("x") as out, output.with_suffix(".err").open("x") as err:
        try:
            result = subprocess.run(argv, stdout=out, stderr=err, timeout=timeout, check=False)
            return {"command": argv, "exit": result.returncode, "seconds": time.monotonic() - begin}
        except subprocess.TimeoutExpired:
            return {"command": argv, "exit": None, "timeout": True, "seconds": time.monotonic() - begin}


def run_arm(m, arm, config, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"arm": arm, "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(), "started_wall_ns": time.time_ns(), "config": json.loads(Path(config).read_text())}
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(m["remote"], output / "load.jsonl", stop))
    thread.start()
    begin = time.monotonic()
    try:
        if arm != "pair":
            args = [m["baseline"], "-config", config, "-train", m["train_pattern"], "-validation", m["validation_pattern"], "-validation-batches", str(m["validation_batches"]), "-timeout", str(timeout) + "s"]
            if arm == "m4":
                args = ["ssh", "-o", "BatchMode=yes", m["remote"], shlex.join(args)]
            metadata["process"] = command(args, output / "baseline", timeout + 30)
            metadata["evaluations"] = [r for r in json_lines(output / "baseline.out") if r.get("event") == "evaluation"]
            metadata["status"] = "complete" if metadata["process"]["exit"] == 0 and any(r.get("event") == "complete" for r in json_lines(output / "baseline.out")) else "failed"
        else:
            common = [m["cluster"], "submit", "diloco", "-principal-state-dir", m["controller"], "-coordinator-endpoint", m["coordinator_endpoint"], "-coordinator-id", m["coordinator_id"]]
            args = common + ["-config", config, "-worker-binary", m["worker"], "-train", "p8-train", "-dataset-id", m["train_hash"], "-validation", "p8-validation", "-validation-id", m["validation_hash"], "-validation-batches", str(m["validation_batches"]), "-workers", "2", "-max-runtime-seconds", str(timeout), "-max-memory-bytes", str(4 << 30), "-discover", "off"]
            for node in m["nodes"]:
                args.extend(["-node", node])
            metadata["process"] = command(args, output / "submission", min(timeout, 300))
            responses = json_lines(output / "submission.out")
            if not responses:
                raise ValueError("submission returned no JSON; inspect stderr for a possibly accepted run ID before retrying")
            metadata["run_id"] = responses[-1]["run_id"]
            with (output / "status.jsonl").open("x") as log:
                while time.monotonic() - begin < timeout + 300:
                    r = subprocess.run(common + ["-status", "-run-id", metadata["run_id"]], text=True, capture_output=True, timeout=90, check=False)
                    observation = {"elapsed_seconds": time.monotonic() - begin, "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
                    log.write(json.dumps(observation) + "\n")
                    log.flush()
                    try:
                        status = json.loads(r.stdout)
                    except ValueError:
                        time.sleep(2)
                        continue
                    if status.get("run_phase") in ("COMPLETE", "FAILED") or status.get("admission") == "ABORTED":
                        metadata["status"] = status
                        break
                    time.sleep(1)
                else:
                    metadata["status"] = "observer_timeout; inspect managed run before cleanup"
            records = history(m["coordinator_state"], metadata["run_id"])
            (output / "history.json").write_text(json.dumps(records, indent=2) + "\n")
    except Exception as exc:
        metadata["error"] = str(exc)
        metadata.setdefault("status", "failed")
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - begin
        metadata["ended_wall_ns"] = time.time_ns()
        stop.set()
        thread.join()
        (output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    status = metadata["status"]
    return status == "complete" or isinstance(status, dict) and status.get("run_phase") == "COMPLETE"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arm", choices=["m1", "m4", "pair"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--require-idle", action="store_true", help="require three quiet-host samples before a final arm")
    a = p.parse_args()
    if not 1 <= a.timeout <= 14400:
        p.error("timeout must be 1..14400 seconds")
    manifest = json.loads(a.manifest.read_text())
    if a.require_idle:
        # A blocked preflight submits no job and is not a failed training arm.
        print(json.dumps({"preflight": preflight(manifest["remote"])}), flush=True)
    raise SystemExit(0 if run_arm(manifest, a.arm, a.config, a.output, a.timeout) else 1)
