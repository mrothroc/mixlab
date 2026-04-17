"""RunPod serverless handler for mixlab."""

import json
import os
import subprocess
import tempfile

import runpod


def run_shell_commands(commands, label, timeout):
    """Run a list of shell commands, streaming output. Returns (stdout, stderr) or raises."""
    all_stdout = []
    all_stderr = []
    for i, cmd in enumerate(commands):
        print(f"[{label}[{i}]] {cmd}", flush=True)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            all_stdout.append(line)
        proc.wait(timeout=timeout)
        stderr = proc.stderr.read()
        if stderr:
            for line in stderr.splitlines():
                print(f"[{label}[{i}] stderr] {line}", flush=True)
            all_stderr.append(stderr)
        if proc.returncode != 0:
            return None, {
                "error": f"{label}[{i}] failed", "cmd": cmd,
                "stdout": "\n".join(all_stdout), "stderr": "\n".join(all_stderr),
                "exit_code": proc.returncode,
            }
    return "\n".join(all_stdout), None


def handler(job):
    job_input = job["input"]
    timeout = job_input.get("timeout", 3600)
    output = {}

    # --- Setup commands (before mixlab) ---
    setup_cmds = job_input.get("setup", [])
    if setup_cmds:
        setup_out, err = run_shell_commands(setup_cmds, "setup", timeout)
        if err:
            return err
        output["setup_stdout"] = setup_out

    # --- Main mixlab command ---
    mode = job_input.get("mode", "smoke")
    cmd = ["mixlab", "-mode", mode]

    config_json = job_input.get("config_json")
    config_path = job_input.get("config")
    tmp_config = None

    if config_json and not config_path:
        tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_json, tmp_config)
        tmp_config.close()
        config_path = tmp_config.name

    if config_path:
        cmd.extend(["-config", config_path])
    if job_input.get("train"):
        cmd.extend(["-train", job_input["train"]])
    if job_input.get("safetensors"):
        cmd.extend(["-safetensors", job_input["safetensors"]])
    if job_input.get("safetensors_load"):
        cmd.extend(["-safetensors-load", job_input["safetensors_load"]])
    if job_input.get("quantize"):
        cmd.extend(["-quantize", job_input["quantize"]])
    if job_input.get("output"):
        cmd.extend(["-output", job_input["output"]])
    if job_input.get("checkpoint_dir"):
        cmd.extend(["-checkpoint-dir", job_input["checkpoint_dir"]])
    if job_input.get("checkpoint_every"):
        cmd.extend(["-checkpoint-every", str(job_input["checkpoint_every"])])
    if job_input.get("max_tokens"):
        cmd.extend(["-max-tokens", str(job_input["max_tokens"])])
    if job_input.get("temperature"):
        cmd.extend(["-temperature", str(job_input["temperature"])])

    try:
        stdout_lines = []
        stderr_lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
        stderr_lines = proc.stderr.read().splitlines()
        if stderr_lines:
            for line in stderr_lines:
                print(f"[stderr] {line}", flush=True)
        output["stdout"] = "\n".join(stdout_lines)
        output["stderr"] = "\n".join(stderr_lines)
        output["exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_config:
            os.unlink(tmp_config.name)

    # --- Post-processing commands (after mixlab) ---
    post_cmds = job_input.get("post", [])
    if post_cmds:
        post_out, err = run_shell_commands(post_cmds, "post", timeout)
        if err:
            output["post_error"] = err
        else:
            output["post_stdout"] = post_out

    return output


runpod.serverless.start({"handler": handler})
