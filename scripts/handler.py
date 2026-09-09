"""RunPod serverless handler for mixlab."""

import json
import math
import os
import shlex
import subprocess
import tempfile
from string import Template

from handler_process import run_process

# Flags whose value is meaningful when zero, so they cannot be guarded on truthiness.
_ZERO_VALUED_FLAGS = {"temperature": "-temperature"}

_FLAGS = {
    "train": "-train",
    "safetensors": "-safetensors",
    "safetensors_load": "-safetensors-load",
    "resume": "-resume",
    "quantize": "-quantize",
    "output": "-output",
    "checkpoint_dir": "-checkpoint-dir",
    "checkpoint_every": "-checkpoint-every",
    "max_tokens": "-max-tokens",
}


def format_command(command):
    """Return a loggable command string for either shell strings or argv lists."""
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        return shlex.join(str(part) for part in command)
    return str(command)


def expand_command_argv(command, env=None):
    """Expand environment variables in argv-style commands without invoking a shell."""
    mapping = os.environ if env is None else env
    return [Template(str(part)).safe_substitute(mapping) for part in command]


def build_mixlab_command(job_input, config_path):
    """Build the mixlab argv for a job."""
    if job_input.get("resume") and job_input.get("safetensors_load"):
        raise ValueError("resume and safetensors_load are mutually exclusive: "
                         "choose checkpoint resume or weights-only loading")
    cmd = ["mixlab", "-mode", job_input.get("mode", "smoke")]
    if config_path:
        cmd.extend(["-config", config_path])
    for key, flag in _FLAGS.items():
        if job_input.get(key):
            cmd.extend([flag, str(job_input[key])])
    for key, flag in _ZERO_VALUED_FLAGS.items():
        if job_input.get(key) is not None:
            cmd.extend([flag, str(job_input[key])])
    return cmd


def build_job_env(job_input, config_path):
    """Build the environment for a job's child processes.

    Returns a fresh mapping rather than mutating os.environ: warm workers reuse the
    handler process, so a job's env must not leak into the next job on that worker.
    """
    env = dict(os.environ)
    if config_path:
        env["MIXLAB_CONFIG"] = config_path
    for key, value in (job_input.get("env") or {}).items():
        env[str(key)] = str(value)
    return env


def run_shell_commands(commands, label, timeout, env=None):
    """Run setup/post commands. Return (stdout, error_details)."""
    all_stdout = []
    all_stderr = []
    for i, command in enumerate(commands):
        print(f"[{label}[{i}]] {format_command(command)}", flush=True)
        if isinstance(command, str):
            popen_args = command
            use_shell = True
        elif isinstance(command, (list, tuple)):
            popen_args = expand_command_argv(command, env)
            use_shell = False
        else:
            return None, {
                "error": f"{label}[{i}] invalid command",
                "cmd": command,
                "stdout": "\n".join(all_stdout),
                "stderr": "\n".join(all_stderr),
                "exit_code": None,
            }

        failure = None
        try:
            result = run_process(popen_args, timeout, shell=use_shell, env=env,
                                 stderr_prefix=f"[{label}[{i}] stderr] ")
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(popen_args, None, exc.output or "", exc.stderr or "")
            failure = f"{label}[{i}] timeout"
        except OSError as exc:
            result = subprocess.CompletedProcess(popen_args, None, "", str(exc))
            failure = f"{label}[{i}] failed: {exc}"
        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)
        if failure or result.returncode != 0:
            return None, {
                "error": failure or f"{label}[{i}] failed", "cmd": command,
                "stdout": "\n".join(all_stdout), "stderr": "\n".join(all_stderr),
                "exit_code": result.returncode,
            }
    return "\n".join(all_stdout), None


def handler(job):
    job_input = job["input"]
    timeout = job_input.get("timeout", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        return {"error": "timeout must be a positive finite number of seconds"}

    config_json = job_input.get("config_json")
    config_path = job_input.get("config")
    tmp_config = None

    try:
        if config_json and not config_path:
            tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            with tmp_config:
                json.dump(config_json, tmp_config)
            config_path = tmp_config.name
        return run_job(job_input, config_path, timeout)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if tmp_config:
            os.unlink(tmp_config.name)


def run_job(job_input, config_path, timeout):
    output = {}
    env = build_job_env(job_input, config_path)
    # Validate before setup commands can have side effects.
    cmd = build_mixlab_command(job_input, config_path)

    # --- Setup commands (before mixlab) ---
    setup_cmds = job_input.get("setup", [])
    if setup_cmds:
        setup_out, err = run_shell_commands(setup_cmds, "setup", timeout, env=env)
        if err:
            return err
        output["setup_stdout"] = setup_out

    # --- Main mixlab command ---
    try:
        result = run_process(cmd, timeout, env=env)
        output.update(stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)
    except subprocess.TimeoutExpired as exc:
        output.update(error="timeout", stdout=exc.output or "", stderr=exc.stderr or "", exit_code=None)
        return output

    # --- Post-processing commands (after mixlab) ---
    # Run before config cleanup so post commands can reference the config file.
    post_cmds = job_input.get("post", [])
    if post_cmds:
        post_out, err = run_shell_commands(post_cmds, "post", timeout, env=env)
        if err:
            output["post_error"] = err
        else:
            output["post_stdout"] = post_out

    return output


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
