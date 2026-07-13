"""RunPod serverless handler for mixlab."""

import json
import os
import shlex
import subprocess
import tempfile
from string import Template

# Flags whose value is meaningful when zero, so they cannot be guarded on truthiness.
_ZERO_VALUED_FLAGS = {"temperature": "-temperature"}

_FLAGS = {
    "train": "-train",
    "safetensors": "-safetensors",
    "safetensors_load": "-safetensors-load",
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
    """Run a list of shell commands, streaming output. Returns (stdout, stderr) or raises."""
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

        proc = subprocess.Popen(popen_args, shell=use_shell, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
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
        proc.stdout.close()
        proc.stderr.close()
        if proc.returncode != 0:
            return None, {
                "error": f"{label}[{i}] failed", "cmd": command,
                "stdout": "\n".join(all_stdout), "stderr": "\n".join(all_stderr),
                "exit_code": proc.returncode,
            }
    return "\n".join(all_stdout), None


def handler(job):
    job_input = job["input"]
    timeout = job_input.get("timeout", 3600)
    output = {}

    config_json = job_input.get("config_json")
    config_path = job_input.get("config")
    tmp_config = None

    if config_json and not config_path:
        tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_json, tmp_config)
        tmp_config.close()
        config_path = tmp_config.name

    env = build_job_env(job_input, config_path)

    # --- Setup commands (before mixlab) ---
    setup_cmds = job_input.get("setup", [])
    if setup_cmds:
        setup_out, err = run_shell_commands(setup_cmds, "setup", timeout, env=env)
        if err:
            return err
        output["setup_stdout"] = setup_out

    # --- Main mixlab command ---
    cmd = build_mixlab_command(job_input, config_path)

    try:
        stdout_lines = []
        stderr_lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            stdout_lines.append(line)
        proc.wait(timeout=timeout)
        stderr_lines = proc.stderr.read().splitlines()
        proc.stdout.close()
        proc.stderr.close()
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

    # --- Post-processing commands (after mixlab) ---
    # Run before config cleanup so post commands can reference the config file.
    post_cmds = job_input.get("post", [])
    if post_cmds:
        post_out, err = run_shell_commands(post_cmds, "post", timeout, env=env)
        if err:
            output["post_error"] = err
        else:
            output["post_stdout"] = post_out

    if tmp_config:
        os.unlink(tmp_config.name)

    return output


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
