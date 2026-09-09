"""Opt-in committed-step watchdog and bounded native diagnostics for one trainer."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from handler_process import run_process


class TrainingStall(RuntimeError):
    is_training_stall = True

    def __init__(self, seconds, directory):
        super().__init__(f"training made no optimizer progress for {seconds:g}s")
        self.diagnostics = str(directory)
        self.output = ""
        self.stderr = ""


def validate_watchdog_input(job_input):
    seconds = job_input.get("stall_timeout")
    directory = job_input.get("stall_dump_dir")
    if seconds is None:
        if directory is not None:
            raise ValueError("stall_dump_dir requires stall_timeout")
        return
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("stall_timeout must be positive finite seconds")
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        raise ValueError("stall_timeout requires an absolute stall_dump_dir on persistent storage")
    if job_input.get("mode", "smoke") != "arch":
        raise ValueError("stall_timeout supports single-process arch training only")


class TrainingWatchdog:
    def __init__(self, seconds, progress_path, directory, *, clock=time.monotonic):
        self.seconds = seconds
        self.progress_path = Path(progress_path)
        self.directory = Path(directory)
        self.clock = clock
        self.pid = None
        self.committed = 0
        self.last_progress = 0
        self.record = None
        self.last_read_error = None

    def start(self, pid):
        self.pid = pid
        self.last_progress = self.clock()

    def check(self):
        try:
            with self.progress_path.open() as source:
                record = json.loads(source.read(4096))
            count = record.get("optimizer_steps")
            if record.get("pid") != self.pid or type(count) is not int or count < 0:
                raise ValueError("invalid progress PID or optimizer count")
            self.record = record
            self.last_read_error = None
            if count > self.committed:
                self.committed = count
                self.last_progress = self.clock()
        except (OSError, ValueError, AttributeError) as exc:
            self.last_read_error = str(exc)
        if self.clock() - self.last_progress >= self.seconds:
            try:
                self.capture()
            except Exception as exc:
                # Capture is best effort, but supervision must still kill/reap.
                error = TrainingStall(self.seconds, self.directory)
                error.args = (f"{error}; diagnostic capture failed: {exc}",)
                raise error from exc
            raise TrainingStall(self.seconds, self.directory)

    def capture(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = {"pid": self.pid, "optimizer_steps": self.committed,
                    "seconds_without_progress": self.clock() - self.last_progress,
                    "progress": self.record, "progress_read_error": self.last_read_error}
        (self.directory / "stall.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # /proc stays useful when the container disallows ptrace or GDB is absent.
        proc = Path("/proc") / str(self.pid)
        snapshots = {}
        for name in ("status", "wchan", "stat"):
            snapshots[name] = _read_proc(proc / name)
        try:
            tasks = sorted((proc / "task").iterdir())[:256]
        except OSError:
            tasks = []
        snapshots["threads"] = {task.name: {name: _read_proc(task / name)
                                for name in ("stat", "wchan", "stack")} for task in tasks}
        (self.directory / "proc.json").write_text(json.dumps(snapshots, indent=2) + "\n")
        try:
            self._command("native-stacks.txt", ["gdb", "-q", "-nx", "-batch",
                          "-iex", "set auto-load off", "-ex", "set pagination off",
                          "-ex", "set debuginfod enabled off", "-p", str(self.pid),
                          "-ex", "info threads", "-ex", "thread apply all bt 64",
                          "-ex", "detach"], 30)
        finally:
            # An interrupted debugger can leave the inferior stopped. The caller
            # kills the trainer group after capture regardless of debugger status.
            try:
                os.kill(self.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        self._command("nvidia-smi.txt", ["nvidia-smi"], 5)

    def _command(self, name, command, timeout):
        try:
            result = run_process(command, timeout, stream_output=False, capture_limit=4 * 1024 * 1024)
            text = f"exit_code={result.returncode}\n{result.stdout}\n{result.stderr}\n"
        except subprocess.TimeoutExpired as exc:
            text = f"diagnostic timed out after {timeout}s\n{exc.output or ''}\n{exc.stderr or ''}\n"
        except OSError as exc:
            text = f"diagnostic unavailable: {exc}\n"
        (self.directory / name).write_text(text)


def _read_proc(path):
    try:
        with path.open() as source:
            return source.read(8192)
    except OSError as exc:
        return f"unavailable: {exc}"
