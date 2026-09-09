"""POSIX subprocess streaming for the RunPod handler (no RunPod dependency)."""

from collections import deque
import os
import selectors
import signal
import subprocess
import time


_CAPTURE_LIMIT = 8 * 1024 * 1024
_READ_SIZE = 64 * 1024
_TRUNCATED = "[earlier output truncated by handler]\n"


class _Output:
    def __init__(self, prefix, limit):
        self.prefix = prefix
        self.limit = limit
        self.chunks = deque()
        self.size = 0
        self.truncated = False
        self.pending = b""

    def feed(self, data):
        self.chunks.append(data)
        self.size += len(data)
        while self.size > self.limit:
            first = self.chunks.popleft()
            excess = self.size - self.limit
            removed = min(excess, len(first))
            self.size -= removed
            if removed < len(first):
                self.chunks.appendleft(first[removed:])
            self.truncated = True

        self.pending += data
        while self.pending:
            newline = self.pending.find(b"\n", 0, _READ_SIZE)
            if newline >= 0:
                line, self.pending = self.pending[:newline], self.pending[newline + 1:]
            elif len(self.pending) >= _READ_SIZE:
                line, self.pending = self.pending[:_READ_SIZE], self.pending[_READ_SIZE:]
            else:
                break
            print(self.prefix + line.decode("utf-8", errors="replace").rstrip("\r"), flush=True)

    def finish(self):
        if self.pending:
            print(self.prefix + self.pending.decode("utf-8", errors="replace"), flush=True)
            self.pending = b""

    def text(self):
        text = b"".join(self.chunks).decode("utf-8", errors="replace")
        return (_TRUNCATED if self.truncated else "") + "\n".join(text.splitlines())


def _kill_group(proc):
    # The shell may have exited while a descendant still owns an output pipe.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_process(command, timeout, *, env=None, shell=False,
                stderr_prefix="[stderr] ", capture_limit=_CAPTURE_LIMIT):
    """Drain both streams without readline blocking; enforce a wall-clock deadline.

    Return CompletedProcess with separate text tails. TimeoutExpired includes
    output captured so far. The child is a session leader so timeout/error
    cleanup also kills shell descendants that inherited its pipes.
    """
    if capture_limit <= 0:
        raise ValueError("capture_limit must be positive")
    deadline = time.monotonic() + timeout
    out = _Output("", capture_limit)
    err = _Output(stderr_prefix, capture_limit)
    proc = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            bufsize=0, env=env, start_new_session=True)
    try:
        with selectors.DefaultSelector() as selector:
            for stream, capture in ((proc.stdout, out), (proc.stderr, err)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, capture)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(min(remaining, 0.1)):
                    try:
                        data = os.read(key.fd, _READ_SIZE)
                    except BlockingIOError:
                        continue
                    if data:
                        key.data.feed(data)
                    else:
                        selector.unregister(key.fileobj)
                        key.data.finish()
            proc.wait(timeout=max(0, deadline - time.monotonic()))
        return subprocess.CompletedProcess(command, proc.returncode, out.text(), err.text())
    except BaseException as exc:
        _kill_group(proc)
        # Never enter an unbounded communicate/read after a failed process.
        proc.wait(timeout=5)
        if isinstance(exc, subprocess.TimeoutExpired):
            out.finish()
            err.finish()
            exc.output = out.text()
            exc.stderr = err.text()
        raise
    finally:
        proc.stdout.close()
        proc.stderr.close()
