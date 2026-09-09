"""Bounded, best-effort dashboard logging, independent of process supervision."""

import builtins
import queue
import threading
import time


class LogSink:
    def __init__(self):
        # Messages are already chunked to 64 KiB by handler_process.
        self.queue = queue.Queue(maxsize=64)
        threading.Thread(target=self._run, name="dashboard-log", daemon=True).start()

    def emit(self, message):
        try:
            self.queue.put_nowait((builtins.print, message))
        except queue.Full:
            pass  # Returned stdout/stderr tails remain independent and intact.

    def _run(self):
        while True:
            printer, message = self.queue.get()
            try:
                printer(message, flush=True)
            except Exception:
                pass  # A broken dashboard must not kill or deadlock training.
            finally:
                self.queue.task_done()

    def flush(self, seconds=0.1):
        end = time.monotonic() + seconds
        while self.queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.005)


# One worker across warm jobs; a permanently blocked sink cannot leak a thread
# per command. Do not wait unboundedly for it, including at interpreter exit.
_sink = LogSink()


def emit(message):
    _sink.emit(message)


def flush():
    _sink.flush()
