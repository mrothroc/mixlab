import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handler
from handler_process import run_process
from handler_watchdog import TrainingStall, TrainingWatchdog, validate_watchdog_input


class WatchdogTest(unittest.TestCase):
    def test_only_new_commits_reset_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [0]
            progress = Path(tmp) / "progress"
            watch = TrainingWatchdog(10, progress, tmp, clock=lambda: now[0])
            watch.start(123)
            progress.write_text(json.dumps({"pid": 123, "optimizer_steps": 53025, "step": 53025}))
            now[0] = 9
            watch.check()
            self.assertEqual(watch.last_progress, 9)
            # Attempts, refreshed mtimes and log activity are not commits.
            progress.write_text(json.dumps({"pid": 123, "optimizer_steps": 53025, "step": 53026}))
            now[0] = 18
            watch.check()
            self.assertEqual(watch.last_progress, 9)
            now[0] = 19
            with patch.object(watch, "capture") as capture, self.assertRaises(TrainingStall):
                watch.check()
            capture.assert_called_once()

    def test_missing_malformed_wrong_pid_or_backward_progress_cannot_keep_alive(self):
        for record in (None, "{", "[]", '{"pid":456,"optimizer_steps":999}',
                       '{"pid":123,"optimizer_steps":-1}',
                       '{"pid":123,"optimizer_steps":true}'):
            with self.subTest(record=record), tempfile.TemporaryDirectory() as tmp:
                now = [0]
                p = Path(tmp) / "progress"
                watch = TrainingWatchdog(10, p, tmp, clock=lambda: now[0])
                watch.start(123)
                if record is not None:
                    p.write_text(record)
                now[0] = 10
                with patch.object(watch, "capture"), self.assertRaises(TrainingStall):
                    watch.check()

    def test_capture_errors_still_raise_training_stall(self):
        watch = TrainingWatchdog(1, "/missing", "/missing", clock=lambda: 2)
        watch.start(123)
        watch.last_progress = 0
        with patch.object(watch, "capture", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(TrainingStall, "capture failed: disk full"):
            watch.check()

    def test_capture_is_bounded_and_keeps_debugger_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch = TrainingWatchdog(10, "/missing", tmp)
            watch.start(123)
            outcomes = [subprocess.TimeoutExpired("gdb", 30, output="partial stack"),
                        OSError("nvidia-smi not found")]
            with patch("handler_watchdog.run_process", side_effect=outcomes) as run, \
                    patch("handler_watchdog.os.kill") as kill:
                watch.capture()
            self.assertEqual(run.call_args_list[0].args[1], 30)
            self.assertEqual(run.call_args_list[1].args[1], 5)
            self.assertFalse(run.call_args_list[0].kwargs["stream_output"])
            self.assertIn("partial stack", (Path(tmp) / "native-stacks.txt").read_text())
            self.assertIn("unavailable", (Path(tmp) / "nvidia-smi.txt").read_text())
            self.assertTrue((Path(tmp) / "stall.json").exists())
            self.assertTrue((Path(tmp) / "proc.json").exists())
            kill.assert_called_once()

    def test_live_noisy_stall_captures_before_killing_and_returns_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch = TrainingWatchdog(.3, Path(tmp) / "progress", tmp)
            captured = []
            def capture():
                os.kill(watch.pid, 0)  # Must still be alive when capture starts.
                captured.append(watch.pid)
            with patch.object(watch, "capture", side_effect=capture), patch("builtins.print"):
                start = time.monotonic()
                with self.assertRaises(TrainingStall) as caught:
                    run_process([sys.executable, "-c",
                                 "import time\nwhile True:\n print('still logging',flush=True); time.sleep(.01)"],
                                5, watchdog=watch)
            self.assertLess(time.monotonic() - start, 2)
            self.assertEqual(captured, [watch.pid])
            self.assertIn("still logging", caught.exception.output)
            with self.assertRaises(ProcessLookupError):
                os.kill(watch.pid, 0)

    def test_closed_pipes_do_not_bypass_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            watch = TrainingWatchdog(.2, Path(tmp) / "progress", tmp)
            with patch.object(watch, "capture"), self.assertRaises(TrainingStall):
                run_process([sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(30)"],
                            5, watchdog=watch)

    def test_real_progress_file_keeps_a_silent_child_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress"
            watch = TrainingWatchdog(.5, progress, tmp)
            code = ("import os,time,json,pathlib\n"
                    f"p=pathlib.Path({str(progress)!r})\n"
                    "for i in range(1,5):\n"
                    " t=p.with_suffix('.tmp')\n"
                    " t.write_text(json.dumps({'pid':os.getpid(),'optimizer_steps':i,'step':i}))\n"
                    " t.replace(p)\n"
                    " time.sleep(.2)\n")
            with patch.object(watch, "capture") as capture:
                result = run_process([sys.executable, "-c", code], 5, watchdog=watch)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(watch.committed, 4)
            capture.assert_not_called()


class WatchdogHandlerTest(unittest.TestCase):
    def test_validation(self):
        validate_watchdog_input({})
        for seconds in (0, -1, True, "10", float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                validate_watchdog_input({"stall_timeout": seconds, "stall_dump_dir": "/tmp/x", "mode": "arch"})
        for config in ({"stall_timeout": 10}, {"stall_dump_dir": "/tmp/x"},
                       {"stall_timeout": 10, "stall_dump_dir": "relative", "mode": "arch"},
                       {"stall_timeout": 10, "stall_dump_dir": "/tmp/x", "mode": "arch_race"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_watchdog_input(config)
        validate_watchdog_input({"stall_timeout": 10, "stall_dump_dir": "/tmp/x", "mode": "arch"})

    def test_invalid_watchdog_fails_before_setup(self):
        with patch.object(handler, "run_shell_commands") as setup:
            result = handler.handler({"input": {"stall_timeout": -1, "setup": ["echo unsafe"]}})
        self.assertIn("stall_timeout", result["error"])
        setup.assert_not_called()

    def test_handler_enables_local_progress_and_persists_stall_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            envs = []
            def stalled(command, timeout, *, env, watchdog):
                envs.append(dict(env))
                self.assertEqual(env["MIXLAB_DEBUG_PTRACER_PID"], str(os.getpid()))
                raise TrainingStall(watchdog.seconds, watchdog.directory)
            with patch.object(handler, "run_process", side_effect=stalled):
                result = handler.handler({"input": {"mode": "arch", "stall_timeout": 10, "stall_dump_dir": tmp}})
            self.assertIn("no optimizer progress", result["error"])
            self.assertTrue(Path(result["diagnostics"]).is_dir())
            self.assertFalse(Path(envs[0]["MIXLAB_PROGRESS_FILE"]).parent.exists())

    def test_timing_is_a_go_boolean_flag(self):
        self.assertEqual(handler.build_mixlab_command({"timing": True}, None), ["mixlab", "-mode", "smoke", "-timing"])
        self.assertEqual(handler.build_mixlab_command({"timing": False}, None), ["mixlab", "-mode", "smoke"])
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                handler.build_mixlab_command({"timing": value}, None)
        cmd = handler.build_mixlab_command({"telemetry_out": "/volume/progress.jsonl"}, None)
        self.assertEqual(cmd[-2:], ["-telemetry-out", "/volume/progress.jsonl"])

    def test_watchdog_environment_is_main_process_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            def main(command, timeout, *, env, watchdog):
                self.assertIn("MIXLAB_PROGRESS_FILE", env)
                return subprocess.CompletedProcess(command, 0, "ok", "")
            with patch.object(handler, "run_process", side_effect=main), \
                    patch.object(handler, "run_shell_commands", return_value=("ok", None)) as shell:
                result = handler.handler({"input": {"mode": "arch", "stall_timeout": 10,
                    "stall_dump_dir": tmp, "setup": ["true"], "post": ["true"]}})
            self.assertEqual(result["exit_code"], 0)
            for call in shell.call_args_list:
                self.assertNotIn("MIXLAB_PROGRESS_FILE", call.kwargs["env"])
                self.assertNotIn("MIXLAB_DEBUG_PTRACER_PID", call.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
