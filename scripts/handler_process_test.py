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


class ProcessStreamingTest(unittest.TestCase):
    def run_python(self, code, timeout=5, **kwargs):
        with patch("builtins.print"):
            return run_process([sys.executable, "-c", code], timeout, **kwargs)

    def test_stderr_flood_before_stdout_cannot_deadlock(self):
        result = self.run_python(
            "import sys; sys.stderr.write('e' * (2 * 1024 * 1024)); "
            "sys.stderr.flush(); print('finished', flush=True)")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "finished")
        self.assertEqual(result.stderr, "e" * (2 * 1024 * 1024))

    def test_both_streams_flood_and_remain_separate(self):
        result = self.run_python("import os\nfor _ in range(128):\n"
                                 " os.write(1, b'o' * 16384)\n"
                                 " os.write(2, b'e' * 16384)\n")
        self.assertEqual(result.stdout, "o" * (128 * 16384))
        self.assertEqual(result.stderr, "e" * (128 * 16384))

    def test_silent_partial_line_and_continuous_output_timeout(self):
        for code in (
            "import time; time.sleep(30)",
            "import os,time; os.write(1,b'partial'); os.write(2,b'warning'); time.sleep(30)",
            "import os,time\nwhile True:\n os.write(1,b'x'*4096); time.sleep(.001)",
            "import os,time; os.close(1); os.close(2); time.sleep(30)",
        ):
            with self.subTest(code=code):
                start = time.monotonic()
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    self.run_python(code, timeout=0.3)
                self.assertLess(time.monotonic() - start, 3)
                if "partial" in code:
                    self.assertEqual(caught.exception.output, "partial")
                    self.assertEqual(caught.exception.stderr, "warning")

    def test_no_newline_and_utf8_preserved_in_result(self):
        result = self.run_python(
            "import os,time; os.write(1, b'\\xe2'); time.sleep(.01); "
            "os.write(1, b'\\x82\\xac tail'); os.write(2,b'last error')")
        self.assertEqual(result.stdout, "\u20ac tail")
        self.assertEqual(result.stderr, "last error")

    def test_streaming_stderr_arrives_before_process_exit(self):
        # The child requires a response to its stderr line before it exits.
        with tempfile.TemporaryDirectory() as tmp:
            ack = Path(tmp) / "ack"
            def acknowledge(message, **_):
                if message == "[stderr] ready":
                    ack.touch()
            code = ("import sys,time,pathlib; print('ready',file=sys.stderr,flush=True); "
                    f"p=pathlib.Path({str(ack)!r})\n"
                    "while not p.exists(): time.sleep(.01)\nprint('acknowledged')")
            with patch("builtins.print", side_effect=acknowledge):
                result = run_process([sys.executable, "-c", code], 3)
            self.assertEqual(result.stdout, "acknowledged")

    def test_capture_is_bounded_and_marked_but_draining_continues(self):
        result = self.run_python("import os; os.write(2,b'x'*100000+b'END'); print('done')",
                                 capture_limit=128)
        self.assertEqual(result.stdout, "done")
        self.assertEqual(result.stderr, "[earlier output truncated by handler]\n" + "x" * 125 + "END")

    def test_timeout_kills_descendants_even_after_parent_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "survivor"
            child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).touch()"
            code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
            with self.assertRaises(subprocess.TimeoutExpired):
                self.run_python(code, timeout=0.3)
            time.sleep(1.1)
            self.assertFalse(marker.exists(), "descendant survived the timeout")

    def test_logging_error_kills_and_reaps_child(self):
        real_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        with patch("handler_process.subprocess.Popen", side_effect=spawn), \
                patch("builtins.print", side_effect=BrokenPipeError("log sink closed")):
            with self.assertRaises(BrokenPipeError):
                run_process([sys.executable, "-c", "import time; print('hi',flush=True); time.sleep(30)"], 5)
        self.assertIsNotNone(children[0].returncode)
        self.assertTrue(children[0].stdout.closed)
        self.assertTrue(children[0].stderr.closed)


class HandlerProcessIntegrationTest(unittest.TestCase):
    def test_setup_and_post_stderr_flood(self):
        for stage in ("setup", "post"):
            with self.subTest(stage=stage), patch("builtins.print"):
                out, error = handler.run_shell_commands([[sys.executable, "-c",
                    "import sys; sys.stderr.write('e'*2000000); sys.stderr.flush(); print('ok')"]], stage, 5)
                self.assertIsNone(error)
                self.assertEqual(out, "ok")

    def test_setup_and_post_failures_preserve_stderr_and_exit_code(self):
        for stage in ("setup", "post"):
            with self.subTest(stage=stage), patch("builtins.print"):
                out, error = handler.run_shell_commands([[sys.executable, "-c",
                    "import sys; print('before'); print('diagnostic',file=sys.stderr); sys.exit(7)"]], stage, 5)
                self.assertIsNone(out)
                self.assertEqual(error["exit_code"], 7)
                self.assertEqual(error["stdout"], "before")
                self.assertEqual(error["stderr"], "diagnostic")

    def test_shell_timeout_returns_error(self):
        with patch("builtins.print"):
            out, error = handler.run_shell_commands(["sleep 30"], "setup", 0.2)
        self.assertIsNone(out)
        self.assertEqual(error["error"], "setup[0] timeout")

    def test_main_flood_result_and_temp_config_cleanup(self):
        config_paths = []
        def command(_, path):
            self.assertTrue(Path(path).exists())
            config_paths.append(path)
            return [sys.executable, "-c", "import sys; sys.stderr.write('e'*2000000); print('ok')"]
        with patch.object(handler, "build_mixlab_command", side_effect=command), patch("builtins.print"):
            result = handler.handler({"input": {"config_json": {"name": "test"}, "timeout": 5}})
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "ok")
        self.assertEqual(result["stderr"], "e" * 2000000)
        self.assertFalse(Path(config_paths[0]).exists())

    def test_config_cleanup_on_setup_failure_main_timeout_and_spawn_failure(self):
        for failure in ("setup", "timeout", "spawn", "post"):
            paths = []
            def command(_, path):
                paths.append(path)
                if failure == "spawn":
                    return ["/nonexistent-mixlab-executable"]
                if failure == "timeout":
                    return [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"]
                return [sys.executable, "-c", "print('ok')"]
            job = {"config_json": {"name": "test"}, "timeout": 0.3}
            if failure == "setup":
                job["setup"] = ["exit 9"]
            if failure == "post":
                job["post"] = ["exit 9"]
            with self.subTest(failure=failure), patch.object(handler, "build_mixlab_command", side_effect=command), patch("builtins.print"):
                result = handler.handler({"input": job})
                self.assertIn("post_error" if failure == "post" else "error", result)
                if failure == "timeout":
                    self.assertEqual(result["stdout"], "started")
                self.assertFalse(Path(paths[0]).exists())

    def test_invalid_timeouts_fail_before_process_start(self):
        for timeout in (0, -1, float("nan"), float("inf"), True, None, "10"):
            with self.subTest(timeout=timeout), patch.object(handler, "run_process") as run:
                result = handler.handler({"input": {"timeout": timeout}})
                self.assertIn("timeout must be", result["error"])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
