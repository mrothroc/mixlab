import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handler


class RunShellCommandsTest(unittest.TestCase):
    def setUp(self):
        self.old_config = os.environ.get("MIXLAB_CONFIG")
        os.environ["MIXLAB_CONFIG"] = "/tmp/config with spaces.json"

    def tearDown(self):
        if self.old_config is None:
            os.environ.pop("MIXLAB_CONFIG", None)
        else:
            os.environ["MIXLAB_CONFIG"] = self.old_config

    def test_shell_command_expands_environment(self):
        command = (
            f"{sys.executable} -c 'import sys; print(sys.argv[1:])' "
            "-config \"$MIXLAB_CONFIG\" -safetensors-load /tmp/weights.st"
        )

        stdout, err = handler.run_shell_commands([command], "post", 5)

        self.assertIsNone(err)
        self.assertIn("-config", stdout)
        self.assertIn("/tmp/config with spaces.json", stdout)
        self.assertIn("-safetensors-load", stdout)
        self.assertIn("/tmp/weights.st", stdout)

    def test_argv_command_expands_environment_without_shell(self):
        command = [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1:])",
            "-config",
            "$MIXLAB_CONFIG",
            "-safetensors-load",
            "/tmp/weights.st",
        ]

        stdout, err = handler.run_shell_commands([command], "post", 5)

        self.assertIsNone(err)
        self.assertIn("-config", stdout)
        self.assertIn("/tmp/config with spaces.json", stdout)
        self.assertIn("-safetensors-load", stdout)
        self.assertIn("/tmp/weights.st", stdout)


class BuildMixlabCommandTest(unittest.TestCase):
    def test_resume_paths_are_forwarded_verbatim(self):
        for path in ("/volume/checkpoints", "/volume/checkpoints/step_100/manifest.json",
                     "/volume/checkpoints with spaces/weights.safetensors"):
            with self.subTest(path=path):
                self.assertEqual(handler.build_mixlab_command(
                    {"mode": "arch", "resume": path}, "/config.json"),
                    ["mixlab", "-mode", "arch", "-config", "/config.json", "-resume", path])

    def test_resume_conflicts_with_weights_only_load(self):
        with self.assertRaisesRegex(ValueError, "resume and safetensors_load are mutually exclusive"):
            handler.build_mixlab_command(
                {"resume": "/checkpoint", "safetensors_load": "/weights.st"}, None)

    def test_existing_checkpoint_write_command_unchanged(self):
        self.assertEqual(handler.build_mixlab_command({
            "mode": "arch", "train": "/data/*.bin", "checkpoint_dir": "/checkpoints",
            "checkpoint_every": 100,
        }, None), ["mixlab", "-mode", "arch", "-train", "/data/*.bin",
                   "-checkpoint-dir", "/checkpoints", "-checkpoint-every", "100"])

    def test_empty_resume_does_not_conflict_with_weights_load(self):
        self.assertEqual(handler.build_mixlab_command({
            "resume": "", "safetensors_load": "/weights.st",
        }, None), ["mixlab", "-mode", "smoke", "-safetensors-load", "/weights.st"])

    def test_handler_rejects_conflict_before_setup_and_cleans_config(self):
        with patch.object(handler, "run_shell_commands") as setup, \
                patch.object(handler.subprocess, "Popen") as popen, \
                patch.object(handler.os, "unlink", wraps=os.unlink) as unlink:
            result = handler.handler({"input": {
                "resume": "/checkpoint", "safetensors_load": "/weights.st",
                "config_json": {"model_dim": 16}, "setup": ["echo setup"],
            }})
            self.assertIn("mutually exclusive", result["error"])
            setup.assert_not_called()
            popen.assert_not_called()
            config_paths = [Path(call.args[0]) for call in unlink.call_args_list
                            if str(call.args[0]).endswith(".json")]
            self.assertEqual(len(config_paths), 1)
            self.assertFalse(config_paths[0].exists())

    def test_temperature_zero_is_forwarded(self):
        """temperature=0 selects greedy decoding and must not be dropped as falsy."""
        cmd = handler.build_mixlab_command({"mode": "generate", "temperature": 0}, None)

        self.assertIn("-temperature", cmd)
        self.assertEqual(cmd[cmd.index("-temperature") + 1], "0")

    def test_temperature_omitted_when_absent(self):
        cmd = handler.build_mixlab_command({"mode": "generate"}, None)

        self.assertNotIn("-temperature", cmd)

    def test_config_path_and_mode(self):
        cmd = handler.build_mixlab_command({"mode": "arch"}, "/examples/x.json")

        self.assertEqual(cmd[:3], ["mixlab", "-mode", "arch"])
        self.assertIn("-config", cmd)
        self.assertEqual(cmd[cmd.index("-config") + 1], "/examples/x.json")


class BuildJobEnvTest(unittest.TestCase):
    def test_job_env_is_forwarded_as_strings(self):
        env = handler.build_job_env({"env": {"MIXLAB_FLAG": 1}}, None)

        self.assertEqual(env["MIXLAB_FLAG"], "1")

    def test_config_path_exported(self):
        env = handler.build_job_env({}, "/tmp/cfg.json")

        self.assertEqual(env["MIXLAB_CONFIG"], "/tmp/cfg.json")

    def test_does_not_mutate_process_env(self):
        """Warm workers reuse the process, so job env must not leak between jobs."""
        handler.build_job_env({"env": {"MIXLAB_LEAK_CHECK": "1"}}, None)

        self.assertNotIn("MIXLAB_LEAK_CHECK", os.environ)

    def test_env_reaches_shell_command(self):
        env = handler.build_job_env({"env": {"MIXLAB_TEST_VAR": "hello"}}, None)

        stdout, err = handler.run_shell_commands(
            ['echo "$MIXLAB_TEST_VAR"'], "post", 5, env=env
        )

        self.assertIsNone(err)
        self.assertIn("hello", stdout)

    def test_env_reaches_argv_command(self):
        env = handler.build_job_env({"env": {"MIXLAB_TEST_VAR": "hello"}}, None)

        stdout, err = handler.run_shell_commands(
            [[sys.executable, "-c", "import sys; print(sys.argv[1])", "$MIXLAB_TEST_VAR"]],
            "post",
            5,
            env=env,
        )

        self.assertIsNone(err)
        self.assertIn("hello", stdout)


if __name__ == "__main__":
    unittest.main()
