import os
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
