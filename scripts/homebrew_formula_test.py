import tempfile
import unittest
from pathlib import Path

import homebrew_formula as formula

TAG = "v0.118.0"
REVISION = "7b72184d055e8a17fccffed66360e7e4ae16d182"


class HomebrewFormulaTests(unittest.TestCase):
    def source(self):
        return (formula.ROOT / formula.SOURCE).read_text()

    def test_source_in_the_repository_is_a_template(self):
        # If a rendered formula were committed as the source, every later
        # release would silently republish the same version.
        text = self.source()
        for placeholder in formula.PLACEHOLDERS:
            self.assertIn(placeholder, text)

    def test_renders_the_release_tag_and_commit(self):
        text = formula.render(self.source(), TAG, REVISION)
        self.assertIn(f'tag:      "{TAG}"', text)
        self.assertIn(f'revision: "{REVISION}"', text)
        self.assertIn(f"at {TAG} by the publish-homebrew workflow", text)
        self.assertNotRegex(text, r"@[A-Z_]+@")
        self.assertIn('MLX_TESTED_MINIMUM = "', text)
        self.assertTrue(text.lstrip().startswith("# Rendered from packaging/homebrew/mixlab.rb"))

    def test_refuses_tags_that_are_not_final_releases(self):
        for tag in ("0.118.0", "v0.118", "v0.119.0-rc.1", "latest", "v0.118.0\n", ""):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "not a final release"):
                formula.render(self.source(), tag, REVISION)

    def test_refuses_revisions_that_are_not_full_commit_shas(self):
        for revision in (REVISION[:12], REVISION.upper(), REVISION[:-1], "g" * 40, ""):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "40-character"):
                formula.render(self.source(), TAG, revision)

    def test_missing_or_unknown_placeholders_fail(self):
        with self.assertRaisesRegex(ValueError, "missing @RELEASE_REVISION@"):
            formula.render(self.source().replace("@RELEASE_REVISION@", REVISION), TAG, REVISION)
        with self.assertRaisesRegex(ValueError, "unknown placeholders"):
            formula.render(self.source() + "# @RELEASE_DATE@\n", TAG, REVISION)

    def test_command_line_writes_the_output_or_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "mixlab.rb"
            formula.main(["--tag", TAG, "--revision", REVISION, "--output", str(output)])
            self.assertIn(f'tag:      "{TAG}"', output.read_text())

            failed = Path(temp) / "failed.rb"
            with self.assertRaises(SystemExit) as exit_:
                formula.main(["--tag", "v0.119.0-rc.1", "--revision", REVISION, "--output", str(failed)])
            self.assertEqual(exit_.exception.code, 1)
            self.assertFalse(failed.exists())


if __name__ == "__main__":
    unittest.main()
