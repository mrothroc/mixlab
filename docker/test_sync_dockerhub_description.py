#!/usr/bin/env python3
"""Tests for sync_dockerhub_description.py.

The guards are the whole point of this script. Docker Hub's PATCH endpoint
happily accepts an empty full_description, so a sync that silently sends one
would REPLACE a working overview with a blank page and report success. That is
the same failure class the doc generators guard against: replacing a real fact
with no fact while claiming to have worked.

Network calls are not exercised here. Everything below is the validation layer
that runs before any request is made.
"""
import importlib.util
import os
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "sync_dockerhub_description",
    os.path.join(os.path.dirname(__file__), "sync_dockerhub_description.py"),
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


class TestLoadDescription(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def write(self, text, name="DOCKERHUB.md"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_reads_a_normal_file(self):
        path = self.write("# mixlab\n\nExplore ML architectures fast.\n")
        self.assertIn("Explore ML architectures", sync.load_description(path))

    def test_missing_file_raises(self):
        # A renamed or moved file must abort the sync, never blank the page.
        with self.assertRaises(sync.DescriptionError):
            sync.load_description(os.path.join(self.tmp, "nope.md"))

    def test_empty_file_raises(self):
        with self.assertRaises(sync.DescriptionError):
            sync.load_description(self.write(""))

    def test_whitespace_only_file_raises(self):
        # Degenerate in exactly the way an empty one is, and easier to produce.
        with self.assertRaises(sync.DescriptionError):
            sync.load_description(self.write("\n\n   \n\t\n"))

    def test_oversized_file_raises(self):
        # Docker Hub caps full_description at 25,000 characters. Over the cap the
        # API truncates rather than erroring, so catch it here where we can say why.
        with self.assertRaises(sync.DescriptionError):
            sync.load_description(self.write("x" * (sync.MAX_DESCRIPTION_CHARS + 1)))

    def test_file_at_the_exact_limit_is_accepted(self):
        text = "x" * sync.MAX_DESCRIPTION_CHARS
        self.assertEqual(len(sync.load_description(self.write(text))),
                         sync.MAX_DESCRIPTION_CHARS)

    def test_unicode_survives_the_round_trip(self):
        # The overview contains arrows and em-dashes; a bad encoding assumption
        # here would corrupt the published page.
        path = self.write("# mixlab\n\nconfig → trained → Hugging Face — MIT\n")
        self.assertIn("→ trained →", sync.load_description(path))


class TestPayload(unittest.TestCase):
    def test_payload_is_json_with_full_description(self):
        body = sync.build_payload("# mixlab\n\n\"quoted\" & <tagged>\n")
        self.assertEqual(body["full_description"],
                         "# mixlab\n\n\"quoted\" & <tagged>\n")
        self.assertEqual(list(body), ["full_description"])

    def test_payload_encodes_without_shell_escaping_hazards(self):
        # Built with json.dumps rather than string interpolation: markdown is
        # full of quotes, backticks and newlines that break naive construction.
        import json
        raw = json.dumps(sync.build_payload("a\"b`c\nd"))
        self.assertEqual(json.loads(raw)["full_description"], "a\"b`c\nd")


class TestCredentialHandling(unittest.TestCase):
    def test_missing_credentials_is_a_skip_not_a_failure(self):
        # Mirrors the existing Docker Hub push step: a build without the secret
        # configured should no-op, not break the pipeline.
        self.assertTrue(sync.should_skip(user="", token="tok"))
        self.assertTrue(sync.should_skip(user="michaelrothrock", token=""))

    def test_complete_credentials_do_not_skip(self):
        self.assertFalse(sync.should_skip(user="michaelrothrock", token="tok"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
