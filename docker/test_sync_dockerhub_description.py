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


class TestShortDescription(unittest.TestCase):
    """The one-line description is a separate Docker Hub field, and it drifts.

    On 2026-09-15 mixlab-cuda's short description still read "sm_80/86/89" after
    the long one was corrected to include sm_90 — the same understatement, in the
    field that shows in search results. It lives in the same file as the overview
    so the two are edited together, and it is REQUIRED: a file without one would
    otherwise leave a stale line published while reporting success.
    """

    def test_extracts_the_marker_and_strips_it_from_the_body(self):
        short, body = sync.split_description(
            "<!-- short: A one-line pitch. -->\n# Title\n\nBody text.\n")
        self.assertEqual(short, "A one-line pitch.")
        self.assertNotIn("<!-- short:", body)
        self.assertTrue(body.lstrip().startswith("# Title"))

    def test_marker_is_tolerated_anywhere_in_the_file(self):
        short, body = sync.split_description(
            "# Title\n\n<!-- short: Later marker. -->\nBody.\n")
        self.assertEqual(short, "Later marker.")
        self.assertNotIn("<!-- short:", body)

    def test_whitespace_around_the_marker_is_ignored(self):
        short, _ = sync.split_description("<!--   short:   Padded.   -->\n# T\n")
        self.assertEqual(short, "Padded.")

    def test_missing_marker_raises(self):
        with self.assertRaises(sync.DescriptionError):
            sync.split_description("# Title\n\nNo marker here.\n")

    def test_empty_marker_raises(self):
        with self.assertRaises(sync.DescriptionError):
            sync.split_description("<!-- short:    -->\n# Title\n")

    def test_over_the_hub_limit_raises(self):
        over = "x" * (sync.MAX_SHORT_DESCRIPTION_CHARS + 1)
        with self.assertRaises(sync.DescriptionError):
            sync.split_description(f"<!-- short: {over} -->\n# T\n")

    def test_exactly_at_the_limit_is_accepted(self):
        exact = "x" * sync.MAX_SHORT_DESCRIPTION_CHARS
        short, _ = sync.split_description(f"<!-- short: {exact} -->\n# T\n")
        self.assertEqual(len(short), sync.MAX_SHORT_DESCRIPTION_CHARS)

    def test_stripping_leaves_a_non_empty_body(self):
        # A file that is nothing but a marker would publish a blank overview.
        with self.assertRaises(sync.DescriptionError):
            sync.split_description("<!-- short: Only a marker. -->\n")

    def test_every_shipped_overview_carries_one(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("DOCKERHUB.md", "DOCKERHUB-cuda.md", "DOCKERHUB-cuda-base.md"):
            path = os.path.join(root, "docker", name)
            with open(path, encoding="utf-8") as handle:
                short, body = sync.split_description(handle.read())
            self.assertTrue(short, f"{name} has an empty short description")
            self.assertLessEqual(len(short), sync.MAX_SHORT_DESCRIPTION_CHARS)
            self.assertIn("mixlab", body)


class TestPayload(unittest.TestCase):
    def test_payload_is_json_with_full_description(self):
        body = sync.build_payload("# mixlab\n\n\"quoted\" & <tagged>\n")
        self.assertEqual(body["full_description"],
                         "# mixlab\n\n\"quoted\" & <tagged>\n")
        self.assertEqual(list(body), ["full_description"])

    def test_payload_carries_the_short_description_when_given(self):
        body = sync.build_payload("# mixlab\n", short="One line.")
        self.assertEqual(body["description"], "One line.")
        self.assertEqual(sorted(body), ["description", "full_description"])

    def test_payload_touches_no_other_repository_fields(self):
        # A PATCH that included is_private or similar could change repo settings.
        body = sync.build_payload("# mixlab\n", short="One line.")
        self.assertEqual(sorted(body), ["description", "full_description"])

    def test_payload_encodes_without_shell_escaping_hazards(self):
        # Built with json.dumps rather than string interpolation: markdown is
        # full of quotes, backticks and newlines that break naive construction.
        import json
        raw = json.dumps(sync.build_payload("a\"b`c\nd"))
        self.assertEqual(json.loads(raw)["full_description"], "a\"b`c\nd")


class TestTokenSource(unittest.TestCase):
    """The token is read from a named env var, never from argv.

    Cloud Build refuses to bind one secret version to two env names, so the
    caller has to be able to say which variable already holds the secret. A
    build that got this wrong failed validation before running any step.
    """

    def test_reads_the_named_variable(self):
        os.environ["TEST_TOKEN_VAR"] = "s3cret"
        try:
            self.assertEqual(sync.read_token("TEST_TOKEN_VAR"), "s3cret")
        finally:
            del os.environ["TEST_TOKEN_VAR"]

    def test_reads_the_cloud_build_variable_name(self):
        # The pipeline passes --token-env=DOCKER_TOKEN, reusing the push step's
        # binding rather than declaring a second one.
        os.environ["DOCKER_TOKEN"] = "from-push-step"
        try:
            self.assertEqual(sync.read_token("DOCKER_TOKEN"), "from-push-step")
        finally:
            del os.environ["DOCKER_TOKEN"]

    def test_unset_variable_yields_empty_not_an_error(self):
        # An unset token is a skip, not a failure — same as the push step.
        os.environ.pop("DEFINITELY_UNSET_VAR", None)
        self.assertEqual(sync.read_token("DEFINITELY_UNSET_VAR"), "")

    def test_empty_env_name_raises(self):
        with self.assertRaises(sync.DescriptionError):
            sync.read_token("")


class TestCredentialHandling(unittest.TestCase):
    def test_missing_credentials_is_a_skip_not_a_failure(self):
        # Mirrors the existing Docker Hub push step: a build without the secret
        # configured should no-op, not break the pipeline.
        self.assertTrue(sync.should_skip(user="", token="tok"))
        self.assertTrue(sync.should_skip(user="michaelrothrock", token=""))

    def test_complete_credentials_do_not_skip(self):
        self.assertFalse(sync.should_skip(user="michaelrothrock", token="tok"))


class TestTargetParsing(unittest.TestCase):
    def test_parses_repo_and_path(self):
        self.assertEqual(sync.parse_target("mixlab-cuda=docker/DOCKERHUB-cuda.md"),
                         ("mixlab-cuda", "docker/DOCKERHUB-cuda.md"))

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(sync.parse_target(" mixlab = docker/DOCKERHUB.md "),
                         ("mixlab", "docker/DOCKERHUB.md"))

    def test_missing_separator_raises(self):
        with self.assertRaises(sync.DescriptionError):
            sync.parse_target("mixlab")

    def test_empty_side_raises(self):
        for spec in ("=docker/DOCKERHUB.md", "mixlab=", "=", " = "):
            with self.assertRaises(sync.DescriptionError):
                sync.parse_target(spec)


class TestPublishedArchClaims(unittest.TestCase):
    """The GPU architecture list must match what the build actually compiles.

    On 2026-09-15 four files disagreed: README.md and docker/DOCKERHUB.md claimed
    sm_80/86/89 while the build config and docker/README.md said 80;86;89;90. The
    published image's own layer history settled it — ARCHS=80;86;89;90 — meaning
    the overview live on Docker Hub understated GPU support and turned away H100
    users. The list is derivable from the build config, so it is checked here
    instead of being hand-maintained in five places.
    """

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BUILD_CONFIG = os.path.join(REPO_ROOT, "docker", "cloudbuild-golf-mlx-cuda.yaml")
    CLAIM_FILES = ("README.md", "docker/DOCKERHUB.md",
                   "docker/DOCKERHUB-cuda.md", "docker/DOCKERHUB-cuda-base.md")

    def built_archs(self):
        """The _ARCHS substitution, the authoritative list. Missing = abort."""
        import re
        with open(self.BUILD_CONFIG, encoding="utf-8") as handle:
            match = re.search(r'^\s*_ARCHS:\s*"([0-9;]+)"', handle.read(), re.M)
        self.assertIsNotNone(
            match, f"_ARCHS not found in {self.BUILD_CONFIG} — cannot verify claims")
        archs = [a for a in match.group(1).split(";") if a]
        self.assertGreaterEqual(len(archs), 1, "_ARCHS parsed to an empty list")
        return archs

    def claims_in(self, relative_path):
        import re
        with open(os.path.join(self.REPO_ROOT, relative_path), encoding="utf-8") as h:
            return set(re.findall(r"sm_(\d+)", h.read()))

    def test_no_file_claims_an_architecture_the_build_does_not_compile(self):
        built = set(self.built_archs())
        for relative_path in self.CLAIM_FILES:
            extra = self.claims_in(relative_path) - built
            self.assertFalse(
                extra,
                f"{relative_path} claims sm_{sorted(extra)} but _ARCHS is "
                f"{sorted(built)} — the image does not support it")

    def test_the_user_facing_overviews_list_every_built_architecture(self):
        # Understating support is the failure that actually happened: an H100
        # user reads sm_80/86/89 and goes elsewhere.
        built = set(self.built_archs())
        for relative_path in ("docker/DOCKERHUB.md", "docker/DOCKERHUB-cuda.md"):
            missing = built - self.claims_in(relative_path)
            self.assertFalse(
                missing,
                f"{relative_path} omits sm_{sorted(missing)}, which the build "
                f"compiles — it understates GPU support")


if __name__ == "__main__":
    unittest.main(verbosity=2)
