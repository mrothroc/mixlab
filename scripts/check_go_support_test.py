import io
import unittest
from contextlib import redirect_stdout

import check_go_support as check

# Shape of https://go.dev/dl/?mode=json: the currently supported releases, newest
# patch of each, trimmed to the fields the check reads.
RELEASES = [
    {"version": "go1.27.1", "stable": True},
    {"version": "go1.26.8", "stable": True},
]


class ToolchainVersionTest(unittest.TestCase):
    def test_reads_the_toolchain_line(self):
        text = "module example\n\ngo 1.26.0\n\ntoolchain go1.27.1\n"
        self.assertEqual(check.toolchain_version(text), (1, 27, 1))

    def test_refuses_a_go_mod_without_a_toolchain_line(self):
        # The go directive is the importers' minimum, not the build toolchain.
        with self.assertRaisesRegex(ValueError, "toolchain"):
            check.toolchain_version("module example\n\ngo 1.26.0\n")

    def test_refuses_two_toolchain_lines(self):
        with self.assertRaises(ValueError):
            check.toolchain_version("toolchain go1.27.1\ntoolchain go1.26.8\n")


class SupportedReleasesTest(unittest.TestCase):
    def test_reads_newest_patch_per_series(self):
        self.assertEqual(check.supported_releases(RELEASES), {(1, 27): 1, (1, 26): 8})

    def test_ignores_unstable_releases(self):
        releases = RELEASES + [{"version": "go1.28rc1", "stable": False}]
        self.assertEqual(set(check.supported_releases(releases)), {(1, 27), (1, 26)})

    def test_an_empty_list_is_an_error_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, "no supported"):
            check.supported_releases([])
        with self.assertRaisesRegex(ValueError, "no supported"):
            check.supported_releases([{"version": "go1.28rc1", "stable": False}])


class CheckTest(unittest.TestCase):
    def run_check(self, version):
        out = io.StringIO()
        with redirect_stdout(out):
            ok = check.check(version, check.supported_releases(RELEASES))
        return ok, out.getvalue()

    def test_newest_patch_of_a_supported_series_passes_quietly(self):
        ok, out = self.run_check((1, 27, 1))
        self.assertTrue(ok)
        self.assertNotIn("::", out)

    def test_older_supported_series_passes(self):
        ok, _ = self.run_check((1, 26, 8))
        self.assertTrue(ok)

    def test_stale_patch_passes_with_a_warning(self):
        # govulncheck fails the build when a missing patch fixes reachable code.
        ok, out = self.run_check((1, 27, 0))
        self.assertTrue(ok)
        self.assertIn("::warning::", out)
        self.assertIn("go1.27.1", out)

    def test_unsupported_series_fails(self):
        ok, out = self.run_check((1, 24, 13))
        self.assertFalse(ok)
        self.assertIn("::error::", out)
        self.assertIn("go1.24.13", out)

    def test_series_newer_than_any_release_fails(self):
        ok, _ = self.run_check((1, 28, 0))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
