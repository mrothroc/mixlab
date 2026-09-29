import unittest
import shutil
import subprocess
import generate_macos_cask as cask


class CaskTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ruby"), "Ruby is optional")
    def test_ruby_syntax(self):
        result = subprocess.run(["ruby", "-c"], input=cask.render("v1.2.3", "a"*64),
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_signed_artifact_and_stable_path(self):
        result = cask.from_checksums("v1.2.3", "a" * 64 + "  mixlab-v1.2.3-macos-arm64.dmg\n")
        self.assertIn('sha256 "' + "a" * 64 + '"', result)
        self.assertIn('suite "mixlab-macos-arm64", target: "Mixlab"', result)
        self.assertIn('#{appdir}/Mixlab/mixlab-cluster', result)
        for bad in ("Cellar", "codesign", "xattr", "strip", "no_check", "zap"):
            self.assertNotIn(bad, result)

    def test_refuses_ambiguous_or_unverified_checksums(self):
        for tag, text in [("latest", ""), ("v1.2.3", "bad mixlab-v1.2.3-macos-arm64.dmg"),
                          ("v1.2.3", ("a"*64+"  mixlab-v1.2.3-macos-arm64.dmg\n")*2)]:
            with self.assertRaises(ValueError):
                cask.from_checksums(tag, text)
