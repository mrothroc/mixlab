import unittest
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import package_macos as package


class MacOSPackageTests(unittest.TestCase):
    def test_pipeline_is_offline_unless_notarization_requested(self):
        for notarize, status in [(False, 'Accepted'), (True, 'Accepted'), (True, 'Invalid')]:
            with self.subTest(notarize=notarize, status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'source'
                prefix = Path(temp) / 'mlx'
                root.mkdir()
                (root / 'LICENSE').write_text('Mixlab license')
                (root / 'THIRD_PARTY_NOTICES.md').write_text('Third-party notices fixture')
                for name in ['LICENSE', 'include/mlx/mlx.h', 'lib/libmlx.dylib', 'lib/mlx.metallib']:
                    path = prefix / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture')
                output = Path(temp) / 'output'
                args = argparse.Namespace(output=output, identity='Developer ID Application: Test',
                                          mlx_prefix=prefix, allow_dirty=False, notarize=notarize,
                                          notary_profile='test-profile')
                calls = []

                def fake_run(*command, **kwargs):
                    command = [str(x) for x in command]
                    calls.append(command)
                    if command[:3] == ['git', 'status', '--porcelain']:
                        return ''
                    if command[:2] == ['git', 'rev-parse']:
                        return 'revision'
                    if command[:2] == ['go', 'build']:
                        Path(command[command.index('-o') + 1]).write_bytes(b'binary')
                    if command[0] == 'otool':
                        return 'binary:' if command[1] == '-L' else ''
                    if command[-1] == '-version':
                        return Path(command[0]).name + ' test-build\nworker_protocol v1'
                    if command[:2] == ['hdiutil', 'create']:
                        staged = Path(command[command.index('-srcfolder') + 1]) / 'mixlab-macos-arm64'
                        self.assertEqual((staged / 'THIRD_PARTY_NOTICES.md').read_text(), 'Third-party notices fixture')
                        receipt = json.loads((staged / 'build-receipt.json').read_text())
                        self.assertIn('THIRD_PARTY_NOTICES.md', receipt['files'])
                        self.assertTrue(receipt['experimental_managed_cluster'])
                        self.assertNotIn('development_cluster_scaffold', receipt)
                        self.assertNotIn('scaffold', (staged / 'INSTALL.txt').read_text())
                        Path(command[-1]).write_bytes(b'disk image')
                    if command[:3] == ['xcrun', 'notarytool', 'submit']:
                        return json.dumps({'status': status, 'id': 'test-id'})
                    return ''

                with patch.object(package, 'ROOT', root), patch.object(package.platform, 'system', return_value='Darwin'), \
                     patch.object(package.platform, 'machine', return_value='arm64'), patch.object(package, 'run', side_effect=fake_run):
                    if status == 'Invalid':
                        with self.assertRaisesRegex(ValueError, 'notarization not accepted'):
                            package.build(args)
                        self.assertFalse((output / 'SHA256SUMS').exists())
                    else:
                        package.build(args)
                        self.assertTrue((output / 'SHA256SUMS').exists())
                self.assertFalse(list(output.glob('.stage-*')))
                submitted = any(c[:3] == ['xcrun', 'notarytool', 'submit'] for c in calls)
                stapled = any(c[:3] == ['xcrun', 'stapler', 'staple'] for c in calls)
                self.assertEqual(submitted, notarize)
                self.assertEqual(stapled, notarize and status == 'Accepted')

    def test_dependency_parser_preserves_spaces(self):
        output = "binary:\n\t/a path/libmlx.dylib (compatibility version 0.0.0, current version 0.0.0)\n"
        self.assertEqual(package.dependencies(output), ["/a path/libmlx.dylib"])
        with self.assertRaises(ValueError):
            package.dependencies("binary:\n unexpected")

    def test_only_system_or_bundled_libraries_allowed(self):
        for source in ("/opt/homebrew/lib/libmlx.dylib", "@rpath/libmlx.dylib", "@loader_path/libmlx.dylib"):
            self.assertEqual(package.local_dependency(source, {"libmlx.dylib"}), "@loader_path/libmlx.dylib")
        for source in ("/usr/lib/libSystem.B.dylib", "/System/Library/Frameworks/Metal.framework/Metal"):
            self.assertEqual(package.local_dependency(source, set()), source)
        for source in ("/opt/homebrew/lib/libunexpected.dylib", "@rpath/missing.dylib", "/usr/library/not-system.dylib"):
            with self.assertRaises(ValueError):
                package.local_dependency(source, {"libmlx.dylib"})

    def test_rpath_parser(self):
        self.assertEqual(package.rpaths("""Load command 10
          cmd LC_RPATH
      cmdsize 64
         path /a path/lib (offset 12)
Load command 11
          cmd LC_RPATH
      cmdsize 32
         path @executable_path (offset 12)
"""), ["/a path/lib", "@executable_path"])

    def test_signing_uses_runtime_timestamp_and_verifies(self):
        with patch.object(package, "run") as run:
            package.sign(Path("candidate"), "Developer ID Application: Example", "com.mixlab.mixlab")
            self.assertIn("--timestamp", run.call_args_list[0].args)
            self.assertIn("runtime", run.call_args_list[0].args)
            self.assertIn("--verify", run.call_args_list[1].args)
            self.assertNotIn("--deep", run.call_args_list[0].args)

    def test_matched_build_and_protocol_required(self):
        a = "mixlab (devel) (abc-dirty)\nworker_protocol v1"
        b = "mixlab-cluster (devel) (abc-dirty)\nworker_protocol v1\nscaffold"
        package.matched_identity(a, b)
        for bad in (b.replace("abc", "def"), b.replace("v1", "v2"), "empty"):
            with self.assertRaises(ValueError):
                package.matched_identity(a, bad)

    def test_audit_rejects_homebrew_or_rpath_residue(self):
        with patch.object(package, "run", return_value="binary:\n\t/opt/homebrew/lib/libmlx.dylib (compatibility version 0, current version 0)"):
            with self.assertRaises(ValueError):
                package.audit(Path("binary"), {"libmlx.dylib"})
        with patch.object(package, "run", side_effect=["binary:", "cmd LC_RPATH\ncmdsize 32\npath /tmp/lib (offset 12)"]):
            with self.assertRaises(ValueError):
                package.audit(Path("binary"), set())


if __name__ == "__main__":
    unittest.main()
