import unittest
import argparse
import json
from pathlib import Path
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import package_macos as package


def make_fixture(temp, mlx_version=(0, 32, 1), tested=("0.32.0", "0.33.0")):
    """Source tree and MLX prefix with everything build() reads, including the
    MLX version header and the formula's tested-range constants."""
    root = Path(temp) / 'source'
    prefix = Path(temp) / 'mlx'
    root.mkdir()
    (root / 'LICENSE').write_text('Mixlab license')
    (root / 'THIRD_PARTY_NOTICES.md').write_text('Third-party notices fixture')
    (root / 'packaging' / 'homebrew').mkdir(parents=True)
    (root / 'packaging' / 'homebrew' / 'mixlab.rb').write_text(
        'class Mixlab < Formula\n'
        f'  MLX_TESTED_MINIMUM = "{tested[0]}".freeze\n'
        f'  MLX_TESTED_BELOW = "{tested[1]}".freeze\n'
        'end\n')
    for name in ['LICENSE', 'include/mlx/mlx.h', 'lib/libmlx.dylib', 'lib/mlx.metallib']:
        path = prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    major, minor, patch_level = mlx_version
    (prefix / 'include/mlx/version.h').write_text(
        '#pragma once\n'
        f'#define MLX_VERSION_MAJOR {major}\n'
        f'#define MLX_VERSION_MINOR {minor}\n'
        f'#define MLX_VERSION_PATCH {patch_level}\n')
    return root, prefix


def namespace(output, prefix, **overrides):
    values = dict(output=output, identity='Developer ID Application: Test', mlx_prefix=prefix,
                  allow_dirty=False, notarize=False, notary_profile='test-profile', release=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_tools(calls, version='test-build', status='Accepted', dirty=False, on_image=None):
    def run(*command, **kwargs):
        command = [str(x) for x in command]
        calls.append(command)
        if command[:3] == ['git', 'status', '--porcelain']:
            return ' M changed.go' if dirty else ''
        if command[:2] == ['git', 'rev-parse']:
            return 'revision'
        if command[:2] == ['go', 'build']:
            Path(command[command.index('-o') + 1]).write_bytes(b'binary')
        if command[0] == 'otool':
            return 'binary:' if command[1] == '-L' else ''
        if command[-1] == '-version':
            return f'{Path(command[0]).name} {version}\nworker_protocol v1'
        if command[:2] == ['hdiutil', 'create']:
            if on_image:
                on_image(command)
            Path(command[-1]).write_bytes(b'disk image')
        if command[:3] == ['xcrun', 'notarytool', 'submit']:
            return json.dumps({'status': status, 'id': 'test-id'})
        return ''
    return run


@contextmanager
def patched(root, run):
    with patch.object(package, 'ROOT', root), patch.object(package.platform, 'system', return_value='Darwin'), \
         patch.object(package.platform, 'machine', return_value='arm64'), patch.object(package, 'run', side_effect=run):
        yield


def started_build(calls):
    return any(c[:2] == ['go', 'build'] for c in calls)


class MacOSPackageTests(unittest.TestCase):
    def test_pipeline_is_offline_unless_notarization_requested(self):
        for notarize, status in [(False, 'Accepted'), (True, 'Accepted'), (True, 'Invalid')]:
            with self.subTest(notarize=notarize, status=status), tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp)
                output = Path(temp) / 'output'
                args = namespace(output, prefix, notarize=notarize)
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

    def test_mlx_outside_the_formula_tested_range_is_refused(self):
        # The formula is the one place the tested MLX range is recorded; a
        # packaged build must not bundle an MLX the formula would refuse.
        for mlx, allowed in [((0, 32, 0), True), ((0, 32, 1), True), ((0, 33, 0), False),
                             ((0, 31, 9), False), ((1, 0, 0), False)]:
            with self.subTest(mlx=mlx), tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp, mlx_version=mlx)
                output = Path(temp) / 'output'
                calls, seen = [], {}

                def keep_receipt(command):
                    stage = Path(command[command.index('-srcfolder') + 1]) / 'mixlab-macos-arm64'
                    seen.update(json.loads((stage / 'build-receipt.json').read_text()))

                with patched(root, fake_tools(calls, on_image=keep_receipt)):
                    if allowed:
                        package.build(namespace(output, prefix))
                        self.assertEqual(seen['mlx_version'], '.'.join(map(str, mlx)))
                    else:
                        with self.assertRaisesRegex(ValueError, 'outside the tested range'):
                            package.build(namespace(output, prefix))
                        self.assertFalse(started_build(calls))
                        self.assertFalse(output.exists())

    def test_unreadable_mlx_version_or_tested_range_fails_loudly(self):
        # A range check that silently passes when it cannot parse its inputs
        # is the vacuous guard this project has been bitten by before.
        def drop_patch_macro(root, prefix):
            header = prefix / 'include/mlx/version.h'
            header.write_text(header.read_text().replace('#define MLX_VERSION_PATCH 1\n', ''))

        def drop_upper_bound(root, prefix):
            formula = root / 'packaging' / 'homebrew' / 'mixlab.rb'
            formula.write_text(formula.read_text().replace('MLX_TESTED_BELOW', 'SOMETHING_ELSE'))

        def delete_header(root, prefix):
            (prefix / 'include/mlx/version.h').unlink()

        for breakage in (drop_patch_macro, drop_upper_bound, delete_header):
            with self.subTest(breakage=breakage.__name__), tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp)
                breakage(root, prefix)
                calls = []
                with patched(root, fake_tools(calls)):
                    with self.assertRaisesRegex(ValueError, 'cannot read'):
                        package.build(namespace(Path(temp) / 'output', prefix))
                self.assertFalse(started_build(calls))

    def test_release_mode_preconditions(self):
        for override, dirty, message in [
            (dict(notarize=False), False, 'requires --notarize'),
            (dict(allow_dirty=True), False, 'cannot use --allow-dirty'),
            (dict(release='0.119.0'), False, 'must look like'),
            (dict(release='latest'), False, 'must look like'),
            (dict(release='v0.119'), False, 'must look like'),
            ({}, True, 'dirty'),
        ]:
            with self.subTest(override=override, dirty=dirty), tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp)
                values = dict(release='v0.119.0', notarize=True)
                values.update(override)
                calls = []
                with patched(root, fake_tools(calls, dirty=dirty)):
                    with self.assertRaisesRegex(ValueError, message):
                        package.build(namespace(Path(temp) / 'output', prefix, **values))
                self.assertFalse(started_build(calls))

    def test_release_mode_labels_the_artifact_as_the_release(self):
        def build(release, version, notarize):
            with tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp)
                output = Path(temp) / 'output'
                seen = {}

                def inspect(command):
                    stage = Path(command[command.index('-srcfolder') + 1]) / 'mixlab-macos-arm64'
                    seen['volname'] = command[command.index('-volname') + 1]
                    seen['install'] = (stage / 'INSTALL.txt').read_text()
                    seen['receipt'] = json.loads((stage / 'build-receipt.json').read_text())

                with patched(root, fake_tools([], version=version, on_image=inspect)):
                    package.build(namespace(output, prefix, release=release, notarize=notarize))
                seen['images'] = sorted(p.name for p in output.glob('*.dmg'))
                seen['sums'] = (output / 'SHA256SUMS').read_text()
                return seen

        release = build('v0.119.0', 'v0.119.0 (abc1234, 2026-09-28T00:00:00Z)', True)
        self.assertEqual(release['images'], ['mixlab-v0.119.0-macos-arm64.dmg'])
        self.assertIn('mixlab-v0.119.0-macos-arm64.dmg', release['sums'])
        self.assertEqual(release['volname'], 'Mixlab v0.119.0')
        self.assertIn('v0.119.0', release['install'])
        self.assertIn('experimental', release['install'].lower())
        for phrase in ('NOT A RELEASE', 'PRIVATE', 'CANDIDATE'):
            self.assertNotIn(phrase, release['install'].upper() if phrase != 'PRIVATE' else release['install'])
        self.assertEqual(release['receipt']['format'], 'mixlab_macos_release_v1')
        self.assertEqual(release['receipt']['release'], 'v0.119.0')

        # The documented private-candidate flow is unchanged.
        candidate = build(None, 'test-build', False)
        self.assertEqual(candidate['images'], ['mixlab-macos-arm64.dmg'])
        self.assertEqual(candidate['volname'], 'Mixlab Candidate')
        self.assertIn('NOT A RELEASE', candidate['install'])
        self.assertEqual(candidate['receipt']['format'], 'mixlab_macos_candidate_v1')
        self.assertNotIn('release', candidate['receipt'])

    def test_release_refuses_binaries_that_do_not_report_the_tag(self):
        # Building the wrong ref, or a dirty one, must stop before anything is
        # packaged or sent to Apple. v0.119.01 guards against a bare prefix match.
        for version in ('v0.118.0 (abc1234, 2026-09-28T00:00:00Z)',
                        'v0.119.0+dirty (abc1234-dirty, 2026-09-28T00:00:00Z)',
                        'v0.119.01 (abc1234, 2026-09-28T00:00:00Z)',
                        '(devel) (abc1234, 2026-09-28T00:00:00Z)'):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                root, prefix = make_fixture(temp)
                calls = []
                with patched(root, fake_tools(calls, version=version)):
                    with self.assertRaisesRegex(ValueError, 'not release v0.119.0'):
                        package.build(namespace(Path(temp) / 'output', prefix, release='v0.119.0', notarize=True))
                self.assertFalse(any(c[:2] == ['hdiutil', 'create'] for c in calls))
                self.assertFalse(any(c[:3] == ['xcrun', 'notarytool', 'submit'] for c in calls))

    def test_tested_range_is_read_from_the_real_formula_source(self):
        # Pins the packager to the file the formula is actually rendered from.
        low, high = package.tested_mlx_range(package.ROOT)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
