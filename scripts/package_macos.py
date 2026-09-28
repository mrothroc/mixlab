#!/usr/bin/env python3
"""Build a relocatable, signed macOS distribution of mixlab and mixlab-cluster.

By default this builds a private acceptance candidate and notarization is
opt-in. --release vX.Y.Z builds the tagged release instead: it must be
notarized, built from a clean tree, and the binaries must report that version.
Either way this never publishes a release or changes the source Homebrew
installation. Managed cluster operation is experimental and limited to
trusted, administrator-controlled hosts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")
LIBRARIES = ("libmlx.dylib", "libjaccl.dylib")
RELEASE_VERSION = re.compile(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?")


def run(*args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, text=True,
                          stdout=subprocess.PIPE, **kwargs).stdout.strip()


def dependencies(output):
    result = []
    for line in output.splitlines()[1:]:
        if " (compatibility version " not in line:
            raise ValueError(f"unrecognized otool dependency: {line}")
        result.append(line.strip().split(" (compatibility version ", 1)[0])
    return result


def rpaths(output):
    return re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset \d+\)", output)


def local_dependency(dep, packaged):
    if dep.startswith(SYSTEM_PREFIXES):
        return dep
    name = Path(dep).name
    if name not in packaged:
        raise ValueError(f"unbundled non-system dependency: {dep}")
    return "@loader_path/" + name


def relocate(path, packaged):
    # Dylib IDs also appear in `otool -L`; normalize before walking imports.
    if path.suffix == ".dylib":
        run("install_name_tool", "-id", "@loader_path/" + path.name, path)
    for dep in dependencies(run("otool", "-L", path)):
        replacement = local_dependency(dep, packaged)
        if replacement != dep:
            run("install_name_tool", "-change", dep, replacement, path)
    for value in set(rpaths(run("otool", "-l", path))):
        run("install_name_tool", "-delete_rpath", value, path)
    audit(path, packaged)


def audit(path, packaged):
    for dep in dependencies(run("otool", "-L", path)):
        if dep != local_dependency(dep, packaged):
            raise ValueError(f"non-relocatable dependency in {path}: {dep}")
    if rpaths(run("otool", "-l", path)):
        raise ValueError(f"unexpected library search path in {path}")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sign(path, identity, identifier=None):
    args = ["codesign", "--force", "--sign", identity, "--timestamp", "--options", "runtime"]
    if identifier:
        args.extend(["--identifier", identifier])
    run(*args, path)
    run("codesign", "--verify", "--strict", "--verbose=2", path)


def matched_identity(trainer, cluster):
    left, right = trainer.splitlines(), cluster.splitlines()
    if len(left) < 2 or len(right) < 2:
        raise ValueError("missing version/protocol identity")
    if left[0].removeprefix("mixlab ") != right[0].removeprefix("mixlab-cluster "):
        raise ValueError("trainer and cluster build identities differ")
    if left[1] != right[1] or not left[1].startswith("worker_protocol "):
        raise ValueError("trainer and cluster worker protocols differ")


def mlx_version(prefix):
    header = prefix / "include/mlx/version.h"
    if not header.is_file():
        raise ValueError(f"cannot read MLX version: missing {header}")
    text = header.read_text()
    parts = []
    for name in ("MAJOR", "MINOR", "PATCH"):
        found = re.search(rf"^#define MLX_VERSION_{name} (\d+)\s*$", text, re.MULTILINE)
        if not found:
            raise ValueError(f"cannot read MLX_VERSION_{name} from {header}")
        parts.append(int(found.group(1)))
    return tuple(parts)


def tested_mlx_range(root):
    """The formula is the single record of which MLX releases were tested."""
    formula = root / "Formula" / "mixlab.rb"
    text = formula.read_text() if formula.is_file() else ""
    bounds = []
    for name in ("MLX_TESTED_MINIMUM", "MLX_TESTED_BELOW"):
        found = re.search(rf'^\s*{name}\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, re.MULTILINE)
        if not found:
            raise ValueError(f"cannot read {name} from {formula}")
        bounds.append(tuple(int(x) for x in found.groups()))
    return bounds[0], bounds[1]


def check_mlx(prefix, root):
    version = mlx_version(prefix)
    low, high = tested_mlx_range(root)
    dotted = lambda v: ".".join(map(str, v))
    if not low <= version < high:
        raise ValueError(f"MLX {dotted(version)} is outside the tested range >={dotted(low)} <{dotted(high)} "
                         "recorded in Formula/mixlab.rb; run the -tags mlx suite before widening it")
    return dotted(version)


def check_release_version(release, trainer):
    first = trainer.splitlines()[0]
    if first != f"mixlab {release}" and not first.startswith(f"mixlab {release} ("):
        raise ValueError(f"built trainer reports {first!r}, not release {release}; build the tagged, clean commit")


def build(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this candidate packager currently requires Apple Silicon macOS")
    if args.notarize and not args.notary_profile:
        raise ValueError("--notarize requires --notary-profile (Keychain name, not a password)")
    if not args.identity.startswith("Developer ID Application:"):
        raise ValueError("--identity must be a Developer ID Application identity")
    release = args.release
    if release is not None:
        if not RELEASE_VERSION.fullmatch(release):
            raise ValueError("--release must look like vX.Y.Z or vX.Y.Z-suffix")
        if not args.notarize:
            raise ValueError("--release requires --notarize; a release is never shipped unnotarized")
        if args.allow_dirty:
            raise ValueError("--release cannot use --allow-dirty; a release is built from its clean tag")
    dirty = bool(run("git", "status", "--porcelain", cwd=ROOT))
    if dirty and not args.allow_dirty:
        raise ValueError("working tree is dirty; use --allow-dirty only for private acceptance candidates")
    output = args.output.expanduser().absolute()
    if output.resolve().is_relative_to(ROOT):
        raise ValueError("output must be outside the source repository")
    if output.exists():
        raise ValueError("output directory must not already exist")
    prefix = args.mlx_prefix.resolve()
    for file in ("include/mlx/mlx.h", "lib/libmlx.dylib", "lib/mlx.metallib", "LICENSE"):
        if not (prefix / file).is_file():
            raise ValueError(f"missing MLX build input: {prefix / file}")
    bundled_mlx = check_mlx(prefix, ROOT)
    output.mkdir(parents=True, mode=0o700)
    # Only our private staging area is automatically deleted. Failed notary
    # receipts/artifacts remain in the explicit output for diagnosis.
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=output) as temporary:
        stage = Path(temporary) / "mixlab-macos-arm64"
        stage.mkdir()
        env = os.environ.copy()
        env["CGO_ENABLED"] = "1"
        env["CGO_CFLAGS"] = f"-I{prefix}/include " + env.get("CGO_CFLAGS", "")
        env["CGO_CXXFLAGS"] = f"-I{prefix}/include " + env.get("CGO_CXXFLAGS", "")
        env["CGO_LDFLAGS"] = f"-L{prefix}/lib " + env.get("CGO_LDFLAGS", "")
        print("Building matched trainer and cluster candidate", flush=True)
        run("go", "build", "-trimpath", "-tags", "mlx", "-o", stage / "mixlab", "./cmd/mixlab", cwd=ROOT, env=env)
        env["CGO_ENABLED"] = "0"
        run("go", "build", "-trimpath", "-o", stage / "mixlab-cluster", "./cmd/mixlab-cluster", cwd=ROOT, env=env)
        packaged = {name for name in LIBRARIES if (prefix / "lib" / name).is_file()}
        for name in sorted(packaged) + ["mlx.metallib"]:
            shutil.copyfile(prefix / "lib" / name, stage / name)
        shutil.copyfile(ROOT / "LICENSE", stage / "LICENSE-mixlab")
        shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", stage / "THIRD_PARTY_NOTICES.md")
        shutil.copyfile(prefix / "LICENSE", stage / "LICENSE-mlx")
        receipt = {"format": "mixlab_macos_release_v1" if release else "mixlab_macos_candidate_v1",
                   "revision": run("git", "rev-parse", "HEAD", cwd=ROOT),
                   "dirty": dirty, "architecture": "arm64", "experimental_managed_cluster": True,
                   "mlx_version": bundled_mlx, "mlx_source_sha256": digest(prefix / "lib/libmlx.dylib")}
        if release:
            receipt["release"] = release
        for name in sorted(packaged) + ["mixlab", "mixlab-cluster"]:
            path = stage / name
            path.chmod(0o755)
            relocate(path, packaged)
            sign(path, args.identity, "com.mixlab." + name.replace(".dylib", ""))
        trainer = run(stage / "mixlab", "-version")
        cluster = run(stage / "mixlab-cluster", "-version")
        matched_identity(trainer, cluster)
        if release:
            check_release_version(release, trainer)
        receipt["trainer_version"] = trainer
        receipt["cluster_version"] = cluster
        receipt["files"] = {p.name: digest(p) for p in sorted(stage.iterdir())}
        (stage / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if release:
            install = (f"mixlab {release} for macOS on Apple Silicon.\n"
                       "Copy this entire directory to a stable installation path.\n")
        else:
            install = ("PRIVATE R1.1 ACCEPTANCE CANDIDATE; NOT A RELEASE.\n"
                       "Copy this entire directory to a private, stable installation path.\n")
        (stage / "INSTALL.txt").write_text(
            install +
            "Keep the executables, dylibs and mlx.metallib together. Do not re-sign or strip them.\n"
            "mixlab-cluster provides experimental managed training on trusted hosts.\n"
            "Signing/notarization is not cluster authentication or a firewall exemption.\n"
            "R1.1 assumes trusted, administrator-controlled hosts.\n")
        image = output / (f"mixlab-{release}-macos-arm64.dmg" if release else "mixlab-macos-arm64.dmg")
        run("hdiutil", "create", "-srcfolder", stage.parent, "-volname",
            f"Mixlab {release}" if release else "Mixlab Candidate", "-format", "UDZO", image)
        sign(image, args.identity)
    if args.notarize:
        print("Submitting candidate disk image to Apple for notarization", flush=True)
        result = json.loads(run("xcrun", "notarytool", "submit", image, "--keychain-profile",
                                args.notary_profile, "--wait", "--timeout", "20m", "--output-format", "json"))
        (output / "notarization.json").write_text(json.dumps(result, indent=2) + "\n")
        if result.get("status") != "Accepted":
            raise ValueError(f"notarization not accepted; see {output / 'notarization.json'}")
        run("xcrun", "stapler", "staple", image)
        run("xcrun", "stapler", "validate", image)
        run("spctl", "--assess", "--type", "open", "--context", "context:primary-signature", "--verbose=2", image)
    (output / "SHA256SUMS").write_text(f"{digest(image)}  {image.name}\n")
    print(image, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--mlx-prefix", type=Path, default=Path("/opt/homebrew/opt/mlx"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--notarize", action="store_true")
    parser.add_argument("--notary-profile")
    parser.add_argument("--release", metavar="vX.Y.Z",
                        help="build this tagged release instead of a private candidate; "
                             "requires --notarize and a clean tree")
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"package-macos: {error}\n")


if __name__ == "__main__":
    main()
