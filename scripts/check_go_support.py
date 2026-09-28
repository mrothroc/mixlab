#!/usr/bin/env python3
"""Fail when go.mod's toolchain is a Go release that no longer gets security fixes.

Go supports its two newest major releases. mixlab kept building on Go 1.24 for
seven months after support ended, because nothing ever asked: the toolchain was
fine on the day it was chosen and only became a liability later. This runs on
every push and on a weekly schedule, so going out of support fails the build
even when nothing in the repository changes.

The list of supported releases is go.dev's own (https://go.dev/dl/?mode=json,
the newest patch of each supported series). A stale patch in a supported series
only warns: the govulncheck step next to this one fails the build when a missing
patch fixes code mixlab reaches.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

RELEASES_URL = "https://go.dev/dl/?mode=json"


def toolchain_version(go_mod):
    found = re.findall(r"^toolchain go(\d+)\.(\d+)\.(\d+)\s*$", go_mod, re.MULTILINE)
    if len(found) != 1:
        raise ValueError("go.mod must have exactly one 'toolchain goX.Y.Z' line; it names the build toolchain")
    return tuple(int(part) for part in found[0])


def supported_releases(releases):
    """{(major, minor): newest patch} for each supported series."""
    supported = {}
    for release in releases:
        match = re.fullmatch(r"go(\d+)\.(\d+)(?:\.(\d+))?", release.get("version", ""))
        if not release.get("stable") or not match:
            continue
        major, minor, patch = (int(part or 0) for part in match.groups())
        supported[(major, minor)] = max(patch, supported.get((major, minor), 0))
    if not supported:
        raise ValueError("go.dev listed no supported Go releases; refusing to pass on an empty list")
    return supported


def check(version, supported):
    name = "go{}.{}.{}".format(*version)
    series = version[:2]
    listed = ", ".join("go{}.{}.{}".format(*s, p) for s, p in sorted(supported.items(), reverse=True))
    if series not in supported:
        print(f"::error::go.mod builds with {name}, which is not a supported Go release "
              f"(supported: {listed}). Move the toolchain line in go.mod to a supported release.")
        return False
    newest = supported[series]
    if version[2] < newest:
        print(f"::warning::go.mod builds with {name}; go{series[0]}.{series[1]}.{newest} is available.")
    else:
        print(f"{name} is supported (supported: {listed}).")
    return True


def main():
    root = Path(__file__).resolve().parent.parent
    version = toolchain_version((root / "go.mod").read_text())
    with urllib.request.urlopen(RELEASES_URL, timeout=30) as response:
        releases = json.load(response)
    return 0 if check(version, supported_releases(releases)) else 1


if __name__ == "__main__":
    sys.exit(main())
