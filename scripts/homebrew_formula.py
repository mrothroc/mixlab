#!/usr/bin/env python3
"""Render the Homebrew formula for one final release.

The formula users install lives in mrothroc/homebrew-tap, but its source is
packaging/homebrew/mixlab.rb in this repository. The release tag and its commit
are the only values that change per release, so they are placeholders there.
Rendering fails rather than producing a formula with a missing, leftover, or
malformed value. Prereleases are refused: they are never published to Homebrew.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("packaging/homebrew/mixlab.rb")
PLACEHOLDERS = ("@RELEASE_TAG@", "@RELEASE_REVISION@")
FINAL_TAG = re.compile(r"v\d+\.\d+\.\d+")
REVISION = re.compile(r"[0-9a-f]{40}")


def render(template, tag, revision):
    if not FINAL_TAG.fullmatch(tag):
        raise ValueError(f"tag {tag!r} is not a final release (vX.Y.Z); prereleases are not published to Homebrew")
    if not REVISION.fullmatch(revision):
        raise ValueError(f"revision {revision!r} is not a full 40-character lowercase commit SHA")
    for placeholder in PLACEHOLDERS:
        if placeholder not in template:
            raise ValueError(f"template is missing {placeholder}")
    rendered = template.replace("@RELEASE_TAG@", tag).replace("@RELEASE_REVISION@", revision)
    leftover = sorted(set(re.findall(r"@[A-Z_]+@", rendered)))
    if leftover:
        raise ValueError(f"template has unknown placeholders: {leftover}")
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="final release tag, vX.Y.Z")
    parser.add_argument("--revision", required=True, help="full commit SHA the tag points at")
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        text = render(args.source.read_text(), args.tag, args.revision)
    except (ValueError, OSError) as error:
        parser.exit(1, f"homebrew-formula: {error}\n")
    args.output.write_text(text)


if __name__ == "__main__":
    main()
