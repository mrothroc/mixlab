"""Render a signature-preserving cask from a tagged DMG checksum (no publishing)."""
import argparse
from pathlib import Path
import re


def render(tag, checksum):
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?", tag):
        raise ValueError("expected release tag vX.Y.Z")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("expected exact SHA256 of notarized, stapled DMG")
    return f'''cask "mixlab-signed" do
  version "{tag[1:]}"
  sha256 "{checksum}"

  url "https://github.com/mrothroc/mixlab/releases/download/v#{{version}}/mixlab-v#{{version}}-macos-arm64.dmg"
  name "Mixlab"
  desc "Signed training and managed cluster executables"
  homepage "https://github.com/mrothroc/mixlab"

  depends_on arch: :arm64
  depends_on macos: :tahoe

  suite "mixlab-macos-arm64", target: "Mixlab"
  binary "#{{appdir}}/Mixlab/mixlab"
  binary "#{{appdir}}/Mixlab/mixlab-cluster"

  caveats <<~EOS
    Unlink the source-built mixlab formula before installing this cask.
    Enroll first, then use mixlab-cluster authority install and agent install.
    Services require a logged-in GUI session and initial Local Network approval.
    Before upgrading or uninstalling: finish jobs, then stop agent and authority
    services. After upgrading: agent reapprove -worker-binary "$(command -v mixlab)"
    on each node, then start services. Identities and datasets are preserved.
    Uninstall service registrations explicitly before removing this cask.
  EOS
end
'''


def from_checksums(tag, text):
    name = f"mixlab-{tag}-macos-arm64.dmg"
    matches = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            matches.append(parts[0])
    if len(matches) != 1:
        raise ValueError("checksum file must contain exactly one matching release DMG")
    return render(tag, matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = from_checksums(args.release, args.checksums.read_text())
    with args.output.open("x") as output:
        output.write(content)


if __name__ == "__main__":
    main()
