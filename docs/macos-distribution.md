# macOS Distribution Candidates

`scripts/package_macos.py` builds two kinds of disk image. By default it builds a
private **acceptance candidate**, the maintainer workflow described below. With
`--release vX.Y.Z`, the Release workflow builds the **tagged release** for each `v*`
tag (see [Releasing](releasing.md)). Tagged images are not yet an announced install
channel: the Homebrew cask that will point at them does not exist, and the
acceptance checks below still apply. `mixlab-cluster` implements experimental enrollment,
node hosting and fixed-cohort submission, including managed checkpoint/resume.
The R1.1 candidate passed signed-package M1/M4 acceptance. Each published build
still requires release approval; signing alone is not functional or security
acceptance.

## Build And Notarize

The candidate packager requires Apple Silicon macOS, Xcode command-line tools,
Go, the supported MLX installation, a Developer ID Application identity with its
private key in Keychain, and a notarization Keychain profile. Never put passwords
or private signing keys in repository files, command arguments, or logs.

```bash
python3 scripts/package_macos.py \
  --output /tmp/mixlab-macos-candidate \
  --identity 'Developer ID Application: Your Name (TEAMID)' \
  --notarize --notary-profile mixlab-notary
```

Every build, candidate or release, refuses an MLX outside the range recorded in
`packaging/homebrew/mixlab.rb` (`MLX_TESTED_MINIMUM` up to but excluding `MLX_TESTED_BELOW`). The
version is read from the MLX headers being bundled. The formula stays the one record
of which MLX was tested, so a Homebrew MLX bump fails the build before anything is
signed. It also fails loudly if either the header or the formula cannot be read.

The output directory must not exist and must be outside the source repository.
Dirty source is rejected unless
`--allow-dirty` is explicitly supplied for a private test build. Omitting
`--notarize` produces a locally signed candidate without submitting it to Apple.
The script never creates a release, pushes a tag, installs software, or changes
the firewall. Its staging directory is removed automatically; the explicit
output directory remains for inspection and must be removed after acceptance.

Both executables are built from the same working tree and their build/protocol
identities must match. The package bundles MLX libraries, `mlx.metallib`, and
Mixlab/MLX licenses. Library imports are made relative to the installed files;
unexpected external dependencies and residual library search paths are errors.
Load paths are finalized before signing. Each Mach-O executable and dylib is
signed individually with hardened runtime and a secure timestamp, without
disabling library validation. The `mlx.metallib` Metal resource is included in
the hash receipt and signed disk image; it is not a Mach-O dylib. The disk
image is signed, optionally notarized, stapled,
and assessed. A receipt records input provenance and signed file hashes; the
outer image contains that receipt, and `SHA256SUMS` hashes the final image.

This workflow uses a DMG so the notarization ticket can be stapled without a
Developer ID Installer certificate. A future signed PKG would require that
additional identity. Apple Distribution and Mac Installer Distribution are
not substitutes for the Developer ID identities used outside the App Store.

## Release Builds

`--release vX.Y.Z` differs from a candidate build in five ways:

- It requires `--notarize`. A release is never shipped unnotarized.
- It refuses `--allow-dirty` and a dirty tree.
- Both binaries must report exactly `vX.Y.Z` from `-version`, so building the wrong
  commit, or a modified one, fails before anything is packaged or sent to Apple.
- The image is `mixlab-vX.Y.Z-macos-arm64.dmg` with volume name `Mixlab vX.Y.Z`, and
  its `INSTALL.txt` describes a release rather than a private candidate.
- The receipt records format `mixlab_macos_release_v1`, the release version, and the
  bundled MLX version.

In CI the certificate lives in a temporary keychain. `codesign --keychain` alone does
not find an identity there, so the workflow adds that keychain to the search list and
makes it the default, then checks the identity's SHA-1 against
`MACOS_SIGNING_IDENTITY_SHA1` before signing anything. After notarization it copies the
image contents out the way a user would, uninstalls Homebrew MLX, and runs `-version`
and `-mode smoke` on Metal. That proves the packaged binaries run only on the libraries
they carry.

## Acceptance

Copy the **entire** `mixlab-macos-arm64` directory from the mounted image to a
stable, administrator-controlled location. Keep its binaries and resources
together. Do not strip, patch, or re-sign installed files. The existing Homebrew
formula builds from source and is not the signature-preserving distribution
channel for these candidates.

Before treating this as shippable, verify on another Mac:

- Gatekeeper assessment of the quarantined image, notarization verification of
  copied code (`codesign --verify --strict --check-notarization`), and actual
  execution with quarantine retained. `spctl --type execute` may reject a bare
  CLI as "does not seem to be an app"; it is not an app-bundle acceptance test.
- Signature and receipt hash preservation after copying the directory.
- Real Metal training with the packaged MLX library/resources, not only
  `-version`, and no dependence on Homebrew library search paths.
- Incoming connections to the real agent with the Application Firewall enabled
  and its actual policy recorded.
- Private installation permissions and complete process/artifact cleanup.

Notarization/Gatekeeper approval and Application Firewall policy are separate.
On first launch, macOS may require approval even when signature and notarization
checks pass. Use the Mac's local desktop or Screen Sharing to approve the
specific executable in System Settings > Privacy & Security, then confirm the
follow-up Open dialog. Allow Anyway alone may not complete the launch. Keep
quarantine intact during acceptance; removing it would bypass the distribution
check. A headless SSH session is not a substitute for this interactive approval.

Developer ID signing does not override administrator policy, block-all mode,
or a specific deny rule. Do not instruct users to disable the firewall; use a
scoped administrator-approved exception when policy requires it. Signing also
does not replace cluster enrollment, peer TLS authentication, or authorization.
R1.1 assumes trusted, administrator-controlled hosts; hostile local processes
are outside its threat model.

See [Apple's notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
and [firewall security](https://support.apple.com/guide/security/firewall-security-in-macos-seca0e83763f/web).
