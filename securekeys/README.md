# Protected Signing Keys

Internal R1.1 storage adapter, not an enrollment service. `securekeys` generates
and stores independent Ed25519 signing and X25519 envelope keys behind public,
scoped handles. It does not issue certificates or decide who
may sign a job. Trust's purpose-restricted signer remains the policy boundary.

## Backends

- `OpenKeychain` uses macOS Security.framework with cgo. Each context has an
  exact service namespace and each generated key a random account identifier.
  Queries target only the selected default Keychain, not the global search list.
  Authentication UI is disabled; a locked or unavailable Keychain returns an
  error. Missing items never trigger key regeneration or backend fallback.
- `OpenFile` uses an existing `statehome.Path`. Owner-only files contain the
  private seed; they are **not encrypted at rest** by Mixlab. The directory must
  be mode 0700 and files mode 0600, with ownership, symlink, hard-link, ACL, and
  bounded-read checks. Publication cannot overwrite an existing key.
- macOS compositions should prefer Keychain for new identities. File storage is
  an explicit fallback choice, not an automatic response to authentication or
  read errors. Linux uses the file adapter without additional services. Native
  Keychain access is unavailable in CGO-disabled builds; callers get a clear
  error rather than an implicit file store.

`OpenSelected` centralizes those defaults for **new** contexts. Reopened
contexts must pass their persisted backend explicitly. Unknown backend names
fail, and a Keychain error never triggers a file fallback.

Handles contain version, backend, context scope, random ID, and public key only.
Scope is a composition-owned 256-bit context identifier encoded as lowercase
hex; obtain it from local context identity, never a remote request. Opening a
handle checks its backend/scope and the public key derived from its stored seed.
Every signature re-reads the protected record, so deletion invalidates already
opened signer handles. `Close` releases native Keychain references.

Keychain seeds are generic-password items: this does not claim Secure Enclave
or hardware-nonexportable Ed25519 keys. Private bytes exist transiently in the
adapter process for runtime Ed25519 operations, never in public handles or
argv/environment. Buffers are cleared where practical; Go/runtime allocations
do not provide a guaranteed forensic memory-erasure contract.

The adapter uses Apple's [SecItem API](https://developer.apple.com/documentation/security/secitemadd(_:_:))
with an explicit [destination Keychain](https://developer.apple.com/documentation/security/ksecusekeychain).
See Apple's [macOS Keychain implementation guidance](https://developer.apple.com/documentation/technotes/tn3137-on-mac-keychains).

## Lifecycle And Failure

`Generate` returns its candidate handle even if publication reports an error.
A file may have been renamed before its durability check failed. The owning
workflow must reconcile the exact handle, not blindly retry and lose track of
an orphan key. Durable owners use `PrepareSigning`/`PrepareEnvelope`, persist
`Pending.Handle()` first, then `Publish` and always `Close`. Preparing touches no
backend. Closing clears the in-memory serialized seed. Retrying a live pending
publication accepts only the exact same stored key. `Inspect` checks an existing
handle without exposing private bytes. See [key lifecycle](../trust/keylifecycle/README.md)
for durable intent/recovery/rotation; enrollment must wire that state machine
into approved issuance. Deleting a key is not certificate revocation.

The supported host model is trusted, administrator-controlled machines. These
adapters reject unsafe paths and corrupt records, but do not protect against
the administrator or another process with equivalent user authority rewriting
state. No backup, rotation, automatic migration, or recovery policy is implied.

## Verification

```bash
go test ./securekeys ./statehome ./trust/... -count=1
go test -race ./securekeys ./statehome ./trust/... -count=1
go test -tags keychaintest ./securekeys -count=1 -timeout=60s
```

The opt-in native test creates a disposable Keychain with a generated password,
does not open the login Keychain or change the default, tests locked-store
rejection, and deletes the test Keychain. Tests without that tag use file storage
and a fake Keychain backend. No test accesses user signing certificates.
