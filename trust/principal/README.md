# Installed Principal State

Owns principal credential storage, protected-handle binding, trust refresh and
same-key renewal publication. It accepts the existing initial bootstrap record
and the general principal record used by enrolled nodes/controllers/coordinators.

`View` permits inspection of expired credentials/stale signed history; `Active`
requires fresh eligible trust and a live protected signing handle. Neither path
creates missing keys, replaces a root, or changes a backend. Node identities also
require the exact active X25519 handle certified by their leaf.

`Install` writes validated state in an existing private enrollment staging
context. The enrollee workflow atomically promotes that complete context to an
absent destination. `Renew` preserves identity and key material and rejects
expired/revoked old identities. `Refresh` only advances signed trust.
