# Credential Encryption Adapter

This adapter pins CIRCL v1.6.4's RFC 9180 HPKE base mode to
DHKEM(X25519, HKDF-SHA256), HKDF-SHA256 and ChaCha20Poly1305. It provides
encryption, not sender authentication. The future credential-envelope owner
must verify a signed envelope, exact recipient/job/attempt/audience binding,
expiry and replay state before releasing plaintext to an authorized use port.
No enrollment or application authorization is implemented by this package.

The known-answer test uses the first encryption from mode 0, KEM 32, KDF 1,
AEAD 3 in CIRCL's `hpke/testdata/vectors_rfc9180_5f503c5.json.gz`. It verifies
encapsulation and ciphertext exactly, not just a same-library round trip.
Additional tests reject tampered context, associated data, keys, ciphertext,
noncanonical and low-order public keys, missing entropy and oversized inputs.

X25519 keys are generated independently from Ed25519 identity keys. The
`securekeys` adapter owns persistence and never exports private key material.
Clearing temporary byte slices is best effort: Go and the crypto library may
retain internal copies until garbage collection. This is not a memory-erasure
or hostile-local-process security boundary.
