# Local Bootstrap Boundary

This package composes protected key lifecycle, certificate issuance, and
principal publication for explicit local initialization. It does not enroll
remote peers, grant leases, listen on a socket, or import trainer/GPU code.

The authority journal freezes IDs, paths, backend, key handles, issued DER,
signed endpoints, and the initial snapshot. Principal stages contain their own
key lifecycle and public trust; durable directory promotion never copies key
material or replaces a destination. An uncertain promotion is reconciled only
against the exact expected credential record and protected signing handle.

Journal failure tests exercise every bootstrap commit before/after publication.
Missing previously-started key material fails closed. Recovery verifies expired
initial snapshots as history, never as fresh operational authorization. Serving
and renewal remain separate work; this is not the full enrollment state machine.
