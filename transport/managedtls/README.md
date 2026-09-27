# Normal Managed TLS

This adapter is for authenticated managed control traffic, never provisional
enrollment. It opens no public listener and knows no CA keys, leases, jobs, or
approval policy. A local composition supplies its protected principal signer
and a current-trust verification port with the expected peer identity.

Both peers use full TLS 1.3 handshakes and the normal HTTP/1.1 profile. Default
Web PKI hostname verification is replaced by mandatory pinned-cluster URI
identity verification through the trust port. No system roots, TOFU, session
tickets, verification-disable option, or anonymous client fallback is used.

`AuthenticateHTTP` repeats authentication before each application request,
including keepalive requests after revocation or snapshot expiry. The returned
principal is identity evidence, not authorization; the application owns that
decision. `HTTPClient` uses a new handshake per request and disallows plaintext,
redirect following, and environment proxies, preventing stale connections from
sending a new command before checking the server's current trust. Callers must
bound and strictly decode response bodies according to their own contracts.

The dedicated provisional enrollment profile and stale-snapshot refresh route
must be separate adapters. Do not weaken this profile to implement them. Long
lived collective channels have their own admission and revocation cleanup
rules, not the per-request control policy.
