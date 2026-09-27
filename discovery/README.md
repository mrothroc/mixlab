# Untrusted Discovery

`Provider` supplies endpoint hints to application composition, not trust rules.
Authenticate an endpoint before obtaining capabilities or enrolling. Never pin
a root, select a policy, or trust a node ID from a discovery claim. Authority
addresses additionally require the root-signed endpoint allowlist.

Adapters are mDNS/DNS-SD on the local link, explicit configured host:port
addresses with multicast off, and an in-memory deterministic fake. Browses
return at most 256 deduplicated endpoints in stable endpoint order. mDNS
browses last at most 10 seconds. Advertisements stop on context cancellation
or explicit close. TXT has a closed, bounded schema and no arbitrary metadata.
The DNS implementation is pinned in `go.mod`; application code does not parse
wire DNS itself.

`authority serve -trust-advertise mdns` and `enrollment serve
-enrollment-advertise mdns` advertise their concrete listener. The default is
`off`. No OS hostname or unrelated interface inventory is published. Discovery
does not promise to cross VLANs, VPNs, or the internet. Explicit coordinator
addresses continue to use exactly the same enrollment protocol.

The node advertisement schema is present for the managed agent composition;
these adapters alone do not make remote training operational.
