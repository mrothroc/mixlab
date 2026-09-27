# Dedicated Enrollment Transport

Internal adapter for full TLS 1.3 enrollment connections. It uses a dedicated
ALPN profile, disables session resumption/tickets, observes the peer and local
destination address/interface, and derives a one-use 32-byte exporter for one
request. The interface owns the destination IP; it does not prove physical
packet ingress on a multihomed host. Trusted-LAN policy is an explicit local
address plus private-source-CIDR restriction on administrator-controlled hosts,
not a routing, tunnel or packet-filter security boundary.

The caller owns root/policy acceptance. Normal authenticated operations belong
to `transport/managedtls`; enrollment connections cannot carry those operations.
Neither adapter approves enrollment, loads private state, or authorizes jobs.

The HTTP helper operates only on an already completed dedicated TLS connection.
It has bounded headers, request/idle timeouts, no proxy/redirect/reconnect behavior,
and an exact origin. Handlers own body limits and the bootstrap-only route list.
`ServeConnection` closes live evidence on exit; composition must also call the
trust service's disconnect port to durably expire unfinished requests.
