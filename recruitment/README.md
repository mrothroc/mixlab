# Authenticated Cohort Selection

Selection consumes addresses from discovery, never its identity or capability
claims. Each address is queried through a fresh pinned-cluster TLS connection;
the response is bound to the authenticated node identity and current trust.
The pure selector additionally checks freshness, worker build/MLX/device/dtype/
custom-op compatibility, resource policy, local dataset identity and availability.

One bounded scan yields either the complete requested cohort or no cohort.
Ranks are assigned by authenticated node ID. Duplicate addresses cannot add
votes. Conflicting node observations or an authenticated negative sibling
exclude the node. Probe time/free-memory fluctuations are not identity changes,
but every address must independently satisfy freshness and resource checks.
At commit, the first still-live address in deterministic endpoint order wins.

Selection is an observation, not a reservation. Prepare must recheck executable,
dataset, resources and the exclusive lease. Durable all-or-abort recruitment and
the public managed-submit workflow remain separate integration work.
