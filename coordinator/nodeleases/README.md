# Coordinator Allocation Keeper

`Keeper` implements the coordinator's allocation-readiness port and a bounded
periodic renewal/cleanup step. Its node client uses coordinator-owned identity,
not the submitting terminal's credentials. Every response is checked against
the original slot, job, lease, run, coordinator and commit digest.

Readiness requires completed authority handoff on every original allocation.
Expired or stopped allocations fail the fixed run; the adapter has no recruit
or replacement operation. Renewal keys bind the lease version, and a lost
response is resolved by rereading state. Terminal runs request release but are
not reported released until node management confirms physical cleanup.

The service composition must call `Tick` on a bounded cadence while the run is
active or cleaning up, retry transient transport failures, and surface persistent
failure. Merely constructing this adapter does not install a background service.
