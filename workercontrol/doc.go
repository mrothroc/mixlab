// Package workercontrol provides the Phase 1 mixlab_worker_control_v1 foundation.
// It owns local envelope/framing, per-child admission, and bounded stream
// verification, not domain payload schemas, authorization, or idempotency policy.
// Payload kinds and versions must be checked by their owning context before use.
//
// InspectPeerIdentity obtains OS peer PID/UID from a connected Unix stream on
// Darwin/Linux. The local subpackage supplies private socket provisioning,
// descriptor bootstrap, handshake, deadlines, and bounded bidirectional I/O.
// It binds inspected evidence to the supervised live child and supplies it to
// Session.Authenticate. Message-supplied identity is NOT evidence. The launcher
// must attest the approved binary/build and assignment; this package compares
// their binding but cannot verify executable provenance.
//
// The adapter must deliver the capability via a protected descriptor or owner-
// only file (never argv/environment), protect the connection, enforce deadlines
// and aggregate resource limits, and close the session on attempt termination.
// A restart requires a new attempt and session. Stream verification does not
// publish bytes: callers must stage bytes until Finish succeeds. These pure
// contracts alone do not constitute a protected OS transport implementation.
package workercontrol
