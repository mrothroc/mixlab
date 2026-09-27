// Package local provides protected, single-child Unix-stream worker control
// on Darwin/Linux. It imports only the worker-control contract and state-home
// safety adapter, never training, trust, or job-policy packages.
//
// Listen uses an existing private per-attempt directory. Pass its returned
// anonymous-pipe read descriptor only to the approved child via ExtraFiles.
// After Start, Accept binds the one-use proof to the actual supervised child
// PID/UID. Connect consumes and closes the inherited descriptor, authenticates
// the agent's kernel PID/UID, and proves the session before any domain message.
// Credentials never enter argv, environment, or files on disk.
//
// Callers MUST keep the child supervised, close Server when that child exits,
// and use a fresh directory/session/attempt on restart. This adapter does not
// launch processes, attest executables, verify assignment signatures, interpret
// domain payloads, or publish streamed artifacts. The launcher owns those tasks.
// Trusted administrator-controlled hosts are assumed; hostile same-UID/root
// processes and passed-descriptor/PID reuse attacks are not an isolation claim.
//
// Every blocking call requires a context deadline. Read/write errors, invalid
// messages, replay, exhausted budgets, and cancellation close the connection.
// Each direction permits one concurrent reader/writer with bounded framing;
// there are no background queues. Budgets exclude the bounded handshake.
package local
