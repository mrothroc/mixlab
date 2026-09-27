# Protected Local Worker Transport

Internal R1.1 foundation, not a public managed-training entrypoint. The existing
trainer does not yet use this adapter and `mixlab-cluster` remains a scaffold.
No remote service or listener starts implicitly.

## Ownership And Lifecycle

1. Worker hosting approves the executable/build and signed assignment, creates
   a fresh per-attempt runtime directory, and validates it through `statehome`.
2. `Listen` creates `control.sock` inside that owner-only directory and an
   anonymous pipe. It returns the pipe's read descriptor for `exec.Cmd.ExtraFiles`.
   Only the socket path and descriptor number may appear in launch arguments;
   credentials must never appear in argv/environment or persistent files.
3. After `Start`, hosting supplies the actual supervised child PID/UID and
   immutable `workercontrol.Binding` to `Server.Accept`. Admission generates a
   CSPRNG one-use capability and sends bootstrap metadata/proof through the
   pipe. The parent's duplicate descriptor is closed by the server.
4. Before starting training or other children, the worker calls `Connect` with
   its inherited descriptor. This consumes/closes the descriptor, verifies the
   agent's kernel PID/UID, and sends its proof over the protected Unix stream.
5. The agent verifies the child through kernel evidence, checks its complete
   binding and proof, and acknowledges admission. No domain message is accepted
   before this handshake. Failed admission consumes the session and closes it;
   there is no reconnect or weaker-authentication fallback.
6. Both sides use `Conn.Send`/`Receive`. The owning contexts validate message
   payload kinds, versions, semantics and idempotency. Sequence numbers are
   caller-supplied, positive and strictly increasing independently in each
   direction. Stream recipients stage artifacts until stream verification ends.
7. Hosting closes the server when the child exits or admission fails, reaps the
   child, then removes its runtime directory. `Server.Close` unblocks I/O and
   removes only its own socket inode. It does not kill/reap processes or delete
   the caller's directory. A restart uses a new attempt and new session.

The launcher must not release/reuse the supervised process identity while
admission is pending. OS peer credentials identify the process that established
the connection, not a later recipient of a passed socket. R1.1 assumes trusted,
administrator-controlled hosts, not isolation from hostile root/same-UID users.
This adapter authenticates a local connection; it does not verify executable
signatures, cluster certificates, job approval, or training assignments.

## Wire And Limits

The handshake metadata uses a `mixlab_worker_control_v1` readiness envelope,
payload kind `worker_session_v1`, payload version 1, sequence 1 and correlation
ID `session`. The metadata is canonical JSON containing `binding` and `agent`;
noncanonical/unknown/case-alias fields are rejected. This frame is capped at
4096 bytes. Exactly 32 raw capability bytes follow it. Descriptor bootstrap
must then reach EOF. Proof bytes are not JSON/base64 encoded or logged, and
temporary proof buffers are cleared after use. Go does not promise forensic
erasure of all runtime memory copies.

The accepted socket replies with `mixlab_worker_ready_v1` plus newline. After
admission, each direction begins its own ordinary envelope sequence; the
handshake sequence is not part of the application sequence.

Every operation requires a live context deadline. There is no unbounded queue;
Unix-stream backpressure applies. Owners supply frame, total wire-byte and
message budgets independently enforced in both directions. These exclude the
bounded handshake. Framing errors, invalid bindings, replay, budget exhaustion,
timeouts, and cancellation close the connection. `Close` is idempotent.

Darwin and Linux use kernel Unix-socket PID/UID inspection. Inherited blocking
pipe descriptors are duplicated as close-on-exec and registered with Go's
poller so cancellation and read deadlines work after exec.

## Tests

```bash
go test -race ./workercontrol/... -count=3
go test ./workercontrol/local -run '^$' -fuzz '^FuzzHello$' -fuzztime=10s
```

Native tests launch real children with descriptor-delivered credentials, test
both identities, reject wrong proofs/builds/replay, and exercise startup death,
backpressure, deadlines, cancellation, bounded framing, and socket cleanup.
Run outside a sandbox that prohibits Unix socket binding. CI includes these
packages on Linux; cross-compilation alone is not native Linux acceptance.
