# Distributed Training Requirements And Release Specification

Status: In progress

Audience: implementation agents and maintainers

## Purpose

Extend mixlab from a single-process trainer into a distributed training system
that can use:

- a heterogeneous pool of Apple-silicon Macs connected by an ordinary
  gigabit LAN; and
- one or more CUDA workers rented from RunPod and connected over the public
  internet.

The intended end state is hierarchical:

```text
durable global coordinator and checkpoint
                 |
        infrequent outer updates
                 |
        +--------+---------+
        |                  |
  Mac learner pool    CUDA learner island
  MLX on Metal        MLX on CUDA
  one Mac/learner     optional NCCL DDP within island
```

Per-step communication is allowed within a tightly connected learner group.
Communication between learner islands is infrequent and exchanges parameter
deltas, not per-step gradients.

On a school LAN, background node daemons advertise availability and accept
authenticated structured jobs. Discovery selects a cohort before launch; it
does not change DDP or DiLoCo membership while a training attempt is running.

This document defines stable requirements, architectural boundaries, public
surface proposals, release targets, and acceptance gates. It is deliberately
implementation-oriented. An agent implementing a release must satisfy the
requirements and tests assigned to that release and stop at its review gate.

## R0 Prerequisite

The MLX upgrade and initial benchmark work are outside this specification.
They may proceed independently without a design review.

R1 assumes:

- mixlab is pinned to an MLX release that provides the current C++ distributed
  API, TCP ring on Metal, and NCCL on CUDA;
- existing Metal and CUDA training paths pass their required parity and smoke
  tests after that upgrade; and
- custom primitive regressions caused by the upgrade have been resolved.

R1 must fail at build or startup with an actionable error if the selected MLX
build lacks its required distributed backend. It must not silently fall back
to single-process training.

## Release Targets

| Target | Deliverable | Release claim |
|--------|-------------|---------------|
| R0 | Pin and validate a current MLX distributed-capable build. | Independent prerequisite; no design phase required. |
| R1 | Fixed-world DDP over MLX ring on Metal and NCCL on CUDA. | Correct same-backend data parallelism and the shared distributed trainer foundation. No general gigabit-LAN speedup claim. |
| R1.1 | Background node daemon, self-managed cluster trust, trusted-LAN/provisioned/verified enrollment policies, untrusted LAN discovery followed by authenticated capability lookup, leases, encrypted collective transport, and remote fixed-cohort launch. | Idle lab hosts can be enrolled at the assurance level appropriate to the environment and recruited without SSH or externally managed CA/VPN infrastructure. Discovery is dynamic; membership within each launched training attempt remains fixed. |
| R2 | Fixed-cohort synchronous DiLoCo among single-process Mac learners. | First release eligible to claim that idle Macs on gigabit LAN accelerate a training target. |
| R3 | One fixed DiLoCo cohort containing Metal and RunPod CUDA learners. | Mixed-backend training without a cross-backend or WAN collective. |
| R4 | An NCCL DDP group acting as one CUDA DiLoCo learner. | Hierarchical scaling inside a CUDA island. |
| R5 | Elastic or asynchronous learners based on a separately approved design. | Experimental tolerance for changing and unequal-speed capacity. |

The core dependency path is R0 -> R1 -> R1.1 -> R2 -> R3. R1.1 may be
implemented in parallel with late R1 hardware validation after the immutable
membership contract is fixed. R4 extends R3 for multi-GPU CUDA capacity. R5
may begin only after R3 recovery and convergence evidence exists; it does not
block R4.

### R1 implementation status

R1 Phase 1, the internal DDP walking skeleton, is implemented. It provides:

- immutable ordered membership and local launch identity;
- an opaque, strictly initialized MLX group runtime with startup identity
  agreement;
- a C++-owned numerator/denominator optimizer-stage boundary;
- causal loss-normalizer metadata and zero-denominator skipping;
- distributed-mode submit/collect ordering; and
- singleton and two-rank ring startup verification.

This phase does not expose a public `mode=ddp` and does not yet perform
multi-rank gradient reduction. The ordered collective transaction, gradient
bucketing, accumulation, rank-disjoint sampling, and two-rank optimizer parity
remain gated to the subsequent R1 releases.

## References

The algorithm and backend contracts are based on:

- [MLX distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [MLX data parallelism](https://ml-explore.github.io/mlx/build/html/examples/data_parallelism.html)
- [DiLoCo](https://arxiv.org/abs/2311.08105)
- [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428)
- [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [RFC 6762: Multicast DNS](https://www.rfc-editor.org/rfc/rfc6762)
- [RFC 6763: DNS-Based Service Discovery](https://www.rfc-editor.org/rfc/rfc6763)
- [RFC 8446: TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446)
- [RFC 9266: TLS 1.3 channel bindings](https://www.rfc-editor.org/rfc/rfc9266)
- [RFC 6189: short authentication strings](https://www.rfc-editor.org/rfc/rfc6189)
- [RFC 5280: Internet X.509 PKI certificate profile](https://www.rfc-editor.org/rfc/rfc5280)
- [RFC 9180: Hybrid Public Key Encryption](https://www.rfc-editor.org/rfc/rfc9180)

The normative behavior is this document, not an inferred behavior from a
reference implementation.

## Goals

### G1. Correct replicated data parallelism

Provide fixed-world synchronous data-parallel training for:

- multiple Macs using the MLX TCP ring backend; and
- multiple CUDA devices using the MLX NCCL backend.

The distributed update must be mathematically equivalent, within documented
floating-point tolerance, to a single-process update over the same effective
global batch.

### G2. Low-communication training on a Mac LAN

Provide synchronous DiLoCo rounds in which each Mac trains a full model replica
for many inner AdamW steps and periodically submits a parameter-space outer
gradient. Gigabit Ethernet must be used only at round boundaries.

### G3. Mixed Metal and CUDA learners

Allow Metal and CUDA learners to participate in one logical training run
without requiring a collective spanning the two backends. Workers exchange
versioned safetensors artifacts through a durable coordinator.

### G4. Hierarchical CUDA scaling

Allow a multi-GPU CUDA DDP group to act as one DiLoCo learner. Only the learner
leader communicates with the global coordinator.

### G5. Eventual elastic operation

After fixed-cohort behavior is validated, allow learners to join, leave,
restart, or submit at different rates using a separately reviewed
Decoupled-DiLoCo design.

### G6. Preserve mixlab's existing contract

When distributed training is absent or disabled:

- existing configs retain their current graph, weight layout, RNG behavior,
  checkpoint format, and output;
- `training.batch_tokens` retains its current meaning;
- existing single-process resume remains valid; and
- no distributed service, socket, or background thread is started.

### G7. Recruit idle LAN hosts without interactive login

Allow an administrator to enroll each lab machine once using an explicitly
selected trusted-LAN auto-enrollment window, a protected provisioning file, or
human-verified pairing; run a background mixlab daemon; discover available
machines on the local network; and submit a job that reserves and launches a
compatible fixed cohort without SSH.

Discovery and availability may change continuously. A training attempt's
membership may not change after its membership manifest is committed.

The complete enrollment, identity, certificate, revocation, and encrypted
transport path is provided by the mixlab binary. A normal deployment must not
require OpenSSL commands, a public certificate, an external CA service, a VPN,
or installation of a second agent. Use of an operating-system credential store
is an implementation adapter, not separately maintained infrastructure.

## Non-Goals

The release train does not include:

- synchronous per-step training across Metal and CUDA;
- per-step communication between the Mac LAN and RunPod;
- tensor, pipeline, context, or expert parallelism;
- FSDP or ZeRO parameter sharding;
- a parameter server that receives every minibatch gradient;
- bit-identical Metal and CUDA kernels;
- automatic dataset transfer to lab machines or RunPod;
- arbitrary remote command execution by the coordinator;
- trusting mDNS advertisements as authentication or authorization;
- enabling trusted-LAN auto-enrollment on a public endpoint or using it for
  controller, coordinator, or external-worker credentials;
- integrating an enterprise or public CA in R1.1;
- public-internet peer-to-peer access to lab Macs;
- automatic RunPod provisioning in R1-R3;
- gradient or delta compression before an uncompressed correctness baseline;
- Muon, NorMuon, LAMB, SWA, data2vec, distillation, RTD, or multihead support
  in the initial DiLoCo release; or
- elastic/asynchronous behavior before R5.

Authentication does not provide Byzantine training tolerance. An enrolled
participant assigned to a run can fail, stall, or submit incorrect values
within the operations its role permits; detecting malicious gradient or model
poisoning is outside R1-R4.

Architecture-race scheduling across machines is useful adjacent work, but it is
not part of this specification.

## Terminology

| Term | Definition |
|------|------------|
| Rank | One process in a fixed synchronous DDP group. |
| World | The complete set of ranks participating in a DDP collective. |
| Learner | One independently progressing model replica. A learner may contain one process or an internal DDP group. |
| Learner leader | Rank zero of a learner; the only process that talks to the global coordinator. |
| Island | One or more learners with similar placement or connectivity, such as the school Mac LAN or a RunPod pod. |
| Inner step | One attempted local optimizer update inside a learner. |
| Outer round | A period of local inner training followed by one global DiLoCo update. |
| Base parameters | The authoritative global parameters at the beginning of an outer round, written as `theta_r`. |
| Local parameters | A learner's parameters after its inner steps, written as `theta_r_i`. |
| Outer gradient | The parameter delta `g_i = theta_r - theta_r_i`. The sign is normative. |
| Effective denominator | The exact count or weight used by an objective to turn its loss sum into its reported scalar mean. |
| Coordinator | The durable control-plane process that owns global round state and the outer optimizer. |
| Artifact store | Content-addressed storage for model, delta, and worker-state safetensors. |
| DDP group membership | Immutable ordered member and rank assignment for one synchronous process group, including run, group, generation, backend, and canonical members hash. A process-local rank is a view of this shared value, not part of the shared hash. |
| Learner cohort membership | Immutable ordered learner-member assignment for one DiLoCo run, including run, generation, and canonical cohort hash. Allocation provider/capability data is attached to run-plan slots but is not part of membership identity. It is distinct from any DDP group inside a learner. |
| Membership generation | Monotonic identifier changed only when a different member or rank/learner assignment is committed within the same run and membership scope. Relaunching the same assignment uses a new launch-attempt ID, not a new generation. |
| Launch attempt | One attempt to start a previously selected membership assignment. Retries and restarts have distinct launch-attempt IDs even when membership is unchanged. |
| Node daemon | Background control-plane process that advertises a machine, reports authenticated capabilities, leases its accelerator, and supervises worker child processes. |
| Discovery record | Untrusted hint that a possible daemon exists at an address. It is never proof of node identity or capability. |
| Cluster trust domain | A mixlab-managed cryptographic namespace rooted in one persistent cluster signing identity. It is local application state, not an external CA service. |
| Mixlab state home | The per-user filesystem root for mixlab-owned durable state. It defaults to `~/.mixlab` on macOS and Linux and contains context-separated authority, principal, daemon, and worker directories. It is not a dataset, checkpoint, or artifact root. |
| Cluster identity fingerprint and phrase | SHA-256 of the canonical cluster-root public key and its versioned human-readable word encoding. They identify a root but are public and grant no enrollment authority. |
| Principal identity | A stable cluster-scoped ID and role proven by a cluster-signed certificate and possession of its private key. Display names, hostnames, addresses, process IDs, and mDNS fields are not principal identities. |
| Enrollment approval policy | The cluster-trust decision strategy for one enrollment request: `trusted_lan`, `provisioned`, or `verified`. It affects bootstrap authorization only, not the resulting certificate or later application authorization. |
| Enrollment request | A pending, expiring trust-context record binding the requested role, locally generated public-key request, observed channel evidence, and selected approval policy. Proposed names and addresses remain untrusted metadata until approval. |
| Enrollment channel binding | Immutable evidence from the dedicated bootstrap transport binding one full TLS 1.3 connection/exporter to its observed peer address and receiving interface. Transport produces it; cluster trust interprets it under an approval policy. |
| Enrollment approval record | A signed cluster-trust decision binding one approved request, policy, root, endpoint, role/purpose, public-key request, evidence digest, and expiry. It authorizes issuance only and grants no application operation. |
| Enrollment invitation / provisioning file | A short-lived, single-use protected bootstrap value containing the pinned cluster identity, invitation purpose, allowed resulting role, and high-entropy authorization secret. It never contains a cluster or controller private key. |
| Enrollment consumption receipt | A cluster-trust-signed, non-secret proof that one invitation was atomically consumed for an exact purpose, request hash, endpoint audience, and admission/recovery binding. It is evidence, not application admission. |
| Verified-pairing phrase | A per-request short authentication string derived from the live TLS channel binding, cluster root, requested role, endpoint audience, nonces, and certificate-request hash. It identifies one live request but is not a reusable secret. |
| Authority endpoint set | Root-signed routing metadata naming one or more same-binary cluster-trust authority endpoints. It is an address hint plus audience binding, not proof supplied by the address itself. |
| Trust snapshot | A dedicated-snapshot-signer-signed, monotonic, expiring statement of active issuers, role eligibility, authority endpoints, and revoked principal or certificate identifiers. It carries no resource/action grants. |
| Trust evidence | Durable identity and trust-snapshot data attached to a signed proof so later validation survives ordinary certificate renewal/expiry without treating a signer-supplied timestamp as a security anchor. |
| Workload identity | An ephemeral certificate and key binding one child process to its run, job, launch attempt, role, and, for DDP, exact group member and membership hash. |
| Secure group transport plan | A signed, immutable mapping from one DDP membership and launch attempt to job-scoped peer certificate fingerprints and encrypted transport endpoints. It configures connectivity but cannot create or alter membership. |
| Artifact access grant | A signed, expiring, principal-bound authorization for exact artifact operations within one run/worker scope. An artifact digest is identity, not authorization. |
| Reservation lease | Time-bounded exclusive claim on a daemon's advertised accelerator for one controller and job. |
| Prepared learner allocation | Provider-neutral proof that resources for one learner slot are prepared. It names a provider and carries only that provider's typed allocation reference. |
| Run plan | Immutable staged description of a fixed DiLoCo cohort and its training contract. Workers may verify its hash before it is committed. |
| Run commit | Small atomic commit record that makes one staged run plan authoritative after admission succeeds. |

## Bounded Contexts, Shared Kernels, And Dependency Rules

A bounded context owns domain decisions and the meaning of its state. A shared
kernel owns immutable value semantics used by several contexts. An adapter owns
translation to an external mechanism such as MLX, mDNS, HTTP, or a filesystem;
it does not acquire the domain decisions of the context that calls it.

This distinction is normative. Merely placing several responsibilities in one
Go directory does not merge their ownership. If existing package constraints
temporarily colocate them, internal packages and interfaces must preserve the
dependency directions below.

### Shared kernels

| Shared kernel | Release | Owns | Must not own |
|---------------|---------|------|--------------|
| Distributed identity | R1 | Immutable run/group/member/learner IDs, `DDPGroupMembership`, `LearnerCohortMembership`, ordered member descriptors, generation validation, and canonical hashes. | Cryptographic principal identity or certificate validation, selecting members, assigning ranks or learners, launch-attempt lifecycle, MLX initialization, training, discovery, or I/O. |
| Artifact reference | R1/R2 | Content digest, size, media/schema kind, and immutable `ArtifactRef` value validation. | Storage, upload authorization, checkpoint meaning, or resume policy. |
| Event envelope | R1 | Correlation identity, timestamp/sequence, schema version, severity, and bounded attribute rules. | Domain event meaning, aggregation policy, or control decisions. |

Protocol and manifest versions belong to the context that defines the protocol.
The distributed-identity kernel must not become a registry of unrelated daemon,
coordinator, artifact, and telemetry versions.
When a membership value contains a participant ID, the kernel treats it as an
opaque stable value; authenticated transport and the receiving context prove
and authorize the corresponding principal.

### Bounded contexts

| Context | Release | Owns and publishes | Consumes through ports | Must not own |
|---------|---------|--------------------|------------------------|--------------|
| Local training orchestration | R1 | Optimizer-attempt sequencing, validation/early-stop policy, rank-zero side effects, DDP checkpoint semantics, and calls to the local training engine. | Group runtime, trainer, sampler, reproducibility, artifact storage, and event sink ports. | Gradient representation, host recruitment, daemon leases, or global DiLoCo state. |
| Distributed training engine | R1 | Backward pass, accumulation, bucketed reduction, clipping, candidate-state transaction, and atomic local model replacement. | Immutable group membership and MLX group-runtime operations. | Datasets, files, daemon/coordinator APIs, checkpoint policy, or presentation. |
| Data identity and partitioning | R1 | Content-stable dataset identity, deterministic partition plans, sampler state, and effective-count reporting. | Immutable membership/partition inputs and local dataset readers. | MLX groups, host discovery, optimizer state, or allocation decisions. |
| Reproducibility | R1 | Versioned key derivation for initialization and stochastic training domains. | Stable run, learner, rank, step, and microstep values. | Discovery, membership assignment, mutable sampler state, or device-global RNG policy. |
| Cluster trust | R1.1 | Cluster trust-domain initialization and identity phrase, principal/role certificates, enrollment windows/requests/approval policies/invitations/receipts, issuance and renewal policy, signed trust snapshots, revocation, workload-credential issuance, and historical signed-proof validation. | Cryptographic signer/verifier, secure key-store, clock, randomness, enrollment channel evidence, administrator approval, and event ports. | Discovery, network-interface inspection, TLS/HTTP routing, phrase presentation, capabilities, leases, job manifests, membership selection, process launch, training, or allocation decisions. |
| Node management | R1.1 | Local node profile, principal-renewal scheduling trigger, controller authorization for node operations, authenticated capabilities, reservation leases, durable local job state, local dataset catalog, credential-envelope handling, and child supervision. | Authenticated-principal/trust-policy/renewal-client, discovery advertisement, structured job, credential-envelope, artifact, and launcher ports. | Certificate issuance, renewal policy, or revocation; cohort selection, DDP collectives, global rounds, model execution, or arbitrary commands. |
| Recruitment | R1.1 | Compatibility filtering, deterministic selection, all-or-abort LAN reservation/direct-R1 preparation saga, and proposed rank/learner assignments. | Discovery provider, authenticated node-agent client, and principal-signer ports. | Cross-provider run admission, committed coordinator state, private signing keys, MLX launch encoding, gradient synchronization, or outer optimization. |
| Run admission | R2 | Cross-context saga from proposed allocations through staged run plan, prepared allocations, worker admission, lease-authority handoff, run commit, or compensation. | Recruitment/provisioner, coordinator-admission, node-agent, cluster-trust workload-issuer/principal-signer, and event ports. | Trust policy, certificate lifecycle, private signing keys, discovery implementation, model tensors, inner training, or outer optimizer arithmetic. |
| Allocation recovery | R3 | Orchestration and compensation of the recoverable recredential/fencing saga for a failed external process, including requests for the pending-recovery transition, while preserving the exact committed participant, worker, allocation reference, and cohort generation. | Global-coordination recovery grant/pending-incarnation ports, external provisioner, cluster-trust invitation/workload ports, artifact authorization, and event ports. | Cohort/allocation replacement, trust policy, certificate issuance, authoritative incarnation/run state, global optimizer state, learner-private-state meaning, or model execution. |
| Learner runtime | R2 | Round acquisition, verified base-model installation, exactly `H` inner updates, learner-private state, outer-update construction, and idempotent submission. | Local training, coordinator client, artifact storage, allocation keeper, and event ports. | Cohort admission, authoritative global state, outer aggregation, or node discovery. |
| Global coordination | R2/R3 | Staged run plans, committed fixed cohorts, global rounds, outer optimizer, update authorization, run/worker-scoped artifact access grants, R3 external recovery grants plus failed/pending/active incarnation fencing, global checkpoints, and authoritative run state. | Authenticated principal, principal signer, artifact storage/use-ledger, evaluator, allocation-keeper, and event ports. | Certificate validation or issuance, private signing keys, artifact byte storage, recovery-saga orchestration, discovery internals, process launch encoding, model execution, or learner-private AdamW state. |

R3 adds allocation-provider adapters rather than changing these contexts. A LAN
node-agent allocation and an externally provisioned RunPod allocation both
produce a `PreparedLearnerAllocation`. R4 changes the implementation inside one
learner slot by adding a fixed DDP group; it does not expose internal ranks to
global coordination.

R5 must add or revise bounded contexts for elastic rendezvous/membership,
data-range leasing, and quorum/staleness aggregation during its required design
review. Those responsibilities are intentionally not assigned to R1-R4
contexts by implication.

### Infrastructure and application adapters

| Adapter | Owns | Called by |
|---------|------|-----------|
| Discovery provider | mDNS/DNS-SD, explicit-address, and fake browse/advertise translation. | Enrollment composition, recruitment, and node management. |
| Launcher/rendezvous | Backend-neutral `LaunchPlan` validation plus MLX ring/NCCL child environment and hostfile materialization. | Explicit R1 launcher or node management. |
| Group runtime | MLX group lifecycle, authoritative backend/rank/world size, startup identity exchange, and collective wrappers. | Local training orchestration and distributed training engine. |
| Secure collective transport | TLS 1.3 byte transport, invocation of the cluster-trust verifier, exact authenticated-workload-to-plan binding, endpoint confinement, and translation to loopback MLX ring endpoints. It does not issue certificates, interpret collectives, authorize application operations, or choose peers. | Launcher/rendezvous only; the group runtime sees validated loopback launch inputs and does not import the transport adapter. |
| Secure key store | Storage of private keys and protected handles using macOS Keychain when available or strict owner-only files. It does not issue identities or decide trust. | Cluster trust and composition roots. |
| State-home resolver | Resolution and validation of the portable per-user state root, context-specific directory names, overrides, ownership, permissions, and unambiguous selection. It never parses state contents or decides trust, enrollment, node, worker, run, or recovery policy. | Composition roots only. |
| Trust authority client/locator | Resolution of root-signed authority endpoints, local in-process invocation, authenticated remote trust-protocol calls, and fake implementations. It does not issue credentials or choose trust policy. | Cluster trust application services and composition roots. |
| Enrollment bootstrap transport | Dedicated provisional TLS profile, TLS-exporter channel evidence, observed peer/local-interface metadata, and bounded request polling. It does not discover endpoints, present phrases, select an approval policy, approve a request, assign a principal, or issue a certificate. | Enrollment composition root and cluster-trust application ports. |
| Enrollment approval UI | Display of trust-computed cluster/request phrases, fingerprints, warnings, and pending metadata plus collection of explicit local confirmations. It does not compute cryptographic values, interpret a match, choose policy, or issue/approve a credential. | Enrollment composition root through cluster-trust presentation/approval ports. |
| Artifact authorization gateway | Authenticated request binding plus the durable atomic use/idempotency ledger for already issued artifact grants. It does not decide semantic access, issue grants, interpret artifact contents, or store bytes. | Coordinator API transport, using global-coordination authorization decisions. |
| Artifact storage | Content-addressed byte storage, streaming checksum/size enforcement, atomic blob publication, and authorization hooks. | Owning contexts through `ArtifactStore`; it does not interpret checkpoint semantics. |
| Observability | Event transport, storage, aggregation, and presentation. | Every context emits its own semantic events through an `EventSink`. |
| CLI/API transport | Normal TLS 1.3 mechanics, cluster-certificate verification through trust ports, strict decoding, deadlines, and mapping from a verified connection to an `AuthenticatedPrincipal` plus application command. Provisional enrollment is confined to the separate bootstrap adapter. | Composition roots; transport handlers contain no authorization, optimizer, lease, or scheduling policy. |

### Required cross-context contracts

| Contract | Authoritative producer | Consumers |
|----------|------------------------|-----------|
| `DDPGroupMembership` | Explicit launcher or recruitment; verified by group runtime. | Group runtime, local training, trainer, sampler, and DDP checkpoints. |
| `LearnerCohortMembership` | Global coordination when the run is committed. | Learner runtime, round/update validation, global checkpoints, and observability. |
| `LaunchPlan` | Explicit launcher or run-admission application service. | Launcher/rendezvous adapter and node management. |
| `AuthenticatedPrincipal` | Authenticated transport after certificate validation through cluster trust. | Node management, run admission, recruitment clients, and global coordination authorization. |
| `PrincipalSigner` | Cluster trust for one locally available, non-revoked principal key handle. | Recruitment, run admission, global coordination, and composition roots signing canonical context-owned bytes. |
| `ClusterIdentity`, `EnrollmentRequest`, `EnrollmentApprovalRecord`, `EnrollmentInvitation`, `EnrollmentConsumptionReceipt`, `AuthorityEndpointSet`, `TrustSnapshot`, and `TrustEvidence` | Cluster trust. | Enrollment composition/UI, trust-authority client/locator, authenticated transport, signed context-owned records, run admission, recovery, and node management trust-policy checks. |
| `EnrollmentChannelBinding` | Enrollment bootstrap transport from one completed TLS 1.3 handshake plus observed connection metadata. | Cluster trust only as evidence evaluated by the selected enrollment policy. |
| `EnrollmentPresentation` | Cluster trust from canonical identity/request values and the selected policy. | Enrollment approval UI for display only. |
| `ClientPhraseConfirmation` | The enrolling user, captured by the client-side approval UI and bound to the exact presentation/request/channel digest. | Cluster trust through the bounded bootstrap route; it confirms the client-side comparison but cannot approve issuance by itself. |
| `AdministratorConfirmation` | The locally present administrator, captured by the enrollment approval UI and bound to the exact presentation/request digest. | Cluster trust through `AdministratorApprovalPort`; it is input to a policy decision, not an issued credential. |
| `WorkloadIdentity` | Cluster trust credential issuer after receiving a context-owned run/job binding. | Node management, secure collective transport, learner/coordinator transport, and external-enrollment adapter. |
| `SecureGroupTransportPlan` | Launcher/rendezvous after receiving immutable membership plus trust-issued workload-certificate bindings. | Node management and secure collective transport. |
| `NodeCapabilities` | Authenticated node management. | Recruitment only as a selection input. |
| `PreparedLearnerAllocation` | A provisioner such as LAN recruitment or external enrollment. | Run admission and global coordination. |
| `ExternalRecoveryGrant` | Global coordination for one failed committed external slot and one pending replacement attempt; it does not contain a not-yet-created certificate-request hash. | Allocation recovery, external provisioner, and cluster trust as an opaque workload-issuance binding. |
| `RunPlan` and `RunCommit` | Global coordination. | Run admission, learner runtime, global recovery, and observability. |
| `RoundAssignment` | Global coordination. | Learner runtime. |
| `OuterUpdateManifest` | Learner runtime. | Global coordination. |
| `UpdateSubmission` | Learner client transport. | Coordinator authentication/allocation adapter, which passes only a validated `OuterUpdateManifest` to domain admission. |
| `ArtifactRef` | Artifact storage after verified publication. | Context-owned manifests and recovery code. |
| `ArtifactAccessGrant` | Global coordination from a staged/committed slot, round assignment, or validated recovery operation. | Coordinator API authorization adapter and artifact storage authorization hook. |
| `ScopedApplicationCredential` | The bounded context that owns the protected application action; R2 normally needs none beyond mTLS. | Cluster-trust credential sealer and the exact intended child transport. |
| `CredentialEnvelope` | Cluster-trust credential-sealing service after receiving an opaque scoped application credential plus node/job/audience binding. | Node management for daemon-launched jobs; never artifact storage. |
| `EventEnvelope` | Emitting context. | Observability sinks and presentation. |

Normative dependency rules:

- Selection contexts create assignments; the distributed-identity shared kernel
  only validates and hashes them.
- Cluster trust proves principal identities and issues scoped credentials; it
  never selects members, assigns ranks, grants leases, or admits updates.
- All three enrollment policies terminate in the same cluster-trust request,
  approval, issuance, and audit state machine. Policy adapters may supply
  different approval evidence but cannot mint different certificate profiles
  or bypass role restrictions.
- Discovery supplies candidate enrollment addresses only. It never chooses a
  cluster root or approval policy; cluster trust never browses mDNS.
- The bootstrap transport supplies TLS-exporter and observed-network evidence
  but cannot interpret `trusted_lan`, `provisioned`, or `verified` policy.
  Phrase display and button/terminal input are presentation adapters; cluster
  trust alone validates confirmations and records the approval.
- The state-home resolver validates and selects context directories but never
  parses their contents. Each composition passes only the required resolved
  directory/repository to a context; no domain context traverses sibling
  authority, principal, daemon, or worker directories.
- `PrincipalSigner` signs an opaque caller-owned canonical digest plus
  context/audience and returns `TrustEvidence`; it never parses or owns the
  signed manifest/grant's domain semantics.
- `PrincipalSigner` is not a raw signing oracle. Its request includes a closed,
  versioned signature-purpose enum, digest, context, and audience. Cluster trust
  enforces role-to-purpose policy (`authority` only for enumerated
  trust-administration attestations, `controller` for node-job/transport control
  records, `coordinator` for run/round/recovery/artifact grants, and `worker`
  only for its own scoped worker proofs) before using a key handle. Certificate
  issuance and trust-snapshot signing use separate CA-purpose handles and are
  never exposed through `PrincipalSigner`.
- Authenticated transport proves a peer and returns an
  `AuthenticatedPrincipal`; the receiving bounded context alone decides
  whether that principal may perform the requested operation.
- A new process start always creates a new launch-attempt ID. A membership
  generation changes only when its ordered assignment changes.
- The launcher/rendezvous adapter is the only code that materializes MLX
  hostfiles or child environment variables. The group runtime is the only
  training-process code that reads them.
- All other contexts receive immutable group or cohort membership values; they
  do not rediscover rank, world size, or learner identity.
- The initialized MLX group is authoritative for runtime rank and world size.
  The group runtime rejects disagreement with expected DDP membership before
  trainer or sampler creation.
- The sampler receives rank/learner and partition values. It never imports or
  calls MLX.
- Gradients and optimizer tensors never cross into Go merely to implement DDP
  synchronization.
- Custom C++ trainer code never opens datasets, checkpoints, daemon/control
  sockets, or daemon state. Only the MLX group runtime uses the loopback
  collective endpoints materialized by the launcher.
- The node daemon does not initialize MLX in its long-lived process. It invokes
  a launcher adapter and supervises a separate worker process so GPU state,
  crashes, and environment are isolated per job.
- Discovery records never grant authority. Identity and capability claims are
  accepted only over an authenticated node-agent connection.
- DDP membership identity and cryptographic principal identity are distinct.
  A rank must satisfy both the immutable membership check and its workload
  certificate binding; neither check substitutes for the other.
- For an R1.1-launched MLX ring, raw collective bytes never traverse a
  non-loopback interface. The secure collective transport owns network
  encryption and peer authentication; the group runtime retains collective
  ordering and membership-consistency checks.
- Job launch accepts a structured, versioned manifest. Neither scheduler nor
  coordinator may submit arbitrary commands or shell fragments.
- Artifact storage verifies bytes; the context that owns a checkpoint or
  manifest decides its semantic validity and resume policy.
- Each context owns the meaning and schema version of its events. Observability
  sinks never make training, admission, scheduling, or lease decisions.
- Checkpoints record membership identity and generation but cannot change
  membership.

## System Invariants

These invariants apply across releases.

### I1. One authoritative global state

The coordinator owns the authoritative global model and outer optimizer state
for DiLoCo. A learner checkpoint is never implicitly promoted to the global
checkpoint.

For DDP without DiLoCo, every rank has an identical replicated state and rank
zero writes the authoritative checkpoint.

### I2. Compatible programs, not guessed compatibility

Before training or accepting an update, participants must agree on:

- config hash;
- serialized IR/program hash;
- ordered weight-layout hash containing every weight name, shape, and dtype;
- inner optimizer-plan hash;
- dataset identity; and
- every context-owned protocol/manifest version used by that interaction.

A mismatch is a hard error. Tensor position alone is not sufficient identity.

### I3. Globally consistent optimizer decisions

No DDP rank may independently commit or skip an optimizer step. Loss,
gradient, and candidate-state finiteness decisions are collective decisions.
All ranks advance attempted-step, committed-step, schedule, QAT, and program
phase state identically.

### I4. Exact collective order

Every DDP rank must issue the same collectives in the same order with matching
shape and dtype. A conditional collective is allowed only after a preceding
collective has made the condition identical on every rank.

### I5. Correct loss weighting

Equal averaging of per-rank mean gradients is allowed only when the effective
denominator is provably equal on all ranks.

Otherwise, if local gradients are gradients of a local mean,

```text
local_mean_grad_i = grad(local_loss_sum_i / denominator_i)
```

the global gradient must be:

```text
global_grad =
    sum_i(denominator_i * local_mean_grad_i)
    / sum_i(denominator_i)
```

The same numerator/denominator rule applies across gradient-accumulation
microsteps. An implementation must not infer a denominator from tensor shape
when the objective uses a mask or validity weight.

Objectives that combine independently normalized component losses are not
supported until each normalization group has explicit aggregation semantics.

### I6. Initialization and stochastic training use different RNG domains

- Parameter initialization is identical across replicas and does not include a
  rank or learner identifier.
- Dropout, masking, corruption, augmentation, and other stochastic training
  operations include a stable rank or learner identifier so replicas do not
  consume identical stochastic samples.
- RNG keys include optimizer step and accumulation microstep.
- A resumed fixed-world run reproduces the same keys.

`OP_RANDOM_NORMAL` must either be converted to keyed RNG before it is enabled
in distributed training or cause distributed validation to reject the program.
Silent use of the global MLX RNG is forbidden.

### I7. Manifests commit artifacts

Large tensor files are written and checksummed before the owning context
publishes its designated commit manifest or record. Publishing that designated
record is the atomic commit marker. A staged R2 run plan is explicitly not a
commit marker; its matching run-commit record is. Incomplete files and staged
records without their context-defined commit marker are ignored during
recovery.

For every signed or self-identifying format, canonicalization explicitly
defines the unhashed payload. Its digest excludes the digest/signature fields,
and the signature covers that digest plus its context/audience binding. No
format computes a hash over a field containing that same hash.

Every durable signed proof additionally embeds canonical `TrustEvidence`:
cluster ID, signer principal/role, certificate serial, the complete signer
certificate chain, the complete signed `TrustSnapshot` accepted when the
record was committed, signature algorithm, and a recorded signing time. The
time is diagnostic only and is never trusted to order a proof against
revocation. Ordinary leaf renewal or expiry does not invalidate an earlier
proof whose bytes were valid and durably journaled by its owning context.
At initial acceptance, the owning context verifies the signer against current
trust and atomically appends the proof digest plus accepted snapshot generation
to its own monotonic commit journal. Cluster trust supplies verification rules
and evidence parsing; it does not own the run/job/artifact journal or decide
that an application record committed.

Revocation records have an explicit mode and reason. A `prospective`
revocation rejects new handshakes and signatures beginning with the snapshot
generation that carries it. It preserves a proof only when the proof digest
was already present in the owning context's durable, monotonic commit journal
before that generation; presenting old trust evidence later cannot manufacture
a historical commit. A `compromise` revocation invalidates every historical
proof from the named principal or certificate serial. It has no
administrator-supplied compromise time, because a stolen key could backdate a
record. Any run whose authoritative commit depends on invalidated trust
evidence fails closed for explicit administrative recovery; no context
silently re-signs or transfers authority. Compromise of the cluster root,
certificate issuer, or trust-snapshot signer is a cluster-compromise/rekey
event, not an ordinary leaf revocation. Certificate renewal for the same
non-revoked principal preserves identity; replacing the principal requires a
context-owned recovery transition, not a trust-layer alias.

### I8. Idempotent network mutation

Every submitted outer update has a stable update ID. Retrying an HTTP request
must not apply the update twice. Coordinator state transitions use compare-and-
swap semantics on run and round version.

### I9. No hidden batch-size semantics

`training.batch_tokens` remains local microbatch tokens per process.
Distributed logs and manifests must report:

- local microbatch tokens;
- accumulation microsteps;
- world size or learner count;
- effective valid tokens;
- attempted and committed inner steps; and
- effective global batch tokens for DDP.

Learning-rate scaling is never automatic.

### I10. Rank-zero side effects

In DDP, only learner rank zero may:

- print normal progress;
- write telemetry artifacts;
- run validation;
- make early-stop decisions;
- write model/checkpoint files; or
- communicate with the global coordinator.

Other ranks participate in the corresponding synchronization and receive the
resulting decision.

### I11. Trust, authorization, membership, and transport are distinct

No address, hostname, display name, discovery field, rank claim, worker ID, or
bearer value is itself a principal identity.

- Provisioned enrollment pins the cluster root from its protected provisioning
  file. Verified enrollment pins it only after human comparison of the stable
  cluster phrase and live request phrase. `trusted_lan` is the sole explicit
  exception: it performs first-connection TOFU inside an administrator-opened,
  bounded local enrollment window. No mode treats an mDNS field as root proof.
- A TLS handshake proves certificate validity and private-key possession and
  yields an `AuthenticatedPrincipal`; the receiving bounded context performs a
  separate role- and resource-specific authorization decision.
- A workload certificate is accepted only for its exact run, job, launch
  attempt, role, audience, expiry, and, for DDP, member/rank and membership
  hash.
- An authenticated DDP member must also pass the immutable MLX
  rank-to-membership consistency exchange.
- Collective traffic crossing a non-loopback interface is mutually
  authenticated and encrypted. A signed startup exchange over an unprotected
  socket is not a substitute for transport integrity.

Trust authorization means only that a credential is eligible to represent a
cluster role and scope. It never grants a lease, job mutation, coordinator
round, update, or artifact operation. The minimum authorization map is:

| Interaction | Cryptographic prerequisite | Resource/action authorization owner |
|-------------|----------------------------|-------------------------------------|
| Trusted-LAN enrollment | Provisional channel plus policy-compliant observed interface/CIDR; root accepted by explicit TOFU mode | Cluster trust, limited to auto-approving a `node` request within the active window |
| Provisioned enrollment | Provisioning-file-pinned authority plus valid purpose/secret | Cluster trust, limited to the file's one credential bootstrap |
| Verified enrollment | Human-confirmed cluster phrase and request-specific TLS-exporter phrase | Cluster trust, limited to the explicitly approved request/role |
| Renewal or workload issuance | Eligible principal/authority plus signed trust binding | Cluster trust, limited to credential lifecycle |
| Stale trust refresh | Newer dedicated-snapshot-signer value verified against pinned root; no client principal | Cluster trust, limited to monotonic snapshot installation |
| Node capabilities, lease, prepare, start, cancel, or logs | `controller`/prepared-successor principal as applicable | Node management against local node/lease/job policy |
| Worker registration, round lease, or outer update | Exact `worker` workload principal | Run admission/global coordination against staged or committed slot and allocation proof |
| Coordinator artifact read/write | Exact coordinator or active worker workload principal plus an operation-specific grant | Global coordination owns grant issuance and semantic validation; the coordinator API adapter atomically accounts for use, and learner runtime only requests/consumes |
| Pending recovery state read and activation | Exact `PENDING_RECOVERY` workload principal plus recovery grant/receipt | Global coordination permits only the exact private-state read and recovery activation; allocation recovery orchestrates, learner runtime restores |
| DDP byte connection | Exact workload principal in `SecureGroupTransportPlan` | Secure transport admits the connection; group runtime separately verifies membership/rank |

## Public Configuration Contract

Add an optional `distributed` object to `TrainingSpec`:

```jsonc
{
  "training": {
    "batch_tokens": 8192,
    "distributed": {
      "mode": "ddp",
      "backend": "auto",
      "gradient_accumulation_steps": 4,
      "gradient_bucket_bytes": 33554432
    }
  }
}
```

The Go type should be a pointer so omission is distinguishable from an enabled
zero value:

```go
type DistributedSpec struct {
    Mode                      string
    Backend                   string
    GradientAccumulationSteps int
    GradientBucketBytes       int

    InnerSteps        int
    OuterOptimizer    string
    OuterLR           float64
    OuterMomentum     float64
    Aggregation       string
}
```

JSON names use the snake-case names shown below.
The implementation must separately track required-field presence where a zero
value is valid. In particular, an explicit `outer_momentum: 0` is valid but an
omitted `outer_momentum` is not.

`DistributedSpec` contains algorithm and local-training semantics only.
Membership, pool selection, node addresses, leases, coordinator credentials,
TLS keys, artifact-storage credentials, and RunPod provisioning never enter the
architecture config or its compatibility hash. Deployment manifests reference
the immutable config hash and carry their own context-owned operational fields.

### Common fields

| Field | Release | Values and behavior |
|-------|---------|---------------------|
| `mode` | R1 | Required when the object is present. R1: `ddp`; R2 adds `diloco`. |
| `backend` | R1 | `auto`, `ring`, or `nccl`. Empty defaults to `auto`. Applies to an internal DDP learner only. |
| `gradient_accumulation_steps` | R1 | Positive integer; default 1. |
| `gradient_bucket_bytes` | R1 | Positive integer; default 33,554,432 bytes. |

### DiLoCo fields

| Field | Release | Values and behavior |
|-------|---------|---------------------|
| `inner_steps` | R2 | Required positive number of committed AdamW updates per outer round. Recommended initial experiment: 500. |
| `outer_optimizer` | R2 | R2 supports only `nesterov`. |
| `outer_lr` | R2 | Required positive scalar. No silent research default. |
| `outer_momentum` | R2 | Required value in `[0,1)`. No silent research default. |
| `aggregation` | R2 | Required: `uniform` or `effective_tokens`. |

`training.steps` remains the number of inner optimizer attempts per learner.
R2 requires it to be divisible by `inner_steps`. Supporting a shorter final
round is deferred.

Unknown fields are rejected. Because nested types with custom JSON unmarshalling
can bypass a parent decoder's `DisallowUnknownFields`, `DistributedSpec` must
itself decode strictly (or have an equivalently tested strict parent path).
Tests must cover an unknown field inside `training.distributed`.
Defaults and validation must be documented in `docs/config-reference.md` and
the training guide in the same change that exposes them.

### Validation rules

R1:

- `mode=ddp` requires the initialized MLX group's size to be greater than one;
  use ordinary single-process training otherwise.
- `MLX_WORLD_SIZE` is not a universal validation input. NCCL launchers may set
  it, while ring launchers derive membership from the MLX hostfile.
- `backend=ring` is Metal/macOS only in the supported matrix.
- `backend=nccl` is CUDA only.
- `backend=auto` selects NCCL on CUDA and ring on Metal.
- `backend=auto` is resolved to an explicit backend before strict MLX group
  initialization. Mixlab must not pass MLX an unrestricted fallback backend.
- A world may not contain both Metal and CUDA ranks.
- `arch_race` rejects distributed config.
- Distributed lookahead submission is disabled.

R2:

- `mode=diloco` requires `training.optimizer=adamw`.
- `inner_steps`, outer LR, outer momentum, and aggregation are explicit.
- QAT, SWA, data2vec, distillation, RTD, invariance, minimal-pair auxiliary
  training, PLL margin training, multihead objectives, and scheduled topology
  changes are rejected until they have dedicated semantics.
- A learner may have world size one or may be an R1 DDP group.

The validator must list every incompatible field in one error where practical,
rather than failing one field at a time.

## Command-Line Contract

### Portable persistent-state convention

Every stateful mode accepts the global optional override:

```text
-state-home PATH
```

The mixlab state home resolves in this order:

1. an exact context-specific flag such as `-cluster-state-dir`,
   `-principal-state-dir`, `-daemon-state-dir`, or `-worker-state-dir` for that
   one context;
2. `-state-home`;
3. the non-secret `MIXLAB_STATE_HOME` environment value; or
4. the current user's home directory plus `.mixlab`.

The portable default is therefore `~/.mixlab` on both macOS and Linux,
including headless CUDA/RunPod images. Resolution is performed by a
composition-owned state-home adapter before constructing a domain application.
Domain contexts receive an already resolved context directory or repository
port; they do not read environment variables, choose defaults, enumerate other
contexts, or infer authority from co-location.

The canonical logical layout is:

```text
~/.mixlab/
  clusters/<cluster-id>/authority/
  principals/<cluster-id>/<role>/<principal-id>/
  daemons/<cluster-id>/<node-id>/
  workers/<cluster-id>/<worker-id>/
  staging/enrollment/<random-id>/
```

`clusters/.../authority` exists only on a machine intentionally given cluster
authority state. An ordinary enrolled Mac, Linux node, or RunPod worker must
not receive or create it. Principal identity/trust state remains separate from
daemon lease/job state even though a node ID equals its principal ID. Worker
state contains learner-runtime recovery state, not authority state.
Coordinator checkpoints, model artifacts, datasets, and logs can be large or
operator-managed and do not implicitly live under the state home; their
existing explicit paths remain separate.

`cluster-init` generates the cluster/principal IDs before atomically publishing
their directories. Enrollment creates an owner-only random staging directory,
generates the private key there or in the secure key store, and atomically
publishes the final principal directory only after successful validation and
issuance. TLS exporter material is never written to staging. Failed or expired
enrollment removes its staging state.

When an exact directory flag is absent, a command may use an existing default
directory only if exactly one entry matches the required cluster, role, and
context. Zero matches produces a not-enrolled/not-initialized error; multiple
matches fail with an ambiguity error and require the exact flag. Mixlab never
selects the first directory, replaces another identity, or merges two
clusters. Overrides are resolved to clean absolute paths without shell
interpolation and change location only; they do not bypass context separation
or any ownership, permission, link, and file-type check.

The state home and every mixlab-created directory descendant must be owned by
the effective user, must not itself be a symlink, and use mode `0700`;
protected regular files use mode `0600`. Creation and replacement use
same-filesystem temporary files, `fsync` where durability is required, and
atomic rename. Unsafe ownership, group/world access, symlinks, traversal,
hard-link substitution, and non-regular protected files fail closed.

On macOS the secure-key-store adapter prefers Keychain and stores only opaque
handles plus public state under `~/.mixlab`. On Linux, including headless
RunPod, the mandatory no-extra-service backend is strict owner-only key files
inside the relevant context directory. The strict-file backend remains an
available macOS fallback. Runtime sockets, PID files, and locks are ephemeral
runtime-adapter state and must not be treated as durable authority, principal,
daemon, or worker state.

### R1 DDP

R1 continues to use `-mode arch`. Processes are launched by `mlx.launch`, a
cluster scheduler, or equivalent environment setup. Mixlab consumes the MLX
rank/backend environment through the group-runtime boundary rather than
inventing a second rank protocol.

Example shape:

```bash
mlx.launch --backend ring --hostfile mac-ring.json -- \
  ./mixlab -mode arch -config config.json -train 'data/train_*.bin'
```

The exact launcher syntax must be verified against the pinned MLX release and
documented; the command above is illustrative until that verification.

R1.1 generates equivalent launch inputs and starts one local worker through
each reserved daemon. It does not change the R1 worker CLI or group semantics.
An explicit R1 launcher predating R1.1 is an administrator-operated primitive
and requires a trusted isolated network. The user-facing school-LAN workflow
must use the R1.1 managed trust and secure collective transport path.

### R1.1 node daemon and recruitment

Add these single-binary trust lifecycle modes:

```text
-mode cluster-init
-cluster-state-dir PATH
-controller-principal-state-dir PATH
-coordinator-principal-state-dir PATH
-trust-listen ADDRESS
-trust-advertise mdns|off

-mode cluster-invite
-cluster-state-dir PATH
-invite-purpose node-enrollment|controller-enrollment|coordinator-enrollment|external-worker-bootstrap
-invite-output PATH
-invite-ttl DURATION
-bootstrap-endpoint URL
-coordinator-principal-id ID

-mode cluster-enrollment
-cluster-state-dir PATH
-principal-state-dir PATH
-enrollment-listen ADDRESS
-enrollment-advertise mdns|off
-enrollment-policy trusted-lan|verified
-enrollment-ttl DURATION
-enrollment-max-nodes N
-enrollment-interface NAME
-enrollment-cidr CIDR
-enrollment-allow-purpose PURPOSE

-mode cluster-enroll
-principal-state-dir PATH
-enrollment-policy trusted-lan|provisioned|verified
-enrollment-provisioning-file PATH
-enrollment-coordinator URL
-enrollment-discover mdns|off

-mode cluster-revoke
-cluster-state-dir PATH
-principal-id ID
-certificate-serial SERIAL
-revocation-reason TEXT
-revocation-mode prospective|compromise
-pool local
-daemon-address ADDRESS

-mode trust-authority
-cluster-state-dir PATH
-trust-listen ADDRESS
-trust-advertise mdns|off
```

`cluster-init` creates the trust domain, root identity, and first authority,
controller, and coordinator principals. It atomically writes the controller
and coordinator credentials to their two resolved principal-state directories;
neither directory is inside the resolved cluster-authority directory, and
neither receives an authority-principal or CA-purpose private-key handle.
`cluster-invite` writes a protected, short-lived, single-use provisioning file;
it never prints the invitation secret in normal logs. `EnrollmentInvitation`
remains the internal contract name, while the public policy name is
`provisioned`. For a principal-enrollment provisioning file, unless an
already-running trust authority endpoint is selected, `cluster-invite` also
runs the temporary enrollment endpoint until the file is consumed, expires, or
the user cancels it; this is not a permanent CA service. An
`external-worker-bootstrap` invitation instead requires
`bootstrap-endpoint` and `coordinator-principal-id`; it binds the exact
coordinator external-preflight endpoint/audience and never redirects the
worker to a generic trust endpoint.

`cluster-enrollment` is the source-side composition for interactive LAN
enrollment. It advertises and serves the enrollment endpoint, renders the
stable cluster identity and pending verified requests, and calls cluster trust
through an in-process application port. `principal-state-dir` supplies the
source coordinator identity; a cluster-authority directory must resolve either
through `cluster-state-dir` or unambiguously under the state home, but is opened
only by the colocated cluster-trust application, never coordinator domain code.
Only `trusted-lan` and `verified` are valid source policies.
`enrollment-allow-purpose` is repeatable, defaults to `node-enrollment`, and is
constrained by the policy rules below; `trusted-lan` rejects every value except
`node-enrollment`.
Interface/CIDR are mandatory for `trusted-lan` and optional additional
restrictions for `verified`; TTL and maximum approvals apply to both.

`cluster-enroll` is the client composition. For `trusted-lan` or `verified`, it
uses the explicit coordinator URL or browses the enrollment discovery service;
if discovery returns more than one candidate it fails and requires an explicit
selection. For `provisioned`, the protected provisioning file is required and
supplies the pinned root, endpoint, purpose, and secret. A client with a
non-empty principal-state directory never replaces its existing root or
identity automatically. For a node, `principal-state-dir` is the daemon's
protected identity subdirectory. The target host needs only the mixlab binary
plus either network access to an open enrollment endpoint or its provisioning
file. `cluster-enroll` accepts only the three principal-enrollment purposes;
`train-worker` consumes external bootstrap or recovery invitations through the
external provisioner.

Batch tooling may generate multiple distinct provisioning files, but one file
must never enroll multiple nodes. `cluster-revoke` publishes the next
snapshot-signer-signed trust snapshot and asks reachable daemons to synchronize
it.
The composition root obtains push targets from the discovery provider selected
by `pool` plus any explicit daemon addresses; cluster trust itself never
discovers nodes. It acts on trust state and does not edit a node lease or job
aggregate. `revocation-mode` defaults to `prospective`. `compromise` is an explicit
high-impact mode that invalidates all historical proofs for the named
principal or certificate; it accepts no administrator-supplied compromise
timestamp. Exactly one of `principal-id` or `certificate-serial` is required.
`trust-authority` is an optional same-binary background composition
for multi-controller or unattended renewal. The normal single-controller
workflow does not require it: `cluster-invite`, `nodes`, and `submit` can host
the same authority application transiently from local authority state.

Add these operation modes:

```text
-mode daemon
-daemon-listen ADDRESS
-daemon-state-dir PATH
-daemon-advertise mdns|off

-mode nodes
-pool local
-principal-state-dir PATH
-cluster-state-dir PATH

-mode submit
-pool local
-principal-state-dir PATH
-cluster-state-dir PATH
-workers N
-config PATH
-train DATASET_SELECTOR
-coordinator URL
```

`daemon` is long-running and may be installed as a per-user background
service. `nodes` lists authenticated capability and availability results, not
raw discovery advertisements. `submit` drives the run-admission application
service. For R1 it reserves a compatible world, creates immutable DDP
membership, and starts structured jobs. For R2+ it coordinates proposed
allocations, a staged coordinator run plan, preparation, registration, and run
commit. It reads `training.distributed.mode` from the config. `-coordinator` is
required for a DiLoCo submission and omitted for a directly supervised R1 DDP
submission. For R1 DDP, the `submit` process is the launcher and remains alive
to monitor and terminate the fixed world.

The exact service-install mechanism is platform packaging work. It must not be
required for foreground daemon operation or automated tests.

Node profile and cached operational trust state live in the resolved daemon
directory. Cluster authority state lives in the resolved cluster-authority
directory. Private keys are represented by protected key-store handles and are
never stored in architecture config. Only the cluster-trust application
service opens cluster authority state. A primary `submit`/run-admission
composition may host that service in the same binary and process through its
credential-issuer port; recruitment and admission code still never opens
signing keys directly. A secondary controller calls an authenticated authority
endpoint instead. Long-lived node, controller, and coordinator roles open their
own resolved principal directory containing their certificate, public roots/
snapshots, and private-key handle. An externally started worker instead opens
its resolved worker directory for learner-private state, public trust material,
and its run-scoped workload-key handle; a daemon-launched worker uses its
protected per-job directory. None receives root, certificate-issuer, or
snapshot-signer private-key handles merely because roles are colocated on one
machine.
For the provisioned policy, node, controller, and coordinator bootstrap values
are supplied through protected provisioning files or standard input, never as
a command-line value or environment variable. Verified enrollment has no
bearer secret file.
R3 permits an ephemeral external-worker provisioning value in a RunPod
protected environment secret because that is the provider's bootstrap channel;
it remains single-use and is cleared after exchange. State-directory file
adapters apply the portable state-home ownership, permission, link, and atomic
publication rules.

### R2+ coordinator

Add:

```text
-mode train-coordinator
-config PATH
-checkpoint-dir PATH
-distributed-listen ADDRESS
-distributed-artifact-dir PATH
-principal-state-dir PATH
```

The coordinator is CPU-capable and must not initialize MLX merely to serve or
aggregate float32 safetensors. It obtains its server identity and trust roots
from managed coordinator-principal state. Normal operation exposes no
certificate, private-key, or client-CA flags and does not modify the system
trust store.

For R3, `train-coordinator` may explicitly cohost the same enrollment-source
component described by the `cluster-enrollment` flags on its distributed
endpoint, with local `cluster-state-dir` and
`external-worker-bootstrap` as the allowed purpose. It permits `verified` but
rejects `trusted-lan`. This is composition reuse: coordinator handlers/UI call
cluster-trust and external-provisioner ports and do not acquire enrollment
policy or issuance ownership.

### R2+ learner

Add:

```text
-mode train-worker
-config PATH
-train GLOB
-coordinator URL
-worker-id ID
-island-id ID
-worker-state-dir PATH
-enrollment-policy provisioned|verified
-enrollment-provisioning-file PATH
```

`worker-id` is stable across restarts and identifies private inner optimizer
state. `island-id` groups operational telemetry; it does not change the R2
aggregation formula.

Non-loopback coordinator listen addresses require a managed coordinator
certificate. A learner pins the cluster identity supplied by its provisioning
file or human-verified enrollment; certificate verification is never disabled
by an insecure convenience flag.

Provisioned workers bootstrap from a protected provisioning file,
`MIXLAB_ENROLLMENT_PROVISIONING_FILE`, or a RunPod-protected
`MIXLAB_EXTERNAL_WORKER_PROVISIONING` value. A bearer value may be used only
once to complete enrollment and obtain a run-scoped workload identity.
Verified workers perform the same phrase/approval protocol at the explicit
coordinator URL without a bearer file. `trusted_lan` is invalid for
`train-worker`. Ongoing coordinator requests use the resulting workload
identity. A node daemon receives a per-job
workload certificate over its mutually authenticated control connection and
materializes its locally generated workload key/certificate as owner-only
files. If an owning application context requires an additional scoped secret,
it is delivered in a `CredentialEnvelope`. The daemon passes only protected
paths to the child. Daemon-launched secrets never appear in the architecture
config, job/run manifest, content-addressed artifact store, environment, or a
flag likely to appear in process listings. The explicitly named
RunPod-protected provisioning value is the only R3 environment bootstrap
exception and is cleared after consumption.

### R3 external worker recovery

Add:

```text
-mode recover-worker
-principal-state-dir PATH
-cluster-state-dir PATH
-coordinator URL
-run-id ID
-worker-id ID
-invite-output PATH
```

`recover-worker` drives the allocation-recovery application service. It asks
global coordination for an `ExternalRecoveryGrant` and only then asks cluster
trust for the bound internal `external-worker-recovery` invitation. It never
creates an unbound recovery invitation. `cluster-state-dir` is optional when a
remote authority endpoint is available. The recovery grant authorizes a
pending attempt before the replacement process exists and therefore contains
no certificate-request hash. The replacement worker generates its key and
request only after receiving the invitation; invitation consumption binds that
request to the grant.

## Shared Identity And Data Requirements

### Content-stable dataset identity

The current resumable dataset hash includes absolute paths and modification
times, which cannot identify copies of one dataset on different hosts.

Before R1 release, add a distributed dataset identity based on logical content:

```text
SHA-256(
  canonical dataset manifest
  + sorted logical shard names
  + each shard's byte size
  + each shard's SHA-256
)
```

The expensive shard digest may be cached in a sidecar keyed by size and a
local change detector, but the committed distributed identity is content
based. Absolute paths never contribute.

Workers map the same dataset identity to local paths. Dataset copying and cache
population are operational concerns outside this release train.

### Counter-based distributed sampler

The current loader shuffles a process-local stream and resume restores position
by replaying batches. That is insufficient for distributed ownership.

Add a sampler abstraction whose logical state is serializable without replay:

```go
type SamplerState struct {
    DatasetID     string
    Seed          uint64
    Epoch         uint64
    GlobalCursor  uint64
    PartitionID   string
    PartitionSize int
}
```

Requirements:

- For fixed-world DDP, logical samples or shuffle chunks are deterministically
  partitioned without overlap across ranks.
- For fixed-cohort DiLoCo, each learner receives a stable partition.
- Record-oriented datasets partition records; flat token datasets partition
  aligned shuffle chunks.
- No partition crosses record boundaries.
- The sampler reports effective valid tokens and examples for every batch.
- State is stored directly in checkpoints; resume does not replay all previous
  batches.
- Two hosts with different local shard paths but the same dataset identity
  produce the same logical partition.

R5 will replace fixed partitions with leased data ranges. Do not make fixed
rank count part of the dataset file format.

## R1: Fixed-World DDP

### Purpose

Establish the numerical and process-control foundation used by later releases
and provide useful synchronous scaling for workloads whose compute time
dominates gigabit communication.

### Immutable DDP membership contract

The shared distributed-identity kernel defines values conceptually equivalent
to:

```go
type DDPGroupMember struct {
    MemberID string
    Rank     int
}

type DDPGroupMembership struct {
    RunID          string
    GroupID        string
    Generation     uint64
    Backend        string
    OrderedMembers []DDPGroupMember
    MembersHash    string
}

type LocalGroupView struct {
    Membership     DDPGroupMembership
    LocalMemberID  string
    LocalRank      int
    LaunchAttemptID string
}
```

`MembersHash` identifies the complete ordered rank assignment without
requiring data, trainer, or checkpoint code to understand launcher-specific
hostfile formats. `WorldSize` is the validated length of `OrderedMembers`
rather than a separately mutable source of truth. All ranks agree on the
complete `DDPGroupMembership`; only `LocalGroupView` differs.

The launcher supplies a unique canonical `MemberID` for each rank. R1.1 uses
the enrolled node ID plus the assigned process role. An explicit R1 launcher
may derive it from a canonical launcher descriptor for a fresh run. Exact
resume requires the stored ordered member descriptors; if a launcher cannot
reproduce stable member IDs, exact distributed resume is unavailable and must
fail rather than guessing from rank count alone.

R1.1 job manifests provide all membership fields before launch. For a fresh
explicit-launcher R1 run without those fields, strict MLX initialization first
establishes rank/world size, rank zero generates the run/group nonce, and a
bounded startup exchange gathers member IDs and distributes the canonical
membership before trainer or sampler creation. Generation is zero. Exact
resume supplies the stored run/group and ordered membership instead of
generating new values.

Membership is immutable from strict process-group initialization until the
worker exits. Rank failure fails the group. Relaunching the identical ordered
assignment retains its generation and uses a new `LaunchAttemptID`. Replacing,
adding, removing, or reordering a member within the same run creates a new
generation and is outside R1 exact resume. R1 does not shrink, grow, or replace
a rank in a live group.

The group-runtime API returns an opaque group handle plus the authoritative
MLX rank and world size. Its bounded startup identity exchange verifies the
runtime rank-to-member mapping, backend, generation, and members hash against
the expected group membership before the trainer or sampler starts. MLX group
objects are not stored in package-global booleans or rediscovered by downstream
contexts.

### Supported R1 surface

Required:

- Metal plus TCP ring;
- CUDA plus NCCL;
- causal objective;
- AdamW;
- fixed sequence length;
- equal local `batch_tokens`;
- gradient accumulation;
- full replicated model and optimizer state;
- rank-zero checkpoint, validation, logging, and early stop; and
- same-world-size exact resume at an optimizer-step boundary.

Add before declaring broader support:

- MLM/MNTP only after unequal-mask normalization parity tests pass;
- classification only after `valid_mask` normalization parity passes; and
- additional optimizers only after two-rank update parity exists for every
  optimizer group behavior.

Explicitly unsupported in the first R1 implementation:

- combined/multihead losses;
- distillation, data2vec, RTD, invariance, PLL-margin, and other auxiliary
  losses;
- Mamba-3 low-memory/chunked/fused optimizer-update paths;
- dynamic sequence length or weight-layout changes;
- changing world size on resume; and
- multiple outstanding submitted training steps.

Unsupported programs fail during validation, before process-group
initialization where possible.

### C++ ownership

Distributed gradient synchronization belongs in `IRTrainer`, not in Go.
The current Go trainer interface exposes submit/collect operations but not
gradients. Moving full gradients through cgo would add device-host copies and
make collective ordering harder to prove.

Add a C++ context conceptually equivalent to:

```cpp
struct DistributedContext {
  bool enabled;
  std::string backend;
  std::optional<mlx::core::distributed::Group> world;
  int rank;
  int world_size;
  size_t gradient_bucket_bytes;
  int accumulation_steps;
};
```

Use the actual MLX C++ types and optional/handle ownership exposed by the
pinned release; the sketch above is not a requirement that `Group` be default
constructible. Do not wrap MLX collectives in a custom MLX primitive or a long
`custom_vjp`.

### Trainer refactor

Refactor the current fused conceptual operation:

```text
value_and_grad -> sanitize/clip -> optimizer -> transaction
```

into explicit internal stages:

```text
compute local mean loss and gradients
convert local mean gradients to numerator gradients
accumulate numerator gradients and denominator
collective pre-update finite decision
bucketed all-sum of numerator gradients and denominator
divide by global denominator
clip global averaged gradients
compute candidate optimizer state
collective candidate-state finite decision
commit or rollback identically
```

The single-process path must use the same refactored stages with a singleton
context, preserving existing results.

Do not expose gradients through the public Go `GPUTrainer` interface merely to
implement R1.

### Objective normalization contract

Every supported prepared training batch must supply a scalar
`loss_normalizer` whose value exactly matches the denominator used by its loss
op.

Examples:

- fixed causal LM: number of target tokens included in the loss;
- MLM/MNTP: sum of the effective loss mask;
- classification: sum of valid example weights.

The loss remains the local scalar mean for telemetry. Before accumulation, each
gradient is multiplied by `loss_normalizer`. After all-reduce, numerator
gradients are divided by the all-reduced normalizer.

A zero normalizer is a globally coordinated skipped microstep. It is not
replaced with one. If every rank has zero denominator for an optimizer attempt,
the attempt is skipped globally with an explicit reason.

### Gradient accumulation

For `K = gradient_accumulation_steps`:

- calculate `K` local microsteps without applying the optimizer;
- accumulate float32 numerator gradients and a float64 or sufficiently exact
  scalar denominator;
- perform one distributed gradient reduction;
- apply clipping and one optimizer update; and
- advance the learning-rate schedule once.

`training.steps` counts optimizer attempts, not microsteps. Add separate
telemetry counters for microsteps and effective tokens.

Accumulation buffers are cleared after either a committed or globally skipped
optimizer attempt.

### Bucketing

- Use deterministic ordered weight metadata as bucket order.
- Separate buckets by reduction dtype.
- R1 reduces gradients in float32.
- Default target bucket size is 32 MiB.
- Flatten/concatenate compatible adjacent gradients, perform one `all_sum`, and
  restore views/slices without changing weight order.
- Log total gradient bytes, bucket count, all-reduce duration, and effective
  bandwidth.
- Bucket boundaries are identical across ranks and included in startup
  agreement diagnostics.

One collective per weight is not an acceptable released implementation.

### Collective transaction

Each optimizer attempt follows this order:

1. Compute all local microsteps.
2. Produce a local scalar `pre_update_bad` covering non-finite loss,
   gradients, denominator, or backward diagnostics.
3. `all_max(pre_update_bad)`.
4. If bad globally, every rank discards accumulation and records the same
   skipped optimizer attempt. No gradient-bucket collective is issued.
5. Otherwise all-sum denominator and gradient buckets in canonical order.
6. Compute global gradient norm and clip after reduction.
7. Build candidate weights and optimizer state transactionally.
8. Produce local `candidate_bad`.
9. `all_max(candidate_bad)`.
10. Commit on every rank only when the global result is good; otherwise roll
    back on every rank.

If a rank throws before it can participate in the next required collective,
the process must terminate and cause the launcher to fail the world. Continuing
with a partial world is forbidden in R1.

### Initialization agreement

After weights and optimizer state are initialized but before the first batch:

- compare run, group, membership-generation, ordered-member/members-hash, and
  group-protocol/backend/world metadata;
- compare config, IR, weight-layout, optimizer, and dataset hashes;
- verify initial parameter checksums;
- verify canonical bucket metadata; and
- verify all ranks report the same scheduled training phase.

Use collectives to report mismatches coherently. Errors include rank and the
first mismatching field.

### Data and RNG

- Rank-specific sampler partitions are disjoint.
- Dropout and objective RNG keys include rank and accumulation microstep.
- Initial parameter keys do not include rank.
- Validation uses rank zero only and retains current deterministic behavior.
- Early-stop and phase decisions are encoded by rank zero into a scalar or
  fixed-size control tensor and distributed to every rank.

### Scheduling and lookahead

Disable the current submit-next-step-before-collect optimization in distributed
mode. Restore it only after a dedicated test proves identical collective order
under every phase, objective, and error transition.

All program-cache transitions, QAT transitions, recurrence activation, and
head changes are either rejected in the first R1 support matrix or driven by
the shared optimizer-attempt counter.

### DDP checkpoint format

Add a new distributed resume manifest rather than silently changing
`mixlab_resume_v1`.

The manifest contains:

- global optimizer attempt and committed step;
- run ID, group ID, membership generation, members hash, ordered member/rank
  assignment, world size, and backend;
- the launch-attempt ID that produced the checkpoint, for audit only;
- local microbatch and accumulation configuration;
- effective global token counters;
- config/program/weight/optimizer/dataset hashes;
- sampler state sufficient for direct resume;
- full replicated model and optimizer state;
- early-stop and scheduler state; and
- exact-supported-world-size metadata.

Rank zero writes the full model and optimizer state. Other ranks do not write
duplicate tensors. All ranks synchronize before and after publication.

R1 resume requires the same group membership generation, ordered member/rank
assignment, backend kind, local batch size, and accumulation count. The resumed
processes use a new launch-attempt ID. A topology mismatch is rejected. The
replicated model artifact may still be used as an explicit weights-only warm
start in a different topology.

### R1 acceptance

Automated:

- Omitted distributed config preserves existing single-process tests.
- Config validation rejects unsupported combinations.
- World size one is rejected for `mode=ddp`.
- Ring startup succeeds without a synthetic `MLX_WORLD_SIZE` when the
  initialized hostfile group has two ranks.
- Expected DDP membership that disagrees with MLX rank, world size, backend,
  ordered startup identities, generation, or members hash fails before
  trainer/sampler creation.
- Relaunching the same ordered members changes launch-attempt ID without
  changing membership generation; changing one member requires a new
  generation and cannot exact-resume an R1 checkpoint.
- A dependency test keeps discovery, node-agent, and sampler code outside the
  MLX group-runtime and C++ trainer boundaries.
- A deterministic two-rank ring test matches a single-process AdamW update over
  the same global causal batch within `1e-5` max absolute parameter error.
- Accumulation `K=4` matches the equivalent unaccumulated global batch within
  the same tolerance.
- An unequal-denominator fixture verifies numerator-weighted reduction before
  MLM/MNTP is enabled.
- Injecting a NaN on exactly one rank causes both ranks to skip and retain
  identical parameters and optimizer counters.
- A rank/program hash mismatch fails before the first training collective.
- Same-world-size checkpoint/resume matches uninterrupted distributed training
  within the normal resume tolerance.
- No test relies on sleeps for correctness; process tests use bounded
  readiness and completion signals.

Manual hardware:

- M1 Max and M4 Max complete a two-rank TCP-ring causal run over gigabit
  Ethernet.
- A CUDA host completes a two-rank NCCL run.
- Logs report compute, wait, collective, effective bandwidth, and global
  tokens/sec separately.

Review gate:

- Inspect update parity and global skip evidence.
- Inspect Ethernet communication measurements.
- Inspect the immutable membership/group-runtime API before R1.1 builds its
  launcher integration.
- Confirm unsupported paths fail early.
- Do not begin R2 until the refactored single-process path has no material
  regression and R1 checkpoints recover correctly.

## R1.1: Daemon Discovery And Fixed-Cohort Recruitment

### Purpose

Allow users to enroll lab machines once, leave mixlab running in the
background, inspect currently available capacity, and recruit compatible
machines for a job without logging into each host.
Enrollment may be zero-touch inside an explicitly trusted LAN, unattended with
a one-use provisioning file, or interactive with human-verified phrases.

R1.1 separates dynamic discovery from dynamic training membership:

```text
continuously discover authenticated available nodes
                  |
                  v
     reserve and prepare compatible nodes
                  |
                  v
 produce one immutable assignment proposal
                  |
                  v
 commit fixed R1 membership or enter R2 run admission
```

Nodes may appear or disappear before reservation. Once a job launch commits,
R1 or R2 failure semantics apply. R1.1 does not add rank replacement, world
resize, DiLoCo quorum changes, or late learners.

### Self-managed cluster trust

R1.1 implements a private mixlab trust domain inside the mixlab binary. It
uses TLS 1.3 and a private X.509 hierarchy generated with the language/runtime
cryptography library. Users do not generate certificates, run a CA daemon,
install a VPN, modify an operating-system trust store, or invoke an external
cryptography command.

`cluster-init` creates:

- a random cluster ID;
- a long-lived cluster root key and self-signed root certificate;
- distinct active certificate-issuer and trust-snapshot-signing identities
  authorized by that root;
- the first authority TLS, controller, and coordinator principals; and
- versioned issuance, renewal, and trust-snapshot policy.

The cluster root public key is also its stable human-verifiable identity:

```text
root_spki_v1       = canonical DER SubjectPublicKeyInfo(root public key)
cluster_fingerprint = SHA-256(root_spki_v1)
cluster_phrase_v1   = first 88 fingerprint bits as eight 11-bit indexes
                      into the embedded mixlab_wordlist_v1
```

Every UI that displays the phrase also makes the full lowercase hexadecimal
fingerprint available. The word list and bit ordering are immutable versioned
protocol data with test vectors. The phrase and fingerprint are public,
deterministic, and confer no enrollment or application authority. Root
rotation changes both and requires an explicit re-trust/re-enrollment
ceremony; issuer, snapshot-signer, or leaf renewal does not.

The initial controller and coordinator credentials are written as separate
principal-state records containing only their own private-key handles and
public trust material. Their processes do not inherit authority keys.

Cluster initialization also signs an `AuthorityEndpointSet` containing the
configured trust audience and one or more endpoints. Copies live in each
principal state and later trust snapshots. An mDNS authority advertisement
uses `_mixlab-trust._tcp.local.` only as an address hint; the client accepts an
endpoint solely when its `authority` certificate and audience match the
root-signed endpoint set.

The root, certificate issuer, and snapshot signer are files/handles in local
cluster authority state; there is no separately deployed CA service. The same
binary may expose the trust-lifecycle routes transiently or as part of
`submit`/run admission. On macOS the default secure-key-store adapter uses
Keychain when available and stores its opaque handles under the portable
principal/authority layout. On Linux, including headless CUDA/RunPod systems,
the default no-extra-service adapter uses owner-only atomic files beneath
`~/.mixlab`; the same strict-file adapter is the macOS fallback. Both reject
unsafe ownership or permissions, symlinks, hard-link substitution, and path
traversal. Losing or compromising every copy of the root, certificate issuer,
or trust-snapshot signer is an explicit cluster-compromise/rekey event; daemons
never reconstruct or silently replace one.

The cluster-trust application exposes `TrustAuthorityClient` and
`TrustSnapshotDistributor` ports. A composition root selects a local
in-process authority service when it was given `cluster-state-dir`, a remote
mutually authenticated client resolved from the signed authority endpoint set,
or a deterministic fake for tests.

`nodes` and `submit` accept an exact `cluster-state-dir` optionally. When it is
present, or when exactly one applicable authority directory resolves beneath
the state home, they may host the local authority service for the command
lifetime; otherwise they use the remote client. An ambiguous local match fails
before opening any authority state. Workload issuance, renewal, and new
trust-snapshot publication fail with a distinct authority-unavailable error if
neither local nor remote authority is available. They never fall back to
copying issuer keys into controller or coordinator state. A continuously
running `trust-authority` mode is an optional convenience, not a separately
installed dependency.

R1.1 certificates use versioned, purpose-separated mixlab profiles. Every
principal or workload certificate contains:

- cluster ID;
- random stable principal ID;
- exactly one role: `authority`, `controller`, `node`, `coordinator`, or
  `worker`;
- random serial number;
- issue and expiry times;
- appropriate TLS client/server and signing key usages; and
- exactly one canonical URI subject alternative name of the form
  `urn:mixlab:principal:<cluster-id>:<role>:<principal-id>`.

The certificate-issuer and trust-snapshot-signer certificates are separate
CA-purpose identities, not application principals or roles. The issuer has
only the certificate-signing usages required by the profile. The snapshot
signer has only trust-snapshot signing usage. An `authority` principal is a TLS
client/server identity for the authority endpoint and cannot sign
certificates, trust snapshots, or application manifests. No TLS principal key
is also a CA-purpose key.

The v1 cryptographic suite uses Ed25519 for root, issuer, snapshot signer,
principal, workload, and manifest signatures; SHA-256 for content and binding
digests; TLS 1.3 with the vetted TLS implementation's standard AEAD suites for
connections; at least 256 bits from the operating-system CSPRNG for keys and
provisioning secrets; and at least 128 bits for nonces and certificate
serials. TLS signing keys are not reused for application-layer envelope
encryption. A node principal separately
registers an X25519 envelope public key signed as part of its principal
credential. `CredentialEnvelope` uses RFC 9180 HPKE with
DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, and ChaCha20-Poly1305 through a vetted
library compiled into the binary. Hand-implemented cryptographic primitives or
shelling out to a cryptography tool are forbidden.

A workload certificate carries a critical versioned mixlab binding extension.
All workloads bind cluster, role, principal/participant, run, job,
launch-attempt, audience, issue, and expiry values. A DDP workload additionally
binds group ID, membership generation/hash, member ID, and rank. A DiLoCo
worker additionally binds worker ID, staged run-plan hash, credential
generation, and active/pending incarnation scope. An R3 pending-recovery
workload also binds its recovery-grant digest. Missing, duplicate,
unknown-version, or semantically mismatched binding extensions fail certificate
validation before an application handler or MLX process starts.

DNS names, IP addresses, display names, and common-name strings are routing or
presentation data only. They never replace the canonical principal identity.
Certificate and trust-policy lifetimes are versioned constants with fake-clock
tests. Cluster trust owns renewal eligibility and issuance; it does not own a
daemon or process scheduler. Each long-lived composition owns a
`PrincipalRenewalScheduler` trigger using its local principal state and a
`TrustAuthorityClient`. A node daemon starts it during startup/background
operation; controller and coordinator compositions start the same scheduler
while long-running. It attempts renewal when one third of certificate lifetime
remains, with deterministic principal-derived jitter and five minutes of clock
skew allowance, then uses bounded exponential retry until expiry. Principal
state persists certificate expiry, signed authority endpoint set, and last
attempt/outcome, but no issuance policy or authority key.

Renewal requires the still-valid existing principal and role over mutual TLS.
If renewal misses expiry, a node advertises as non-recruitable and every
long-lived role fails closed; recovery requires a new purpose-specific
principal-enrollment approval under one of the policies permitted for that
purpose. There is no anonymous or automatic post-expiry recovery. Workload
certificates are shorter lived than principal
certificates, cannot outlive the bound job plus its cleanup allowance, and are
renewed only by their owning admission/recovery workflow rather than this
generic scheduler. There is no verification-disable flag.

The trust context publishes signed, monotonic, expiring `TrustSnapshot` values
containing the snapshot-signing certificate chain, active certificate issuers,
role eligibility, authority endpoints, and revoked principal IDs or
certificate serials with `prospective` or `compromise` mode and reason. Every
snapshot is signed only by the distinct trust-snapshot signer and embeds its
chain to the cluster root; an authority TLS principal can never sign one. It
never contains a node-operation, lease, run, update, or artifact grant. A
daemon stores only a newer valid generation. Revocation is pushed to reachable
daemons immediately. Before an offline or stale daemon becomes recruitable or
accepts a new lease, it must synchronize a fresh snapshot-signer-signed
snapshot. This makes the revocation
freshness/availability tradeoff explicit without requiring a permanently
running CA service.

The cluster root public certificate/fingerprint/phrase, principal
certificates, enrollment window/request/approval public metadata, consumed
invitation IDs, and trust snapshots are not secrets. A complete unconsumed
provisioning file is secret because it contains the bearer authorization value.
Root, issuer, principal, provisioning-secret, envelope, and workload private
material is secret and follows the redaction/protected-storage rules below.
Provisional TLS-exporter/channel evidence is sensitive ephemeral state: it is
never logged or persisted after the request terminates.

### Enrollment and principal identity

R1.1 exposes three user-facing approval policies:

| Policy | Root authentication and approval | Intended use |
|--------|----------------------------------|--------------|
| `trusted_lan` ("Easy") | Client explicitly accepts the first root presented by one discovered/selected endpoint; cluster trust auto-approves a conforming `node` request inside a bounded administrator-opened window. | Isolated, trusted lab networks where convenience is worth first-enrollment TOFU risk. |
| `provisioned` ("Provisioning file") | A protected, one-use provisioning file pins the root/endpoint and authorizes one purpose and role; no live source operator is required. | Manual distribution followed by unattended auto-enrollment, privileged roles, and external workers. |
| `verified` ("Verified/manual") | A user compares the stable cluster phrase out of band; both sides then compare a request-specific phrase and the source operator approves the exact request. | Fully manual interactive enrollment without moving a secret file. |

The canonical protocol value is `trusted_lan`; the CLI spells it
`trusted-lan`. The other names are identical on wire and CLI.

The normative user journeys are:

1. **Easy:** the source administrator opens a short trusted-LAN window and sees
   its interface, allowed CIDRs, expiry, remaining node count, TOFU warning,
   and cluster phrase. A user who explicitly starts `cluster-enroll` with the
   `trusted-lan` policy discovers the sole source and receives a node identity
   without per-request source approval.
2. **Provisioning file:** the source administrator creates a distinct
   single-use file for each machine and transfers it through an appropriately
   protected channel. The remote user supplies that file to `cluster-enroll`;
   enrollment completes without a simultaneously present source operator.
3. **Verified/manual:** the source administrator opens a verified window and
   communicates its stable cluster phrase out of band. The remote user
   discovers or names the source, compares that phrase, and submits a locally
   generated key. Both users compare the request phrase, after which the source
   administrator approves the exact pending request.

These flows are composition and presentation behavior. The commands do not
implement their own trust decisions: they pass selected policy, transport
evidence, and explicit confirmations to cluster trust.

These are approval strategies inside cluster trust, not separate identity
systems. Every approved principal receives the same certificate profile,
renewal/revocation behavior, trust snapshot, and later mutual-TLS enforcement.
The common durable state machine is:

```text
PENDING -> APPROVED -> ISSUED
        -> REJECTED
        -> EXPIRED
```

Cluster trust owns the state and transition rules. The bootstrap transport
supplies channel/network evidence, the discovery provider supplies candidate
addresses, and the CLI supplies human input; none can issue a certificate or
turn proposed display metadata into a principal identity. An enrollment
request grants no node, lease, run, artifact, or workload operation.

Every approval produces an immutable `EnrollmentApprovalRecord` containing the
request/window ID, selected policy, cluster/root fingerprint, endpoint
audience, requested purpose/role, certificate-request hash, policy-specific
evidence digest, approval journal sequence, expiry, authority-principal
`TrustEvidence`, and signature. For principal enrollment it is consumed
internally by issuance. For external bootstrap it is the opaque trust evidence
that run admission consumes before asking for a workload credential. Neither
the enrollment source composition nor admission can manufacture or reinterpret
it. For `provisioned`, its policy-evidence digest binds the corresponding
`EnrollmentConsumptionReceipt`; external bootstrap/recovery consumers receive
both values and must validate their cross-reference. The approval record uses
a closed authority `PrincipalSigner` approval-record purpose, never the root,
certificate issuer, or snapshot signer.

#### Trusted-LAN auto-enrollment

Opening `trusted_lan` creates a durable `EnrollmentWindow` containing a random
window ID, exact enrollment endpoint/audience, `node` as the only allowed role,
creation/expiry time, positive maximum enrollment count, exact receiving
interface, allowed source CIDR set, and audit principal. The default TTL is ten
minutes; the v1 maximum is one hour. The source refuses an empty/unbounded CIDR,
an unspecified interface, an externally routed/public policy, or a non-positive
maximum count. Closing, expiry, authority restart after expiry, or reaching the
count makes the window unusable.

The client discovers or explicitly selects an enrollment endpoint, performs the
dedicated provisional TLS 1.3 handshake, obtains the proposed root/endpoint
chain, displays and records its cluster phrase, and explicitly applies
first-connection TOFU because the user chose `trusted_lan`. It never derives
trust from the mDNS record itself. After pinning that root it validates the
source certificate chain, coordinator role, and endpoint audience. The client
then generates its key locally and sends proof of possession plus a `node`
certificate request. The transport
supplies the server with the observed peer address and receiving interface;
cluster trust auto-approves only when they satisfy the active window, endpoint,
role, quota, rate, and duplicate-key rules.

If discovery returns more than one enrollment source, the client fails and
requires an explicit endpoint. If principal state already contains a root,
credential, or principal identity, the client fails rather than replacing it.
Provisional state remains in a separate temporary location and is erased on
failure. Failure never falls through to another approval policy. `trusted_lan`
cannot issue `authority`, `controller`, `coordinator`, or external-worker
credentials and is forbidden for public-internet/RunPod bootstrap.

This is the specification's sole TOFU exception. It changes only bootstrap
assurance; it does not permit plaintext operation, anonymous renewal, automatic
re-enrollment after expiry/revocation, or relaxed application authorization.

#### Provisioned enrollment

An internal `EnrollmentInvitation`, presented to users as a provisioning file,
is a canonical protected file containing:

- invitation format version and random invitation ID;
- cluster ID, complete root certificate, full fingerprint, and phrase version;
- purpose-specific endpoint URL/audience, invitation purpose, and allowed
  resulting principal or workload role;
- a cryptographically random high-entropy one-time secret;
- issue and expiry times; and
- an explicit one-use limit.

The authority stores only the verifier/hash needed to consume it. A provisioning
file never contains a cluster, issuer, or controller private key. Invitation
purpose is distinct from certificate role. Node, controller, and coordinator
purposes may issue only the matching long-lived principal role.
`external-worker-bootstrap` reserves/consumes a bootstrap claim but issues no
principal certificate before a staged run binding exists;
`external-worker-recovery` may issue only a replacement workload certificate
for the same already committed external participant/allocation under the R3
recovery rules. Normal files are single-purpose and single-use. Batch
enrollment creates distinct files; a reusable lab-wide bearer token is
forbidden.

Provisioned enrollment proceeds as follows:

1. The new process loads the protected file, pins its root, and requires the
   exact purpose-specific endpoint/audience before connecting.
2. It generates its private key locally and sends a certificate request plus
   proof of the one-time secret over server-authenticated TLS.
3. For principal enrollment, the authority validates and atomically consumes
   the file, purpose, role, and request before issuing the matching
   certificate.
4. For external bootstrap/recovery, the coordinator external-preflight handler
   invokes `TrustAuthorityClient`; cluster trust returns a signed single-use
   consumption receipt plus the corresponding approval record. Neither is a
   workload credential; admission/recovery must still supply its context-owned
   run-plan or recovery binding before workload issuance.
5. The client validates the returned chain/identity/snapshot, stores them
   atomically, and destroys the file secret and temporary state.

An `EnrollmentConsumptionReceipt` contains the invitation ID/purpose, cluster
ID, exact endpoint audience, certificate-request hash, opaque
admission/recovery binding digest, trust-snapshot generation, consumption
journal sequence, authority-principal `TrustEvidence`, and signature. Cluster
trust creates it only after the atomic one-use transition. It is signed through
the closed authority `PrincipalSigner` receipt purpose, never by the
certificate issuer or snapshot signer. Admission/recovery may validate and
consume this evidence but cannot issue or reinterpret it. Replay, concurrent
double consumption, wrong-purpose use, and a server outside the pinned root/
audience fail closed.

#### Human-verified enrollment

The source opens a durable `verified` window containing its random ID, exact
endpoint/audience, explicit purpose/role allowlist, TTL, maximum approvals,
pending-request/rate bounds, and audit principal, then displays the stable
cluster phrase and full fingerprint. Unlike `trusted_lan`, it need not be
CIDR-local because every exact request requires human approval. The client
discovers its endpoint through mDNS when local or uses an explicit routed URL.
mDNS fields, including any claimed fingerprint, are presentation hints only.

The client first completes the dedicated provisional TLS 1.3 handshake and
reads the proposed root and source certificate. Before generating or sending a
certificate request, its local cluster-trust application derives the root
phrase and the presentation adapter asks the remote user to compare it with
the source through an independent channel such as voice, video, or an
in-person display. Rejecting the comparison destroys the connection and
provisional state. Accepting it pins the root, validates the source certificate
chain/role/audience against that root, and permits the client to generate its
private key locally.

After proof of possession and certificate-request submission, cluster trust
creates a pending request with a random request ID. The v1 request phrase is:

```text
request_context = SHA-256(canonical(
  protocol version, cluster fingerprint, endpoint audience, requested purpose
  and role, request ID, certificate-request hash, client nonce, server nonce
))
channel_binding = TLS-Exporter(
  "EXPORTER-mixlab-enrollment-v1", request_context, 32 bytes
)
request_phrase_v1 = first 55 bits of SHA-256(
  "mixlab-enrollment-sas-v1" || channel_binding || request_context
) as five mixlab_wordlist_v1 words
```

Zero-RTT, provisional TLS session resumption, exporter reuse, and moving a
request between connections are forbidden. The client and source UI render the
same request ID/phrase. The source also shows requested purpose/role, proposed
display name, observed network metadata, and full certificate-request
fingerprint; every field except the cryptographic bindings is labeled
untrusted. The remote user confirms that its phrase matches, and the source
operator approves that exact pending request. Cluster trust records both
confirmations before issuing. A phrase is public comparison data, not a bearer
secret; knowledge of it cannot approve another request.

Verified enrollment defaults to `node`. A locally present administrator may
explicitly open a window for a listed privileged purpose, but the choice and
warning are part of the signed/audited request; it never inherits
`trusted_lan` auto-approval. R3 may use the same verified mechanics for an
`external-worker-bootstrap` receipt, after which run admission still owns the
workload binding.

For every policy, an approved principal enrollee validates the returned chain,
cluster ID, role, principal ID, and initial trust snapshot before atomically
storing its key handle and public trust state. Controller enrollment and
external-worker bootstrap/recovery use the same trust context with distinct
purposes; they do not reuse the node-agent lease/job API.

Trust lifecycle routes are versioned separately from node-agent and coordinator
APIs:

```text
POST /v1/trust/enrollments/{invitation_id}/consume
POST /v1/trust/enrollment-requests
GET  /v1/trust/enrollment-requests/{request_id}
POST /v1/trust/enrollment-requests/{request_id}/client-confirmations
POST /v1/trust/principals/{principal_id}/renew
POST /v1/trust/workloads
GET  /v1/trust/snapshots/latest
PUT  /v1/trust/snapshots/{generation}
```

The invitation-consumption route is server-authenticated against the
provisioning-file-pinned root and authorizes exactly one matching request. The
request/create, request/read, and client-confirmation routes use only the
dedicated, rate/size/deadline-limited provisional enrollment transport and can
create, observe, confirm, reject, or receive the result of that one request;
they expose no general trust or application operation. A request ID is not
authorization: every follow-up is bound to the same live
`EnrollmentChannelBinding`; connection loss expires the provisional request
and a retry creates a new request/phrase. Approval is not available to any
network client: the local source UI submits its exact operator decision
through an in-process `AdministratorApprovalPort`, and cluster trust applies
the selected policy. A future remote-administration protocol requires a
separate design and must not reinterpret a coordinator certificate as
enrollment-administrator authority.

Renewal is mutually authenticated and accepts only the same principal and role
before expiry. Snapshot publication accepts only a valid newer value signed by
the dedicated trust-snapshot signer and chaining directly to the pinned root.
Workload issuance is mutually authenticated,
requires an authorized controller/coordinator role plus a signed context-owned
job/run binding, and can issue only a short-lived matching workload
certificate. These handlers invoke cluster-trust application services; they
cannot inspect or mutate node leases, jobs, memberships, coordinator runs, or
artifacts. Node-agent and coordinator transports consume their results through
trust ports rather than reimplementing certificate policy. The `GET` route
reads the current public signed snapshot from an authority service; the `PUT`
route installs it at a daemon or other principal state.

Snapshot refresh has a deliberate stale-trust recovery path. Even when its
local snapshot is expired, a daemon keeps a narrowly scoped trust listener that
accepts only:

- a TLS 1.3 request on the dedicated trust-snapshot route; and
- a newer `TrustSnapshot` for the same cluster whose dedicated
  snapshot-signer chain/signature and root-signed endpoint/audience binding
  validate directly against the daemon's pinned root without consulting the
  expired snapshot.

Client authentication is not required for this one route because the snapshot
is public and its distinct snapshot-signer signature is the authorization; the
endpoint's TLS key supplies transport/routing identity only and can never sign
the snapshot. Accepting only a newer valid generation makes replay
non-mutating. The route has strict size, rate, and deadline limits.

That path cannot query capabilities, issue or renew credentials, acquire or
renew a lease, prepare/start a job, read logs/artifacts, or change job
authority. It exists only to install the newer snapshot atomically.

While trust is stale, the daemon advertises and reports itself as
non-recruitable and rejects all new lease, prepare, start, and authority
operations. For an already bound job it may accept status, cancel, and release
from the exact previously recorded controller/coordinator certificate until
that certificate or lease expires; it does not extend the lease. Snapshot
expiry by itself does not tear down an already authenticated workload channel,
but installing a snapshot that revokes the active node/workload fails and
cleans up the job. After refresh, all normal connections are revalidated
against the new generation. Authority outage therefore reduces availability
without triggering new TOFU, anonymous recovery, raw-socket fallback, or loss
of the ability to cancel stale work.

A controller, coordinator, or worker with a still-valid certificate but stale
local snapshot may fetch the latest public snapshot from an authority endpoint;
the authority authenticates it against current trust before serving any
renewal or workload operation. It performs no application operation until
refresh succeeds. An expired or currently revoked principal cannot use refresh
as renewal or reinstatement and requires the explicit enrollment/recovery
policy applicable to its role.

Every daemon has:

- a stable random node ID;
- a persistent private key and cluster-signed `node` certificate;
- a cluster ID and latest accepted trust-snapshot generation;
- a human-readable display name that is not used as identity; and
- a durable state directory with owner-only permissions.

The node ID equals the node principal ID. Node management authorizes a
controller from the `AuthenticatedPrincipal`, current trust snapshot, requested
operation, and affected lease/job; it does not parse certificates or issue
identities. Enrollment is an explicit one-time cluster action: either the
administrator opens a bounded auto window, supplies one provisioning file, or
approves one verified request. A
discovered but unenrolled, expired, stale-trust, or revoked daemon is never
considered available capacity.

The daemon runs without root privileges. Platform service installation may
start it at login, but the same binary and protocol work in the foreground.

### Discovery provider

The enrollment source composition advertises itself as a cluster enrollment
coordinator using:

```text
_mixlab-enroll._tcp.local.
```

Its bounded TXT data contains the enrollment protocol version, endpoint port,
source role/audience, claimed cluster ID/root fingerprint, allowed policy
labels, and optional display name. These values let a user distinguish
candidates but are not proof. The client-side cluster-trust application derives
the identity phrase from the root actually presented by the endpoint and
applies its selected enrollment policy. The discovery adapter returns
candidate addresses to the enrollment composition; cluster trust never browses
mDNS.

An enrolled node daemon uses:

```text
_mixlab._tcp.local.
```

The advertisement is deliberately small:

- discovery protocol version;
- node ID or public-key fingerprint;
- cluster ID;
- control API port; and
- optional display name.

Do not advertise dataset names, filesystem paths, job details, credentials,
device inventory, or bearer tokens in mDNS TXT records.

mDNS output is an untrusted address hint. The controller connects to the
candidate, authenticates its node identity, verifies cluster enrollment, and
then obtains capabilities over the node-agent API. Duplicate, malformed,
spoofed, stale, or wrong-cluster advertisements are ignored.

Discovery is represented behind a narrow provider interface so routed
networks can later use explicit addresses or a coordinator-backed registry
without changing leases, job manifests, or the node-agent API. R1.1 requires:

- mDNS browse and advertise for enrollment sources and enrolled nodes on the
  local link;
- an `off` mode with explicit configured addresses for tests and networks
  where multicast is blocked; and
- injectable fake discovery for deterministic automated tests.

mDNS is link-local and need not cross multicast domains. The same enrollment
protocol works with an explicit routed coordinator URL; MLX ring connectivity
is separately determined from authenticated routable endpoints after
enrollment.

R1.1 does not promise discovery across VLANs, VPNs, or the public internet.

### Authenticated capabilities

After authentication, the daemon reports a versioned capability document
containing at least:

- node ID, display name, OS, architecture, and daemon protocol version;
- mixlab version, commit/build ID, and supported job-manifest versions;
- MLX runtime/build version;
- device kind, name, and available device/unified memory;
- available distributed backends;
- supported compute dtypes and custom-op capability identifiers;
- current availability and lease state;
- resource limits, including maximum job/artifact size;
- locally registered dataset IDs and logical selectors; and
- monotonic capability generation and observation time.

The scheduler matches explicit job requirements. It does not infer that two
different build IDs, custom-op sets, or dataset IDs are compatible.

Capabilities are revalidated during job preparation. A discovery-time result
is not sufficient if the daemon changed before the lease was acquired.

### Exclusive reservation leases

One daemon accelerator may be owned by at most one active lease unless a
future resource-partitioning design explicitly allows otherwise.

Lease identity includes:

- lease ID and idempotency key;
- controller ID;
- optional successor coordinator ID for an R2 job;
- proposed run ID;
- node ID and capability generation;
- creation, expiry, and renewal deadline; and
- state-machine version.

The daemon lease state machine is:

```text
AVAILABLE
  -> RESERVED
  -> PREPARED
  -> RUNNING
  -> RELEASING
  -> AVAILABLE

RESERVED or PREPARED -> EXPIRED -> RELEASING -> AVAILABLE
RESERVED or PREPARED or RUNNING -> FAILED -> RELEASING -> AVAILABLE
```

Lease acquisition is compare-and-swap and idempotent. Concurrent controllers
cannot both reserve the same accelerator. Expired reservations are released
only after any child has exited or been terminated. `FAILED`, `EXPIRED`, and
`RELEASING` are recoverable cleanup states, not permanent accelerator owners.
A daemon restart loads durable lease/job state, reconciles any surviving child,
and must never launch a second worker merely because in-memory state was lost.

Lease and job are separate aggregates joined by immutable lease/job IDs. A
lease may enter `RUNNING` only for its one prepared job; a running child keeps
the lease non-available; and terminal job reconciliation drives the lease
through `RELEASING`. Job status cannot independently mark an accelerator
available.

For R1, the submitting controller remains the lease owner. For R2, preparation
records an authenticated successor coordinator and grants it narrowly scoped
renew/cancel authority that becomes usable when the worker starts. The
submitting controller retains compensation/cancel authority until the R2 run
commit exists. The run commit then makes the coordinator the primary renewal
and terminal-release authority. Per-daemon authority finalization is
idempotent and may complete after coordinator restart; it is not falsely
described as one atomic transaction across independent daemons.

### Cohort selection and two-phase launch

Recruitment is an all-or-abort saga with compensating release, not a distributed
atomic transaction. The submitting controller:

1. Browses candidates and authenticates their node-agent endpoints.
2. Queries live capabilities and filters against the job requirements.
3. Selects the requested number of nodes using a deterministic, logged policy.
4. Acquires a lease from every selected daemon.
5. Proposes ordered ranks or learner IDs.
6. For direct R1 DDP, creates one immutable `DDPGroupMembership`.
7. Creates and signs one canonical R1 node-job manifest per assigned process,
   with every manifest bound to the same assignment and launch-attempt ID.
8. Sends `prepare` to every daemon.
9. Each daemon generates a per-job transport key locally and returns a
   certificate request bound to its node-job manifest hash.
10. The cluster-trust issuer signs one short-lived DDP workload certificate
    per prepared member. The launcher/rendezvous adapter constructs one
    canonical `SecureGroupTransportPlan` mapping the immutable membership to
    those certificate fingerprints and job-scoped endpoints; the controller
    application signs the canonical bytes through its principal-signer port.
11. The controller activates that plan at every daemon and receives the same
    membership, manifest, and transport-plan hashes from all of them.
12. Starts R1 workers only after every daemon acknowledges the complete
    bindings.

For R2, recruitment stops after step 5 and returns proposed `node_agent`
learner slots plus reservation references to run admission. Run admission must
first obtain a staged coordinator run-plan hash; only then does it construct
plan-bound node-job manifests, ask the node-agent provider to prepare
`PreparedLearnerAllocation` values, and start learners as specified by R2.

Preparation verifies local dataset identity, artifact availability, free
resources, mixlab/MLX compatibility, backend support, writable job directory,
manifest signature, authenticated controller authorization, and current trust
snapshot without initializing MLX. Key generation and certificate-request
creation do not initialize MLX.

For direct R1, if any reservation or preparation fails, the controller aborts
every prepared job and requests release of all leases. Cleanup is complete only
when every daemon acknowledges release or its lease reaches a reconciled
terminal cleanup state. If start partially succeeds, the controller cancels
the attempt; any worker that cannot form its fixed group exits through R1
bounded startup failure. For R2 the run-admission context owns the equivalent
cross-provider compensation described below. A retry always has a new
launch-attempt ID. It retains membership generation only if the complete
ordered assignment is unchanged; any changed assignment requires a new
generation.

R1.1 does not automatically restart a failed training run from checkpoint.
That policy remains explicit orchestration until fixed-world recovery evidence
exists.

### Structured job manifest

Format identifier: `mixlab_node_job_v1`.

The canonical signed manifest contains:

- node-job format version, job ID, launch-attempt ID, run ID, and membership
  generation;
- group ID plus complete DDP membership, or worker ID plus learner-assignment
  and staged run-plan hash;
- ordered node IDs and rank or learner assignments as applicable;
- backend and expected world size where applicable;
- config content hash and immutable config artifact or embedded canonical
  config;
- program, weight-layout, optimizer-plan, and dataset identities when already
  known;
- logical dataset selector, never an arbitrary controller filesystem path;
- optional checkpoint/artifact references with size and SHA-256;
- allowed mixlab mode and typed mode parameters;
- CPU, memory, disk, runtime, and log-size limits;
- controller identity, optional successor coordinator identity, credential
  binding metadata without secret material, creation time, expiry, and nonce;
- the expected secure-transport mode and transport-plan binding rules; and
- controller `TrustEvidence` and manifest signature.

The manifest has no shell command, setup script, post-command, environment
map, dynamic library path, or arbitrary argument vector. The daemon constructs
an allowlisted child environment and invokes a known mixlab worker mode.

### Launcher boundary

The node agent passes a typed `LaunchPlan` to the launcher/rendezvous adapter;
node-agent lease and supervision code does not encode MLX variables itself.
The adapter materializes backend-specific inputs in a private per-job
directory. Ring workers receive the same canonical MLX hostfile and expected
DDP membership. NCCL workers receive the same rendezvous/world metadata plus
their assigned rank and device. The child verifies the initialized MLX group
against the manifest before loading training data.

`LaunchPlan` is a validated in-process value derived from a signed node-job
manifest. It names a known worker mode and binary/build identity, immutable
config/data/artifact references, DDP membership or learner slot, resource
limits, job directory, credential-file path, workload-identity handle,
secure-transport-plan hash, and launch-attempt ID. It contains no
caller-supplied environment map, raw argument vector, shell text, dynamic
library path, or executable path. The launcher adapter owns the fixed mapping
from this value to an allowlisted child environment and arguments.

### Secure collective transport

Every R1.1-launched multi-host MLX ring uses a same-binary secure collective
transport adapter. The adapter may be an internally supervised helper process
or an in-process launcher component, but it is not a separately installed
service. It exposes only loopback endpoints to the raw MLX ring process and
owns every connection that crosses a host boundary.

Format identifier: `mixlab_secure_group_transport_v1`.

The canonical signed plan includes:

- run, group, generation, membership hash, and launch-attempt ID;
- transport protocol version and TLS policy version;
- for each ordered member: member ID, rank, node principal ID, job ID,
  workload-certificate fingerprint, and bounded transport endpoint;
- creation time, expiry, signer principal, and nonce; and
- canonical plan hash, signer `TrustEvidence`, and signature.

The plan is derived from immutable membership; it cannot add, remove, reorder,
or rename a member. A daemon accepts it only when its local prepared-job,
membership, node-job manifest, local certificate request, controller
principal, and expiry all match. Its job-private key never leaves that daemon.

Host-crossing channels use mutual TLS 1.3. In addition to normal chain,
validity, role, and revocation checks, each endpoint verifies that the peer
certificate fingerprint and workload bindings exactly match the expected
member in the plan. The adapter refuses wildcard peers, unrelated cluster
certificates, DNS-name substitution, expired launch attempts, and endpoints
outside the signed plan. It exposes no general-purpose forwarding interface.

The worker-visible MLX hostfile contains only local loopback-shim addresses.
Remote endpoints from the signed plan exist only in the secure adapter's
private configuration; raw MLX never receives a LAN/WAN peer address and never
listens on a non-loopback interface. Failure of a secure transport helper fails
the fixed group. Transport credentials and endpoint state are removed during
terminal job cleanup.

The secure adapter treats collective bytes as opaque. It does not parse tensor
messages, infer rank, choose topology, implement collective ordering, or
initialize MLX. The group runtime still performs the bounded startup identity
exchange and remains authoritative for MLX rank/world size. This preserves the
boundary between cryptographic peer authentication, connectivity translation,
and DDP semantics.

### Credential envelope

`CredentialEnvelope` is a transport value, never a content-addressed artifact.
It is not an application credential issuer. The bounded context that owns the
protected action first creates a `ScopedApplicationCredential` and its
semantic authorization. It passes opaque secret bytes plus the exact intended
node/job/run/worker/audience binding to the cluster-trust credential-sealing
port. Trust validates only the envelope binding contract, seals the opaque
bytes to the enrolled node's separate X25519 key, and signs the envelope; it
never parses the application secret or decides what the secret authorizes.

The envelope contains issuer/controller identity, intended
node/job/run/worker and audience, an opaque credential-kind identifier,
issue/expiry time, nonce, HPKE ciphertext, issuer `TrustEvidence`, and
signature. Its signed binding hash appears in the node-job manifest, but
neither ciphertext nor plaintext does. Normal R2 coordinator authorization
uses the workload certificate and allocation proof and does not require a
second bearer credential.

The daemon accepts an envelope only over the authenticated prepare request,
verifies every binding, and decrypts it into owner-only per-job state. A
daemon-generated workload private key never enters the envelope; the envelope
does not carry certificate-signing authority. The daemon deletes all material
during terminal cleanup. It passes only protected file paths or key-store
handles to the child and never emits a credential through logs, status, crash
reports, environment, or child arguments.

### Node-agent API

Version node-agent routes independently from the R2 coordinator:

```text
GET    /v1/agent/capabilities
POST   /v1/agent/leases
PUT    /v1/agent/leases/{lease_id}/renew
PUT    /v1/agent/leases/{lease_id}/authority
DELETE /v1/agent/leases/{lease_id}
PUT    /v1/agent/jobs/{job_id}/prepare
PUT    /v1/agent/jobs/{job_id}/transport
POST   /v1/agent/jobs/{job_id}/start
POST   /v1/agent/jobs/{job_id}/cancel
GET    /v1/agent/jobs/{job_id}
GET    /v1/agent/jobs/{job_id}/logs
```

Mutating requests carry idempotency keys and expected state-machine versions.
The transport route binds one signed transport plan and workload certificate
to the matching prepared job and is safe to retry; it cannot alter membership
or other job fields. The authority route accepts only a prepared successor
identity and either
activates its bounded running-job authority or finalizes primary authority
using a matching run-commit proof. Bodies reject unknown fields and have strict
size limits. Start, authority, cancel, and release are safe to retry. Status
distinguishes reserved, prepared, starting, running, exited, failed, canceled,
and expired.

### Dataset and artifact behavior

R1.1 does not copy training datasets automatically. An administrator registers
local dataset roots or logical selectors with the daemon. Preparation resolves
the submitted selector locally and verifies the content-stable dataset ID.
Different machines may use different paths for the same dataset ID.

Node management owns the local selector-to-root catalog and path authorization.
The data-identity context owns content hashing and returns a dataset identity;
it does not read daemon enrollment or lease state. Only the selector and
verified dataset ID cross the boundary. Local filesystem paths do not enter
node capabilities, job manifests, or coordinator state.

Small immutable config and checkpoint artifacts may be transferred through the
bounded node-agent artifact path or fetched from the R2 coordinator. Transfers
are content-addressed, size-limited, streamed, checksummed, and staged before
atomic publication. A client-provided path never selects an arbitrary daemon
file.

### Security requirements

- Cluster creation, all three enrollment policies, provisioning-file
  generation/consumption, certificate issuance, renewal, revocation, TLS, and
  encrypted ring transport execute through the mixlab binary without external
  CA, VPN, or cryptography executables.
- Node-agent connections use mutual TLS 1.3 even on the local LAN. Both peers
  verify cluster, role, principal ID, validity, key usage, trust generation,
  and revocation before the transport yields an `AuthenticatedPrincipal`,
  except the signed-snapshot-only stale refresh path, which yields no
  application principal or node operation.
- Trust-on-first-use is forbidden except when the user explicitly selects
  `trusted_lan` and the authority has a matching bounded auto-enrollment window.
  There is no generic verification-disable flag, implicit fallback, or TOFU
  reuse for normal TLS.
- Node management checks controller authorization for every lease and job
  mutation after authentication; a valid cluster certificate alone grants no
  node operation.
- Job manifests are signed and bound to controller, node, nonce, and expiry.
- Each multi-host R1.1 ring uses a signed secure transport plan and mutually
  authenticated encrypted channels bound to the exact workload certificates,
  launch attempt, and DDP membership hash.
- Any additional context-owned application credential is sealed as an
  encrypted, job-bound envelope over the authenticated node-agent connection
  and materialized only as an owner-only file. Its plaintext and bearer value
  are never part of a manifest or content-addressed artifact. R2 normally uses
  its workload certificate instead of such a bearer.
- Lease-authority changes require the prepared successor identity and a signed
  run/worker binding; possession of a run ID alone grants nothing.
- mDNS never carries credentials and is not an authorization input. In
  `trusted_lan`, authorization comes from the explicit trust-owned window plus
  transport-observed interface/CIDR/quota evidence, not the advertisement.
- Stable cluster phrases and verified request phrases are comparison data, not
  secrets or bearer credentials. The client-side cluster-trust application
  computes them locally; UI text received from a peer is never accepted as the
  computed value.
- Job directories reject symlinks, traversal, and paths outside configured
  roots.
- The daemon passes only allowlisted environment values and closes inherited
  file descriptors before spawning a child.
- Cancellation terminates the complete worker process group and eventually
  releases its lease.
- Logs redact credentials, enrollment material, signatures, and protected
  storage/backend URLs.
- The daemon exposes no general command-execution endpoint.
- Root, certificate-issuer, or snapshot-signer compromise is a cluster trust
  compromise requiring explicit rekey/re-enrollment; the software never hides
  it by silently creating a new root.

### R1.1 observability

Record:

- cluster ID, principal ID/role, certificate serial suffix, trust-snapshot
  generation, and enrollment/renewal/revocation outcome without secret
  material;
- enrollment policy, window/request ID, source principal, requested role,
  root/certificate-request fingerprint suffixes, state transition, observed
  interface/CIDR decision, and approval/expiry reason; never a provisioning
  secret or reusable channel material;
- discovery provider and discovery-to-authentication failures;
- node and capability generation;
- lease ID, state transitions, contention, expiry, and renewal;
- selection filters and rejection reasons;
- job/membership manifest hash and generation;
- prepare/start/cancel timing by node;
- child exit status and bounded log references; and
- lease cleanup outcome.

Normal user output identifies selected and rejected nodes without exposing
addresses or credentials unnecessarily.

### R1.1 acceptance

Automated:

- Cluster initialization produces distinct valid root, certificate-issuer,
  snapshot-signer, authority TLS, controller, and coordinator identities with
  exact profiles/key usages, writes separate principal-state directories
  atomically beneath a disposable state home, and never places CA-purpose
  handles in them. Node/worker fixtures contain no authority directory.
- Concurrent use of one provisioning file has exactly one winner. Replayed,
  expired, wrong-purpose/resulting-role, wrong-cluster, altered, low-entropy,
  and already-consumed invitations fail without issuing a certificate.
- Cluster fingerprint/phrase test vectors are stable across certificate
  serialization and leaf/issuer renewal and change when the root public key
  changes.
- Trusted-LAN windows enforce explicit interface/CIDR, node-only role, TTL,
  maximum count, rate limits, endpoint audience, duplicate-key rejection, and
  non-empty principal-state protection. Expired/closed windows and public or
  out-of-scope requests cannot enroll; restart preserves an unexpired window's
  original bounds/counter but expires its pending channels. Failure never falls
  through to another policy.
- Verified pairing uses a fresh full TLS 1.3 handshake/exporter and binds the
  request phrase to root, endpoint, role, nonces, request ID, and certificate
  request. MITM-separated channels, altered fields, mismatched human
  confirmations, exporter reuse, 0-RTT/resumption, approval of another pending
  request, and approval after expiry all fail.
- Provisioned enrollment rejects a wrong-root or endpoint/audience mismatch
  before sending its one-time proof; verified enrollment exposes a
  man-in-the-middle as a phrase mismatch before generating/sending its
  certificate request. Node and worker private keys are generated locally and
  never appear in server state or protocol captures.
- Fake-clock tests cover principal/workload expiry, scheduled authenticated
  renewal/jitter/retry, missed-expiry reenrollment, trust-snapshot expiry,
  monotonic update, and revocation. Stale-trust daemons cannot become
  recruitable.
- Local transient and remote authority clients produce identical trust
  results. Authority outage fails issuance without key copying or approval-
  policy fallback; a stale daemon accepts only a newer
  dedicated-snapshot-signer value chaining to its pinned root plus bounded
  existing-job status/cancel/release.
- Key-store contract tests cover Keychain and strict-file fakes; unsafe
  permissions, symlinks, traversal, partial writes, and secret logging fail.
- Authentication/authorization tests prove that a valid node, worker, or
  unapproved controller certificate cannot perform a controller lease/job
  operation.
- Fake discovery deterministically models add, update, expiry, duplication,
  and removal without sleeps.
- Spoofed, wrong-cluster, revoked, and unauthenticated candidates never appear
  as recruitable nodes.
- Capability filtering rejects incompatible backend, MLX, custom-op, dataset,
  memory, and manifest versions with explicit reasons.
- Concurrent lease acquisition has exactly one winner; retry is idempotent;
  expiry and renewal obey a fake clock.
- Partial reservation and preparation failures eventually reconcile and
  release the entire cohort, including injected failures during compensation.
- Every prepared daemon reports the same assignment/run-plan hash and the
  expected hash for its distinct node-job manifest.
- Secure transport plans reject changed membership, endpoint, rank,
  certificate fingerprint, launch attempt, signature, or expiry. A valid
  cluster certificate for a different job cannot connect.
- The ring adapter exposes raw MLX endpoints on loopback only, encrypts every
  host-crossing test stream, and fails closed when its helper or authenticated
  peer disappears.
- Replayed, expired, altered, oversized, or incorrectly signed manifests are
  rejected without launching a child.
- Daemon restart cannot double-launch a worker or forget an active lease.
- Crashes before, during, and after successor-authority activation converge to
  exactly one authorized running child and a recoverable lease owner.
- Cancel terminates the worker process group and returns the node to available.
- Daemon and discovery tests run without an MLX installation and prove the
  daemon parent does not initialize the GPU runtime.
- No API or manifest field permits arbitrary shell execution.

Manual LAN:

- Run `cluster-init` and record the displayed cluster phrase/full fingerprint.
- Under disposable user homes, verify both Macs default to `~/.mixlab`, an
  explicit `-state-home` relocates the complete logical layout, and enrolled
  node homes contain principal/daemon state but no authority directory.
- Exercise `trusted_lan` with a two-node maximum: the M1 Max and M4 Max discover
  the one enrollment coordinator and auto-enroll without files or per-request
  approval. Verify an out-of-CIDR requester, third requester, and privileged
  role are rejected.
- Reinitialize disposable principal state and enroll one Mac with a distinct
  protected provisioning file; no live source approval is required and reuse
  fails.
- Reinitialize disposable principal state and enroll one Mac through
  `verified`: compare the stable phrase, compare the request phrase, approve
  the exact pending key, and verify a simultaneous wrong request is not
  approved.
- For all three policies, verify no system trust-store changes, external CA
  tools, VPN, or second agent are required and that resulting node certificate
  behavior is identical.
- Start foreground or background daemons on the M1 Max and M4 Max.
- `nodes` discovers and authenticates both over the target school LAN.
- Submit a two-rank R1 ring job and complete training without SSH or an
  interactive login on either worker. Verify host-crossing ring traffic uses
  the secure adapter and no raw MLX listener is LAN-reachable.
- Attempt to connect a different enrolled node and a stale workload
  certificate to the ring; both fail before MLX startup.
- Make one host busy and verify selection excludes it without disturbing its
  existing process.
- Disable mDNS and complete the same flow with explicit daemon addresses.
- Verify lease and child cleanup after success, cancellation, preparation
  failure, and one killed worker.

Review gate:

- Threat-model root/issuer custody, provisioning delivery, discovery spoofing,
  explicit trusted-LAN TOFU, verified human comparison, enrollment-window
  exposure, certificate renewal/revocation, stale trust, remote launch,
  collective interception, and local path confinement.
- Review the versioned certificate profile, cryptographic choices, key-store
  adapters, local/remote authority topology, stale-trust recovery, historical
  proof rules, and cluster-rekey procedure.
- Inspect two-phase launch and lease-recovery evidence.
- Confirm cluster trust does not own node leases, recruitment, membership, or
  coordinator authorization; transport adapters do not make authorization
  decisions; and no discovery, node-agent, or trust dependency entered the
  sampler, trainer, distributed training engine, or C++ code.
- Do not make R1.1 a LAN acceleration claim; it improves deployment and
  recruitment, not the DDP communication algorithm.
- Do not begin the R2 user-facing recruitment workflow until R1.1 cleanup and
  authentication gates pass.

## R2: Fixed-Cohort Synchronous DiLoCo On Macs

### Purpose

Make a school lab of otherwise idle Macs useful over gigabit Ethernet by
communicating once every many local AdamW steps instead of every step.

R2 is an experimental optimization mode. Its release criterion is convergence
and wall-clock evidence, not merely a functioning protocol.

R2 uses R1.1 to discover, reserve, and launch idle Mac learners. Users do not
construct the fixed cohort by logging into each machine. Explicitly configured
workers remain a diagnostic/development path, not the primary lab workflow.

### Cohort recruitment and launch

The fixed cohort uses a provider-neutral learner slot conceptually equivalent
to:

```go
type LearnerMember struct {
    WorkerID          string
    ParticipantID     string
    IslandID          string
    DatasetPartitionID string
    InnerGroupHash    string // empty for a single-process learner
}

type LearnerCohortMembership struct {
    RunID           string
    Generation      uint64
    OrderedLearners []LearnerMember
    CohortHash      string
}

type LearnerSlot struct {
    Member         LearnerMember
    CapabilityHash string
    Allocation     AllocationRef
}

type AllocationRef struct {
    Provider string // "node_agent" in R2; R3 adds "external"
    RefHash  string // hash of provider-specific, non-secret allocation proof
}

type PreparedLearnerAllocation struct {
    Slot                 LearnerSlot
    PreparationProofHash string
    AuthorityVersion     uint64
    ExpiresAt            time.Time
}
```

`ParticipantID` is the enrolled node ID for an R2 LAN learner and an
authenticated external-worker identity for an R3 RunPod learner. `AllocationRef`
is a strict discriminated union in the wire format. A `node_agent` allocation
contains typed node, lease, and job references; an `external` allocation
contains a durable external allocation ID plus the hash of its consumed
single-use bootstrap receipt. Neither variant carries a credential or
arbitrary provider payload.

`LearnerMember` and `LearnerCohortMembership` belong to the distributed-identity
shared kernel. `LearnerSlot`, capability evidence, and `AllocationRef` belong
to the DiLoCo run-plan model. Replacing an allocation reference without
changing `LearnerMember` is an operational recovery only when the active
release's membership policy explicitly allows it; R2-R4 do not allow such
replacement after commit. R3 process recredentialing preserves the existing
external `AllocationRef` and therefore is not an allocation replacement.

A worker ID identifies learner-private AdamW/sampler state; it is not a mutable
hostname or process ID. The coordinator, not the recruitment scheduler or
shared identity package, becomes authoritative for the ordered
`LearnerCohortMembership` when it publishes the run commit.

Before round zero, run admission executes this recoverable saga:

1. Create durable coordinator admission state for the proposed run and base
   checkpoint.
2. Use R1.1 recruitment to authenticate/filter nodes, acquire reservations,
   assign proposed worker IDs, and return proposed `node_agent` learner slots.
3. Ask global coordination to validate the proposed slots and stage the exact
   canonical `mixlab_distributed_run_v1` run-plan bytes. Its hash is now stable,
   but the cohort is not committed.
4. Prepare one signed `train-worker` job per node, bound to the staged run-plan
   hash and successor coordinator. Each daemon generates its worker key
   locally; cluster trust signs the certificate request as a run/worker-scoped
   workload identity and returns it through the protected credential path.
5. After every provider returns a `PreparedLearnerAllocation` bound to that
   plan, start workers. Start activates the coordinator's bounded
   renewal/cancel authority.
6. Each worker verifies the same run plan and global base checkpoint, then
   registers using provider-specific allocation proof.
7. After every staged learner registers exactly once, the coordinator publishes
   `mixlab_distributed_run_commit_v1`. Registration is closed and cohort
   membership becomes authoritative.
8. Finalize primary lease authority at every LAN daemon. This step is
   idempotent and recoverable from the committed run record.
9. Open round zero only after allocation authority is reconciled.

The saga journal is durable and records only IDs, hashes, state-machine
versions, deadlines, and protected credential handles. The supported R2
deployment runs the admission application service alongside or under the same
durability boundary as the coordinator, while preserving separate packages and
state ownership. `submit` initiates and observes the saga; it is not the only
copy of its state. If the initiating client disappears before commit,
unstarted reservations expire and the successor coordinator cancels any
started, expired staged admission.

Failure before the run commit triggers run-admission compensation: cancel
started jobs, revoke unused worker credentials, and release every allocation.
Failure after the run commit is a fixed-cohort R2 failure recovered by the
durable coordinator. The interactive `submit` process may exit after commit;
the coordinator renews and ultimately releases provider allocations. Daemon
and coordinator restart logic must not create a second worker for the same
committed worker ID.

Discovery remains active operationally, but nodes discovered after run commit
do not join the run. A vanished or failed worker pauses or fails the fixed
cohort under the existing R2 rules. Replacing it with a newly discovered node
would require a new cohort generation and is deferred to R5.

### Learner runtime boundary

The learner runtime is the application boundary around one independently
progressing replica. It:

- presents its run-scoped workload identity and is authorized by the
  coordinator as exactly one committed learner slot;
- acquires one `RoundAssignment` at a time;
- verifies and installs the assignment's base model through the local training
  engine;
- drives exactly `H` inner updates using its private optimizer, sampler, and RNG
  state;
- checkpoints that private state under its worker ID;
- constructs and publishes one outer-update artifact and manifest; and
- retries coordinator/artifact requests idempotently.

It does not select the cohort, renew an unrelated node lease, aggregate outer
updates, mutate global state, or interpret discovery records. Node supervision
may restart the learner process, but only the learner runtime interprets and
restores learner-private training state.

The learner runtime calls the local training engine through operations such as
`InstallGlobalWeights`, `RunInnerAttempt`, `SnapshotParameters`, and
`RestorePrivateState`. The engine does not call coordinator or artifact APIs.
R4 may implement those operations with an internal DDP group while preserving
the same learner-facing contract.

### Normative algorithm

For outer round `r`, coordinator state contains:

```text
theta_r       global parameters
momentum_r    outer Nesterov momentum
```

Each fixed learner `i`:

1. Receives and verifies `theta_r`.
2. Replaces its model parameters with `theta_r`.
3. Retains its private AdamW first/second moments and local scheduler state.
4. Performs exactly `H = inner_steps` committed local optimizer updates.
5. Computes the float32 outer gradient:

   ```text
   g_i = theta_r - theta_r_i
   ```

6. Uploads `g_i` and its metadata.

The coordinator computes:

```text
g_bar = sum_i(weight_i * g_i) / sum_i(weight_i)
momentum_(r+1) = outer_momentum * momentum_r + g_bar
update_(r+1) = g_bar + outer_momentum * momentum_(r+1)
theta_(r+1) = theta_r - outer_lr * update_(r+1)
```

This Nesterov variant is normative. Unit tests must lock the formula and sign.
Outer weight decay is zero.

Aggregation weights:

- `uniform`: `weight_i = 1`;
- `effective_tokens`: `weight_i` is the number of effective valid training
  tokens reported for the round.

R2 requires all configured learners to submit exactly once. It has no quorum,
staleness, or late-update behavior.

An accepted R2 update has exactly `H` attempted steps, `H` committed steps,
and zero skipped steps. A globally skipped inner optimizer attempt caused by
zero denominator, non-finite values, or a rejected candidate fails that
worker's round; the worker must not silently run extra data until it
accumulates `H` commits. This keeps schedule, sampler, token accounting, and
comparison semantics unambiguous. A future retry policy requires a new
protocol decision.

### Inner optimizer state

Each learner owns separate AdamW moments. They are not averaged or sent to the
coordinator as part of an outer update. They persist across rounds while model
parameters are replaced by the new global parameters.

This is a distinct state category:

- global state: model plus outer Nesterov momentum;
- learner-private state: AdamW moments, local counters, sampler cursor, and RNG
  state.

Losing learner-private state prevents exact R2 resume for that learner.
R3 adds durable private-state artifacts for ephemeral CUDA workers.

### Fixed cohort

The committed `LearnerCohortMembership` lists the exact learner members
participating in R2; the run plan binds each member to one provider-neutral
learner slot.

- Registration closes at run commit before round zero.
- Every committed learner slot equals its staged slot and has one matching
  prepared allocation and authenticated registration.
- A round remains open until every listed learner submits a valid update.
- Duplicate update IDs are idempotently acknowledged.
- A second distinct update from one worker for the same round is rejected.
- A worker failure fails or pauses the run; it does not shrink the cohort.
- Restarting a worker with its private state may resume its assigned round.
- Newly discovered or newly idle nodes remain available for another run; they
  cannot enter the current run.

### Coordinator state machine

```text
CREATED
  -> PLAN_STAGED
  -> PREPARING
  -> STARTING
  -> ADMITTING
  -> COMMITTING
  -> COHORT_COMMITTED
  -> ROUND_OPEN
  -> AGGREGATING
  -> PUBLISHING
  -> ROUND_OPEN (next round)
  -> COMPLETE

Any pre-commit state -> CANCELING -> CANCELED
Any committed state -> FAILED
```

The coordinator durably records every transition and the idempotency/proof
values needed to resume it. `PLAN_STAGED` fixes the run-plan bytes but grants no
training membership. Publishing the run-commit record is the only
`COMMITTING -> COHORT_COMMITTED` transition. A crash during per-daemon authority
finalization resumes from `COHORT_COMMITTED` and cannot restage or replace a
learner.

Only a fully checksummed global round manifest advances the current round.
Recovery first finds the committed run plan/commit pair and then resumes from
the latest complete round. Pre-commit recovery either continues the same
admission saga or compensates it; it never treats a staged plan as a run.

### Artifact formats

#### Coordinator run plan

Format identifier: `mixlab_distributed_run_v1`.

This immutable staged plan fixes the candidate cohort and training contract but
does not commit membership. Its canonical bytes and hash are finalized before
worker preparation and registration. It contains:

- run ID, run-plan format version, membership generation, cluster trust-domain
  ID, coordinator principal ID/role, and coordinator workload audience;
- config, program, weight-layout, inner-optimizer, and dataset hashes;
- base-checkpoint artifact identity;
- ordered provider-neutral learner slots containing worker ID, participant ID,
  capability hash, island ID, dataset partition ID, provider kind, and typed
  non-secret allocation-reference hash, plus optional inner-DDP-group hash;
- outer optimizer and aggregation configuration;
- artifact-store identity and bounded coordinator endpoints; and
- creation/expiry time and complete plan SHA-256.

The coordinator stages these bytes durably and returns their hash to run
admission. A worker may verify the staged plan by hash, but a staged plan without
a valid run commit is not permission to train. Changing membership requires a
new staged plan and cohort hash. Changing only pre-commit capability/allocation
evidence still requires new staged-plan bytes even when the membership hash is
unchanged. Staged bytes are never edited in place.

The plan expiry is an admission deadline. An expired uncommitted plan must be
compensated and cannot later commit. Once a matching run commit exists, that
deadline no longer expires the committed run.

#### Coordinator run commit

Format identifier: `mixlab_distributed_run_commit_v1`.

This small record is the atomic commit marker for the staged plan. It contains:

- run ID, run-plan SHA-256, membership generation, and cohort hash;
- ordered worker registration proofs bound to every staged learner slot;
- provider-specific preparation/allocation proof hashes;
- signed node-job manifest hashes for `node_agent` allocations and external
  enrollment proof hashes for `external` allocations;
- cluster trust-domain ID, coordinator principal ID, and run-commit format
  version;
- commit time and complete commit-record SHA-256; and
- coordinator `TrustEvidence` and Ed25519 signature.

The coordinator publishes this record only after every staged learner registers
with matching identity, capability, allocation, base checkpoint, and run-plan
hash. The record commits the exact `LearnerCohortMembership`; changing one
learner creates a new run or a future R5 membership generation.

#### Global round manifest

Format identifier: `mixlab_distributed_round_v1`.

Required fields:

```json
{
  "format": "mixlab_distributed_round_v1",
  "run_id": "...",
  "run_plan_sha256": "...",
  "run_commit_sha256": "...",
  "membership_generation": 0,
  "cohort_hash": "...",
  "round": 12,
  "model_file": "...",
  "outer_state_file": "...",
  "config_hash": "...",
  "program_hash": "...",
  "weight_layout_hash": "...",
  "inner_optimizer_hash": "...",
  "dataset_id": "...",
  "outer_optimizer": {
    "kind": "nesterov",
    "lr": 0.7,
    "momentum": 0.9
  },
  "accepted_update_ids": ["..."],
  "total_effective_tokens": 0,
  "created_at": "RFC3339 timestamp"
}
```

Tensor files contain float32 tensors and have SHA-256 values in the full
manifest type even if abbreviated above.

#### Outer update manifest

Format identifier: `mixlab_outer_update_v1`.

Required fields:

- update ID, run ID, round, base round;
- run-plan and run-commit hashes;
- membership generation, cohort hash, participant ID, worker ID, and island ID;
- config/program/weight/optimizer/dataset hashes;
- delta file name, size, SHA-256, dtype, and tensor count;
- attempted, committed, and skipped inner steps;
- effective tokens and examples;
- local loss summary;
- delta L2 norm and maximum absolute value;
- local start/end time; and
- outer-update format version.

The wire-level `UpdateSubmission` wraps the semantic outer-update manifest with
authenticated worker identity, active workload credential generation and
launch-attempt ID, and a typed allocation-proof reference.
Transport/application code validates that proof through the allocation-keeper
port and records provider-specific audit metadata before passing the semantic
manifest to update admission. Provider kind, node ID, lease ID, node-job hash,
external enrollment ID, credential generation, and launch-attempt ID never
enter aggregation or outer-optimizer inputs.

The delta safetensors file uses the model's canonical weight names and stores
`theta_r - theta_r_i`, not the inverse.

### Artifact-store abstraction

Define a narrow content-addressed interface:

```go
type ArtifactStore interface {
    Put(ctx context.Context, expectedSHA256 string, r io.Reader, size int64) (ArtifactRef, error)
    Open(ctx context.Context, ref ArtifactRef) (io.ReadCloser, error)
    Stat(ctx context.Context, ref ArtifactRef) (ArtifactInfo, error)
}
```

R2 requires a filesystem implementation and authenticated coordinator
upload/download endpoints. Storage keys derive from SHA-256, not
client-provided paths. Clients cannot request arbitrary coordinator filesystem
paths.

Knowledge of an `ArtifactRef` or digest grants no access. Every non-local
worker read/write request carries an `ArtifactAccessGrant` and the transport's
`AuthenticatedPrincipal`. The canonical signed grant contains:

- grant ID, format version, cluster/run ID, principal/participant ID, worker
  ID, and coordinator audience;
- allowed operation (`read`, `create`, or bounded resumable upload);
- exact existing `ArtifactRef`, or expected digest, size, media/schema kind,
  and upload ID for a not-yet-published artifact;
- semantic access class (`global_base`, `outer_update`, or
  `learner_private`) used by global coordination to apply policy;
- issue/expiry time, maximum uses, and nonce; and
- issuer principal, grant digest, `TrustEvidence`, and signature.

Global coordination normally issues grants only from staged/committed slots
and active round assignments; the sole R3 exception is the exact read-only
pending-recovery grant defined below. Base/global artifacts are readable by the
intended committed workers. Outer-update creation is limited to the assigned
worker/round. Learner-private artifacts are readable/writable only by the exact
worker principal and coordinator recovery path; another worker in the same run
is denied. The API authorization adapter verifies the authenticated principal
and asks global coordination to validate current run/round policy before the
artifact-store adapter enforces the grant's byte limits.

The coordinator API authorization adapter owns a durable, atomic grant-use
ledger keyed by grant ID, operation/upload ID, and request idempotency key. It
reserves or consumes an allowed use before calling storage. A retry with the
same idempotency key returns the same outcome and does not increment use;
concurrent distinct operations cannot exceed `maximum uses`, including across
coordinator restart. One bounded resumable-upload grant binds exactly one
upload session, so its chunks are not separate uses. Global coordination owns
semantic grant issuance/validation; the adapter owns replay/accounting
mechanics; artifact storage owns byte enforcement/publication and does not
issue grants, keep the authorization ledger, or interpret access class.

R3 may add an S3-compatible implementation without changing manifests.
In the R3 baseline only the coordinator-side storage adapter communicates with
S3. A presigned/backend URL is never returned to a worker: every worker
artifact operation remains on the managed mutual-TLS coordinator API. Any
coordinator-internal storage credential is short-lived, operation-scoped, and
never stored in a manifest or log. Direct worker-to-object-store transfer is a
future protocol requiring its own authenticated-transport review.

The store owns blob integrity and publication only. Local training owns DDP
checkpoint compatibility; learner runtime owns learner-private checkpoint and
outer-update semantics; global coordination owns run, round, and outer-state
manifests. An `ArtifactStore` implementation must not decide whether a topology
may resume or whether an update is admissible.

### Round-assignment contract

Format identifier: `mixlab_round_assignment_v1`.

Global coordination issues a round assignment only to one authenticated worker
in the committed cohort. It contains:

- assignment ID, run ID, run-plan/run-commit hashes, cohort generation/hash,
  worker ID, round, and base round;
- base-model and outer-state artifact references;
- config/program/weight/inner-optimizer/dataset hashes;
- dataset partition ID and exactly `H` required inner attempts/commits;
- required previous learner-private-state hash when exact continuation applies;
- issue/expiry time and round-assignment format version; and
- an authenticated coordinator binding.

It contains no discovery record, node address, shell command, or provider lease
details. Allocation health is checked before issuance through an
allocation-keeper port. Retrying acquisition returns the same active assignment
for that worker/round; it never creates a second unit of work.

### R2 API

Version control-plane routes under `/v1`.

Minimum operations:

```text
POST /v1/runs
POST /v1/runs/{run_id}/workers/register
POST /v1/runs/{run_id}/admission/commit
POST /v1/runs/{run_id}/admission/cancel
POST /v1/runs/{run_id}/round-leases
POST /v1/runs/{run_id}/updates
GET  /v1/runs/{run_id}
GET  /v1/artifacts/{sha256}
PUT  /v1/artifacts/{sha256}
```

Requirements:

- `POST /v1/runs` validates and durably stages immutable run-plan bytes; it does
  not commit the cohort.
- Admission commit verifies every expected registration and preparation proof,
  publishes exactly one run-commit record, and is idempotent.
- Admission cancel is accepted only before commit and drives compensation; it
  can never uncommit a cohort.
- A coordinator round lease assigns DiLoCo round work to an already committed
  learner. It is distinct from the R1.1 node-agent reservation lease.
- JSON bodies reject unknown fields.
- Worker registration requires a mutually authenticated workload certificate
  bound to the exact participant, worker, run, staged run-plan hash, and
  coordinator audience. It also includes the capability hash, membership
  generation, base-checkpoint hash, and one strict provider-specific allocation
  proof.
- A `node_agent` registration proves node, lease, and signed job bindings. An
  `external` registration proves its single-use enrollment/allocation binding.
- The coordinator accepts registration only for an exact learner slot in the
  staged `mixlab_distributed_run_v1` plan.
- Upload sizes are declared and bounded before allocation.
- Artifact reads and writes require an exact principal-bound
  `ArtifactAccessGrant`; digest knowledge or membership in the same run is
  insufficient. The API authorization adapter durably reserves/consumes the
  grant use with the request idempotency key before storage access.
- Artifact hashes are verified while streaming.
- Mutating requests carry idempotency IDs.
- Errors have stable machine-readable codes plus human-readable context.
- Workers only initiate outbound connections.

All non-loopback coordinator traffic uses mutual TLS 1.3 with managed cluster
identities, including R2 on a school LAN. The learner verifies a
cluster-signed `coordinator` principal with the expected cluster, endpoint
audience, validity, and revocation state. The coordinator transport verifies a
`worker` workload principal, then global coordination separately authorizes
that principal against the exact staged or committed learner slot.

Node-agent-launched learners receive their signed workload certificate through
the protected job-preparation path; their private key was generated locally
during prepare. Any distinct application credential follows the opaque
credential-envelope path. Externally
provisioned learners use their protected file only for bootstrap, pin the
cluster root from it, generate their key locally, and exchange the single-use
secret for a scoped workload certificate. Human-verified external learners pin
the root through phrase comparison and obtain the same scoped workload
certificate after exact request approval. Plain HTTP is allowed only on
loopback in local automated tests.

### Model replacement

Add a trainer operation that atomically:

1. flushes pending device work;
2. validates the complete incoming weight layout;
3. stages every replacement weight;
4. evaluates finiteness;
5. swaps all weights together; and
6. preserves local AdamW moments and counters.

Looping over public `SetWeightGPU` calls and leaving a partially replaced model
after an error is not acceptable.

### Validation and telemetry

The coordinator evaluates each newly published global checkpoint, either
locally or through one designated evaluator. Worker-local validation is
diagnostic only.

Record per round:

- inner compute time by learner;
- coordinator wait time and straggler time;
- uploaded/downloaded bytes and duration;
- delta norm per learner;
- cosine similarity of each delta to the aggregate;
- aggregation weights;
- outer update norm;
- global validation metrics;
- total worker compute and effective tokens; and
- wall-clock time from run start.

### R2 acceptance

Automated:

- A pure-Go two-worker quadratic oracle locks delta sign, weighted aggregation,
  Nesterov state, and round transitions.
- Coordinator and run-admission restart at every pre-commit state-machine
  boundary either continue the same staged plan or compensate all allocations;
  a staged plan is never mistaken for a committed run.
- Coordinator restart during run-commit publication or allocation-authority
  finalization publishes exactly one commit and converges every allocation to
  coordinator ownership without double-launching.
- Coordinator restart at every round state-machine boundary either resumes the
  previous committed round or completes the pending publish exactly once.
- Retried artifact and update submissions are idempotent.
- Corrupt, oversized, non-finite, wrong-shaped, stale-round, and hash-mismatched
  updates are rejected without mutating global state.
- The R1.1 prepared allocations, staged run plan, run commit, and registered
  worker cohort contain exactly the same learner slots and generation.
- A provider-specific allocation proof can be validated without exposing its
  fields to the outer optimizer or aggregation code.
- Valid cluster certificates with the wrong role, worker ID, run, run-plan
  hash, coordinator audience, expiry, or allocation binding are rejected
  before learner registration or artifact access.
- A node discovered after registration closes is not admitted to the current
  run.
- Coordinator restart resumes lease renewal and worker ownership without
  double-launching a learner.
- A tiny two-learner causal run produces finite global checkpoints and resumes
  with private AdamW state intact.
- `outer_lr=1`, zero outer momentum, and one worker reduce to replacing the
  global model with that worker's local model.

Experimental gate:

- Pre-register the configs, datasets, seeds, inner-step values, outer
  hyperparameters, and comparison metric before running the final experiment.
- Compare M1 Max plus M4 Max over gigabit Ethernet against the M4 Max alone.
- Recruit and launch both Macs through R1.1 without SSH or interactive login
  on either learner.
- Primary metric is wall-clock time to a fixed validation-loss target.
- Report total compute, total tokens, and final validation loss so extra
  compute is not mistaken for sample-efficiency improvement.
- At least one representative mixlab workload must reach the target faster
  than the M4-only baseline without a material final-loss regression.

Review gate:

- If the algorithm works but does not improve time to target, retain it as
  experimental and do not claim LAN acceleration.
- Do not begin R3 until global and learner-private recovery has been exercised.

## R3: Fixed-Round Metal Plus RunPod CUDA

### Purpose

Run one synchronous DiLoCo cohort containing at least one Metal learner on the
school LAN and one CUDA learner on RunPod. No collective crosses the WAN.

### Architecture

- The coordinator and authoritative artifact store are durable and independent
  of a RunPod worker lifecycle.
- A RunPod worker makes outbound authenticated HTTPS requests.
- School-LAN Metal learners may still be recruited through R1.1.
- RunPod discovery does not use mDNS or require an inbound public node-agent
  port; the explicit coordinator URL plus either provisioning material or a
  verified pairing channel supplies enrollment.
- An `ExternalEnrollmentProvisioner` adapter accepts an outbound RunPod
  preflight connection under either `provisioned` or `verified`; `trusted_lan`
  is rejected. Provisioned bootstrap pins the exact coordinator endpoint/
  audience in a single-use `external-worker-bootstrap` file. Verified bootstrap
  pins it through the cluster/request phrase protocol and explicit approval.
  The coordinator proves the named `coordinator` certificate. The provisioner
  passes the worker-generated request and policy evidence to cluster trust
  through `TrustAuthorityClient`; cluster trust returns the signed
  `EnrollmentApprovalRecord`. The provisioner consumes that record, validates
  capabilities, and produces an `external` learner slot. After plan binding,
  cluster trust issues the run-scoped workload certificate. The adapter does
  not gain enrollment-policy, node-agent lease, discovery,
  certificate-issuance, or command-execution semantics.
- Run admission combines `node_agent` and `external` slots before asking global
  coordination to stage one provider-neutral run plan. The outer optimizer and
  round state machine never branch on allocation provider.
- The same model checkpoint and ordered weight layout load on Metal and CUDA.
- Each worker has its own local dataset copy with the agreed dataset identity.
- R2 round and optimizer semantics remain unchanged.

The coordinator must not live solely on an ephemeral RunPod volume.

Automatic RunPod creation remains out of scope. The user or an external system
starts the RunPod worker with either a protected single-use provisioning value
created by the same mixlab binary or an interactive verified connection to the
explicit coordinator URL. The first pins the root/endpoint before revealing
its secret; the second pins them through human phrase comparison before sending
the certificate request. The worker generates its key locally, reports
capabilities over the outbound connection, and waits in preflight state until
run admission either binds it to a staged plan or rejects/expires it. A preflight
registration is not a principal certificate, cohort membership, or authority
to acquire round work.

A headless Linux worker uses the same state-home resolver as macOS. Its default
is the container user's `~/.mixlab`; an operator who wants local state to
survive pod recreation points `-state-home` or `MIXLAB_STATE_HOME` at a
protected mounted volume. This changes storage location only. Coordinator-owned
run/checkpoint state and the R3 recovery protocol remain authoritative, and the
worker state home never contains cluster-authority state.

### Capability registration

Worker registration records:

- OS and architecture;
- MLX version and git/build identity;
- backend: Metal or CUDA;
- device name and memory;
- supported compute dtypes;
- available distributed backends;
- mixlab commit/build ID;
- program and custom-op capability list; and
- maximum accepted artifact size.

Capabilities are diagnostic and validation inputs. They do not permit workers
to train different program semantics.

### Durable private worker state

R3 adds content-addressed storage for each worker's private AdamW and sampler
state at a round boundary. This state is never aggregated. Durable private
state is required for any learner declared ephemeral or used in the R3 recovery
claim; an explicitly non-ephemeral learner may keep a local-only copy but then
cannot claim recreated-worker exact resume.

- A recreated RunPod worker with the same worker ID restores its latest private
  state before continuing.
- Private state references are visible only to that worker and the coordinator.
- If private state is missing, fixed-cohort exact resume fails rather than
  silently resetting AdamW.
- An explicit future warm-rejoin policy may permit reset state; R3 does not.

### External process recovery and recredentialing

An R3 external allocation is a durable coordinator/provider record, not a pod
process ID. Its `AllocationRef`, `ParticipantID`, and `WorkerID` survive
recreation of the RunPod process. R3 permits recredentialing only within that
same allocation identity; it does not replace the allocation, learner member,
cohort generation, or dataset partition.

When a committed external process is confirmed failed and its required
learner-private state is available, global coordination may publish one signed
`ExternalRecoveryGrant` containing:

- run-plan/commit and cohort hashes;
- unchanged participant ID, worker ID, and external allocation-reference hash;
- failed and replacement launch-attempt IDs plus the next credential
  generation;
- required learner-private-state artifact/hash and last committed round;
- coordinator audience, issue/expiry time, nonce, and idempotency ID; and
- coordinator `TrustEvidence` and signature.

The grant authorizes one pending replacement attempt, not a key that cannot yet
exist, and therefore contains no certificate-request hash. Allocation recovery
asks cluster trust to create an `external-worker-recovery` invitation bound to
that grant and to the exact coordinator recovery-preflight endpoint/audience.
The recreated worker pins the same cluster root, connects to that endpoint, and
only then generates a new key and certificate request locally. The external
provisioner proves that the same allocation remains valid and sends the
invitation, request hash, and opaque recovery grant to cluster trust through
`TrustAuthorityClient`. Trust atomically consumes the invitation and returns a
signed receipt binding its ID, the recovery-grant digest, and that request
hash.

Allocation recovery presents the receipt and provider proof to global
coordination. Global coordination confirms the exact failed slot, durably
fences the old credential generation, and transitions only the designated new
incarnation to `PENDING_RECOVERY`. Cluster trust then signs a short-lived
pending-recovery workload certificate for the unchanged participant/worker,
new launch attempt, credential generation, coordinator audience, and recovery
grant. Trust does not decide that the slot is recoverable or activate it.

`PENDING_RECOVERY` is a deliberately restricted global-coordination state.
Global coordination issues exactly one read-only `ArtifactAccessGrant` bound
to the pending workload principal, recovery-grant ID, and required
learner-private `ArtifactRef`, with `maximum uses = 1`; an interrupted retry
must reuse the same idempotency key. That principal may use only that grant on
the mutual-TLS artifact API and call the exact `/recover` operation. It cannot
register as active, obtain any other artifact grant, acquire round work, renew
work, upload an update, submit an update, or mutate run state. The API
authorization adapter accounts for this grant through the same durable use
ledger as every other artifact operation.

Learner runtime restores and verifies the required private state, then calls
`/recover` with its restored-state hash. Global coordination validates the
pending principal, recovery grant/receipt, unchanged external allocation
proof, exact required artifact/hash, and learner-reported restored-state hash,
then atomically transitions `PENDING_RECOVERY -> ACTIVE`. Only the active
credential generation/launch attempt can subsequently register for normal
work. A delayed old process or certificate remains fenced. Missing private
state, a changed allocation reference or participant, concurrent live
incarnation, invalid receipt/grant, or failed provider proof aborts the pending
attempt, revokes its workload credential, and leaves the committed slot failed
without changing cohort membership.

This is a cross-context allocation-recovery saga, not admission of a new
learner and not R5 elasticity. Global coordination owns the recovery decision
and fencing, cluster trust owns invitation/workload credential lifecycle, the
external provisioner owns allocation proof, artifact authorization protects
private state, and learner runtime owns semantic restore.

R3 adds:

```text
POST /v1/runs/{run_id}/workers/{worker_id}/recovery-grants
POST /v1/runs/{run_id}/workers/{worker_id}/recover
```

The first operation is controller-authenticated, idempotently creates at most
one active recovery grant, and requires the slot to be failed/fenced with an
available private-state reference. Invitation consumption plus provider proof
creates the pending incarnation as described above. The second route requires
the exact pending workload principal, recovery grant and consumption receipt,
unchanged external allocation proof, and restored-state hash, then atomically
activates the new credential generation. Neither route can change run-plan or
cohort bytes.

### Security requirements

- A provisioned external bootstrap/recovery uses server-authenticated TLS 1.3
  pinned by its file and can only consume that value and submit its bound
  request. A verified bootstrap uses the dedicated provisional TLS/exporter
  pairing path and explicit approval. `trusted_lan` is always rejected. After
  issuance, all public-internet traffic uses mutual TLS 1.3 with the managed
  mixlab trust domain; no public CA or system trust-store installation is
  required.
- Bootstrap bearer secrets come from a RunPod secret/environment value or
  protected provisioning file, are high entropy, expiring, role-scoped, and
  single-use, and are replaced by a run-scoped workload identity.
- The worker authenticates the coordinator against the provisioning-file-
  pinned or human-confirmed root before sending its bootstrap proof/request.
  The coordinator authorizes the resulting worker principal separately against
  its external allocation and committed learner slot.
- The worker never receives an object-store URL or credential in R3; artifact
  traffic remains on the mutual-TLS coordinator API while only the
  coordinator-side adapter contacts S3.
- Logs redact credentials and protected storage/backend URLs.
- Safetensors is the only accepted tensor artifact format.
- The coordinator validates declared size, checksum, tensor count, names,
  shapes, dtype, and finiteness before accepting an update.
- Worker requests cannot cause command execution or arbitrary file access.
- An enrolled but faulty worker can fail a fixed round; it cannot mutate an
  already committed checkpoint.

### R3 acceptance

- One Metal and one CUDA learner complete multiple fixed rounds from one base
  checkpoint.
- One committed provider-neutral run plan contains a LAN `node_agent` slot and
  a RunPod `external` slot without requiring node/lease/job fields on the
  external slot.
- Both backends independently match a shared forward/loss fixture before the
  first round.
- Provisioned and verified external-bootstrap fixtures produce the same
  run-scoped workload profile; `trusted_lan` external bootstrap is rejected.
- Killing and recreating the RunPod worker uses one recovery grant/invitation,
  preserves participant/worker/allocation identity, fences the old credential
  generation, enters `PENDING_RECOVERY`, reads only the exact private-state
  artifact, restores it, atomically becomes active, and finishes the assigned
  round.
- Killing and restarting the coordinator recovers the latest committed global
  round.
- No WAN request is made per inner optimizer step.
- WAN bytes per effective token and RunPod cost to the validation target are
  reported.
- A deliberately incompatible custom-op build is rejected during registration
  or provider preflight before run commit.
- A RunPod worker bootstraps through either a single-use provisioning file or a
  verified pairing request, authenticates the managed coordinator identity
  without public CA roots, and uses its issued workload certificate for all
  subsequent requests.
- A headless Linux/RunPod worker uses the same `~/.mixlab` logical state layout
  with strict-file key storage, needs no desktop keyring service, and contains
  no cluster-authority state.
- Reuse of the bootstrap secret and workload certificates bound to another
  run, worker, coordinator, or expired allocation is rejected.
- Recredentialing with a changed allocation reference, missing private state,
  stale recovery grant, or still-active incarnation is rejected without
  changing the fixed cohort.

Review gate:

- Confirm Metal/CUDA numerical differences do not destabilize the selected
  experiment.
- Confirm the durable coordinator is not coupled to worker availability.
- Inspect external recredential/fencing evidence and confirm it preserves the
  exact committed participant and allocation.
- Verify cross-worker learner-private artifact access is denied.

## R4: Multi-GPU CUDA Learner Islands

### Purpose

Allow an internal fixed-world NCCL DDP group to contribute one outer update as
one learner.

### Requirements

- Global coordination sees one learner slot and one worker ID. Its optional
  `inner_group` subrecord contains the hash of a `DDPGroupMembership`, but
  internal rank IDs are not outer cohort members.
- R1 DDP runs all `H` inner steps inside the CUDA learner.
- Internal DDP ranks use disjoint data and globally correct gradient
  normalization.
- The learner leader implements the R2 learner-runtime port. Only it registers,
  acquires round work, downloads/uploads coordinator artifacts, and submits the
  one outer update.
- The leader snapshots the learner's base/final parameters and constructs the
  outer-gradient artifact only after all ranks complete the same `H` commits.
- Installing global parameters is a group transaction: the leader obtains the
  artifact; every rank stages and validates the complete layout; a collective
  ready decision precedes local swap; and a post-swap checksum agreement
  verifies identical weights. Failure before agreement fails the learner
  attempt.
- The coordinator aggregation weight describes the learner island as a whole,
  using its total effective tokens, not rank-zero tokens.
- Internal rank failure fails the learner attempt. R4 does not shrink the DDP
  world dynamically.
- The R4 baseline is a same-host CUDA/NCCL island. A future multi-host NCCL
  island may not claim R4 support until its host-crossing collective transport
  satisfies the same authenticated-encryption and workload-binding invariant
  as R1.1; provider-network location alone is not authentication.
- R4 remains replicated DDP; FSDP/ZeRO is a separate future release train.

### Hierarchical learner-private state

Format identifier: `mixlab_learner_private_state_v1`.

The learner leader publishes one semantic private-state manifest containing:

- worker ID, current base round/model hash, and outer run-plan/commit hashes;
- inner `DDPGroupMembership` and the launch-attempt ID that wrote the state;
- one verified replicated model/AdamW/schedule-state artifact reference;
- ordered per-rank entries containing member ID, rank, sampler state, RNG
  counter state, and attempted/committed/skipped counters;
- config/program/weight/optimizer/dataset hashes; and
- artifact sizes, hashes, manifest format version, and complete manifest hash.

Rank-zero publication does not imply rank-zero ownership of rank-private sampler
or RNG state. Every rank contributes its private state before the leader
publishes the manifest. Exact resume requires the same inner DDP membership
generation and ordered rank mapping, while using a new launch-attempt ID.
Artifact storage verifies bytes; the learner runtime validates group topology
and semantic restore compatibility.

### R4 acceptance

- A two-GPU NCCL learner produces the same outer gradient as a one-process
  learner over the equivalent inner batches within tolerance.
- Replacing the global model leaves every internal rank with identical weights
  and private optimizer counters.
- Injecting a replacement validation failure on one rank prevents every rank
  from committing the new global model.
- Only one update is visible to the coordinator.
- Internal DDP communication is reported separately from WAN communication.
- RunPod pod restart recovery restores the replicated optimizer state and every
  rank's sampler/RNG state from the complete learner-private manifest.

## R5: Elastic And Asynchronous Learners

### Purpose

Remove the fixed-cohort barrier so idle lab Macs and preemptible CUDA capacity
can join, leave, and progress at different rates.

R5 is research-gated. Do not implement it by merely accepting late R2 updates.
A dedicated design review must lock its optimizer and staleness semantics.

### Required design inputs

The R5 design must explicitly define:

- separate learner-cohort and inner-DDP membership-generation creation, commit,
  and transition boundaries;
- launch-attempt identity independently from membership generation;
- how R1.1 discovery and node leases feed the elastic rendezvous service;
- whether a membership change restarts an inner DDP group or only changes an
  outer learner cohort;
- learner lease duration and renewal;
- minimum aggregation quorum;
- adaptive grace window;
- base-version and maximum-staleness policy;
- whether and how stale deltas are scaled, rebased, or rejected;
- dynamic token-weighted merging;
- data-range leasing and duplicate-sample policy;
- exactly-once update application;
- learner behavior after missing a synchronization;
- private optimizer-state behavior after a long absence;
- fairness between fast CUDA and slow Mac learners;
- outer optimizer behavior under variable update cadence; and
- checkpoint recovery with in-flight updates.

The R5 review must publish an updated context map and package dependency graph
for an elastic membership/rendezvous context, data-range lease context, and
quorum/staleness aggregation policy. It may reuse R1-R4 value contracts but may
not silently move their ownership into discovery, node management, artifact
storage, or observability.

The design should begin from Decoupled DiLoCo's independent learners, central
synchronizer, minimum quorum, grace window, and token-weighted merge rather
than inventing asynchronous parameter averaging from scratch.

### Minimum R5 acceptance

- Deterministic simulation covers fast, slow, failed, duplicated, delayed, and
  rejoining learners.
- Chaos tests kill learners and the coordinator at every protocol phase.
- Duplicate and out-of-window updates never apply twice or mutate a committed
  checkpoint.
- Validation quality is compared with fixed-round R3 at matched total compute.
- Goodput, staleness distribution, dropped compute, WAN usage, and time to
  validation target are reported.
- Release remains labeled experimental until more than one model family and
  dataset reproduce acceptable convergence.

## Cross-Cutting Checkpoint Requirements

### Global state

Global distributed checkpoints include:

- model weights;
- outer optimizer state when applicable;
- run-plan and run-commit references for DiLoCo;
- committed learner-cohort generation/hash, or DDP group
  generation/members-hash, as applicable;
- global round/update counters;
- scheduler and early-stop state;
- content hashes and context-owned manifest/protocol versions;
- aggregate token/example counters;
- accepted update IDs needed for idempotent recovery; and
- references to committed learner updates.

### Learner-private state

Learner-private checkpoints include:

- inner optimizer tensors and counters;
- immutable learner-cohort membership and worker slot;
- immutable inner DDP group membership and ordered rank assignment when
  applicable;
- the writing launch-attempt ID for audit, not topology equivalence;
- local learning-rate schedule state;
- sampler state;
- local RNG state or sufficient counter state;
- current base round and base model hash;
- attempted/committed/skipped step counters; and
- DDP world metadata for a multi-rank learner.

Artifact storage owns none of these semantic requirements. DDP local training,
learner runtime, and global coordination each validate their own checkpoint
format and recovery invariants before opening referenced artifact bytes.

### Recovery levels

The CLI and logs distinguish:

- exact resume: all required global and private state is present and topology
  requirements match;
- weights-only warm start: model weights are loaded but optimizer/sampler
  continuity is intentionally discarded; and
- unsupported resume: fail with the mismatching invariant.

No command silently downgrades exact resume to a warm start.

## Compatibility Matrix

| Capability | R1 DDP | R1.1 Recruitment | R2 DiLoCo | R3 Mixed | R4 CUDA island | R5 Elastic |
|------------|--------|-------------------|-----------|----------|----------------|------------|
| Metal | Ring | Discover/reserve/launch | Single-process learner | Yes | N/A | Yes |
| CUDA | NCCL | Protocol-compatible node agent | Optional single learner | Yes | NCCL DDP | Yes |
| Metal+CUDA one collective | No | N/A | No | No | No | No |
| AdamW | Required | Job capability constraint | Required | Required | Required | Required |
| Muon/LAMB | Later parity work | Capability only | No | No | No | Separate research |
| Gradient accumulation | Yes | Job manifest constraint | Inner learner may use it | Yes | Yes | Yes |
| Dynamic discovery before launch | No | Yes | Yes | Yes | Yes | Yes |
| Membership change after launch | No | No | No | No | No | Policy-defined |
| Fixed cohort | Yes | Produces prepared assignments | Coordinator commits | Coordinator commits | Coordinator commits | No |
| Full model replica | Yes | N/A | Yes | Yes | Yes | Yes |
| Exact same-topology resume | Yes | Relaunches only a compatible topology | Yes | Yes | Yes | Policy-dependent |
| Per-step WAN traffic | N/A | No | No | No | No | No |
| Self-managed cluster identity | Trusted-launcher prerequisite | Yes | Yes | Yes | Yes | Yes |
| Enrollment bootstrap | Trusted launcher | `trusted_lan`, `provisioned`, or `verified` | Inherits node identity | External: `provisioned` or `verified` | Inherits learner identity | Requires R5 review |
| Host-crossing collective encryption | Trusted-network prerequisite | Required for managed ring | N/A | No cross-island collective | Required if multi-host | Policy-defined |

## Observability Requirements

Extend existing telemetry rather than creating a disconnected logging system.

Every distributed record includes:

- run ID;
- round;
- group/cohort ID, membership generation, launch-attempt ID, participant ID,
  job ID, and provider-specific allocation identifiers where applicable;
- authenticated cluster/principal role and non-secret certificate/trust
  generation identifiers where applicable;
- learner, island, worker, rank, and world identifiers where applicable;
- backend and device;
- mixlab/MLX build identity;
- local attempted and committed steps;
- effective tokens/examples;
- compute, data, collective, coordinator-wait, upload, and download durations;
- bytes communicated;
- loss and validation metrics;
- gradient or delta norm;
- skip/failure reason; and
- base/global version.

Each emitting context owns the meaning and schema of its event payload.
`EventEnvelope` supplies correlation and bounded transport metadata only.
Observability sinks may aggregate, store, and present events but cannot turn a
metric or missing event into a training, scheduling, lease, or admission
decision.

Normal console progress is coordinator/rank-zero only. Diagnostic worker logs
remain available with identifiers. Do not read back per-weight statistics every
step.

Required summary metrics:

```text
effective global tokens/sec
useful compute goodput
DDP parallel efficiency
LAN/WAN bytes per effective token
round straggler fraction
time to validation target
RunPod dollars to validation target
```

## Failure Semantics

### R1

Any rank loss or unhandled rank error fails the world. The launcher is
responsible for terminating surviving ranks. No partial continuation.

### R1.1

- Discovery loss alone does not stop a prepared or running job; the
  authenticated node-agent connection and child status are authoritative.
- Expired or stale trust prevents new recruitment and lease acquisition.
  Synchronizing a revocation for the active node or workload fails and cleans
  up that job; revoking a controller removes its future mutation authority
  without granting authority to another principal implicitly.
- Authority unavailability prevents enrollment, renewal, workload issuance,
  and stale-snapshot recovery but does not grant fallback trust. Existing valid
  jobs follow their credential/lease deadlines and retain bounded
  status/cancel/release behavior.
- Enrollment-source loss expires its pending verified requests. Authority
  restart restores only unexpired durable windows, expires every pending
  channel-bound request, re-evaluates window bounds, and never widens a policy
  or auto-approves a verified request.
- Failure to reserve or prepare any selected node aborts the complete
  uncommitted assignment and drives every lease through reconciled cleanup.
- Partial direct-R1 start cancels every worker in that launch attempt.
- Loss of a secure collective helper or failure of its peer authentication
  fails the fixed DDP world; it never falls back to a raw LAN socket.
- Daemon restart reconciles durable lease/job state before accepting work.
- For R2, started but uncommitted workers remain under the run-admission saga
  and are compensatable. Fixed-cohort R2 failure semantics begin only when the
  run-commit record is published.
- R1.1 never recruits a replacement into a live group or committed R2 cohort.

### R2-R4

- Invalid updates are rejected without global mutation.
- A staged run plan without a matching run commit is ignored for round
  recovery and grants no cohort membership.
- Fixed-cohort worker loss pauses or fails the open round.
- A worker may retry from its last private checkpoint.
- R3 may recreate a failed external process only through the committed-slot
  recovery grant/recredential saga; all other participant/allocation changes
  remain fixed-cohort failures.
- Coordinator restart reconciles the run plan/commit, allocation authority, and
  then the latest committed round.
- A publication failure leaves the previous round authoritative.

### R5

Failure behavior is governed by the separately reviewed quorum and staleness
design.

## Security And Resource Limits

Implementations must:

- treat discovery data as untrusted and require authenticated node identity
  before capability, lease, or launch decisions;
- provide cluster initialization, explicitly bounded trusted-LAN enrollment,
  protected one-time provisioning, human-verified pairing, certificate
  issuance/renewal/revocation, and trust-state synchronization inside the
  mixlab binary without depending on an external CA, VPN, trust-store install,
  or cryptography command;
- confine provisional unauthenticated-root TLS to enrollment-only routes,
  disable 0-RTT/resumption there, bind verified phrases to TLS-exporter and the
  full request, and expose no generic certificate-verification bypass;
- separate certificate validation, application authorization, immutable
  membership, and allocation proof checks;
- require mutual TLS 1.3 and signed structured manifests for daemon job launch;
- require mutual TLS 1.3 with managed identities for every non-loopback
  coordinator connection except the narrowly scoped provisional/provisioned
  enrollment routes, which grant no run or artifact operation and become
  managed mutual TLS immediately after issuance;
- require mutually authenticated encryption for every collective byte stream
  crossing a non-loopback interface;
- scope workload credentials to a run/worker; seal any additional
  daemon-launched application secret through an encrypted job-bound envelope;
  and keep plaintext credentials out of manifests and artifact storage;
- require principal-bound `ArtifactAccessGrant` validation for every remote
  artifact operation and atomically enforce use/idempotency limits; a digest or
  same-run membership grants nothing;
- retain and validate complete `TrustEvidence` for durable signed records
  across ordinary certificate renewal/expiry; signer time is not a security
  anchor, prospective revocation preserves only previously journaled proofs,
  and compromise revocation invalidates all proofs for the compromised
  principal/certificate;
- validate provider-specific allocation proofs through an authorization port
  before admitting a learner, without exposing those fields to optimization
  code;
- bound JSON body, tensor artifact, tensor count, tensor rank, and dimension
  sizes;
- stream uploads and checksums rather than buffering complete models in RAM;
- use temporary files followed by atomic rename;
- reject symlinks and path traversal in filesystem storage;
- resolve macOS, Linux, and headless CUDA state through the same `~/.mixlab`
  logical layout; validate effective-user ownership and strict modes before
  opening context state; and never create cluster-authority state on an
  ordinary node or worker;
- use constant-time token comparison where appropriate;
- use cryptographically secure randomness for IDs, keys, nonces, enrollment
  request/window handles, provisioning secrets, and certificate serials;
- enforce certificate role, key usage, audience, lifetime, trust generation,
  and revocation rather than accepting any certificate chaining to the root;
- apply read/write deadlines and cancellable contexts;
- avoid logging secrets or protected storage/backend URLs;
- reject non-finite global parameters, deltas, and outer state;
- retain enough audit metadata to identify which updates formed a checkpoint;
  and
- never deserialize executable code from a worker.

## Expected Implementation Boundaries

These names are guidance for layout, but ownership and dependency direction are
normative. Existing top-level packages may be migrated incrementally; temporary
colocation must use internal packages/interfaces and may not create reverse
imports. Keep Go files below the repository's 1000-line limit.

### `statehome/`

- Leaf infrastructure for `-state-home`/`MIXLAB_STATE_HOME`/`~/.mixlab`
  precedence, canonical logical directory construction, strict ownership/mode/
  link validation, safe enumeration, ambiguity errors, and atomic staging.
- Returns resolved typed context paths to composition roots; it does not open
  or parse domain state. Cluster/principal/worker IDs and the principal-role
  namespace are opaque caller-supplied safe path segments, not state-home
  domain concepts.
- No trust policy, key generation, certificate handling, enrollment
  transitions, daemon/job recovery, learner recovery, artifact storage, or
  coordinator semantics.
- No imports from `trust`, `cluster`, `admission`, `recovery`, `coordinator`,
  `learner`, `train`, or `gpu`.

### `distributed/`

- Pure immutable DDP-group/cohort membership, member/learner identities,
  generation rules, launch-attempt identity, and canonical hashing.
- No daemon, coordinator, node-job, artifact, or telemetry protocol versions.
- No MLX, filesystem, network, trainer, sampler, or daemon implementation.

This is a leaf shared kernel.

### `trust/`

- Cluster-trust bounded context: cluster/root/certificate-issuer/
  snapshot-signer state, cluster fingerprint/phrase, purpose-separated
  certificate profiles, enrollment windows/requests/approval policies/
  invitations/receipts, certificate requests and issuance, authenticated
  renewal, trust snapshots, revocation, and workload-identity binding.
- Public immutable `PrincipalIdentity`, `AuthenticatedPrincipal`,
  `ClusterIdentity`, `EnrollmentRequest`, `EnrollmentChannelBinding`,
  `EnrollmentPresentation`, `ClientPhraseConfirmation`,
  `AdministratorConfirmation`,
  `EnrollmentApprovalRecord`, `EnrollmentInvitation`,
  `EnrollmentConsumptionReceipt`, `AuthorityEndpointSet`, `TrustSnapshot`,
  `TrustEvidence`, and `WorkloadIdentity` value contracts.
- Three closed approval-policy implementations that consume normalized
  channel/operator/provisioning evidence and all enter one issuance state
  machine. No policy can define a different certificate profile.
- `PrincipalSigner` and historical-proof verifier ports; context owners supply
  canonical bytes/digests, retain schema/commit-journal meaning, and never use a
  signer-provided timestamp as the revocation ordering anchor.
- Child adapters for X.509/profile verification, Ed25519/HPKE operations,
  macOS Keychain, strict-file key storage, and fake clock/randomness/key
  stores. Their base paths/handles are supplied by composition; trust does not
  resolve the user's home or enumerate state-home siblings. TLS connection
  mechanics remain transport adapters.
- `TrustAuthorityClient`, authority service, root-signed endpoint-set, and
  snapshot-distributor ports with local, remote, and fake adapters; target
  discovery remains outside the context.
- No discovery, address/interface inspection, TLS/exporter mechanics, phrase
  presentation, capability filtering, node lease/job state, membership
  selection, run admission, coordinator authorization, process launch,
  artifact storage, or training behavior.

The trust context may validate a caller-supplied context-owned binding before
issuing a credential, but it does not interpret or become authoritative for
the run, job, membership, or allocation that supplied the binding. Its
versioned workload scope stores opaque IDs and hashes and does not import
`distributed` or `diloco` models.

### `transport/`

- Normal managed TLS client/server adapters that yield
  `AuthenticatedPrincipal` only after trust validation.
- A separately scoped enrollment bootstrap adapter that performs full TLS 1.3
  handshakes, disables 0-RTT/resumption, exports
  `EnrollmentChannelBinding`, reports observed peer/local-interface metadata,
  and exposes only bounded enrollment routes.
- No mDNS browsing, approval-policy selection, identity-phrase acceptance,
  human approval, certificate issuance, lease/run authorization, or training
  behavior.

The provisional adapter may carry an untrusted proposed root to the enrollment
application but cannot install it. Only cluster trust, under the selected
policy and required confirmation evidence, returns an approved root/identity
for atomic principal-state storage.

### `artifact/`

- `ArtifactRef`, `ArtifactInfo`, and the narrow content-addressed
  `ArtifactStore` port.
- Streaming checksum/size enforcement and filesystem/S3-compatible adapters in
  implementation subpackages.
- Remote adapters require an authenticated principal plus context-issued
  `ArtifactAccessGrant` and call an authorization port; they do not infer
  access from a digest.
- In R3 the S3 adapter is coordinator-side only; workers never receive
  object-store URLs or bypass the mutual-TLS coordinator API.
- No DDP, learner-private, run, round, or resume semantic decisions.

### `repro/`

- Pure versioned initialization and stochastic-operation key derivation.
- No mutable global RNG, MLX initialization, discovery, or sampler ownership.

### `observability/`

- `EventEnvelope`, `EventSink`, bounded attribute rules, and sink/presentation
  adapters.
- Domain contexts define their event payloads; this package makes no control
  decisions.

### `arch/`

- `DistributedSpec`, defaults, validation, and config hashing.
- Distributed feature-compatibility checks.
- Program and weight-layout hashes using canonical serialization.

### `data/`

- Content-stable dataset identity.
- Counter-based sampler and serializable state.
- Rank/learner partitioning for flat and record-oriented datasets.

### `launch/`

- Backend-neutral `LaunchPlan` validation.
- MLX ring hostfile and NCCL/rank child-environment adapters.
- `SecureGroupTransportPlan` validation and a same-binary TLS 1.3
  proxy/transport adapter that exposes raw MLX endpoints only on loopback.
- Process startup/rendezvous helpers that do not initialize MLX or interpret
  training state.

Only this package materializes MLX launch inputs or transport endpoint
translation. It consumes immutable membership and trust-issued workload
identity values but does not issue certificates or choose members. Node
management calls it through an interface. It returns canonical plan bytes for
the controller application to sign and never opens controller or authority
private keys.

### `gpu/`

- MLX group initialization and collective wrappers.
- Deterministic gradient bucketing.
- `IRTrainer` gradient-accumulation and collective transaction stages.
- Atomic full-weight replacement while preserving inner optimizer state.
- C bridge additions for distributed metadata and operations.

This package consumes `distributed` values but does not import `cluster`,
`admission`, `coordinator`, `learner`, or network transport packages.

### `train/`

- Local single-process and DDP training orchestration.
- Consumption of immutable DDP/cohort membership without launcher/environment
  discovery.
- Loss-normalizer construction for supported objectives.
- DDP checkpoint manifest and semantic resume validation.
- Ports to `gpu`, `data`, `repro`, `artifact`, and `observability`.

This package does not contain daemon APIs, recruitment, run admission, the
DiLoCo coordinator, artifact-store implementations, or HTTP handlers.

### `cluster/discovery/`

- Discovery-provider interface and mDNS/DNS-SD, explicit-address, and fake
  adapters.
- Enrollment-source and enrolled-node service records as untrusted address/
  presentation hints only.

### `cluster/agent/`

- Local node profile, principal-aware node-operation authorization,
  authenticated capabilities, local dataset catalog, leases, durable daemon
  job state, credential-envelope handling, node-agent client/server protocol,
  and structured child supervision.
- Owns the daemon lifecycle trigger for principal renewal and persists attempt
  scheduling metadata; consumes trust renewal policy/issuance through
  `TrustAuthorityClient` and never implements it.
- Consumes authenticated principal, trust-snapshot, enrollment, and workload
  credential ports from `trust`; it does not issue, renew, or revoke
  certificates or parse TLS handshakes.
- Calls `launch`; does not encode MLX launch variables itself.

### `cluster/recruit/`

- Compatibility filtering, deterministic selection, reservation/preparation
  saga within the LAN provider, proposed DDP/learner assignments, and LAN
  prepared-allocation adapter.
- Consumes discovery and authenticated node-agent client ports.
- Does not define coordinator run plans or commit cohorts.

No `cluster` package imports `gpu`, `train`, or initializes MLX.

### `diloco/`

- Pure provider-neutral `LearnerSlot`, allocation-reference union,
  run-plan/commit, round-assignment, outer-update, and learner-private manifest
  models.
- Pure outer-gradient validation and Nesterov aggregation/update functions.
- No network, filesystem, MLX, discovery, node-agent, or process-launch code.

The global-coordination context owns these protocol meanings even though
learner and admission clients consume their public wire/value types.

### `coordinator/`

- Durable coordinator-side plan/admitting/commit states, staged run plans, run
  commits, fixed cohorts, round state machine, update admission, outer optimizer
  orchestration, global checkpoint semantics, and coordinator API.
- Ports for artifact storage/authorization, evaluation, allocation
  keeping/authorization, workload-credential requests, and events.
- Issuance/validation of run/worker/round-scoped `ArtifactAccessGrant` values;
  the blob adapter still owns byte-limit enforcement and publication.
- A coordinator API authorization adapter durably and atomically accounts for
  grant use/idempotency; global coordination remains the sole semantic grant
  issuer/validator.
- R3 failed, `PENDING_RECOVERY`, and active incarnation states, including the
  exact pending private-state grant and atomic recovery activation.
- Provider-specific allocation-keeper adapters may call node-agent or external
  provider clients, but optimizer/domain code sees only validated allocation
  status.

### `learner/`

- R2-R4 learner-runtime application service and coordinator client.
- Base-model verification/installation, inner-loop driving through `train`,
  learner-private checkpoint semantics, delta construction, artifact transfer,
  and idempotent outer-update submission.
- R4 learner-leader and per-rank private-state collection.

This package does not own cohort admission or global aggregation.
It requests and consumes `ArtifactAccessGrant` values and never issues or
semantically validates one.

### `admission/`

- The recoverable application saga composing recruitment/provisioner adapters,
  coordinator admission, cluster-trust workload-certificate requests through a
  port, optional context-owned credential sealing, worker start/registration,
  allocation-authority finalization, and compensation.
- Durable saga journal containing state versions, deadlines, IDs/hashes, and
  protected credential handles.
- External-enrollment provisioner adapter for manually started RunPod workers.

It does not define node leases, run-plan schema, learner training, or optimizer
state; it invokes their owning contexts through ports.

### `recovery/`

- R3 allocation-recovery application saga for an already committed external
  slot: recovery-grant acquisition, single-use recovery invitation,
  provider-proof revalidation, request-bound consumption receipt,
  old-credential fencing, `PENDING_RECOVERY` workload request, exact
  private-state restore, activation, and compensation.
- Durable journal contains only existing/new launch-attempt and credential
  generations, IDs/hashes, deadlines, and protected handles.
- Consumes coordinator, trust, external-provisioner, artifact-authorization,
  and event ports.

It cannot stage/commit a cohort, replace `AllocationRef`/`ParticipantID`/
`WorkerID`, issue certificates itself, or interpret learner-private tensors.

### `cmd/mixlab/`

- Cluster initialization/provisioning, trusted-LAN/verified enrollment-source
  hosting, enrollment client/pending-request UI, principal enrollment,
  revocation, optional trust-authority hosting, daemon, node-listing,
  submission, coordinator, worker, and external-worker recovery CLI modes and
  flags.
- Protected-file/environment credential loading and composition of concrete
  state-home, trust, key-store, normal/provisional TLS, discovery,
  phrase/prompt, daemon, launch, and application adapters.
- Mode-specific validation and help.

The CLI package is a composition root and presentation adapter. It displays
trust-computed values and returns explicit input through approval ports; it
does not compute phrase values or decide approval itself. It contains no
resource scheduling, lease, checkpoint, or optimizer domain logic.

### `docs/`

Each public field and flag is added to the canonical config and CLI references
in the same release. Add operational recipes only after the corresponding
acceptance gate passes.

### Required dependency direction

The intended import/use direction is:

```text
cmd/mixlab
  -> statehome (portable path resolution and validation only)
  -> trust application -> X.509/HPKE and key-store adapters
  -> transport -> trust authentication/enrollment ports
  -> cluster/discovery (enrollment and node address hints)
  -> admission -> cluster/recruit -> cluster/discovery
  |             |               -> cluster/agent client
  |             |               -> diloco value contracts
  |             -> coordinator client
  |             -> trust workload-issuer/principal-signer ports
  |             -> external-enrollment adapter
  -> recovery -> coordinator recovery client
  |           -> trust invitation/workload ports
  |           -> external-provider and artifact-authorization ports
  -> cluster/agent server -> trust authentication/policy ports
  |                       -> launch -> secure collective transport
  -> coordinator -> diloco -> distributed
  |              -> trust authenticated-principal/principal-signer ports
  |              -> artifact
  |              -> observability
  |              -> allocation-keeper adapters
  |                   -> cluster/agent client or external-provider client
  -> learner -> train -> gpu
             |      -> data
             |      -> repro
             |      -> artifact
             -> diloco

all contexts may emit through observability and use public immutable
identity/artifact values, but trust, distributed, artifact, and observability
packages never import their consumers
```

Automated dependency tests must reject reverse edges, particularly
`trust -> distributed/diloco/transport/cluster/admission/coordinator/launch`,
`distributed -> trust`, `cluster -> gpu/train`, `gpu ->
cluster/coordinator/trust`, `artifact -> train/coordinator/learner`,
`observability -> domain contexts`, `coordinator <-> admission` cycles, and
`coordinator <-> recovery` cycles, plus any
`statehome -> trust/cluster/admission/recovery/coordinator/learner/train/gpu`
edge.

## Test Strategy

### Default non-MLX CI

Must cover:

- config parsing/defaults/rejections;
- canonical hashes;
- DDP-group/cohort membership and node-job manifest canonicalization;
- membership-generation versus launch-attempt transition properties;
- required package dependency direction and forbidden reverse imports;
- state-home resolution precedence on macOS/Linux, canonical layout,
  zero/one/multiple-match behavior, context separation, owner/mode/link
  rejection, atomic enrollment publication/cleanup, and the invariant that
  node/worker initialization cannot create or open an authority directory;
- sampler partition and resume properties;
- versioned reproducibility key derivation;
- cluster initialization, certificate profile, canonical URI identities, and
  role/key-usage validation;
- Ed25519 signature fixtures, HPKE envelope wrong-recipient/tamper fixtures,
  workload critical-extension validation, signer role-to-purpose rejection,
  and CSPRNG failure propagation;
- cluster fingerprint/word-encoding fixtures and the common enrollment request/
  approval/issuance state machine across all three policies;
- trusted-LAN window scope/quota/TTL/rate/role enforcement and explicit TOFU
  with no fallback; provisioning entropy/expiry/single-consumption and pinned
  root/endpoint plus receipt/approval-record cross-binding; verified
  TLS-exporter/request-phrase binding and exact human confirmations;
- purpose-specific endpoint/audience binding, authenticated renewal, monotonic
  trust snapshots, and revocation;
- fake-clock principal-renewal scheduling, deterministic jitter, authority
  outage retry, expiry fail-closed, and explicit post-expiry reenrollment;
- authority endpoint-set validation, local/remote/fake
  `TrustAuthorityClient`, authority-unavailable errors, and invitation
  consumption recovery across authority crash/restart;
- distinct issuer/snapshot-signer/authority-TLS key-usage rejection,
  expired-snapshot snapshot-signer-only refresh, forbidden stale-node
  operations, bounded existing-job cancel/release, and post-refresh
  reauthentication;
- workload-certificate binding to run/job/member/rank/membership/audience;
- complete historical `TrustEvidence` validation after certificate
  renewal/expiry, prospective revocation only for pre-journaled proofs,
  all-history compromise revocation, forged backdating rejection, and signer
  revocation;
- secure key-store contracts for macOS Keychain handles and macOS/Linux
  strict-file storage, including headless operation with no keyring service;
- discovery-provider behavior with an injected fake;
- node identity, capability filtering, leases, and two-phase launch;
- launcher-adapter translation without MLX initialization in the daemon;
- secure-transport-plan canonicalization and loopback-only MLX endpoint
  translation without MLX initialization;
- credential-envelope binding, expiry, redaction, and protected-file handling;
- daemon restart reconciliation and structured child cancellation;
- staged run-plan/run-commit distinction, strictness, and atomic publication;
- run-admission crash/compensation at every transition;
- external allocation-recovery crash/compensation, old-incarnation fencing,
  request-bound invitation receipt, `PENDING_RECOVERY` operation denial, exact
  one-artifact read, activation after private-state restore, and exact identity
  preservation;
- provider-neutral LAN/external learner-slot validation;
- outer optimizer oracles;
- coordinator state machine;
- idempotency;
- artifact-store security, durable atomic grant expiry/use/idempotency bounds
  across retry/concurrency/restart, resumable-upload session accounting,
  coordinator-only S3 access, and cross-worker/cross-run private-artifact
  denial even when the digest is known;
- HTTP protocol and recovery;
- telemetry serialization; and
- single-process backward compatibility.

### MLX Metal tests

Must cover:

- singleton refactored trainer parity;
- two-process ring update parity;
- bucket boundaries and dtype handling;
- global non-finite skip;
- distributed resume; and
- atomic global weight replacement preserving AdamW state.

Tests requiring multiple hosts or optional data must skip with an explicit
reason when unavailable.

### CUDA tests

Run on a CUDA host before releasing CUDA changes:

- two-rank NCCL parity;
- CUDA custom-op capability checks;
- Metal/CUDA checkpoint interchange;
- multi-GPU learner outer-gradient parity; and
- multi-rank learner-private sampler/RNG restore;
- group-wide reject/commit behavior for global model replacement; and
- RunPod recovery smoke tests.

GitHub CI does not compile CUDA kernels, so a green default CI run is not
sufficient evidence.

### Network tests

Use local ephemeral ports and fault-injecting transports to test:

- wrong-root, wrong-role, expired, revoked, stale-trust, and valid-but-
  unauthorized TLS peers;
- invitation replay and concurrent consumption;
- enrollment-source discovery spoof/ambiguity, trusted-LAN in/out-of-scope
  peers and window closure/restart, provisional TLS MITM phrase mismatch,
  exporter/request substitution, concurrent verified approvals, and
  post-issuance mutual-TLS equivalence across all policies;
- local and remote authority selection, authority outage, expired-snapshot
  refresh, and recovery after authority/daemon restart;
- spoofed/stale discovery records and authentication failure;
- concurrent lease acquisition, renewal, and expiry;
- partial cohort reservation, preparation, and launch;
- coordinator successor-authority activation/finalization with failure at every
  transition;
- pre-commit admission cancellation and post-commit coordinator recovery;
- encrypted credential delivery, mutual coordinator/worker authentication, and
  rejection of plaintext non-loopback connections;
- secure collective handshake/streaming, exact peer-plan binding, loopback raw
  endpoints, tamper detection, helper failure, and credential cleanup;
- mixed `node_agent` and `external` allocation registration;
- external-worker recredential handshake, old-credential fencing, and
  unchanged committed participant/allocation proof, pending-only private-state
  grant, forbidden pending round/update operations, and atomic activation;
- artifact grant enforcement and cross-worker/cross-run denial with a known
  digest, including concurrent replay and coordinator restart;
- daemon restart and worker cancellation;
- truncated uploads;
- checksum mismatch;
- duplicate requests;
- coordinator restart;
- worker retry;
- connection loss;
- slow upload;
- deadline expiry; and
- stale or wrong-round submissions.

Tests must not depend on arbitrary sleeps.

## Release Discipline

For each release:

1. Lock the supported surface and experiment parameters before final evidence.
2. Implement pure correctness paths before overlap, compression, or
   performance shortcuts.
3. Preserve a single-process regression baseline.
4. Pass context-boundary and forbidden-dependency tests.
5. Produce the required automated and hardware evidence.
6. Update canonical documentation.
7. Stop at the review gate.

A later release may expand an earlier support matrix, but it may not weaken an
invariant or reinterpret an existing checkpoint/protocol format.

Any new cross-context command, event, value type, or dependency must be added to
the context/contract map in the same change. A context may evolve its own
versioned protocol, but another context may not duplicate or privately
reinterpret that protocol's authoritative model.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-25 | Treat R0 as an independent prerequisite rather than an architecture release. | The MLX upgrade and benchmark work can proceed immediately while the distributed contract is designed. |
| 2026-07-25 | Use synchronous DDP only inside one backend/connectivity domain. | No supported high-performance collective spans Metal and CUDA, and WAN lock-step would be dominated by latency and stragglers. |
| 2026-07-25 | Make DiLoCo the primary school-lab scaling mode. | Gigabit Ethernet cannot economically exchange full gradients every small mixlab step, but periodic parameter deltas amortize communication. |
| 2026-07-25 | Keep learner AdamW state private across outer rounds. | This follows DiLoCo and avoids tripling communication; private state is persisted for recovery rather than aggregated. |
| 2026-07-25 | Use explicit outer-gradient sign `base - local`. | Outer SGD with LR 1 then reduces to parameter averaging; locking the sign prevents a catastrophic but superficially plausible implementation error. |
| 2026-07-25 | Require fixed cohorts before elastic/asynchronous learners. | Correct global state, recovery, and convergence must be demonstrated before adding staleness and quorum semantics. |
| 2026-07-25 | Keep the global coordinator durable and GPU-independent. | RunPod workers are burst/preemptible capacity and must not own the only authoritative checkpoint. |
| 2026-07-26 | Separate dynamic discovery from dynamic membership in R1.1. | Machines may be found and reserved continuously while every launched R1 group and R2 cohort remains immutable, preserving collective and optimizer correctness. |
| 2026-07-26 | Treat mDNS as an untrusted address hint and authenticate the node-agent API. | Multicast discovery is spoofable and cannot authorize remote job launch. |
| 2026-07-26 | Keep the daemon outside the MLX process. | A long-lived control-plane service must survive jobs, isolate GPU state and crashes, and supervise only structured mixlab child modes. |
| 2026-07-26 | Separate DDP-group membership, learner-cohort membership, and launch-attempt identity. | R4 nests a DDP group inside one learner, while process restarts do not necessarily change either membership assignment. |
| 2026-07-26 | Add explicit learner-runtime and run-admission boundaries. | Local inner training/private state and cross-service cohort admission are distinct from both node recruitment and authoritative global optimization. |
| 2026-07-26 | Stage an immutable run plan and commit it with a separate atomic record. | Workers must bind to stable plan bytes before registration without creating a circular manifest hash or treating an incomplete admission as a run. |
| 2026-07-26 | Make learner allocation provider-neutral. | LAN daemons have node leases and job manifests, while manually provisioned RunPod workers do not; global optimization must not depend on either provider. |
| 2026-07-26 | Keep blob storage separate from checkpoint semantics. | Content-addressed storage validates bytes, while DDP, learner, and coordinator contexts own restore and admission meaning. |
| 2026-07-26 | Require mutual TLS for non-loopback coordinator traffic. | A school LAN is not a safe transport for plaintext bearer credentials or model artifacts, and both coordinator and worker require cryptographic identity. |
| 2026-07-26 | Embed a private mixlab trust domain in the single binary. | Users should enroll machines and RunPod workers without installing or maintaining a CA service, public certificate, VPN, system trust root, or second agent. |
| 2026-07-26 | Make cluster trust a bounded context separate from node management and transport. | Issuance and revocation policy, node leases, application authorization, and TLS mechanics have different state, failure modes, and reasons to change. |
| 2026-07-26 | Protect managed multi-host collectives with a same-binary secure transport adapter. | Membership agreement alone does not authenticate or protect raw MLX bytes on a shared school LAN; loopback confinement plus job-bound mutual TLS supplies peer authentication, integrity, and confidentiality without changing DDP semantics. |
| 2026-07-26 | Support local, transient, or optional background same-binary trust authority hosting. | Primary commands can issue and refresh credentials without external CA infrastructure, while secondary controllers use a narrow authenticated client and never receive issuer keys; authority outage fails closed and stale daemons retain only refresh/cancel/cleanup operations. |
| 2026-07-26 | Separate invitation purpose from certificate role. | Node/controller/coordinator enrollment creates principal credentials, while external bootstrap and recovery only authorize a later run-bound workload certificate through their owning admission/recovery workflow. |
| 2026-07-26 | Recredential recreated RunPod processes without replacing committed allocation identity. | A durable external participant/allocation and coordinator recovery grant preserve fixed-cohort semantics while fencing the failed process and allowing a new locally generated key. |
| 2026-07-26 | Bind artifact access to signed principal/run/worker grants. | Content digests identify bytes but must not expose another worker's private state or authorize uploads. |
| 2026-07-26 | Validate durable signatures with embedded trust evidence and journal ordering, never signer-supplied time. | Renewal/expiry should not break previously committed proofs; prospective revocation preserves only pre-journaled records, while compromise revocation invalidates all history for the compromised key because it could backdate signatures. |
| 2026-07-26 | Separate certificate issuance, snapshot signing, and authority TLS identities. | Stale-snapshot recovery must be verifiable from the pinned root without allowing a network endpoint key to mint certificates or trust state. |
| 2026-07-26 | Use a restricted `PENDING_RECOVERY` incarnation before activating a recreated external worker. | A replacement needs one exact private-state read before it can prove restoration, but must not acquire rounds, updates, or general artifact access first. |
| 2026-07-26 | Offer `trusted_lan`, `provisioned`, and `verified` as policies over one enrollment state machine. | Isolated labs need bounded zero-touch setup, unattended machines need one-use provisioning, and remote users need human-verifiable pairing; all must converge on identical managed identities without moving issuance into discovery, transport, or UI code. |
| 2026-07-26 | Derive the stable cluster phrase from the root public key and the request phrase from TLS-exporter plus the complete request. | The first lets people verify which cluster they reached; the second binds approval to one live connection and locally generated client key without turning either phrase into a bearer credential. |
| 2026-07-26 | Use `~/.mixlab` as the portable per-user state home on macOS and Linux. | One context-separated layout works for school Macs, headless CUDA hosts, and RunPod without root access or platform-specific packaging; a composition-owned resolver preserves domain boundaries, while Keychain and strict-file key stores remain replaceable adapters. |
