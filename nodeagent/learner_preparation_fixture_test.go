package nodeagent

import (
	"context"
	"crypto"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workerprobe"
)

type learnerPreparationFixture struct {
	s         *Store
	actor     trust.AuthenticatedPrincipal
	anchor    trust.Anchor
	view      trust.VerifiedSnapshot
	chain     [][]byte
	key       crypto.Signer
	now       time.Time
	profile   Profile
	reserve   Reserve
	lease     Lease
	manifest  nodejob.LearnerManifest
	accepted  nodejob.AcceptedLearner
	authority statehome.Path
	material  bootstrap.AuthorityMaterial
}

func preparationCheck(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func newLearnerPreparationFixture(t *testing.T) *learnerPreparationFixture {
	t.Helper()
	f := &learnerPreparationFixture{now: leaseNow}
	f.s, f.actor = leaseFixture(t)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	preparationCheck(t, err)
	path := func(name string, kind statehome.Kind) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, name)}, statehome.Context{Kind: kind})
		preparationCheck(t, err)
		return p
	}
	ca := path("authority-root", statehome.Authority)
	cfg := bootstrap.Config{Cluster: f.s.cluster, Authority: ca, Backend: "file", Endpoint: "https://authority.example:7443", Audience: "preparation-test"}
	for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
		id := []string{strings.Repeat("e", 32), f.actor.Principal, strings.Repeat("f", 32)}[i]
		cfg.Principals = append(cfg.Principals, bootstrap.Target{Role: role, Principal: id, Final: path(string(role), statehome.Principal), Staging: path("stage-"+string(role), statehome.Enrollment)})
	}
	_, err = bootstrap.Initialize(context.Background(), cfg, f.now)
	preparationCheck(t, err)
	material, err := bootstrap.LoadAuthority(ca, f.now)
	preparationCheck(t, err)
	f.authority, f.material = ca, material
	f.anchor = material.Anchor
	f.view, err = trust.VerifySnapshot(f.anchor, material.InitialSnapshot, f.now)
	preparationCheck(t, err)
	controller, err := principal.Open(cfg.Principals[1].Final, f.now)
	preparationCheck(t, err)
	t.Cleanup(func() { preparationCheck(t, controller.Close()) })
	identity, key, err := controller.Active(f.now)
	preparationCheck(t, err)
	f.chain, f.key = identity.Chain, key
	f.actor, err = trust.AuthenticatePrincipal(f.anchor, f.view, f.chain, f.now)
	preparationCheck(t, err)
	f.profile = profileFixture(f.s)
	f.profile.Datasets = append(f.profile.Datasets, LocalDataset{Dataset: Dataset{Selector: "toy.validation", ID: strings.Repeat("c", 64)}, TrainPattern: "/private/local/validation.bin"})
	preparationCheck(t, f.s.InstallProfile(context.Background(), f.profile))
	f.reserve = reservation("1")
	f.reserve.TTLSeconds = 300
	f.lease, err = f.s.Reserve(context.Background(), f.actor, f.reserve, f.now)
	preparationCheck(t, err)
	var plan run.Plan
	b, err := os.ReadFile("../diloco/testdata/plan.json")
	preparationCheck(t, err)
	preparationCheck(t, json.Unmarshal(b, &plan))
	member := distributed.LearnerMember{WorkerID: strings.Repeat("2", 32), ParticipantID: f.s.node, IslandID: "metal", DatasetPartitionID: "partition0"}
	plan.Cohort, err = distributed.NewLearnerCohort(f.lease.Run, 1, []distributed.LearnerMember{member})
	preparationCheck(t, err)
	allocation := run.Allocation{Provider: "node_agent", NodeID: f.s.node, LeaseID: f.lease.ID, JobID: strings.Repeat("3", 32)}
	allocation.RefHash, err = run.Digest(allocation)
	preparationCheck(t, err)
	plan.Slots = []run.Slot{{Member: member, CapabilityHash: strings.Repeat("a", 64), Allocation: allocation}}
	plan.Coordinator = run.Coordinator{ClusterID: f.s.cluster, PrincipalID: cfg.Principals[2].Principal, Role: "coordinator", Audience: "preparation"}
	plan.CreatedAt, plan.ExpiresAt = f.now.Unix(), f.now.Add(10*time.Minute).Unix()
	config := json.RawMessage(`{"model_dim":16}`)
	plan.Contract.ConfigHash = nodejob.Hash(config)
	plan.Contract.DatasetHash = f.profile.Datasets[0].ID
	plan.SHA256, err = plan.Digest()
	preparationCheck(t, err)
	f.manifest = nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: f.actor.Principal, Attempt: strings.Repeat("4", 32), WorkloadGeneration: 1, Plan: plan, Slot: plan.Slots[0], Config: config, DatasetSelector: f.profile.Datasets[0].Selector, Limits: f.profile.Limits}
	f.accepted = f.accept(t, f.manifest)
	return f
}

func (f *learnerPreparationFixture) accept(t *testing.T, m nodejob.LearnerManifest) nodejob.AcceptedLearner {
	t.Helper()
	q, err := m.SigningRequest()
	preparationCheck(t, err)
	proof, err := trust.SignPrincipalProof(f.anchor, f.chain, f.key, f.view, q, f.now)
	preparationCheck(t, err)
	accepted, err := nodejob.AcceptLearner(f.anchor, f.view, f.actor, f.s.node, nodejob.SignedLearner{Manifest: m, Proof: proof}, f.now)
	preparationCheck(t, err)
	return accepted
}

func (f *learnerPreparationFixture) clock() time.Time { return f.now }

func (f *learnerPreparationFixture) freshView(t *testing.T, revocations []trust.Revocation) trust.VerifiedSnapshot {
	t.Helper()
	keys, err := securekeys.OpenSelected(f.material.SnapshotKey.Backend, f.authority, f.material.SnapshotKey.Scope)
	preparationCheck(t, err)
	defer func() { preparationCheck(t, keys.Close()) }()
	key, err := keys.Signer(f.material.SnapshotKey)
	preparationCheck(t, err)
	payload := f.material.InitialSnapshot.Payload
	payload.Generation++
	payload.IssuedAt, payload.ExpiresAt = f.now.Unix(), f.now.Add(trust.SnapshotLifetime).Unix()
	payload.Revocations = revocations
	signed, err := trust.SignSnapshot(f.anchor, payload, f.material.SnapshotSigner, key, f.now)
	preparationCheck(t, err)
	view, err := trust.VerifySnapshot(f.anchor, signed, f.now)
	preparationCheck(t, err)
	return view
}

func (f *learnerPreparationFixture) ports() PreparationPorts {
	return PreparationPorts{Clock: f.clock, Probe: func(context.Context) (workerprobe.Report, error) { return f.profile.Probe, nil }, DatasetID: func(_ context.Context, path string) (string, error) {
		for _, d := range f.profile.Datasets {
			if d.TrainPattern == path {
				return d.ID, nil
			}
		}
		return "", os.ErrNotExist
	}}
}

func (f *learnerPreparationFixture) numerical() workerprobe.Plan {
	c := f.manifest.Plan.Contract
	return workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: c.BuildHash, ConfigHash: c.ConfigHash, ProgramHash: c.ProgramHash, WeightLayoutHash: c.WeightLayoutHash, OptimizerHash: c.InnerOptimizerHash, DType: "fp32", Learner: &workerprobe.LearnerPlan{Outer: f.manifest.Plan.Outer, BatchTokens: 16, SequenceLength: 4, Precision: &precision.Policy{ComputeDType: "float32", Backend: "metal"}}}
}
