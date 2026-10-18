package integration

import (
	"context"
	"crypto"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerprobe"
)

type recoveryNodeTrust struct {
	mu      sync.Mutex
	view    *trust.VerifiedSnapshot
	refresh func(context.Context) error
}

func (n *recoveryNodeTrust) pin(v trust.VerifiedSnapshot) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.view = &v
}

func (n *recoveryNodeTrust) current(f *tlsFixture) trust.VerifiedSnapshot {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.view != nil {
		return *n.view
	}
	return f.view
}

// Give rollover tests an independently cached node view without changing the
// shared admission fixture or introducing a package-global trust override.
func installRecoveryNodeTrust(t *testing.T, x *learnerHTTPFixture, n *recoveryNodeTrust) {
	t.Helper()
	f := x.f
	ctx := context.Background()
	root := learnerHTTPPath(t, statehome.Agent)
	cert := x.policy.ServerConfig().Certificates[0]
	sign := func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(f.a, cert.Certificate, cert.PrivateKey.(crypto.Signer), n.current(f), q, f.clock())
	}
	credentialPath := func(scope workload.Scope) (statehome.Path, error) {
		return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root.Dir(), "job-"+scope.Job+"-"+scope.Attempt)}, statehome.Context{Kind: statehome.Agent})
	}
	wp := nodeagent.WorkloadPorts{Clock: f.clock, Initialize: func(ctx context.Context, scope workload.Scope) error {
		x.initializations.Add(1)
		p, err := credentialPath(scope)
		if err != nil {
			return err
		}
		c, err := nodecredentials.InitializeTransport(ctx, p, f.a, scope, "file")
		if err != nil {
			return err
		}
		return c.Close()
	}, Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		p, err := credentialPath(scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		c, err := nodecredentials.OpenTransport(p, f.a, scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		defer func() { _ = c.Close() }()
		return c.PrepareRequest(ctx, sign, n.current(f), f.clock())
	}, AuthorizeRequest: func(ctx context.Context, scope workload.Scope, original workload.SignedRequest) (workload.RequestAuthorization, error) {
		p, err := credentialPath(scope)
		if err != nil {
			return workload.RequestAuthorization{}, err
		}
		return authorizeLearnerHTTPWorkload(ctx, p, f.a, n.current(f), sign, original, f.clock())
	}}
	plan := x.prepare.Signed.Manifest.Plan
	c := plan.Contract
	probe := workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: c.BuildHash, ConfigHash: c.ConfigHash, ProgramHash: c.ProgramHash, WeightLayoutHash: c.WeightLayoutHash, OptimizerHash: c.InnerOptimizerHash, DType: "fp32", Learner: &workerprobe.LearnerPlan{Outer: plan.Outer, BatchTokens: 16, SequenceLength: 4, Precision: &precision.Policy{ComputeDType: "float32", Backend: "metal"}}}
	jobs, err := clusterapp.NewNodeJobs(ctx, clusterapp.NodeJobOptions{
		Store: x.store, Node: x.capabilities.Node, Anchor: f.a, CredentialRoot: root,
		Workload: wp, RelayAddress: "127.0.0.1:32499", Clock: f.clock,
		CurrentTrust: func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
			return f.a, n.current(f), nil
		}, RefreshTrust: n.refresh, SignRecovery: sign,
		Preparation: nodeagent.PreparationPorts{Clock: f.clock,
			Probe: func(context.Context) (workerprobe.Report, error) { return x.capabilities.Probe, nil },
			DatasetID: func(_ context.Context, path string) (string, error) {
				if path != "/private/local/train.bin" {
					return "", fmt.Errorf("nonlocal dataset path")
				}
				return x.requirements.DatasetID, nil
			}, ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return fmt.Errorf("unexpected artifact") }},
		LearnerProbe: func(context.Context, []byte) (workerprobe.Plan, error) { return probe, nil },
	})
	check(t, err)
	h, err := clusterapp.NodeManagementHandler(x.store, x.policy, f.clock, learnerHTTPJobs{jobs, &x.calls})
	check(t, err)
	x.url = serveCapabilities(t, x.policy, h).URL
}
