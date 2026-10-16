//go:build darwin || linux

package integration

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerjob"
)

// Rebind the existing TLS fixture to this real supervised executable before
// publishing a fresh coordinator repository. No immutable state is rewritten.
func drainRelayFixture(t *testing.T, build string) (*relayFixture, statehome.Path, json.RawMessage) {
	t.Helper()
	x := newLearnerRelay(t)
	config, err := json.Marshal(struct {
		Clock int64 `json:"clock"`
	}{x.f.clock().Unix()})
	check(t, err)
	x.s.Plan.Contract.ConfigHash, err = workerjob.Digest(json.RawMessage(config))
	check(t, err)
	x.s.Plan.Contract.BuildHash = build
	x.s.Plan.Outer.TotalSteps = x.s.Plan.Outer.InnerSteps
	x.s.Plan.SHA256, err = x.s.Plan.Digest()
	check(t, err)
	x.s.Commit.Binding.PlanHash = x.s.Plan.SHA256
	for i := range x.s.Commit.Registrations {
		x.s.Commit.Registrations[i].PlanHash = x.s.Plan.SHA256
		x.s.Commit.Registrations[i].Contract = x.s.Plan.Contract
	}
	x.s.Commit.Binding.CommitHash, err = x.s.Commit.Digest()
	check(t, err)
	for i := range x.s.Assignments {
		x.s.Assignments[i].Binding = x.s.Commit.Binding
		x.s.Assignments[i].Contract = x.s.Plan.Contract
	}
	repoPath := artifactPath(t)
	repo, err := coordinatorlocal.Open(repoPath)
	check(t, err)
	check(t, repo.Save(context.Background(), x.s.Plan.Cohort.RunID, 0, coordinator.Change{Next: x.s}))
	x.g.Repository = repo
	x.model.Repository = repo
	x.model.Allocations = relayAllocations{repo, x.s.Plan.Cohort.RunID}
	for i := range x.options {
		x.bindings[i].PlanHash = x.s.Plan.SHA256
		x.identities[i] = x.issue(t, x.bindings[i])
		policy, err := managedtls.New(x.identities[i], x.f.verify(trust.Coordinator, x.f.serverID), x.f.clock)
		check(t, err)
		x.options[i].Policy = func() (*managedtls.Policy, error) { return policy, nil }
		x.options[i].Plan, x.options[i].Commit, x.options[i].Workload = x.s.Plan, *x.s.Commit, x.bindings[i]
		x.clients[i], err = clusterapp.NewLearnerClientWithPolicy(x.options[i])
		check(t, err)
	}
	return x, repoPath, config
}

func drainWorkerDirectory(t *testing.T) statehome.Path {
	t.Helper()
	base, err := filepath.EvalSymlinks(os.TempDir())
	check(t, err)
	if len(base) > 50 {
		base = "/private/tmp"
	}
	dir, err := os.MkdirTemp(base, "drain-")
	check(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	check(t, err)
	return p
}
