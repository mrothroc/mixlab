package clusterapp

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestNodeWorkloadPortsUseEnrolledProtectedIdentity(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	v, err := a.Current(ctx, testNow)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, v, testNow)
	check(t, err)
	defer i.Clear()
	stage, final := enrolleePaths(t, p)
	c, err := enrollee.Prepare(ctx, stage, final, a.Anchor, trust.Node, "file", testNow)
	check(t, err)
	defer func() { _ = c.Close() }()
	k, x, err := c.Keys()
	check(t, err)
	q, err := enrollment.NewRequest(i, k, x, testNow)
	check(t, err)
	r, err := a.Enrollment.Consume(ctx, q, i.Secret, v, testNow)
	check(t, err)
	check(t, c.CompleteProvisioned(ctx, i, q, r, testNow))
	node, err := principal.Open(final, testNow)
	check(t, err)
	defer func() { _ = node.Close() }()
	root, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "agent-credentials")}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	check(t, root.Ensure())
	ports, err := NodeWorkloadPorts(node, root, func() time.Time { return testNow })
	check(t, err)
	id := strings.Repeat("a", 32)
	scope := workload.Scope{Cluster: a.Anchor.Cluster(), Controller: id, Node: r.Approval.Principal, Lease: id, Run: id, Job: id, Attempt: id, Group: id, Member: id, Workload: id, Generation: 1, MembershipHash: strings.Repeat("a", 64), ManifestHash: strings.Repeat("b", 64), Created: testNow.Unix(), AdmitUntil: testNow.Add(time.Minute).Unix(), Deadline: testNow.Add(time.Hour).Unix()}
	check(t, ports.Initialize(ctx, scope))
	check(t, ports.Initialize(ctx, scope))
	intentPath := filepath.Join(root.Dir(), "job-"+scope.Job+"-"+scope.Attempt, "key-intent-workload.json")
	if _, err := os.Stat(intentPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("empty context initialization generated a key", err)
	}
	first, err := ports.Request(ctx, scope)
	check(t, err)
	again, err := ports.Request(ctx, scope)
	check(t, err)
	if !bytes.Equal(first.Request.CSR, again.Request.CSR) || first.Proof.Evidence.Principal != scope.Node {
		t.Fatal("request was not bound to protected node identity")
	}
	sign, err := first.Request.SigningRequest()
	check(t, err)
	_, err = trust.VerifyProof(a.Anchor, v, first.Proof, sign, testNow)
	check(t, err)
	other := scope
	other.Node = id
	if err := ports.Initialize(ctx, other); err == nil {
		t.Fatal("created a key context for another node")
	}
	other = scope
	other.Lease = strings.Repeat("b", 32)
	if err := ports.Initialize(ctx, other); err == nil {
		t.Fatal("rebound existing credential to another lease")
	}
	if _, err := ports.Request(ctx, other); err == nil {
		t.Fatal("requested credential for another lease")
	}
	cleanup := nodeagent.CleanupRequest{Job: scope.Job, Attempt: scope.Attempt, Workload: scope, WorkloadInitialized: true}
	check(t, DestroyNodeWorkload(ctx, root, a.Anchor, cleanup))
	check(t, DestroyNodeWorkload(ctx, root, a.Anchor, cleanup))
	if _, err := ports.Request(ctx, scope); err == nil {
		t.Fatal("cleanup allowed the prepared key to revive")
	}
	cleanup.Workload.Job, cleanup.Job = strings.Repeat("c", 32), strings.Repeat("c", 32)
	if err := DestroyNodeWorkload(ctx, root, a.Anchor, cleanup); err == nil {
		t.Fatal("missing initialized key directory treated as destroyed")
	}
	cleanup.WorkloadInitialized = false
	check(t, DestroyNodeWorkload(ctx, root, a.Anchor, cleanup))
	cleanup.Attempt = strings.Repeat("d", 32)
	if err := DestroyNodeWorkload(ctx, root, a.Anchor, cleanup); err == nil {
		t.Fatal("cleanup accepted mismatched attempt")
	}
}
