package recruitment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// These fakes test saga ordering and durable compensation. Cryptographic
// admission and real HTTP/worker cleanup are covered by trust/integration.
type launchFixture struct {
	t                                     *testing.T
	launch                                *Launch
	plan                                  LaunchPlan
	ports                                 LaunchPorts
	leases                                map[string]nodeagent.Lease
	jobs                                  map[string]nodeagent.Job
	requests                              map[string]nodeagent.Reserve
	activateCount, startCount, issueCount int
	fail                                  string
	failed                                bool
}

func newLaunchFixture(t *testing.T) *launchFixture {
	t.Helper()
	r, _ := fixture()
	plan := LaunchPlan{Cluster: r.Cluster, Controller: strings.Repeat("f", 32), Run: strings.Repeat("d", 32), Attempt: strings.Repeat("e", 32), TTLSeconds: 300, Selection: Selection{Policy: SelectionPolicy, Requirements: r}}
	for i := range 2 {
		_, o := fixture()
		o.Capabilities.Node = strings.Repeat(fmt.Sprint(i+1), 32)
		o.Capabilities.TransportEndpoint = fmt.Sprintf("127.0.0.%d:7446", i+1)
		plan.Selection.Selected = append(plan.Selection.Selected, Candidate{fmt.Sprintf("node%d.local:7445", i), o.Capabilities})
	}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "attempt")}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	l, err := BeginLaunch(path, plan)
	if err != nil {
		t.Fatal(err)
	}
	x := &launchFixture{t: t, launch: l, plan: plan, leases: map[string]nodeagent.Lease{}, jobs: map[string]nodeagent.Job{}, requests: map[string]nodeagent.Reserve{}}
	x.ports = LaunchPorts{Clock: func() time.Time { return selectionNow }, Reserve: x.reserve, LeaseStatus: x.leaseStatus, AbortReservation: x.abortReservation,
		Renew: func(context.Context, Candidate, nodeagent.LeaseCommand) (nodeagent.Lease, error) {
			return nodeagent.Lease{}, errors.New("unexpected renewal")
		}, BuildJobs: x.buildJobs, Prepare: x.prepare, Grant: func(_ context.Context, q workload.SignedRequest) (workload.Grant, error) {
			return workload.Grant{Request: q}, x.inject("grant")
		}, Issue: x.issue, BuildTransport: x.buildTransport, Activate: x.activate, Start: x.start, Status: x.status}
	return x
}
func (x *launchFixture) inject(point string) error {
	if x.fail == point && !x.failed {
		x.failed = true
		return errors.New("injected lost response at " + point)
	}
	return nil
}
func (x *launchFixture) reserve(ctx context.Context, c Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
	if err := ctx.Err(); err != nil {
		return nodeagent.Lease{}, err
	}
	_, r, err := x.launch.load()
	if err != nil {
		x.t.Fatal(err)
	}
	found := false
	for _, n := range r.Nodes {
		if n.Candidate.Capabilities.Node == c.Capabilities.Node {
			found = n.Touched && n.Reserve == q
		}
	}
	if !found {
		x.t.Fatal("remote reserve before durable intent")
	}
	if previous, ok := x.requests[c.Capabilities.Node]; ok {
		if previous != q {
			x.t.Fatal("ambiguous reservation retried with new identity")
		}
		return x.leases[c.Capabilities.Node], nil
	}
	x.requests[c.Capabilities.Node] = q
	l := nodeagent.Lease{Format: nodeagent.LeaseVersion, ID: c.Capabilities.Node, Node: c.Capabilities.Node, Controller: x.plan.Controller, Run: q.Run, CapabilityGeneration: q.CapabilityGeneration, Version: 1, State: nodeagent.Reserved, Created: selectionNow.Unix(), Expires: selectionNow.Add(300 * time.Second).Unix(), RenewBy: selectionNow.Add(150 * time.Second).Unix()}
	x.leases[c.Capabilities.Node] = l
	return l, x.inject("reserve" + c.Capabilities.Node[:1])
}
func (x *launchFixture) leaseStatus(_ context.Context, c Candidate, id string) (nodeagent.Lease, error) {
	l := x.leases[c.Capabilities.Node]
	if l.ID != id {
		return l, errors.New("unknown lease")
	}
	if l.State == nodeagent.Releasing {
		l.State = nodeagent.Released
		l.Version++
		x.leases[c.Capabilities.Node] = l
	}
	return l, nil
}
func (x *launchFixture) abortReservation(_ context.Context, c Candidate, q nodeagent.Reserve) (nodeagent.ReservationAbort, error) {
	_, r, err := x.launch.load()
	if err != nil {
		x.t.Fatal(err)
	}
	if r.Phase != "aborting" {
		x.t.Fatal("release without durable abort")
	}
	found := false
	for _, n := range r.Nodes {
		if n.Touched && n.Reserve == q {
			found = true
		}
	}
	if !found {
		x.t.Fatal("release not recorded before send")
	}
	b, _ := json.Marshal(q)
	out := nodeagent.ReservationAbort{Node: c.Capabilities.Node, Controller: x.plan.Controller, RequestHash: nodejob.Hash(b)}
	if l, ok := x.leases[c.Capabilities.Node]; ok {
		if l.State != nodeagent.Released {
			l.State = nodeagent.Releasing
			l.Version++
			x.leases[c.Capabilities.Node] = l
		}
		out.Lease = &l
	}
	return out, x.inject("release")
}
func (x *launchFixture) buildJobs(_ context.Context, p LaunchPlan, leases []nodeagent.Lease) ([]nodejob.Signed, error) {
	if err := x.inject("jobs"); err != nil {
		return nil, err
	}
	var members []distributed.DDPGroupMember
	var assigned []nodejob.Member
	for i, c := range p.Selection.Selected {
		members = append(members, distributed.DDPGroupMember{MemberID: c.Capabilities.Node, Rank: i})
		assigned = append(assigned, nodejob.Member{Node: c.Capabilities.Node, MemberID: c.Capabilities.Node, Rank: i})
	}
	membership, err := distributed.NewDDPGroupMembership(p.Run, strings.Repeat("c", 32), 1, "ring", members)
	if err != nil {
		x.t.Fatal(err)
	}
	var out []nodejob.Signed
	for i, l := range leases {
		config := json.RawMessage(`{"model_dim":16}`)
		r := p.Selection.Requirements
		m := nodejob.Manifest{Version: nodejob.Version, Job: l.Node, Attempt: p.Attempt, Lease: l.ID, Node: l.Node, Controller: p.Controller, Nonce: l.Node, Membership: membership, Members: assigned, Rank: i, BuildID: r.BuildID, Config: config, ConfigHash: nodejob.Hash(config), ProgramHash: r.BuildID, WeightLayoutHash: r.BuildID, OptimizerHash: r.BuildID, DatasetSelector: r.DatasetSelector, DatasetID: r.DatasetID, Artifacts: []nodejob.ArtifactRef{}, Mode: "arch", Transport: "tls13-ring", Limits: r.Limits, Created: selectionNow.Unix(), Expires: selectionNow.Add(time.Minute).Unix()}
		q, err := m.SigningRequest()
		if err != nil {
			x.t.Fatal(err)
		}
		out = append(out, nodejob.Signed{Manifest: m, Proof: trust.SignedProof{Request: q}})
	}
	return out, nil
}
func (x *launchFixture) prepare(_ context.Context, c Candidate, version uint64, s nodejob.Signed) (nodeagent.PreparedWorkload, error) {
	l := x.leases[c.Capabilities.Node]
	if l.Version != version {
		x.t.Fatal("wrong lease version")
	}
	l.State = nodeagent.Prepared
	l.Version++
	l.Job = s.Manifest.Job
	x.leases[l.Node] = l
	j := nodeagent.Job{Format: nodeagent.JobVersion, ID: s.Manifest.Job, Attempt: s.Manifest.Attempt, Lease: l.ID, Controller: l.Controller, ManifestHash: s.Proof.Request.Digest, Version: 3, State: nodeagent.JobPrepared}
	x.jobs[l.Node] = j
	scope, err := s.Manifest.WorkloadScope(x.plan.Cluster, l.Node)
	if err != nil {
		x.t.Fatal(err)
	}
	return nodeagent.PreparedWorkload{Job: j, Request: workload.SignedRequest{Request: workload.Request{Version: workload.Version, Scope: scope}}}, x.inject("prepare" + c.Capabilities.Node[:1])
}
func (x *launchFixture) issue(_ context.Context, g workload.Grant) (workload.Result, error) {
	x.issueCount++
	s := g.Request.Request.Scope
	b := trust.WorkloadBinding{Version: trust.WorkloadBindingVersion, Cluster: s.Cluster, Role: trust.Worker, Principal: s.Workload, Participant: s.Node, Run: s.Run, Job: s.Job, Lease: s.Lease, ManifestHash: s.ManifestHash, Attempt: s.Attempt, Audience: workload.Audience, IssuedAt: s.Created, ExpiresAt: s.Deadline, Group: s.Group, Generation: s.Generation, MembershipHash: s.MembershipHash, Member: s.Member, Rank: s.Rank}
	return workload.Result{Binding: b, Chain: [][]byte{[]byte(s.Node), []byte("issuer"), []byte("root")}}, x.inject("issue")
}
func (x *launchFixture) buildTransport(_ context.Context, p LaunchPlan, jobs []nodejob.Signed, results []workload.Result) (grouptransport.Signed, error) {
	plan := grouptransport.Plan{Version: grouptransport.Version, TLSPolicy: grouptransport.TLSPolicy, Cluster: p.Cluster, Controller: p.Controller, Attempt: p.Attempt, Nonce: p.Attempt, Membership: jobs[0].Manifest.Membership, LoopbackPorts: []int{22000, 22001}, Created: selectionNow.Unix(), Expires: results[0].Binding.ExpiresAt}
	for i, j := range jobs {
		r := results[i]
		plan.Members = append(plan.Members, grouptransport.Member{Node: j.Manifest.Node, Job: j.Manifest.Job, MemberID: j.Manifest.Members[i].MemberID, Rank: i, ManifestHash: j.Proof.Request.Digest, Endpoint: p.Selection.Selected[i].Capabilities.TransportEndpoint, CertificateHash: nodejob.Hash(r.Chain[0]), Chain: r.Chain, Binding: r.Binding})
	}
	q, err := plan.SigningRequest()
	if err != nil {
		x.t.Fatal(err)
	}
	return grouptransport.Signed{Plan: plan, Proof: trust.SignedProof{Request: q}}, x.inject("transport")
}
func (x *launchFixture) activate(_ context.Context, c Candidate, v uint64, p grouptransport.Signed, _ workload.Grant, _ workload.Result) (nodeagent.Job, error) {
	x.activateCount++
	j := x.jobs[c.Capabilities.Node]
	if j.Version != v {
		x.t.Fatal("wrong preparation version")
	}
	j.Version++
	j.TransportHash = p.Proof.Request.Digest
	x.jobs[c.Capabilities.Node] = j
	return j, x.inject("activate" + c.Capabilities.Node[:1])
}
func (x *launchFixture) start(_ context.Context, c Candidate, q nodeagent.StartCommand) (nodeagent.Job, error) {
	if x.activateCount != 2 {
		x.t.Fatal("start before complete transport acknowledgement")
	}
	_, r, err := x.launch.load()
	if err != nil {
		x.t.Fatal(err)
	}
	for _, n := range r.Nodes {
		if n.Activated == nil {
			x.t.Fatal("activation barrier not durable")
		}
	}
	x.startCount++
	j := x.jobs[c.Capabilities.Node]
	if j.ID != q.Job || j.Version != q.ExpectedVersion {
		x.t.Fatal("wrong start")
	}
	j.State = nodeagent.JobRunning
	j.Version++
	x.jobs[c.Capabilities.Node] = j
	return j, x.inject("start" + c.Capabilities.Node[:1])
}
func (x *launchFixture) status(_ context.Context, c Candidate, id string) (nodeagent.Job, error) {
	j := x.jobs[c.Capabilities.Node]
	if j.ID != id {
		x.t.Fatal("wrong status")
	}
	if err := x.inject("status"); err != nil {
		return j, err
	}
	j.State = nodeagent.JobExited
	x.jobs[c.Capabilities.Node] = j
	l := x.leases[c.Capabilities.Node]
	l.State = nodeagent.Released
	l.Version++
	x.leases[c.Capabilities.Node] = l
	return j, nil
}

func TestLaunchAllOrAbortAndLostResponses(t *testing.T) {
	for _, failure := range []string{"", "reserve1", "reserve2", "jobs", "prepare1", "prepare2", "grant", "issue", "transport", "activate1", "activate2", "start1", "start2", "status"} {
		t.Run(failure, func(t *testing.T) {
			x := newLaunchFixture(t)
			x.fail = failure
			err := x.launch.Run(context.Background(), x.ports)
			if (err != nil) != (failure != "") {
				t.Fatal(failure, err)
			}
			_, r, err := x.launch.load()
			if err != nil {
				t.Fatal(err)
			}
			want := "succeeded"
			if failure != "" {
				want = "aborted"
			}
			if r.Phase != want {
				t.Fatal(r.Phase, want)
			}
			for _, l := range x.leases {
				if l.State != nodeagent.Released {
					t.Fatal("leaked reservation", l)
				}
			}
			before := x.startCount
			reopened, err := OpenLaunch(x.launch.path)
			if err != nil {
				t.Fatal(err)
			}
			_ = reopened.Run(context.Background(), x.ports)
			if x.startCount != before {
				t.Fatal("terminal attempt restarted")
			}
		})
	}
}

func TestLaunchInterruptedReserveRecoveryAndMissingJournal(t *testing.T) {
	x := newLaunchFixture(t)
	old, r, err := x.launch.load()
	if err != nil {
		t.Fatal(err)
	}
	r.Phase = "reserving"
	r.Nodes[0].Touched = true
	if err := x.launch.save(&old, r); err != nil {
		t.Fatal(err)
	}
	_, err = x.reserve(context.Background(), r.Nodes[0].Candidate, r.Nodes[0].Reserve)
	if err != nil {
		t.Fatal(err)
	}
	// Crash before storing the returned lease. Recovery must exact-retry reserve,
	// then compensate; it must never prepare or start even the first rank.
	if err := x.launch.Run(context.Background(), x.ports); err == nil {
		t.Fatal("interrupted launch reported success")
	}
	if x.startCount != 0 || len(x.jobs) != 0 {
		t.Fatal("recovery continued launching")
	}
	_, r, err = x.launch.load()
	if err != nil || r.Phase != "aborted" {
		t.Fatal(r.Phase, err)
	}
	if err := os.Remove(filepath.Join(x.launch.path.Dir(), launchFile)); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenLaunch(x.launch.path); err == nil {
		t.Fatal("opened missing published journal")
	}
	if _, err := BeginLaunch(x.launch.path, x.plan); err == nil {
		t.Fatal("reset missing published journal")
	}
}

func TestLaunchCleanupRetriesLostReleaseAndWaitsForProof(t *testing.T) {
	x := newLaunchFixture(t)
	old, r, err := x.launch.load()
	if err != nil {
		t.Fatal(err)
	}
	r.Phase = "reserving"
	r.Nodes[0].Touched = true
	if err := x.launch.save(&old, r); err != nil {
		t.Fatal(err)
	}
	l, err := x.reserve(context.Background(), r.Nodes[0].Candidate, r.Nodes[0].Reserve)
	if err != nil {
		t.Fatal(err)
	}
	r.Nodes[0].Lease = &l
	if err := x.launch.save(&old, r); err != nil {
		t.Fatal(err)
	}
	x.fail = "release"
	if err := x.launch.Abort(context.Background(), x.ports); err != nil {
		t.Fatal(err)
	}
	_, r, err = x.launch.load()
	if err != nil || r.Phase != "aborted" || !r.Nodes[0].Released {
		t.Fatal(r, err)
	}
	if !reflect.DeepEqual(x.requests[l.Node], r.Nodes[0].Reserve) {
		t.Fatal("reservation changed")
	}
}

func TestLaunchDoesNotStartAnotherRankAfterPeerLeaseEnds(t *testing.T) {
	x := newLaunchFixture(t)
	start := x.ports.Start
	x.ports.Start = func(ctx context.Context, c Candidate, q nodeagent.StartCommand) (nodeagent.Job, error) {
		j, err := start(ctx, c, q)
		l := x.leases[c.Capabilities.Node]
		l.State = nodeagent.Releasing
		x.leases[l.Node] = l
		return j, err
	}
	if err := x.launch.Run(context.Background(), x.ports); err == nil {
		t.Fatal("partial cohort reported success")
	}
	if x.startCount != 1 {
		t.Fatal("started another rank after peer termination", x.startCount)
	}
}

func TestLaunchRejectsExpiredOrUncommittedRenewal(t *testing.T) {
	for _, invalid := range []string{"expired", "unchanged-version", "released"} {
		t.Run(invalid, func(t *testing.T) {
			x := newLaunchFixture(t)
			original := x.ports.LeaseStatus
			x.ports.LeaseStatus = func(ctx context.Context, c Candidate, id string) (nodeagent.Lease, error) {
				l, err := original(ctx, c, id)
				l.RenewBy = selectionNow.Unix()
				return l, err
			}
			x.ports.Renew = func(_ context.Context, c Candidate, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
				l := x.leases[c.Capabilities.Node]
				l.Version++
				switch invalid {
				case "expired":
					l.Expires = selectionNow.Unix()
				case "unchanged-version":
					l.Version = q.ExpectedVersion
				case "released":
					l.State = nodeagent.Released
				}
				return l, nil
			}
			if err := x.launch.Run(context.Background(), x.ports); err == nil {
				t.Fatal("accepted bad renewal")
			}
			if len(x.jobs) != 0 {
				t.Fatal("prepared after invalid renewal")
			}
		})
	}
}

func TestLaunchAbortRetainsUnreachableCompensation(t *testing.T) {
	x := newLaunchFixture(t)
	old, r, err := x.launch.load()
	if err != nil {
		t.Fatal(err)
	}
	r.Phase = "reserving"
	r.Nodes[0].Touched = true
	if err := x.launch.save(&old, r); err != nil {
		t.Fatal(err)
	}
	original := x.ports.AbortReservation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	x.ports.AbortReservation = func(context.Context, Candidate, nodeagent.Reserve) (nodeagent.ReservationAbort, error) {
		cancel()
		return nodeagent.ReservationAbort{}, errors.New("node unreachable")
	}
	if err := x.launch.Abort(ctx, x.ports); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	_, r, err = x.launch.load()
	if err != nil || r.Phase != "aborting" || r.Nodes[0].Released {
		t.Fatal(r.Phase, err)
	}
	x.ports.AbortReservation = original
	if err := x.launch.Abort(context.Background(), x.ports); err != nil {
		t.Fatal(err)
	}
	if len(x.leases) != 0 {
		t.Fatal("recovery allocated a new reservation")
	}
}

func TestLaunchRejectsRewoundLeaseVersion(t *testing.T) {
	x := newLaunchFixture(t)
	original := x.ports.Reserve
	x.ports.Reserve = func(ctx context.Context, c Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
		l, err := original(ctx, c, q)
		l.Version += 10 // Status subsequently returns the older persisted version.
		return l, err
	}
	if err := x.launch.Run(context.Background(), x.ports); err == nil {
		t.Fatal("rewound lease accepted")
	}
	if len(x.jobs) != 0 {
		t.Fatal("built jobs from stale lease")
	}
}

func TestLaunchRejectsChangedRunningTransport(t *testing.T) {
	x := newLaunchFixture(t)
	original := x.ports.Status
	x.ports.Status = func(ctx context.Context, c Candidate, id string) (nodeagent.Job, error) {
		j, err := original(ctx, c, id)
		j.TransportHash = strings.Repeat("f", 64)
		return j, err
	}
	if err := x.launch.Run(context.Background(), x.ports); err == nil {
		t.Fatal("changed running transport accepted")
	}
}
