package clusterapp

import (
	"bytes"
	"context"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

type recruitmentAuthorizationFixture struct {
	a          *Authority
	controller *principal.Store
	signer     RecruitmentOptions
	now        time.Time
	view       trust.VerifiedSnapshot
	nodeKey    ed25519.PrivateKey
	nodeChain  [][]byte
	request    workload.SignedRequest
	inbox      *admission.Inbox
	inboxPath  statehome.Path
}

func newRecruitmentAuthorizationFixture(t *testing.T) *recruitmentAuthorizationFixture {
	t.Helper()
	ctx := context.Background()
	endpoint := "https://127.0.0.1:7443"
	path := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, path, testNow))
	a, err := OpenAuthority(ctx, path, testNow)
	check(t, err)
	t.Cleanup(func() { check(t, a.Close()) })
	resolve := func(name string) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(path.Dir()), name)}, statehome.Context{Kind: statehome.Principal})
		check(t, err)
		return p
	}
	controller, err := principal.Open(resolve("controller"), testNow)
	check(t, err)
	t.Cleanup(func() { check(t, controller.Close()) })
	cs, _, err := controller.Active(testNow)
	check(t, err)
	view, err := a.Current(ctx, testNow)
	check(t, err)
	invite, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, view, testNow)
	check(t, err)
	defer invite.Clear()
	_, key, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	q, err := enrollment.NewRequest(invite, key, envelope.PublicKey().Bytes(), testNow)
	check(t, err)
	node, err := a.Enrollment.Consume(ctx, q, invite.Secret, view, testNow)
	check(t, err)
	f := &recruitmentAuthorizationFixture{a: a, controller: controller, now: testNow, view: view, nodeKey: key, nodeChain: node.Chain}
	f.signer = RecruitmentOptions{Principal: controller, Clock: func() time.Time { return f.now }}
	_, leases := admissionLeaseSetup(t, 2)
	submission := leases.q
	submission.Owner = cs.Principal
	submission.Launch.Cluster = cs.Cluster
	submission.Launch.Controller = cs.Principal
	submission.Launch.Selection.Requirements.Cluster = cs.Cluster
	submission.Launch.Selection.Selected[0].Capabilities.Node = node.Approval.Principal
	for i := range submission.Launch.Selection.Selected {
		caps := &submission.Launch.Selection.Selected[i].Capabilities
		caps.ObservedAt, caps.ProbeObservedAt = testNow.Unix(), testNow.Unix()
	}
	submission.Specification.Coordinator.ClusterID = cs.Cluster
	submission.Specification.CreatedAt = testNow.Unix()
	submission.Deadline = testNow.Add(time.Hour).Unix()
	f.inboxPath = resolve("authorization-inbox")
	check(t, f.inboxPath.Ensure())
	f.inbox, err = admission.OpenInbox(f.inboxPath)
	check(t, err)
	check(t, f.inbox.Accept(ctx, submission))
	id := func() string { value, err := controllerRandomID(); check(t, err); return value }
	scope := workload.Scope{Cluster: cs.Cluster, Controller: cs.Principal, Node: node.Approval.Principal, Lease: id(), Run: submission.Launch.Run, Job: id(), Attempt: submission.Launch.Attempt, Workload: id(), Generation: 1, ManifestHash: strings.Repeat("b", 64), Created: testNow.Unix(), AdmitUntil: submission.Deadline, Deadline: testNow.Add(3 * time.Hour).Unix(), Profile: workload.LearnerVersion, Worker: id(), PlanHash: strings.Repeat("c", 64), Coordinator: submission.Specification.Coordinator.PrincipalID}
	_, transportKey, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	request, err := workload.NewRequest(scope, transportKey)
	check(t, err)
	sign, err := request.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(a.Anchor, node.Chain, key, view, sign, testNow)
	check(t, err)
	f.request = workload.SignedRequest{Request: request, Proof: proof}
	return f
}

func (f *recruitmentAuthorizationFixture) advance(t *testing.T, now time.Time) {
	t.Helper()
	f.now = now
	var err error
	f.view, err = f.a.Current(context.Background(), now)
	check(t, err)
	snapshot, err := f.a.Snapshots.Load(now)
	check(t, err)
	check(t, f.controller.Refresh(context.Background(), snapshot, now))
}

func (f *recruitmentAuthorizationFixture) authorization(t *testing.T) workload.RequestAuthorization {
	t.Helper()
	request, err := f.request.ReauthorizationRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a.Anchor, f.nodeChain, f.nodeKey, f.view, request, f.now)
	check(t, err)
	return workload.RequestAuthorization{Version: workload.RequestAuthorizationVersion, Proof: proof}
}

func recruitmentAuthorizationJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	check(t, err)
	return b
}

func TestRecruitmentSupplementalAuthorizationAfterOriginalProofExpires(t *testing.T) {
	f := newRecruitmentAuthorizationFixture(t)
	f.advance(t, time.Unix(f.request.Proof.Evidence.Snapshot.Payload.ExpiresAt+1, 0))
	ctx := context.Background()
	if _, err := f.signer.grant(ctx, f.request); err == nil {
		t.Fatal("ordinary grant accepted expired node evidence")
	}
	original := recruitmentAuthorizationJSON(t, f.request)
	grant, err := f.signer.grantWithAuthorization(ctx, f.request, f.authorization(t))
	check(t, err)
	if !bytes.Equal(original, recruitmentAuthorizationJSON(t, grant.Request)) {
		t.Fatal("supplemental grant replaced the original signed request")
	}
	request, err := f.request.GrantRequest()
	check(t, err)
	_, err = trust.VerifyProof(f.a.Anchor, f.view, grant.Proof, request, f.now)
	check(t, err)
}

// This composes the real controller, durable inbox and authority issuance
// service. It is not an installed-service or HTTP transport test.
func TestRecruitmentAuthorizedGrantLostACKKeepsPersistedGrant(t *testing.T) {
	f := newRecruitmentAuthorizationFixture(t)
	ctx := context.Background()
	auth := f.authorization(t)
	scope := f.request.Request.Scope
	grant, err := f.inbox.WorkloadGrant(ctx, scope.Run, scope.Job, f.request, func() (workload.Grant, error) {
		return f.signer.grantWithAuthorization(ctx, f.request, auth)
	})
	check(t, err)
	want := recruitmentAuthorizationJSON(t, grant)
	authorized, err := f.signer.authorizeGrant(ctx, grant, auth)
	check(t, err)
	identity, _, err := f.controller.Active(f.now)
	check(t, err)
	first, err := f.a.Workloads.IssueAuthorized(ctx, identity.Chain, authorized, f.view, f.now)
	check(t, err)
	// Lose the response, let both original proof snapshots expire, then reopen
	// the inbox. Only supplemental authorization may be rebuilt for replay.
	f.advance(t, time.Unix(grant.Proof.Evidence.Snapshot.Payload.ExpiresAt+1, 0))
	reopened, err := admission.OpenInbox(f.inboxPath)
	check(t, err)
	loaded, err := reopened.WorkloadGrant(ctx, scope.Run, scope.Job, f.request, func() (workload.Grant, error) {
		t.Fatal("lost-ACK replay rebuilt the original grant")
		return workload.Grant{}, nil
	})
	check(t, err)
	if !bytes.Equal(want, recruitmentAuthorizationJSON(t, loaded)) {
		t.Fatal("reopened inbox changed the original grant")
	}
	if _, err := f.signer.authorizeGrant(ctx, loaded, auth); err == nil {
		t.Fatal("cached grant bypassed expired supplemental authorization")
	}
	fresh, err := f.signer.authorizeGrant(ctx, loaded, f.authorization(t))
	check(t, err)
	check(t, workload.VerifyAuthorizedGrant(f.a.Anchor, f.view, fresh, f.now))
	if bytes.Equal(recruitmentAuthorizationJSON(t, authorized.ControllerAuthorization), recruitmentAuthorizationJSON(t, fresh.ControllerAuthorization)) {
		t.Fatal("controller supplemental authorization was not refreshed")
	}
	identity, _, err = f.controller.Active(f.now)
	check(t, err)
	again, err := f.a.Workloads.IssueAuthorized(ctx, identity.Chain, fresh, f.view, f.now)
	check(t, err)
	if first.GrantHash != again.GrantHash || !bytes.Equal(first.Chain[0], again.Chain[0]) {
		t.Fatal("lost-ACK replay changed grant identity or reissued the credential")
	}
	_, replacementKey, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	replacement, err := workload.NewRequest(scope, replacementKey)
	check(t, err)
	replacementRequest, err := replacement.SigningRequest()
	check(t, err)
	replacementProof, err := trust.SignPrincipalProof(f.a.Anchor, f.nodeChain, f.nodeKey, f.view, replacementRequest, f.now)
	check(t, err)
	changed := workload.SignedRequest{Request: replacement, Proof: replacementProof}
	if _, err := reopened.WorkloadGrant(ctx, scope.Run, scope.Job, changed, func() (workload.Grant, error) {
		t.Fatal("changed request rebuilt the persisted grant")
		return workload.Grant{}, nil
	}); err == nil {
		t.Fatal("cached grant accepted a substituted signed CSR")
	}
	stored, err := f.inboxPath.ReadFile(scope.Run + "-" + scope.Job + "-workload-grant")
	check(t, err)
	if !bytes.Equal(want, stored) {
		t.Fatal("supplemental replay rewrote persisted grant bytes")
	}
}

func TestRecruitmentSupplementalAuthorizationRejectsSubstitutions(t *testing.T) {
	f := newRecruitmentAuthorizationFixture(t)
	ctx := context.Background()
	auth := f.authorization(t)
	for _, mode := range []string{"missing", "signature", "csr", "node", "controller", "cluster", "ordinary", "recovery"} {
		t.Run(mode, func(t *testing.T) {
			var request workload.SignedRequest
			check(t, json.Unmarshal(recruitmentAuthorizationJSON(t, f.request), &request))
			var supplemental workload.RequestAuthorization
			check(t, json.Unmarshal(recruitmentAuthorizationJSON(t, auth), &supplemental))
			switch mode {
			case "missing":
				supplemental = workload.RequestAuthorization{}
			case "signature":
				supplemental.Proof.Signature[0] ^= 1
			case "csr":
				request.Request.CSR[len(request.Request.CSR)-1] ^= 1
			case "node":
				request.Request.Scope.Node = strings.Repeat("1", 32)
			case "controller":
				request.Request.Scope.Controller = strings.Repeat("1", 32)
			case "cluster":
				request.Request.Scope.Cluster = strings.Repeat("1", 32)
			case "ordinary":
				request.Request.Scope.Profile = ""
			case "recovery":
				request.Request.Scope.Coordinator = request.Request.Scope.Controller
			}
			if _, err := f.signer.grantWithAuthorization(ctx, request, supplemental); err == nil {
				t.Fatal("substituted request or supplemental proof accepted")
			}
			grant, err := f.signer.grantWithAuthorization(ctx, f.request, auth)
			check(t, err)
			grant.Request = request
			if _, err := f.signer.authorizeGrant(ctx, grant, supplemental); err == nil {
				t.Fatal("cached grant replay bypassed supplemental verification")
			}
		})
	}
}

func TestRecruitmentSupplementalAuthorizationRejectsRevocationAndDeadline(t *testing.T) {
	for _, mode := range []string{"node_revoked", "controller_revoked", "admission_expired", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			f := newRecruitmentAuthorizationFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			auth := f.authorization(t)
			grant, err := f.signer.grantWithAuthorization(ctx, f.request, auth)
			check(t, err)
			switch mode {
			case "node_revoked", "controller_revoked":
				id := f.request.Request.Scope.Node
				if mode == "controller_revoked" {
					id = f.request.Request.Scope.Controller
				}
				next, err := f.a.Snapshots.Revoke(ctx, "principal", id, "prospective", "supplemental authorization test", f.now)
				check(t, err)
				check(t, f.controller.Refresh(ctx, next, f.now))
			case "admission_expired":
				f.advance(t, time.Unix(f.request.Request.Scope.AdmitUntil, 0))
				auth = f.authorization(t)
			case "canceled":
				cancel()
			}
			if _, err := f.signer.grantWithAuthorization(ctx, f.request, auth); err == nil {
				t.Fatal("terminal authorization failure accepted before grant")
			}
			if _, err := f.signer.authorizeGrant(ctx, grant, auth); err == nil {
				t.Fatal("terminal authorization failure accepted for persisted grant")
			}
		})
	}
}
