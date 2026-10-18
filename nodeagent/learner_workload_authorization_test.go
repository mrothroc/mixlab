package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestAuthorizeLearnerWorkloadBindingAndAbort(t *testing.T) {
	for _, mode := range []string{"exact", "changed_csr", "changed_proof", "changed_scope", "aborted_before", "aborted_during", "canceled_during", "actor_expired_during", "lease_expired_during", "wrong_supplement"} {
		t.Run(mode, func(t *testing.T) {
			s, actor, reserve, id := learnerCredentialFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			now := leaseNow
			p := WorkloadPorts{Clock: func() time.Time { return now }, Initialize: func(context.Context, workload.Scope) error { return nil }, Request: learnerCredentialRequest}
			original, err := s.PrepareLearnerWorkload(ctx, actor, id, p)
			if err != nil {
				t.Fatal(err)
			}
			before, _, err := s.loadLearner(id)
			if err != nil {
				t.Fatal(err)
			}
			want, err := original.ReauthorizationRequest()
			if err != nil {
				t.Fatal(err)
			}
			auth := workload.RequestAuthorization{Version: workload.RequestAuthorizationVersion, Proof: trust.SignedProof{Request: want, Evidence: trust.TrustEvidence{Cluster: original.Request.Scope.Cluster, Principal: original.Request.Scope.Node, Role: trust.Node}}}
			calls := 0
			p.AuthorizeRequest = func(ctx context.Context, scope workload.Scope, request workload.SignedRequest) (workload.RequestAuthorization, error) {
				calls++
				if scope != original.Request.Scope {
					t.Error("port received another scope")
				}
				if mode == "aborted_during" {
					if _, err := s.AbortReservation(ctx, actor, reserve, leaseNow); err != nil {
						return workload.RequestAuthorization{}, err
					}
				}
				if mode == "canceled_during" {
					cancel()
				}
				if mode == "actor_expired_during" {
					now = actor.ExpiresAt
				}
				if mode == "lease_expired_during" {
					now = leaseNow.Add(time.Minute)
				}
				out := auth
				if mode == "wrong_supplement" {
					out.Proof.Request.Digest = "changed"
				}
				return out, nil // The real credential port attests only its saved request.
			}
			input := original
			switch mode {
			case "changed_csr":
				input, err = learnerCredentialRequest(ctx, original.Request.Scope)
				if err != nil {
					t.Fatal(err)
				}
			case "changed_proof":
				input.Proof.Signature = []byte("different original proof")
			case "changed_scope":
				scope := original.Request.Scope
				scope.Workload = actor.Principal
				input, err = learnerCredentialRequest(ctx, scope)
				if err != nil {
					t.Fatal(err)
				}
			case "aborted_before":
				if _, err := s.AbortReservation(ctx, actor, reserve, leaseNow); err != nil {
					t.Fatal(err)
				}
			}
			inputBefore, _ := json.Marshal(input)
			out, err := s.AuthorizeLearnerWorkload(ctx, actor, id, input, p)
			if mode == "exact" {
				if err != nil || out.Proof.Request != want || calls != 1 {
					t.Fatal("exact reauthorization failed", err)
				}
			} else if err == nil || out.Version != "" {
				t.Fatal("changed or fenced authorization escaped")
			}
			if (mode == "changed_scope" || mode == "aborted_before") && calls != 0 {
				t.Fatal("invalid admission reached credential signer")
			}
			after, _, err := s.loadLearner(id)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatal("reauthorization rewrote learner journal", err)
			}
			inputAfter, _ := json.Marshal(input)
			if !bytes.Equal(inputBefore, inputAfter) {
				t.Fatal("reauthorization changed original request")
			}
		})
	}
}

func TestAuthorizeLearnerWorkloadRequiresInitializedIntent(t *testing.T) {
	s, actor, _, id := learnerCredentialFixture(t)
	_, j, err := s.loadLearner(id)
	if err != nil {
		t.Fatal(err)
	}
	scope, err := learnerScope(j, actor.Principal, false)
	if err != nil {
		t.Fatal(err)
	}
	original, err := learnerCredentialRequest(context.Background(), scope)
	if err != nil {
		t.Fatal(err)
	}
	p := WorkloadPorts{Clock: func() time.Time { return leaseNow }, AuthorizeRequest: func(context.Context, workload.Scope, workload.SignedRequest) (workload.RequestAuthorization, error) {
		t.Error("authorization reached credential port before initialization")
		return workload.RequestAuthorization{}, nil
	}}
	if _, err := s.AuthorizeLearnerWorkload(context.Background(), actor, id, original, p); err == nil {
		t.Fatal("authorization created a new credential intent")
	}
}
