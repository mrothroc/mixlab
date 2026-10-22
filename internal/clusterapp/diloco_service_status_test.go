package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"syscall"
	"testing"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

func TestDiLoCoNodeFailurePublicStatus(t *testing.T) {
	s, saga, fixture := admissionMaintenanceService(t)
	ctx := context.Background()
	node := fixture.q.Launch.Selection.Selected[0].Capabilities.Node
	client, err := NewNodeClientWithPolicy("127.0.0.1:7445", node, func() (*managedtls.Policy, error) {
		return nil, fmt.Errorf("PRIVATE-TLS-MATERIAL: %w", &net.OpError{Op: "dial", Net: "tcp", Err: syscall.EHOSTUNREACH})
	})
	check(t, err)
	s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
		return client.request(ctx, http.MethodPost, "/v1/agent/leases", nil, nil)
	}
	check(t, s.step(ctx, fixture.q.Launch.Run))
	o, err := s.Inbox.Observation(ctx, fixture.q.Launch.Run)
	check(t, err)
	status := DiLoCoSubmissionStatus{Admission: "ADMITTING"}
	status.setObservation(o)
	if !status.ServiceError || status.ServiceNode != node || status.ServiceFailure == nil || status.ServiceFailure.Reason != "route_or_permission_denied" || status.NextRetryAt <= o.At {
		t.Fatalf("missing actionable node failure: %+v", status)
	}
	if !strings.Contains(status.ServiceFailure.Hint, "allowlist") || !strings.Contains(status.ServiceFailure.Hint, "does not prove") {
		t.Fatal("missing qualified Local Network guidance")
	}
	b, err := json.Marshal(status)
	check(t, err)
	if strings.Contains(string(b), "PRIVATE-TLS-MATERIAL") || strings.Contains(string(b), "127.0.0.1") {
		t.Fatalf("raw failure exposed: %s", b)
	}
	check(t, saga.Cancel(ctx))
	status.Admission = "ABORTING"
	status.setObservation(o)
	if !status.AwaitingNode {
		t.Fatal("abort pending node not explicit")
	}
	status.setObservation(admission.Observation{At: o.At + 1})
	if status.ServiceFailure != nil || status.ServiceError || status.ServiceNode != "" || status.AwaitingNode || status.NextRetryAt != 0 {
		t.Fatal("successful observation retained stale failure")
	}
}

func TestDiLoCoStatusSanitizesLegacyAndUnknownObservation(t *testing.T) {
	for _, reason := range []string{"", "private-secret-certificate"} {
		s := DiLoCoSubmissionStatus{}
		s.setObservation(admission.Observation{At: 1, Error: "raw key", FailureClass: reason, FailureNode: "secret-path"})
		b, err := json.Marshal(s)
		check(t, err)
		if s.ServiceFailure.Reason != "unavailable_or_unauthenticated" || s.ServiceNode != "" || strings.Contains(string(b), "secret") || strings.Contains(string(b), "raw key") {
			t.Fatal(string(b))
		}
	}
}

func TestDiLoCoJoinedFailuresKeepNodeAndClassTogether(t *testing.T) {
	first, second := strings.Repeat("a", 32), strings.Repeat("b", 32)
	effect := errors.Join(&nodeFailure{node: first, err: managedtls.ErrPeerRejected}, &nodeFailure{node: second, err: syscall.EHOSTUNREACH})
	o, _ := serviceObservation("run", 100, admission.Observation{}, admission.Snapshot{}, admission.Snapshot{}, true, effect)
	if o.FailureNode != first || o.FailureClass != "tls_rejected" {
		t.Fatalf("class came from a different node: %+v", o)
	}
}
