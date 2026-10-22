//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
)

func TestDiLoCoStatusPrintsSafeFailureHint(t *testing.T) {
	status := clusterapp.DiLoCoSubmissionStatus{Admission: "ABORTING", ServiceError: true, ServiceNode: strings.Repeat("b", 32), AwaitingNode: true, NextRetryAt: 1234,
		ServiceFailure: &clusterdiagnostic.Failure{Reason: "route_or_permission_denied", Hint: "untrusted-secret"}}
	var out, stderr bytes.Buffer
	if err := reportDiLoCoStatus(&out, &stderr, status); err != nil {
		t.Fatal(err)
	}
	var got clusterapp.DiLoCoSubmissionStatus
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ServiceNode != status.ServiceNode || !got.AwaitingNode || got.NextRetryAt != 1234 {
		t.Fatal(got)
	}
	for _, want := range []string{"route_or_permission_denied", "allowlist", "LaunchAgent", "does not prove"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("hint missing %q: %s", want, &stderr)
		}
	}
	if strings.Contains(stderr.String(), "untrusted-secret") {
		t.Fatal("printed untrusted guidance")
	}
}
