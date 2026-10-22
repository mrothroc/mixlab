package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/internal/clusterservice"
)

func TestDoctorNamesDifferentInstalledAgentState(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	m, err := serviceManager()
	if err != nil {
		t.Fatal(err)
	}
	p, _, err := m.Paths("agent")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	installed, inspected := filepath.Join(home, "old-agent"), filepath.Join(home, "manual-agent")
	spec := clusterservice.Spec{Role: "agent", Binary: "/opt/mixlab/mixlab-cluster", Arguments: []string{"agent", "-agent-state-dir", installed}}
	b, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.WriteFile("service.json", b); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	// Missing manual state is intentional; registration evidence is independent
	// of role readiness and must not be presented as the inspected installation.
	_ = runDoctor([]string{"-agent-state-dir", inspected}, &out, &stderr)
	var result struct {
		Checks []doctorCheck `json:"checks"`
	}
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err, &stderr)
	}
	for _, c := range result.Checks {
		if c.Check != "agent_service" {
			continue
		}
		for _, want := range []string{"installed OS registration", installed, inspected, "mismatch"} {
			if !strings.Contains(c.Detail, want) {
				t.Errorf("missing %q: %+v", want, c)
			}
		}
		if c.Status == "ok" {
			t.Error("different registration reported ready")
		}
		if _, err := os.Stat(inspected); !os.IsNotExist(err) {
			t.Error("doctor created inspected state")
		}
		return
	}
	t.Fatal("missing agent_service row")
}
