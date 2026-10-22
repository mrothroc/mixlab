package main

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/mrothroc/mixlab/internal/clusterservice"
)

// A registration is not evidence that a manually started role is running.
// Read only validated persisted arguments; never print the full argv or keys.
func doctorService(ctx context.Context, m clusterservice.Manager, role, inspected string) doctorCheck {
	out := doctorCheck{Check: role + "_service", Status: "unknown", Detail: "No readable installed OS registration; this check does not discover manual processes."}
	p, _, err := m.Paths(role)
	if err != nil {
		return out
	}
	s, err := clusterservice.Read(p)
	if err != nil || s.Role != role {
		return out
	}
	paths := []string{}
	state := ""
	for i, arg := range s.Arguments {
		if i+1 >= len(s.Arguments) {
			continue
		}
		if strings.HasSuffix(arg, "-state-dir") || arg == "-checkpoint-dir" || arg == "-artifact-dir" {
			paths = append(paths, arg+"="+s.Arguments[i+1])
		}
		if arg == "-agent-state-dir" {
			state = s.Arguments[i+1]
		}
	}
	out.Detail = "installed OS registration: " + strings.Join(paths, ", ") + ". This is not manual-role readiness."
	if role == "agent" && inspected != "" && filepath.Clean(state) != filepath.Clean(inspected) {
		out.Status = "mismatch"
		out.Detail += fmt.Sprintf(" State directory mismatch: doctor is inspecting %s.", inspected)
		return out
	}
	if _, err := m.Control(ctx, role, "status"); err == nil {
		out.Status = "info"
		out.Detail += " OS manager reports the registered job; check authenticated trust and peer readiness separately."
	} else {
		out.Detail += " OS manager cannot confirm the registered job; use the role's status command."
	}
	return out
}
