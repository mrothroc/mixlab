package main

import (
	"bytes"
	"strings"
	"testing"
)

// The agent's control listener serves an anonymous trust-snapshot route, so a
// wildcard bind would expose it on every interface. bootstrapEndpoint already
// refuses that at agent_unix.go:57, well before the bind; this pins the
// behavior so the guard cannot be dropped when that call site is refactored.
func TestAgentListenRejectsWildcardBind(t *testing.T) {
	for _, address := range []string{"0.0.0.0:7445", "[::]:7445", ":7445"} {
		t.Run(address, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runAgent([]string{"-agent-listen", address}, &stdout, &stderr)
			if code == 0 {
				t.Fatalf("accepted wildcard bind %q", address)
			}
			if !strings.Contains(stderr.String(), "wildcard bind such as 0.0.0.0 is refused") {
				t.Fatalf("wrong rejection for %q: %s", address, stderr.String())
			}
		})
	}
}

// The documented default and LAN example must keep working; the check must not
// reject the addresses the docs tell administrators to use.
func TestAgentListenAcceptsDocumentedAddresses(t *testing.T) {
	for _, address := range []string{"127.0.0.1:7445", "192.168.1.20:7445"} {
		var stdout, stderr bytes.Buffer
		// These fail later for missing state, not at the address policy.
		runAgent([]string{"-agent-listen", address}, &stdout, &stderr)
		if strings.Contains(stderr.String(), "wildcard bind such as 0.0.0.0 is refused") {
			t.Fatalf("rejected documented address %q: %s", address, stderr.String())
		}
	}
}
