package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestAgentUsageRejectsUnexpectedArguments(t *testing.T) {
	for _, argument := range []string{"serve", "stray"} {
		t.Run(argument, func(t *testing.T) {
			args := []string{"agent", "serve", "-agent-state-dir", t.TempDir()}
			if argument == "stray" {
				args = []string{"agent", "-agent-state-dir", t.TempDir(), argument}
			}
			var stdout, stderr bytes.Buffer
			if code := run(args, &stdout, &stderr); code != 2 {
				t.Fatalf("exit = %d, want 2: %s", code, &stderr)
			}
			for _, text := range []string{"unexpected argument \"" + argument + "\"", "mixlab-cluster agent [flags]", "no serve subcommand"} {
				if !strings.Contains(stderr.String(), text) {
					t.Errorf("missing %q in stderr: %s", text, &stderr)
				}
			}
		})
	}
}

func TestAgentUsageRejectsInvalidAdvertisement(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"agent", "-agent-state-dir", t.TempDir(), "-agent-advertise", "bonjour"}, &stdout, &stderr); code != 2 {
		t.Fatalf("exit = %d, want 2: %s", code, &stderr)
	}
	if !strings.Contains(stderr.String(), `-agent-advertise "bonjour": advertisement must be mdns or off`) {
		t.Errorf("silent or unclear advertisement error: %q", &stderr)
	}
}
