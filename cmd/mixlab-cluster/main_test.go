package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/internal/buildinfo"
)

func TestCLI(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		code int
		want string
	}{
		{"version", []string{"-version"}, 0, buildinfo.Report("mixlab-cluster") + "\n" + developmentNotice + "\n"},
		{"help", []string{"-help"}, 0, "Usage: mixlab-cluster -version | -help"},
		{"empty", nil, 2, developmentNotice},
		{"enrollment", []string{"enroll"}, 2, "explicit policy"},
		{"agent missing state", []string{"agent", "-agent-state-dir", filepath.Join(t.TempDir(), "missing")}, 1, "agent:"},
		{"future flag", []string{"-listen", ":8080"}, 2, "flag provided but not defined"},
		{"version with subcommand", []string{"-version", "agent"}, 2, "unexpected argument"},
		{"help with subcommand", []string{"-help", "enroll"}, 2, "unexpected argument"},
		{"false version", []string{"-version=false"}, 2, developmentNotice},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if got := run(tc.args, &stdout, &stderr); got != tc.code {
				t.Fatalf("exit = %d, want %d; stdout=%q stderr=%q", got, tc.code, stdout.String(), stderr.String())
			}
			out, unused := stdout.String(), stderr.String()
			if tc.code != 0 {
				out, unused = unused, out
			}
			if unused != "" || !strings.Contains(out, tc.want) {
				t.Fatalf("output=%q unused stream=%q, want %q", out, unused, tc.want)
			}
			if tc.name == "version" && out != tc.want {
				t.Fatalf("version output=%q, want exactly %q", out, tc.want)
			}
			if tc.name == "help" {
				if !strings.Contains(out, developmentNotice) {
					t.Fatal("help must identify development-only status")
				}
				for _, line := range strings.Split(out, "\n") {
					if strings.HasPrefix(line, "  -") && line != "  -help" && line != "  -version" {
						t.Fatalf("unexpected public flag: %s", line)
					}
				}
			}
		})
	}
}
