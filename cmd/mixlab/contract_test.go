package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectContractHelp(t *testing.T) {
	for _, name := range []string{"config", "contract-profile", "contract-require"} {
		if !flagGroupContains(modeFlagGroups["inspect-contract"], name) {
			t.Fatal("missing help", name)
		}
	}
}

func TestInspectContractStubCLI(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the portable CLI")
	}
	binary := filepath.Join(t.TempDir(), "mixlab")
	build := exec.Command("go", "build", "-o", binary, ".")
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("stub build: %v\n%s", err, out)
	}
	for _, tc := range []struct {
		args []string
		fail bool
	}{
		{[]string{"-config", "../../arch/testdata/execution_contracts/ttt.json", "-contract-profile", "native-ttt-stateful", "-contract-require", "streaming"}, false},
		{[]string{"-config", "../../examples/plain_3L.json"}, false},
		{[]string{"-config", "../../examples/plain_3L.json", "-contract-require", "complete"}, true},
		{[]string{"-config", "../../arch/testdata/execution_contracts/recurrent.json", "-contract-profile", "native-ttt-stateful"}, true},
		{nil, true},
	} {
		var stdout, stderr bytes.Buffer
		cmd := exec.Command(binary, append([]string{"-mode", "inspect-contract"}, tc.args...)...)
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if (err != nil) != tc.fail {
			t.Fatalf("args=%v err=%v stderr=%s", tc.args, err, stderr.String())
		}
		if strings.Contains(stderr.String(), "MLX backend unavailable") {
			t.Fatal("inspection reached GPU dispatch")
		}
		if tc.fail {
			if stdout.Len() != 0 || !json.Valid(stderr.Bytes()) {
				t.Fatalf("invalid error channels stdout=%s stderr=%s", &stdout, &stderr)
			}
		} else {
			if !json.Valid(stdout.Bytes()) {
				t.Fatalf("non-JSON stdout: %s", &stdout)
			}
		}
	}
}
