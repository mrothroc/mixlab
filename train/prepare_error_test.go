package train

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunPrepareReturnsBoundedStderrTailWithoutPythonDependencies(t *testing.T) {
	binDir := t.TempDir()
	python := filepath.Join(binDir, "python3")
	pythonScript := `#!/bin/sh
if [ "$1" = "-c" ]; then
  exit 0
fi
i=1
while [ "$i" -le 25 ]; do
  printf 'prepare-diagnostic-%02d\n' "$i" >&2
  i=$((i + 1))
done
exit 7
`
	if err := os.WriteFile(python, []byte(pythonScript), 0o700); err != nil {
		t.Fatal(err)
	}
	scriptDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(scriptDir, "prepare.py"), []byte("# invoked by fake python\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)
	t.Setenv("MIXLAB_SCRIPTS", scriptDir)

	err := runPrepare(PrepareOptions{
		Input:       filepath.Join(t.TempDir(), "input.txt"),
		Output:      filepath.Join(t.TempDir(), "prepared"),
		InputFormat: "text",
		VocabSize:   32,
	})
	if err == nil {
		t.Fatal("runPrepare unexpectedly succeeded")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 {
		t.Fatalf("error did not retain subprocess cause: %v", err)
	}
	message := err.Error()
	for _, want := range []string{
		"prepare.py failed using MIXLAB_SCRIPTS scripts",
		"prepare.py stderr:",
		"prepare-diagnostic-06",
		"prepare-diagnostic-25",
	} {
		if !strings.Contains(message, want) {
			t.Fatalf("error missing %q:\n%s", want, message)
		}
	}
	if strings.Contains(message, "prepare-diagnostic-05") {
		t.Fatalf("error retained more than %d stderr lines:\n%s", prepareStderrTailLines, message)
	}
}

func TestBoundedTailBufferCapsCapturedBytes(t *testing.T) {
	buffer := newBoundedTailBuffer(8)
	if _, err := buffer.Write([]byte("abcdef")); err != nil {
		t.Fatal(err)
	}
	if _, err := buffer.Write([]byte("ghijkl")); err != nil {
		t.Fatal(err)
	}
	if got, want := buffer.String(), "efghijkl"; got != want {
		t.Fatalf("tail=%q want=%q", got, want)
	}
	if !buffer.truncated {
		t.Fatal("buffer did not record truncation")
	}
}
