package train

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunValidate(t *testing.T) {
	t.Run("missing config", func(t *testing.T) {
		err := runValidate("")
		if err == nil || !strings.Contains(err.Error(), "-config is required") {
			t.Fatalf("runValidate() error = %v, want required-config error", err)
		}
	})

	t.Run("valid config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "valid.json")
		if err := os.WriteFile(path, []byte(`{
			"name": "validate_test",
			"model_dim": 8,
			"vocab_size": 16,
			"seq_len": 4,
			"blocks": [{"type": "plain", "heads": 2}],
			"training": {"steps": 1, "batch_tokens": 4}
		}`), 0o600); err != nil {
			t.Fatal(err)
		}

		output := captureStdout(t, func() {
			if err := runValidate(path); err != nil {
				t.Fatalf("runValidate(valid): %v", err)
			}
		})
		if !strings.Contains(output, "valid config") || !strings.Contains(output, "objective=causal") {
			t.Fatalf("runValidate output = %q", output)
		}
	})

	t.Run("invalid config", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "invalid.json")
		if err := os.WriteFile(path, []byte(`{
			"name": "validate_test",
			"model_dim": 8,
			"vocab_size": 16,
			"seq_len": 4,
			"blocks": [{"type": "plain", "heads": 3}]
		}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := runValidate(path); err == nil {
			t.Fatal("runValidate(invalid) unexpectedly succeeded")
		}
	})
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	defer func() { os.Stdout = old }()

	fn()
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	return string(data)
}
