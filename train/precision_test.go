package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestResumePrecision(t *testing.T) {
	on := gpu.PrecisionPolicy{ComputeDType: "float32", Backend: "cuda", TF32Allowed: true}
	if err := checkResumePrecision(&on, on); err != nil {
		t.Fatal(err)
	}
	for _, p := range []gpu.PrecisionPolicy{{ComputeDType: "float32", Backend: "cuda"}, {ComputeDType: "bf16", Backend: "cuda", TF32Allowed: true}, {ComputeDType: "float32", Backend: "metal"}} {
		if err := checkResumePrecision(&on, p); err == nil {
			t.Fatalf("allowed precision change to %+v", p)
		}
	}
	if err := checkResumePrecision(nil, on); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("legacy: %v", err)
	}
	for _, value := range []any{resumeManifest{Precision: &on}, distributedResumeManifest{Precision: &on}} {
		b, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var m struct {
			Precision gpu.PrecisionPolicy `json:"precision"`
		}
		if err = json.Unmarshal(b, &m); err != nil || m.Precision != on {
			t.Fatalf("roundtrip %s %v", b, err)
		}
	}
}

func TestPrecisionRunManifest(t *testing.T) {
	cfg := &ArchConfig{Name: "precision"}
	for _, save := range []bool{false, true} {
		dir := t.TempDir()
		opts := TrainOptions{CheckpointDir: dir}
		path := filepath.Join(dir, "run.json")
		if save {
			opts.SafetensorsPath = filepath.Join(dir, "model.safetensors")
			path = opts.SafetensorsPath + ".run.json"
		}
		if err := startTrainingPrecision(cfg, opts); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var m struct {
			Format    string
			Precision gpu.PrecisionPolicy
		}
		if err = json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		if m.Format != "mixlab_run_v1" || m.Precision.Backend != gpu.PrecisionBackend() || m.Precision.ComputeDType != "float32" {
			t.Fatalf("manifest=%s", b)
		}
	}
}

func TestAllowTF32ResumeHash(t *testing.T) {
	cfg := &ArchConfig{}
	a, err := resumeConfigHash(cfg)
	if err != nil {
		t.Fatal(err)
	}
	f := false
	cfg.Training.AllowTF32 = &f
	b, err := resumeConfigHash(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if a == b {
		t.Fatal("precision override absent from config hash")
	}
}

func TestResumePrecisionMismatchBeforeStateLoad(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 10
	policy, err := trainingPrecision(cfg)
	if err != nil {
		t.Fatal(err)
	}
	policy.TF32Allowed = !policy.TF32Allowed
	dir := t.TempDir()
	for _, name := range []string{"model.safetensors", "state.safetensors"} {
		if err = os.WriteFile(filepath.Join(dir, name), []byte("must not be loaded"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(dir, "step_000001.resume.json")
	m := resumeManifest{Format: resumeCheckpointFormat, GlobalStep: 1, ModelFile: "model.safetensors", StateFile: "state.safetensors", Precision: &policy, Optimizer: gpu.TrainerOptimizerStats{AttemptedSteps: 1, CommittedSteps: 1}}
	if err = atomicWriteJSON(path, m); err != nil {
		t.Fatal(err)
	}
	if _, err = prepareResumeRun(cfg, "unused", path, nil); err == nil || !strings.Contains(err.Error(), "precision mismatch") {
		t.Fatalf("resume error: %v", err)
	}
	runPath := filepath.Join(dir, "run.json")
	before := []byte("last-good provenance")
	if err = os.WriteFile(runPath, before, 0o600); err != nil {
		t.Fatal(err)
	}
	if err = startTrainingPrecision(cfg, TrainOptions{Resume: path, CheckpointDir: dir}); err == nil || !strings.Contains(err.Error(), "precision mismatch") {
		t.Fatalf("startup: %v", err)
	}
	after, err := os.ReadFile(runPath)
	if err != nil || string(after) != string(before) {
		t.Fatal("failed resume overwrote provenance")
	}
}
