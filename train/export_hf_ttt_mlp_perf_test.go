package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestExportHFTTTMLPInferencePerformance is an opt-in release gate for the
// reported D384, 14-mixer, 43-token Hugging Face inference shape.
func TestExportHFTTTMLPInferencePerformance(t *testing.T) {
	if os.Getenv("HF_TTT_MLP_PERF") != "1" {
		t.Skip("set HF_TTT_MLP_PERF=1 to run the TTT-MLP HF performance gate")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("python HF dependencies unavailable via %q: %v", python, err)
	}

	var blocks strings.Builder
	for layer := 0; layer < 14; layer++ {
		if layer > 0 {
			blocks.WriteByte(',')
		}
		blocks.WriteString(`{"type":"ttt_mlp","heads":6,"chunk_size":16}`)
		blocks.WriteByte(',')
		blocks.WriteString(`{"type":"swiglu"}`)
	}
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, fmt.Sprintf(`{
		"name":"ttt_mlp_hf_perf",
		"model_dim":384,
		"vocab_size":512,
		"seq_len":64,
		"mlp_mult":4.0,
		"blocks":[%s],
		"training":{"steps":1,"batch_tokens":64,"seed":6401}
	}`, blocks.String()))
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	script, err := filepath.Abs(filepath.Join("testdata", "hf_ttt_mlp_perf.py"))
	if err != nil {
		t.Fatalf("resolve HF TTT performance script: %v", err)
	}
	cmd := exec.Command(
		python,
		script,
		"--dir", outDir,
		"--seq-len", "43",
		"--batch", "1",
		"--iterations", "3",
		"--min-cpu-speedup", "3",
		"--mps",
	)
	output, err := cmd.CombinedOutput()
	t.Logf("hf_ttt_mlp_perf.py output:\n%s", output)
	if err != nil {
		t.Fatalf("TTT-MLP HF performance gate failed: %v", err)
	}
}
