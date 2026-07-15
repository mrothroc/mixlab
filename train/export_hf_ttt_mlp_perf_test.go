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
	python := hfTTTMLPPython(t)
	outDir := exportTTTMLPPerformanceFixture(t, 384, 14, 64, 64)
	script := hfTTTMLPPerformanceScript(t)
	cmd := hfTTTMLPPerformanceCommand(t,
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

// TestExportHFTTTMLPCompiledTrainingContract checks the exported classifier's
// full-graph forward/backward contract without requiring a CUDA toolchain.
func TestExportHFTTTMLPCompiledTrainingContract(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1 to run the TTT-MLP HF compile contract")
	}
	python := hfTTTMLPPython(t)
	outDir := exportTTTMLPPerformanceFixture(t, 16, 1, 8, 16)
	script := hfTTTMLPPerformanceScript(t)
	compileBackend := os.Getenv("HF_TTT_MLP_COMPILE_CONTRACT_BACKEND")
	if compileBackend == "" {
		compileBackend = "eager"
	}
	cmd := hfTTTMLPPerformanceCommand(t,
		python,
		script,
		"--dir", outDir,
		"--mode", "train",
		"--seq-len", "8",
		"--batch", "2",
		"--iterations", "1",
		"--warmup-iterations", "0",
		"--train-device", "cpu",
		"--compile-backend", compileBackend,
		"--gradient-parity",
		"--check-padding-guard",
	)
	output, err := cmd.CombinedOutput()
	t.Logf("hf_ttt_mlp_perf.py compile-contract output:\n%s", output)
	if err != nil {
		t.Fatalf("TTT-MLP HF compile contract failed: %v", err)
	}
}

// TestExportHFTTTMLPTrainingPerformance is an opt-in CUDA release gate for
// steady-state torch.compile fine-tuning throughput at sequence length 512.
func TestExportHFTTTMLPTrainingPerformance(t *testing.T) {
	if os.Getenv("HF_TTT_MLP_TRAIN_PERF") != "1" {
		t.Skip("set HF_TTT_MLP_TRAIN_PERF=1 to run the TTT-MLP HF training gate")
	}
	python := hfTTTMLPPython(t)
	if err := exec.Command(python, "-c", "import torch; assert torch.cuda.is_available()").Run(); err != nil {
		t.Skipf("CUDA is unavailable via %q: %v", python, err)
	}
	outDir := exportTTTMLPPerformanceFixture(t, 384, 14, 512, 512)
	script := hfTTTMLPPerformanceScript(t)
	minSpeedup := os.Getenv("HF_TTT_MLP_MIN_COMPILE_SPEEDUP")
	if minSpeedup == "" {
		minSpeedup = "1.0"
	}
	cmd := hfTTTMLPPerformanceCommand(t,
		python,
		script,
		"--dir", outDir,
		"--mode", "train",
		"--seq-len", "512",
		"--batch", "1",
		"--iterations", "5",
		"--warmup-iterations", "2",
		"--train-device", "cuda",
		"--compile-backend", "inductor",
		"--compile-mode", "reduce-overhead",
		"--min-compile-speedup", minSpeedup,
	)
	output, err := cmd.CombinedOutput()
	t.Logf("hf_ttt_mlp_perf.py training output:\n%s", output)
	if err != nil {
		t.Fatalf("TTT-MLP HF training performance gate failed: %v", err)
	}
}

func hfTTTMLPPerformanceCommand(t *testing.T, name string, args ...string) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Env = append(os.Environ(), "HF_MODULES_CACHE="+filepath.Join(t.TempDir(), "hf_modules"))
	return cmd
}

func hfTTTMLPPython(t *testing.T) string {
	t.Helper()
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("python HF dependencies unavailable via %q: %v", python, err)
	}
	return python
}

func hfTTTMLPPerformanceScript(t *testing.T) string {
	t.Helper()
	script, err := filepath.Abs(filepath.Join("testdata", "hf_ttt_mlp_perf.py"))
	if err != nil {
		t.Fatalf("resolve HF TTT performance script: %v", err)
	}
	return script
}

func exportTTTMLPPerformanceFixture(t *testing.T, modelDim, layers, seqLen, batchTokens int) string {
	t.Helper()
	var blocks strings.Builder
	for layer := 0; layer < layers; layer++ {
		if layer > 0 {
			blocks.WriteByte(',')
		}
		heads := modelDim / 64
		if heads < 1 {
			heads = 1
		}
		fmt.Fprintf(&blocks, `{"type":"ttt_mlp","heads":%d,"chunk_size":16}`, heads)
		blocks.WriteByte(',')
		blocks.WriteString(`{"type":"swiglu"}`)
	}
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, fmt.Sprintf(`{
		"name":"ttt_mlp_hf_training_perf",
		"model_dim":%d,
		"vocab_size":512,
		"seq_len":%d,
		"mlp_mult":4.0,
		"blocks":[%s],
		"training":{"steps":1,"batch_tokens":%d,"seed":6401}
	}`, modelDim, seqLen, blocks.String(), batchTokens))
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	return outDir
}
