//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFAttentionBiasNativePythonParity(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend unavailable")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("Python HF dependencies: %v", err)
	}
	script, err := filepath.Abs(filepath.Join("testdata", "hf_parity_check.py"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name             string
		qkv, out, masked bool
		extra            string
	}{
		{"out_only_causal", false, true, false, ""},
		{"qkv_only_causal_gqa_gate", true, false, false, `,"kv_heads":1,"attn_value_gate":true`},
		{"both", true, true, false, ""},
		{"out_only_masked", false, true, true, ""},
		{"qkv_only_shared_relative_masked", true, false, true, `,"kv_heads":1,"attn_value_gate":true,"relative_attention":"deberta_p2c_c2p","relative_attention_window":3,"relative_attention_parameterization":"shared_qk_reuse"`},
		{"out_only_differential", false, true, false, `,"differential_attention":true`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := attentionBiasFixture(tc.qkv, tc.out, tc.extra)
			if tc.masked {
				config = strings.Replace(config, `"steps":1`, `"steps":1,"objective":"mntp","mlm_mask_token_id":1`, 1)
			}
			runNativePythonParityCase(t, python, script, config, tc.masked, false, false, nonzeroAttentionBiases)
		})
	}
	t.Run("patch_classifier_output_bias", func(t *testing.T) {
		runAttentionBiasPatchParity(t, python)
	})
}

func runAttentionBiasPatchParity(t *testing.T, python string) {
	t.Helper()
	rawConfig := strings.Replace(patchModelJSON("cls", "learned_xy", "none"), `"heads":2`, `"heads":2,"attn_qkv_bias":false,"attn_out_bias":true`, 1)
	dir := t.TempDir()
	cp, wp, _ := writeHFExportFixtureWithMutators(t, dir, rawConfig, nonzeroAttentionBiases)
	cfg, err := LoadArchConfigQuiet(cp)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := loadSafetensorsWeights(wp, shapes)
	if err != nil {
		t.Fatal(err)
	}
	raw := patchRawBatch()
	want := clsForward(t, cfg, weights, raw)
	out := filepath.Join(dir, "hf")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cp, SafetensorsLoad: wp, OutputDir: out}); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal(raw.frames)
	expected, _ := json.Marshal(want)
	script := `
import json, sys, torch
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], trust_remote_code=True).eval()
b = m.blocks[0]
assert b.wq.bias is None and b.wk.bias is None and b.wv.bias is None
assert b.wo.bias is not None and b.wo.bias.abs().max().item() > 0
x = torch.tensor(json.loads(sys.argv[2])).reshape(2,6,8)
with torch.no_grad():
    actual = m(input_values=x).logits
expected = torch.tensor(json.loads(sys.argv[3])).reshape(2,3)
diff = (actual-expected).abs().max().item()
print(f'output-bias patch classifier native/HF diff={diff:.3e}')
assert diff < 1e-4
`
	output, err := exec.Command(python, "-c", script, out, string(input), string(expected)).CombinedOutput()
	t.Logf("%s", output)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAttentionBiasTinyTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend unavailable")
	}
	path := filepath.Join(t.TempDir(), "train_000.bin")
	writeInferenceShard(t, path, []uint16{1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10})
	for _, tc := range []struct {
		name     string
		qkv, out bool
		extra    string
	}{
		{"out_only", false, true, ""},
		{"qkv_only", true, false, ""},
		{"parallel_out", false, true, `,"parallel_residual":true`},
		{"parallel_qkv", true, false, `,"parallel_residual":true`},
		{"kv_out", false, true, `,"kv_source":1`},
		{"kv_qkv", true, false, `,"kv_source":1`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := attentionBiasFixture(tc.qkv, tc.out, tc.extra)
			if strings.HasPrefix(tc.name, "parallel") {
				raw = strings.Replace(raw, `}],`, `},{"type":"geglu"}],`, 1)
			}
			if strings.HasPrefix(tc.name, "kv") {
				raw = strings.Replace(raw, `"blocks":[`, `"blocks":[{"type":"plain","heads":2},`, 1)
			}
			c, err := ParseArchConfig([]byte(raw), tc.name)
			if err != nil {
				t.Fatal(err)
			}
			c.Training.Steps = 4
			c.Training.Optimizer = "adamw"
			c.Training.LR = 1e-4
			c.Training.GradClip = 1
			result, err := runTrain(c, path, TrainOptions{LogEvery: 0, ValEvery: 0})
			if err != nil {
				t.Fatal(err)
			}
			if math.IsNaN(result.LastLoss) || math.IsInf(result.LastLoss, 0) {
				t.Fatalf("nonfinite loss %g", result.LastLoss)
			}
		})
	}
}
