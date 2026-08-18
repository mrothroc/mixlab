//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestExportHFS4DContinuousNativePythonParity(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1 to run native continuous S4D classification parity")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("HF parity dependencies unavailable via %q: %v", python, err)
	}

	configs := []struct {
		name     string
		config   string
		mutators []hfExportWeightMutator
	}{
		{name: "bilinear_bidirectional_glu_post_residual", config: hfS4DContinuousConfig},
		{name: "zoh_unidirectional_fixed_b_pre_norm", config: `{
          "name":"s4d_continuous_zoh_export","model_dim":8,"seq_len":6,
          "positional_embedding":"none","norm_type":"rmsnorm","norm_placement":"pre",
          "input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
          "blocks":[{"type":"s4d","state_size":4}],
          "training":{"objective":"classification","classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},"steps":1,"batch_tokens":6,"seed":20260808}
        }`},
		{name: "zoh_unidirectional_sobolev_filter", config: `{
          "name":"s4d_continuous_sobolev_export","model_dim":8,"seq_len":6,
          "positional_embedding":"none","norm_type":"rmsnorm","norm_placement":"pre",
          "input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
          "blocks":[{"type":"s4d","state_size":4,"freq_scale":3,
            "sobolev_filter":{"beta_init":-0.5,"learning_rate":0.004}}],
          "training":{"objective":"classification","classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},"steps":1,"batch_tokens":6,"seed":20260809}
		}`},
		{name: "zoh_unidirectional_bounded_layer_sobolev", config: `{
          "name":"s4d_continuous_bounded_sobolev_export","model_dim":8,"seq_len":6,
          "positional_embedding":"none","norm_type":"rmsnorm","norm_placement":"pre",
          "input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
          "blocks":[{"type":"s4d","state_size":4,"freq_scale":3,
            "sobolev_filter":{"beta_init":0.5,"granularity":"layer","bounds":[-2,2],"trainable":false}}],
          "training":{"objective":"classification","classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},"steps":1,"batch_tokens":6,"seed":20260810}
        }`},
		// n_ssm < model_dim is the only case where the grouped channel mapping is
		// observable: with the default n_ssm == model_dim every group covers one
		// channel, so an interleaved and a contiguous expansion agree and cannot
		// distinguish a wrong template.
		{name: "zoh_unidirectional_grouped_trainable_b", config: `{
          "name":"s4d_continuous_grouped_export","model_dim":8,"seq_len":6,
          "positional_embedding":"none","norm_type":"rmsnorm","norm_placement":"pre",
          "input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
          "blocks":[{"type":"s4d","state_size":4,"n_ssm":2,"trainable_b":true}],
          "training":{"objective":"classification","classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},"steps":1,"batch_tokens":6,"seed":20260818}
        }`, mutators: []hfExportWeightMutator{distinctS4DGroupRows}},
	}
	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			runExportHFS4DContinuousNativePythonParity(t, python, tc.config, tc.mutators...)
		})
	}
}

// distinctS4DGroupRows gives every grouped S4D tensor a different value per
// group row. The initializers make all rows identical (log_A_real is a
// constant, A_imag depends only on the state index, B is ones/zeros), and with
// identical rows an interleaved and a contiguous group expansion produce the
// same kernel — so a grouped fixture cannot detect a wrong channel mapping
// unless the rows actually differ.
func distinctS4DGroupRows(weights [][]float32, shapes []WeightShape) error {
	for i, shape := range shapes {
		switch shape.Name {
		case "s4d_log_A_real", "s4d_A_imag", "s4d_B_real", "s4d_B_imag":
		default:
			continue
		}
		if len(shape.Shape) != 2 {
			continue
		}
		rows, cols := shape.Shape[0], shape.Shape[1]
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				// Give each group a clearly different pole: distinct decay and a
				// distinct oscillation frequency, so swapping which channels read
				// which group changes the output far beyond FFT rounding.
				switch shape.Name {
				case "s4d_log_A_real":
					weights[i][r*cols+c] = float32(math.Log(0.5) - float64(r)*1.5)
				case "s4d_A_imag":
					weights[i][r*cols+c] = float32(float64(r+1) * 2.0 * math.Pi * float64(c+1))
				case "s4d_B_real":
					weights[i][r*cols+c] = float32(1.0 + float64(r))
				case "s4d_B_imag":
					weights[i][r*cols+c] = float32(float64(r) * 0.75)
				}
			}
		}
	}
	return nil
}

func runExportHFS4DContinuousNativePythonParity(t *testing.T, python, config string, mutators ...hfExportWeightMutator) {
	t.Helper()
	dir := t.TempDir()
	cfgPath, weightsPath, _ := writeHFExportFixtureWithMutators(t, dir, config, mutators...)
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights: %v", err)
	}
	program, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	trainerInterface, err := initGPUTrainer(program, cfg, weights, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	frames := []float32{-1.0, -0.4, 0.2, 0.9, 0.35, -0.15}
	raw := trainBatch{
		frames: frames, labels: []int32{2}, validMask: repeatFloat32Train(cfg.SeqLen, 1),
	}
	batch, err := prepareClassificationBatch(cfg, raw, cfg.SeqLen, cfg.SeqLen)
	if err != nil {
		t.Fatalf("prepareClassificationBatch: %v", err)
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(batch, 1, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatalf("native S4D classifier forward: %v", err)
	}
	nativeLogits, err := readTrainerOutput(trainer, "classification_logits", []int{1, 3})
	if err != nil {
		t.Fatalf("read native classifier logits: %v", err)
	}

	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	framesJSON, _ := json.Marshal(frames)
	logitsJSON, _ := json.Marshal(nativeLogits)
	script := `
import json
import sys
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    sys.argv[1], trust_remote_code=True
)
model.eval()
values = torch.tensor(json.loads(sys.argv[2]), dtype=torch.float32).reshape(1, 6, 1)
expected = torch.tensor(json.loads(sys.argv[3]), dtype=torch.float64).reshape(1, 3)
with torch.no_grad():
    actual = model(input_values=values).logits.to(torch.float64)
    batched = model(input_values=values.repeat(2, 1, 1)).logits
diff = (actual - expected).abs().max().item()
print(f"s4d_continuous_native_classifier_parity: max_logit_diff={diff:.3e}")
if diff >= 1e-3:
    raise SystemExit(f"native classifier diff {diff:.3e} >= 1.000e-03")
if model.main_input_name != "input_values":
    raise SystemExit(f"unexpected main input name {model.main_input_name!r}")
if tuple(batched.shape) != (2, 3):
    raise SystemExit(f"unexpected batched classifier shape {tuple(batched.shape)}")

try:
    model(input_values=values, attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]))
except ValueError as error:
    if "fixed unpadded records" not in str(error):
        raise
else:
    raise SystemExit("continuous S4D export accepted a padded attention mask")
`
	output, err := exec.Command(
		python, "-c", script, outDir, string(framesJSON), string(logitsJSON),
	).CombinedOutput()
	t.Logf("native continuous S4D classifier parity output:\n%s", output)
	if err != nil {
		t.Fatalf("native continuous S4D classifier parity failed: %v", err)
	}
}
