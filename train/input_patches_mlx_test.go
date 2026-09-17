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

func TestLinearPatchesAugmentedTrainingResume(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy unavailable")
	}
	dir := t.TempDir()
	config := strings.Replace(patchModelJSON("cls", "learned_xy", "none"), `"steps":30`, `"steps":4`, 1)
	cp := filepath.Join(dir, "model.json")
	if err := os.WriteFile(cp, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	input, labels, output := filepath.Join(dir, "images.npy"), filepath.Join(dir, "labels.tsv"), filepath.Join(dir, "dataset")
	frames := make([]float32, 6*48)
	for i := range frames {
		frames[i] = float32(i%23-11) / 11
	}
	writeNPYFloat32(t, input, []int{6, 6, 8}, frames)
	if err := os.WriteFile(labels, []byte("0\t0\n1\t1\n2\t2\n3\t0\n4\t1\n5\t2\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := runPrepare(PrepareOptions{ConfigPath: cp, Input: input, InputFormat: "continuous", Output: output, LabelFile: labels, ContinuousModality: "image"}); err != nil {
		t.Fatal(err)
	}
	load := func() *ArchConfig {
		c, e := LoadArchConfigQuiet(cp)
		if e != nil {
			t.Fatal(e)
		}
		return c
	}
	pattern := filepath.Join(output, "train_*.bin")
	checkpoints := filepath.Join(dir, "checkpoints")
	fullPath, resumePath := filepath.Join(dir, "full.safetensors"), filepath.Join(dir, "resumed.safetensors")
	if _, err := runTrain(load(), pattern, TrainOptions{SafetensorsPath: fullPath, CheckpointDir: checkpoints, CheckpointEvery: 2, LogEvery: 100, ValEvery: 100}); err != nil {
		t.Fatal(err)
	}
	if _, err := runTrain(load(), pattern, TrainOptions{SafetensorsPath: resumePath, Resume: filepath.Join(checkpoints, resumeManifestFilename(2)), LogEvery: 100, ValEvery: 100}); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(load())
	if err != nil {
		t.Fatal(err)
	}
	a, err := loadSafetensorsWeights(fullPath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	b, err := loadSafetensorsWeights(resumePath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		for j := range a[i] {
			if math.IsNaN(float64(a[i][j])) || math.IsInf(float64(a[i][j]), 0) || math.Abs(float64(a[i][j]-b[i][j])) > 1e-6 {
				t.Fatalf("resumed %s[%d]=%g uninterrupted=%g", shapes[i].Name, j, b[i][j], a[i][j])
			}
		}
	}
	t.Logf("matched %d tensors after augmented training resume", len(a))
}

func TestLinearPatchesNativeTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	for _, pool := range []string{"mean", "cls"} {
		t.Run(pool, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(patchModelJSON(pool, "learned_xy", "none")), t.Name())
			if err != nil {
				t.Fatal(err)
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			weights := initWeightData(shapes, 23, "normal", .15)
			raw := patchRawBatch()
			initial := clsForward(t, cfg, weights, raw)
			adapter := *cfg.InputAdapter
			adapter.Augment = nil
			evalCfg := *cfg
			evalCfg.InputAdapter = &adapter
			if diff := maxAbsDiffBidirectional(initial, clsForward(t, &evalCfg, weights, raw)); diff != 0 {
				t.Fatalf("eval augmented: %g", diff)
			}
			p, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			tr, err := initGPUTrainer(p, cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.CloseTrainer()
			batch, err := prepareClassificationBatch(cfg, raw, 12, 6)
			if err != nil {
				t.Fatal(err)
			}
			first, err := tr.EvaluateObjectiveGPU(batch, 2, 6)
			if err != nil {
				t.Fatal(err)
			}
			for step := 0; step < 30; step++ {
				loss, err := tr.(*mlxGPUTrainer).TrainObjectiveStepGPU(batch, 2, 6, .001)
				if err != nil {
					t.Fatal(err)
				}
				if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
					t.Fatalf("nonfinite step %d", step)
				}
			}
			last, err := tr.EvaluateObjectiveGPU(batch, 2, 6)
			if err != nil {
				t.Fatal(err)
			}
			if last >= first {
				t.Fatalf("loss %g -> %g", first, last)
			}
			trained, err := readTrainerWeights(tr)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"input_adapter_coord_x", "input_adapter_coord_y", "input_adapter_proj"} {
				i := weightShapeIndex(shapes, name)
				if maxAbsDiffBidirectional(weights[i], trained[i]) < 1e-7 {
					t.Fatalf("%s received no update", name)
				}
			}
			t.Logf("%s loss %g -> %g", pool, first, last)
		})
	}
}

func TestLinearPatchesNativeHFParity(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1")
	}
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skip("HF dependencies unavailable")
	}
	for _, tt := range []struct{ pool, coords, pos, clsPos string }{
		{"mean", "none", "none", ""}, {"mean", "learned_xy", "rope", ""},
		{"cls", "learned_xy", "none", ""}, {"cls", "learned_xy", "learned_absolute", ""},
		{"cls", "learned_xy", "learned_absolute", "middle"}, {"cls", "learned_xy", "learned_absolute", "tail"},
	} {
		t.Run(tt.pool+"_"+tt.coords+"_"+tt.pos+"_"+tt.clsPos, func(t *testing.T) {
			dir := t.TempDir()
			config := patchModelJSON(tt.pool, tt.coords, tt.pos)
			if tt.clsPos != "" {
				config = strings.Replace(config, `"pooling":"cls"`, `"pooling":"cls","cls_position":"`+tt.clsPos+`"`, 1)
			}
			cp, wp, _ := writeHFExportFixtureWithMutators(t, dir, config)
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
x = torch.tensor(json.loads(sys.argv[2])).reshape(2,6,8)
with torch.no_grad():
    actual = m(input_values=x).logits
    m.config.input_adapter.pop('augment', None)
    unaugmented = m(input_values=x).logits
expected = torch.tensor(json.loads(sys.argv[3])).reshape(2,3)
diff = (actual-expected).abs().max().item()
print(f'patch native/HF logit diff={diff:.3e}')
assert diff < 1e-4
torch.testing.assert_close(actual, unaugmented, atol=0, rtol=0)
# Independent coordinate/CLS positioning oracle using explicit raster indices.
with torch.no_grad():
    projected = m.input_adapter_norm(m.input_adapter(x))
    if m.input_adapter_coord_x is not None:
        for i in range(6):
            projected[:,i] += m.input_adapter_coord_x[i % 3] + m.input_adapter_coord_y[i // 3]
    if m.cls_token is not None:
        index = {'head': 0, 'middle': 3, 'tail': 6}[m.config.cls_position]
        projected = torch.cat((projected[:,:index], m.cls_token.unsqueeze(0).expand(2,-1,-1), projected[:,index:]),1)
    if m.position_embeddings is not None:
        projected += m.position_embeddings.weight[:projected.shape[1]].unsqueeze(0)
    torch.testing.assert_close(m._embed_features(input_values=x), projected, atol=1e-6, rtol=1e-6)
try:
    m(input_values=x[:,:5])
except ValueError:
    pass
else:
    raise AssertionError('accepted partial image')
`
			output, err := exec.Command(python, "-c", script, out, string(input), string(expected)).CombinedOutput()
			t.Logf("%s", output)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
