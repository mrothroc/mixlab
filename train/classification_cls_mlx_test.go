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

func clsForward(t *testing.T, cfg *ArchConfig, weights [][]float32, raw trainBatch) []float32 {
	t.Helper()
	p, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := initGPUTrainer(p, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	b, err := prepareClassificationBatch(cfg, raw, cfg.Training.BatchTokens, cfg.SeqLen)
	if err != nil {
		t.Fatal(err)
	}
	B := cfg.Training.BatchTokens / cfg.SeqLen
	if _, err = tr.(*mlxGPUTrainer).EvaluateObjectiveGPUWithOutputs(b, B, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatal(err)
	}
	out, err := readTrainerOutput(tr.(*mlxGPUTrainer), "classification_logits", []int{B, 3})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestCLSPoolingNativeTrainingAndPadding(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	cfg, err := ParseArchConfig([]byte(clsContinuousConfig), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 23, "normal", 0.15)
	raw := trainBatch{frames: []float32{-1, 0.3, 0.8, 0.4, 99, -77}, labels: []int32{2}, validMask: []float32{1, 1, 1, 1, 0, 0}}
	long := clsForward(t, cfg, weights, raw)
	shortCfg := *cfg
	shortCfg.SeqLen = 4
	shortCfg.Training.BatchTokens = 4
	shortRaw := trainBatch{frames: raw.frames[:4], labels: raw.labels, validMask: raw.validMask[:4]}
	short := clsForward(t, &shortCfg, weights, shortRaw)
	if diff := maxAbsDiffBidirectional(long, short); diff > 1e-5 {
		t.Fatalf("padding changed logits: %g", diff)
	}
	batchCfg := *cfg
	batchCfg.Training.BatchTokens = 12
	batchRaw := trainBatch{
		frames: append(append([]float32(nil), raw.frames...), raw.frames...),
		labels: []int32{2, 1}, validMask: []float32{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1},
	}
	batched := clsForward(t, &batchCfg, weights, batchRaw)
	if diff := maxAbsDiffBidirectional(long, batched[:3]); diff > 1e-5 {
		t.Fatalf("batch changed first row logits: %g", diff)
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
	b, err := prepareClassificationBatch(cfg, raw, 6, 6)
	if err != nil {
		t.Fatal(err)
	}
	first, err := tr.EvaluateObjectiveGPU(b, 1, 6)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 30; step++ {
		loss, err := tr.(*mlxGPUTrainer).TrainObjectiveStepGPU(b, 1, 6, 0.001)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("nonfinite step %d", step)
		}
	}
	last, err := tr.EvaluateObjectiveGPU(b, 1, 6)
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
	i := weightShapeIndex(shapes, "cls_token")
	if maxAbsDiffBidirectional(trained[i], weights[i]) < 1e-7 {
		t.Fatal("CLS received no update")
	}
	t.Logf("CLS training loss %g -> %g; padded/unpadded parity passed", first, last)
}

func TestCLSPoolingNativeHFParity(t *testing.T) {
	t.Run("frames", func(t *testing.T) { runCLSPoolingNativeHFParity(t, clsContinuousConfig) })
	t.Run("tokens", func(t *testing.T) {
		config := strings.Replace(clsContinuousConfig, `"input_adapter":{"kind":"linear_frames","feature_dim":1,"norm":"none"}`, `"vocab_size":8`, 1)
		runCLSPoolingNativeHFParity(t, config)
	})
}

func runCLSPoolingNativeHFParity(t *testing.T, config string) {
	t.Helper()
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
	dir := t.TempDir()
	configPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, config)
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	raw := trainBatch{x: []int{1, 2, 3, 4, 7, 6}, frames: []float32{-1, 0.3, 0.8, 0.4, 99, -77}, labels: []int32{2}, validMask: []float32{1, 1, 1, 1, 0, 0}}
	fullOnly := cfg.EffectiveCLSPosition() == "middle"
	for _, block := range cfg.Blocks {
		fullOnly = fullOnly || block.Type == "s4d"
	}
	if fullOnly {
		raw.validMask = []float32{1, 1, 1, 1, 1, 1}
	}
	want := clsForward(t, cfg, weights, raw)
	fullRaw := raw
	fullRaw.validMask = []float32{1, 1, 1, 1, 1, 1}
	full := clsForward(t, cfg, weights, fullRaw)
	outDir := filepath.Join(dir, "hf")
	if cfg.LinearFramesEnabled() {
		tokenizerDir = ""
	}
	if err := RunExportHF(ExportHFOptions{ConfigPath: configPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatal(err)
	}
	wantJSON, _ := json.Marshal(want)
	fullJSON, _ := json.Marshal(full)
	script := `
import json, sys, torch
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], trust_remote_code=True).eval()
x = torch.tensor([-1., .3, .8, .4, 99., -77.]).reshape(1,6,1)
continuous = m.input_adapter_kind == 'linear_frames'
key = 'input_values' if continuous else 'input_ids'
if not continuous:
    x = torch.tensor([[1,2,3,4,7,6]])
mask = torch.tensor([[1,1,1,1,0,0]])
full_only = sys.argv[4] == 'true'
if full_only:
    mask = torch.ones_like(mask)
with torch.no_grad():
    actual = m(**{key:x}, attention_mask=mask).logits
    if not full_only:
        short = m(**{key:x[:,:4]}).logits
    batched = m(**{key:x.repeat(2,1,1) if continuous else x.repeat(2,1)}, attention_mask=mask.repeat(2,1)).logits
expected = torch.tensor(json.loads(sys.argv[2])).reshape(1,3)
diff = (actual-expected).abs().max().item()
print(f"CLS native/HF max logit diff={diff:.3e}")
assert diff <= 1e-4
if not full_only:
    torch.testing.assert_close(actual, short, atol=1e-5, rtol=1e-5)
else:
    try:
        m(**{key:x}, attention_mask=torch.tensor([[1,1,1,1,0,0]]))
    except ValueError:
        pass
    else:
        raise AssertionError('middle/S4D accepted padded input')
torch.testing.assert_close(actual.expand(2,-1), batched, atol=1e-5, rtol=1e-5)
assert tuple(m.cls_token.shape) == (1,8)
if continuous and __import__('os').environ.get('HF_CLS_VIT_REFERENCE') == '1':
    # Import the published implementation directly; unrelated package __init__
    # modules pull in torchvision. The frame adapter is matched explicitly:
    # this fixture starts with projected patches, without patch LayerNorms.
    import importlib.util, importlib.metadata
    path = importlib.metadata.distribution('vit-pytorch').locate_file('vit_pytorch/vit.py')
    spec = importlib.util.spec_from_file_location('reference_vit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.ViT(image_size=(2,3), patch_size=1, channels=1, dim=8,
                     depth=1, heads=2, dim_head=4, mlp_dim=m.blocks[0].ff1.weight.shape[1],
                     num_classes=3, pool='cls').eval()
    ref.to_patch_embedding[1] = torch.nn.Identity()
    ref.to_patch_embedding[3] = torch.nn.Identity()
    def linear(dst, src):
        dst.weight.copy_(src.weight.T)
        if dst.bias is not None:
            dst.bias.zero_() if src.bias is None else dst.bias.copy_(src.bias)
    with torch.no_grad():
        linear(ref.to_patch_embedding[2], m.input_adapter)
        ref.cls_token.copy_(m.cls_token)
        ref.pos_embedding.copy_(m.position_embeddings.weight)
        a, f = ref.transformer.layers[0]
        b = m.blocks[0]
        a.norm.load_state_dict(b.norm.state_dict())
        a.to_qkv.weight.copy_(torch.cat([b.wq.weight.T, b.wk.weight.T, b.wv.weight.T]))
        linear(a.to_out[0], b.wo)
        f.net[0].load_state_dict(b.ffn_norm.state_dict())
        linear(f.net[1], b.ff1)
        linear(f.net[4], b.ff2)
        ref.transformer.norm.load_state_dict(m.final_norm.state_dict())
        ref.mlp_head.load_state_dict(m.classifier.state_dict())
        expected_full = torch.tensor(json.loads(sys.argv[3])).reshape(1,3)
        actual_ref = ref(x.reshape(1,1,2,3))
    ref_diff = (actual_ref-expected_full).abs().max().item()
    print(f"vit-pytorch {importlib.metadata.version('vit-pytorch')} matched-adapter CLS/native diff={ref_diff:.3e}")
    assert ref_diff <= 1e-4
`
	fullOnlyJSON, _ := json.Marshal(fullOnly)
	out, err := exec.Command(python, "-c", script, outDir, string(wantJSON), string(fullJSON), string(fullOnlyJSON)).CombinedOutput()
	t.Logf("%s", out)
	if err != nil {
		t.Fatal(err)
	}
}

func TestCLSPoolingNativeAdaptersAndPositions(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	for _, adapter := range []string{"token", "linear_frames", "discrete_codebooks"} {
		for _, pos := range []string{"none", "rope", "learned_absolute"} {
			t.Run(adapter+"/"+pos, func(t *testing.T) {
				text := strings.Replace(clsContinuousConfig, `"learned_absolute"`, `"`+pos+`"`, 1)
				var doc map[string]any
				if err := json.Unmarshal([]byte(text), &doc); err != nil {
					t.Fatal(err)
				}
				switch adapter {
				case "token":
					delete(doc, "input_adapter")
					doc["vocab_size"] = 8
				case "discrete_codebooks":
					doc["input_adapter"] = map[string]any{"kind": "discrete_codebooks", "num_codebooks": 2, "codebook_vocab_size": 8, "fusion": "attention_mlp", "norm": "layernorm"}
				}
				encoded, _ := json.Marshal(doc)
				cfg, err := ParseArchConfig(encoded, t.Name())
				if err != nil {
					t.Fatal(err)
				}
				shapes, err := computeWeightShapes(cfg)
				if err != nil {
					t.Fatal(err)
				}
				weights := initWeightData(shapes, 23, "normal", 0.1)
				raw := trainBatch{x: []int{1, 2, 3, 4, 7, 6}, frames: []float32{-1, .3, .8, .4, 99, -77}, codebooks: []int32{1, 2, 3, 4, 5, 6, 1, 3, 7, 7, 6, 6}, labels: []int32{2}, validMask: []float32{1, 1, 1, 1, 0, 0}}
				long := clsForward(t, cfg, weights, raw)
				shortCfg := *cfg
				shortCfg.SeqLen = 4
				shortCfg.Training.BatchTokens = 4
				short := clsForward(t, &shortCfg, weights, raw)
				if diff := maxAbsDiffBidirectional(long, short); diff > 1e-5 {
					t.Fatalf("padding diff %g", diff)
				}
				for _, v := range long {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite logits")
					}
				}
			})
		}
	}
}
