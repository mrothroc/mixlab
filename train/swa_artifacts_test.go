package train

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeWeightReader struct {
	weights [][]float32
}

func (f fakeWeightReader) ReadWeights() ([][]float32, error) {
	return cloneWeights(f.weights), nil
}

func TestSWAFinalAndAveragedSafetensorArtifacts(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	swa := cloneWeights(weights)
	swa[0][0] += 1
	dir := t.TempDir()
	base := filepath.Join(dir, "model.safetensors")

	artifacts, err := exportTrainingSafetensorsArtifacts(cfg, fakeWeightReader{weights: weights}, shapes, TrainOptions{
		SafetensorsPath: base,
		Quantize:        "none",
	}, swa)
	if err != nil {
		t.Fatalf("exportTrainingSafetensorsArtifacts: %v", err)
	}

	wantFinal := filepath.Join(dir, "model.final.safetensors")
	wantSWA := filepath.Join(dir, "model.swa.safetensors")
	if artifacts.FinalPath != wantFinal || artifacts.SWAPath != wantSWA {
		t.Fatalf("artifacts = %+v, want final=%s swa=%s", artifacts, wantFinal, wantSWA)
	}
	finalWeights, err := loadSafetensorsWeights(artifacts.FinalPath, shapes)
	if err != nil {
		t.Fatalf("load final weights: %v", err)
	}
	swaWeights, err := loadSafetensorsWeights(artifacts.SWAPath, shapes)
	if err != nil {
		t.Fatalf("load swa weights: %v", err)
	}
	if finalWeights[0][0] == swaWeights[0][0] {
		t.Fatalf("final and SWA weights did not differ after EMA update: %g", finalWeights[0][0])
	}
	if finalWeights[0][0] != weights[0][0] {
		t.Fatalf("final weight = %g, want live %g", finalWeights[0][0], weights[0][0])
	}
	if swaWeights[0][0] != swa[0][0] {
		t.Fatalf("swa weight = %g, want averaged %g", swaWeights[0][0], swa[0][0])
	}
	if _, err := os.Stat(base); err == nil {
		t.Fatalf("unsuffixed base artifact %s was written; expected standardized final/SWA names", base)
	}
}

func TestFinalSafetensorArtifactPreservesLegacyPathWithoutSWA(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	artifacts, err := exportTrainingSafetensorsArtifacts(cfg, fakeWeightReader{weights: weights}, shapes, TrainOptions{
		SafetensorsPath: path,
		Quantize:        "none",
	}, nil)
	if err != nil {
		t.Fatalf("exportTrainingSafetensorsArtifacts: %v", err)
	}
	if artifacts.FinalPath != path || artifacts.SWAPath != "" {
		t.Fatalf("artifacts = %+v, want only %s", artifacts, path)
	}
	if _, err := loadSafetensorsWeights(path, shapes); err != nil {
		t.Fatalf("load final weights: %v", err)
	}
}

func TestSWACheckpointArtifactsUseFinalAndSuffixedNames(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	swa := cloneWeights(weights)
	swa[0][0] += 1
	dir := t.TempDir()

	artifacts, err := writeCheckpoint(cfg, fakeWeightReader{weights: weights}, shapes, dir, 12, swa)
	if err != nil {
		t.Fatalf("writeCheckpoint: %v", err)
	}

	wantFinal := filepath.Join(dir, "step_000012.final.safetensors")
	wantSWA := filepath.Join(dir, "step_000012.swa.safetensors")
	if artifacts.FinalPath != wantFinal || artifacts.SWAPath != wantSWA {
		t.Fatalf("checkpoint artifacts = %+v, want final=%s swa=%s", artifacts, wantFinal, wantSWA)
	}
	for _, path := range []string{artifacts.FinalPath, artifacts.SWAPath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
	}
}

func TestSWAWeightsCanBeSelectedForHFExport(t *testing.T) {
	dir := t.TempDir()
	cfgPath, _, tokenizerDir := writeHFExportFixture(t, dir, `{
	  "name": "swa_hf_export",
	  "model_dim": 8,
	  "vocab_size": 16,
	  "seq_len": 3,
	  "blocks": [{"type": "plain", "heads": 2}],
	  "training": {"steps": 1, "lr": 0.001, "batch_tokens": 3, "seed": 7}
	}`)
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	swa := cloneWeights(weights)
	swa[0][0] += 7
	artifacts, err := exportTrainingSafetensorsArtifacts(cfg, fakeWeightReader{weights: weights}, shapes, TrainOptions{
		SafetensorsPath: filepath.Join(dir, "trained.safetensors"),
		Quantize:        "none",
	}, swa)
	if err != nil {
		t.Fatalf("exportTrainingSafetensorsArtifacts: %v", err)
	}

	outDir := filepath.Join(dir, "hf")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: artifacts.SWAPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF with SWA weights: %v", err)
	}
	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported HF safetensors: %v", err)
	}
	embed, err := decodeSafetensorFloat32("embed_tokens.weight", shapes[0].Shape, tensors)
	if err != nil {
		t.Fatalf("decode HF embed: %v", err)
	}
	if embed[0] != swa[0][0] {
		t.Fatalf("HF export used embed[0]=%g, want selected SWA weight %g", embed[0], swa[0][0])
	}
}

func TestSWAEMAValidationBandCPUOracle(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	_ = shapes
	ema := cloneWeights(weights)
	next := cloneWeights(weights)
	for i := range next {
		for j := range next[i] {
			next[i][j] += 0.001
		}
	}
	updateEMAWeights(ema, next, 0.9)

	tokens := [][]int{{0, 1, 2, 3}}
	targets := [][]int{{1, 2, 3, 4}}
	liveLoss := cpuCrossEntropy(runNativeCPUForward(t, cfg, weights, tokens), targets)
	emaLoss := cpuCrossEntropy(runNativeCPUForward(t, cfg, ema, tokens), targets)
	if !isFinitePositive(liveLoss) {
		t.Fatalf("live validation loss is not finite positive: %g", liveLoss)
	}
	if !isFinitePositive(emaLoss) {
		t.Fatalf("EMA validation loss is not finite positive: %g", emaLoss)
	}
	if delta := absFloat64(liveLoss - emaLoss); delta > 0.25 {
		t.Fatalf("EMA validation loss drift = %g, live=%g ema=%g", delta, liveLoss, emaLoss)
	}
}

func TestSWADocsAndExampleConfig(t *testing.T) {
	cfg, err := LoadArchConfig(filepath.Join("..", "examples", "swa_ema_tiny.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig(swa_ema_tiny): %v", err)
	}
	if cfg.Training.SWAStart <= 0 || cfg.Training.SWAInterval <= 0 || cfg.Training.SWADecay <= 0 {
		t.Fatalf("SWA example did not enable averaging: start=%d interval=%d decay=%g", cfg.Training.SWAStart, cfg.Training.SWAInterval, cfg.Training.SWADecay)
	}
	required := map[string][]string{
		filepath.Join("..", "docs", "config-reference.md"): {
			"-swa-start",
			"-swa-decay",
			"-swa-interval",
			".final.safetensors",
			".swa.safetensors",
			"averaged weights",
		},
		filepath.Join("..", "docs", "hf-export.md"): {
			"swa_ema_tiny",
			".swa.safetensors",
		},
		filepath.Join("..", "examples", "README.md"): {
			"swa_ema_tiny.json",
			"swa_start",
			"swa_decay",
			"averaged weights",
		},
	}
	for path, needles := range required {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		source := string(data)
		if strings.Contains(source, "TODO") || strings.Contains(source, "TBD") {
			t.Fatalf("%s contains unfinished SWA documentation wording", path)
		}
		for _, needle := range needles {
			if !strings.Contains(source, needle) {
				t.Fatalf("%s missing %q", path, needle)
			}
		}
	}
}

func isFinitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func absFloat64(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func smallSWAArtifactConfig() *ArchConfig {
	cfg := &ArchConfig{
		Name:      "swa_artifact_test",
		ModelDim:  8,
		VocabSize: 16,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Steps = 1
	cfg.Training.LR = 1e-3
	cfg.Training.Seed = 7
	cfg.Training.BatchTokens = 4
	cfg.Training.ApplyDefaults()
	return cfg
}

func smallSWAArtifactWeights(t *testing.T, cfg *ArchConfig) ([]WeightShape, [][]float32) {
	t.Helper()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	return shapes, weights
}
