//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

// TestEnergyMultiheadExampleRunsTwoMLXSteps exercises the full energy head path
// end to end — minimal-pair batch assembly, the pairwise ranking loss forward,
// and the backward/optimizer update — which the op oracle and IR-shape tests do
// not cover on their own.
func TestEnergyMultiheadExampleRunsTwoMLXSteps(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	cfg, err := LoadArchConfig(filepath.Join("examples", "multihead_mntp_energy_tiny.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	cfg.Training.Steps = 2
	cfg.Training.WarmupSteps = 0
	cfg.Training.HoldSteps = 0
	cfg.Training.WarmdownSteps = 0

	tmp := t.TempDir()
	pairsPath := filepath.Join(tmp, "pairs.train.jsonl")
	writeMinimalPairFixture(t, pairsPath, cfg.VocabSize, cfg.SeqLen, 8)
	// Absolute path bypasses config-relative resolution.
	cfg.Training.MinimalPair.Path = pairsPath

	trainDir := filepath.Join(tmp, "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), rtdSmokeTokens(cfg.VocabSize, 4096))

	result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 0, ValEvery: 0})
	if err != nil {
		t.Fatalf("runTrain: %v", err)
	}
	for name, v := range map[string]float64{"first": result.FirstLoss, "last": result.LastLoss} {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s loss=%g, want finite positive", name, v)
		}
	}
}

func TestScoreEBMReadsEnergyHeadOutputMLX(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	cfg := parseTrainMinimalPairConfig(t, `"minimal_pair": {"path": "pairs.jsonl"}`)
	scoreBatch := 2
	cfg.Training.BatchTokens = scoreBatch * cfg.SeqLen
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		Objective:       "multihead",
		DropoutInactive: true,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainerIface.CloseTrainer()
	evaluator, ok := trainerIface.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer does not implement diffusionGenerationEvaluator")
	}
	energies, _, err := scoreEBMSequencesDetailed(cfg, evaluator, [][]int{{1, 2, 3}, {1, 3, 2}}, nil, scoreBatch, false)
	if err != nil {
		t.Fatalf("scoreEBMSequencesDetailed: %v", err)
	}
	if len(energies) != 2 {
		t.Fatalf("energies len=%d, want 2", len(energies))
	}
	for i, energy := range energies {
		if math.IsNaN(float64(energy)) || math.IsInf(float64(energy), 0) {
			t.Fatalf("energy[%d]=%g, want finite", i, energy)
		}
	}
}

func writeMinimalPairFixture(t *testing.T, path string, vocabSize, seqLen, pairs int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create minimal pair fixture: %v", err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	span := vocabSize - 2
	if span <= 0 {
		span = 1
	}
	for p := 0; p < pairs; p++ {
		clean := make([]int, seqLen)
		corrupt := make([]int, seqLen)
		for i := 0; i < seqLen; i++ {
			clean[i] = (p+i)%span + 2
			// Swap two adjacent tokens for the corrupt member.
			corrupt[i] = (p+i)%span + 2
		}
		if seqLen >= 2 {
			corrupt[0], corrupt[1] = corrupt[1], corrupt[0]
		}
		rec := minimalPairRecord{
			ID:      "mp_fixture_" + string(rune('a'+p)),
			Clean:   clean,
			Corrupt: corrupt,
			Family:  "word_order",
		}
		if err := enc.Encode(rec); err != nil {
			t.Fatalf("encode minimal pair record: %v", err)
		}
	}
}
