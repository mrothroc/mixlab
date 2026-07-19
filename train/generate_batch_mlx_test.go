//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
)

func TestBulkReplayGenerationMLXDeterministic(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := &ArchConfig{
		Name: "bulk_generate_mlx", ModelDim: 8, VocabSize: 8, SeqLen: 4,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.BatchTokens = cfg.SeqLen
	cfg.Training.Seed = 19
	dir, configPath, weightsPath := newBulkGenerationFixture(t, cfg)

	outputs := make([][]byte, 2)
	for run, batchSize := range []int{1, 3} {
		outputPath := filepath.Join(dir, "samples_"+string(rune('a'+run))+".txt")
		err := runGenerateWithOptions(GenerateOptions{
			ConfigPath: configPath, SafetensorsLoad: weightsPath,
			MaxTokens: 2, Temperature: 1, Prompt: "token_ids:1",
			NumSamples: 4, GenerationBatch: batchSize, GenerationSeed: 7, OutputPath: outputPath,
		})
		if err != nil {
			t.Fatal(err)
		}
		outputs[run], err = os.ReadFile(outputPath)
		if err != nil {
			t.Fatal(err)
		}
	}
	if string(outputs[0]) != string(outputs[1]) {
		t.Fatalf("generation differs between batch widths 1 and 3:\n%s\n---\n%s", outputs[0], outputs[1])
	}
	lines := strings.Split(strings.TrimSpace(string(outputs[0])), "\n")
	if len(lines) != 4 {
		t.Fatalf("output lines=%d want=4: %q", len(lines), outputs[0])
	}
	for _, line := range lines {
		if strings.Contains(line, "generated token_ids:") || len(strings.Split(line, ",")) != 3 {
			t.Fatalf("invalid machine generation record %q", line)
		}
	}
}

func TestBatchedGenerationMLXThroughputProbe(t *testing.T) {
	if os.Getenv("MIXLAB_GENERATION_PERF_PROBE") != "1" {
		t.Skip("set MIXLAB_GENERATION_PERF_PROBE=1 to run native generation throughput comparison")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := &ArchConfig{
		Name: "bulk_generate_perf", ModelDim: 64, VocabSize: 32, SeqLen: 32,
		TieEmbeddings: true,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4}, {Type: "swiglu"},
			{Type: "plain", Heads: 4}, {Type: "swiglu"},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.BatchTokens = cfg.SeqLen
	cfg.Training.Seed = 29
	dir, configPath, weightsPath := newBulkGenerationFixture(t, cfg)

	outputs := make([][]byte, 2)
	durations := make([]time.Duration, 2)
	for run, batchSize := range []int{1, 16} {
		outputPath := filepath.Join(dir, fmt.Sprintf("perf_batch_%d.txt", batchSize))
		start := time.Now()
		err := runGenerateWithOptions(GenerateOptions{
			ConfigPath: configPath, SafetensorsLoad: weightsPath,
			MaxTokens: 16, Temperature: 0.8, TopK: 16, Prompt: "token_ids:1",
			NumSamples: 128, GenerationBatch: batchSize, GenerationSeed: 31, OutputPath: outputPath,
		})
		durations[run] = time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		outputs[run], err = os.ReadFile(outputPath)
		if err != nil {
			t.Fatal(err)
		}
	}
	if string(outputs[0]) != string(outputs[1]) {
		t.Fatal("throughput probe output differs between batch widths 1 and 16")
	}
	if durations[1] >= durations[0] {
		t.Fatalf("batched generation did not improve throughput: batch1=%s batch16=%s", durations[0], durations[1])
	}
	t.Logf("generation throughput: batch1=%s batch16=%s speedup=%.2fx", durations[0], durations[1], float64(durations[0])/float64(durations[1]))
}

func TestGenerationGatheredLogitsMLXMatchFullEval(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := &ArchConfig{
		Name: "gathered_generation_logits", ModelDim: 8, VocabSize: 9, SeqLen: 4,
		TieEmbeddings: false,
		LogitSoftcap:  3,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.BatchTokens = 3 * cfg.SeqLen
	cfg.Training.Seed = 37
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	fullProgram, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	generationProgram, err := arch.BuildGenerationIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	fullTrainer, err := initGPUTrainer(fullProgram, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer fullTrainer.CloseTrainer()
	generationTrainer, err := initGPUTrainer(generationProgram, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer generationTrainer.CloseTrainer()

	contexts := [][]int{{1}, {2, 3}, {4, 5, 6}}
	xTok := make([]int, cfg.Training.BatchTokens)
	yTok := make([]int, cfg.Training.BatchTokens)
	positions := make([]int, len(contexts))
	for row, context := range contexts {
		fillGenerationBatchRow(xTok, yTok, positions, row, cfg.SeqLen, context)
	}
	if _, err := fullTrainer.EvaluateGPU(xTok, yTok, len(contexts), cfg.SeqLen); err != nil {
		t.Fatal(err)
	}
	fullLogits, err := readTrainerOutput(fullTrainer, "logits", []int{cfg.Training.BatchTokens, cfg.VocabSize})
	if err != nil {
		t.Fatal(err)
	}
	batched, ok := generationTrainer.(batchedCausalGenerationEvaluator)
	if !ok {
		t.Fatal("generation trainer does not expose batched evaluator")
	}
	gathered, err := batched.EvaluateGenerationGPU(xTok, yTok, positions, len(contexts), cfg.SeqLen)
	if err != nil {
		t.Fatal(err)
	}
	for row, position := range positions {
		want := fullLogits[(row*cfg.SeqLen+position)*cfg.VocabSize : (row*cfg.SeqLen+position+1)*cfg.VocabSize]
		got := gathered[row*cfg.VocabSize : (row+1)*cfg.VocabSize]
		for token := range want {
			diff := got[token] - want[token]
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-5 {
				t.Fatalf("row=%d position=%d token=%d gathered=%g full=%g diff=%g", row, position, token, got[token], want[token], diff)
			}
		}
	}
}

func newBulkGenerationFixture(t testing.TB, cfg *ArchConfig) (dir, configPath, weightsPath string) {
	t.Helper()
	dir = t.TempDir()
	configPath = filepath.Join(dir, "config.json")
	configJSON, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, configJSON, 0o644); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	weightsPath = filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(weightsPath, cfg, shapes, weights); err != nil {
		t.Fatal(err)
	}
	return dir, configPath, weightsPath
}
