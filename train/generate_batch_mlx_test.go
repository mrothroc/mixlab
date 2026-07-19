//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestBulkReplayGenerationMLXDeterministic(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	cfg := &ArchConfig{
		Name: "bulk_generate_mlx", ModelDim: 8, VocabSize: 8, SeqLen: 4,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = arch.ObjectiveCausal
	cfg.Training.BatchTokens = cfg.SeqLen
	cfg.Training.Seed = 19
	configPath := filepath.Join(dir, "config.json")
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
	weightsPath := filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(weightsPath, cfg, shapes, weights); err != nil {
		t.Fatal(err)
	}

	outputs := make([][]byte, 2)
	for run := range outputs {
		outputPath := filepath.Join(dir, "samples_"+string(rune('a'+run))+".txt")
		err := runGenerateWithOptions(GenerateOptions{
			ConfigPath: configPath, SafetensorsLoad: weightsPath,
			MaxTokens: 2, Temperature: 1, Prompt: "token_ids:1",
			NumSamples: 4, GenerationSeed: 7, OutputPath: outputPath,
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
		t.Fatalf("bulk generation is not deterministic:\n%s\n---\n%s", outputs[0], outputs[1])
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
