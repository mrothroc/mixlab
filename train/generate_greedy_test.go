package train

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// -temperature 0 selects greedy decoding. The RunPod handler and docker/README
// promised this from July 2026, but the CLI rejected any temperature <= 0, so
// a job asking for deterministic output failed with "-temperature must be > 0".
func TestGreedyDecodingPicksTheHighestLogit(t *testing.T) {
	logits := []float32{0.1, 2.5, -1, 2.4}
	for seed := int64(0); seed < 20; seed++ {
		got, err := sampleNextToken(logits, 0, 0, rand.New(rand.NewSource(seed)))
		if err != nil {
			t.Fatal(err)
		}
		if got != 1 {
			t.Fatalf("seed %d: greedy chose token %d, want 1", seed, got)
		}
	}
}

func TestGreedyDecodingIgnoresTopKAndBreaksTiesByLowestToken(t *testing.T) {
	logits := []float32{0, 3, 1, 3}
	for _, topK := range []int{0, 1, 2, 4} {
		got, err := sampleNextToken(logits, 0, topK, rand.New(rand.NewSource(7)))
		if err != nil {
			t.Fatal(err)
		}
		if got != 1 {
			t.Fatalf("top-k %d: greedy chose token %d, want 1 (lowest of the tied maxima)", topK, got)
		}
	}
}

func TestGreedyDecodingNeverPicksAMaskedToken(t *testing.T) {
	// Grammar constraints mask with -Inf before sampling.
	masked := float32(math.Inf(-1))
	got, err := sampleNextToken([]float32{masked, -5, masked, -7}, 0, 0, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("greedy chose token %d, want 1", got)
	}
	if _, err := sampleNextToken([]float32{masked, masked}, 0, 0, rand.New(rand.NewSource(1))); err == nil || !strings.Contains(err.Error(), "no finite logits") {
		t.Fatalf("all-masked greedy step: got %v, want the no-finite-logits error", err)
	}
}

func TestGreedyDecodingDoesNotConsumeTheRNG(t *testing.T) {
	// Later stochastic steps in the same stream must not shift.
	rng := rand.New(rand.NewSource(3))
	if _, err := sampleNextToken([]float32{1, 2}, 0, 0, rng); err != nil {
		t.Fatal(err)
	}
	if got, want := rng.Int63(), rand.New(rand.NewSource(3)).Int63(); got != want {
		t.Fatal("greedy decoding consumed a draw from the sampling RNG")
	}
}

func TestSamplingTemperatureMustBeFiniteAndNotNegative(t *testing.T) {
	for _, temperature := range []float32{-0.5, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := sampleNextToken([]float32{1, 2}, temperature, 0, rand.New(rand.NewSource(1))); err == nil {
			t.Errorf("temperature %g was accepted", temperature)
		}
	}
}

func TestGenerationPlanAcceptsGreedyAndRejectsNegativeTemperature(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 8}
	if _, err := buildGenerationPlan(GenerateOptions{Temperature: 0}, cfg, nil); err != nil {
		t.Fatalf("temperature 0 (greedy) rejected: %v", err)
	}
	for _, temperature := range []float32{-1, float32(math.NaN())} {
		_, err := buildGenerationPlan(GenerateOptions{Temperature: temperature}, cfg, nil)
		if err == nil || !strings.Contains(err.Error(), "-temperature") {
			t.Errorf("temperature %g: got %v, want a -temperature error", temperature, err)
		}
	}
}
