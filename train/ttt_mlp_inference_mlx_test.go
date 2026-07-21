//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

func TestTTTMLPInferenceSessionReplayContinuationResetAndIsolation(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, cfg := newTTTMLPInferenceFixture(t)
	tokens := []uint16{1, 2, 3, 4, 5}

	replay, err := NewInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession replay: %v", err)
	}
	replayLogits, err := replay.EvalLogits(tokens)
	if err != nil {
		t.Fatalf("replay EvalLogits: %v", err)
	}
	if err := replay.Close(); err != nil {
		t.Fatal(err)
	}

	session, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatalf("NewTTTMLPInferenceSession: %v", err)
	}
	defer session.Close()
	oneShot, err := session.NewState()
	if err != nil {
		t.Fatal(err)
	}
	defer oneShot.Close()
	statefulLogits, err := session.Prefill(oneShot, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("stateful prefill: %v", err)
	}
	if diff := maxAbsDiffTTTStateful(replayLogits, statefulLogits); diff > 2e-4 {
		t.Fatalf("full-prefix replay vs stateful L_inf=%g want <=2e-4", diff)
	}

	split, err := session.NewState()
	if err != nil {
		t.Fatal(err)
	}
	defer split.Close()
	first, err := session.Prefill(split, []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	second, err := session.Prefill(split, []int{3})
	if err != nil {
		t.Fatal(err)
	}
	last, err := session.Decode(split, 4)
	if err != nil {
		t.Fatal(err)
	}
	continued := append(append(first, second...), last...)
	if diff := maxAbsDiffTTTStateful(continued, statefulLogits); diff > 2e-4 {
		t.Fatalf("partial continuation L_inf=%g want <=2e-4", diff)
	}

	if err := split.Reset(); err != nil {
		t.Fatal(err)
	}
	resetLogits, err := session.Prefill(split, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffTTTStateful(resetLogits, statefulLogits); diff > 1e-6 {
		t.Fatalf("reset L_inf=%g want <=1e-6", diff)
	}

	requestA, _ := session.NewState()
	requestB, _ := session.NewState()
	defer requestA.Close()
	defer requestB.Close()
	if _, err := session.Prefill(requestA, []int{9, 8, 7}); err != nil {
		t.Fatal(err)
	}
	isolated, err := session.Prefill(requestB, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffTTTStateful(isolated, statefulLogits); diff > 1e-6 {
		t.Fatalf("request state leaked L_inf=%g want <=1e-6", diff)
	}
	adaptedState, _ := session.NewState()
	freshState, _ := session.NewState()
	defer adaptedState.Close()
	defer freshState.Close()
	if _, err := session.PrefillLast(adaptedState, []int{11, 12, 13}); err != nil {
		t.Fatal(err)
	}
	adaptedProbe, err := session.Decode(adaptedState, 4)
	if err != nil {
		t.Fatal(err)
	}
	freshProbe, err := session.Decode(freshState, 4)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffTTTStateful(adaptedProbe, freshProbe); diff < 1e-7 {
		t.Fatalf("stream adaptation did not change probe logits: L_inf=%g", diff)
	}
	stats := session.Stats()
	if stats.Tokens == 0 || stats.Evaluations == 0 || stats.LiveStates < 2 {
		t.Fatalf("invalid runtime telemetry: %+v", stats)
	}

	if err := requestA.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Decode(requestA, 1); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("decode with closed state error=%v", err)
	}
	if got := session.Config(); got == nil || got.Name != cfg.Name {
		t.Fatalf("Config()=%v want %q", got, cfg.Name)
	}
}

func TestTTTMLPInferenceLongContextBenchmark(t *testing.T) {
	if os.Getenv("MIXLAB_TTT_MLP_LONG_BENCH") != "1" {
		t.Skip("set MIXLAB_TTT_MLP_LONG_BENCH=1 to run the 32k backend benchmark")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath := newTTTMLPLongBenchmarkFixture(t)
	session, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	state, err := session.NewState()
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	firstStart := time.Now()
	if _, err := session.Decode(state, 1); err != nil {
		t.Fatal(err)
	}
	firstToken := time.Since(firstStart)
	t.Logf("TTT_MLP_BENCH device=%q first_token_ms=%.3f", gpu.DeviceName(), float64(firstToken.Microseconds())/1000)

	for _, length := range []int{512, 2048, 8192, 32768} {
		tokens := make([]int, length)
		for i := range tokens {
			tokens[i] = (i*17 + 3) % session.Config().VocabSize
		}
		if err := state.Reset(); err != nil {
			t.Fatal(err)
		}
		if _, err := session.PrefillLast(state, tokens); err != nil {
			t.Fatalf("warm length %d: %v", length, err)
		}
		if err := state.Reset(); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if _, err := session.PrefillLast(state, tokens); err != nil {
			t.Fatalf("timed length %d: %v", length, err)
		}
		elapsed := time.Since(start)
		memory := gpu.MemoryStatsSnapshot()
		t.Logf("TTT_MLP_BENCH context=%d elapsed_ms=%.3f tok_per_sec=%.1f active_mib=%.2f cache_mib=%.2f peak_mib=%.2f programs=%d",
			length, float64(elapsed.Microseconds())/1000, float64(length)/elapsed.Seconds(),
			float64(memory.ActiveBytes)/(1<<20), float64(memory.CacheBytes)/(1<<20), float64(memory.PeakBytes)/(1<<20), session.Stats().ProgramVariants)
	}
	if err := state.Reset(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if _, err := session.Decode(state, i%session.Config().VocabSize); err != nil {
			t.Fatalf("warm decode %d: %v", i, err)
		}
	}
	if err := state.Reset(); err != nil {
		t.Fatal(err)
	}
	const decodeTokens = 512
	decodeStart := time.Now()
	for i := 0; i < decodeTokens; i++ {
		if _, err := session.Decode(state, i%session.Config().VocabSize); err != nil {
			t.Fatalf("timed decode %d: %v", i, err)
		}
	}
	decodeElapsed := time.Since(decodeStart)
	t.Logf("TTT_MLP_BENCH decode_tokens=%d decode_ms_per_token=%.3f decode_tok_per_sec=%.1f programs=%d",
		decodeTokens, float64(decodeElapsed.Microseconds())/1000/decodeTokens,
		decodeTokens/decodeElapsed.Seconds(), session.Stats().ProgramVariants)
}

func TestTTTMLPInferenceStateRejectsDifferentSession(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, _ := newTTTMLPInferenceFixture(t)
	first, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	state, err := first.NewState()
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	second, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if _, err := second.Decode(state, 1); err == nil || !strings.Contains(err.Error(), "different session") {
		t.Fatalf("cross-session state error=%v", err)
	}
}

func TestGenerateUsesTTTMLPStateBeyondConfiguredSeqLen(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, _ := newTTTMLPInferenceFixture(t)
	if err := runGenerate(configPath, weightsPath, 1, 1, 0, "token_ids:1,2,3,4,5"); err != nil {
		t.Fatalf("stateful generate beyond seq_len: %v", err)
	}
}

func TestTTTMLPBulkGenerationClosesEachSampleState(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, cfg := newTTTMLPInferenceFixture(t)
	session, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	plan := generationPlan{numSamples: 3, baseSeed: 23, eosTokenID: -1}
	opts := GenerateOptions{MaxTokens: 2, Temperature: 1, Prompt: "token_ids:1,2"}
	var output bytes.Buffer
	err = runGenerationSamples(plan, nil, &output, func(_ int, rng *rand.Rand) (generationSample, error) {
		return generateTTTMLPStatefulSample(session, cfg, opts, plan.eosTokenID, rng, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := session.Stats().LiveStates; got != 0 {
		t.Fatalf("live TTT states after bulk generation=%d want=0", got)
	}
	if got := len(strings.Split(strings.TrimSpace(output.String()), "\n")); got != plan.numSamples {
		t.Fatalf("output lines=%d want=%d", got, plan.numSamples)
	}
}

func TestTTTMLPStatefulGenerationAppliesLogitProcessor(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, cfg := newTTTMLPInferenceFixture(t)
	session, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	opts := GenerateOptions{
		MaxTokens: 2, Temperature: 1, Prompt: "token_ids:1,2",
		NewLogitProcessor: func() LogitProcessor { return &fixedTokenProcessor{token: 7} },
	}
	sample, err := generateTTTMLPStatefulSample(session, cfg, opts, -1, rand.New(rand.NewSource(3)), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := sample.tokens, []int{1, 2, 7, 7}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens=%v want=%v", got, want)
	}
}

func TestTTTMLPGenerationRejectsBatchedPersistentState(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	configPath, weightsPath, _ := newTTTMLPInferenceFixture(t)
	err := runGenerateWithOptions(GenerateOptions{
		ConfigPath: configPath, SafetensorsLoad: weightsPath,
		MaxTokens: 2, Temperature: 1, Prompt: "token_ids:1,2",
		NumSamples: 3, GenerationBatch: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "batched persistent inference state") {
		t.Fatalf("TTT batched generation error=%v", err)
	}
}

func newTTTMLPInferenceFixture(t *testing.T) (string, string, *ArchConfig) {
	t.Helper()
	dir := t.TempDir()
	cfg := &ArchConfig{
		Name: "ttt_mlp_stateful_test", ModelDim: 16, VocabSize: 32, SeqLen: 4,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "ttt_mlp", Heads: 2, ChunkSize: 4}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = "causal"
	cfg.Training.BatchTokens = cfg.SeqLen
	cfg.Training.Steps = 1
	cfg.Training.Seed = 17
	cfg.Training.LR = 1e-3
	configPath := filepath.Join(dir, "config.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
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
	return configPath, weightsPath, cfg
}

func newTTTMLPLongBenchmarkFixture(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	cfg := &ArchConfig{
		Name: "ttt_mlp_stateful_long_bench", ModelDim: 64, VocabSize: 1024, SeqLen: 32,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "ttt_mlp", Heads: 4, ChunkSize: 16}, {Type: "swiglu"}},
		Training:      DefaultTrainingSpec(),
	}
	cfg.Training.Objective = "causal"
	cfg.Training.BatchTokens = 32
	cfg.Training.Steps = 1
	cfg.Training.Seed = 42
	cfg.Training.LR = 3e-4
	configPath := filepath.Join(dir, "config.json")
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
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
	return configPath, weightsPath
}

func maxAbsDiffTTTStateful(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	maxDiff := 0.0
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}
