package train

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGenerationPromptTokens_DefaultRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	got, err := generationPromptTokens("", 10, rng)
	if err != nil {
		t.Fatalf("generationPromptTokens: %v", err)
	}
	if len(got) != 1 || got[0] < 0 || got[0] >= 10 {
		t.Fatalf("unexpected prompt tokens: %v", got)
	}
}

func TestGenerationPromptTokens_ExplicitIDs(t *testing.T) {
	got, err := generationPromptTokens("token_ids:0, 2,5", 10, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatalf("generationPromptTokens: %v", err)
	}
	want := []int{0, 2, 5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("generationPromptTokens = %v, want %v", got, want)
	}
}

func TestGenerationBatch(t *testing.T) {
	xTok, yTok, lastPos := generationBatch([]int{4, 7, 9}, 6)
	if lastPos != 2 {
		t.Fatalf("lastPos = %d, want 2", lastPos)
	}
	if !reflect.DeepEqual(xTok, []int{4, 7, 9, 0, 0, 0}) {
		t.Fatalf("xTok = %v", xTok)
	}
	if !reflect.DeepEqual(yTok, []int{7, 9, 9, 0, 0, 0}) {
		t.Fatalf("yTok = %v", yTok)
	}
}

func TestGenerationSeedDerivationPreservesFirstSampleAndPrefix(t *testing.T) {
	const base int64 = 77
	if got := generationSeedForSample(base, 0); got != base {
		t.Fatalf("sample zero seed=%d want=%d", got, base)
	}
	seen := map[int64]bool{}
	for i := 0; i < 32; i++ {
		seed := generationSeedForSample(base, i)
		if seen[seed] {
			t.Fatalf("duplicate derived seed %d at sample %d", seed, i)
		}
		seen[seed] = true
	}
	first := generatedSampleLines(t, generationPlan{numSamples: 3, baseSeed: base, eosTokenID: -1})
	second := generatedSampleLines(t, generationPlan{numSamples: 8, baseSeed: base, eosTokenID: -1})
	if !reflect.DeepEqual(first, second[:len(first)]) {
		t.Fatalf("sample prefix changed with N: first=%v second=%v", first, second)
	}
}

func TestBuildGenerationPlanDefaultsEOSAndValidation(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 9, Training: TrainingSpec{Seed: 42}}
	base := GenerateOptions{Temperature: 1}
	plan, err := buildGenerationPlan(base, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.numSamples != 1 || plan.batchSize != 1 || plan.baseSeed != 42 || plan.eosTokenID != -1 || !plan.legacyOutput {
		t.Fatalf("default plan=%+v", plan)
	}
	vocab := trainTestDNAVocabulary()
	plan, err = buildGenerationPlan(base, cfg, vocab)
	if err != nil {
		t.Fatal(err)
	}
	if plan.eosTokenID != 2 {
		t.Fatalf("sequence vocabulary EOS=%d want=2", plan.eosTokenID)
	}
	explicit := 3
	if _, err := buildGenerationPlan(GenerateOptions{Temperature: 1, EOSTokenID: &explicit}, cfg, vocab); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("mismatched vocabulary EOS error=%v", err)
	}
	for name, opts := range map[string]GenerateOptions{
		"negative samples": {Temperature: 1, NumSamples: -1},
		"negative batch":   {Temperature: 1, GenerationBatch: -1},
		"negative max":     {Temperature: 1, MaxTokens: -1},
		"zero temperature": {},
		"negative top-k":   {Temperature: 1, TopK: -1},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := buildGenerationPlan(opts, cfg, nil); err == nil {
				t.Fatal("invalid generation options were accepted")
			}
		})
	}
}

func TestBatchedGenerationMatchesBatchOneAndPreservesOrder(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 7, SeqLen: 6}
	opts := GenerateOptions{MaxTokens: 4, Temperature: 0.9, TopK: 4, Prompt: "token_ids:1"}
	run := func(batchSize int) string {
		t.Helper()
		plan := generationPlan{numSamples: 9, batchSize: batchSize, baseSeed: 77, eosTokenID: -1}
		var out bytes.Buffer
		if err := runBatchedGenerationSamples(cfg, deterministicBatchedGenerationEvaluator{vocabSize: cfg.VocabSize}, opts, plan, nil, &out); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	if one, many := run(1), run(4); one != many {
		t.Fatalf("batched output differs from batch one:\n%s\n---\n%s", one, many)
	}
}

func TestBatchedGenerationStopsRowsIndependentlyAndPadsFinalWave(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 6, SeqLen: 6}
	evaluator := &scriptedBatchedGenerationEvaluator{
		vocabSize: cfg.VocabSize,
		tokens: [][]int{
			{2, 3, 4},
			{5, 2, 5},
			{5, 5, 2},
			{2, 4, 4},
		},
	}
	plan := generationPlan{numSamples: 4, batchSize: 3, baseSeed: 5, eosTokenID: 2}
	var out bytes.Buffer
	err := runBatchedGenerationSamples(cfg, evaluator, GenerateOptions{
		MaxTokens: 4, Temperature: 1, TopK: 1, Prompt: "token_ids:1",
	}, plan, nil, &out)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := out.String(), "1,2\n1,3,2\n1,4,5,2\n1,2\n"; got != want {
		t.Fatalf("batched EOS output=%q want=%q", got, want)
	}
	if evaluator.calls != 4 {
		t.Fatalf("evaluator calls=%d want=4", evaluator.calls)
	}
	if !evaluator.finalWavePadded {
		t.Fatal("final partial wave did not duplicate its active row into padding rows")
	}
}

func TestGenerateReplaySampleStopsAtEOS(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 5, SeqLen: 8}
	evaluator := &fakeCausalGenerationEvaluator{vocabSize: cfg.VocabSize, tokens: []int{2, 4, 4}}
	sample, err := generateReplaySample(cfg, evaluator, GenerateOptions{
		MaxTokens: 5, Temperature: 1, Prompt: "token_ids:1",
	}, 2, rand.New(rand.NewSource(1)), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sample.tokens, []int{1, 2}) {
		t.Fatalf("EOS sample=%v", sample.tokens)
	}
	if evaluator.calls != 1 {
		t.Fatalf("evaluator calls=%d want=1", evaluator.calls)
	}
}

func TestRunGenerationSamplesLegacyAndMachineOutput(t *testing.T) {
	legacy := generationPlan{numSamples: 1, baseSeed: 11, eosTokenID: -1, legacyOutput: true}
	var legacyOut bytes.Buffer
	err := runGenerationSamples(legacy, nil, &legacyOut, func(_ int, _ *rand.Rand) (generationSample, error) {
		return generationSample{tokens: []int{1, 2, 3}, notice: "stopped at seq_len limit (3 tokens)"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	wantLegacy := "stopped at seq_len limit (3 tokens)\ngenerated token_ids:1,2,3\n"
	if legacyOut.String() != wantLegacy {
		t.Fatalf("legacy output=%q want=%q", legacyOut.String(), wantLegacy)
	}

	machine := generationPlan{numSamples: 3, baseSeed: 11, eosTokenID: -1}
	var machineOut bytes.Buffer
	err = runGenerationSamples(machine, nil, &machineOut, func(i int, _ *rand.Rand) (generationSample, error) {
		return generationSample{tokens: []int{1, i + 2}, notice: "must not leak"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := machineOut.String(), "1,2\n1,3\n1,4\n"; got != want {
		t.Fatalf("machine output=%q want=%q", got, want)
	}
}

func TestGenerationOutputFileStreamsDecodedRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "samples.txt")
	output, err := openGenerationOutput(path)
	if err != nil {
		t.Fatal(err)
	}
	plan := generationPlan{numSamples: 2, baseSeed: 9, eosTokenID: 2}
	err = runGenerationSamples(plan, trainTestDNAVocabulary(), output.writer, func(i int, _ *rand.Rand) (generationSample, error) {
		if i == 0 {
			return generationSample{tokens: []int{1, 4, 5, 2}}, nil
		}
		return generationSample{tokens: []int{1, 6, 7, 2}}, nil
	})
	if closeErr := output.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(body), "AC\nGT\n"; got != want {
		t.Fatalf("decoded output=%q want=%q", got, want)
	}
}

func generatedSampleLines(t *testing.T, plan generationPlan) []string {
	t.Helper()
	var out bytes.Buffer
	err := runGenerationSamples(plan, nil, &out, func(_ int, rng *rand.Rand) (generationSample, error) {
		return generationSample{tokens: []int{rng.Intn(1_000_000)}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(out.String()), "\n")
}

type fakeCausalGenerationEvaluator struct {
	vocabSize int
	tokens    []int
	calls     int
}

func (f *fakeCausalGenerationEvaluator) EvaluateGPU(_ []int, _ []int, _, _ int) (float32, error) {
	f.calls++
	return 0, nil
}

func (f *fakeCausalGenerationEvaluator) ReadOutput(_ string, shape []int) ([]float32, error) {
	logits := make([]float32, shape[0]*shape[1])
	for i := range logits {
		logits[i] = -100
	}
	token := f.tokens[f.calls-1]
	for row := 0; row < shape[0]; row++ {
		logits[row*f.vocabSize+token] = 100
	}
	return logits, nil
}

type deterministicBatchedGenerationEvaluator struct{ vocabSize int }

func (f deterministicBatchedGenerationEvaluator) EvaluateGenerationGPU(xTok, _ []int, positions []int, batchSize, seqLen int) ([]float32, error) {
	logits := make([]float32, batchSize*f.vocabSize)
	for row := 0; row < batchSize; row++ {
		last := xTok[row*seqLen+positions[row]]
		for token := 0; token < f.vocabSize; token++ {
			logits[row*f.vocabSize+token] = float32(((last+1)*(token+3))%11) / 5
		}
	}
	return logits, nil
}

type scriptedBatchedGenerationEvaluator struct {
	vocabSize       int
	tokens          [][]int
	calls           int
	finalWavePadded bool
}

func (f *scriptedBatchedGenerationEvaluator) EvaluateGenerationGPU(xTok, _ []int, _ []int, batchSize, seqLen int) ([]float32, error) {
	if f.calls >= len(f.tokens) {
		return nil, fmt.Errorf("unexpected evaluator call %d", f.calls)
	}
	if f.calls == len(f.tokens)-1 && batchSize >= 3 {
		f.finalWavePadded = reflect.DeepEqual(xTok[:seqLen], xTok[seqLen:2*seqLen]) && reflect.DeepEqual(xTok[:seqLen], xTok[2*seqLen:3*seqLen])
	}
	logits := make([]float32, batchSize*f.vocabSize)
	for i := range logits {
		logits[i] = -100
	}
	for row, token := range f.tokens[f.calls] {
		logits[row*f.vocabSize+token] = 100
	}
	f.calls++
	return logits, nil
}
