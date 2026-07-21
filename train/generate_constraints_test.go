package train

import (
	"bytes"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

type fixedTokenProcessor struct {
	token      int
	prompt     []int
	accepted   []int
	finishFail bool
}

func (p *fixedTokenProcessor) Start(prompt []int) error {
	p.prompt = append([]int(nil), prompt...)
	return nil
}

func (p *fixedTokenProcessor) Mask(_ int, logits []float32) error {
	for token := range logits {
		if token != p.token {
			logits[token] = float32(math.Inf(-1))
		}
	}
	return nil
}

func (p *fixedTokenProcessor) Accept(token int) error {
	p.accepted = append(p.accepted, token)
	return nil
}

func (p *fixedTokenProcessor) Finish() error {
	if p.finishFail {
		return errNoFiniteLogits
	}
	return nil
}

func TestConstrainedSamplingMasksBeforeTemperatureAndTopK(t *testing.T) {
	processor := &fixedTokenProcessor{token: 2}
	if err := processor.Start([]int{0}); err != nil {
		t.Fatal(err)
	}
	logits := []float32{100, 90, -100, 80}
	next, err := constrainAndSampleNextToken(processor, 0, logits, 0.2, 1, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if next != 2 || !reflect.DeepEqual(processor.accepted, []int{2}) {
		t.Fatalf("next=%d accepted=%v, want token 2", next, processor.accepted)
	}
}

func TestSampleNextTokenRejectsAllMaskedAndNonFinite(t *testing.T) {
	if _, err := sampleNextToken([]float32{float32(math.Inf(-1)), float32(math.Inf(-1))}, 1, 0, rand.New(rand.NewSource(1))); err == nil || !strings.Contains(err.Error(), "no finite logits") {
		t.Fatalf("all-masked error=%v", err)
	}
	if _, err := sampleNextToken([]float32{0, float32(math.NaN())}, 1, 0, rand.New(rand.NewSource(1))); err == nil || !strings.Contains(err.Error(), "token 1") {
		t.Fatalf("NaN error=%v", err)
	}
	processor := &fixedTokenProcessor{token: 3}
	if err := processor.Start(nil); err != nil {
		t.Fatal(err)
	}
	if _, err := constrainAndSampleNextToken(processor, 7, []float32{0, 1, 2}, 1, 0, rand.New(rand.NewSource(1))); err == nil || !strings.Contains(err.Error(), "grammar permits no continuation at step 7") {
		t.Fatalf("constrained dead-end error=%v", err)
	}
}

func TestNilProcessorSamplingParity(t *testing.T) {
	logits := []float32{1, -2, 0.5, 4, 3}
	for seed := int64(0); seed < 50; seed++ {
		want, err := sampleNextToken(append([]float32(nil), logits...), 0.7, 4, rand.New(rand.NewSource(seed)))
		if err != nil {
			t.Fatal(err)
		}
		got, err := constrainAndSampleNextToken(nil, 0, append([]float32(nil), logits...), 0.7, 4, rand.New(rand.NewSource(seed)))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("seed=%d got=%d want=%d", seed, got, want)
		}
	}
}

func TestBatchedGenerationUsesIndependentProcessors(t *testing.T) {
	created := 0
	factory := func() LogitProcessor {
		created++
		return &fixedTokenProcessor{token: created}
	}
	cfg := &ArchConfig{VocabSize: 4, SeqLen: 4}
	opts := GenerateOptions{MaxTokens: 1, Temperature: 1, TopK: 1, Prompt: "token_ids:0", NewLogitProcessor: factory}
	plan := generationPlan{numSamples: 2, batchSize: 2, baseSeed: 1, eosTokenID: -1}
	var output bytes.Buffer
	if err := runBatchedGenerationSamples(cfg, deterministicBatchedGenerationEvaluator{vocabSize: cfg.VocabSize}, opts, plan, nil, &output); err != nil {
		t.Fatal(err)
	}
	if created != 2 {
		t.Fatalf("processor instances=%d want=2", created)
	}
	if got, want := output.String(), "0,1\n0,2\n"; got != want {
		t.Fatalf("output=%q want=%q", got, want)
	}
}

func TestGenerationRejectsIncompleteGrammarAtLimit(t *testing.T) {
	cfg := &ArchConfig{VocabSize: 4, SeqLen: 4}
	evaluator := &fakeCausalGenerationEvaluator{vocabSize: cfg.VocabSize, tokens: []int{1}}
	processor := &fixedTokenProcessor{token: 1, finishFail: true}
	_, err := generateReplaySample(cfg, evaluator, GenerateOptions{
		MaxTokens: 1, Temperature: 1, Prompt: "token_ids:0", NewLogitProcessor: func() LogitProcessor { return processor },
	}, -1, rand.New(rand.NewSource(1)), nil)
	if err == nil || !strings.Contains(err.Error(), "does not complete the grammar") {
		t.Fatalf("incomplete grammar error=%v", err)
	}
}
