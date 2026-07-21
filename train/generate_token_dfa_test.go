package train

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
)

func balancedTokenDFATable() tokenDFATableJSON {
	return tokenDFATableJSON{
		Format: tokenDFAFormat, Version: tokenDFAVersion, VocabSize: 5, StartState: "start", EOSTokenIDs: []int{4},
		States: []tokenDFAStateJSON{
			{Name: "start", Transitions: map[string]string{"0": "body"}},
			{Name: "body", Transitions: map[string]string{"1": "open", "3": "body", "4": "done"}},
			{Name: "open", Transitions: map[string]string{"2": "body", "3": "open"}},
			{Name: "done", Accept: true},
		},
	}
}

func TestTokenDFAConsumesPromptMasksAndRequiresAcceptingFinish(t *testing.T) {
	dfa, err := compileTokenDFA(balancedTokenDFATable(), 5, 4)
	if err != nil {
		t.Fatal(err)
	}
	processor := &tokenDFAProcessor{dfa: dfa}
	if err := processor.Start([]int{0}); err != nil {
		t.Fatal(err)
	}
	logits := []float32{1, 1, 1, 1, 1}
	if err := processor.Mask(0, logits); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []int{0, 2} {
		if !math.IsInf(float64(logits[forbidden]), -1) {
			t.Fatalf("token %d was not masked: %v", forbidden, logits)
		}
	}
	if err := processor.Accept(1); err != nil {
		t.Fatal(err)
	}
	if err := processor.Finish(); err == nil || !strings.Contains(err.Error(), "not accepting") || !errors.Is(err, ErrGrammarIncomplete) {
		t.Fatalf("open grammar finish error=%v", err)
	}
	if err := processor.Accept(2); err != nil {
		t.Fatal(err)
	}
	if err := processor.Accept(4); err != nil {
		t.Fatal(err)
	}
	if err := processor.Finish(); err != nil {
		t.Fatal(err)
	}
}

func TestTokenDFATableValidation(t *testing.T) {
	base := balancedTokenDFATable()
	badVocab := base
	badVocab.VocabSize = 6
	if _, err := compileTokenDFA(badVocab, 5, 4); err == nil || !strings.Contains(err.Error(), "model vocab_size") {
		t.Fatalf("vocab mismatch error=%v", err)
	}
	badEOS := base
	badEOS.States = append([]tokenDFAStateJSON(nil), base.States...)
	badEOS.States[1].Transitions = map[string]string{"4": "body"}
	if _, err := compileTokenDFA(badEOS, 5, 4); err == nil || !strings.Contains(err.Error(), "accepting state") {
		t.Fatalf("EOS target error=%v", err)
	}
	if _, err := compileTokenDFA(base, 5, -1); err == nil || !strings.Contains(err.Error(), "requires -eos-token-id") {
		t.Fatalf("missing EOS error=%v", err)
	}
}

func TestTokenDFABalancedParenthesesAcrossSeeds(t *testing.T) {
	dfa, err := compileTokenDFA(balancedTokenDFATable(), 5, 4)
	if err != nil {
		t.Fatal(err)
	}
	for seed := int64(0); seed < 1_000; seed++ {
		processor := &tokenDFAProcessor{dfa: dfa}
		if err := processor.Start([]int{0}); err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(seed))
		depth := 0
		terminated := false
		for step := 0; step < 64; step++ {
			logits := []float32{0, 1, 3, -1, 4}
			next, err := constrainAndSampleNextToken(processor, step, logits, 1, 0, rng)
			if err != nil {
				t.Fatalf("seed=%d step=%d: %v", seed, step, err)
			}
			switch next {
			case 1:
				depth++
			case 2:
				depth--
			case 4:
				terminated = true
			}
			if depth < 0 {
				t.Fatalf("seed=%d produced negative depth", seed)
			}
			if terminated {
				break
			}
		}
		if !terminated || depth != 0 {
			t.Fatalf("seed=%d terminated=%t depth=%d", seed, terminated, depth)
		}
		if err := processor.Finish(); err != nil {
			t.Fatalf("seed=%d finish: %v", seed, err)
		}
	}
}
