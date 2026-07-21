package train

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGBNFParserAcceptsRecursiveAndRepeatedGrammar(t *testing.T) {
	grammar, err := parseGBNF(`
root ::= object
object ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= "\"x\"" ws ":" ws number
number ::= "-"? [0-9]+
ws ::= [ \t\n\r]*
`)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{`{"x":12}`, `{ "x" : -7,"x":0 }`} {
		state, err := newGBNFState(grammar)
		if err != nil {
			t.Fatal(err)
		}
		state, viable, err := state.consumeBytes([]byte(input))
		if err != nil || !viable || !state.accepting() {
			t.Fatalf("input=%q viable=%t accepting=%t err=%v", input, viable, state.accepting(), err)
		}
	}
	state, _ := newGBNFState(grammar)
	state, viable, err := state.consumeBytes([]byte(`{"x":`))
	if err != nil || !viable || state.accepting() {
		t.Fatalf("incomplete input viable=%t accepting=%t err=%v", viable, state.accepting(), err)
	}
	if _, viable, err := state.consumeBytes([]byte(`x}`)); err != nil || viable {
		t.Fatalf("invalid suffix viable=%t err=%v", viable, err)
	}
}

func TestGBNFTokenizerCandidatesSupportPartialBPEPieces(t *testing.T) {
	dir := t.TempDir()
	tokenizerPath := filepath.Join(dir, "tokenizer.json")
	tokenizer := `{
  "added_tokens":[{"id":0,"content":"<bos>","special":true},{"id":1,"content":"<eos>","special":true}],
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},
  "model":{"type":"BPE","vocab":{"<bos>":0,"<eos>":1,"a":2,"b":3,"ab":4},"merges":[]}
}`
	if err := os.WriteFile(tokenizerPath, []byte(tokenizer), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &ArchConfig{VocabSize: 5}
	factory, err := buildGBNFLogitProcessorFactory(`root ::= "ab"`, tokenizerPath, "", "", cfg, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	processor := factory()
	if err := processor.Start([]int{0}); err != nil {
		t.Fatal(err)
	}
	logits := []float32{0, 0, 0, 0, 0}
	if err := processor.Mask(0, logits); err != nil {
		t.Fatal(err)
	}
	for _, token := range []int{2, 4} {
		if math.IsInf(float64(logits[token]), -1) {
			t.Fatalf("viable token %d was masked: %v", token, logits)
		}
	}
	for _, token := range []int{0, 1, 3} {
		if !math.IsInf(float64(logits[token]), -1) {
			t.Fatalf("invalid token %d was not masked: %v", token, logits)
		}
	}
	if err := processor.Accept(4); err != nil {
		t.Fatal(err)
	}
	logits = []float32{0, 0, 0, 0, 0}
	if err := processor.Mask(1, logits); err != nil {
		t.Fatal(err)
	}
	for token := range logits {
		allowed := !math.IsInf(float64(logits[token]), -1)
		if allowed != (token == 1) {
			t.Fatalf("completed grammar logits=%v", logits)
		}
	}
	if err := processor.Accept(1); err != nil {
		t.Fatal(err)
	}
	if err := processor.Finish(); err != nil {
		t.Fatal(err)
	}
}

func TestGBNFValidationAndPromptModes(t *testing.T) {
	if _, err := parseGBNF(`value ::= "x"`); err == nil || !strings.Contains(err.Error(), "root") {
		t.Fatalf("missing root error=%v", err)
	}
	if _, err := parseGBNF(`root ::= [α]`); err == nil || !strings.Contains(err.Error(), "ASCII") {
		t.Fatalf("Unicode class error=%v", err)
	}
	if got := normalizeGrammarPromptMode("IGNORE"); got != grammarPromptIgnore {
		t.Fatalf("prompt mode=%q", got)
	}
}

func TestGBNFLiteralDefinitionMarkerAndRawByteEscape(t *testing.T) {
	grammar, err := parseGBNF(`root ::= "::=" "\xFF"`)
	if err != nil {
		t.Fatal(err)
	}
	state, err := newGBNFState(grammar)
	if err != nil {
		t.Fatal(err)
	}
	state, viable, err := state.consumeBytes([]byte{':', ':', '=', 0xff})
	if err != nil || !viable || !state.accepting() {
		t.Fatalf("viable=%t accepting=%t err=%v", viable, state.accepting(), err)
	}
}

// TestGBNFParserRejectsAdversarialInputWithoutCrashOrOOM locks in the bounds
// that keep untrusted grammar text from crashing (fatal stack overflow) or
// exhausting memory before a limit check fires.
func TestGBNFParserRejectsAdversarialInputWithoutCrashOrOOM(t *testing.T) {
	cases := []struct {
		name    string
		grammar string
	}{
		{"deeply nested groups", "root ::= " + strings.Repeat("(", 5000) + `"x"` + strings.Repeat(")", 5000)},
		{"many oversized repeats", "root ::= " + strings.Repeat(`"a"{0,1024} `, 300)},
		{"overflow repeat count", `root ::= "a"{99999999999999999999}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseGBNF(tc.grammar); err == nil {
				t.Fatalf("expected %s grammar to be rejected", tc.name)
			}
		})
	}
}
