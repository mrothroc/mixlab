package train

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestMLMWordBoundaryLUTSupportedTokenizers(t *testing.T) {
	tests := []struct {
		name   string
		json   string
		scheme string
		starts []uint8
	}{
		{
			name:   "metaspace bpe",
			json:   `{"added_tokens":[{"id":3,"content":"[MASK]","special":true}],"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"BPE","vocab":{"▁play":0,"ing":1,"▁ball":2,"[MASK]":3,"tail":4}}}`,
			scheme: "sentencepiece", starts: []uint8{1, 0, 1, 0, 0},
		},
		{
			name:   "metaspace unigram",
			json:   `{"added_tokens":[{"id":3,"content":"[MASK]","special":true}],"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"Unigram","vocab":[["▁play",0],["ing",0],["▁ball",0],["[MASK]",0],["tail",0]]}}`,
			scheme: "sentencepiece", starts: []uint8{1, 0, 1, 0, 0},
		},
		{
			name:   "bytelevel",
			json:   `{"added_tokens":[{"id":3,"content":"[MASK]","special":true}],"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"ByteLevel","add_prefix_space":true}]},"model":{"type":"BPE","vocab":{"Ġplay":0,"ing":1,"Ġball":2,"[MASK]":3,"tail":4}}}`,
			scheme: "bytelevel", starts: []uint8{1, 0, 1, 0, 0},
		},
		{
			name:   "wordpiece",
			json:   `{"added_tokens":[{"id":3,"content":"[MASK]","special":true}],"pre_tokenizer":{"type":"Whitespace"},"model":{"type":"WordPiece","continuing_subword_prefix":"##","vocab":{"play":0,"##ing":1,"ball":2,"[MASK]":3,"##tail":4}}}`,
			scheme: "wordpiece", starts: []uint8{1, 0, 1, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTokenizerFixture(t, tt.json)
			starts, eligible, scheme, err := mlmWordBoundaryLUTFromTokenizer(path, 5, 3)
			if err != nil {
				t.Fatalf("mlmWordBoundaryLUTFromTokenizer: %v", err)
			}
			if scheme != tt.scheme {
				t.Fatalf("scheme=%q, want %q", scheme, tt.scheme)
			}
			if !reflect.DeepEqual(starts, tt.starts) {
				t.Fatalf("starts=%v, want %v", starts, tt.starts)
			}
			if want := []uint8{1, 1, 1, 0, 1}; !reflect.DeepEqual(eligible, want) {
				t.Fatalf("eligible=%v, want %v", eligible, want)
			}
		})
	}
}

func TestMLMWordBoundaryLUTRejectsUnsafeMetadata(t *testing.T) {
	tests := []struct {
		name  string
		json  string
		want  string
		vocab int
	}{
		{name: "bytelevel missing prefix", json: `{"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false},"model":{"type":"BPE","vocab":{"Ġa":0,"b":1}}}`, want: "add_prefix_space"},
		{name: "metaspace missing prefix", json: `{"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"never"},"model":{"type":"BPE","vocab":{"▁a":0,"b":1}}}`, want: "prepend_scheme"},
		{name: "out of range id", json: `{"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"BPE","vocab":{"▁a":0,"b":2}}}`, want: "invalid id"},
		{name: "missing id", json: `{"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"BPE","vocab":{"▁a":0,"b":2}}}`, want: "missing token id", vocab: 3},
		{name: "duplicate id", json: `{"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"BPE","vocab":{"▁a":0,"b":0}}}`, want: "duplicate token id"},
		{name: "unsupported", json: `{"pre_tokenizer":{"type":"Whitespace"},"model":{"type":"BPE","vocab":{"a":0,"b":1}}}`, want: "no supported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vocab := tt.vocab
			if vocab == 0 {
				vocab = 2
			}
			_, _, _, err := mlmWordBoundaryLUTFromTokenizer(writeTokenizerFixture(t, tt.json), vocab, 1)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestConfigureMLMWordBoundariesForTraining(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "train_00000.bin"), []byte("fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := objectiveTestConfig()
	cfg.VocabSize = 5
	cfg.Training.MLMMaskTokenID = 3
	cfg.Training.MLMMaskUnit = "whole_word"
	if _, err := configureMLMWordBoundariesForTraining(cfg, filepath.Join(dir, "train_*.bin")); err == nil || !strings.Contains(err.Error(), "requires tokenizer.json") {
		t.Fatalf("missing tokenizer error=%v", err)
	}
	json := `{"added_tokens":[{"id":3,"content":"[MASK]","special":true}],"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"},"model":{"type":"BPE","vocab":{"▁play":0,"ing":1,"▁ball":2,"[MASK]":3,"tail":4}}}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(json), 0o644); err != nil {
		t.Fatal(err)
	}
	source, err := configureMLMWordBoundariesForTraining(cfg, filepath.Join(dir, "train_*.bin"))
	if err != nil {
		t.Fatalf("configureMLMWordBoundariesForTraining: %v", err)
	}
	if !strings.Contains(source, "scheme=sentencepiece") || len(cfg.Training.MLMWordStart) != 5 || len(cfg.Training.MLMMaskEligible) != 5 {
		t.Fatalf("source=%q starts=%v eligible=%v", source, cfg.Training.MLMWordStart, cfg.Training.MLMMaskEligible)
	}
}

func writeTokenizerFixture(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
