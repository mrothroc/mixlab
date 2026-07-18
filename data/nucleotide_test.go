package data

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func testDNAVocabulary() NucleotideVocabulary {
	return NucleotideVocabulary{
		Format: NucleotideVocabularyFormat, Version: NucleotideVocabularyVersion,
		Alphabet: NucleotideAlphabetDNA, InvalidSymbolPolicy: "error",
		AmbiguousSymbols: []string{"N"},
		Tokens:           map[string]int{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3, "A": 4, "C": 5, "G": 6, "T": 7, "N": 8},
		Complements:      map[string]string{"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"},
	}
}

func TestNucleotideVocabularyEncodeDecodeAndComplement(t *testing.T) {
	vocab := testDNAVocabulary()
	ids, err := vocab.Encode("acgtn")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !reflect.DeepEqual(ids, []int{4, 5, 6, 7, 8}) {
		t.Fatalf("ids=%v", ids)
	}
	decoded, err := vocab.Decode([]int{1, 4, 5, 6, 7, 8, 2, 4})
	if err != nil || decoded != "ACGTN" {
		t.Fatalf("Decode=%q err=%v", decoded, err)
	}
	complement, err := vocab.ComplementTokenIDs()
	if err != nil {
		t.Fatalf("ComplementTokenIDs: %v", err)
	}
	if complement[4] != 7 || complement[5] != 6 || complement[8] != 8 {
		t.Fatalf("complement=%v", complement)
	}
}

func TestLoadNucleotideVocabularyRejectsUnknownAndTrailingJSON(t *testing.T) {
	dir := t.TempDir()
	for _, body := range []string{
		`{"format":"mixlab.nucleotide_vocabulary","version":1,"alphabet":"dna","invalid_symbol_policy":"error","tokens":{},"unknown":1}`,
		`{"format":"mixlab.nucleotide_vocabulary","version":1,"alphabet":"dna","invalid_symbol_policy":"error","tokens":{}} {}`,
	} {
		path := filepath.Join(dir, "vocab.json")
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadNucleotideVocabulary(path); err == nil {
			t.Fatalf("LoadNucleotideVocabulary accepted %q", body)
		}
	}
}

func TestNucleotideVocabularyRejectsContractDrift(t *testing.T) {
	tests := map[string]func(*NucleotideVocabulary){
		"special id": func(v *NucleotideVocabulary) {
			v.Tokens["<BOS>"] = 8
			v.Tokens["N"] = 1
		},
		"invalid policy":    func(v *NucleotideVocabulary) { v.InvalidSymbolPolicy = "replace" },
		"undeclared symbol": func(v *NucleotideVocabulary) { v.Tokens["R"] = 9 },
		"bad complement":    func(v *NucleotideVocabulary) { v.Complements["A"] = "A" },
		"not complement closed": func(v *NucleotideVocabulary) {
			v.AmbiguousSymbols = []string{"R"}
			delete(v.Tokens, "N")
			v.Tokens["R"] = 8
			delete(v.Complements, "N")
			v.Complements["R"] = "Y"
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			vocab := testDNAVocabulary()
			mutate(&vocab)
			if err := vocab.Validate(); err == nil {
				t.Fatal("Validate accepted a drifted nucleotide vocabulary")
			}
		})
	}
}
