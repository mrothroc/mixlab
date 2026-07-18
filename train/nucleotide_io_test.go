package train

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func trainTestDNAVocabulary() *data.NucleotideVocabulary {
	return &data.NucleotideVocabulary{
		Format: data.NucleotideVocabularyFormat, Version: data.NucleotideVocabularyVersion,
		Alphabet: data.NucleotideAlphabetDNA, InvalidSymbolPolicy: "error", AmbiguousSymbols: []string{"N"},
		Tokens:      map[string]int{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3, "A": 4, "C": 5, "G": 6, "T": 7, "N": 8},
		Complements: map[string]string{"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"},
	}
}

func TestGenerationPromptNucleotideSequence(t *testing.T) {
	got, err := generationPromptTokensWithVocabulary("sequence:ACGTN", 9, rand.New(rand.NewSource(1)), trainTestDNAVocabulary())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int{1, 4, 5, 6, 7, 8}) {
		t.Fatalf("prompt ids=%v", got)
	}
}

func TestScoreEBMSequenceStringFieldsEncode(t *testing.T) {
	vocab := trainTestDNAVocabulary()
	single, err := encodeScoreEBMSequenceFields(scoreEBMInputRecord{ID: "s", Sequence: "acgt"}, vocab)
	if err != nil {
		t.Fatal(err)
	}
	if single.Sequence != "ACGT" || !reflect.DeepEqual(single.Tokens, []int{4, 5, 6, 7}) {
		t.Fatalf("single=%+v", single)
	}
	pair, err := encodeScoreEBMSequenceFields(scoreEBMInputRecord{ID: "p", CleanSequence: "AC", CorruptSequence: "GT"}, vocab)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pair.Clean, []int{4, 5}) || !reflect.DeepEqual(pair.Corrupt, []int{6, 7}) {
		t.Fatalf("pair=%+v", pair)
	}
	if _, err := encodeScoreEBMSequenceFields(scoreEBMInputRecord{ID: "bad", Sequence: "AC"}, nil); err == nil {
		t.Fatal("sequence fields accepted without vocabulary")
	}
}
