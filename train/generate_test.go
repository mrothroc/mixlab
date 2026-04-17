package train

import (
	"math/rand"
	"reflect"
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
