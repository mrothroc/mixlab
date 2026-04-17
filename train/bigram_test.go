package train

import (
	"reflect"
	"testing"
)

func TestComputeBigramIDs_KnownReference(t *testing.T) {
	tokens := []int32{10, 20, 30, 40, 50}
	got, err := ComputeBigramIDs(tokens, len(tokens), 97)
	if err != nil {
		t.Fatalf("ComputeBigramIDs: %v", err)
	}
	want := []int32{96, 50, 2, 58, 26}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ComputeBigramIDs=%v want %v", got, want)
	}
}

func TestComputeBigramIDs_Disabled(t *testing.T) {
	got, err := ComputeBigramIDs([]int32{1, 2, 3}, 3, 0)
	if err != nil {
		t.Fatalf("ComputeBigramIDs: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil bigram ids when disabled, got %v", got)
	}
}
