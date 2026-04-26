package train

import (
	"reflect"
	"testing"
)

func TestComputeTrigramIDs_SequenceBoundaries(t *testing.T) {
	tokens := []int32{10, 20, 30, 40, 50, 60}
	got, err := ComputeTrigramIDs(tokens, 2, 3, 97)
	if err != nil {
		t.Fatalf("ComputeTrigramIDs: %v", err)
	}
	want := []int32{0, 0, 61, 0, 0, 55}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ComputeTrigramIDs=%v want %v", got, want)
	}
}

func TestComputeTrigramIDs_Disabled(t *testing.T) {
	got, err := ComputeTrigramIDs([]int32{1, 2, 3}, 1, 3, 0)
	if err != nil {
		t.Fatalf("ComputeTrigramIDs: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil trigram ids when disabled, got %v", got)
	}
}
