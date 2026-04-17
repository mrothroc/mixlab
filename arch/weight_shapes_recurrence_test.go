package arch

import "testing"

func TestCollectWeightShapesWithRecurrence_OmitsCopyBlockWeights(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	recurrence := []int{0, 1, 2, 3, 2, 3, 2, 3}

	metas, err := CollectWeightShapesWithBigramAndRecurrence(64, 256, 32, DefaultFFNMultiplier, false, false, false, false, 0, 0, blocks, recurrence)
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigramAndRecurrence: %v", err)
	}
	if len(metas) != 25 {
		t.Fatalf("weight shapes len=%d want 25", len(metas))
	}

	untied, err := CollectWeightShapes(64, 256, 32, DefaultFFNMultiplier, false, false, false, false, blocks)
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	if len(untied) != 47 {
		t.Fatalf("untied weight shapes len=%d want 47", len(untied))
	}
}
