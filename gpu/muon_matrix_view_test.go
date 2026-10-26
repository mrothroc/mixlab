package gpu

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestMuonMatrixViewShapes(t *testing.T) {
	for _, tc := range []struct {
		view        MuonMatrixView
		shape, want []int
	}{
		{MuonMatrixIdentity, []int{2, 3}, []int{2, 3}},
		{MuonMatrixOHWI, []int{5, 2, 3, 7}, []int{5, 42}},
		{MuonMatrixIHWO, []int{7, 2, 3, 5}, []int{5, 42}},
		{MuonMatrixOHWI, []int{1, 5, 5, 27}, []int{1, 675}},
		{MuonMatrixIdentity, []int{2, 2, 2, 2}, nil},
		{MuonMatrixIHWO, []int{2, 3}, nil},
		{MuonMatrixOHWI, []int{2, 0, 3, 4}, nil},
		{MuonMatrixView(99), []int{2, 3, 4, 5}, nil},
	} {
		got, err := tc.view.Shape(tc.shape)
		if (err != nil) != (tc.want == nil) || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%v %v got %v %v", tc.view, tc.shape, got, err)
		}
	}
}

func TestMuonDefaultViewPreservesOptimizerJSON(t *testing.T) {
	b, err := json.Marshal(WeightOptimizer{GroupIndex: 2, Decay: true})
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"GroupIndex":2,"Decay":true,"Frozen":false}` {
		t.Fatalf("legacy resume hash would change: %s", b)
	}
}
