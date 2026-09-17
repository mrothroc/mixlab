package train

import (
	"reflect"
	"strings"
	"testing"
)

func clsPositionConfig(t *testing.T, position string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(strings.Replace(clsContinuousConfig, `"pooling":"cls"`, `"pooling":"cls","cls_position":"`+position+`"`, 1)), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestCLSPositionBatch(t *testing.T) {
	cfg := clsPositionConfig(t, "tail")
	raw := trainBatch{labels: []int32{1, 2}, validMask: []float32{1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0}}
	b, err := prepareClassificationBatch(cfg, raw, 12, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(b.clsInsertPositions, []int32{1, 2, 3, 4, 0, 5, 6, 8, 9, 7, 10, 11, 12, 13}) || !reflect.DeepEqual(b.classificationPos, []int32{4, 9}) {
		t.Fatalf("indices=%v readout=%v", b.clsInsertPositions, b.classificationPos)
	}
	// Insertion preserves content order and never puts CLS behind padding.
	for row, index := range []int{4, 2} {
		if b.clsInsertPositions[row*7+index] != int32(row*7) {
			t.Fatal("wrong CLS index")
		}
	}
	cfg = clsPositionConfig(t, "middle")
	cfg.SeqLen = 64
	mask := make([]float32, 128)
	for i := range mask {
		mask[i] = 1
	}
	indices, positions, err := clsBatchPositions(cfg, mask, 64)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(positions, []int32{32, 97}) {
		t.Fatal(positions)
	}
	for row := 0; row < 2; row++ {
		for pos := 0; pos < 65; pos++ {
			want := pos
			if pos < 32 {
				want++
			} else if pos == 32 {
				want = 0
			}
			if indices[row*65+pos] != int32(row*65+want) {
				t.Fatal("content permuted")
			}
		}
	}
	mask[63] = 0
	if _, _, err = clsBatchPositions(cfg, mask, 64); err == nil || !strings.Contains(err.Error(), "full-length") {
		t.Fatal(err)
	}
	cfg = clsPositionConfig(t, "tail")
	for _, bad := range [][]float32{{1, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {1, .5, 0, 0, 0, 0}} {
		if _, _, err = clsBatchPositions(cfg, bad, 6); err == nil {
			t.Fatal("accepted invalid mask")
		}
	}
}
