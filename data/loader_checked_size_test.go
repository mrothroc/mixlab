package data

import (
	"math"
	"strings"
	"testing"
)

// An absurd batch shape must fail before allocating, not wrap to a short buffer.
func TestRecordBatchSizesRejectOverflow(t *testing.T) {
	codebooks := &Loader{codebooks: &codebookSequenceStream{seqLen: 1, numCodebooks: 2}}
	if _, err := codebooks.nextCodebookBatch(math.MaxInt, 1); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("codebook batch error = %v, want overflow", err)
	}
	continuous := &Loader{continuous: &continuousSequenceStream{seqLen: 1, featureDim: 2}}
	if _, err := continuous.nextContinuousBatch(math.MaxInt, 1); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("continuous batch error = %v, want overflow", err)
	}
}
