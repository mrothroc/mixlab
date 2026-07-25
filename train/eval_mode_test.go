package train

import (
	"strings"
	"testing"
)

func TestEvalModeRejectsNegativeClassificationBatchCapBeforeGPUSetup(t *testing.T) {
	err := runEvalModeWithOptions("", "", "", EvalModeOptions{ValBatches: -1})
	if err == nil || !strings.Contains(err.Error(), "-val-batches") {
		t.Fatalf("error=%v", err)
	}
}
