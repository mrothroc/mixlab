package arch

import (
	"strings"
	"testing"
)

// Token IDs are int32 in the IR and on the GPU, so a larger vocabulary cannot
// be represented and would wrap token IDs silently.
func TestVocabSizeMustFitInt32(t *testing.T) {
	const tooLarge = `{"model_dim": 16, "vocab_size": 2147483649, "seq_len": 4, "blocks": [{"type": "plain", "heads": 2}], "training": {"batch_tokens": 4}}`
	if _, err := ParseArchConfig([]byte(tooLarge), "too-large"); err == nil || !strings.Contains(err.Error(), "vocab_size") {
		t.Fatalf("vocab_size 2^31+1 accepted or wrong error: %v", err)
	}
	const largest = `{"model_dim": 16, "vocab_size": 2147483647, "seq_len": 4, "blocks": [{"type": "plain", "heads": 2}], "training": {"batch_tokens": 4}}`
	if _, err := ParseArchConfig([]byte(largest), "largest"); err != nil {
		t.Fatalf("vocab_size 2^31-1 rejected: %v", err)
	}
}
