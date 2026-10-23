package train

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// Absurd shapes must fail with an overflow error before allocating, not wrap
// to a short buffer.
func TestBatchSizesRejectOverflow(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.bin")
	if _, err := loadCharFeatures(missing, math.MaxInt, 257, 2); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("char features error = %v, want overflow before reading the file", err)
	}
	if _, _, err := clsBatchPositions(clsPositionConfig(t, "tail"), nil, math.MaxInt); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("cls positions error = %v, want overflow", err)
	}
	cfg := parseTrainMultiheadConfig(t, `"training": {
		"objective": "multihead",
		"steps": 1,
		"lr": 0.001,
		"batch_tokens": 8,
		"seed": 11,
		"mlm_mask_prob": 1.0,
		"mlm_mask_token_id": 31,
		"heads": [
			{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
			{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
			 "diffusion": {"block_size": 2, "min_mask_fraction": 1.0, "max_mask_fraction": 1.0}}
		]
	}`)
	if _, err := prepareMultiheadBatch(cfg, trainBatch{}, 0, math.MaxInt, 1); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("multihead batch error = %v, want overflow", err)
	}
}
