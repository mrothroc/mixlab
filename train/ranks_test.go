package train

import (
	"math"
	"testing"
)

func TestTargetRankFromLogitsBasic(t *testing.T) {
	// Three vocab entries; target is the argmax => rank 0.
	logits := []float32{0.1, 2.0, 0.5}
	rank, err := targetRankFromLogits(logits, 3, 1)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 0 {
		t.Fatalf("rank = %d, want 0 (target is argmax)", rank)
	}
}

func TestTargetRankFromLogitsLastPlace(t *testing.T) {
	// Target has the lowest logit => rank vocab-1.
	logits := []float32{2.0, 0.5, 0.1}
	rank, err := targetRankFromLogits(logits, 3, 2)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 2 {
		t.Fatalf("rank = %d, want 2 (target is last)", rank)
	}
}

func TestTargetRankFromLogitsMiddle(t *testing.T) {
	// logits[0]=3.0 wins. Target id=2 (logit=2.0) has one entry above it.
	logits := []float32{3.0, 0.5, 2.0, 1.0}
	rank, err := targetRankFromLogits(logits, 4, 2)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 1 {
		t.Fatalf("rank = %d, want 1", rank)
	}
}

func TestTargetRankFromLogitsTieBreakAscendingID(t *testing.T) {
	// Three entries tied at the top. Tie-break: lower token ID ranks higher.
	// Target=2 has two equal-logit entries with smaller IDs (0 and 1), so rank=2.
	logits := []float32{1.0, 1.0, 1.0, 0.5}
	rank, err := targetRankFromLogits(logits, 4, 2)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 2 {
		t.Fatalf("rank = %d, want 2 (tie-broken by ascending token ID)", rank)
	}

	// Target=0 with the same logits ranks first under tie-break: rank 0.
	rank, err = targetRankFromLogits(logits, 4, 0)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 0 {
		t.Fatalf("rank = %d, want 0 (target=0 wins tie)", rank)
	}
}

func TestTargetRankFromLogitsTieMixedWithGreater(t *testing.T) {
	// One entry strictly greater, plus a tie with the target on a smaller ID.
	// vocab IDs:    0    1    2    3
	logits := []float32{2.5, 1.0, 1.0, 0.5}
	// target=2: id=0 strictly greater (count 1) + id=1 equal-and-smaller (count 1) => rank 2
	rank, err := targetRankFromLogits(logits, 4, 2)
	if err != nil {
		t.Fatalf("targetRankFromLogits: %v", err)
	}
	if rank != 2 {
		t.Fatalf("rank = %d, want 2", rank)
	}
}

func TestTargetRankFromLogitsTargetOutOfRange(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	if _, err := targetRankFromLogits(logits, 3, 3); err == nil {
		t.Fatal("targetRankFromLogits(target=vocab) succeeded, want error")
	}
}

func TestTargetRankFromLogitsShortRow(t *testing.T) {
	logits := []float32{1.0, 2.0}
	if _, err := targetRankFromLogits(logits, 3, 0); err == nil {
		t.Fatal("targetRankFromLogits with short row succeeded, want error")
	}
}

func TestTargetNLLFromLogitsKnownRow(t *testing.T) {
	// Three-entry softmax: logits=[1, 2, 3]
	// log_sum_exp = log(e + e^2 + e^3) ≈ 3.40760596
	// For target=2 (logit=3): nll = log_sum_exp - 3 ≈ 0.40760596
	// For target=0 (logit=1): nll = log_sum_exp - 1 ≈ 2.40760596
	logits := []float32{1.0, 2.0, 3.0}
	cases := []struct {
		target uint16
		want   float64
	}{
		{target: 2, want: 0.40760596},
		{target: 1, want: 1.40760596},
		{target: 0, want: 2.40760596},
	}
	for _, tc := range cases {
		nll, err := targetNLLFromLogits(logits, 3, tc.target)
		if err != nil {
			t.Fatalf("targetNLLFromLogits(target=%d): %v", tc.target, err)
		}
		if math.Abs(float64(nll)-tc.want) > 1e-5 {
			t.Fatalf("targetNLLFromLogits(target=%d) = %g, want %g", tc.target, nll, tc.want)
		}
	}
}

func TestTargetNLLFromLogitsLargeShifted(t *testing.T) {
	// Large logits must not overflow because of the log-sum-exp max-shift.
	logits := []float32{1000.0, 1001.0, 1002.0}
	nll, err := targetNLLFromLogits(logits, 3, 2)
	if err != nil {
		t.Fatalf("targetNLLFromLogits: %v", err)
	}
	// Same shape as [0,1,2] → expected ≈ 0.40760596
	if math.Abs(float64(nll)-0.40760596) > 1e-4 {
		t.Fatalf("nll = %g, want ≈ 0.40760596", nll)
	}
}

func TestTargetRankFromLogitsVocabExceedsUint16(t *testing.T) {
	// A rank value that would not fit in uint16 must surface as an error.
	vocab := 1 << 17
	logits := make([]float32, vocab)
	for i := range logits {
		// Higher token IDs have higher logits; target=0 has the lowest, so rank = vocab-1.
		logits[i] = float32(i)
	}
	if _, err := targetRankFromLogits(logits, vocab, 0); err == nil {
		t.Fatal("expected error for rank exceeding uint16 range")
	}
}
