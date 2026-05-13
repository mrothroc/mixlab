package train

import (
	"fmt"
	"math"
)

// targetNLLFromLogits computes -log(softmax(logits)[target]) in fp32 using a
// max-shifted log-sum-exp, matching the GPU per-token cross-entropy reduction
// (see gpu/ir.cpp cross_entropy_per_token). Returns an error if logits is
// shorter than vocab or target is outside the vocab.
func targetNLLFromLogits(logits []float32, vocab int, target uint16) (float32, error) {
	if vocab <= 0 {
		return 0, fmt.Errorf("invalid vocab=%d", vocab)
	}
	if len(logits) < vocab {
		return 0, fmt.Errorf("logits row too short: got=%d want>=%d", len(logits), vocab)
	}
	tgt := int(target)
	if tgt < 0 || tgt >= vocab {
		return 0, fmt.Errorf("target=%d out of range for vocab=%d", tgt, vocab)
	}
	maxLogit := logits[0]
	for j := 1; j < vocab; j++ {
		if logits[j] > maxLogit {
			maxLogit = logits[j]
		}
	}
	sumExp := 0.0
	for j := range vocab {
		sumExp += math.Exp(float64(logits[j] - maxLogit))
	}
	logNorm := float64(maxLogit) + math.Log(sumExp)
	return float32(logNorm - float64(logits[tgt])), nil
}

// targetRankFromLogits returns the 0-indexed rank of target under the row of
// logits. Rank 0 means target has the highest logit; rank vocab-1 means it has
// the lowest.
//
// Ties are broken deterministically by token ID ascending: among entries that
// share a logit with the target, only those with strictly smaller token IDs
// count as ranked higher. This guarantees reproducible ranks across runs.
//
// Returns an error if logits is shorter than vocab, if target is outside the
// vocab, or if the computed rank does not fit in uint16.
func targetRankFromLogits(logits []float32, vocab int, target uint16) (uint16, error) {
	if vocab <= 0 {
		return 0, fmt.Errorf("invalid vocab=%d", vocab)
	}
	if len(logits) < vocab {
		return 0, fmt.Errorf("logits row too short: got=%d want>=%d", len(logits), vocab)
	}
	tgt := int(target)
	if tgt < 0 || tgt >= vocab {
		return 0, fmt.Errorf("target=%d out of range for vocab=%d", tgt, vocab)
	}
	targetLogit := logits[tgt]
	rank := 0
	for j := range vocab {
		if j == tgt {
			continue
		}
		lj := logits[j]
		if lj > targetLogit {
			rank++
		} else if lj == targetLogit && j < tgt {
			rank++
		}
	}
	if rank > math.MaxUint16 {
		return 0, fmt.Errorf("rank=%d exceeds uint16 range; vocab=%d target=%d", rank, vocab, tgt)
	}
	return uint16(rank), nil
}
