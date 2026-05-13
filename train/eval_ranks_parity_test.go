//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/logprobs"
	"github.com/mrothroc/mixlab/ranks"
)

// TestRunEvalRanksAlignedWithLogprobs runs eval with both -logprobs-out and
// -ranks-out and validates the position-by-position guarantees from the
// feature spec:
//   - token IDs in ranks.bin match logprobs.bin
//   - ranks are in [0, vocab)
//   - rank 0 implies target == argmax(logits) under the same model
//   - per-position NLL written matches CPU-derived NLL from EvalLogits (single
//     pass parity, not cross-mode against the GPU NLL fast path)
func TestRunEvalRanksAlignedWithLogprobs(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	ranksPath := filepath.Join(dataDir, "ranks.bin")
	if err := RunEvalLogprobsAndRanks(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, logprobsPath, ranksPath); err != nil {
		t.Fatalf("RunEvalLogprobsAndRanks: %v", err)
	}

	logprobBlob, err := os.ReadFile(logprobsPath)
	if err != nil {
		t.Fatalf("ReadFile(logprobs): %v", err)
	}
	logprobHeader, logprobRecs, err := logprobs.Read(bytes.NewReader(logprobBlob))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}
	if logprobHeader.Magic != logprobs.Magic {
		t.Fatalf("logprob magic = %#x, want %#x", logprobHeader.Magic, logprobs.Magic)
	}

	ranksBlob, err := os.ReadFile(ranksPath)
	if err != nil {
		t.Fatalf("ReadFile(ranks): %v", err)
	}
	ranksHeader, rankRecs, err := ranks.Read(bytes.NewReader(ranksBlob))
	if err != nil {
		t.Fatalf("ranks.Read: %v", err)
	}
	if ranksHeader.Magic != ranks.Magic {
		t.Fatalf("ranks magic = %#x, want %#x", ranksHeader.Magic, ranks.Magic)
	}
	if ranksHeader.VocabSize != logprobHeader.VocabSize {
		t.Fatalf("vocab mismatch: logprobs=%d ranks=%d", logprobHeader.VocabSize, ranksHeader.VocabSize)
	}
	if ranksHeader.TotalTokens != logprobHeader.TotalTokens {
		t.Fatalf("totalTokens mismatch: logprobs=%d ranks=%d", logprobHeader.TotalTokens, ranksHeader.TotalTokens)
	}

	vocab := int(ranksHeader.VocabSize)
	if len(rankRecs) != len(logprobRecs) {
		t.Fatalf("len(rankRecs) = %d, want %d", len(rankRecs), len(logprobRecs))
	}

	// Token-ID alignment + rank range.
	for i := range rankRecs {
		if rankRecs[i].TokenID != logprobRecs[i].TokenID {
			t.Fatalf("record[%d] token mismatch: logprobs=%d ranks=%d", i, logprobRecs[i].TokenID, rankRecs[i].TokenID)
		}
		if int(rankRecs[i].Rank) >= vocab {
			t.Fatalf("record[%d] rank=%d not in [0, %d)", i, rankRecs[i].Rank, vocab)
		}
	}

	// Independent ground-truth: run EvalLogits on the same input and check
	// rank semantics row-by-row.
	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	defer func() { _ = sess.Close() }()

	batchTokens := fixture.cfg.Training.BatchTokens
	pairs := len(rankRecs)
	// Use the same byte window the eval path used: full eval truncated to
	// complete batches starting from offset 0.
	truncTokens := fixture.evalTokens[:pairs+1]
	allLogits, err := sess.EvalLogits(truncTokens)
	if err != nil {
		t.Fatalf("EvalLogits: %v", err)
	}
	if len(allLogits) != pairs*vocab {
		t.Fatalf("EvalLogits returned %d floats, want %d", len(allLogits), pairs*vocab)
	}
	if pairs%batchTokens != 0 {
		t.Fatalf("pairs=%d not multiple of batchTokens=%d", pairs, batchTokens)
	}

	for i, rec := range rankRecs {
		row := allLogits[i*vocab : (i+1)*vocab]

		// Rank semantics check: count strictly-greater entries + equal-and-smaller-ID entries.
		tgt := int(rec.TokenID)
		tgtLogit := row[tgt]
		expectedRank := 0
		for j, lj := range row {
			if j == tgt {
				continue
			}
			if lj > tgtLogit {
				expectedRank++
			} else if lj == tgtLogit && j < tgt {
				expectedRank++
			}
		}
		if int(rec.Rank) != expectedRank {
			t.Fatalf("record[%d] rank = %d, want %d (tgt=%d, tgtLogit=%v)", i, rec.Rank, expectedRank, tgt, tgtLogit)
		}

		// Rank 0 must mean target is the argmax (deterministic argmax: first
		// token ID with the maximum logit).
		if rec.Rank == 0 {
			argmax := 0
			for j := 1; j < vocab; j++ {
				if row[j] > row[argmax] {
					argmax = j
				}
			}
			if argmax != tgt {
				t.Fatalf("record[%d] rank=0 but argmax=%d != target=%d", i, argmax, tgt)
			}
		}

		// NLL parity: the written logprob equals the CPU-derived log-sum-exp NLL
		// from the same logits row (sanity check that the export path used the
		// same logits source).
		gotNLL := logprobRecs[i].NLL
		wantNLL, err := targetNLLFromLogits(row, vocab, rec.TokenID)
		if err != nil {
			t.Fatalf("targetNLLFromLogits[%d]: %v", i, err)
		}
		if math.Abs(float64(gotNLL-wantNLL)) > 1e-4 {
			t.Fatalf("record[%d] NLL = %g, want ≈ %g (diff=%g)", i, gotNLL, wantNLL, math.Abs(float64(gotNLL-wantNLL)))
		}
	}
}

// TestRunEvalRanksOnly verifies that -ranks-out alone (without -logprobs-out)
// produces a valid ranks file.
func TestRunEvalRanksOnly(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	ranksPath := filepath.Join(dataDir, "ranks.bin")
	if err := RunEvalLogprobsAndRanks(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, "", ranksPath); err != nil {
		t.Fatalf("RunEvalLogprobsAndRanks(ranks only): %v", err)
	}
	if _, err := os.Stat(ranksPath); err != nil {
		t.Fatalf("ranks file not created: %v", err)
	}
	blob, err := os.ReadFile(ranksPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	header, recs, err := ranks.Read(bytes.NewReader(blob))
	if err != nil {
		t.Fatalf("ranks.Read: %v", err)
	}
	if header.Magic != ranks.Magic {
		t.Fatalf("magic = %#x, want %#x", header.Magic, ranks.Magic)
	}
	if len(recs) == 0 {
		t.Fatal("no records emitted")
	}
}

// TestRunEvalRanksRejectsEmptyOutputs ensures the entry point fails fast when
// neither flag is set.
func TestRunEvalRanksRejectsEmptyOutputs(t *testing.T) {
	fixture := newInferenceSessionFixture(t)
	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	if err := RunEvalLogprobsAndRanks(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, "", ""); err == nil {
		t.Fatal("RunEvalLogprobsAndRanks with no outputs succeeded, want error")
	}
}
