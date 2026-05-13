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
	"github.com/mrothroc/mixlab/uncertainty"
)

func TestRunEvalUncertaintyAlignedWithLogprobsAndRanks(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	logprobsPath := filepath.Join(dataDir, "logprobs.bin")
	ranksPath := filepath.Join(dataDir, "ranks.bin")
	uncertaintyPath := filepath.Join(dataDir, "uncertainty.bin")
	if err := RunEvalLogprobsRanksAndUncertainty(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, logprobsPath, ranksPath, uncertaintyPath); err != nil {
		t.Fatalf("RunEvalLogprobsRanksAndUncertainty: %v", err)
	}

	logprobBlob, err := os.ReadFile(logprobsPath)
	if err != nil {
		t.Fatalf("ReadFile(logprobs): %v", err)
	}
	logprobHeader, logprobRecs, err := logprobs.Read(bytes.NewReader(logprobBlob))
	if err != nil {
		t.Fatalf("logprobs.Read: %v", err)
	}

	ranksBlob, err := os.ReadFile(ranksPath)
	if err != nil {
		t.Fatalf("ReadFile(ranks): %v", err)
	}
	ranksHeader, rankRecs, err := ranks.Read(bytes.NewReader(ranksBlob))
	if err != nil {
		t.Fatalf("ranks.Read: %v", err)
	}

	uncertaintyBlob, err := os.ReadFile(uncertaintyPath)
	if err != nil {
		t.Fatalf("ReadFile(uncertainty): %v", err)
	}
	uncertaintyHeader, uncertaintyRecs, err := uncertainty.Read(bytes.NewReader(uncertaintyBlob))
	if err != nil {
		t.Fatalf("uncertainty.Read: %v", err)
	}
	if uncertaintyHeader.Magic != uncertainty.Magic {
		t.Fatalf("uncertainty magic = %#x, want %#x", uncertaintyHeader.Magic, uncertainty.Magic)
	}

	if uncertaintyHeader.VocabSize != logprobHeader.VocabSize || uncertaintyHeader.VocabSize != ranksHeader.VocabSize {
		t.Fatalf("vocab mismatch: logprobs=%d ranks=%d uncertainty=%d", logprobHeader.VocabSize, ranksHeader.VocabSize, uncertaintyHeader.VocabSize)
	}
	if uncertaintyHeader.TotalTokens != logprobHeader.TotalTokens || uncertaintyHeader.TotalTokens != ranksHeader.TotalTokens {
		t.Fatalf("totalTokens mismatch: logprobs=%d ranks=%d uncertainty=%d", logprobHeader.TotalTokens, ranksHeader.TotalTokens, uncertaintyHeader.TotalTokens)
	}
	if len(uncertaintyRecs) != len(logprobRecs) || len(uncertaintyRecs) != len(rankRecs) {
		t.Fatalf("record count mismatch: logprobs=%d ranks=%d uncertainty=%d", len(logprobRecs), len(rankRecs), len(uncertaintyRecs))
	}

	vocab := int(uncertaintyHeader.VocabSize)
	maxEntropy := math.Log(float64(vocab))
	for i := range uncertaintyRecs {
		rec := uncertaintyRecs[i]
		if rec.TokenID != logprobRecs[i].TokenID || rec.TokenID != rankRecs[i].TokenID {
			t.Fatalf("record[%d] token mismatch: logprobs=%d ranks=%d uncertainty=%d", i, logprobRecs[i].TokenID, rankRecs[i].TokenID, rec.TokenID)
		}
		if rec.Top1Prob < float32(1.0/float64(vocab)-1e-6) || rec.Top1Prob > 1.0+1e-6 {
			t.Fatalf("record[%d] top1_prob=%g outside [1/vocab, 1]", i, rec.Top1Prob)
		}
		if rec.Entropy < -1e-6 || rec.Entropy > float32(maxEntropy+1e-5) {
			t.Fatalf("record[%d] entropy=%g outside [0, log(vocab)]", i, rec.Entropy)
		}
		if rec.Margin < -1e-6 || rec.Margin > rec.Top1Prob+1e-6 {
			t.Fatalf("record[%d] margin=%g outside [0, top1_prob=%g]", i, rec.Margin, rec.Top1Prob)
		}
	}

	sess, err := NewInferenceSession(fixture.configPath, fixture.weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	defer func() { _ = sess.Close() }()

	pairs := len(uncertaintyRecs)
	truncTokens := fixture.evalTokens[:pairs+1]
	allLogits, err := sess.EvalLogits(truncTokens)
	if err != nil {
		t.Fatalf("EvalLogits: %v", err)
	}
	if len(allLogits) != pairs*vocab {
		t.Fatalf("EvalLogits returned %d floats, want %d", len(allLogits), pairs*vocab)
	}

	for i, rec := range uncertaintyRecs {
		row := allLogits[i*vocab : (i+1)*vocab]
		wantTop1, wantEntropy, wantMargin, err := uncertaintyFromLogits(row, vocab)
		if err != nil {
			t.Fatalf("uncertaintyFromLogits[%d]: %v", i, err)
		}
		if math.Abs(float64(rec.Top1Prob-wantTop1)) > 1e-5 {
			t.Fatalf("record[%d] top1 = %g, want %g", i, rec.Top1Prob, wantTop1)
		}
		if math.Abs(float64(rec.Entropy-wantEntropy)) > 1e-5 {
			t.Fatalf("record[%d] entropy = %g, want %g", i, rec.Entropy, wantEntropy)
		}
		if math.Abs(float64(rec.Margin-wantMargin)) > 1e-5 {
			t.Fatalf("record[%d] margin = %g, want %g", i, rec.Margin, wantMargin)
		}
	}
}

func TestRunEvalUncertaintyOnly(t *testing.T) {
	fixture := newInferenceSessionFixture(t)

	dataDir := t.TempDir()
	trainPattern := filepath.Join(dataDir, "train_*.bin")
	valPath := filepath.Join(dataDir, "val_000.bin")
	writeInferenceShard(t, valPath, fixture.evalTokens)
	writeInferenceLUTs(t, dataDir, fixture.cfg.VocabSize)

	uncertaintyPath := filepath.Join(dataDir, "uncertainty.bin")
	if err := RunEvalLogprobsRanksAndUncertainty(fixture.configPath, trainPattern, fixture.weightsPath, dataDir, "", "", uncertaintyPath); err != nil {
		t.Fatalf("RunEvalLogprobsRanksAndUncertainty(uncertainty only): %v", err)
	}
	if _, err := os.Stat(uncertaintyPath); err != nil {
		t.Fatalf("uncertainty file not created: %v", err)
	}
	blob, err := os.ReadFile(uncertaintyPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	header, recs, err := uncertainty.Read(bytes.NewReader(blob))
	if err != nil {
		t.Fatalf("uncertainty.Read: %v", err)
	}
	if header.Magic != uncertainty.Magic {
		t.Fatalf("magic = %#x, want %#x", header.Magic, uncertainty.Magic)
	}
	if len(recs) == 0 {
		t.Fatal("no records emitted")
	}
}
