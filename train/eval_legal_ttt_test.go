package train

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type fakeLegalTTTTrainer struct {
	events     []string
	scoreCalls int
	trainCalls int
	trainLRs   []float32
}

func (t *fakeLegalTTTTrainer) TrainStepGPU(_ []int, _ []int, _, _ int, lr float32) (float32, error) {
	t.events = append(t.events, fmt.Sprintf("train:%d", t.trainCalls))
	t.trainCalls++
	t.trainLRs = append(t.trainLRs, lr)
	return 0.5, nil
}

func (t *fakeLegalTTTTrainer) SubmitStepGPU(_ []int, _ []int, _, _ int, _ float32) error {
	return nil
}

func (t *fakeLegalTTTTrainer) CollectLossGPU() (float32, error) { return 0, nil }
func (t *fakeLegalTTTTrainer) FlushGPU() error                  { return nil }
func (t *fakeLegalTTTTrainer) SetQATGPU(string) error           { return nil }
func (t *fakeLegalTTTTrainer) SetWeightGPU(string, []float32) error {
	return nil
}
func (t *fakeLegalTTTTrainer) EvaluateGPU(_ []int, _ []int, _, _ int) (float32, error) {
	t.events = append(t.events, fmt.Sprintf("score:%d", t.scoreCalls))
	t.scoreCalls++
	return 1, nil
}
func (t *fakeLegalTTTTrainer) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	return t.EvaluateGPU(batch.x, batch.y, batchSize, seqLen)
}
func (t *fakeLegalTTTTrainer) EvaluatePerTokenGPU(_ []int, _ []int, _, _ int) ([]float32, error) {
	return nil, nil
}
func (t *fakeLegalTTTTrainer) EvaluateLoRATTTGPU(_ []int, _ []int, _, _, _ int, _ float32, _ int) (float32, error) {
	return 0, nil
}
func (t *fakeLegalTTTTrainer) CloseTrainer() {}

func TestLegalChunkSGDLR(t *testing.T) {
	const base = float32(0.01)
	got := []float32{
		legalChunkSGDLR(base, "cosine", 0, 4),
		legalChunkSGDLR(base, "cosine", 1, 4),
		legalChunkSGDLR(base, "cosine", 2, 4),
		legalChunkSGDLR(base, "cosine", 3, 4),
	}
	want := []float32{
		base,
		base * float32(0.5*(1+math.Cos(math.Pi*1/4))),
		base * 0.5,
		base * float32(0.5*(1+math.Cos(math.Pi*3/4))),
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-8 {
			t.Fatalf("lr[%d] = %g, want %g", i, got[i], want[i])
		}
	}
	if got := legalChunkSGDLR(base, "constant", 3, 4); got != base {
		t.Fatalf("constant lr = %g, want %g", got, base)
	}
}

func TestLegalChunkSGDScoreFirstOrdering(t *testing.T) {
	dir := t.TempDir()
	writeLegalTTTShard(t, filepath.Join(dir, "val_00.bin"), []uint16{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0})
	writeLegalTTTLUTs(t, dir, 8)
	momentum := 0.9
	cfg := &ArchConfig{
		Name:      "legal_order",
		ModelDim:  8,
		VocabSize: 8,
		SeqLen:    2,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}},
		Training:  TrainingSpec{BatchTokens: 4},
		Eval: &EvalSpec{
			TTTMode:       "legal_chunk_sgd",
			ChunkTokens:   8,
			TTTEpochs:     2,
			TTTLR:         0.01,
			TTTMomentum:   &momentum,
			TTTLRSchedule: "cosine",
		},
	}
	trainer := &fakeLegalTTTTrainer{}
	if err := runFullEvalLegalChunkSGDWithTrainer(cfg, filepath.Join(dir, "val_*.bin"), trainer, dir); err != nil {
		t.Fatalf("runFullEvalLegalChunkSGDWithTrainer: %v", err)
	}
	wantPrefix := []string{"score:0", "score:1", "train:0", "train:1", "train:2", "train:3", "score:2", "score:3"}
	if len(trainer.events) != 12 {
		t.Fatalf("events=%v, want 12 events", trainer.events)
	}
	for i, want := range wantPrefix {
		if trainer.events[i] != want {
			t.Fatalf("event[%d]=%q, want %q; events=%v", i, trainer.events[i], want, trainer.events)
		}
	}
	if trainer.scoreCalls != 4 || trainer.trainCalls != 8 {
		t.Fatalf("scoreCalls=%d trainCalls=%d, want 4 and 8", trainer.scoreCalls, trainer.trainCalls)
	}
	if trainer.trainLRs[0] != 0.01 {
		t.Fatalf("first chunk lr=%g, want 0.01", trainer.trainLRs[0])
	}
	if math.Abs(float64(trainer.trainLRs[4]-0.005)) > 1e-8 {
		t.Fatalf("second chunk lr=%g, want 0.005", trainer.trainLRs[4])
	}
}

func writeLegalTTTShard(t *testing.T, path string, tokens []uint16) {
	t.Helper()
	const (
		shardMagic   = 20240520
		shardVersion = 1
		headerInts   = 256
	)
	header := make([]int32, headerInts)
	header[0] = shardMagic
	header[1] = shardVersion
	header[2] = int32(len(tokens))
	buf := make([]byte, headerInts*4+len(tokens)*2)
	for i, v := range header {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	off := headerInts * 4
	for i, tok := range tokens {
		binary.LittleEndian.PutUint16(buf[off+i*2:], tok)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeLegalTTTLUTs(t *testing.T, dir string, vocab int) {
	t.Helper()
	bytesPerToken := make([]byte, vocab*2)
	hasLeading := make([]byte, vocab)
	isBoundary := make([]byte, vocab)
	for i := 0; i < vocab; i++ {
		binary.LittleEndian.PutUint16(bytesPerToken[i*2:], 1)
		isBoundary[i] = 1
	}
	if err := os.WriteFile(filepath.Join(dir, "bytes_per_token.bin"), bytesPerToken, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "has_leading_space.bin"), hasLeading, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "is_boundary_token.bin"), isBoundary, 0o644); err != nil {
		t.Fatal(err)
	}
}
