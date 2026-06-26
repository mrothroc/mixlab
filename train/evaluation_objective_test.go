package train

import (
	"reflect"
	"testing"
)

type recordingObjectiveEvaluator struct {
	batch     objectiveBatch
	batchSize int
	seqLen    int
	calls     int
}

func (r *recordingObjectiveEvaluator) EvaluateObjectiveGPU(batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	r.batch = batch
	r.batchSize = batchSize
	r.seqLen = seqLen
	r.calls++
	return 1.25, nil
}

func TestEvaluateTokensViaObjectiveGPUWrapsTokenBatch(t *testing.T) {
	trainer := &recordingObjectiveEvaluator{}
	x := []int{1, 2, 3, 4}
	y := []int{2, 3, 4, 5}

	loss, err := evaluateTokensViaObjectiveGPU(trainer, x, y, 2, 2)
	if err != nil {
		t.Fatalf("evaluateTokensViaObjectiveGPU: %v", err)
	}
	if loss != 1.25 {
		t.Fatalf("loss=%g, want 1.25", loss)
	}
	if trainer.calls != 1 {
		t.Fatalf("calls=%d, want 1", trainer.calls)
	}
	if trainer.batchSize != 2 || trainer.seqLen != 2 {
		t.Fatalf("shape=(%d,%d), want (2,2)", trainer.batchSize, trainer.seqLen)
	}
	if !reflect.DeepEqual(trainer.batch.x, x) {
		t.Fatalf("wrapped x=%v, want %v", trainer.batch.x, x)
	}
	if !reflect.DeepEqual(trainer.batch.y, y) {
		t.Fatalf("wrapped y=%v, want %v", trainer.batch.y, y)
	}
	if trainer.batch.lossMask != nil {
		t.Fatalf("lossMask=%v, want nil for plain token evaluation", trainer.batch.lossMask)
	}
}
