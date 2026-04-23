package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

type fakeTTTTrainer struct {
	loss       float32
	delta      float32
	loraDelta  float32
	evalCalls  int
	trainCalls int
	loraCalls  int
	lrs        []float32
	events     []string
	pending    *float32
}

func (t *fakeTTTTrainer) TrainStepGPU(_ []int, _ []int, _, _ int, lr float32) (float32, error) {
	t.trainCalls++
	t.lrs = append(t.lrs, lr)
	t.events = append(t.events, "train")
	t.loss -= t.delta
	return t.loss, nil
}

func (t *fakeTTTTrainer) SubmitStepGPU(_ []int, _ []int, _, _ int, lr float32) error {
	loss, err := t.TrainStepGPU(nil, nil, 0, 0, lr)
	if err != nil {
		return err
	}
	t.pending = &loss
	return nil
}

func (t *fakeTTTTrainer) CollectLossGPU() (float32, error) {
	if t.pending == nil {
		return 0, nil
	}
	loss := *t.pending
	t.pending = nil
	return loss, nil
}

func (t *fakeTTTTrainer) FlushGPU() error {
	t.pending = nil
	return nil
}

func (t *fakeTTTTrainer) EvaluateGPU(_ []int, _ []int, _, _ int) (float32, error) {
	t.evalCalls++
	t.events = append(t.events, "eval")
	return t.loss, nil
}

func (t *fakeTTTTrainer) EvaluateLoRATTTGPU(_ []int, _ []int, _, _, tttSteps int, lr float32, rank int) (float32, error) {
	t.loraCalls++
	t.lrs = append(t.lrs, lr)
	t.events = append(t.events, "lora_eval")
	if rank <= 0 {
		return 0, nil
	}
	return t.loss - float32(tttSteps)*t.loraDelta, nil
}

func (t *fakeTTTTrainer) CloseTrainer() {}

func testValSet() *data.ValSet {
	return &data.ValSet{Batches: []data.ValBatch{
		{X: []int{1, 2}, Y: []int{2, 3}},
		{X: []int{3, 4}, Y: []int{4, 5}},
	}}
}

func TestMeanValidationLossWithTTTScoreFirst(t *testing.T) {
	trainer := &fakeTTTTrainer{loss: 1.0, delta: 0.2}

	got, err := meanValidationLossWithTTT(testValSet(), trainer, 1, 2, "full", 1, 1e-5, 4)
	if err != nil {
		t.Fatalf("meanValidationLossWithTTT: %v", err)
	}
	if math.Abs(got-0.9) > 1e-6 {
		t.Fatalf("TTT loss = %.6f, want 0.900000", got)
	}
	if trainer.evalCalls != 2 || trainer.trainCalls != 2 {
		t.Fatalf("calls eval=%d train=%d, want 2/2", trainer.evalCalls, trainer.trainCalls)
	}
	wantEvents := []string{"eval", "train", "eval", "train"}
	for i, want := range wantEvents {
		if trainer.events[i] != want {
			t.Fatalf("event[%d]=%q, want %q (score-first order)", i, trainer.events[i], want)
		}
	}
	for _, lr := range trainer.lrs {
		if lr != 1e-5 {
			t.Fatalf("ttt lr = %g, want 1e-5", lr)
		}
	}
}

func TestMeanValidationLossWithTTTDisabledMatchesNormalEval(t *testing.T) {
	valSet := testValSet()
	normalTrainer := &fakeTTTTrainer{loss: 1.0, delta: 0.2}
	tttDisabledTrainer := &fakeTTTTrainer{loss: 1.0, delta: 0.2}

	normal, err := meanValidationLoss(valSet, normalTrainer, 1, 2)
	if err != nil {
		t.Fatalf("meanValidationLoss: %v", err)
	}
	disabled, err := meanValidationLossWithTTT(valSet, tttDisabledTrainer, 1, 2, "full", 0, 1e-5, 4)
	if err != nil {
		t.Fatalf("meanValidationLossWithTTT disabled: %v", err)
	}
	if normal != disabled {
		t.Fatalf("disabled TTT loss = %.6f, want normal %.6f", disabled, normal)
	}
	if tttDisabledTrainer.trainCalls != 0 {
		t.Fatalf("disabled TTT train calls = %d, want 0", tttDisabledTrainer.trainCalls)
	}
}

func TestMeanValidationLossWithTTTMultipleStepsPerBatch(t *testing.T) {
	trainer := &fakeTTTTrainer{loss: 1.0, delta: 0.1}

	got, err := meanValidationLossWithTTT(testValSet(), trainer, 1, 2, "full", 2, 3e-5, 4)
	if err != nil {
		t.Fatalf("meanValidationLossWithTTT: %v", err)
	}
	if math.Abs(got-0.9) > 1e-6 {
		t.Fatalf("TTT loss = %.6f, want 0.900000", got)
	}
	if trainer.trainCalls != 4 {
		t.Fatalf("train calls = %d, want 4", trainer.trainCalls)
	}
}

func TestMeanValidationLossWithLoRATTTImprovesOverNoTTT(t *testing.T) {
	normalTrainer := &fakeTTTTrainer{loss: 1.0, delta: 0.2}
	loraTrainer := &fakeTTTTrainer{loss: 1.0, loraDelta: 0.15}

	normal, err := meanValidationLoss(testValSet(), normalTrainer, 1, 2)
	if err != nil {
		t.Fatalf("meanValidationLoss: %v", err)
	}
	lora, err := meanValidationLossWithTTT(testValSet(), loraTrainer, 1, 2, "lora", 2, 5e-5, 8)
	if err != nil {
		t.Fatalf("meanValidationLossWithTTT lora: %v", err)
	}
	if !(lora < normal) {
		t.Fatalf("LoRA-TTT loss = %.6f, want < normal %.6f", lora, normal)
	}
	if loraTrainer.trainCalls != 0 {
		t.Fatalf("LoRA-TTT full-train calls = %d, want 0", loraTrainer.trainCalls)
	}
	if loraTrainer.loraCalls != 2 {
		t.Fatalf("LoRA-TTT eval calls = %d, want 2", loraTrainer.loraCalls)
	}
	wantEvents := []string{"lora_eval", "lora_eval"}
	for i, want := range wantEvents {
		if loraTrainer.events[i] != want {
			t.Fatalf("event[%d]=%q, want %q", i, loraTrainer.events[i], want)
		}
	}
}
