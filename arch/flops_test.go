package arch

import "testing"

func TestEstimateFLOPsPlainBlock(t *testing.T) {
	cfg := flopsTestConfig([]BlockSpec{{Type: "plain", Heads: 2}})

	got := EstimateFLOPs(cfg)

	const (
		B   = int64(2)
		T   = int64(4)
		D   = int64(8)
		V   = int64(32)
		H   = int64(2)
		HD  = int64(4)
		FFN = int64(16)
	)
	wantForward := int64(0)
	wantForward += 3 * 2 * B * T * D * D   // QKV projections
	wantForward += 2 * B * H * T * T * HD  // attention scores
	wantForward += 2 * B * H * T * T * HD  // attention weighted sum
	wantForward += 2 * B * T * D * D       // output projection
	wantForward += 2 * 2 * B * T * D * FFN // FFN path
	wantForward += 2 * B * T * D * V       // LM head
	wantTraining := 3 * wantForward
	wantPerToken := wantTraining / (B * T)

	if got.ForwardFLOPs != wantForward {
		t.Fatalf("ForwardFLOPs=%d, want %d", got.ForwardFLOPs, wantForward)
	}
	if got.TrainingFLOPs != wantTraining {
		t.Fatalf("TrainingFLOPs=%d, want %d", got.TrainingFLOPs, wantTraining)
	}
	if got.FLOPsPerToken != wantPerToken {
		t.Fatalf("FLOPsPerToken=%d, want %d", got.FLOPsPerToken, wantPerToken)
	}
}

func TestEstimateFLOPsMoreLayersScaleBlockCost(t *testing.T) {
	oneLayer := flopsTestConfig([]BlockSpec{{Type: "plain", Heads: 2}})
	twoLayer := flopsTestConfig([]BlockSpec{
		{Type: "plain", Heads: 2},
		{Type: "plain", Heads: 2},
	})

	one := EstimateFLOPs(oneLayer)
	two := EstimateFLOPs(twoLayer)
	lmHead := int64(2 * 2 * 4 * 8 * 32)
	oneBlock := one.ForwardFLOPs - lmHead
	twoBlocks := two.ForwardFLOPs - lmHead

	if twoBlocks != 2*oneBlock {
		t.Fatalf("two block FLOPs=%d, want exactly 2x one block FLOPs=%d", twoBlocks, 2*oneBlock)
	}
}

func TestEstimateFLOPsGQAIsCheaperThanMHA(t *testing.T) {
	mha := flopsTestConfig([]BlockSpec{{Type: "plain", Heads: 4}})
	gqa := flopsTestConfig([]BlockSpec{{Type: "plain", Heads: 4, KVHeads: 2}})

	mhaEstimate := EstimateFLOPs(mha)
	gqaEstimate := EstimateFLOPs(gqa)

	if gqaEstimate.ForwardFLOPs >= mhaEstimate.ForwardFLOPs {
		t.Fatalf("GQA ForwardFLOPs=%d, want less than MHA ForwardFLOPs=%d", gqaEstimate.ForwardFLOPs, mhaEstimate.ForwardFLOPs)
	}
}

func flopsTestConfig(blocks []BlockSpec) *ArchConfig {
	return &ArchConfig{
		Name:      "flops-test",
		ModelDim:  8,
		VocabSize: 32,
		SeqLen:    4,
		MLPMult:   2,
		Blocks:    blocks,
		Training: TrainingSpec{
			BatchTokens: 8,
		},
	}
}
