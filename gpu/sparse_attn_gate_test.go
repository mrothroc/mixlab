//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestSparseAttnGateWeightUpdatesDuringTraining(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	cfg := &ir.ArchConfig{
		Name:      "sparse_attn_gate_grad",
		ModelDim:  128,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []ir.BlockSpec{
			{Type: "plain", Heads: 8, SparseAttnGate: true},
		},
		Training: ir.TrainingSpec{Steps: 1, LR: 1e-2, BatchTokens: 4},
	}
	prog, err := ir.BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	metas, err := collectWeightShapesForConfig(cfg)
	if err != nil {
		t.Fatalf("collectWeightShapesForConfig: %v", err)
	}

	gateIdx := -1
	for i, meta := range metas {
		if meta.Name == "attn_gate_w" {
			gateIdx = i
			break
		}
	}
	handles, err := initTrainerWeights(metas)
	if err != nil {
		t.Fatalf("initTrainerWeights: %v", err)
	}
	defer FreeHandles(handles)

	if gateIdx < 0 {
		t.Fatal("missing attn_gate_w in weight metadata")
	}
	if got, want := metas[gateIdx].Shape, []int{8, 12}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("attn_gate_w shape = %v, want %v", got, want)
	}

	trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          0.01,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			WeightDecay: 0.0,
		}},
		Weights:       uniformWeightOptimizers(len(metas)),
		DefaultBaseLR: 0.01,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	before := make([]float32, shapeProduct(metas[gateIdx].Shape))
	if err := TrainerReadWeight(trainer, gateIdx, before); err != nil {
		t.Fatalf("TrainerReadWeight(before): %v", err)
	}

	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: []int32{1, 2, 3, 4}},
	}
	loss, err := TrainerStep(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss: %g", loss)
	}

	after := make([]float32, shapeProduct(metas[gateIdx].Shape))
	if err := TrainerReadWeight(trainer, gateIdx, after); err != nil {
		t.Fatalf("TrainerReadWeight(after): %v", err)
	}

	maxDiff := float32(0)
	for i := range before {
		d := float32(math.Abs(float64(after[i] - before[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff <= 1e-8 {
		t.Fatalf("attn_gate_w did not update; max diff=%g", maxDiff)
	}
}

func TestSparseAttnGateParallelResidualTrainStep(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	blocks := make([]ir.BlockSpec, 0, 8)
	for i := 0; i < 4; i++ {
		blocks = append(blocks,
			ir.BlockSpec{Type: "plain", Heads: 4, SparseAttnGate: true},
			ir.BlockSpec{Type: "swiglu"},
		)
	}
	cfg := &ir.ArchConfig{
		Name:             "sparse_attn_gate_parallel_residual",
		ModelDim:         320,
		VocabSize:        8192,
		SeqLen:           1024,
		ParallelResidual: true,
		Blocks:           blocks,
		Training: ir.TrainingSpec{
			Steps:       1,
			LR:          1e-3,
			Seed:        42,
			BatchTokens: 1024,
			GradClip:    1.0,
			WeightDecay: 0.04,
		},
	}

	prog, err := ir.BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	metas, err := collectWeightShapesForConfig(cfg)
	if err != nil {
		t.Fatalf("collectWeightShapesForConfig: %v", err)
	}
	handles, err := initTrainerWeights(metas)
	if err != nil {
		t.Fatalf("initTrainerWeights: %v", err)
	}
	defer FreeHandles(handles)

	trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          0.001,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			WeightDecay: 0.04,
		}},
		Weights:       uniformWeightOptimizers(len(metas)),
		DefaultBaseLR: 0.001,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	tokens := make([]int32, cfg.SeqLen)
	targets := make([]int32, cfg.SeqLen)
	for i := 0; i < cfg.SeqLen; i++ {
		tokens[i] = int32(i % 128)
		targets[i] = int32((i + 1) % 128)
	}
	loss, err := TrainerStep(trainer, []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, cfg.SeqLen}, Data: tokens},
		{Name: "targets", DType: TensorInt32, Shape: []int{cfg.SeqLen}, Data: targets},
	})
	if err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss: %g", loss)
	}
}

func uniformWeightOptimizers(n int) []WeightOptimizer {
	out := make([]WeightOptimizer, n)
	for i := range out {
		out[i] = WeightOptimizer{GroupIndex: 0, Decay: false}
	}
	return out
}

func shapeProduct(shape []int) int {
	total := 1
	for _, d := range shape {
		total *= d
	}
	return total
}

func collectWeightShapesForConfig(cfg *ir.ArchConfig) ([]ir.WeightMeta, error) {
	return ir.CollectWeightShapesWithNgramsRecurrenceAndParallel(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		cfg.EffectiveMLPMult(),
		cfg.TieEmbeddings,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
		cfg.Blocks,
		cfg.Recurrence,
	)
}

func initTrainerWeights(metas []ir.WeightMeta) ([]int64, error) {
	handles := make([]int64, len(metas))
	for i, meta := range metas {
		data := make([]float32, shapeProduct(meta.Shape))
		switch {
		case meta.InitValue != 0:
			for j := range data {
				data[j] = meta.InitValue
			}
		case meta.IsNormScale || meta.InitOne:
			for j := range data {
				data[j] = 1.0
			}
		default:
			for j := range data {
				data[j] = 0.01 * float32((j%17)+1)
			}
		}
		if meta.Name == "attn_gate_w" {
			for j := range data {
				data[j] = 0
			}
		}
		rows, cols := 1, len(data)
		if len(meta.Shape) == 2 {
			rows, cols = meta.Shape[0], meta.Shape[1]
		}
		handle, err := FromData(data, rows, cols)
		if err != nil {
			FreeHandles(handles[:i])
			return nil, err
		}
		handles[i] = handle
	}
	return handles, nil
}
