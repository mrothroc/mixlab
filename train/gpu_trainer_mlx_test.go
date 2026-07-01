//go:build mlx && cgo && (darwin || linux)

package train

import (
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestMLXGPUTrainerMakeInputs_ExtendedTargets(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 40,
		tokBuf:             make([]int32, 32),
		tgtBuf:             make([]int32, 32),
	}

	xTok := make([]int, 32)
	yTok := make([]int, 32)
	for i := range xTok {
		xTok[i] = 100 + i
		yTok[i] = i + 1
	}

	inputs, err := trainer.makeInputs(xTok, yTok, 2, 16)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}
	if len(inputs) != 2 {
		t.Fatalf("len(inputs) = %d, want 2", len(inputs))
	}

	targets, ok := inputs[1].Data.([]int32)
	if !ok {
		t.Fatalf("targets data type = %T, want []int32", inputs[1].Data)
	}
	want := []int32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
		1, 5, 9, 13,
		17, 21, 25, 29,
	}
	if !reflect.DeepEqual(inputs[1].Shape, []int{40}) {
		t.Fatalf("targets shape = %v, want [40]", inputs[1].Shape)
	}
	if !reflect.DeepEqual(targets, want) {
		t.Fatalf("targets = %v, want %v", targets, want)
	}
}

func TestMLXGPUTrainerMakeInputs_StandardTargetsUnchanged(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 0,
		tokBuf:             make([]int32, 8),
		tgtBuf:             make([]int32, 8),
	}

	xTok := []int{10, 11, 12, 13, 14, 15, 16, 17}
	yTok := []int{1, 2, 3, 4, 5, 6, 7, 8}

	inputs, err := trainer.makeInputs(xTok, yTok, 2, 4)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}

	targets, ok := inputs[1].Data.([]int32)
	if !ok {
		t.Fatalf("targets data type = %T, want []int32", inputs[1].Data)
	}
	want := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if !reflect.DeepEqual(inputs[1].Shape, []int{8}) {
		t.Fatalf("targets shape = %v, want [8]", inputs[1].Shape)
	}
	if !reflect.DeepEqual(targets, want) {
		t.Fatalf("targets = %v, want %v", targets, want)
	}
}

func TestBlockDiffusionMLXInputsMaterializeBoundaries(t *testing.T) {
	trainer := &mlxGPUTrainer{
		lossMaskInput:            true,
		diffusionBlockStartInput: true,
		diffusionBlockEndInput:   true,
		tokBuf:                   make([]int32, 8),
		tgtBuf:                   make([]int32, 8),
		lossMaskBuf:              make([]float32, 8),
		diffusionBlockStartBuf:   make([]int32, 2),
		diffusionBlockEndBuf:     make([]int32, 2),
	}
	batch := objectiveBatch{
		x:                   []int{10, 11, 12, 13, 20, 21, 22, 23},
		y:                   []int{10, 11, 12, 13, 20, 21, 22, 23},
		lossMask:            []float32{0, 1, 0, 0, 0, 0, 1, 0},
		diffusionBlockStart: []int32{1, 4},
		diffusionBlockEnd:   []int32{3, 8},
	}
	inputs, err := trainer.makeObjectiveInputs(batch, 2, 4)
	if err != nil {
		t.Fatalf("makeObjectiveInputs(block_diffusion): %v", err)
	}
	start := tensorInputByName(t, inputs, "diffusion_block_start")
	if start.DType != gpu.TensorInt32 || !reflect.DeepEqual(start.Shape, []int{2}) {
		t.Fatalf("diffusion_block_start dtype/shape=%d/%v, want int32/[2]", start.DType, start.Shape)
	}
	if !reflect.DeepEqual(start.Data, []int32{1, 4}) {
		t.Fatalf("diffusion_block_start data=%v, want [1 4]", start.Data)
	}
	end := tensorInputByName(t, inputs, "diffusion_block_end")
	if end.DType != gpu.TensorInt32 || !reflect.DeepEqual(end.Shape, []int{2}) {
		t.Fatalf("diffusion_block_end dtype/shape=%d/%v, want int32/[2]", end.DType, end.Shape)
	}
	if !reflect.DeepEqual(end.Data, []int32{3, 8}) {
		t.Fatalf("diffusion_block_end data=%v, want [3 8]", end.Data)
	}
	lossMask := tensorInputByName(t, inputs, "loss_mask")
	if !reflect.DeepEqual(lossMask.Data, batch.lossMask) {
		t.Fatalf("loss_mask data=%v, want %v", lossMask.Data, batch.lossMask)
	}

	batch.diffusionBlockEnd = nil
	if _, err := trainer.makeObjectiveInputs(batch, 2, 4); err == nil {
		t.Fatal("makeObjectiveInputs without diffusion_block_end succeeded, want error")
	}
}

func tensorInputByName(t *testing.T, inputs []gpu.TensorInput, name string) gpu.TensorInput {
	t.Helper()
	for _, input := range inputs {
		if input.Name == name {
			return input
		}
	}
	t.Fatalf("missing tensor input %q in %v", name, inputs)
	return gpu.TensorInput{}
}

func TestMLXGPUTrainerPrepareTargets_InvalidExtendedShape(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 19,
		tgtBuf:             make([]int32, 16),
	}

	if _, _, err := trainer.prepareTargets(2, 8, 16); err == nil {
		t.Fatal("prepareTargets succeeded, want error")
	}
}

func TestMLXGPUTrainerMakeObjectiveInputs_DiffusionBlocks(t *testing.T) {
	trainer := &mlxGPUTrainer{
		tokBuf:                   make([]int32, 6),
		tgtBuf:                   make([]int32, 6),
		diffusionBlockStartInput: true,
		diffusionBlockEndInput:   true,
		diffusionBlockStartBuf:   make([]int32, 2),
		diffusionBlockEndBuf:     make([]int32, 2),
	}
	batch := objectiveBatch{
		x:                   []int{10, 11, 12, 20, 21, 22},
		y:                   []int{11, 12, 13, 21, 22, 23},
		diffusionBlockStart: []int32{0, 3},
		diffusionBlockEnd:   []int32{2, 6},
	}

	inputs, err := trainer.makeObjectiveInputs(batch, 2, 3)
	if err != nil {
		t.Fatalf("makeObjectiveInputs: %v", err)
	}
	start := findTensorInput(t, inputs, "diffusion_block_start")
	if !reflect.DeepEqual(start.Shape, []int{2}) {
		t.Fatalf("diffusion_block_start shape=%v, want [2]", start.Shape)
	}
	if start.DType != gpu.TensorInt32 {
		t.Fatalf("diffusion_block_start dtype=%d, want TensorInt32", start.DType)
	}
	if got := start.Data.([]int32); !reflect.DeepEqual(got, []int32{0, 3}) {
		t.Fatalf("diffusion_block_start data=%v, want [0 3]", got)
	}
	end := findTensorInput(t, inputs, "diffusion_block_end")
	if !reflect.DeepEqual(end.Shape, []int{2}) {
		t.Fatalf("diffusion_block_end shape=%v, want [2]", end.Shape)
	}
	if end.DType != gpu.TensorInt32 {
		t.Fatalf("diffusion_block_end dtype=%d, want TensorInt32", end.DType)
	}
	if got := end.Data.([]int32); !reflect.DeepEqual(got, []int32{2, 6}) {
		t.Fatalf("diffusion_block_end data=%v, want [2 6]", got)
	}
}

func TestMLXGPUTrainerMakeObjectiveInputs_RTDGeneratorUsesRawBatchShape(t *testing.T) {
	trainer := &mlxGPUTrainer{
		rtdGeneratorInput:     true,
		rtdGeneratorBatchSize: 2,
		rtdGeneratorSeqLen:    3,
		tokBuf:                make([]int32, 12),
		tgtBuf:                make([]int32, 12),
		rtdGeneratorTokBuf:    make([]int32, 6),
		rtdGeneratorTgtBuf:    make([]int32, 6),
		rtdGeneratorLossBuf:   make([]float32, 6),
	}
	batch := objectiveBatch{
		x:                    []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		y:                    []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
		rtdGeneratorX:        []int{21, 22, 23, 24, 25, 26},
		rtdGeneratorY:        []int{31, 32, 33, 34, 35, 36},
		rtdGeneratorLossMask: []float32{1, 0, 1, 0, 1, 0},
	}

	inputs, err := trainer.makeObjectiveInputs(batch, 4, 3)
	if err != nil {
		t.Fatalf("makeObjectiveInputs: %v", err)
	}
	toks := findTensorInput(t, inputs, "rtd_generator_tokens")
	if !reflect.DeepEqual(toks.Shape, []int{2, 3}) {
		t.Fatalf("rtd_generator_tokens shape=%v, want [2 3]", toks.Shape)
	}
	if got := toks.Data.([]int32); !reflect.DeepEqual(got, []int32{21, 22, 23, 24, 25, 26}) {
		t.Fatalf("rtd_generator_tokens data=%v", got)
	}
	targets := findTensorInput(t, inputs, "rtd_generator_targets")
	if !reflect.DeepEqual(targets.Shape, []int{6}) {
		t.Fatalf("rtd_generator_targets shape=%v, want [6]", targets.Shape)
	}
	if got := targets.Data.([]int32); !reflect.DeepEqual(got, []int32{31, 32, 33, 34, 35, 36}) {
		t.Fatalf("rtd_generator_targets data=%v", got)
	}
	mask := findTensorInput(t, inputs, "rtd_generator_loss_mask")
	if !reflect.DeepEqual(mask.Shape, []int{6}) {
		t.Fatalf("rtd_generator_loss_mask shape=%v, want [6]", mask.Shape)
	}
	if got := mask.Data.([]float32); !reflect.DeepEqual(got, []float32{1, 0, 1, 0, 1, 0}) {
		t.Fatalf("rtd_generator_loss_mask data=%v", got)
	}
}

func TestMLXGPUTrainerMakeObjectiveInputs_RequiresDiffusionBlocks(t *testing.T) {
	trainer := &mlxGPUTrainer{
		tokBuf:                   make([]int32, 4),
		tgtBuf:                   make([]int32, 4),
		diffusionBlockStartInput: true,
		diffusionBlockEndInput:   true,
		diffusionBlockStartBuf:   make([]int32, 2),
		diffusionBlockEndBuf:     make([]int32, 2),
	}
	batch := objectiveBatch{
		x: []int{10, 11, 20, 21},
		y: []int{11, 12, 21, 22},
	}

	_, err := trainer.makeObjectiveInputs(batch, 2, 2)
	if err == nil || !strings.Contains(err.Error(), "diffusion_block_start") {
		t.Fatalf("makeObjectiveInputs error=%v, want missing diffusion_block_start", err)
	}
}

func findTensorInput(t *testing.T, inputs []gpu.TensorInput, name string) gpu.TensorInput {
	t.Helper()
	for _, input := range inputs {
		if input.Name == name {
			return input
		}
	}
	t.Fatalf("input %q not found in %+v", name, inputs)
	return gpu.TensorInput{}
}

func TestCopyWeightDataTransposesEmbedToHead(t *testing.T) {
	src := []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	dst := make([]float32, len(src))
	if err := copyWeightData(dst, []int{2, 3}, src, []int{3, 2}, "head", "embed"); err != nil {
		t.Fatalf("copyWeightData: %v", err)
	}
	want := []float32{
		1, 3, 5,
		2, 4, 6,
	}
	if !reflect.DeepEqual(dst, want) {
		t.Fatalf("dst = %v, want %v", dst, want)
	}
}

func TestBuildTrainerOptimizerSpec_MuonEqR(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "muon_eq_r_optimizer",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Optimizer = "muon_eq_r"
	cfg.Training.ApplyDefaults()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}

	matrixGroup := -1
	for i, shape := range shapes {
		if len(shape.Shape) == 2 && shape.Name != "embed" && shape.Name != "head" {
			matrixGroup = spec.Weights[i].GroupIndex
			break
		}
	}
	if matrixGroup < 0 {
		t.Fatal("no matrix weight group found")
	}
	group := spec.Groups[matrixGroup]
	if group.Kind != gpu.OptimizerMuon {
		t.Fatalf("matrix group kind=%d want Muon", group.Kind)
	}
	if !group.RowNormalize {
		t.Fatal("matrix group RowNormalize=false, want true")
	}
}

func TestBuildTrainerOptimizerSpec_NorMuon(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "normuon_optimizer",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Optimizer = "normuon"
	cfg.Training.ApplyDefaults()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}

	matrixGroup := -1
	for i, shape := range shapes {
		if len(shape.Shape) == 2 && shape.Name != "embed" && shape.Name != "head" {
			matrixGroup = spec.Weights[i].GroupIndex
			break
		}
	}
	if matrixGroup < 0 {
		t.Fatal("no matrix weight group found")
	}
	group := spec.Groups[matrixGroup]
	if group.Kind != gpu.OptimizerMuon {
		t.Fatalf("matrix group kind=%d want Muon", group.Kind)
	}
	if group.MuonNormalization != gpu.MuonNormalizationNorMuon {
		t.Fatalf("matrix group MuonNormalization=%d want NorMuon", group.MuonNormalization)
	}
}

func TestBuildTrainerOptimizerSpec_LAMBWholeModel(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "lamb_optimizer",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Optimizer = "lamb"
	cfg.Training.LAMBBeta1 = 0.87
	cfg.Training.LAMBBeta2 = 0.997
	cfg.Training.LAMBEps = 1e-5
	cfg.Training.ApplyDefaults()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}

	seenEmbed := false
	seenHead := false
	seenScalar := false
	seenMatrix := false
	for i, shape := range shapes {
		group := spec.Groups[spec.Weights[i].GroupIndex]
		if group.Kind != gpu.OptimizerLAMB {
			t.Fatalf("weight %q group kind=%d want LAMB", shape.Name, group.Kind)
		}
		if group.Beta1 != 0.87 || group.Beta2 != 0.997 || group.Epsilon != 1e-5 {
			t.Fatalf("weight %q LAMB hyperparams=%+v", shape.Name, group)
		}
		switch {
		case shape.Name == "embed":
			seenEmbed = true
		case shape.Name == "head":
			seenHead = true
		case shape.IsNormScale:
			seenScalar = true
		case len(shape.Shape) == 2:
			seenMatrix = true
		}
	}
	if !seenEmbed || !seenHead || !seenScalar || !seenMatrix {
		t.Fatalf("did not see all optimizer classes: embed=%v head=%v scalar=%v matrix=%v", seenEmbed, seenHead, seenScalar, seenMatrix)
	}
}

func TestBuildTrainerOptimizerSpec_ParsedAdamWZeroWeightDecay(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "adamw_zero_decay",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 16,
			"optimizer": "adamw",
			"weight_decay": 0.0
		}
	}`), "adamw_zero_decay")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}

	for _, name := range []string{"embed", "head", "final_norm", "wq"} {
		group := optimizerGroupForWeightName(t, spec, shapes, name)
		if group.Kind != gpu.OptimizerAdamW {
			t.Fatalf("%s group kind=%d want AdamW", name, group.Kind)
		}
		if group.WeightDecay != 0 {
			t.Fatalf("%s group weight_decay=%g want 0", name, group.WeightDecay)
		}
	}
}

func TestBuildTrainerOptimizerSpec_ParsedLAMBMixedWeightDecay(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "lamb_mixed_decay",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 16,
			"optimizer": "lamb",
			"weight_decay": 0.3,
			"embed_weight_decay": 0.0,
			"matrix_weight_decay": 0.0,
			"scalar_weight_decay": 0.02,
			"head_weight_decay": 0.04
		}
	}`), "lamb_mixed_decay")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}

	for name, wantDecay := range map[string]float32{
		"embed":      0,
		"head":       0.04,
		"final_norm": 0.02,
		"wq":         0,
	} {
		group := optimizerGroupForWeightName(t, spec, shapes, name)
		if group.Kind != gpu.OptimizerLAMB {
			t.Fatalf("%s group kind=%d want LAMB", name, group.Kind)
		}
		if group.WeightDecay != wantDecay {
			t.Fatalf("%s group weight_decay=%g want %g", name, group.WeightDecay, wantDecay)
		}
	}
}

func TestBuildTrainerOptimizerSpec_CautiousWeightDecay(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "cautious_weight_decay_optimizer",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Steps = 20
	cfg.Training.CautiousWeightDecay = true
	cfg.Training.CautiousWeightDecayActivationFrac = 0.25
	cfg.Training.ApplyDefaults()

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatalf("buildTrainerOptimizerSpec: %v", err)
	}
	for i, group := range spec.Groups {
		if !group.CautiousWeightDecay {
			t.Fatalf("group %d CautiousWeightDecay=false, want true", i)
		}
		if group.CautiousWeightDecayActivationStep != 5 {
			t.Fatalf("group %d activation step=%d want 5", i, group.CautiousWeightDecayActivationStep)
		}
	}
}

func optimizerGroupForWeightName(t *testing.T, spec gpu.TrainerOptimizerSpec, shapes []WeightShape, name string) gpu.OptimizerGroup {
	t.Helper()
	for i, shape := range shapes {
		if shape.Name == name {
			if i >= len(spec.Weights) {
				t.Fatalf("weight optimizer index %d out of range for %q", i, name)
			}
			groupIdx := spec.Weights[i].GroupIndex
			if groupIdx < 0 || groupIdx >= len(spec.Groups) {
				t.Fatalf("group index %d out of range for %q", groupIdx, name)
			}
			return spec.Groups[groupIdx]
		}
	}
	t.Fatalf("missing weight %q in shapes", name)
	return gpu.OptimizerGroup{}
}
