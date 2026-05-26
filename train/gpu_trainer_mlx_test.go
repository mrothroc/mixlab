//go:build mlx && cgo && (darwin || linux)

package train

import (
	"reflect"
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

func TestMLXGPUTrainerPrepareTargets_InvalidExtendedShape(t *testing.T) {
	trainer := &mlxGPUTrainer{
		declaredTargetSize: 19,
		tgtBuf:             make([]int32, 16),
	}

	if _, _, err := trainer.prepareTargets(2, 8, 16); err == nil {
		t.Fatal("prepareTargets succeeded, want error")
	}
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
