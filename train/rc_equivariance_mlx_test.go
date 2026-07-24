//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func testRCEquivariantConfig(t *testing.T, objective string) *ArchConfig {
	t.Helper()
	classification := ""
	if objective == arch.ObjectiveClassification {
		classification = `,"classification":{"num_labels":2,"pooling":"mean"}`
	}
	raw := `{
		"name":"rc_equivariance_mlx",
		"model_dim":8,
		"vocab_size":9,
		"seq_len":6,
		"tie_embeddings":true,
		"rc_equivariant":true,
		"blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional"},{"type":"swiglu"}],
		"training":{
			"objective":"` + objective + `",
			"mlm_mask_token_id":3,
			"optimizer":"adamw",
			"steps":12,
			"lr":0.001,
			"grad_clip":1.0,
			"weight_decay":0.0,
			"seed":19,
			"batch_tokens":6` + classification + `
		}
	}`
	cfg, err := ParseArchConfig([]byte(raw), "rc_equivariance_mlx")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.DatasetNucleotideAlphabet = "dna"
	cfg.Training.DatasetNucleotideComplement = []int{0, 1, 2, 3, 7, 6, 5, 4, 8}
	cfg.Training.DatasetTokenEligible = []uint8{0, 0, 0, 0, 1, 1, 1, 1, 1}
	cfg.Training.DatasetBOSID = 1
	cfg.Training.DatasetEOSID = 2
	cfg.Training.DatasetPADID = 0
	if objective == arch.ObjectiveClassification {
		cfg.Training.DatasetRecordFraming = true
		cfg.Training.DatasetClassification = true
		cfg.Training.DatasetNumLabels = 2
	} else {
		cfg.Training.DatasetSequencePacking = true
	}
	return cfg
}

func testRCObjectiveBatch(t *testing.T, cfg *ArchConfig, tokens []int, objective string, label int32) objectiveBatch {
	t.Helper()
	raw := trainBatch{
		x:            append([]int(nil), tokens...),
		y:            append([]int(nil), tokens...),
		lossMask:     make([]float32, len(tokens)),
		segmentIDs:   make([]int32, len(tokens)),
		maskEligible: []uint8{0, 1, 1, 1, 1, 0},
		labels:       []int32{label},
		validMask:    []float32{1, 1, 1, 1, 1, 1},
	}
	prepared := objectiveBatch{
		x:          append([]int(nil), tokens...),
		y:          append([]int(nil), tokens...),
		lossMask:   make([]float32, len(tokens)),
		segmentIDs: make([]int32, len(tokens)),
	}
	for i := range prepared.lossMask {
		prepared.lossMask[i] = 1
	}
	if objective == arch.ObjectiveClassification {
		var err error
		prepared, err = prepareClassificationBatch(cfg, raw, len(tokens), cfg.SeqLen)
		if err != nil {
			t.Fatal(err)
		}
		prepared.segmentIDs = make([]int32, len(tokens))
	}
	if err := attachRCEquivariantInputs(cfg, raw, &prepared, len(tokens), cfg.SeqLen); err != nil {
		t.Fatal(err)
	}
	return prepared
}

func TestRCEquivariantMLMLogitsTransformNumerically(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := testRCEquivariantConfig(t, arch.ObjectiveMLM)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	// Use a non-palindromic pair to exercise both position and vocabulary maps.
	forwardTokens := []int{1, 4, 4, 5, 6, 2}
	reverseTokens := []int{1, 5, 6, 7, 7, 2}
	forward := testRCObjectiveBatch(t, cfg, forwardTokens, arch.ObjectiveMLM, 0)
	reverse := testRCObjectiveBatch(t, cfg, reverseTokens, arch.ObjectiveMLM, 0)

	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(forward, 1, cfg.SeqLen, []string{"logits", "rc_equivariant_hidden"}); err != nil {
		t.Fatal(err)
	}
	forwardLogits, err := trainer.ReadOutput("logits", []int{cfg.SeqLen, cfg.VocabSize})
	if err != nil {
		t.Fatal(err)
	}
	forwardHidden, err := trainer.ReadOutput("rc_equivariant_hidden", []int{1, cfg.SeqLen, 2 * cfg.ModelDim})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(reverse, 1, cfg.SeqLen, []string{"logits", "rc_equivariant_hidden"}); err != nil {
		t.Fatal(err)
	}
	reverseLogits, err := trainer.ReadOutput("logits", []int{cfg.SeqLen, cfg.VocabSize})
	if err != nil {
		t.Fatal(err)
	}
	reverseHidden, err := trainer.ReadOutput("rc_equivariant_hidden", []int{1, cfg.SeqLen, 2 * cfg.ModelDim})
	if err != nil {
		t.Fatal(err)
	}
	align := forward.rcAlignmentPositions
	complement := cfg.Training.DatasetNucleotideComplement
	maxDiff := float32(0)
	for pos := 0; pos < cfg.SeqLen; pos++ {
		rcPos := int(align[pos])
		for token := 0; token < cfg.VocabSize; token++ {
			got := reverseLogits[pos*cfg.VocabSize+token]
			want := forwardLogits[rcPos*cfg.VocabSize+complement[token]]
			diff := float32(math.Abs(float64(got - want)))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	if maxDiff > 2e-5 {
		t.Fatalf("reverse-complement logit equivariance max diff=%g", maxDiff)
	}
	maxHiddenDiff := float32(0)
	for pos := 0; pos < cfg.SeqLen; pos++ {
		rcPos := int(align[pos])
		for channel := 0; channel < cfg.ModelDim; channel++ {
			for _, pair := range [][2]int{{channel, cfg.ModelDim + channel}, {cfg.ModelDim + channel, channel}} {
				got := reverseHidden[pos*2*cfg.ModelDim+pair[0]]
				want := forwardHidden[rcPos*2*cfg.ModelDim+pair[1]]
				diff := float32(math.Abs(float64(got - want)))
				if diff > maxHiddenDiff {
					maxHiddenDiff = diff
				}
			}
		}
	}
	if maxHiddenDiff > 2e-5 {
		t.Fatalf("reverse-complement hidden equivariance max diff=%g", maxHiddenDiff)
	}
}

func TestRCEquivariantClassificationIsInvariantAndTrains(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := testRCEquivariantConfig(t, arch.ObjectiveClassification)
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	forward := testRCObjectiveBatch(t, cfg, []int{1, 4, 4, 5, 6, 2}, arch.ObjectiveClassification, 0)
	reverse := testRCObjectiveBatch(t, cfg, []int{1, 5, 6, 7, 7, 2}, arch.ObjectiveClassification, 0)
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(forward, 1, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatal(err)
	}
	forwardLogits, err := trainer.ReadOutput("classification_logits", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(reverse, 1, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatal(err)
	}
	reverseLogits, err := trainer.ReadOutput("classification_logits", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := range forwardLogits {
		if diff := math.Abs(float64(forwardLogits[i] - reverseLogits[i])); diff > 2e-5 {
			t.Fatalf("classification logit %d differs under RC: forward=%g reverse=%g diff=%g", i, forwardLogits[i], reverseLogits[i], diff)
		}
	}

	first, err := trainer.EvaluateObjectiveGPU(forward, 1, cfg.SeqLen)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(forward, 1, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(forward, 1, cfg.SeqLen)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("classification loss did not decrease: first=%g last=%g", first, last)
	}
}

func TestRCEquivariantGatedDeltaNetClassificationIsInvariant(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := testRCEquivariantConfig(t, arch.ObjectiveClassification)
	cfg.Blocks = []arch.BlockSpec{
		{Type: "gated_deltanet", Heads: 2, DK: 4},
		{Type: "swiglu"},
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	forward := testRCObjectiveBatch(t, cfg, []int{1, 4, 4, 5, 6, 2}, arch.ObjectiveClassification, 0)
	reverse := testRCObjectiveBatch(t, cfg, []int{1, 5, 6, 7, 7, 2}, arch.ObjectiveClassification, 0)
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(forward, 1, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatal(err)
	}
	forwardLogits, err := trainer.ReadOutput("classification_logits", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(reverse, 1, cfg.SeqLen, []string{"classification_logits"}); err != nil {
		t.Fatal(err)
	}
	reverseLogits, err := trainer.ReadOutput("classification_logits", []int{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := range forwardLogits {
		if diff := math.Abs(float64(forwardLogits[i] - reverseLogits[i])); diff > 2e-5 {
			t.Fatalf("Gated DeltaNet classification logit %d differs under RC: forward=%g reverse=%g diff=%g", i, forwardLogits[i], reverseLogits[i], diff)
		}
	}
}
