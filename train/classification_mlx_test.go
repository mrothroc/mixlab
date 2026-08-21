//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestNativeClassificationTinyTrainingPlainAndRecurrent(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	for _, block := range []string{
		`{"type":"plain","heads":2}`,
		`{"type":"gated_deltanet","heads":2,"d_k":4}`,
		`{"type":"gated_deltanet","heads":2,"d_k":4,"bidirectional":true}`,
		`{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`,
	} {
		t.Run(block, func(t *testing.T) {
			raw := fmt.Sprintf(`{
				"name":"classification_smoke",
				"model_dim":8,
				"vocab_size":8,
				"seq_len":4,
				"tie_embeddings":true,
				"blocks":[%s],
				"training":{
					"objective":"classification",
					"classification":{"num_labels":2,"pooling":"last"},
					"optimizer":"adamw",
					"steps":30,
					"lr":0.003,
					"grad_clip":1.0,
					"weight_decay":0.0,
					"seed":11,
					"batch_tokens":8
				}
			}`, block)
			cfg, err := ParseArchConfig([]byte(raw), "classification_smoke")
			if err != nil {
				t.Fatal(err)
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
			rawBatch := trainBatch{
				x:          []int{1, 3, 3, 2, 1, 6, 6, 2},
				y:          make([]int, 8),
				labels:     []int32{0, 1},
				validMask:  []float32{1, 1, 1, 1, 1, 1, 1, 1},
				segmentIDs: make([]int32, 8),
			}
			batch, err := prepareObjectiveBatch(cfg, rawBatch, 0, arch.ObjectiveClassification)
			if err != nil {
				t.Fatal(err)
			}
			first, err := trainer.EvaluateObjectiveGPU(batch, 2, 4)
			if err != nil {
				t.Fatal(err)
			}
			for step := 0; step < 30; step++ {
				loss, err := trainer.TrainObjectiveStepGPU(batch, 2, 4, float32(cfg.Training.LR))
				if err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
				if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
					t.Fatalf("step %d non-finite loss=%g", step, loss)
				}
			}
			last, err := trainer.EvaluateObjectiveGPU(batch, 2, 4)
			if err != nil {
				t.Fatal(err)
			}
			if !(last < first) {
				t.Fatalf("classification loss did not decrease: first=%g last=%g", first, last)
			}
		})
	}
}

func TestBidirectionalRecurrentClassificationPaddingInvariance(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	for _, block := range []string{
		`{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`,
		`{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`,
		`{"type":"s4d","state_size":4,"bidirectional":true}`,
	} {
		t.Run(block, func(t *testing.T) {
			shortCfg := bidirectionalClassificationMLXConfig(t, block, 4)
			longCfg := bidirectionalClassificationMLXConfig(t, block, 6)
			shortShapes, err := computeWeightShapes(shortCfg)
			if err != nil {
				t.Fatal(err)
			}
			longShapes, err := computeWeightShapes(longCfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(shortShapes) != len(longShapes) {
				t.Fatalf("weight count changes with padding: short=%d long=%d", len(shortShapes), len(longShapes))
			}
			weights := initWeightData(shortShapes, 23, shortCfg.Training.WeightInit, shortCfg.Training.WeightInitStd)
			shortHidden, shortLogits := evaluateBidirectionalOutputs(t, shortCfg, weights, []int{1, 2, 3, 4}, []float32{1, 1, 1, 1})
			longHidden, longLogits := evaluateBidirectionalOutputs(t, longCfg, weights, []int{1, 2, 3, 4, 7, 6}, []float32{1, 1, 1, 1, 0, 0})
			if diff := maxAbsDiffBidirectional(shortLogits, longLogits); diff > 2e-4 {
				t.Fatalf("classification logits changed with right padding: diff=%g short=%v long=%v", diff, shortLogits, longLogits)
			}
			for i := 0; i < 4*shortCfg.ModelDim; i++ {
				if diff := math.Abs(float64(shortHidden[i] - longHidden[i])); diff > 2e-4 {
					t.Fatalf("valid-prefix hidden[%d] differs with right padding: short=%g long=%g diff=%g", i, shortHidden[i], longHidden[i], diff)
				}
			}
			for i := 4 * longCfg.ModelDim; i < len(longHidden); i++ {
				if longHidden[i] != 0 {
					t.Fatalf("padded hidden[%d]=%g want zero", i, longHidden[i])
				}
			}
		})
	}
}

func TestBidirectionalRecurrentMixerUsesFutureContext(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	for _, baseBlock := range []string{
		`{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4`,
		`{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false`,
	} {
		t.Run(baseBlock, func(t *testing.T) {
			causalCfg := bidirectionalClassificationMLXConfig(t, baseBlock+`}`, 4)
			bidirectionalCfg := bidirectionalClassificationMLXConfig(t, baseBlock+`,"bidirectional":true}`, 4)
			shapes, err := computeWeightShapes(causalCfg)
			if err != nil {
				t.Fatal(err)
			}
			weights := initWeightData(shapes, 31, causalCfg.Training.WeightInit, causalCfg.Training.WeightInitStd)
			mask := []float32{1, 1, 1, 1}
			first := []int{1, 2, 3, 4}
			changedFuture := []int{1, 2, 3, 7}
			causalA, _ := evaluateBidirectionalOutputs(t, causalCfg, weights, first, mask)
			causalB, _ := evaluateBidirectionalOutputs(t, causalCfg, weights, changedFuture, mask)
			if diff := maxAbsDiffBidirectional(causalA[:causalCfg.ModelDim], causalB[:causalCfg.ModelDim]); diff > 1e-6 {
				t.Fatalf("causal first position changed from a future token: diff=%g", diff)
			}
			bidirA, _ := evaluateBidirectionalOutputs(t, bidirectionalCfg, weights, first, mask)
			bidirB, _ := evaluateBidirectionalOutputs(t, bidirectionalCfg, weights, changedFuture, mask)
			if diff := maxAbsDiffBidirectional(bidirA[:bidirectionalCfg.ModelDim], bidirB[:bidirectionalCfg.ModelDim]); diff < 1e-7 {
				t.Fatalf("bidirectional first position did not respond to a future token: diff=%g", diff)
			}
		})
	}
}

func bidirectionalClassificationMLXConfig(t *testing.T, block string, seqLen int) *ArchConfig {
	t.Helper()
	raw := fmt.Sprintf(`{
		"model_dim":8,"vocab_size":8,"seq_len":%d,"positional_embedding":"none",
		"blocks":[%s],
		"training":{"objective":"classification","classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},"optimizer":"adamw","steps":2,"lr":0.001,"grad_clip":1,"weight_decay":0,"seed":23,"batch_tokens":%d}
	}`, seqLen, block, seqLen)
	cfg, err := ParseArchConfig([]byte(raw), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func evaluateBidirectionalOutputs(t *testing.T, cfg *ArchConfig, weights [][]float32, tokens []int, validMask []float32) ([]float32, []float32) {
	t.Helper()
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()
	batch := objectiveBatch{
		x: tokens, classificationLabels: []int32{0}, classificationMask: validMask,
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(batch, 1, cfg.SeqLen, []string{"x_hidden", "classification_logits"}); err != nil {
		t.Fatal(err)
	}
	hidden, err := readTrainerOutput(trainer, "x_hidden", []int{1, cfg.SeqLen, cfg.ModelDim})
	if err != nil {
		t.Fatal(err)
	}
	logits, err := readTrainerOutput(trainer, "classification_logits", []int{1, cfg.Training.Classification.NumLabels})
	if err != nil {
		t.Fatal(err)
	}
	return hidden, logits
}

func maxAbsDiffBidirectional(a, b []float32) float32 {
	var max float32
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > max {
			max = diff
		}
	}
	return max
}

func TestNativeClassificationValidationAcceptsPartialFinalBatch(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := nativeClassificationTestConfig()
	cfg.SeqLen = 6
	cfg.Training.BatchTokens = 12
	cfg.Training.DatasetRecordFraming = true
	cfg.Training.DatasetClassification = true
	cfg.Training.DatasetNumLabels = 2
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.CloseTrainer()

	valSet := &data.ValSet{
		TotalExamples: 3, EvaluatedExamples: 3,
		Batches: []data.ValBatch{
			classificationValBatch([]int32{0, 1}, 2),
			classificationValBatch([]int32{1, 1}, 1),
		},
	}
	var predictions bytes.Buffer
	metrics, err := evaluateClassificationValidationWithTrainer(cfg, valSet, trainer, 0, 2, 6, &predictions, nil)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.Examples != 3 || math.IsNaN(metrics.Loss) || math.IsInf(metrics.Loss, 0) {
		t.Fatalf("metrics=%+v", metrics)
	}
	dec := json.NewDecoder(&predictions)
	count := 0
	for {
		var record classificationPredictionRecord
		if err := dec.Decode(&record); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		count++
	}
	if count != 3 {
		t.Fatalf("prediction rows=%d want 3", count)
	}
}
