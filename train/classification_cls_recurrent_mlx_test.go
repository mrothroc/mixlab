//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func clsMixerConfig(t *testing.T, blocks, position string) *ArchConfig {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal([]byte(clsContinuousConfig), &doc); err != nil {
		t.Fatal(err)
	}
	var parsed []any
	if err := json.Unmarshal([]byte(blocks), &parsed); err != nil {
		t.Fatal(err)
	}
	doc["blocks"] = parsed
	doc["training"].(map[string]any)["classification"].(map[string]any)["cls_position"] = position
	raw, _ := json.Marshal(doc)
	cfg, err := ParseArchConfig(raw, t.Name())
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestCLSRecurrentNative(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	for _, blocks := range []string{
		`[{"type":"s4d","state_size":4,"bidirectional":true}]`,
		`[{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}]`,
		`[{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}]`,
		`[{"type":"plain","heads":2,"attention_mask":"bidirectional"},{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":true,"bidirectional":true}]`,
	} {
		for _, position := range []string{"head", "middle", "tail"} {
			t.Run(blocks+"/"+position, func(t *testing.T) {
				cfg := clsMixerConfig(t, blocks, position)
				shapes, err := computeWeightShapes(cfg)
				if err != nil {
					t.Fatal(err)
				}
				weights := initWeightData(shapes, 23, "normal", .15)
				raw := trainBatch{frames: []float32{-1, .3, .8, .4, 99, -77}, labels: []int32{2}, validMask: []float32{1, 1, 1, 1, 0, 0}}
				if position == "middle" {
					raw.validMask = []float32{1, 1, 1, 1, 1, 1}
				}
				logits := clsForward(t, cfg, weights, raw)
				if position != "middle" {
					shortCfg := *cfg
					shortCfg.SeqLen = 4
					shortCfg.Training.BatchTokens = 4
					short := clsForward(t, &shortCfg, weights, raw)
					if diff := maxAbsDiffBidirectional(logits, short); diff > 2e-5 {
						t.Fatalf("padding changes logits by %g", diff)
					}
					batchCfg := *cfg
					batchCfg.Training.BatchTokens = 12
					batchRaw := trainBatch{frames: append(append([]float32{}, raw.frames...), raw.frames...), labels: []int32{2, 1}, validMask: []float32{1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0}}
					batched := clsForward(t, &batchCfg, weights, batchRaw)
					if diff := maxAbsDiffBidirectional(logits, batched[:3]); diff > 2e-5 {
						t.Fatalf("batch changes logits by %g", diff)
					}
				}
				p, err := BuildIRProgramFromConfig(cfg)
				if err != nil {
					t.Fatal(err)
				}
				tr, err := initGPUTrainer(p, cfg, weights, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer tr.CloseTrainer()
				b, err := prepareClassificationBatch(cfg, raw, 6, 6)
				if err != nil {
					t.Fatal(err)
				}
				first, err := tr.EvaluateObjectiveGPU(b, 1, 6)
				if err != nil {
					t.Fatal(err)
				}
				for i := 0; i < 8; i++ {
					loss, err := tr.(*mlxGPUTrainer).TrainObjectiveStepGPU(b, 1, 6, .001)
					if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
						t.Fatalf("step %d loss %g error %v", i, loss, err)
					}
				}
				last, err := tr.EvaluateObjectiveGPU(b, 1, 6)
				if err != nil {
					t.Fatal(err)
				}
				if last >= first {
					t.Fatalf("loss %g -> %g", first, last)
				}
				trained, err := readTrainerWeights(tr)
				if err != nil {
					t.Fatal(err)
				}
				i := weightShapeIndex(shapes, "cls_token")
				if maxAbsDiffBidirectional(trained[i], weights[i]) < 1e-7 {
					t.Fatal("no CLS gradient")
				}
				t.Logf("loss %g -> %g", first, last)
			})
		}
	}
}

func TestCLSPositionNativeHFParity(t *testing.T) {
	for _, position := range []string{"head", "middle", "tail"} {
		t.Run("s4d_"+position, func(t *testing.T) {
			cfg := clsMixerConfig(t, `[{"type":"s4d","state_size":4,"bidirectional":true}]`, position)
			cfg.MLMHead = ""
			raw, _ := json.Marshal(cfg)
			runCLSPoolingNativeHFParity(t, string(raw))
		})
	}
	t.Run("middle_frames", func(t *testing.T) {
		runCLSPoolingNativeHFParity(t, strings.Replace(clsContinuousConfig, `"pooling":"cls"`, `"pooling":"cls","cls_position":"middle"`, 1))
	})
	t.Run("tail_frames", func(t *testing.T) {
		runCLSPoolingNativeHFParity(t, strings.Replace(clsContinuousConfig, `"pooling":"cls"`, `"pooling":"cls","cls_position":"tail"`, 1))
	})
	t.Run("tail_tokens", func(t *testing.T) {
		config := strings.Replace(clsContinuousConfig, `"input_adapter":{"kind":"linear_frames","feature_dim":1,"norm":"none"}`, `"vocab_size":8`, 1)
		runCLSPoolingNativeHFParity(t, strings.Replace(config, `"pooling":"cls"`, `"pooling":"cls","cls_position":"tail"`, 1))
	})
}
