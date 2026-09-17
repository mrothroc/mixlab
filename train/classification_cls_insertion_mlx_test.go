//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestCLSPositionNativeInsertionAndReadout(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX unavailable")
	}
	for _, mode := range []string{"head", "middle", "tail"} {
		t.Run(mode, func(t *testing.T) {
			cfg := clsPositionConfig(t, mode)
			cfg.SeqLen = 64
			cfg.MaxPositions = 65
			cfg.Training.BatchTokens = 128
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			weights := initWeightData(shapes, 23, "normal", .15)
			frames := make([]float32, 128)
			mask := make([]float32, 128)
			for i := range frames {
				frames[i] = float32(i+1) / 128
				mask[i] = 1
			}
			if mode == "tail" {
				for i := 100; i < 128; i++ {
					mask[i] = 0
				}
			}
			raw := trainBatch{frames: frames, validMask: mask, labels: []int32{1, 2}}
			b, err := prepareClassificationBatch(cfg, raw, 128, 64)
			if err != nil {
				t.Fatal(err)
			}
			p, err := BuildEvalIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			p.DeclareOutput("cls_input", arch.TensorFloat32, []int{2, 65, 8})
			tr, err := initGPUTrainer(p, cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.CloseTrainer()
			mt := tr.(*mlxGPUTrainer)
			_, err = mt.EvaluateObjectiveGPUWithOutputs(b, 2, 64, []string{"cls_input", "classification_logits", "x_hidden"})
			if err != nil {
				t.Fatal(err)
			}
			input, err := readTrainerOutput(mt, "cls_input", []int{2, 65, 8})
			if err != nil {
				t.Fatal(err)
			}
			hidden, err := readTrainerOutput(mt, "x_hidden", []int{2, 65, 8})
			if err != nil {
				t.Fatal(err)
			}
			logits, err := readTrainerOutput(mt, "classification_logits", []int{2, 3})
			if err != nil {
				t.Fatal(err)
			}
			cls := weights[weightShapeIndex(shapes, "cls_token")]
			projection := weights[weightShapeIndex(shapes, "head_classifier_proj")]
			bias := weights[weightShapeIndex(shapes, "head_classifier_bias")]
			for row := 0; row < 2; row++ {
				index := 0
				if mode == "middle" {
					index = 32
				} else if mode == "tail" {
					index = 64
					if row == 1 {
						index = 36
					}
				}
				for pos := 0; pos < 65; pos++ {
					for d := 0; d < 8; d++ {
						want := cls[d]
						if pos != index {
							src := pos
							if pos > index {
								src--
							}
							want = frames[row*64+src] * weights[0][d]
						}
						if math.Abs(float64(input[(row*65+pos)*8+d]-want)) > 1e-6 {
							t.Fatalf("wrong insertion row=%d pos=%d channel=%d", row, pos, d)
						}
					}
				}
				// Verify the actual public classifier output against the forced
				// position, without reading an intermediate strided slice view.
				for label := 0; label < 3; label++ {
					want := bias[label]
					for d := 0; d < 8; d++ {
						want += hidden[(row*65+index)*8+d] * projection[d*3+label]
					}
					if diff := math.Abs(float64(want - logits[row*3+label])); diff > 1e-5 {
						t.Fatalf("readout index %d differs by %g", index, diff)
					}
				}
			}
		})
	}
}
