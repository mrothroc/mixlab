//go:build mlx && cgo && (darwin || linux)

package train

import "testing"

func TestMLXTrainerBuildsCharIDsFromTokenIDs(t *testing.T) {
	if !MLXAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg := &ArchConfig{
		Name:            "char_input_test",
		ModelDim:        8,
		VocabSize:       8,
		SeqLen:          2,
		CharVocabSize:   257,
		CharDim:         8,
		CharMaxPerToken: 3,
		Blocks:          []BlockSpec{{Type: "swiglu"}},
		Training:        TrainingSpec{BatchTokens: 4, Seed: 1},
	}
	cfg.Training.ApplyDefaults()
	cfg.CharFeatureIDs = []int32{
		0, 0, 0,
		11, 12, 0,
		21, 22, 23,
		31, 0, 0,
		41, 42, 0,
		51, 0, 0,
		61, 62, 63,
		71, 0, 0,
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainerIface.CloseTrainer()
	trainer := trainerIface.(*mlxGPUTrainer)
	inputs, err := trainer.makeInputs([]int{1, 2, 0, 3}, []int{2, 3, 4, 5}, 2, 2)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}
	var got []int32
	for _, input := range inputs {
		if input.Name == "char_ids" {
			got = input.Data.([]int32)
			break
		}
	}
	want := []int32{
		11, 12, 0,
		21, 22, 23,
		0, 0, 0,
		31, 0, 0,
	}
	if len(got) != len(want) {
		t.Fatalf("char_ids len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("char_ids[%d]=%d want %d (all=%v)", i, got[i], want[i], got)
		}
	}
}
