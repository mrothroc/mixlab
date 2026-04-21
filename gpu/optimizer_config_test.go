package gpu

import "testing"

func TestClassifyWeightOptimizer(t *testing.T) {
	tests := []struct {
		name   string
		weight OptimizerWeightMetadata
		want   optimizerClass
	}{
		{"embed", OptimizerWeightMetadata{Name: "embed", Shape: []int{128, 256}}, optimizerClassEmbed},
		{"bigram_table", OptimizerWeightMetadata{Name: "bigram_table", Shape: []int{64, 32}}, optimizerClassEmbed},
		{"head", OptimizerWeightMetadata{Name: "head", Shape: []int{128, 256}}, optimizerClassHead},
		{"norm", OptimizerWeightMetadata{Name: "final_norm", Shape: []int{128}, IsNormScale: true}, optimizerClassScalar},
		{"scalar_name", OptimizerWeightMetadata{Name: "bigram_scale", Shape: []int{1}}, optimizerClassScalar},
		{"vector", OptimizerWeightMetadata{Name: "bias", Shape: []int{128}}, optimizerClassScalar},
		{"matrix", OptimizerWeightMetadata{Name: "wq", Shape: []int{128, 128}}, optimizerClassMatrix},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := classifyWeightOptimizer(tt.weight)
			if err != nil {
				t.Fatalf("classifyWeightOptimizer(%+v) error = %v", tt.weight, err)
			}
			if got != tt.want {
				t.Fatalf("classifyWeightOptimizer(%+v) = %v, want %v", tt.weight, got, tt.want)
			}
		})
	}
}

func TestClassifyWeightOptimizerRejectsUnclassified(t *testing.T) {
	if _, err := classifyWeightOptimizer(OptimizerWeightMetadata{Name: "cube", Shape: []int{2, 3, 4}}); err == nil {
		t.Fatal("expected error for unclassified weight")
	}
}

func TestBuildTrainerOptimizerSpec(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "head", Shape: []int{4, 10}},
			{Name: "norm", Shape: []int{4}, IsNormScale: true},
			{Name: "wq", Shape: []int{4, 4}},
			{Name: "wk", Shape: []int{4, 4}},
		},
		Embed:         OptimizerSettings{Name: "adamw", LR: 1, Beta1: 0.1, Beta2: 0.2, Epsilon: 0.3, WeightDecay: 0.4},
		Head:          OptimizerSettings{Name: "adamw", LR: 2, Beta1: 0.5, Beta2: 0.6, Epsilon: 0.7, WeightDecay: 0.8},
		Scalar:        OptimizerSettings{Name: "adamw", LR: 3, Beta1: 0.9, Beta2: 0.91, Epsilon: 0.92, WeightDecay: 0.93},
		Matrix:        OptimizerSettings{Name: "muon", LR: 4, Beta1: 0.94, Beta2: 0.95, Epsilon: 0.96, WeightDecay: 0.97, BackendSteps: 5, Nesterov: true},
		MaxGradNorm:   1.25,
		DefaultBaseLR: 0.01,
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	if len(spec.Groups) != 4 {
		t.Fatalf("len(spec.Groups) = %d, want 4", len(spec.Groups))
	}
	if len(spec.Weights) != 5 {
		t.Fatalf("len(spec.Weights) = %d, want 5", len(spec.Weights))
	}
	if spec.Groups[0].Kind != OptimizerAdamW || spec.Groups[0].LR != 1 {
		t.Fatalf("embed group = %+v", spec.Groups[0])
	}
	if spec.Groups[3].Kind != OptimizerMuon || !spec.Groups[3].Nesterov || spec.Groups[3].BackendSteps != 5 {
		t.Fatalf("matrix group = %+v", spec.Groups[3])
	}
	if spec.Weights[2].GroupIndex != 2 || spec.Weights[2].Decay {
		t.Fatalf("scalar weight spec = %+v, want group=2 decay=false", spec.Weights[2])
	}
	if spec.Weights[4].GroupIndex != 3 || !spec.Weights[4].Decay {
		t.Fatalf("second matrix weight spec = %+v, want group=3 decay=true", spec.Weights[4])
	}
	if spec.MaxGradNorm != 1.25 || spec.DefaultBaseLR != 0.01 {
		t.Fatalf("spec lr fields = max_grad_norm=%v default_base_lr=%v", spec.MaxGradNorm, spec.DefaultBaseLR)
	}
}

func TestBuildTrainerOptimizerSpec_AdamWForMatrix(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "wq", Shape: []int{4, 4}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1},
		Head:   OptimizerSettings{Name: "adamw", LR: 2},
		Scalar: OptimizerSettings{Name: "adamw", LR: 3},
		Matrix: OptimizerSettings{Name: "adamw", LR: 4, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	// Matrix group should be AdamW, not Muon
	matrixGroup := spec.Groups[spec.Weights[1].GroupIndex]
	if matrixGroup.Kind != OptimizerAdamW {
		t.Fatalf("matrix group Kind = %d, want AdamW (%d)", matrixGroup.Kind, OptimizerAdamW)
	}
	if matrixGroup.LR != 4 {
		t.Fatalf("matrix group LR = %v, want 4", matrixGroup.LR)
	}
}
