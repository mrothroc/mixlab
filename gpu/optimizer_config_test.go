package gpu

import "testing"

func TestClassifyWeightOptimizer(t *testing.T) {
	tests := []struct {
		name   string
		weight OptimizerWeightMetadata
		want   optimizerClass
	}{
		{"embed", OptimizerWeightMetadata{Name: "embed", Shape: []int{128, 256}}, optimizerClassEmbed},
		{"rtd_generator_embed", OptimizerWeightMetadata{Name: "rtd_generator_embed", Shape: []int{128, 64}}, optimizerClassEmbed},
		{"char_table", OptimizerWeightMetadata{Name: "char_table", Shape: []int{257, 32}}, optimizerClassEmbed},
		{"bigram_table", OptimizerWeightMetadata{Name: "bigram_table", Shape: []int{64, 32}}, optimizerClassEmbed},
		{"trigram_table", OptimizerWeightMetadata{Name: "trigram_table", Shape: []int{64, 32}}, optimizerClassEmbed},
		{"head", OptimizerWeightMetadata{Name: "head", Shape: []int{128, 256}}, optimizerClassHead},
		{"norm", OptimizerWeightMetadata{Name: "final_norm", Shape: []int{128}, IsNormScale: true}, optimizerClassScalar},
		{"scalar_name", OptimizerWeightMetadata{Name: "bigram_scale", Shape: []int{1}}, optimizerClassScalar},
		{"trigram_scale", OptimizerWeightMetadata{Name: "trigram_scale", Shape: []int{1}}, optimizerClassScalar},
		{"smear_gate", OptimizerWeightMetadata{Name: "smear_gate", Shape: []int{12, 1}}, optimizerClassScalar},
		{"smear_scale", OptimizerWeightMetadata{Name: "smear_scale", Shape: []int{1}}, optimizerClassScalar},
		{"backout_lambda", OptimizerWeightMetadata{Name: "backout_lambda", Shape: []int{1}}, optimizerClassScalar},
		{"s4d_B", OptimizerWeightMetadata{Name: "s4d_B_real", Shape: []int{2, 32}}, optimizerClassScalar},
		{"s4d_backward_C", OptimizerWeightMetadata{Name: "s4d_C_backward_real", Shape: []int{64, 32}}, optimizerClassScalar},
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
		Matrix:        OptimizerSettings{Name: "muon", LR: 4, Beta1: 0.94, Beta2: 0.95, Epsilon: 0.96, WeightDecay: 0.97, CautiousWeightDecay: true, CautiousWeightDecayActivationStep: 17, BackendSteps: 5, NewtonSchulzVariant: "polar_express", Nesterov: true},
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
	if spec.Groups[3].Kind != OptimizerMuon || !spec.Groups[3].Nesterov || spec.Groups[3].BackendSteps != 5 || spec.Groups[3].NewtonSchulzVariant != NewtonSchulzPolarExpress {
		t.Fatalf("matrix group = %+v", spec.Groups[3])
	}
	if !spec.Groups[3].CautiousWeightDecay || spec.Groups[3].CautiousWeightDecayActivationStep != 17 {
		t.Fatalf("matrix group cautious weight decay = enabled %v step %d, want enabled step 17", spec.Groups[3].CautiousWeightDecay, spec.Groups[3].CautiousWeightDecayActivationStep)
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

func TestBuildTrainerOptimizerSpecFrozenParameter(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{{Name: "beta", Shape: []int{4}, Frozen: true}},
		Scalar:  OptimizerSettings{Name: "adamw", LR: 0.01},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Weights) != 1 || !spec.Weights[0].Frozen {
		t.Fatalf("weights=%+v", spec.Weights)
	}
}

func TestBuildTrainerOptimizerSpecExtraGroupsAndAllParameterDecay(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "s4d_log_A_real", Shape: []int{2, 4}, Group: "state", ForceNoDecay: true},
			{Name: "s4d_log_dt", Shape: []int{8}, Group: "main", ForceNoDecay: true},
			{Name: "s4d_C_real", Shape: []int{8, 4}, Group: "main"},
			{Name: "s4d_D", Shape: []int{8}, Group: "main"},
			{Name: "bias", Shape: []int{8}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 0.01, WeightDecay: 0.05},
		Head:   OptimizerSettings{Name: "adamw", LR: 0.01, WeightDecay: 0.05},
		Scalar: OptimizerSettings{Name: "adamw", LR: 0.01, WeightDecay: 0.05},
		Matrix: OptimizerSettings{Name: "adamw", LR: 0.01, WeightDecay: 0.05},
		ExtraGroups: map[string]OptimizerSettings{
			"state": {Name: "adamw", LR: 0.001, WeightDecay: 0.05},
			"main":  {Name: "adamw", LR: 0.01, WeightDecay: 0.05},
		},
		DecayAll: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantLR := []float32{0.001, 0.01, 0.01, 0.01, 0.01}
	wantDecay := []bool{false, false, true, true, true}
	for i := range spec.Weights {
		group := spec.Groups[spec.Weights[i].GroupIndex]
		if group.LR != wantLR[i] || spec.Weights[i].Decay != wantDecay[i] {
			t.Fatalf("weight[%d] group=%+v decay=%v want lr=%g decay=%v", i, group, spec.Weights[i].Decay, wantLR[i], wantDecay[i])
		}
	}
}

func TestBuildTrainerOptimizerSpec_BackoutLambdaUsesScalarGroup(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 16}},
			{Name: "backout_lambda", Shape: []int{1}},
			{Name: "wq", Shape: []int{16, 16}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1},
		Head:   OptimizerSettings{Name: "adamw", LR: 2},
		Scalar: OptimizerSettings{Name: "adamw", LR: 3},
		Matrix: OptimizerSettings{Name: "muon", LR: 4},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	backoutSpec := spec.Weights[1]
	backoutGroup := spec.Groups[backoutSpec.GroupIndex]
	if backoutGroup.Kind != OptimizerAdamW || backoutGroup.LR != 3 {
		t.Fatalf("backout group=%+v want scalar AdamW group with LR 3", backoutGroup)
	}
	if backoutSpec.Decay {
		t.Fatalf("backout_lambda should not use weight decay: %+v", backoutSpec)
	}
	if spec.Groups[spec.Weights[2].GroupIndex].Kind != OptimizerMuon {
		t.Fatalf("matrix group=%+v want Muon", spec.Groups[spec.Weights[2].GroupIndex])
	}
}

func TestBuildTrainerOptimizerSpec_S4DParametersUseScalarAdamGroup(t *testing.T) {
	names := []string{
		"s4d_log_dt", "s4d_log_A_real", "s4d_A_imag",
		"s4d_C_real", "s4d_C_imag", "s4d_D",
	}
	weights := make([]OptimizerWeightMetadata, len(names))
	for i, name := range names {
		shape := []int{8, 32}
		if name == "s4d_log_dt" || name == "s4d_D" {
			shape = []int{8}
		}
		weights[i] = OptimizerWeightMetadata{Name: name, Shape: shape}
	}
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: weights,
		Embed:   OptimizerSettings{Name: "adamw", LR: 1},
		Head:    OptimizerSettings{Name: "adamw", LR: 2},
		Scalar:  OptimizerSettings{Name: "adamw", LR: 3, WeightDecay: 0.7},
		Matrix:  OptimizerSettings{Name: "muon", LR: 4, WeightDecay: 0.8},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, weight := range spec.Weights {
		group := spec.Groups[weight.GroupIndex]
		if group.Kind != OptimizerAdamW || group.LR != 3 {
			t.Fatalf("%s group=%+v want scalar AdamW LR 3", names[i], group)
		}
		if weight.Decay {
			t.Fatalf("%s unexpectedly enables weight decay", names[i])
		}
	}
}

func TestBuildTrainerOptimizerSpec_SmearGateUsesScalarGroup(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 16}},
			{Name: "smear_gate", Shape: []int{12, 1}},
			{Name: "smear_scale", Shape: []int{1}},
			{Name: "wq", Shape: []int{16, 16}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1},
		Head:   OptimizerSettings{Name: "adamw", LR: 2},
		Scalar: OptimizerSettings{Name: "adamw", LR: 3},
		Matrix: OptimizerSettings{Name: "muon", LR: 4},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	if spec.Weights[1].GroupIndex != spec.Weights[2].GroupIndex {
		t.Fatalf("smear weights use different groups: gate=%+v scale=%+v", spec.Weights[1], spec.Weights[2])
	}
	smearGroup := spec.Groups[spec.Weights[1].GroupIndex]
	if smearGroup.Kind != OptimizerAdamW || smearGroup.LR != 3 {
		t.Fatalf("smear group=%+v want scalar AdamW group with LR 3", smearGroup)
	}
	if spec.Weights[1].Decay || spec.Weights[2].Decay {
		t.Fatalf("smear weights should not use weight decay: gate=%+v scale=%+v", spec.Weights[1], spec.Weights[2])
	}
	matrixGroup := spec.Groups[spec.Weights[3].GroupIndex]
	if matrixGroup.Kind != OptimizerMuon {
		t.Fatalf("matrix group=%+v want Muon", matrixGroup)
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

func TestBuildTrainerOptimizerSpec_SGD(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "wq", Shape: []int{4, 4}},
		},
		Embed:  OptimizerSettings{Name: "sgd", LR: 0.01, Beta1: 0.9},
		Head:   OptimizerSettings{Name: "sgd", LR: 0.01, Beta1: 0.9},
		Scalar: OptimizerSettings{Name: "sgd", LR: 0.01, Beta1: 0.9},
		Matrix: OptimizerSettings{Name: "sgd", LR: 0.01, Beta1: 0.9},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	if spec.Groups[0].Kind != OptimizerSGD {
		t.Fatalf("embed group Kind = %d, want SGD (%d)", spec.Groups[0].Kind, OptimizerSGD)
	}
	matrixGroup := spec.Groups[spec.Weights[1].GroupIndex]
	if matrixGroup.Kind != OptimizerSGD || matrixGroup.Beta1 != 0.9 {
		t.Fatalf("matrix group = %+v, want SGD momentum 0.9", matrixGroup)
	}
}

func TestBuildTrainerOptimizerSpec_LAMB(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "head", Shape: []int{4, 10}},
			{Name: "norm", Shape: []int{4}, IsNormScale: true},
			{Name: "wq", Shape: []int{4, 4}},
		},
		Embed:  OptimizerSettings{Name: "lamb", LR: 1, Beta1: 0.11, Beta2: 0.91, Epsilon: 1e-6, WeightDecay: 0.01, LAMBTrustRatioCap: 8},
		Head:   OptimizerSettings{Name: "lamb", LR: 2, Beta1: 0.12, Beta2: 0.92, Epsilon: 2e-6, WeightDecay: 0.02, LAMBTrustRatioCap: 9},
		Scalar: OptimizerSettings{Name: "lamb", LR: 3, Beta1: 0.13, Beta2: 0.93, Epsilon: 3e-6, WeightDecay: 0.03, LAMBTrustRatioCap: 10},
		Matrix: OptimizerSettings{Name: "lamb", LR: 4, Beta1: 0.14, Beta2: 0.94, Epsilon: 4e-6, WeightDecay: 0.04, LAMBTrustRatioCap: 11},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	if len(spec.Groups) != 4 {
		t.Fatalf("len(spec.Groups) = %d, want 4", len(spec.Groups))
	}
	for i, group := range spec.Groups {
		if group.Kind != OptimizerLAMB {
			t.Fatalf("group %d Kind=%d, want LAMB (%d)", i, group.Kind, OptimizerLAMB)
		}
	}
	if spec.Groups[0].Beta1 != 0.11 || spec.Groups[1].Beta2 != 0.92 || spec.Groups[2].Epsilon != 3e-6 || spec.Groups[3].WeightDecay != 0.04 || spec.Groups[3].LAMBTrustRatioCap != 11 {
		t.Fatalf("LAMB groups did not preserve settings: %+v", spec.Groups)
	}
	if spec.Weights[2].Decay {
		t.Fatalf("scalar LAMB weight should still not decay: %+v", spec.Weights[2])
	}
	if !spec.Weights[3].Decay {
		t.Fatalf("matrix LAMB weight should decay: %+v", spec.Weights[3])
	}
}

func TestBuildTrainerOptimizerSpec_MuonEqRForMatrix(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "wq", Shape: []int{4, 4}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1},
		Head:   OptimizerSettings{Name: "adamw", LR: 2},
		Scalar: OptimizerSettings{Name: "adamw", LR: 3},
		Matrix: OptimizerSettings{
			Name:         "muon_eq_r",
			LR:           4,
			Beta1:        0.9,
			Beta2:        0.95,
			Epsilon:      1e-8,
			BackendSteps: 5,
			RowNormalize: true,
		},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	matrixGroup := spec.Groups[spec.Weights[1].GroupIndex]
	if matrixGroup.Kind != OptimizerMuon {
		t.Fatalf("matrix group Kind = %d, want Muon (%d)", matrixGroup.Kind, OptimizerMuon)
	}
	if !matrixGroup.RowNormalize {
		t.Fatalf("matrix group RowNormalize=false, want true")
	}
}

func TestBuildTrainerOptimizerSpec_NorMuonForMatrix(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "wq", Shape: []int{4, 4}},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1},
		Head:   OptimizerSettings{Name: "adamw", LR: 2},
		Scalar: OptimizerSettings{Name: "adamw", LR: 3},
		Matrix: OptimizerSettings{
			Name:              "normuon",
			LR:                4,
			Beta1:             0.9,
			Beta2:             0.95,
			Epsilon:           1e-8,
			BackendSteps:      5,
			MuonNormalization: MuonNormalizationNorMuon,
		},
	})
	if err != nil {
		t.Fatalf("BuildTrainerOptimizerSpec error = %v", err)
	}
	matrixGroup := spec.Groups[spec.Weights[1].GroupIndex]
	if matrixGroup.Kind != OptimizerMuon {
		t.Fatalf("matrix group Kind = %d, want Muon (%d)", matrixGroup.Kind, OptimizerMuon)
	}
	if matrixGroup.MuonNormalization != MuonNormalizationNorMuon {
		t.Fatalf("matrix group MuonNormalization=%d, want NorMuon (%d)", matrixGroup.MuonNormalization, MuonNormalizationNorMuon)
	}
}

func TestBuildTrainerOptimizerSpecFreezesModelBuffers(t *testing.T) {
	spec, err := BuildTrainerOptimizerSpec(TrainerOptimizerConfig{
		Weights: []OptimizerWeightMetadata{
			{Name: "embed", Shape: []int{10, 4}},
			{Name: "block_norm_running_mean", Shape: []int{4}, IsBuffer: true},
			{Name: "block_norm_running_var", Shape: []int{4}, IsBuffer: true},
		},
		Embed:  OptimizerSettings{Name: "adamw", LR: 1, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8},
		Head:   OptimizerSettings{Name: "adamw", LR: 1, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8},
		Scalar: OptimizerSettings{Name: "adamw", LR: 1, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8},
		Matrix: OptimizerSettings{Name: "adamw", LR: 1, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Weights) != 3 {
		t.Fatalf("weight specs=%d want 3", len(spec.Weights))
	}
	for i := 1; i < 3; i++ {
		if !spec.Weights[i].Frozen || spec.Weights[i].Decay {
			t.Fatalf("buffer weight[%d]=%+v want frozen and non-decayed", i, spec.Weights[i])
		}
	}
}
