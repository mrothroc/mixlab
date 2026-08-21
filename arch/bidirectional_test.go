package arch

import (
	"strings"
	"testing"
)

func TestBidirectionalMixerValidationAndClassificationPooling(t *testing.T) {
	for _, tc := range []struct {
		name  string
		block string
	}{
		{name: "mamba3 canonical", block: `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`},
		{name: "gated deltanet", block: `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`},
		{name: "s4d", block: `{"type":"s4d","state_size":4,"bidirectional":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := parseBidirectionalTestConfig(t, tc.block, ObjectiveClassification)
			if got := cfg.EffectiveClassificationPooling(); got != ClassificationPoolingMean {
				t.Fatalf("default classification pooling=%q want mean", got)
			}
		})
	}
}

func TestBidirectionalMixerAllowsMaskedObjectives(t *testing.T) {
	for _, objective := range []string{ObjectiveMLM, ObjectiveMNTP} {
		for _, block := range []string{
			`{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`,
			`{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`,
		} {
			t.Run(objective+block, func(t *testing.T) {
				cfg := parseBidirectionalTestConfig(t, block, objective)
				prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: objective})
				if err != nil {
					t.Fatal(err)
				}
				if programDeclaresInput(prog, sequenceValidMaskInput) {
					t.Fatalf("fixed-length masked program unexpectedly declares %s", sequenceValidMaskInput)
				}
				if countOps(prog, OpReverseValidPrefix) != 2 {
					t.Fatal("masked program did not emit both recurrent directions")
				}
			})
		}
	}
}

func TestBidirectionalMixerRejectsLeakyAndUnsupportedUses(t *testing.T) {
	for _, tc := range []struct {
		name      string
		block     string
		objective string
		want      string
	}{
		{name: "mamba causal", block: `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`, objective: ObjectiveCausal, want: "future-token context"},
		{name: "gated causal", block: `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`, objective: ObjectiveCausal, want: "future-token context"},
		{name: "s4d causal", block: `{"type":"s4d","state_size":4,"bidirectional":true}`, objective: ObjectiveCausal, want: "future-token context"},
		{name: "unsupported hgrn2", block: `{"type":"hgrn2","heads":2,"d_state":2,"bidirectional":true}`, objective: ObjectiveMLM, want: "supported only"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(bidirectionalTestConfigJSON(tc.block, tc.objective)), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseArchConfig error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestBidirectionalMixersReuseWeightsAndEmitTwoDirections(t *testing.T) {
	for _, tc := range []struct {
		name   string
		block  string
		scanOp int
	}{
		{name: "mamba3 canonical", block: `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}`, scanOp: OpMamba3CanonicalBlock},
		{name: "gated deltanet", block: `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4}`, scanOp: OpGatedDeltaScan},
	} {
		t.Run(tc.name, func(t *testing.T) {
			causal := parseBidirectionalTestConfig(t, tc.block, ObjectiveClassification)
			bidirectionalBlock := strings.TrimSuffix(tc.block, "}") + `,"bidirectional":true}`
			bidirectional := parseBidirectionalTestConfig(t, bidirectionalBlock, ObjectiveClassification)
			causalCount, _, err := ParameterCountsFromConfig(causal)
			if err != nil {
				t.Fatal(err)
			}
			bidirectionalCount, _, err := ParameterCountsFromConfig(bidirectional)
			if err != nil {
				t.Fatal(err)
			}
			if bidirectionalCount != causalCount {
				t.Fatalf("bidirectional params=%d causal params=%d", bidirectionalCount, causalCount)
			}

			prog, err := BuildTrainingIRProgramFromConfig(bidirectional, TrainingProgramState{Objective: ObjectiveClassification})
			if err != nil {
				t.Fatal(err)
			}
			if !programDeclaresInput(prog, sequenceValidMaskInput) {
				t.Fatalf("classification program does not declare %s", sequenceValidMaskInput)
			}
			if got := countOps(prog, tc.scanOp); got != 2 {
				t.Fatalf("scan op count=%d want 2", got)
			}
			if got := countOps(prog, OpReverseValidPrefix); got != 2 {
				t.Fatalf("reverse-valid-prefix op count=%d want 2", got)
			}
		})
	}
}

func TestUnidirectionalMixerIRRemainsSingleDirection(t *testing.T) {
	cfg := parseBidirectionalTestConfig(t, `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}`, ObjectiveClassification)
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveClassification})
	if err != nil {
		t.Fatal(err)
	}
	if got := countOps(prog, OpMamba3CanonicalBlock); got != 1 {
		t.Fatalf("mamba op count=%d want 1", got)
	}
	if got := countOps(prog, OpReverseValidPrefix); got != 0 {
		t.Fatalf("reverse-valid-prefix op count=%d want 0", got)
	}
	if programDeclaresInput(prog, sequenceValidMaskInput) {
		t.Fatalf("unidirectional program unexpectedly declares %s", sequenceValidMaskInput)
	}
}

func parseBidirectionalTestConfig(t *testing.T, block, objective string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(bidirectionalTestConfigJSON(block, objective)), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func bidirectionalTestConfigJSON(block, objective string) string {
	training := `{"objective":"` + objective + `","steps":1,"batch_tokens":8}`
	switch objective {
	case ObjectiveClassification:
		training = `{"objective":"classification","steps":1,"batch_tokens":8,"classification":{"num_labels":2}}`
	case ObjectiveMLM, ObjectiveMNTP:
		training = `{"objective":"` + objective + `","steps":1,"batch_tokens":8,"mlm_mask_token_id":1}`
	}
	return `{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[` + block + `],"training":` + training + `}`
}
