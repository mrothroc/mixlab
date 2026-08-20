package arch

import (
	"strings"
	"testing"
)

func TestZeroBlockClassificationRequiresExternalInputAdapter(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "causal language model",
			raw:  `{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[],"training":{"objective":"causal","batch_tokens":4}}`,
			want: `training.objective="causal" must define at least one block`,
		},
		{
			name: "masked language model",
			raw:  `{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[],"training":{"objective":"mlm","mlm_mask_token_id":1,"batch_tokens":4}}`,
			want: `training.objective="mlm" must define at least one block`,
		},
		{
			name: "classification token embedding",
			raw:  `{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[],"training":{"objective":"classification","classification":{"num_labels":2},"batch_tokens":4}}`,
			want: `with blocks: [] requires input_adapter.kind="linear_frames" or "discrete_codebooks"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ParseArchConfig([]byte(tt.raw), tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestZeroBlockExternalAdapterClassificationBuildsTrainingAndEvalIR(t *testing.T) {
	for _, tt := range []struct {
		name    string
		adapter string
		input   string
	}{
		{
			name:    "continuous frames",
			adapter: `{"kind":"linear_frames","feature_dim":3,"bias":false,"norm":"none"}`,
			input:   "continuous_frames",
		},
		{
			name:    "discrete codebooks",
			adapter: `{"kind":"discrete_codebooks","num_codebooks":2,"codebook_vocab_size":8,"fusion":"attention_mlp","fusion_hidden_dim":8,"norm":"none"}`,
			input:   "codebook_tokens",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			raw := `{
				"model_dim":8,
				"seq_len":4,
				"positional_embedding":"none",
				"final_norm":false,
				"input_adapter":` + tt.adapter + `,
				"blocks":[],
				"training":{
					"objective":"classification",
					"classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0,"bias":false},
					"batch_tokens":8
				}
			}`
			cfg, err := ParseArchConfig([]byte(raw), tt.name)
			if err != nil {
				t.Fatal(err)
			}
			for _, build := range []struct {
				name string
				fn   func(*ArchConfig) (*Program, error)
			}{
				{name: "training", fn: BuildIRProgramFromConfig},
				{name: "eval", fn: BuildEvalIRProgramFromConfig},
			} {
				t.Run(build.name, func(t *testing.T) {
					prog, err := build.fn(cfg)
					if err != nil {
						t.Fatal(err)
					}
					if !programHasInput(prog, tt.input) || programHasInput(prog, "targets") {
						t.Fatalf("inputs=%+v", prog.Inputs)
					}
					if !programHasOutput(prog, "x_hidden") || !programHasOutput(prog, "classification_logits") || programHasOutput(prog, "logits") {
						t.Fatalf("outputs=%+v", prog.Outputs)
					}
				})
			}
		})
	}
}

func TestZeroBlockDASBProbeParameterCount(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name":"dasb_zero_block_probe",
		"model_dim":1024,
		"seq_len":128,
		"positional_embedding":"none",
		"final_norm":false,
		"input_adapter":{
			"kind":"discrete_codebooks",
			"num_codebooks":2,
			"codebook_vocab_size":1024,
			"fusion":"attention_mlp",
			"fusion_hidden_dim":1024,
			"norm":"none"
		},
		"blocks":[],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":18,"pooling":"mean","bias":false},
			"batch_tokens":128
		}
	}`), "dasb_zero_block_probe")
	if err != nil {
		t.Fatal(err)
	}
	weights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{
		"input_adapter_codebook_embedding",
		"input_adapter_codebook_attn_w1",
		"input_adapter_codebook_attn_b1",
		"input_adapter_codebook_attn_w2",
		"head_classifier_proj",
	}
	if len(weights) != len(wantNames) {
		t.Fatalf("weights=%+v", weights)
	}
	for i, want := range wantNames {
		if weights[i].Name != want {
			t.Fatalf("weight[%d]=%q want=%q", i, weights[i].Name, want)
		}
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const wantParams int64 = 3_166_208
	if params != wantParams || expanded != wantParams {
		t.Fatalf("params=%d expanded=%d want=%d", params, expanded, wantParams)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if prog.NumWeights != len(wantNames) {
		t.Fatalf("IR weights=%d want=%d", prog.NumWeights, len(wantNames))
	}
}
