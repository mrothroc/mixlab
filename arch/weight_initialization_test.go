package arch

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestWeightInitializationControls(t *testing.T) {
	for _, tc := range []struct {
		fields  string
		wantErr bool
	}{
		{`"weight_init":"pytorch_linear_all"`, false},
		{`"position_embedding_init_std":1,"cls_token_init_std":1`, false},
		{`"position_embedding_init_std":0,"cls_token_init_std":0`, false},
		{`"position_embedding_init_std":null,"cls_token_init_std":null`, false},
		{`"position_embedding_init_std":-1`, true},
		{`"cls_token_init_std":-1`, true},
		{`"cls_token_init_std":1e99`, true},
	} {
		raw := []byte(`{"model_dim":16,"vocab_size":32,"seq_len":4,"positional_embedding":"learned_absolute",
			"blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional"}],
			"training":{"objective":"classification","classification":{"num_labels":2,"pooling":"cls"},` + tc.fields + `}}`)
		cfg, err := ParseArchConfig(raw, "init")
		if (err != nil) != tc.wantErr {
			t.Fatalf("%s: err=%v", tc.fields, err)
		}
		if err == nil {
			encoded, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			roundTrip, err := ParseArchConfig(encoded, "roundtrip")
			if err != nil {
				t.Fatal(err)
			}
			if roundTrip.Training.WeightInit != cfg.Training.WeightInit ||
				!reflect.DeepEqual(roundTrip.Training.PositionEmbeddingInitStd, cfg.Training.PositionEmbeddingInitStd) ||
				!reflect.DeepEqual(roundTrip.Training.CLSTokenInitStd, cfg.Training.CLSTokenInitStd) {
				t.Fatal("initialization controls changed during JSON round trip")
			}
		}
	}
	for _, field := range []string{"position_embedding_init_std", "cls_token_init_std"} {
		_, err := ParseArchConfig([]byte(`{"model_dim":16,"vocab_size":32,"seq_len":4,
			"blocks":[{"type":"plain","heads":2}],"training":{"`+field+`":1}}`), "inapplicable")
		if err == nil || !strings.Contains(err.Error(), field) {
			t.Fatalf("ignored inapplicable %s: %v", field, err)
		}
	}
}

func TestOrdinaryAffineMetadataCoverage(t *testing.T) {
	for _, tc := range []struct {
		spec     BlockSpec
		matrices int
	}{
		{BlockSpec{Type: "plain", Heads: 4, KVHeads: 2}, 6},
		{BlockSpec{Type: "swiglu"}, 3},
		{BlockSpec{Type: "geglu"}, 3},
		{BlockSpec{Type: "mlp"}, 2},
		{BlockSpec{Type: "moe", NumExperts: 3, TopK: 2, ExpertBlock: &BlockSpec{Type: "swiglu"}}, 10},
		{BlockSpec{Type: "perceiver", NumLatents: 3}, 14},
		{BlockSpec{Type: "bottleneck", NumLatents: 3}, 14},
		{BlockSpec{Type: "retnet", Heads: 4}, 6},
		{BlockSpec{Type: "rwkv"}, 7},
		{BlockSpec{Type: "legacy_mamba"}, 3},
		{BlockSpec{Type: "gated_linear_ssm"}, 4},
		{BlockSpec{Type: "mamba3-canonical"}, 11},
		{BlockSpec{Type: "cross_attention", Heads: 4}, 6},
		{BlockSpec{Type: "token_blend"}, 1},
	} {
		t.Run(tc.spec.Type, func(t *testing.T) {
			metas, err := BlockWeightShapes(tc.spec, 16, 4, 1, 32)
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, m := range metas {
				if m.LinearFanIn > 0 {
					count++
					if len(m.Shape) != 2 || m.LinearFanIn != m.Shape[0] {
						t.Fatalf("bad affine orientation: %+v", m)
					}
				}
			}
			if count != tc.matrices {
				t.Fatalf("affine matrices=%d want %d", count, tc.matrices)
			}
		})
	}
}

func TestPlainAffineBiasFanIn(t *testing.T) {
	metas, err := builtinBlockWeightShapes(BlockSpec{Type: "plain", Heads: 4, KVHeads: 2, AttnBias: true, FFNBias: true}, 16, 4, 1, 32, 4, false, false)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, m := range metas {
		if m.LinearFanIn <= 0 || len(m.Shape) != 1 {
			continue
		}
		count++
		want := 16
		if m.Name == "ff2_bias" {
			want = 64
		}
		if m.LinearFanIn != want || m.PyTorchLinearFanIn != 0 {
			t.Fatalf("%s fan-in=%d legacy fan-in=%d", m.Name, m.LinearFanIn, m.PyTorchLinearFanIn)
		}
	}
	if count != 6 {
		t.Fatalf("paired biases=%d, want 6", count)
	}
}

func TestLearningRateExplicitZeroRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		input  string
		base   float64
		groups [4]float32
	}{
		{`{}`, 0.0003, [4]float32{0.0003, 0.0003, 0.0003, 0.0003}},
		{`{"lr":null}`, 0.0003, [4]float32{0.0003, 0.0003, 0.0003, 0.0003}},
		{`{"lr":0}`, 0, [4]float32{}},
		{`{"lr":0.01,"embed_lr":0,"matrix_lr":0,"scalar_lr":0,"head_lr":0}`, 0.01, [4]float32{}},
		{`{"lr":0.01,"matrix_lr":0}`, 0.01, [4]float32{0.01, 0, 0.01, 0.01}},
		{`{"lr":0,"head_lr":0.02}`, 0, [4]float32{0, 0, 0, 0.02}},
	} {
		var spec TrainingSpec
		if err := json.Unmarshal([]byte(tc.input), &spec); err != nil {
			t.Fatal(err)
		}
		for round := 0; round < 3; round++ {
			spec.ApplyDefaults()
			got := [4]float32{spec.EmbedLR, spec.MatrixLR, spec.ScalarLR, spec.HeadLR}
			if spec.LR != tc.base || got != tc.groups {
				t.Fatalf("%s round %d: base=%g groups=%v", tc.input, round, spec.LR, got)
			}
			encoded, err := json.Marshal(spec)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(encoded, &spec); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestLearningRateRejectsNegative(t *testing.T) {
	for _, field := range []string{"lr", "embed_lr", "matrix_lr", "scalar_lr", "head_lr"} {
		raw := fmt.Sprintf(`{"model_dim":16,"vocab_size":32,"seq_len":4,"blocks":[{"type":"plain","heads":2}],"training":{%q:-0.01}}`, field)
		if _, err := ParseArchConfig([]byte(raw), "negative_lr"); err == nil {
			t.Fatalf("accepted negative %s", field)
		}
	}
}

func TestClassifierDropoutExplicitZeroRoundTrip(t *testing.T) {
	for _, raw := range []string{`{}`, `{"classifier_dropout":0}`, `{"classifier_dropout":0.2}`} {
		var spec ClassificationSpec
		if err := json.Unmarshal([]byte(raw), &spec); err != nil {
			t.Fatal(err)
		}
		want := float32(0.1)
		if spec.ClassifierDropout != nil {
			want = *spec.ClassifierDropout
		}
		blob, err := json.Marshal(spec)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(blob, &spec); err != nil {
			t.Fatal(err)
		}
		if got := spec.effectiveDropout(0.1); got != want {
			t.Fatalf("%s: dropout=%g want %g", raw, got, want)
		}
	}
}
