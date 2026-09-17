package train

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func attentionBiasFixture(qkv, out bool, extra string) string {
	return fmt.Sprintf(`{
		"model_dim":8,"vocab_size":11,"seq_len":4,"mlp_mult":2,
		"blocks":[{"type":"plain","heads":2,"attn_qkv_bias":%t,"attn_out_bias":%t%s}],
		"training":{"steps":1,"batch_tokens":4,"seed":817}
	}`, qkv, out, extra)
}

// A nonzero, channel-varying fixture detects omitted bias additions. Production
// initialization remains zero; these values stand in for trained biases.
func nonzeroAttentionBiases(weights [][]float32, shapes []WeightShape) error {
	for i, w := range shapes {
		switch w.Name {
		case "wq_bias", "wk_bias", "wv_bias", "wo_bias":
			for j := range weights[i] {
				weights[i][j] = float32((i+3)*(j+1)%17-8) * 0.04
			}
		}
	}
	return nil
}

func TestExportHFAttentionBiasCPUParity(t *testing.T) {
	for _, qkv := range []bool{false, true} {
		for _, out := range []bool{false, true} {
			for _, extra := range []string{"", `,"kv_heads":1,"attn_value_gate":true,"attention_mask":"bidirectional","relative_attention":"deberta_p2c_c2p","relative_attention_window":3,"relative_attention_parameterization":"shared_qk_reuse"`} {
				t.Run(fmt.Sprintf("%t/%t/%s", qkv, out, extra), func(t *testing.T) {
					runExportHFParityCase(t, attentionBiasFixture(qkv, out, extra), [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, dir string) {
						var cfg map[string]any
						readJSON(t, filepath.Join(dir, "config.json"), &cfg)
						b := cfg["blocks"].([]any)[0].(map[string]any)
						if _, ok := b["attn_bias"]; ok {
							t.Fatal("export synthesized conflicting shorthand")
						}
						if b["attn_qkv_bias"] != qkv || b["attn_out_bias"] != out {
							t.Fatalf("wrong exported flags: %v", b)
						}
						var mapping []hfWeightMapping
						readJSON(t, filepath.Join(dir, "weight_map.json"), &mapping)
						tensors, err := loadSafetensors(filepath.Join(dir, "model.safetensors"))
						if err != nil {
							t.Fatal(err)
						}
						for _, proj := range []string{"wq", "wk", "wv", "wo"} {
							want := qkv
							if proj == "wo" {
								want = out
							}
							name := "blocks.0." + proj + ".bias"
							_, exists := tensors[name]
							if containsHFWeight(mapping, name) != want || exists != want {
								t.Fatalf("wrong mapping for %s", name)
							}
						}
					}, scaleHFExportWeightsToTrainedMagnitude, nonzeroAttentionBiases)
				})
			}
		}
	}
}

func TestAttentionBiasLegacyInitializationAndCheckpoint(t *testing.T) {
	split := attentionBiasFixture(true, true, "")
	legacy := strings.Replace(split, `"attn_qkv_bias":true,"attn_out_bias":true`, `"attn_bias":true`, 1)
	lc, err := ParseArchConfig([]byte(legacy), "legacy")
	if err != nil {
		t.Fatal(err)
	}
	sc, err := ParseArchConfig([]byte(split), "split")
	if err != nil {
		t.Fatal(err)
	}
	lw, err := computeWeightShapes(lc)
	if err != nil {
		t.Fatal(err)
	}
	sw, err := computeWeightShapes(sc)
	if err != nil {
		t.Fatal(err)
	}
	for _, init := range []string{"normal", "gpt2", "gptbert", "pytorch_linear"} {
		a := initWeightData(lw, 42, init, 0.02)
		b := initWeightData(sw, 42, init, 0.02)
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("%s initialization changed", init)
		}
		path := filepath.Join(t.TempDir(), "legacy.safetensors")
		if err := exportSafetensors(path, lc, lw, a); err != nil {
			t.Fatal(err)
		}
		loaded, err := loadSafetensorsWeights(path, sw)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(loaded, b) {
			t.Fatal("checkpoint layout changed")
		}
	}
}

func TestExportHFAttentionBiasGPT2(t *testing.T) {
	base := strictGPT2SpecialTokenTestConfig()
	for _, qkv := range []bool{false, true} {
		for _, out := range []bool{false, true} {
			raw := strings.ReplaceAll(base, `"attn_bias":true`, fmt.Sprintf(`"attn_qkv_bias":%t,"attn_out_bias":%t`, qkv, out))
			if raw == base {
				t.Fatal("fixture not replaced")
			}
			cfg, err := ParseArchConfig([]byte(raw), "split-gpt2")
			if err != nil {
				t.Fatal(err)
			}
			err = validateHFGPT2ExportConfig(cfg)
			if qkv && out {
				if err != nil {
					t.Fatal(err)
				}
				dir := t.TempDir()
				cp, wp, tok := writeHFExportFixture(t, dir, raw)
				if err := RunExportHF(ExportHFOptions{ConfigPath: cp, SafetensorsLoad: wp, TokenizerSource: tok, OutputDir: filepath.Join(dir, "hf")}); err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "attn_qkv_bias") || !strings.Contains(err.Error(), "attn_out_bias") {
				t.Fatalf("asymmetric GPT-2 not clearly rejected: %v", err)
			}
		}
	}
}
