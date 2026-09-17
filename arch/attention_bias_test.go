package arch

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func parseAttentionBiasConfig(t *testing.T, blocks, fields string) *ArchConfig {
	t.Helper()
	c, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"model_dim":16,"vocab_size":19,"seq_len":4,"mlp_mult":2,
		"blocks":[%s],%s "training":{"batch_tokens":8,"seed":17}
	}`, blocks, fields)), "attention-bias")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestAttentionBiasParsingAndRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		fields   string
		qkv, out bool
	}{
		{``, false, false},
		{`,"attn_bias":false`, false, false},
		{`,"attn_bias":true`, true, true},
		{`,"attn_qkv_bias":false,"attn_out_bias":true`, false, true},
		{`,"attn_qkv_bias":true,"attn_out_bias":false`, true, false},
		{`,"attn_qkv_bias":true,"attn_out_bias":true`, true, true},
		{`,"attn_qkv_bias":false,"attn_out_bias":false`, false, false},
		{`,"attn_out_bias":true`, false, true},
		{`,"attn_qkv_bias":true`, true, false},
	} {
		t.Run(tc.fields, func(t *testing.T) {
			c := parseAttentionBiasConfig(t, `{"type":"plain","heads":4`+tc.fields+`}`, "")
			for round := 0; round < 3; round++ {
				b := c.Blocks[0]
				if b.AttentionQKVBiasEnabled() != tc.qkv || b.AttentionOutBiasEnabled() != tc.out {
					t.Fatalf("wrong effective settings: %+v", b)
				}
				raw, err := json.Marshal(c)
				if err != nil {
					t.Fatal(err)
				}
				c, err = ParseArchConfig(raw, "roundtrip")
				if err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	for _, field := range []string{"attn_qkv_bias", "attn_out_bias"} {
		for _, legacy := range []bool{false, true} {
			for _, split := range []bool{false, true} {
				var b BlockSpec
				err := json.Unmarshal([]byte(fmt.Sprintf(`{"type":"plain","attn_bias":%t,%q:%t}`, legacy, field, split)), &b)
				if err == nil || !strings.Contains(err.Error(), "attn_bias") || !strings.Contains(err.Error(), field) {
					t.Fatalf("conflict not rejected: %v", err)
				}
			}
		}
		for _, value := range []string{`null`, `"true"`, `1`} {
			var b BlockSpec
			if err := json.Unmarshal([]byte(fmt.Sprintf(`{"type":"plain",%q:%s}`, field, value)), &b); err == nil {
				t.Fatalf("accepted %s=%s", field, value)
			}
		}
		var b BlockSpec
		if err := json.Unmarshal([]byte(fmt.Sprintf(`{"type":"swiglu",%q:false}`, field)), &b); err == nil {
			t.Fatalf("accepted %s on non-plain", field)
		}
	}
	f := false
	if err := validateBlockSpec(BlockSpec{Type: "plain", Heads: 4, AttnBias: true, AttnOutBias: &f}, "direct", "blocks", 0); err == nil {
		t.Fatal("programmatic conflict accepted")
	}
}

func TestAttentionBiasLegacyLayoutAndIR(t *testing.T) {
	for _, mode := range []struct{ name, blocks, fields string }{
		{"plain", `{"type":"plain","heads":4%s}`, ""},
		{"gqa_gate_relative", `{"type":"plain","heads":4,"kv_heads":2,"attn_value_gate":true,"relative_attention":"deberta_p2c_c2p","relative_attention_parameterization":"shared_qk_reuse"%s}`, ""},
		{"diff", `{"type":"plain","heads":2,"differential_attention":true%s}`, ""},
		{"parallel", `{"type":"plain","heads":4%s},{"type":"geglu"}`, `"parallel_residual":true,"block_scales":true,`},
		{"group", `{"type":"plain","heads":4,"parallel_group":3%s},{"type":"hgrn2","heads":4},{"type":"swiglu"}`, `"block_scales":true,`},
		{"skip", `{"type":"plain","heads":4,"skip_attention":true%s}`, ""},
		{"kv", `{"type":"plain","heads":4,"attn_bias":true},{"type":"plain","heads":4,"kv_source":1%s}`, ""},
	} {
		for _, enabled := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%t", mode.name, enabled), func(t *testing.T) {
				old := parseAttentionBiasConfig(t, fmt.Sprintf(mode.blocks, fmt.Sprintf(`,"attn_bias":%t`, enabled)), mode.fields)
				split := parseAttentionBiasConfig(t, fmt.Sprintf(mode.blocks, fmt.Sprintf(`,"attn_qkv_bias":%t,"attn_out_bias":%t`, enabled, enabled)), mode.fields)
				a, err := CollectWeightShapesFromConfig(old)
				if err != nil {
					t.Fatal(err)
				}
				b, err := CollectWeightShapesFromConfig(split)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(a, b) {
					t.Fatal("legacy weight layout changed")
				}
				ap, err := BuildIRProgramFromConfig(old)
				if err != nil {
					t.Fatal(err)
				}
				bp, err := BuildIRProgramFromConfig(split)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(ap, bp) {
					t.Fatal("legacy graph changed")
				}
				if bp.NumWeights != len(b) {
					t.Fatalf("IR=%d metadata=%d", bp.NumWeights, len(b))
				}
			})
		}
	}
}

func TestAttentionBiasSplitShapesAndConsumption(t *testing.T) {
	for _, qkv := range []bool{false, true} {
		for _, out := range []bool{false, true} {
			for _, mode := range []struct {
				name, extra string
				widths      []int
			}{
				{"plain", "", []int{16, 16, 16}},
				{"gqa", `,"kv_heads":2`, []int{16, 8, 8}},
				{"gqa_gate", `,"kv_heads":2,"attn_value_gate":true`, []int{16, 8, 24}},
				{"kv", `,"kv_source":1`, []int{16}},
			} {
				t.Run(fmt.Sprintf("%s/%t/%t", mode.name, qkv, out), func(t *testing.T) {
					var spec BlockSpec
					if err := json.Unmarshal([]byte(fmt.Sprintf(`{"type":"plain","heads":4,"attn_qkv_bias":%t,"attn_out_bias":%t%s}`, qkv, out, mode.extra)), &spec); err != nil {
						t.Fatal(err)
					}
					weights, err := BlockWeightShapes(spec, 16, 4, 2, 19)
					if err != nil {
						t.Fatal(err)
					}
					count, err := BlockWeightCount(spec, false, false)
					if err != nil {
						t.Fatal(err)
					}
					if count != len(weights) {
						t.Fatal("count and shapes disagree")
					}
					var got []int
					for i, w := range weights {
						if strings.HasSuffix(w.Name, "_bias") {
							if !w.InitZero || weights[i-1].Name != strings.TrimSuffix(w.Name, "_bias") {
								t.Fatalf("bad bias metadata %+v", w)
							}
							got = append(got, w.Shape[0])
						}
					}
					var want []int
					if qkv {
						want = append(want, mode.widths...)
					}
					if out {
						want = append(want, 16)
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("got=%v want=%v", got, want)
					}
					blocks := fmt.Sprintf(`{"type":"plain","heads":4,"attn_qkv_bias":%t,"attn_out_bias":%t%s}`, qkv, out, mode.extra)
					if mode.name == "kv" {
						blocks = `{"type":"plain","heads":4},` + blocks
					}
					for _, skip := range []bool{false, true} {
						c := parseAttentionBiasConfig(t, blocks, "")
						c.Blocks[len(c.Blocks)-1].SkipAttention = skip
						p, err := BuildIRProgramFromConfig(c)
						if err != nil {
							t.Fatal(err)
						}
						ws, err := CollectWeightShapesFromConfig(c)
						if err != nil {
							t.Fatal(err)
						}
						if p.NumWeights != len(ws) {
							t.Fatal("emitter consumption mismatch")
						}
					}
				})
			}
		}
	}
	base := parseAttentionBiasConfig(t, `{"type":"plain","heads":4},{"type":"plain","heads":4}`, "")
	a, _ := CollectWeightShapesFromConfig(base)
	for i := range base.Blocks {
		v := true
		base.Blocks[i].AttnOutBias = &v
	}
	b, _ := CollectWeightShapesFromConfig(base)
	if delta := countWeightMetaElements(b) - countWeightMetaElements(a); delta != 2*16 {
		t.Fatalf("output-only delta=%d", delta)
	}
}

func TestAttentionBiasParameterAndFLOPDelta(t *testing.T) {
	// The requested six-layer width-512 setting needs exactly six output biases.
	base := parseAttentionBiasConfig(t, `{"type":"plain","heads":8}`, "")
	base.ModelDim = 512
	base.Blocks = make([]BlockSpec, 6)
	for i := range base.Blocks {
		base.Blocks[i] = BlockSpec{Type: "plain", Heads: 8}
	}
	a, err := CollectWeightShapesFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	before := EstimateFLOPs(base).ForwardFLOPs
	for i := range base.Blocks {
		v := true
		base.Blocks[i].AttnOutBias = &v
	}
	b, err := CollectWeightShapesFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	if delta := countWeightMetaElements(b) - countWeightMetaElements(a); delta != 3072 {
		t.Fatalf("output-only parameter delta=%d want 3072", delta)
	}
	if delta := EstimateFLOPs(base).ForwardFLOPs - before; delta != 6*512*8 {
		t.Fatalf("output-only FLOP delta=%d", delta)
	}
}
