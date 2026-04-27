package arch

import "testing"

func TestBlockWeightCount_GatedDeltaNet(t *testing.T) {
	n, err := BlockWeightCount(BlockSpec{Type: "gated_deltanet", Heads: 4, DK: 8}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(shared): %v", err)
	}
	if n != 13 {
		t.Fatalf("shared gated_deltanet count=%d, want 13", n)
	}

	share := false
	n, err = BlockWeightCount(BlockSpec{Type: "gated_deltanet", Heads: 4, DK: 8, KVShare: &share}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount(unshared): %v", err)
	}
	if n != 14 {
		t.Fatalf("unshared gated_deltanet count=%d, want 14", n)
	}
}

func TestBlockWeightShapes_GatedDeltaNetDefaults(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "gated_deltanet", Heads: 4, DK: 8}, 32, 16, 1, 128, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	if len(metas) != 13 {
		t.Fatalf("len(metas)=%d, want 13", len(metas))
	}
	want := map[string][]int{
		"norm_scale":   {32},
		"wq":           {32, 32},
		"w_kv":         {32, 64},
		"q_conv":       {4, 32},
		"k_conv":       {4, 32},
		"v_conv":       {4, 64},
		"w_a":          {32, 4},
		"A_log":        {4},
		"dt_bias":      {4},
		"w_beta":       {32, 4},
		"w_out_gate":   {32, 64},
		"o_norm_scale": {16},
		"wo":           {64, 32},
	}
	for _, meta := range metas {
		ws, ok := want[meta.Name]
		if !ok {
			t.Fatalf("unexpected weight %q", meta.Name)
		}
		if len(meta.Shape) != len(ws) {
			t.Fatalf("%s rank=%v want %v", meta.Name, meta.Shape, ws)
		}
		for i := range ws {
			if meta.Shape[i] != ws[i] {
				t.Fatalf("%s shape=%v want %v", meta.Name, meta.Shape, ws)
			}
		}
	}
}

func TestEmitGatedDeltaNetIR_UsesGatedDeltaScan(t *testing.T) {
	p := NewProgram(13)
	wi, err := emitBlockIR(p, BlockSpec{Type: "gated_deltanet", Heads: 4, DK: 8}, "x", 0, 64, 16, 2, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 13 {
		t.Fatalf("wi=%d, want 13", wi)
	}
	if n := countOps(p, OpOuter); n != 0 {
		t.Fatalf("outer ops=%d, want 0", n)
	}
	if n := countOps(p, OpMatrixScan); n != 0 {
		t.Fatalf("matrix scan ops=%d, want 0", n)
	}
	if n := countOps(p, OpGatedDeltaScan); n != 1 {
		t.Fatalf("gated delta scan ops=%d, want 1", n)
	}
	if n := countOps(p, OpRMSNorm); n < 4 {
		t.Fatalf("rmsnorm ops=%d, want at least 4", n)
	}
	if n := countOps(p, OpSoftplus); n != 1 {
		t.Fatalf("softplus ops=%d, want 1", n)
	}
	if n := countOps(p, OpSiLU); n < 4 {
		t.Fatalf("silu ops=%d, want at least 4", n)
	}
}

func TestParameterCountsFromConfig_GatedDeltaNetFormula(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "gdn_count",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 16,
		"blocks": [
			{"type": "gated_deltanet", "heads": 4, "d_k": 6},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 3e-4, "batch_tokens": 16}
	}`), "gdn_count")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	got, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if got != expanded {
		t.Fatalf("expanded params=%d, want equal to unique %d", expanded, got)
	}

	const (
		D      = int64(32)
		V      = int64(64)
		heads  = int64(4)
		dk     = int64(6)
		dv     = int64(12)
		ffnDim = int64(85) // round(32 * 2.67)
	)
	base := V*D + D*V + D
	conv := int64(4) * (heads*dk + heads*dk + heads*dv)
	gated := D + D*(heads*dk) + D*(heads*dv) + conv + D*heads + heads + heads + D*heads + D*(heads*dv) + dv + (heads*dv)*D
	swiglu := D + D*ffnDim + D*ffnDim + ffnDim*D
	want := base + gated + swiglu
	if got != want {
		t.Fatalf("params=%d, want %d", got, want)
	}
}
