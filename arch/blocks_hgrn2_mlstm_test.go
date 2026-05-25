package arch

import (
	"strings"
	"testing"
)

func TestBlockWeightShapes_HGRN2DefaultsAndDState(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "hgrn2", Heads: 4}, 64, 16, 1, 128, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes default: %v", err)
	}
	wantDefault := map[string][]int{
		"norm_scale":   {64},
		"w_v":          {64, 64},
		"w_q":          {64, 64},
		"w_f":          {64, 64},
		"o_norm_scale": {16},
		"wo":           {64, 64},
	}
	assertWeightShapes(t, metas, wantDefault)

	metas, err = blockWeightShapes(BlockSpec{Type: "hgrn2", Heads: 4, DState: 24}, 64, 16, 1, 128, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes d_state: %v", err)
	}
	wantDState := map[string][]int{
		"norm_scale":   {64},
		"w_v":          {64, 64},
		"w_q":          {64, 96},
		"w_f":          {64, 96},
		"o_norm_scale": {16},
		"wo":           {64, 64},
	}
	assertWeightShapes(t, metas, wantDState)
}

func TestBlockWeightShapes_MLSTM(t *testing.T) {
	metas, err := blockWeightShapes(BlockSpec{Type: "mlstm", Heads: 3, DK: 8, DV: 10}, 48, 16, 1, 128, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	want := map[string][]int{
		"norm_scale":   {48},
		"wq":           {48, 24},
		"wk":           {48, 24},
		"wv":           {48, 30},
		"w_i":          {48, 3},
		"b_i":          {3},
		"w_f":          {48, 3},
		"b_f":          {3},
		"o_norm_scale": {10},
		"w_out_gate":   {48, 30},
		"wo":           {30, 48},
	}
	assertWeightShapes(t, metas, want)
}

func TestEmitHGRN2IR_UsesHGRN2Scan(t *testing.T) {
	p := NewProgram(6)
	wi, err := emitBlockIR(p, BlockSpec{Type: "hgrn2", Heads: 4, DState: 12}, "x", 0, 64, 16, 2, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 6 {
		t.Fatalf("wi=%d, want 6", wi)
	}
	if n := countOps(p, OpHGRN2Scan); n != 1 {
		t.Fatalf("HGRN2Scan ops=%d, want 1", n)
	}
	if n := countOps(p, OpMLSTMScan); n != 0 {
		t.Fatalf("MLSTMScan ops=%d, want 0", n)
	}
	for _, op := range p.Ops {
		if op.Code == OpHGRN2Scan {
			want := []int{2, 16, 4, 12, 16}
			if !intSlicesEqual(op.IntParams, want) {
				t.Fatalf("HGRN2Scan int params=%v, want %v", op.IntParams, want)
			}
			return
		}
	}
	t.Fatal("missing HGRN2Scan op")
}

func TestEmitMLSTMIR_UsesMLSTMScan(t *testing.T) {
	p := NewProgram(11)
	wi, err := emitBlockIR(p, BlockSpec{Type: "mlstm", Heads: 3, DK: 8, DV: 10}, "x", 0, 48, 16, 2, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 11 {
		t.Fatalf("wi=%d, want 11", wi)
	}
	if n := countOps(p, OpMLSTMScan); n != 1 {
		t.Fatalf("MLSTMScan ops=%d, want 1", n)
	}
	if n := countOps(p, OpHGRN2Scan); n != 0 {
		t.Fatalf("HGRN2Scan ops=%d, want 0", n)
	}
	for _, op := range p.Ops {
		if op.Code == OpMLSTMScan {
			want := []int{2, 16, 3, 8, 10}
			if !intSlicesEqual(op.IntParams, want) {
				t.Fatalf("MLSTMScan int params=%v, want %v", op.IntParams, want)
			}
			return
		}
	}
	t.Fatal("missing MLSTMScan op")
}

func TestParseArchConfig_HGRN2MLSTMValidation(t *testing.T) {
	valid := []byte(`{
		"name": "hgrn2_mlstm_valid",
		"model_dim": 48,
		"vocab_size": 128,
		"seq_len": 16,
		"blocks": [
			{"type": "hgrn2", "heads": 3, "d_state": 12},
			{"type": "swiglu"},
			{"type": "mlstm", "heads": 3, "d_k": 8, "d_v": 10},
			{"type": "geglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 16}
	}`)
	if _, err := ParseArchConfig(valid, "valid"); err != nil {
		t.Fatalf("ParseArchConfig(valid): %v", err)
	}

	cases := []struct {
		name string
		json string
		want string
	}{
		{
			name: "hgrn2 heads",
			json: `{"name":"bad","model_dim":48,"vocab_size":128,"seq_len":16,"blocks":[{"type":"hgrn2","heads":0}],"training":{"batch_tokens":16}}`,
			want: "type=hgrn2 requires heads > 0",
		},
		{
			name: "hgrn2 divisibility",
			json: `{"name":"bad","model_dim":50,"vocab_size":128,"seq_len":16,"blocks":[{"type":"hgrn2","heads":3}],"training":{"batch_tokens":16}}`,
			want: "model_dim=50 divisible by heads=3",
		},
		{
			name: "mlstm dk",
			json: `{"name":"bad","model_dim":48,"vocab_size":128,"seq_len":16,"blocks":[{"type":"mlstm","heads":3,"d_v":8}],"training":{"batch_tokens":16}}`,
			want: "type=mlstm requires d_k > 0",
		},
		{
			name: "mlstm dv",
			json: `{"name":"bad","model_dim":48,"vocab_size":128,"seq_len":16,"blocks":[{"type":"mlstm","heads":3,"d_k":8}],"training":{"batch_tokens":16}}`,
			want: "type=mlstm requires d_v > 0",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(tc.json), tc.name)
			if err == nil {
				t.Fatal("expected parse error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%q, want substring %q", err.Error(), tc.want)
			}
		})
	}
}

func TestParameterCountsFromConfig_HGRN2MLSTM(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "hgrn2_mlstm_count",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"blocks": [
			{"type": "hgrn2", "heads": 4, "d_state": 6},
			{"type": "mlstm", "heads": 4, "d_k": 5, "d_v": 7}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}
	}`), "count")
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
		D  = int64(32)
		V  = int64(64)
		H  = int64(4)
		DS = int64(6)
		DK = int64(5)
		DV = int64(7)
		HD = D / H
	)
	base := V*D + D*V + D
	hgrn2 := D + D*D + 2*D*(H*DS) + HD + D*D
	mlstm := D + 2*D*(H*DK) + D*(H*DV) + D*H + H + D*H + H + DV + D*(H*DV) + (H*DV)*D
	want := base + hgrn2 + mlstm
	if got != want {
		t.Fatalf("params=%d, want %d", got, want)
	}
}

func assertWeightShapes(t *testing.T, metas []WeightMeta, want map[string][]int) {
	t.Helper()
	if len(metas) != len(want) {
		t.Fatalf("len(metas)=%d, want %d", len(metas), len(want))
	}
	for _, meta := range metas {
		ws, ok := want[meta.Name]
		if !ok {
			t.Fatalf("unexpected weight %q", meta.Name)
		}
		if !intSlicesEqual(meta.Shape, ws) {
			t.Fatalf("%s shape=%v want %v", meta.Name, meta.Shape, ws)
		}
	}
}

func intSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
