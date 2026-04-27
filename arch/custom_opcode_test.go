package arch

import (
	"encoding/json"
	"testing"
)

func TestCustomBlockOuter(t *testing.T) {
	var spec BlockSpec
	if err := json.Unmarshal([]byte(`{
		"type": "custom",
		"name": "outer_block",
		"ops": [
			{
				"op": "outer",
				"inputs": ["x", "x"],
				"output": "outer_out"
			}
		]
	}`), &spec); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	prog := NewProgram(0)
	if _, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0); err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(prog.Ops))
	}
	if prog.Ops[0].Code != OpOuter {
		t.Fatalf("expected OpOuter (%d), got %d", OpOuter, prog.Ops[0].Code)
	}
}

func TestCustomBlockScan(t *testing.T) {
	var spec BlockSpec
	if err := json.Unmarshal([]byte(`{
		"type": "custom",
		"name": "scan_block",
		"weights": [
			{"name": "decay", "shape": ["D"]}
		],
		"ops": [
			{
				"op": "scan",
				"inputs": ["x", "decay"],
				"output": "scan_out",
				"params": {"B": "B", "T": "T", "D": "D"}
			}
		]
	}`), &spec); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	prog := NewProgram(0)
	wi, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0)
	if err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if wi != 1 {
		t.Fatalf("expected 1 weight consumed, got %d", wi)
	}
	if len(prog.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(prog.Ops))
	}
	op := prog.Ops[0]
	if op.Code != OpScan {
		t.Fatalf("expected OpScan (%d), got %d", OpScan, op.Code)
	}
	if len(op.IntParams) != 3 || op.IntParams[0] != 2 || op.IntParams[1] != 64 || op.IntParams[2] != 128 {
		t.Fatalf("scan int params = %v, want [2 64 128]", op.IntParams)
	}
}

func TestCustomBlockMatrixScan(t *testing.T) {
	var spec BlockSpec
	if err := json.Unmarshal([]byte(`{
		"type": "custom",
		"name": "matrix_scan_block",
		"ops": [
			{
				"op": "matrix_scan",
				"inputs": ["x", "gate"],
				"output": "scan_out",
				"params": {"B": "B", "T": "T", "Da": "H", "Db": "HD"}
			}
		]
	}`), &spec); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	prog := NewProgram(0)
	if _, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0); err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(prog.Ops))
	}
	op := prog.Ops[0]
	if op.Code != OpMatrixScan {
		t.Fatalf("expected OpMatrixScan (%d), got %d", OpMatrixScan, op.Code)
	}
	if len(op.Inputs) != 2 || op.Inputs[0] != "stream" || op.Inputs[1] != "stream_custom_matrix_scan_block_0_gate" {
		t.Fatalf("matrix scan inputs = %v", op.Inputs)
	}
	if len(op.IntParams) != 4 || op.IntParams[0] != 2 || op.IntParams[1] != 64 || op.IntParams[2] != 1 || op.IntParams[3] != 128 {
		t.Fatalf("matrix scan int params = %v, want [2 64 1 128]", op.IntParams)
	}
}

func TestCustomBlockScanTV(t *testing.T) {
	var spec BlockSpec
	if err := json.Unmarshal([]byte(`{
		"type": "custom",
		"name": "scan_tv_block",
		"ops": [
			{
				"op": "scan_tv",
				"inputs": ["x", "x"],
				"output": "scan_out",
				"params": {"B": "B", "T": "T", "D": "D"}
			}
		]
	}`), &spec); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	prog := NewProgram(0)
	if _, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0); err != nil {
		t.Fatalf("emitCustomBlockIR: %v", err)
	}
	if len(prog.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(prog.Ops))
	}
	op := prog.Ops[0]
	if op.Code != OpScanTV {
		t.Fatalf("expected OpScanTV (%d), got %d", OpScanTV, op.Code)
	}
	if len(op.IntParams) != 3 || op.IntParams[0] != 2 || op.IntParams[1] != 64 || op.IntParams[2] != 128 {
		t.Fatalf("scan_tv int params = %v, want [2 64 128]", op.IntParams)
	}
}

func TestCustomBlockSliceParamOrder(t *testing.T) {
	// Regression test for the bug fixed in arch/custom.go: the JSON-side custom
	// block builder was appending int params in order [axis, start, end, step],
	// but the C++ OP_SLICE handler expects [start, end, step, axis]. This test
	// asserts the int params land in the C++-expected order regardless of the
	// JSON key order used by the user.
	cases := []struct {
		name string
		json string
	}{
		{
			name: "natural order",
			json: `{
				"type": "custom",
				"name": "slice_block",
				"ops": [{
					"op": "slice",
					"inputs": ["x"],
					"output": "y",
					"params": {"start": 0, "end": 12, "step": 1, "axis": 1}
				}]
			}`,
		},
		{
			name: "reversed order",
			json: `{
				"type": "custom",
				"name": "slice_block",
				"ops": [{
					"op": "slice",
					"inputs": ["x"],
					"output": "y",
					"params": {"axis": 1, "step": 1, "end": 12, "start": 0}
				}]
			}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var spec BlockSpec
			if err := json.Unmarshal([]byte(tc.json), &spec); err != nil {
				t.Fatalf("json.Unmarshal: %v", err)
			}
			prog := NewProgram(0)
			if _, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0); err != nil {
				t.Fatalf("emitCustomBlockIR: %v", err)
			}
			if len(prog.Ops) != 1 {
				t.Fatalf("expected 1 op, got %d", len(prog.Ops))
			}
			op := prog.Ops[0]
			if op.Code != OpSlice {
				t.Fatalf("expected OpSlice (%d), got %d", OpSlice, op.Code)
			}
			want := []int{0, 12, 1, 1} // start=0, end=12, step=1, axis=1
			if len(op.IntParams) != 4 {
				t.Fatalf("slice int params = %v, want %v (length 4)", op.IntParams, want)
			}
			for i, w := range want {
				if op.IntParams[i] != w {
					t.Fatalf("slice int param[%d] = %d, want %d (full %v)", i, op.IntParams[i], w, op.IntParams)
				}
			}
		})
	}
}

func TestCustomBlockSliceMissingParam(t *testing.T) {
	// All four slice params are required; missing one should error rather than
	// silently produce a malformed op.
	var spec BlockSpec
	if err := json.Unmarshal([]byte(`{
		"type": "custom",
		"name": "slice_block",
		"ops": [{
			"op": "slice",
			"inputs": ["x"],
			"output": "y",
			"params": {"start": 0, "end": 12, "axis": 1}
		}]
	}`), &spec); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	prog := NewProgram(0)
	_, err := emitCustomBlockIR(prog, spec, "stream", 0, 128, 64, 2, 1024, 0)
	if err == nil {
		t.Fatal("expected error for missing step param")
	}
}
