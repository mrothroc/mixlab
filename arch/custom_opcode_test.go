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
