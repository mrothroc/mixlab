package arch

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// opNameToCode maps JSON op names to IR op codes for custom blocks.
var opNameToCode = map[string]int{
	"matmul":      OpMatMul,
	"add":         OpAdd,
	"sub":         OpSub,
	"mul":         OpMul,
	"scalar_mul":  OpScalarMul,
	"div":         OpDiv,
	"scan":        OpScan,
	"matrix_scan": OpMatrixScan,
	"scan_tv":     OpScanTV,
	"embed":       OpEmbed,
	"outer":       OpOuter,
	"square":      OpSquare,
	"exp":         OpExp,
	"sigmoid":     OpSigmoid,
	"silu":        OpSiLU,
	"gelu":        OpGELU,
	"relu":        OpReLU,
	"tanh":        OpTanh,
	"softmax":     OpSoftmax,
	"reshape":     OpReshape,
	"transpose":   OpTranspose,
	"slice":       OpSlice,
	"concat":      OpConcat,
	"rmsnorm":     OpRMSNorm,
	"rms_norm":    OpRMSNorm,
	"rope":        OpRoPE,
}

// resolveShapeSymbol resolves a slice of symbolic shape dimensions to concrete
// integer dimensions given the model parameters.
//
// Supported symbols: D, H, HD (D/H), FFN (2.67*D), 2D, 3D, 4D, 8D, T, B, V,
// BT, T/2, <float>D (e.g. "1.5D"), or literal integers.
func resolveShapeSymbol(syms []string, D, H, T, B, V int) ([]int, error) {
	if len(syms) == 0 {
		return nil, fmt.Errorf("shape is required")
	}
	out := make([]int, len(syms))
	for i, raw := range syms {
		s := strings.TrimSpace(raw)
		switch s {
		case "D":
			out[i] = D
		case "H":
			out[i] = H
		case "HD":
			if H <= 0 || D%H != 0 {
				return nil, fmt.Errorf("invalid HD with D=%d H=%d", D, H)
			}
			out[i] = D / H
		case "T":
			out[i] = T
		case "B":
			out[i] = B
		case "V":
			out[i] = V
		case "BT":
			out[i] = B * T
		case "2D":
			out[i] = 2 * D
		case "3D":
			out[i] = 3 * D
		case "4D":
			out[i] = 4 * D
		case "8D":
			out[i] = 8 * D
		case "T/2":
			out[i] = T / 2
		case "FFN":
			out[i] = int(2.67 * float64(D))
		default:
			// Try float multiplier pattern: "2.67D", "1.5D", etc.
			if strings.HasSuffix(s, "D") && len(s) > 1 {
				mult, err := strconv.ParseFloat(s[:len(s)-1], 64)
				if err == nil {
					out[i] = int(mult * float64(D))
					continue
				}
			}
			n, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("unknown shape symbol %q (supported: D, H, HD, T, B, V, BT, 2D, 3D, 4D, 8D, FFN, T/2, <number>, or <float>D like 2.67D)", s)
			}
			out[i] = n
		}
		if out[i] <= 0 {
			return nil, fmt.Errorf("invalid non-positive resolved shape dim %d for %q", out[i], s)
		}
	}
	return out, nil
}

// emitCustomBlockIR emits a user-defined block specified entirely in JSON.
// The block declares its weights and a sequence of ops. Variable "x" maps to
// the current stream hidden state; weight names map to their IR weight indices;
// all other names become prefixed temporaries.
func emitCustomBlockIR(
	prog *Program,
	spec BlockSpec,
	stream string,
	wi, D, T, B, V, idx int,
) (int, error) {
	opName := strings.ToLower(strings.TrimSpace(spec.Name))
	if opName == "" {
		opName = "custom"
	}
	heads := spec.Heads
	if heads <= 0 {
		heads = 1
	}
	prefix := tmpName(stream+"_custom_"+opName, idx) + "_"

	// Map declared weight names to IR weight indices.
	weightMap := make(map[string]string, len(spec.Weights))
	for i, ws := range spec.Weights {
		if strings.TrimSpace(ws.Name) == "" {
			return wi, fmt.Errorf("custom block %q weight[%d] missing name", spec.Name, i)
		}
		if _, err := resolveShapeSymbol(ws.Shape, D, heads, T, B, V); err != nil {
			return wi, fmt.Errorf("custom block %q weight %q shape: %w", spec.Name, ws.Name, err)
		}
		weightMap[ws.Name] = weightName(wi)
		wi++
	}

	// resolveName maps a user-facing name to an IR variable name.
	resolveName := func(name string) string {
		if name == "x" {
			return stream
		}
		if w, ok := weightMap[name]; ok {
			return w
		}
		return prefix + name
	}

	// buildOpParams extracts float and int params from the op's Params map.
	buildOpParams := func(op OpSpec) ([]float32, []int, error) {
		var (
			floatParams []float32
			intParams   []int
		)
		if op.Params == nil {
			return floatParams, intParams, nil
		}
		p := op.Params

		if v, ok := p["shape"]; ok {
			ss, err := valueToStringSlice(v)
			if err != nil {
				return nil, nil, fmt.Errorf("shape param: %w", err)
			}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("shape param: %w", err)
			}
			intParams = append(intParams, res...)
		}
		if v, ok := p["axes"]; ok {
			axes, err := valueToIntSlice(v)
			if err != nil {
				return nil, nil, fmt.Errorf("axes param: %w", err)
			}
			intParams = append(intParams, axes...)
		}
		if v, ok := p["axis"]; ok {
			n, err := valueToInt(v)
			if err != nil {
				return nil, nil, fmt.Errorf("axis param: %w", err)
			}
			intParams = append(intParams, n)
		}
		if v, ok := p["start"]; ok {
			n, err := valueToInt(v)
			if err != nil {
				return nil, nil, fmt.Errorf("start param: %w", err)
			}
			intParams = append(intParams, n)
		}
		if v, ok := p["end"]; ok {
			n, err := valueToInt(v)
			if err != nil {
				return nil, nil, fmt.Errorf("end param: %w", err)
			}
			intParams = append(intParams, n)
		}
		if v, ok := p["step"]; ok {
			n, err := valueToInt(v)
			if err != nil {
				return nil, nil, fmt.Errorf("step param: %w", err)
			}
			intParams = append(intParams, n)
		}
		if v, ok := p["head_dim"]; ok {
			n, err := valueToInt(v)
			if err != nil {
				return nil, nil, fmt.Errorf("head_dim param: %w", err)
			}
			intParams = append(intParams, n)
		}
		if v, ok := p["B"]; ok {
			ss := []string{fmt.Sprint(v)}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("b param: %w", err)
			}
			intParams = append(intParams, res[0])
		}
		if v, ok := p["T"]; ok {
			ss := []string{fmt.Sprint(v)}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("t param: %w", err)
			}
			intParams = append(intParams, res[0])
		}
		if v, ok := p["D"]; ok {
			ss := []string{fmt.Sprint(v)}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("d param: %w", err)
			}
			intParams = append(intParams, res[0])
		}
		if v, ok := p["Da"]; ok {
			ss := []string{fmt.Sprint(v)}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("da param: %w", err)
			}
			intParams = append(intParams, res[0])
		}
		if v, ok := p["Db"]; ok {
			ss := []string{fmt.Sprint(v)}
			res, err := resolveShapeSymbol(ss, D, heads, T, B, V)
			if err != nil {
				return nil, nil, fmt.Errorf("db param: %w", err)
			}
			intParams = append(intParams, res[0])
		}
		if v, ok := p["scalar"]; ok {
			f, err := valueToFloat(v)
			if err != nil {
				return nil, nil, fmt.Errorf("scalar param: %w", err)
			}
			floatParams = append(floatParams, f)
		}
		if v, ok := p["eps"]; ok {
			f, err := valueToFloat(v)
			if err != nil {
				return nil, nil, fmt.Errorf("eps param: %w", err)
			}
			floatParams = append(floatParams, f)
		}
		if v, ok := p["base"]; ok {
			f, err := valueToFloat(v)
			if err != nil {
				return nil, nil, fmt.Errorf("base param: %w", err)
			}
			floatParams = append(floatParams, f)
		}
		return floatParams, intParams, nil
	}

	for i, op := range spec.Ops {
		opKey := strings.ToLower(strings.TrimSpace(op.Op))
		code, ok := opNameToCode[opKey]
		if !ok {
			return wi, fmt.Errorf("custom block %q op[%d]: unknown op %q", spec.Name, i, op.Op)
		}
		inputs := make([]string, len(op.Inputs))
		for ii, in := range op.Inputs {
			inputs[ii] = resolveName(in)
		}
		var outputs []string
		switch {
		case len(op.Outputs) > 0:
			outputs = make([]string, len(op.Outputs))
			for oi, out := range op.Outputs {
				outputs[oi] = resolveName(out)
			}
		case strings.TrimSpace(op.Output) != "":
			outputs = []string{resolveName(op.Output)}
		default:
			return wi, fmt.Errorf("custom block %q op[%d]: missing output(s)", spec.Name, i)
		}
		floatParams, intParams, err := buildOpParams(op)
		if err != nil {
			return wi, fmt.Errorf("custom block %q op[%d] params: %w", spec.Name, i, err)
		}
		prog.AddOp(code, inputs, outputs, floatParams, intParams)
	}

	return wi, nil
}

// --- param conversion helpers ---

func valueToInt(v interface{}) (int, error) {
	switch t := v.(type) {
	case float64:
		return int(t), nil
	case float32:
		return int(t), nil
	case int:
		return t, nil
	case int64:
		return int(t), nil
	case json.Number:
		n, err := t.Int64()
		return int(n), err
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(t))
		return n, err
	default:
		return 0, fmt.Errorf("unsupported int value type %T", v)
	}
}

func valueToFloat(v interface{}) (float32, error) {
	switch t := v.(type) {
	case float64:
		return float32(t), nil
	case float32:
		return t, nil
	case int:
		return float32(t), nil
	case json.Number:
		f, err := t.Float64()
		return float32(f), err
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 32)
		return float32(f), err
	default:
		return 0, fmt.Errorf("unsupported float value type %T", v)
	}
}

func valueToIntSlice(v interface{}) ([]int, error) {
	arr, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", v)
	}
	out := make([]int, len(arr))
	for i, elem := range arr {
		n, err := valueToInt(elem)
		if err != nil {
			return nil, fmt.Errorf("element [%d]: %w", i, err)
		}
		out[i] = n
	}
	return out, nil
}

func valueToStringSlice(v interface{}) ([]string, error) {
	arr, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("expected array, got %T", v)
	}
	out := make([]string, len(arr))
	for i, elem := range arr {
		s, ok := elem.(string)
		if !ok {
			return nil, fmt.Errorf("element [%d]: expected string, got %T", i, elem)
		}
		out[i] = s
	}
	return out, nil
}
