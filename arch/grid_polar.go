package arch

import (
	"fmt"
	"math"
	"strconv"
)

func isPolarOp(name string) bool {
	return name == "cartesian_to_polar" || name == "polar_to_cartesian"
}

// polarParameters is the single config-to-IR geometry contract. Dimensions are
// resolved at graph build time; the backend never needs a dataset to resolve them.
// IR integers: K,J,H,W,tx_row_column,tx_col_column; float: dr.
func polarParameters(op OpSpec, inputs [][]int) ([]int, float32, error) {
	if len(inputs) != 2 || len(inputs[0]) != 4 || len(inputs[1]) != 2 || inputs[0][0] != inputs[1][0] {
		return nil, 0, fmt.Errorf("polar sampling requires a rank-4 map and [B,S] vectors with matching batches")
	}
	keys := map[string]bool{"n_rays": true, "dr": true, "tx": true}
	if op.Op == "cartesian_to_polar" {
		keys["n_r"] = true
	} else {
		keys["height"], keys["width"] = true, true
	}
	if len(op.Params) != len(keys) {
		return nil, 0, fmt.Errorf("%s requires exactly %v", op.Op, keys)
	}
	for key := range op.Params {
		if !keys[key] {
			return nil, 0, fmt.Errorf("unsupported polar parameter %q", key)
		}
	}
	k, err := gridInteger(op.Params["n_rays"])
	if err != nil || k <= 0 {
		return nil, 0, fmt.Errorf("n_rays must be a positive int32")
	}
	f, err := strconv.ParseFloat(fmt.Sprint(op.Params["dr"]), 64)
	dr := float32(f)
	if err != nil || dr <= 0 || math.IsNaN(float64(dr)) || math.IsInf(float64(dr), 0) {
		return nil, 0, fmt.Errorf("dr must be finite and positive in float32")
	}
	tx, ok := op.Params["tx"].([]interface{})
	if !ok || len(tx) != 2 {
		return nil, 0, fmt.Errorf("tx must contain two distinct vector column indices in row,column order")
	}
	cols := [2]int{}
	for i, v := range tx {
		cols[i], err = gridInteger(v)
		if err != nil || cols[i] < 0 || cols[i] >= inputs[1][1] {
			return nil, 0, fmt.Errorf("tx column outside vector width %d", inputs[1][1])
		}
	}
	if cols[0] == cols[1] {
		return nil, 0, fmt.Errorf("tx columns must be distinct")
	}
	h, w, j := inputs[0][1], inputs[0][2], 0
	if op.Op == "cartesian_to_polar" {
		j, err = gridInteger(op.Params["n_r"])
		if err != nil || j <= 0 {
			return nil, 0, fmt.Errorf("n_r must be a positive int32")
		}
	} else {
		j = inputs[0][2]
		if k != inputs[0][1] {
			return nil, 0, fmt.Errorf("n_rays must equal the polar input's angular size")
		}
		h, err = gridInteger(op.Params["height"])
		if err != nil || h <= 0 {
			return nil, 0, fmt.Errorf("height must be a positive int32")
		}
		w, err = gridInteger(op.Params["width"])
		if err != nil || w <= 0 {
			return nil, 0, fmt.Errorf("width must be a positive int32")
		}
		bound := math.Ceil(math.Hypot(float64(h-1), float64(w-1))/float64(dr)) + 2
		if float64(j) < bound {
			return nil, 0, fmt.Errorf("polar radius too short: need n_r >= %.0f for %dx%d at dr=%g", bound, h, w, dr)
		}
	}
	if math.IsInf(float64(float32(j-1)*dr), 0) {
		return nil, 0, fmt.Errorf("polar radius exceeds float32")
	}
	for _, shape := range [][]int{{inputs[0][0], k, j, inputs[0][3]}, {inputs[0][0], h, w, inputs[0][3]}} {
		if _, err := gridSize(shape); err != nil {
			return nil, 0, err
		}
	}
	return []int{k, j, h, w, cols[0], cols[1]}, dr, nil
}

func polarShape(op OpSpec, inputs [][]int) ([]int, error) {
	p, _, err := polarParameters(op, inputs)
	if err != nil {
		return nil, err
	}
	if op.Op == "cartesian_to_polar" {
		return []int{inputs[0][0], p[0], p[1], inputs[0][3]}, nil
	}
	return []int{inputs[0][0], p[2], p[3], inputs[0][3]}, nil
}

// PolarInputCheck describes only runtime validation, not sampling or data IO.
type PolarInputCheck struct {
	Input     string `json:"input"`
	Height    int    `json:"height"`
	Width     int    `json:"width"`
	RowColumn int    `json:"row_column"`
	ColColumn int    `json:"col_column"`
}

func GridPolarInputChecks(p *Program) []PolarInputCheck {
	var checks []PolarInputCheck
	for _, op := range p.Ops {
		if (op.Code == OpCartesianToPolar || op.Code == OpPolarToCartesian) && len(op.IntParams) == 6 && len(op.Inputs) == 2 {
			v := op.IntParams
			checks = append(checks, PolarInputCheck{op.Inputs[1], v[2], v[3], v[4], v[5]})
		}
	}
	return checks
}

func (c PolarInputCheck) Validate(values []float32, width int, ids []string) error {
	if width <= max(c.RowColumn, c.ColColumn) || width <= 0 || len(values)%width != 0 {
		return fmt.Errorf("polar vector shape mismatch")
	}
	for b := 0; b < len(values)/width; b++ {
		r, col := values[b*width+c.RowColumn], values[b*width+c.ColColumn]
		if math.IsNaN(float64(r)) || math.IsNaN(float64(col)) || r < 0 || col < 0 || float64(r) > float64(c.Height-1) || float64(col) > float64(c.Width-1) {
			id := fmt.Sprintf("batch row %d", b)
			if b < len(ids) {
				id = fmt.Sprintf("record %q (batch row %d)", ids[b], b)
			}
			return fmt.Errorf("%s: invalid polar transmitter (%g,%g) for %dx%d map", id, r, col, c.Height, c.Width)
		}
	}
	return nil
}
