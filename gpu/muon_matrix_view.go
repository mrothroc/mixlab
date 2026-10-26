package gpu

import "fmt"

// MuonMatrixView changes only the optimizer's view, not parameter storage.
// Values are shared with mlx_bridge.h and muon_matrix_view.h.
type MuonMatrixView int

const (
	MuonMatrixIdentity MuonMatrixView = iota
	MuonMatrixOHWI
	MuonMatrixIHWO
)

func (v MuonMatrixView) Shape(shape []int) ([]int, error) {
	if v == MuonMatrixIdentity {
		if len(shape) != 2 {
			return nil, fmt.Errorf("muon requires rank-2 weights or an explicit convolution matrix view")
		}
		return append([]int(nil), shape...), nil
	}
	if len(shape) != 4 || (v != MuonMatrixOHWI && v != MuonMatrixIHWO) {
		return nil, fmt.Errorf("invalid Muon matrix view %d for shape %v", v, shape)
	}
	for _, n := range shape {
		if n <= 0 {
			return nil, fmt.Errorf("invalid Muon kernel dimensions %v", shape)
		}
	}
	if v == MuonMatrixOHWI {
		return []int{shape[0], shape[1] * shape[2] * shape[3]}, nil
	}
	return []int{shape[3], shape[1] * shape[2] * shape[0]}, nil
}
