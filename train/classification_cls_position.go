package train

import "fmt"

// clsBatchPositions maps [CLS, content...] to insertion order. Readout positions
// index the flattened backbone, not the raw content-token tensor.
func clsBatchPositions(cfg *ArchConfig, mask []float32, seqLen int) ([]int32, []int32, error) {
	mode := cfg.EffectiveCLSPosition()
	if mode == "" || mode == "head" {
		return nil, nil, nil
	}
	rows := len(mask) / seqLen
	indices := make([]int32, rows*(seqLen+1))
	positions := make([]int32, rows)
	for row := 0; row < rows; row++ {
		length := 0
		for i, valid := range mask[row*seqLen : (row+1)*seqLen] {
			if valid != 0 && valid != 1 {
				return nil, nil, fmt.Errorf("cls_position=%s requires a binary valid mask", mode)
			}
			if valid == 1 {
				if length != i {
					return nil, nil, fmt.Errorf("cls_position=%s requires right-padded valid prefixes", mode)
				}
				length++
			}
		}
		if length == 0 {
			return nil, nil, fmt.Errorf("cls_position=%s requires non-empty records", mode)
		}
		index := length
		if mode == "middle" {
			if length != cfg.SeqLen || seqLen != cfg.SeqLen {
				return nil, nil, fmt.Errorf("cls_position=middle requires fixed full-length records at seq_len=%d; row %d has length %d", cfg.SeqLen, row, length)
			}
			index = length / 2
		}
		positions[row] = int32(row*(seqLen+1) + index)
		for i := 0; i <= seqLen; i++ {
			source := i
			if i < index {
				source = i + 1
			} else if i == index {
				source = 0
			}
			indices[row*(seqLen+1)+i] = int32(row*(seqLen+1) + source)
		}
	}
	return indices, positions, nil
}
