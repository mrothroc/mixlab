// Package gridpoints defines the shared grid point-column declaration, without
// depending on model configuration, data loading or augmentation execution.
package gridpoints

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Columns contains ordered row,column pairs. Nil means no declaration.
type Columns [][2]int

func (c *Columns) UnmarshalJSON(raw []byte) error {
	var pairs [][]json.RawMessage
	if err := json.Unmarshal(raw, &pairs); err != nil {
		return fmt.Errorf("vector_points: %w", err)
	}
	if len(pairs) == 0 {
		return fmt.Errorf("vector_points must be a nonempty list of row,column pairs")
	}
	parsed := make(Columns, len(pairs))
	for i, pair := range pairs {
		if len(pair) != 2 {
			return fmt.Errorf("vector_points[%d] must contain exactly two column indices", i)
		}
		for j, value := range pair {
			if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return fmt.Errorf("vector_points[%d][%d] must be an integer", i, j)
			}
			if err := json.Unmarshal(value, &parsed[i][j]); err != nil {
				return fmt.Errorf("vector_points[%d][%d] must be an integer: %w", i, j, err)
			}
		}
	}
	*c = parsed
	return nil
}

func (c Columns) Validate(vectorDim int) error {
	if c == nil {
		return nil
	}
	if len(c) == 0 || vectorDim <= 0 {
		return fmt.Errorf("vector_points requires a nonempty declaration and positive vector_dim")
	}
	seen := map[int]bool{}
	for i, pair := range c {
		for _, column := range pair {
			if column < 0 || column >= vectorDim {
				return fmt.Errorf("vector_points[%d] column %d outside vector_dim %d", i, column, vectorDim)
			}
			if seen[column] {
				return fmt.Errorf("vector_points column %d appears more than once", column)
			}
			seen[column] = true
		}
	}
	return nil
}

func (c Columns) Contains(row, col int) bool {
	for _, pair := range c {
		if pair == [2]int{row, col} {
			return true
		}
	}
	return false
}
