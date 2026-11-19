package arch

import (
	"bytes"
	"encoding/json"
	"fmt"
)

func (s *InputAdapterSpec) UnmarshalJSON(raw []byte) error {
	type plain InputAdapterSpec
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if value, ok := fields["vector_dim"]; ok && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return fmt.Errorf("input_adapter.vector_dim must be a nonnegative integer, not null")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	return decoder.Decode((*plain)(s))
}
