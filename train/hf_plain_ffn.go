package train

import "strings"

func hfPlainFFNActivation(block BlockSpec) string {
	switch strings.ToLower(strings.TrimSpace(block.FFNActivation)) {
	case "", "silu":
		return "silu"
	case "geglu":
		return "geglu"
	case "swiglu":
		return "swiglu"
	default:
		return strings.ToLower(strings.TrimSpace(block.FFNActivation))
	}
}

func hfPlainFFNActivationUsesGate(block BlockSpec) bool {
	switch hfPlainFFNActivation(block) {
	case "geglu", "swiglu":
		return true
	default:
		return false
	}
}
