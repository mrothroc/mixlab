package arch

import "strings"

const (
	PlainFFNActivationSiLU    = "silu"
	PlainFFNActivationGEGLU   = "geglu"
	PlainFFNActivationSwiGLU  = "swiglu"
	PlainFFNActivationGELU    = "gelu"
	PlainFFNActivationGELUNew = "gelu_new"
)

func normalizePlainFFNActivation(activation string) string {
	switch strings.ToLower(strings.TrimSpace(activation)) {
	case "", PlainFFNActivationSiLU:
		return PlainFFNActivationSiLU
	case PlainFFNActivationGEGLU:
		return PlainFFNActivationGEGLU
	case PlainFFNActivationSwiGLU:
		return PlainFFNActivationSwiGLU
	case PlainFFNActivationGELU:
		return PlainFFNActivationGELU
	case "gelu-new", "gelu_new":
		return PlainFFNActivationGELUNew
	default:
		return strings.ToLower(strings.TrimSpace(activation))
	}
}

func plainFFNActivationUsesGate(activation string) bool {
	switch normalizePlainFFNActivation(activation) {
	case PlainFFNActivationGEGLU, PlainFFNActivationSwiGLU:
		return true
	default:
		return false
	}
}
