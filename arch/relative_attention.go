package arch

import "strings"

const (
	RelativeAttentionNone          = "none"
	RelativeAttentionDebertaP2CC2P = "deberta_p2c_c2p"

	defaultRelativeAttentionWindow = 128
)

func normalizeRelativeAttention(raw string) string {
	return strings.ToLower(strings.TrimSpace(raw))
}

func relativeAttentionEnabled(spec BlockSpec) bool {
	return normalizeRelativeAttention(spec.RelativeAttention) == RelativeAttentionDebertaP2CC2P
}

func effectiveRelativeAttentionWindow(spec BlockSpec) int {
	if !relativeAttentionEnabled(spec) {
		return 0
	}
	if spec.RelativeAttentionWindow > 0 {
		return spec.RelativeAttentionWindow
	}
	return defaultRelativeAttentionWindow
}
