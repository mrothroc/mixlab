package arch

import "strings"

const (
	RopeConventionAdjacentPair = "adjacent_pair"
	RopeConventionHalfRotation = "half_rotation"

	ropeConventionCodeAdjacentPair = 0
	ropeConventionCodeHalfRotation = 1
)

func normalizeRopeConvention(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", "adjacent", "adjacent_pair", "adjacent-pair":
		return RopeConventionAdjacentPair
	case "half", "half_rotation", "half-rotation":
		return RopeConventionHalfRotation
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

func ropeConventionCode(v string) int {
	if normalizeRopeConvention(v) == RopeConventionHalfRotation {
		return ropeConventionCodeHalfRotation
	}
	return ropeConventionCodeAdjacentPair
}
