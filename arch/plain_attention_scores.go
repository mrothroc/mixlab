package arch

import (
	"fmt"
	"math"
)

func emitRoPEForConvention(prog *Program, q, k, qOut, kOut string, T, headDim, ropeDims int, convention string) {
	if normalizeRopeConvention(convention) == RopeConventionHalfRotation {
		prog.RoPEWithConvention(q, k, qOut, kOut, T, headDim, ropeDims, 10000.0, convention)
		return
	}
	prog.RoPE(q, k, qOut, kOut, T, headDim, ropeDims, 10000.0)
}

func emitDebertaRelativeProjectionIR(prog *Program, prefix, tableName, keyProjectionName, keyBiasName, queryProjectionName, queryBiasName string, H, kvH, D, headDim, relativeWindow int, sharedQKReuse bool) error {
	if relativeWindow <= 0 {
		relativeWindow = defaultRelativeAttentionWindow
	}
	if D != H*headDim {
		return fmt.Errorf("invalid relative attention dimensions D=%d H=%d head_dim=%d", D, H, headDim)
	}
	if kvH <= 0 || H%kvH != 0 {
		return fmt.Errorf("invalid relative attention GQA dimensions H=%d KV=%d", H, kvH)
	}
	relRows := 2*relativeWindow - 1
	relKeyFlat := prefix + "_rel_key_flat"
	relQueryFlat := prefix + "_rel_query_flat"
	relKeyProjected := relKeyFlat
	relQueryProjected := relQueryFlat
	relKeyH := prefix + "_rel_key_h"
	relQueryH := prefix + "_rel_query_h"
	prog.MatMul(tableName, keyProjectionName, relKeyFlat)
	prog.MatMul(tableName, queryProjectionName, relQueryFlat)
	if keyBiasName != "" {
		relKeyBiased := relKeyFlat + "_biased"
		prog.Add(relKeyFlat, keyBiasName, relKeyBiased)
		relKeyProjected = relKeyBiased
	}
	if queryBiasName != "" {
		relQueryBiased := relQueryFlat + "_biased"
		prog.Add(relQueryFlat, queryBiasName, relQueryBiased)
		relQueryProjected = relQueryBiased
	}
	if sharedQKReuse {
		relKeyKV3 := prefix + "_rel_key_kv3"
		relKeyKVH := prefix + "_rel_key_kvh"
		prog.Reshape(relKeyProjected, []int{relRows, kvH, headDim}, relKeyKV3)
		prog.Transpose(relKeyKV3, []int{1, 0, 2}, relKeyKVH)
		if kvH == H {
			prog.ScalarMul(relKeyKVH, 1.0, relKeyH)
		} else {
			groupSize := H / kvH
			relKeyExp := prefix + "_rel_key_exp"
			relKeyOnes := prefix + "_rel_key_ones"
			relKeyRep := prefix + "_rel_key_rep"
			prog.Reshape(relKeyKVH, []int{kvH, 1, relRows, headDim}, relKeyExp)
			prog.Full([]int{1, groupSize, 1, 1}, 1.0, relKeyOnes)
			prog.Mul(relKeyExp, relKeyOnes, relKeyRep)
			prog.Reshape(relKeyRep, []int{H, relRows, headDim}, relKeyH)
		}
	} else {
		relKey3 := prefix + "_rel_key3"
		prog.Reshape(relKeyProjected, []int{relRows, H, headDim}, relKey3)
		prog.Transpose(relKey3, []int{1, 0, 2}, relKeyH)
	}
	relQuery3 := prefix + "_rel_query3"
	prog.Reshape(relQueryProjected, []int{relRows, H, headDim}, relQuery3)
	prog.Transpose(relQuery3, []int{1, 0, 2}, relQueryH)
	return nil
}

func emitPlainProjectedAttentionScoresIR(prog *Program, prefix, qh, kh string, wi, B, H, kvH, D, T, headDim int, baseScale float32, qkGain float64, ropeDims int, ropeConvention, positionalEmbedding, relativeAttention string, relativeWindow int, relativeParameterization string, qWeightName, qBiasName, kWeightName, kBiasName string, sharedRel sharedRelativeAttentionPlan) (string, string, int, error) {
	relMode := normalizeRelativeAttention(relativeAttention)
	qForScores := qh
	kForScores := kh
	scoreScale := baseScale
	keyForCache := kh
	switch relMode {
	case "", RelativeAttentionNone:
		if normalizePositionalEmbedding(positionalEmbedding) == PositionalEmbeddingRope {
			qRot := prefix + "_q_rot"
			kRot := prefix + "_k_rot"
			emitRoPEForConvention(prog, qh, kh, qRot, kRot, T, headDim, ropeDims, ropeConvention)
			qForScores = qRot
			kForScores = kRot
			keyForCache = kRot
		}
	case RelativeAttentionDebertaP2CC2P:
		if relativeWindow <= 0 {
			relativeWindow = defaultRelativeAttentionWindow
		}
		switch normalizeRelativeAttentionParameterization(relativeParameterization) {
		case RelativeAttentionParamPerBlockProjections:
			if err := emitDebertaRelativeProjectionIR(prog, prefix, weightName(wi), weightName(wi+1), "", weightName(wi+2), "", H, H, D, headDim, relativeWindow, false); err != nil {
				return "", "", wi, err
			}
			wi += 3
		case RelativeAttentionParamSharedQKReuse:
			if !sharedRel.Enabled || sharedRel.WeightIndex < 0 {
				return "", "", wi, fmt.Errorf("shared_qk_reuse relative attention requires a shared relative embedding weight")
			}
			if sharedRel.Window != relativeWindow {
				return "", "", wi, fmt.Errorf("shared_qk_reuse relative attention window=%d does not match shared window=%d", relativeWindow, sharedRel.Window)
			}
			if qWeightName == "" || kWeightName == "" {
				return "", "", wi, fmt.Errorf("shared_qk_reuse relative attention requires local q/k projection weights")
			}
			relTable := weightName(sharedRel.WeightIndex)
			if sharedRel.Norm == RelativeAttentionEmbeddingNormLayerNorm {
				if sharedRel.NormIndex < 0 {
					return "", "", wi, fmt.Errorf("shared_qk_reuse relative attention embedding LayerNorm requires shared norm weights")
				}
				relNorm := prefix + "_shared_rel_norm"
				prog.LayerNorm(relTable, weightName(sharedRel.NormIndex), weightName(sharedRel.NormIndex+1), relNorm, sharedRel.NormEps)
				relTable = relNorm
			}
			if err := emitDebertaRelativeProjectionIR(prog, prefix, relTable, kWeightName, kBiasName, qWeightName, qBiasName, H, kvH, D, headDim, relativeWindow, true); err != nil {
				return "", "", wi, err
			}
		default:
			return "", "", wi, fmt.Errorf("invalid relative_attention_parameterization=%q", relativeParameterization)
		}
		scoreScale = float32(1.0 / math.Sqrt(float64(headDim*3)))
	default:
		return "", "", wi, fmt.Errorf("invalid relative_attention=%q", relativeAttention)
	}

	kt := prefix + "_kt"
	scores := prefix + "_scores"
	scaled := scores + "_scaled"
	prog.Transpose(kForScores, []int{0, 1, 3, 2}, kt)
	prog.MatMul(qForScores, kt, scores)
	prog.ScalarMul(scores, scoreScale, scaled)

	if relMode == RelativeAttentionDebertaP2CC2P {
		if relativeWindow <= 0 {
			relativeWindow = defaultRelativeAttentionWindow
		}
		relKeyH := prefix + "_rel_key_h"
		relQueryH := prefix + "_rel_query_h"
		relBias := prefix + "_deberta_rel_bias"
		relBiasScaled := relBias + "_scaled"
		scoresWithRel := prefix + "_scores_deberta"
		prog.DebertaRelativeBias(qh, kh, relKeyH, relQueryH, relBias, B, T, H, headDim, relativeWindow)
		prog.ScalarMul(relBias, scoreScale, relBiasScaled)
		prog.Add(scaled, relBiasScaled, scoresWithRel)
		scaled = scoresWithRel
	}

	if qkGain > 0 {
		gain := scores + "_qk_gain"
		gained := scores + "_gained"
		prog.Reshape(weightName(wi), []int{1, H, 1, 1}, gain)
		wi++
		prog.Mul(scaled, gain, gained)
		scaled = gained
	}
	return scaled, keyForCache, wi, nil
}
