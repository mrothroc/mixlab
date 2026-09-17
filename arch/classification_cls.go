package arch

import (
	"fmt"
	"strings"
)

const clsInsertionInput = "cls_insert_positions"

func (c *ArchConfig) EffectiveCLSPosition() string {
	if !c.CLSPoolingEnabled() {
		return ""
	}
	position := strings.ToLower(strings.TrimSpace(c.Training.Classification.CLSPosition))
	if position == "" {
		return "head"
	}
	return position
}

// ClassTokenInitMode marks the learned classification token, which draws from
// Normal(0, 0.02) unless the model-wide "normal" weight-init policy supplies a
// standard deviation of its own.
const ClassTokenInitMode = "class_token_normal"

// CLSPoolingEnabled leaves all public batch dimensions in input-token units.
func (c *ArchConfig) CLSPoolingEnabled() bool {
	return c.ClassificationEnabled() && c.EffectiveClassificationPooling() == ClassificationPoolingCLS
}

func (c *ArchConfig) EffectiveBackboneSeqLen() int {
	if c.CLSPoolingEnabled() {
		return c.SeqLen + 1
	}
	return c.SeqLen
}

func validateCLSPooling(c *ArchConfig) error {
	if !c.CLSPoolingEnabled() {
		if c.Training.Classification.CLSPosition != "" {
			return fmt.Errorf("classification cls_position requires pooling=cls")
		}
		return nil
	}
	switch c.EffectiveCLSPosition() {
	case "head", "tail":
	case "middle":
		if c.Training.LengthBucketsChangeShape(c.SeqLen) {
			return fmt.Errorf("classification cls_position=middle requires fixed full-length records, not length bucketing")
		}
	default:
		return fmt.Errorf("classification cls_position must be head, middle, or tail")
	}
	if c.EffectiveNormSpec().Type == NormTypeBatchNorm {
		return fmt.Errorf("classification pooling=cls requires tokenwise normalization, not batchnorm")
	}
	if c.CharVocabSize > 0 || c.BigramVocabSize > 0 || c.TrigramVocabSize > 0 || c.SmearEmbeddings || c.RCEquivarianceEnabled() {
		return fmt.Errorf("classification pooling=cls does not yet support embedding side channels or RC equivariance")
	}
	mixer := false
	for _, b := range c.Blocks {
		switch blockTypeKey(b) {
		case "plain":
			mask := normalizeAttentionMask(b.AttentionMask)
			if mask != AttentionMaskBidirectional && mask != AttentionMaskNone {
				return fmt.Errorf("classification pooling=cls requires bidirectional plain attention")
			}
			mixer = mixer || !b.SkipAttention
		case "mamba3-canonical", "gated_deltanet", "s4d":
			if !b.Bidirectional {
				return fmt.Errorf("classification pooling=cls requires bidirectional %s; unidirectional mixers are unsupported", b.Type)
			}
			mixer = true
		case "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("classification pooling=cls does not yet support block %q", b.Type)
		}
	}
	if !mixer {
		return fmt.Errorf("classification pooling=cls requires at least one bidirectional token mixer")
	}
	if c.EffectiveMaxPositions() < c.EffectiveBackboneSeqLen() {
		return fmt.Errorf("classification pooling=cls requires max_positions >= seq_len + 1")
	}
	return nil
}

func prependCLSInputIR(p *Program, state, weight string, B, T, D int) (string, int) {
	if weight == "" {
		return state, T
	}
	p.Reshape(weight, []int{1, 1, D}, "cls_token_11d")
	p.Full([]int{B, 1, 1}, 1, "cls_broadcast")
	p.Mul("cls_token_11d", "cls_broadcast", "cls_token_b1d")
	p.Concat("cls_token_b1d", state, 1, "cls_input")
	reorderCLSIR(p, "cls_input", B, T+1, D)
	return "cls_input", T + 1
}

// All CLS-bearing tensors use the same per-row permutation of [CLS, content].
func reorderCLSIR(p *Program, name string, B, T, D int) {
	if programDeclaresInput(p, clsInsertionInput) {
		p.Reshape(name, []int{B * T, D}, name+"_flat")
		p.Embed(name+"_flat", clsInsertionInput, name)
	}
}

func expandCLSValidMaskIR(p *Program, B, T int) {
	if !programDeclaresInput(p, sequenceValidMaskInput) {
		return
	}
	p.Full([]int{B, 1}, 1, "cls_valid")
	p.Concat("cls_valid", sequenceValidMaskInput, 1, sequenceValidMaskInput)
	p.Reshape(sequenceValidMaskInput, []int{B, T, 1}, "cls_valid_3d")
	reorderCLSIR(p, "cls_valid_3d", B, T, 1)
	p.Reshape("cls_valid_3d", []int{B, T}, sequenceValidMaskInput)
}
