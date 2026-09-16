package arch

import "fmt"

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
		return nil
	}
	if c.EffectiveNormSpec().Type == NormTypeBatchNorm {
		return fmt.Errorf("classification pooling=cls requires tokenwise normalization, not batchnorm")
	}
	if c.CharVocabSize > 0 || c.BigramVocabSize > 0 || c.TrigramVocabSize > 0 || c.SmearEmbeddings || c.RCEquivarianceEnabled() {
		return fmt.Errorf("classification pooling=cls does not yet support embedding side channels or RC equivariance")
	}
	attention := false
	for _, b := range c.Blocks {
		switch blockTypeKey(b) {
		case "plain":
			mask := normalizeAttentionMask(b.AttentionMask)
			if mask != AttentionMaskBidirectional && mask != AttentionMaskNone {
				return fmt.Errorf("classification pooling=cls requires bidirectional plain attention")
			}
			attention = attention || !b.SkipAttention
		case "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("classification pooling=cls does not yet support block %q", b.Type)
		}
	}
	if !attention {
		return fmt.Errorf("classification pooling=cls requires at least one bidirectional plain block")
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
	return "cls_input", T + 1
}
