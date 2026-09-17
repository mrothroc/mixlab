package arch

import "fmt"

// AttentionQKVBiasEnabled resolves split controls without mutating the legacy
// shorthand, so serialization never emits both spellings of the same setting.
func (b BlockSpec) AttentionQKVBiasEnabled() bool {
	if b.AttnQKVBias != nil {
		return *b.AttnQKVBias
	}
	return b.AttnBias
}

// AttentionOutBiasEnabled resolves the output-projection bias independently.
func (b BlockSpec) AttentionOutBiasEnabled() bool {
	if b.AttnOutBias != nil {
		return *b.AttnOutBias
	}
	return b.AttnBias
}

type attentionBiasOptions struct {
	QKV bool
	Out bool
}

func (b BlockSpec) attentionBiases() attentionBiasOptions {
	return attentionBiasOptions{QKV: b.AttentionQKVBiasEnabled(), Out: b.AttentionOutBiasEnabled()}
}

func (o attentionBiasOptions) weightCount(reuseKV bool) int {
	n := 0
	if o.QKV {
		n++
		if !reuseKV {
			n += 2
		}
	}
	if o.Out {
		n++
	}
	return n
}

func (b BlockSpec) validateAttentionBias() error {
	for _, field := range []struct {
		name  string
		value *bool
	}{{"attn_qkv_bias", b.AttnQKVBias}, {"attn_out_bias", b.AttnOutBias}} {
		if field.value == nil {
			continue
		}
		if b.attnBiasSet || b.AttnBias {
			return fmt.Errorf("attn_bias cannot be combined with %s", field.name)
		}
		if blockTypeKey(b) != "plain" {
			return fmt.Errorf("%s is supported only for type=plain", field.name)
		}
	}
	return nil
}
