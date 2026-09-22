package arch

import "encoding/json"

// TrainingSpec custom JSON (un)marshaling. UnmarshalJSON records which fields
// were present so defaulting can distinguish an explicit zero from an omitted
// field; MarshalJSON preserves explicit LR floors and drops never-set
// zero-valued decay fields so configs round trip without spurious keys.

func (t *TrainingSpec) UnmarshalJSON(data []byte) error {
	type alias TrainingSpec
	var raw alias
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*t = TrainingSpec(raw)

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	_, t.mlmMaskProbSet = fields["mlm_mask_prob"]
	_, t.mlmMaskTokenIDSet = fields["mlm_mask_token_id"]
	_, maskProbSet := fields["mlm_mask_token_prob"]
	_, randomProbSet := fields["mlm_random_token_prob"]
	_, keptProbSet := fields["mlm_kept_unchanged_prob"]
	t.mlmReplacementProbSet = maskProbSet || randomProbSet || keptProbSet
	_, t.hybridCLMFractionSet = fields["hybrid_clm_fraction"]
	_, t.attentionSegmentBoundaryTokenIDSet = fields["attention_segment_boundary_token_id"]
	_, t.warmupStepsSet = fields["warmup_steps"]
	_, t.warmupRatioSet = fields["warmup_ratio"]
	_, t.holdStepsSet = fields["hold_steps"]
	// Unlike the presence flags above, an explicit null counts as omitted: the
	// floor's explicit zero means "decay to zero", and null must not select it.
	t.minLRFractionSet = len(fields["min_lr_fraction"]) > 0 && string(fields["min_lr_fraction"]) != "null"
	t.lrSet = jsonFieldPresent(fields, "lr")
	t.embedLRSet = jsonFieldPresent(fields, "embed_lr")
	t.matrixLRSet = jsonFieldPresent(fields, "matrix_lr")
	t.scalarLRSet = jsonFieldPresent(fields, "scalar_lr")
	t.headLRSet = jsonFieldPresent(fields, "head_lr")
	_, t.weightDecaySet = fields["weight_decay"]
	_, t.embedWeightDecaySet = fields["embed_weight_decay"]
	_, t.matrixWeightDecaySet = fields["matrix_weight_decay"]
	_, t.scalarWeightDecaySet = fields["scalar_weight_decay"]
	_, t.headWeightDecaySet = fields["head_weight_decay"]
	_, t.lambBeta1Set = fields["lamb_beta1"]
	_, t.lambBeta2Set = fields["lamb_beta2"]
	_, t.lambEpsSet = fields["lamb_eps"]
	_, t.lambTrustRatioCapSet = fields["lamb_trust_ratio_cap"]
	_, t.swaDecaySet = fields["swa_decay"]
	_, t.swaIntervalSet = fields["swa_interval"]
	_, t.batchTokensSet = fields["batch_tokens"]
	_, t.batchSizeSet = fields["batch_size"]
	_, t.valEveryStepsSet = fields["val_every_steps"]
	_, t.valExamplesSet = fields["val_examples"]
	return nil
}

func (t TrainingSpec) MarshalJSON() ([]byte, error) {
	type alias TrainingSpec
	data, err := json.Marshal(alias(t))
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	if !t.weightDecaySet && t.WeightDecay == 0 {
		delete(fields, "weight_decay")
	}
	for _, field := range []struct {
		name  string
		set   bool
		value float64
	}{
		{"lr", t.lrSet, t.LR},
		{"embed_lr", t.embedLRSet, float64(t.EmbedLR)},
		{"matrix_lr", t.matrixLRSet, float64(t.MatrixLR)},
		{"scalar_lr", t.scalarLRSet, float64(t.ScalarLR)},
		{"head_lr", t.headLRSet, float64(t.HeadLR)},
	} {
		if field.value == 0 {
			if field.set {
				fields[field.name] = json.RawMessage("0")
			} else {
				delete(fields, field.name)
			}
		}
	}
	if t.minLRFractionSet && t.MinLRFraction == 0 {
		fields["min_lr_fraction"] = json.RawMessage("0")
	}
	if !t.embedWeightDecaySet && t.EmbedWeightDecay == 0 {
		delete(fields, "embed_weight_decay")
	}
	if !t.matrixWeightDecaySet && t.MatrixWeightDecay == 0 {
		delete(fields, "matrix_weight_decay")
	}
	if !t.scalarWeightDecaySet && t.ScalarWeightDecay == 0 {
		delete(fields, "scalar_weight_decay")
	}
	if !t.headWeightDecaySet && t.HeadWeightDecay == 0 {
		delete(fields, "head_weight_decay")
	}
	if !t.lambTrustRatioCapSet && t.LAMBTrustRatioCap == 0 {
		delete(fields, "lamb_trust_ratio_cap")
	}
	if (!t.batchTokensSet && t.BatchTokens == 0) || t.batchTokensDerivedFromBatchSize {
		delete(fields, "batch_tokens")
	}
	return json.Marshal(fields)
}

func jsonFieldPresent(fields map[string]json.RawMessage, name string) bool {
	return len(fields[name]) > 0 && string(fields[name]) != "null"
}
