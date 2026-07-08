package arch

import "encoding/json"

// TrainingSpec custom JSON (un)marshaling. UnmarshalJSON records which fields
// were present so defaulting can distinguish an explicit zero from an omitted
// field; MarshalJSON drops never-set zero-valued decay fields so configs round
// trip without spurious keys.

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
	return json.Marshal(fields)
}
