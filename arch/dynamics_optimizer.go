package arch

// applyDynamicsStateLR leaves ordinary weights and omitted overrides untouched.
// S4D retains its separate dt/A/B selection and legacy optimizer group names.
func applyDynamicsStateLR(metas []WeightMeta, spec BlockSpec) {
	if spec.StateLR == nil {
		return
	}
	for i := range metas {
		if metas[i].Name == "A_log" || metas[i].Name == "dt_bias" {
			metas[i].OptimizerRole = "ssm_state"
			metas[i].OptimizerLR = float32(*spec.StateLR)
		}
	}
}
