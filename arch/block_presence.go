package arch

import "fmt"

// validateBlockPresence keeps zero-depth models limited to explicit external
// representation probes. Token-embedding and language-model zero-depth paths
// need separate semantics and are intentionally not enabled here.
func validateBlockPresence(cfg *ArchConfig, source string) error {
	if cfg == nil || len(cfg.Blocks) > 0 {
		return nil
	}
	objective := cfg.Training.EffectiveObjective()
	if objective != ObjectiveClassification {
		return fmt.Errorf("config %q training.objective=%q must define at least one block", source, objective)
	}
	switch cfg.EffectiveInputAdapterKind() {
	case InputAdapterLinearFrames, InputAdapterDiscreteCodebooks:
		return nil
	case InputAdapterTokenEmbedding:
		return fmt.Errorf(
			"config %q training.objective=%q with blocks: [] requires input_adapter.kind=%q or %q",
			source, objective, InputAdapterLinearFrames, InputAdapterDiscreteCodebooks,
		)
	default:
		// Let input-adapter validation report unsupported kinds with its more
		// specific public-config error.
		return nil
	}
}
