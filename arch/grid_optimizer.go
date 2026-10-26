package arch

import "fmt"

const (
	KernelLayoutOHWI = "ohwi"
	KernelLayoutIHWO = "ihwo"
)

// UsesMuon reports the explicit optimizer policy; it does not change defaults.
func (s TrainingSpec) UsesMuon() bool {
	return s.Optimizer == "muon" || s.Optimizer == "muon_eq_r" || s.Optimizer == "normuon"
}

// Resolve from graph consumers, never from ambiguous rank-4 dimensions.
func resolveGridKernelMetadata(cfg *ArchConfig, metas []WeightMeta) error {
	excluded := map[string]bool{}
	for _, name := range cfg.Training.MuonExclude {
		if name == "" || excluded[name] {
			return fmt.Errorf("training.muon_exclude has empty or duplicate node %q", name)
		}
		excluded[name] = true
	}
	spec := cfg.Blocks[0]
	indices := map[string]int{}
	for i, w := range spec.Weights {
		indices[w.Name] = i
	}
	matched := map[string]bool{}
	for _, op := range spec.Ops {
		layout := ""
		switch op.Op {
		case "conv2d":
			layout = KernelLayoutOHWI
		case "conv_transpose2d":
			layout = KernelLayoutIHWO
		default:
			continue
		}
		i, ok := indices[op.Inputs[1]]
		if !ok {
			if cfg.Training.UsesMuon() {
				return fmt.Errorf("muon grid kernel for node %q must be a declared weight", op.Output)
			}
			continue
		}
		m := &metas[i]
		if m.KernelLayout != "" && (m.KernelLayout != layout || m.MuonExcluded != excluded[op.Output]) {
			if cfg.Training.UsesMuon() {
				return fmt.Errorf("shared grid kernel %q has conflicting layouts or muon_exclude policies", m.Name)
			}
		}
		m.KernelLayout, m.MuonExcluded = layout, excluded[op.Output]
		matched[op.Output] = true
	}
	for name := range excluded {
		if !matched[name] {
			return fmt.Errorf("training.muon_exclude node %q is not a convolution node", name)
		}
	}
	return nil
}
