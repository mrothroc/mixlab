package train

import "fmt"

func preparePatchArgs(path, format string) ([]string, error) {
	if path == "" {
		return nil, nil
	}
	cfg, err := LoadArchConfigQuiet(path)
	if err != nil {
		return nil, err
	}
	if !cfg.LinearPatchesEnabled() {
		return nil, fmt.Errorf("prepare -config currently requires input_adapter.kind=linear_patches")
	}
	if format != "continuous" {
		return nil, fmt.Errorf("linear_patches prepare requires -input-format=continuous")
	}
	return []string{"--patch-seq-len", fmt.Sprint(cfg.SeqLen), "--patch-feature-dim", fmt.Sprint(cfg.InputFeatureDim())}, nil
}
