package train

import "github.com/mrothroc/mixlab/data"

func patchLoaderTransform(cfg *ArchConfig, training bool) *data.PatchTransform {
	if !cfg.LinearPatchesEnabled() {
		return nil
	}
	s := cfg.InputAdapter
	p := &data.PatchTransform{Height: s.Image.Height, Width: s.Image.Width, Channels: s.Image.Channels, Patch: s.Patch, Training: training}
	if s.Augment != nil {
		p.HFlip = s.Augment.HFlip
		p.CropPad = s.Augment.RandomCropPad
		p.PadValue = s.Augment.PadValue
	}
	return p
}

func trainingLoaderOptions(cfg *ArchConfig) data.LoaderOptions {
	opts := effectiveLoaderOptions(cfg)
	opts.PatchTransform = patchLoaderTransform(cfg, true)
	return opts
}
