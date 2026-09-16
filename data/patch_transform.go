package data

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
)

// PatchTransform describes channel-last pixels within raster-ordered patches.
// Training is runtime-only: evaluation always uses geometry without transforms.
type PatchTransform struct {
	Height, Width, Channels, Patch int
	Training, HFlip                bool
	CropPad                        int
	PadValue                       []float32
}

func withoutPatchAugmentation(opts LoaderOptions) LoaderOptions {
	if opts.PatchTransform != nil {
		p := *opts.PatchTransform
		p.Training = false
		opts.PatchTransform = &p
	}
	return opts
}

func validatePatchFiles(pattern string, p *PatchTransform) error {
	if p == nil {
		return nil
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	for _, file := range files {
		lengths, t, f, err := loadContinuousSequenceLengths(file)
		if err != nil {
			return err
		}
		if err := p.Validate(t, f); err != nil {
			return fmt.Errorf("patch shard %q: %w", file, err)
		}
		for _, n := range lengths {
			if int(n) != t {
				return fmt.Errorf("linear_patches requires full-length image records: %q", file)
			}
		}
	}
	return nil
}

func (p PatchTransform) Validate(seqLen, featureDim int) error {
	h, w, c, k := p.Height, p.Width, p.Channels, p.Patch
	if h <= 0 || w <= 0 || c <= 0 || k <= 0 || h%k != 0 || w%k != 0 || h > math.MaxInt32/w || h*w > math.MaxInt32/c {
		return fmt.Errorf("invalid patch image geometry")
	}
	if seqLen != (h/k)*(w/k) || featureDim != k*k*c {
		return fmt.Errorf("patch geometry does not match [T=%d,F=%d]", seqLen, featureDim)
	}
	if p.CropPad < 0 || p.CropPad > (math.MaxInt32-max(h, w))/2 {
		return fmt.Errorf("invalid patch crop padding")
	}
	if len(p.PadValue) != 0 && len(p.PadValue) != c {
		return fmt.Errorf("patch pad_value must have %d channels", c)
	}
	for _, v := range p.PadValue {
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			return fmt.Errorf("patch pad_value must be finite")
		}
	}
	return nil
}

type patchTransformState struct {
	spec       PatchTransform
	seed       uint64
	occurrence uint64
	rng        *rand.Rand
	scratch    []float32
}

func newPatchTransformState(p *PatchTransform, seed int64, t, f int) (*patchTransformState, error) {
	if p == nil {
		return nil, nil
	}
	if err := p.Validate(t, f); err != nil {
		return nil, err
	}
	s := &patchTransformState{spec: *p, seed: uint64(seed)}
	s.spec.PadValue = append([]float32(nil), p.PadValue...)
	if p.Training && (p.HFlip || p.CropPad > 0) {
		s.scratch = make([]float32, t*f)
		s.rng = rand.New(rand.NewSource(0))
	}
	return s, nil
}

func (s *patchTransformState) apply(record []float32) {
	if s.rng == nil {
		return
	}
	// SplitMix64 separates transforms from shuffle RNG and addresses occurrences
	// directly, so replay and loader prefetch cannot change an augmentation.
	z := s.seed + 0x70617463685f6175 + (s.occurrence+1)*0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	s.rng.Seed(int64(z ^ (z >> 31)))
	s.occurrence++
	y, x := 0, 0
	if s.spec.CropPad > 0 {
		y = s.rng.Intn(2*s.spec.CropPad + 1)
		x = s.rng.Intn(2*s.spec.CropPad + 1)
	}
	flip := s.spec.HFlip && s.rng.Intn(2) == 1
	s.spec.transform(s.scratch, record, y, x, flip)
	copy(record, s.scratch)
}

func (p PatchTransform) pixelIndex(y, x, c int) int {
	k := p.Patch
	return (((y/k)*(p.Width/k)+x/k)*k*k+(y%k)*k+x%k)*p.Channels + c
}

// transform directly addresses pad -> crop -> flip in patch storage. This is
// equivalent to unpatchify/repatchify without allocating a padded image.
func (p PatchTransform) transform(dst, src []float32, cropY, cropX int, flip bool) {
	for y := 0; y < p.Height; y++ {
		for x := 0; x < p.Width; x++ {
			sx := x
			if flip {
				sx = p.Width - 1 - x
			}
			sy := y + cropY - p.CropPad
			sx += cropX - p.CropPad
			for c := 0; c < p.Channels; c++ {
				var v float32
				if sy >= 0 && sy < p.Height && sx >= 0 && sx < p.Width {
					v = src[p.pixelIndex(sy, sx, c)]
				} else if len(p.PadValue) > 0 {
					v = p.PadValue[c]
				}
				dst[p.pixelIndex(y, x, c)] = v
			}
		}
	}
}
