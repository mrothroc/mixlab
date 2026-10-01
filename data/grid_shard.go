package data

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const GridShardMagic = 20260930
const GridShardFormat = "mixlab_grid_shard_v1"

// GridProvenance preserves preparation metadata without fitting or applying
// normalization in the runtime. Group lists align with split record order.
type GridProvenance struct {
	Normalization *GridNormalization  `json:"normalization,omitempty"`
	Groups        map[string][]string `json:"groups,omitempty"`
}
type GridNormalization struct {
	FitSplit     string    `json:"fit_split"`
	InputOffset  []float64 `json:"input_offset"`
	InputScale   []float64 `json:"input_scale"`
	TargetOffset []float64 `json:"target_offset,omitempty"`
	TargetScale  []float64 `json:"target_scale,omitempty"`
}

// GridGeometry describes external CHW records. GPU batches are NHWC.
type GridGeometry struct {
	Channels       int `json:"channels"`
	Height         int `json:"height"`
	Width          int `json:"width"`
	TargetChannels int `json:"target_channels"`
}

func (g GridGeometry) Validate() error {
	if g.Channels <= 0 || g.Channels > math.MaxInt32 || g.TargetChannels < 0 || g.TargetChannels > math.MaxInt32 || g.Height <= 0 || g.Width <= 0 {
		return fmt.Errorf("invalid grid geometry %+v", g)
	}
	n := int64(1)
	for _, d := range []int{g.Height, g.Width, g.Channels + g.TargetChannels} {
		if int64(d) > math.MaxInt32 || n > math.MaxInt32/int64(d) {
			return fmt.Errorf("grid geometry overflows: %+v", g)
		}
		n *= int64(d)
	}
	return nil
}

func (m *DatasetManifest) validateGrid() error {
	if m.Grid == nil {
		return fmt.Errorf("grid geometry required")
	}
	if err := m.Grid.Validate(); err != nil {
		return err
	}
	if m.ShardFormat != GridShardFormat || m.Task == nil || m.Task.Type != "dense_regression" || m.Task.NumLabels != 0 {
		return fmt.Errorf("grid requires dense_regression task and %s", GridShardFormat)
	}
	if m.FeatureDType != "float32" && m.FeatureDType != "float16" {
		return fmt.Errorf("grid feature_dtype must be float32 or float16")
	}
	if m.VocabSize != 0 || m.TokenDType != "" || m.FeatureDim != 0 || m.SequenceLayout != "" || m.RecordSeqLen != 0 || m.NumCodebooks != 0 || len(m.SpecialTokenIDs) > 0 {
		return fmt.Errorf("grid manifest cannot contain sequence/token geometry")
	}
	if len(m.Splits) == 0 {
		return fmt.Errorf("grid manifest has no splits")
	}
	for name, s := range m.Splits {
		if !validDatasetIdentifier(name) || s.Pattern == "" || filepath.IsAbs(s.Pattern) || s.Sequences <= 0 || s.Shards <= 0 {
			return fmt.Errorf("invalid grid split %q", name)
		}
		for _, part := range strings.Split(filepath.ToSlash(s.Pattern), "/") {
			if part == ".." {
				return fmt.Errorf("grid split path escapes manifest")
			}
		}
	}
	if p := m.GridProvenance; p != nil {
		if n := p.Normalization; n != nil {
			if n.FitSplit != "train" {
				return fmt.Errorf("grid normalization must be fitted on train only")
			}
			for j, values := range [][]float64{n.InputOffset, n.InputScale, n.TargetOffset, n.TargetScale} {
				want := m.Grid.Channels
				if j >= 2 {
					want = m.Grid.TargetChannels
				}
				if len(values) != want {
					return fmt.Errorf("grid normalization channel count mismatch")
				}
				for _, v := range values {
					if math.IsNaN(v) || math.IsInf(v, 0) || (j%2 == 1 && v <= 0) {
						return fmt.Errorf("invalid grid normalization value")
					}
				}
			}
		}
		for split, groups := range p.Groups {
			s, ok := m.Splits[split]
			if !ok || int64(len(groups)) != s.Sequences {
				return fmt.Errorf("grid grouping metadata count mismatch")
			}
			for _, g := range groups {
				if strings.TrimSpace(g) == "" {
					return fmt.Errorf("empty grid group ID")
				}
			}
		}
	}
	return nil
}

// GridShard holds only record metadata. Payloads are read one record at a time.
type GridShard struct {
	Geometry      GridGeometry
	IDs           []string
	DType         string
	file          *os.File
	start, stride int64
	bytesPerFloat int
	scratch       []byte
}

func OpenGridShard(path string) (s *GridShard, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			_ = f.Close()
		}
	}()
	var header [1024]byte
	if _, err = io.ReadFull(f, header[:]); err != nil {
		return nil, err
	}
	word := func(n int) int { return int(binary.LittleEndian.Uint32(header[n*4:])) }
	if word(0) != GridShardMagic || word(1) != 1 {
		return nil, fmt.Errorf("unsupported grid shard magic/version")
	}
	if word(2) != 1 && word(2) != 2 {
		return nil, fmt.Errorf("invalid grid dtype")
	}
	s = &GridShard{file: f, Geometry: GridGeometry{Channels: word(3), Height: word(4), Width: word(5), TargetChannels: word(6)}, bytesPerFloat: 4, DType: "float32"}
	if word(2) == 2 {
		s.bytesPerFloat = 2
		s.DType = "float16"
	}
	if err = s.Geometry.Validate(); err != nil {
		return nil, err
	}
	count, metaLen := word(7), word(8)
	if count <= 0 || count > 1_000_000 || metaLen <= 0 || metaLen > 16<<20 {
		return nil, fmt.Errorf("invalid grid count/metadata length")
	}
	for n := 9; n < 256; n++ {
		if word(n) != 0 {
			return nil, fmt.Errorf("nonzero reserved grid header word %d", n)
		}
	}
	meta := make([]byte, metaLen)
	if _, err = io.ReadFull(f, meta); err != nil {
		return nil, err
	}
	if err = json.Unmarshal(meta, &s.IDs); err != nil {
		return nil, err
	}
	if len(s.IDs) != count {
		return nil, fmt.Errorf("grid ID count mismatch")
	}
	seen := map[string]bool{}
	for _, id := range s.IDs {
		if strings.TrimSpace(id) == "" || seen[id] {
			return nil, fmt.Errorf("empty/duplicate grid ID %q", id)
		}
		seen[id] = true
	}
	g := s.Geometry
	pixels := int64(g.Height) * int64(g.Width)
	s.stride = pixels * int64(g.Channels+g.TargetChannels) * int64(s.bytesPerFloat)
	if g.TargetChannels > 0 {
		s.stride += (pixels + 7) / 8
	}
	s.start = 1024 + int64(metaLen)
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() != s.start+s.stride*int64(count) {
		return nil, fmt.Errorf("grid payload size mismatch")
	}
	// Bound a single record too: malformed geometry must not request multi-GB scratch.
	if s.stride > 512<<20 {
		return nil, fmt.Errorf("grid record exceeds 512 MiB limit")
	}
	s.scratch = make([]byte, s.stride)
	return s, nil
}

func (s *GridShard) Close() error { return s.file.Close() }

// ReadNHWC decodes into caller-owned batch slices. Mask is broadcast to channels.
func (s *GridShard) ReadNHWC(index int, x, y, mask []float32) error {
	g := s.Geometry
	p := g.Height * g.Width
	if index < 0 || index >= len(s.IDs) || len(x) != p*g.Channels || len(y) != p*g.TargetChannels || len(mask) != len(y) {
		return fmt.Errorf("grid record/index buffer mismatch")
	}
	if _, err := s.file.ReadAt(s.scratch, s.start+int64(index)*s.stride); err != nil {
		return err
	}
	read := func(n int) float32 {
		if s.bytesPerFloat == 2 {
			return decodeGridHalf(binary.LittleEndian.Uint16(s.scratch[n*2:]))
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(s.scratch[n*4:]))
	}
	for c := 0; c < g.Channels; c++ {
		for j := 0; j < p; j++ {
			v := read(c*p + j)
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("grid input contains nonfinite value")
			}
			x[j*g.Channels+c] = v
		}
	}
	for c := 0; c < g.TargetChannels; c++ {
		for j := 0; j < p; j++ {
			v := read(p*g.Channels + c*p + j)
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("grid target contains nonfinite value")
			}
			y[j*g.TargetChannels+c] = v
		}
	}
	if g.TargetChannels > 0 {
		bits := s.scratch[p*(g.Channels+g.TargetChannels)*s.bytesPerFloat:]
		if p%8 != 0 && bits[len(bits)-1]>>uint(p%8) != 0 {
			return fmt.Errorf("grid mask has nonzero padding bits")
		}
		for j := 0; j < p; j++ {
			v := float32((bits[j/8] >> uint(j%8)) & 1)
			for c := 0; c < g.TargetChannels; c++ {
				mask[j*g.TargetChannels+c] = v
			}
		}
	}
	return nil
}

func decodeGridHalf(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 31)
	fraction := uint32(h & 1023)
	switch exp {
	case 0:
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		exp = 1
		for fraction&1024 == 0 {
			fraction <<= 1
			exp--
		}
		fraction &= 1023
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | fraction<<13)
}
