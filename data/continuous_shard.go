package data

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

const (
	continuousSequenceShardMagic   = 20260726
	continuousSequenceShardVersion = 1
	continuousFeatureDTypeFloat32  = 1
)

// ContinuousSequenceShard is one fixed-shape, atomically labeled frame shard.
type ContinuousSequenceShard struct {
	Frames     []float32
	Labels     []int32
	Records    int
	SeqLen     int
	FeatureDim int
}

// LoadContinuousSequenceShard reads [N,T,F] float32 frames plus one int32
// label per record. The exact-size and finite checks keep malformed data from
// reaching GPU matrix operations.
func LoadContinuousSequenceShard(path string) (*ContinuousSequenceShard, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	headerBytes := headerInts * 4
	if len(blob) < headerBytes {
		return nil, fmt.Errorf("continuous sequence shard %q is too small to contain a mixlab header", path)
	}
	readHeader := func(index int) int {
		return int(int32(binary.LittleEndian.Uint32(blob[index*4 : index*4+4])))
	}
	magic := readHeader(0)
	version := readHeader(1)
	records := readHeader(2)
	seqLen := readHeader(3)
	featureDim := readHeader(4)
	dtype := readHeader(5)
	hasLabels := readHeader(6)
	if magic != continuousSequenceShardMagic || version != continuousSequenceShardVersion {
		return nil, fmt.Errorf("continuous sequence shard %q has unsupported magic/version %d/%d", path, magic, version)
	}
	if records <= 0 || seqLen <= 0 || featureDim <= 0 {
		return nil, fmt.Errorf("continuous sequence shard %q has invalid shape N=%d T=%d F=%d", path, records, seqLen, featureDim)
	}
	if dtype != continuousFeatureDTypeFloat32 {
		return nil, fmt.Errorf("continuous sequence shard %q has unsupported dtype code %d", path, dtype)
	}
	if hasLabels != 1 {
		return nil, fmt.Errorf("continuous sequence shard %q is missing atomic classification labels", path)
	}
	maxInt := int64(^uint(0) >> 1)
	frameCount64 := int64(records) * int64(seqLen)
	if frameCount64 > maxInt/int64(featureDim) {
		return nil, fmt.Errorf("continuous sequence shard %q shape is too large", path)
	}
	frameCount64 *= int64(featureDim)
	labelBytes64 := int64(records) * 4
	if labelBytes64 > maxInt-int64(headerBytes) ||
		frameCount64 > (maxInt-int64(headerBytes)-labelBytes64)/4 {
		return nil, fmt.Errorf("continuous sequence shard %q payload is too large", path)
	}
	frameCount := int(frameCount64)
	labelBytes := int(labelBytes64)
	expected := int64(headerBytes) + labelBytes64 + frameCount64*4
	if int64(len(blob)) != expected {
		return nil, fmt.Errorf("continuous sequence shard %q size mismatch: got=%d bytes want=%d bytes", path, len(blob), expected)
	}
	labels := make([]int32, records)
	for i := range labels {
		labels[i] = int32(binary.LittleEndian.Uint32(blob[headerBytes+i*4:]))
		if labels[i] < 0 {
			return nil, fmt.Errorf("continuous sequence shard %q label[%d]=%d must be non-negative", path, i, labels[i])
		}
	}
	frameStart := headerBytes + labelBytes
	frames := make([]float32, frameCount)
	for i := range frames {
		frames[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[frameStart+i*4:]))
		if math.IsNaN(float64(frames[i])) || math.IsInf(float64(frames[i]), 0) {
			return nil, fmt.Errorf("continuous sequence shard %q frame[%d] is non-finite", path, i)
		}
	}
	return &ContinuousSequenceShard{
		Frames: frames, Labels: labels, Records: records, SeqLen: seqLen, FeatureDim: featureDim,
	}, nil
}

type continuousSequenceStream struct {
	files      []string
	fileIdx    int
	shard      *ContinuousSequenceShard
	order      []int
	record     int
	rng        *rand.Rand
	shuffle    bool
	seqLen     int
	featureDim int
}

func newContinuousSequenceStream(pattern string, seed int64, noShuffle bool, seqLen, featureDim int) (*continuousSequenceStream, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no continuous sequence shard files matched %q", pattern)
	}
	rng := rand.New(rand.NewSource(seed))
	if !noShuffle {
		rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}
	stream := &continuousSequenceStream{
		files: files, rng: rng, shuffle: !noShuffle, seqLen: seqLen, featureDim: featureDim,
	}
	if err := stream.loadFile(0); err != nil {
		return nil, err
	}
	return stream, nil
}

func (s *continuousSequenceStream) loadFile(index int) error {
	shard, err := LoadContinuousSequenceShard(s.files[index])
	if err != nil {
		return err
	}
	if shard.SeqLen != s.seqLen || shard.FeatureDim != s.featureDim {
		return fmt.Errorf(
			"continuous shard %q shape [T=%d,F=%d] does not match manifest [T=%d,F=%d]",
			s.files[index], shard.SeqLen, shard.FeatureDim, s.seqLen, s.featureDim,
		)
	}
	order := make([]int, shard.Records)
	for i := range order {
		order[i] = i
	}
	if s.shuffle {
		s.rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}
	s.fileIdx = index
	s.shard = shard
	s.order = order
	s.record = 0
	return nil
}

func (s *continuousSequenceStream) takeRecord() ([]float32, int32, error) {
	if s == nil || s.shard == nil {
		return nil, 0, fmt.Errorf("continuous sequence stream is not initialized")
	}
	if s.record >= len(s.order) {
		next := (s.fileIdx + 1) % len(s.files)
		if err := s.loadFile(next); err != nil {
			return nil, 0, err
		}
	}
	index := s.order[s.record]
	s.record++
	width := s.seqLen * s.featureDim
	start := index * width
	return s.shard.Frames[start : start+width], s.shard.Labels[index], nil
}
