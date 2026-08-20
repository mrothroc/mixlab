package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

const (
	continuousSequenceShardMagic   = 20260726
	continuousSequenceShardVersion = 2
	continuousSequenceShardV1      = 1
	continuousFeatureDTypeFloat32  = 1
)

// ContinuousSequenceShard is one fixed-shape, atomically labeled frame shard.
type ContinuousSequenceShard struct {
	Frames     []float32
	Lengths    []int32
	Labels     []int32
	Records    int
	SeqLen     int
	FeatureDim int
}

func loadContinuousSequenceLengths(path string) ([]int32, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer func() { _ = file.Close() }()
	header := make([]byte, headerInts*4)
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, 0, 0, fmt.Errorf("read continuous sequence shard %q header: %w", path, err)
	}
	readHeader := func(index int) int { return int(int32(binary.LittleEndian.Uint32(header[index*4:]))) }
	magic, version := readHeader(0), readHeader(1)
	records, seqLen, featureDim := readHeader(2), readHeader(3), readHeader(4)
	if magic != continuousSequenceShardMagic || (version != continuousSequenceShardV1 && version != continuousSequenceShardVersion) || records <= 0 || seqLen <= 0 || featureDim <= 0 {
		return nil, 0, 0, fmt.Errorf("continuous sequence shard %q has invalid header", path)
	}
	if _, err := file.Seek(int64(records*4), io.SeekCurrent); err != nil {
		return nil, 0, 0, fmt.Errorf("seek continuous sequence shard %q lengths: %w", path, err)
	}
	lengths := make([]int32, records)
	if version == continuousSequenceShardV1 {
		for i := range lengths {
			lengths[i] = int32(seqLen)
		}
		return lengths, seqLen, featureDim, nil
	}
	raw := make([]byte, records*4)
	if _, err := io.ReadFull(file, raw); err != nil {
		return nil, 0, 0, fmt.Errorf("read continuous sequence shard %q lengths: %w", path, err)
	}
	for i := range lengths {
		lengths[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
		if lengths[i] <= 0 || int(lengths[i]) > seqLen {
			return nil, 0, 0, fmt.Errorf("continuous sequence shard %q length[%d]=%d must be in [1,%d]", path, i, lengths[i], seqLen)
		}
	}
	return lengths, seqLen, featureDim, nil
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
	if magic != continuousSequenceShardMagic || (version != continuousSequenceShardV1 && version != continuousSequenceShardVersion) {
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
	lengthBytes64 := int64(0)
	if version >= continuousSequenceShardVersion {
		lengthBytes64 = labelBytes64
	}
	expected := int64(headerBytes) + labelBytes64 + lengthBytes64 + frameCount64*4
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
	lengths := make([]int32, records)
	frameStart := headerBytes + labelBytes
	if version >= continuousSequenceShardVersion {
		for i := range lengths {
			lengths[i] = int32(binary.LittleEndian.Uint32(blob[frameStart+i*4:]))
			if lengths[i] <= 0 || int(lengths[i]) > seqLen {
				return nil, fmt.Errorf("continuous sequence shard %q length[%d]=%d must be in [1,%d]", path, i, lengths[i], seqLen)
			}
		}
		frameStart += labelBytes
	} else {
		for i := range lengths {
			lengths[i] = int32(seqLen)
		}
	}
	frames := make([]float32, frameCount)
	for i := range frames {
		frames[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[frameStart+i*4:]))
		if math.IsNaN(float64(frames[i])) || math.IsInf(float64(frames[i]), 0) {
			return nil, fmt.Errorf("continuous sequence shard %q frame[%d] is non-finite", path, i)
		}
	}
	return &ContinuousSequenceShard{
		Frames: frames, Lengths: lengths, Labels: labels, Records: records, SeqLen: seqLen, FeatureDim: featureDim,
	}, nil
}

type continuousSequenceStream struct {
	files             []string
	fileIdx           int
	shard             *ContinuousSequenceShard
	order             []int
	record            int
	rng               *rand.Rand
	shuffle           bool
	seqLen            int
	featureDim        int
	lengthBuckets     []int
	bucketBatchSize   int
	fileLengths       [][]int32
	bucketSchedule    []lengthBucketBatch
	bucketCursor      int
	bucketTokenBudget int
}

func newContinuousSequenceStream(
	pattern string,
	seed int64,
	noShuffle bool,
	seqLen, featureDim int,
	lengthBuckets []int,
	bucketBatchSize int,
) (*continuousSequenceStream, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no continuous sequence shard files matched %q", pattern)
	}
	if err := validateLengthBuckets(lengthBuckets); err != nil && len(lengthBuckets) > 0 {
		return nil, err
	}
	lengthsByPath := make(map[string][]int32, len(files))
	for _, file := range files {
		lengths, fileSeqLen, fileFeatureDim, err := loadContinuousSequenceLengths(file)
		if err != nil {
			return nil, err
		}
		if fileSeqLen != seqLen || fileFeatureDim != featureDim {
			return nil, fmt.Errorf("continuous shard %q shape [T=%d,F=%d] does not match manifest [T=%d,F=%d]", file, fileSeqLen, fileFeatureDim, seqLen, featureDim)
		}
		if err := validateRecordLengthsForBuckets(file, lengths, lengthBuckets); err != nil {
			return nil, err
		}
		lengthsByPath[file] = lengths
	}
	rng := rand.New(rand.NewSource(seed))
	if !noShuffle {
		rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}
	stream := &continuousSequenceStream{
		files: files, rng: rng, shuffle: !noShuffle, seqLen: seqLen, featureDim: featureDim,
		lengthBuckets: append([]int(nil), lengthBuckets...), bucketBatchSize: bucketBatchSize,
	}
	stream.fileLengths = make([][]int32, len(files))
	for i, file := range files {
		stream.fileLengths[i] = lengthsByPath[file]
	}
	if bucketBatchSize > 0 {
		if err := stream.rebuildFixedBucketSchedule(); err != nil {
			return nil, err
		}
		return stream, nil
	}
	if err := stream.loadFile(0); err != nil {
		return nil, err
	}
	return stream, nil
}

func (s *continuousSequenceStream) rebuildFixedBucketSchedule() error {
	var rng *rand.Rand
	if s.shuffle {
		rng = s.rng
	}
	schedule, err := buildFixedLengthBucketSchedule(s.fileLengths, s.lengthBuckets, s.bucketBatchSize, rng)
	if err != nil {
		return fmt.Errorf("build fixed-size continuous length buckets: %w", err)
	}
	if len(schedule) == 0 {
		return fmt.Errorf("fixed-size continuous length buckets contain no records")
	}
	s.bucketSchedule, s.bucketCursor = schedule, 0
	return nil
}

func (s *continuousSequenceStream) fixedBucketShard(source int) (*ContinuousSequenceShard, error) {
	if source < 0 || source >= len(s.files) {
		return nil, fmt.Errorf("continuous length-bucket source %d outside [0,%d)", source, len(s.files))
	}
	if s.shard != nil && s.fileIdx == source {
		return s.shard, nil
	}
	shard, err := LoadContinuousSequenceShard(s.files[source])
	if err != nil {
		return nil, err
	}
	if shard.SeqLen != s.seqLen || shard.FeatureDim != s.featureDim {
		return nil, fmt.Errorf("continuous shard %q changed shape after startup validation", s.files[source])
	}
	s.fileIdx, s.shard = source, shard
	return shard, nil
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
	s.bucketSchedule = nil
	s.bucketCursor = 0
	return nil
}

func (s *continuousSequenceStream) nextLengthBucketBatch(tokenBudget int) (Batch, error) {
	if len(s.lengthBuckets) == 0 {
		return Batch{}, fmt.Errorf("continuous length bucketing is disabled")
	}
	for {
		if s.bucketBatchSize > 0 && s.bucketCursor >= len(s.bucketSchedule) {
			if err := s.rebuildFixedBucketSchedule(); err != nil {
				return Batch{}, err
			}
		}
		if s.bucketSchedule == nil {
			if s.bucketTokenBudget != 0 && s.bucketTokenBudget != tokenBudget {
				return Batch{}, fmt.Errorf("length-bucket token budget changed from %d to %d", s.bucketTokenBudget, tokenBudget)
			}
			s.bucketTokenBudget = tokenBudget
			var rng *rand.Rand
			if s.shuffle {
				rng = s.rng
			}
			schedule, err := buildLengthBucketSchedule(s.order, s.shard.Lengths, s.lengthBuckets, tokenBudget, rng)
			if err != nil {
				return Batch{}, fmt.Errorf("build continuous length buckets for %q: %w", s.files[s.fileIdx], err)
			}
			s.bucketSchedule = schedule
		}
		if s.bucketCursor < len(s.bucketSchedule) {
			break
		}
		if err := s.loadFile((s.fileIdx + 1) % len(s.files)); err != nil {
			return Batch{}, err
		}
	}
	plan := s.bucketSchedule[s.bucketCursor]
	s.bucketCursor++
	batchSize := len(plan.indices)
	frames := make([]float32, batchSize*plan.seqLen*s.featureDim)
	labels := make([]int32, batchSize)
	validMask := make([]float32, batchSize*plan.seqLen)
	exampleMask := make([]float32, batchSize)
	storedWidth := s.seqLen * s.featureDim
	batchShards := make(map[int]*ContinuousSequenceShard)
	for row, index := range plan.indices {
		shard := s.shard
		if len(plan.sources) > 0 {
			sourceIndex := plan.sources[row]
			shard = batchShards[sourceIndex]
			if shard == nil {
				var err error
				shard, err = s.fixedBucketShard(sourceIndex)
				if err != nil {
					return Batch{}, err
				}
				batchShards[sourceIndex] = shard
			}
		}
		source := shard.Frames[index*storedWidth : (index+1)*storedWidth]
		copyTimesteps := plan.seqLen
		if copyTimesteps > s.seqLen {
			copyTimesteps = s.seqLen
		}
		copy(frames[row*plan.seqLen*s.featureDim:], source[:copyTimesteps*s.featureDim])
		labels[row] = shard.Labels[index]
		for pos := 0; pos < int(shard.Lengths[index]); pos++ {
			validMask[row*plan.seqLen+pos] = 1
		}
		if row < plan.realRows {
			exampleMask[row] = 1
		}
	}
	return Batch{
		Frames: frames, Labels: labels, ValidMask: validMask, ExampleMask: exampleMask,
		SeqLen: plan.seqLen, BatchSize: batchSize, ExampleCount: plan.realRows,
	}, nil
}

func (s *continuousSequenceStream) takeRecord() ([]float32, int32, int32, error) {
	if s == nil || s.shard == nil {
		return nil, 0, 0, fmt.Errorf("continuous sequence stream is not initialized")
	}
	if s.record >= len(s.order) {
		next := (s.fileIdx + 1) % len(s.files)
		if err := s.loadFile(next); err != nil {
			return nil, 0, 0, err
		}
	}
	index := s.order[s.record]
	s.record++
	width := s.seqLen * s.featureDim
	start := index * width
	return s.shard.Frames[start : start+width], s.shard.Lengths[index], s.shard.Labels[index], nil
}
