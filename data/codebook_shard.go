package data

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

const (
	codebookSequenceShardMagic   = 20260819
	codebookSequenceShardVersion = 1
	codebookTokenDTypeInt32      = 1
)

// CodebookSequenceShard stores fixed [N,T,Q] codebook IDs, valid timestep
// lengths, and one classification label per record.
type CodebookSequenceShard struct {
	Tokens            []int32
	Lengths           []int32
	Labels            []int32
	Records           int
	SeqLen            int
	NumCodebooks      int
	CodebookVocabSize int
}

func loadCodebookSequenceLengths(path string) ([]int32, int, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer func() { _ = file.Close() }()
	header := make([]byte, headerInts*4)
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("read codebook sequence shard %q header: %w", path, err)
	}
	readHeader := func(index int) int { return int(int32(binary.LittleEndian.Uint32(header[index*4:]))) }
	magic, version := readHeader(0), readHeader(1)
	records, seqLen, numCodebooks := readHeader(2), readHeader(3), readHeader(4)
	codebookVocabSize := readHeader(5)
	if magic != codebookSequenceShardMagic || version != codebookSequenceShardVersion || records <= 0 || seqLen <= 0 || numCodebooks <= 0 {
		return nil, 0, 0, 0, fmt.Errorf("codebook sequence shard %q has invalid header", path)
	}
	if _, err := file.Seek(int64(records*4), io.SeekCurrent); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("seek codebook sequence shard %q lengths: %w", path, err)
	}
	raw := make([]byte, records*4)
	if _, err := io.ReadFull(file, raw); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("read codebook sequence shard %q lengths: %w", path, err)
	}
	lengths := make([]int32, records)
	for i := range lengths {
		lengths[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
		if lengths[i] <= 0 || int(lengths[i]) > seqLen {
			return nil, 0, 0, 0, fmt.Errorf("codebook sequence shard %q length[%d]=%d must be in [1,%d]", path, i, lengths[i], seqLen)
		}
	}
	return lengths, seqLen, numCodebooks, codebookVocabSize, nil
}

func LoadCodebookSequenceShard(path string) (*CodebookSequenceShard, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	headerBytes := headerInts * 4
	if len(blob) < headerBytes {
		return nil, fmt.Errorf("codebook sequence shard %q is too small to contain a mixlab header", path)
	}
	readHeader := func(index int) int {
		return int(int32(binary.LittleEndian.Uint32(blob[index*4 : index*4+4])))
	}
	magic, version := readHeader(0), readHeader(1)
	records, seqLen, numCodebooks := readHeader(2), readHeader(3), readHeader(4)
	codebookVocabSize, dtype := readHeader(5), readHeader(6)
	hasLabels, hasLengths := readHeader(7), readHeader(8)
	if magic != codebookSequenceShardMagic || version != codebookSequenceShardVersion {
		return nil, fmt.Errorf("codebook sequence shard %q has unsupported magic/version %d/%d", path, magic, version)
	}
	if records <= 0 || seqLen <= 0 || numCodebooks <= 0 || codebookVocabSize <= 1 {
		return nil, fmt.Errorf("codebook sequence shard %q has invalid shape/domain N=%d T=%d Q=%d V=%d", path, records, seqLen, numCodebooks, codebookVocabSize)
	}
	if dtype != codebookTokenDTypeInt32 || hasLabels != 1 || hasLengths != 1 {
		return nil, fmt.Errorf("codebook sequence shard %q requires int32 tokens, labels, and lengths", path)
	}
	maxInt := int64(^uint(0) >> 1)
	tokenCount64 := int64(records) * int64(seqLen) * int64(numCodebooks)
	metadataCount64 := int64(records) * 2
	if tokenCount64 < 0 || tokenCount64 > maxInt || metadataCount64 > maxInt-tokenCount64 {
		return nil, fmt.Errorf("codebook sequence shard %q payload is too large", path)
	}
	expected := int64(headerBytes) + 4*(metadataCount64+tokenCount64)
	if int64(len(blob)) != expected {
		return nil, fmt.Errorf("codebook sequence shard %q size mismatch: got=%d bytes want=%d bytes", path, len(blob), expected)
	}
	offset := headerBytes
	labels := make([]int32, records)
	for i := range labels {
		labels[i] = int32(binary.LittleEndian.Uint32(blob[offset+i*4:]))
		if labels[i] < 0 {
			return nil, fmt.Errorf("codebook sequence shard %q label[%d]=%d must be non-negative", path, i, labels[i])
		}
	}
	offset += records * 4
	lengths := make([]int32, records)
	for i := range lengths {
		lengths[i] = int32(binary.LittleEndian.Uint32(blob[offset+i*4:]))
		if lengths[i] <= 0 || int(lengths[i]) > seqLen {
			return nil, fmt.Errorf("codebook sequence shard %q length[%d]=%d must be in [1,%d]", path, i, lengths[i], seqLen)
		}
	}
	offset += records * 4
	tokens := make([]int32, int(tokenCount64))
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(blob[offset+i*4:]))
		if tokens[i] < 0 || int(tokens[i]) >= codebookVocabSize {
			codebook := i % numCodebooks
			timestep := (i / numCodebooks) % seqLen
			record := i / (numCodebooks * seqLen)
			return nil, fmt.Errorf("codebook sequence shard %q token[%d,%d,%d]=%d outside [0,%d)", path, record, timestep, codebook, tokens[i], codebookVocabSize)
		}
	}
	return &CodebookSequenceShard{
		Tokens: tokens, Lengths: lengths, Labels: labels, Records: records, SeqLen: seqLen,
		NumCodebooks: numCodebooks, CodebookVocabSize: codebookVocabSize,
	}, nil
}

type codebookSequenceStream struct {
	files             []string
	fileIdx           int
	shard             *CodebookSequenceShard
	order             []int
	record            int
	rng               *rand.Rand
	shuffle           bool
	seqLen            int
	numCodebooks      int
	codebookVocabSize int
	lengthBuckets     []int
	bucketSchedule    []lengthBucketBatch
	bucketCursor      int
	bucketTokenBudget int
}

func newCodebookSequenceStream(pattern string, seed int64, noShuffle bool, seqLen, numCodebooks, codebookVocabSize int, lengthBuckets []int) (*codebookSequenceStream, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no codebook sequence shard files matched %q", pattern)
	}
	if err := validateLengthBuckets(lengthBuckets); err != nil && len(lengthBuckets) > 0 {
		return nil, err
	}
	for _, file := range files {
		lengths, fileSeqLen, fileCodebooks, fileVocab, err := loadCodebookSequenceLengths(file)
		if err != nil {
			return nil, err
		}
		if fileSeqLen != seqLen || fileCodebooks != numCodebooks || fileVocab != codebookVocabSize {
			return nil, fmt.Errorf("codebook shard %q domain [T=%d,Q=%d,V=%d] does not match manifest [T=%d,Q=%d,V=%d]", file, fileSeqLen, fileCodebooks, fileVocab, seqLen, numCodebooks, codebookVocabSize)
		}
		if err := validateRecordLengthsForBuckets(file, lengths, lengthBuckets); err != nil {
			return nil, err
		}
	}
	rng := rand.New(rand.NewSource(seed))
	if !noShuffle {
		rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}
	stream := &codebookSequenceStream{
		files: files, rng: rng, shuffle: !noShuffle, seqLen: seqLen,
		numCodebooks: numCodebooks, codebookVocabSize: codebookVocabSize,
		lengthBuckets: append([]int(nil), lengthBuckets...),
	}
	if err := stream.loadFile(0); err != nil {
		return nil, err
	}
	return stream, nil
}

func (s *codebookSequenceStream) loadFile(index int) error {
	shard, err := LoadCodebookSequenceShard(s.files[index])
	if err != nil {
		return err
	}
	if shard.SeqLen != s.seqLen || shard.NumCodebooks != s.numCodebooks || shard.CodebookVocabSize != s.codebookVocabSize {
		return fmt.Errorf("codebook shard %q domain [T=%d,Q=%d,V=%d] does not match manifest [T=%d,Q=%d,V=%d]", s.files[index], shard.SeqLen, shard.NumCodebooks, shard.CodebookVocabSize, s.seqLen, s.numCodebooks, s.codebookVocabSize)
	}
	order := make([]int, shard.Records)
	for i := range order {
		order[i] = i
	}
	if s.shuffle {
		s.rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	}
	s.fileIdx, s.shard, s.order, s.record = index, shard, order, 0
	s.bucketSchedule = nil
	s.bucketCursor = 0
	return nil
}

func (s *codebookSequenceStream) nextLengthBucketBatch(tokenBudget int) (Batch, error) {
	if len(s.lengthBuckets) == 0 {
		return Batch{}, fmt.Errorf("codebook length bucketing is disabled")
	}
	for {
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
				return Batch{}, fmt.Errorf("build codebook length buckets for %q: %w", s.files[s.fileIdx], err)
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
	Q := s.numCodebooks
	tokens := make([]int32, batchSize*plan.seqLen*Q)
	labels := make([]int32, batchSize)
	validMask := make([]float32, batchSize*plan.seqLen)
	exampleMask := make([]float32, batchSize)
	storedWidth := s.seqLen * Q
	for row, index := range plan.indices {
		source := s.shard.Tokens[index*storedWidth : (index+1)*storedWidth]
		copyTimesteps := plan.seqLen
		if copyTimesteps > s.seqLen {
			copyTimesteps = s.seqLen
		}
		copy(tokens[row*plan.seqLen*Q:], source[:copyTimesteps*Q])
		labels[row] = s.shard.Labels[index]
		for pos := 0; pos < int(s.shard.Lengths[index]); pos++ {
			validMask[row*plan.seqLen+pos] = 1
		}
		if row < plan.realRows {
			exampleMask[row] = 1
		}
	}
	return Batch{
		Codebooks: tokens, Labels: labels, ValidMask: validMask, ExampleMask: exampleMask,
		SeqLen: plan.seqLen, BatchSize: batchSize, ExampleCount: plan.realRows,
	}, nil
}

func (s *codebookSequenceStream) takeRecord() ([]int32, int32, int32, error) {
	if s == nil || s.shard == nil {
		return nil, 0, 0, fmt.Errorf("codebook sequence stream is not initialized")
	}
	if s.record >= len(s.order) {
		if err := s.loadFile((s.fileIdx + 1) % len(s.files)); err != nil {
			return nil, 0, 0, err
		}
	}
	index := s.order[s.record]
	s.record++
	width := s.seqLen * s.numCodebooks
	start := index * width
	return s.shard.Tokens[start : start+width], s.shard.Lengths[index], s.shard.Labels[index], nil
}
