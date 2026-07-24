package data

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
)

const (
	sequenceShardMagic          = 20260718
	sequenceShardVersion        = 1
	labeledSequenceShardMagic   = 20260724
	labeledSequenceShardVersion = 1
)

// LoadSequenceShard reads a record-oriented discrete-sequence shard. The
// payload is an offsets table followed by contiguous uint16 token records.
func LoadSequenceShard(path string) ([][]uint16, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	headerBytes := headerInts * 4
	if len(blob) < headerBytes {
		return nil, fmt.Errorf("sequence shard %q is too small to contain a mixlab header", path)
	}
	magic := int(binary.LittleEndian.Uint32(blob[0:4]))
	version := int(binary.LittleEndian.Uint32(blob[4:8]))
	nTokens := int(binary.LittleEndian.Uint32(blob[8:12]))
	nSequences := int(binary.LittleEndian.Uint32(blob[12:16]))
	if magic != sequenceShardMagic || version != sequenceShardVersion {
		return nil, fmt.Errorf("sequence shard %q has unsupported magic/version %d/%d", path, magic, version)
	}
	if nTokens < 0 || nSequences <= 0 {
		return nil, fmt.Errorf("sequence shard %q has invalid counts tokens=%d sequences=%d", path, nTokens, nSequences)
	}
	offsetBytes := (nSequences + 1) * 8
	expected := headerBytes + offsetBytes + nTokens*2
	if len(blob) != expected {
		return nil, fmt.Errorf("sequence shard %q size mismatch: got=%d bytes want=%d bytes", path, len(blob), expected)
	}
	offsets := make([]uint64, nSequences+1)
	for i := range offsets {
		offsets[i] = binary.LittleEndian.Uint64(blob[headerBytes+i*8:])
	}
	if offsets[0] != 0 || offsets[len(offsets)-1] != uint64(nTokens) {
		return nil, fmt.Errorf("sequence shard %q has invalid terminal offsets %d..%d for %d tokens", path, offsets[0], offsets[len(offsets)-1], nTokens)
	}
	payload := headerBytes + offsetBytes
	records := make([][]uint16, nSequences)
	for i := 0; i < nSequences; i++ {
		start, end := offsets[i], offsets[i+1]
		if start >= end || end > uint64(nTokens) {
			return nil, fmt.Errorf("sequence shard %q record %d has invalid offsets [%d,%d)", path, i, start, end)
		}
		record := make([]uint16, int(end-start))
		for j := range record {
			record[j] = binary.LittleEndian.Uint16(blob[payload+(int(start)+j)*2:])
		}
		records[i] = record
	}
	return records, nil
}

// LoadLabeledSequenceShard reads an atomic sequence-plus-label shard. Labels
// are int32 class IDs aligned one-for-one with the sequence records.
func LoadLabeledSequenceShard(path string) ([][]uint16, []int32, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	headerBytes := headerInts * 4
	if len(blob) < headerBytes {
		return nil, nil, fmt.Errorf("labeled sequence shard %q is too small to contain a mixlab header", path)
	}
	magic := int(binary.LittleEndian.Uint32(blob[0:4]))
	version := int(binary.LittleEndian.Uint32(blob[4:8]))
	nTokens := int(binary.LittleEndian.Uint32(blob[8:12]))
	nSequences := int(binary.LittleEndian.Uint32(blob[12:16]))
	if magic != labeledSequenceShardMagic || version != labeledSequenceShardVersion {
		return nil, nil, fmt.Errorf("labeled sequence shard %q has unsupported magic/version %d/%d", path, magic, version)
	}
	if nTokens < 0 || nSequences <= 0 {
		return nil, nil, fmt.Errorf("labeled sequence shard %q has invalid counts tokens=%d sequences=%d", path, nTokens, nSequences)
	}
	offsetBytes := (nSequences + 1) * 8
	labelBytes := nSequences * 4
	expected := headerBytes + offsetBytes + labelBytes + nTokens*2
	if len(blob) != expected {
		return nil, nil, fmt.Errorf("labeled sequence shard %q size mismatch: got=%d bytes want=%d bytes", path, len(blob), expected)
	}
	offsets := make([]uint64, nSequences+1)
	for i := range offsets {
		offsets[i] = binary.LittleEndian.Uint64(blob[headerBytes+i*8:])
	}
	if offsets[0] != 0 || offsets[len(offsets)-1] != uint64(nTokens) {
		return nil, nil, fmt.Errorf("labeled sequence shard %q has invalid terminal offsets %d..%d for %d tokens", path, offsets[0], offsets[len(offsets)-1], nTokens)
	}
	labels := make([]int32, nSequences)
	labelStart := headerBytes + offsetBytes
	for i := range labels {
		labels[i] = int32(binary.LittleEndian.Uint32(blob[labelStart+i*4:]))
	}
	payload := labelStart + labelBytes
	records := make([][]uint16, nSequences)
	for i := range records {
		start, end := offsets[i], offsets[i+1]
		if start >= end || end > uint64(nTokens) {
			return nil, nil, fmt.Errorf("labeled sequence shard %q record %d has invalid offsets [%d,%d)", path, i, start, end)
		}
		record := make([]uint16, int(end-start))
		for j := range record {
			record[j] = binary.LittleEndian.Uint16(blob[payload+(int(start)+j)*2:])
		}
		records[i] = record
	}
	return records, labels, nil
}

type sequenceStream struct {
	files   []string
	fileIdx int
	records [][]uint16
	labels  []int32
	record  int
	pos     int
	rng     *rand.Rand
	shuffle bool
	labeled bool
}

func newSequenceStreamWithFormat(pattern string, seed int64, noShuffle bool, shardFormat string) (*sequenceStream, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no sequence shard files matched %q", pattern)
	}
	rng := rand.New(rand.NewSource(seed))
	if !noShuffle {
		rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}
	s := &sequenceStream{
		files: files, rng: rng, shuffle: !noShuffle,
		labeled: shardFormat == DatasetShardFormatLabeledSequenceV1,
	}
	if err := s.loadFile(0); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *sequenceStream) loadFile(index int) error {
	var records [][]uint16
	var labels []int32
	var err error
	if s.labeled {
		records, labels, err = LoadLabeledSequenceShard(s.files[index])
	} else {
		records, err = LoadSequenceShard(s.files[index])
	}
	if err != nil {
		return err
	}
	if s.shuffle {
		s.rng.Shuffle(len(records), func(i, j int) {
			records[i], records[j] = records[j], records[i]
			if len(labels) == len(records) {
				labels[i], labels[j] = labels[j], labels[i]
			}
		})
	}
	s.fileIdx = index
	s.records = records
	s.labels = labels
	s.record = 0
	s.pos = 0
	return nil
}

func (s *sequenceStream) advanceRecord() error {
	s.record++
	s.pos = 0
	if s.record < len(s.records) {
		return nil
	}
	next := (s.fileIdx + 1) % len(s.files)
	return s.loadFile(next)
}

func (s *sequenceStream) takeChunk(max int) ([]uint16, error) {
	if max <= 0 {
		return nil, nil
	}
	for attempts := 0; attempts <= len(s.files); attempts++ {
		if s.record >= len(s.records) {
			if err := s.advanceRecord(); err != nil {
				return nil, err
			}
		}
		record := s.records[s.record]
		if s.pos >= len(record) {
			if err := s.advanceRecord(); err != nil {
				return nil, err
			}
			continue
		}
		n := len(record) - s.pos
		if n > max {
			n = max
		}
		out := append([]uint16(nil), record[s.pos:s.pos+n]...)
		s.pos += n
		if s.pos == len(record) {
			if err := s.advanceRecord(); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("sequence shards contain no non-empty records")
}

func (s *sequenceStream) takeLabeledRecord() ([]uint16, int32, error) {
	for attempts := 0; attempts <= len(s.files); attempts++ {
		if s.record >= len(s.records) {
			if err := s.advanceRecord(); err != nil {
				return nil, 0, err
			}
			continue
		}
		index := s.record
		record := append([]uint16(nil), s.records[index]...)
		var label int32
		if s.labeled {
			if len(s.labels) != len(s.records) {
				return nil, 0, fmt.Errorf("labeled sequence shard has %d records but %d labels", len(s.records), len(s.labels))
			}
			label = s.labels[index]
		}
		if err := s.advanceRecord(); err != nil {
			return nil, 0, err
		}
		return record, label, nil
	}
	return nil, 0, fmt.Errorf("sequence shards contain no records")
}
