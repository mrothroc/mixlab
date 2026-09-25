package data

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// DistributedDatasetIdentity uses logical names and content, not host paths,
// timestamps or inode numbers. Files must be immutable during a run.
func DistributedDatasetIdentity(pattern string) (string, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return "", fmt.Errorf("no training shards match %q", pattern)
	}
	manifest, _, _, err := FindDatasetManifest(pattern)
	if err != nil {
		return "", err
	}
	type entry struct {
		Name   string
		Bytes  int64
		SHA256 string
	}
	entries := make([]entry, 0, len(files))
	seen := map[string]bool{}
	for _, path := range files {
		name := filepath.Base(path)
		if seen[name] {
			return "", fmt.Errorf("duplicate logical shard name %q", name)
		}
		seen[name] = true
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		h := sha256.New()
		n, err := io.Copy(h, f)
		closeErr := f.Close()
		if err != nil {
			return "", err
		}
		if closeErr != nil {
			return "", closeErr
		}
		entries = append(entries, entry{name, n, hex.EncodeToString(h.Sum(nil))})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	blob, err := json.Marshal(struct {
		Version  string
		Manifest *DatasetManifest
		Shards   []entry
	}{"mixlab-distributed-dataset-v1", manifest, entries})
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(blob)
	return hex.EncodeToString(h[:]), nil
}

// DistributedSamplerState is rank-neutral so the rank-zero checkpoint can
// restore each rank's own partition without storing duplicate sampler states.
type DistributedSamplerState struct {
	DatasetID     string `json:"dataset_id"`
	Seed          int64  `json:"seed"`
	Epoch         uint64 `json:"epoch"`
	GlobalCursor  uint64 `json:"global_cursor"`
	PartitionID   string `json:"partition_id"`
	PartitionSize int    `json:"partition_size"`
	ChunkTokens   int    `json:"chunk_tokens"`
	ChunkOffset   int    `json:"chunk_offset"`
}

// DistributedLoader owns fixed, aligned token chunks. Each rank uses the same
// counter permutation and a distinct strided ordinal; no prefetch RNG or replay.
// Only one shard is resident per rank, regardless of dataset size.
type DistributedLoader struct {
	files                       []string
	ends                        []int
	state                       DistributedSamplerState
	rank, seqLen, vocab, usable int
	cached                      int
	tokens                      []uint16
	records                     [][]uint16
	manifest                    *DatasetManifest
}

func NewDistributedLoader(pattern string, seed int64, world, rank, seqLen, chunkTokens, vocab int, partitionID string) (*DistributedLoader, error) {
	if world < 2 || rank < 0 || rank >= world || seqLen <= 0 || chunkTokens < seqLen || chunkTokens%seqLen != 0 || vocab <= 0 || partitionID == "" {
		return nil, fmt.Errorf("invalid distributed loader topology/shape; shuffle_chunk_tokens must be a positive multiple of seq_len")
	}
	m, _, found, err := FindDatasetManifest(pattern)
	if err != nil {
		return nil, err
	}
	recordMode := found && m.ShardFormat == DatasetShardFormatSequenceV1 && m.EffectiveSequenceLayout() == DatasetSequenceLayoutOneRecordRow
	if found && !recordMode && (m.ShardFormat != DatasetShardFormatTokenStreamV1 || m.EffectiveSequenceLayout() != DatasetSequenceLayoutContinuousStream) {
		return nil, fmt.Errorf("R1 DDP requires continuous token shards or one_record_per_row sequence shards")
	}
	if recordMode && m.RecordSeqLen != seqLen {
		return nil, fmt.Errorf("record_seq_len %d differs from seq_len %d", m.RecordSeqLen, seqLen)
	}
	id, err := DistributedDatasetIdentity(pattern)
	if err != nil {
		return nil, err
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	l := &DistributedLoader{files: files, rank: rank, seqLen: seqLen, vocab: vocab, cached: -1, state: DistributedSamplerState{DatasetID: id, Seed: seed, PartitionID: partitionID, PartitionSize: world, ChunkTokens: chunkTokens}}
	if recordMode {
		l.manifest = m
		l.state.ChunkTokens = seqLen // A record, never a partial record, is one sample.
	}
	total := 0
	for _, path := range files {
		if recordMode {
			records, err := LoadSequenceShard(path)
			if err != nil {
				return nil, err
			}
			for _, record := range records {
				if len(record) > seqLen-2 {
					return nil, fmt.Errorf("shard %q record exceeds seq_len-2", path)
				}
				for _, token := range record {
					if int(token) >= vocab {
						return nil, fmt.Errorf("shard %q token %d outside vocab_size %d", path, token, vocab)
					}
				}
			}
			total += len(records)
			l.ends = append(l.ends, total)
			continue
		}
		tokens, err := LoadDataShard(path)
		if err != nil {
			return nil, err
		}
		for _, t := range tokens {
			if int(t) >= vocab {
				return nil, fmt.Errorf("shard %q token %d outside vocab_size %d", path, t, vocab)
			}
		}
		// Reserve the next-token target without crossing a shard boundary.
		total += max(0, (len(tokens)-1)/chunkTokens)
		l.ends = append(l.ends, total)
	}
	l.usable = (total / world) * world
	if l.usable == 0 {
		return nil, fmt.Errorf("dataset has %d aligned chunks, fewer than world size %d", total, world)
	}
	return l, nil
}

func (l *DistributedLoader) State() DistributedSamplerState { return l.state }

func (l *DistributedLoader) Restore(s DistributedSamplerState) error {
	w := l.state
	if s.DatasetID != w.DatasetID || s.Seed != w.Seed || s.PartitionID != w.PartitionID || s.PartitionSize != w.PartitionSize || s.ChunkTokens != w.ChunkTokens || s.GlobalCursor >= uint64(l.usable) || s.GlobalCursor%uint64(w.PartitionSize) != 0 || s.ChunkOffset < 0 || s.ChunkOffset >= w.ChunkTokens || s.ChunkOffset%l.seqLen != 0 {
		return fmt.Errorf("distributed sampler state is incompatible or has an invalid cursor")
	}
	l.state = s
	return nil
}

func (l *DistributedLoader) NextBatch(batchTokens int) (Batch, error) {
	if batchTokens <= 0 || batchTokens%l.seqLen != 0 {
		return Batch{}, fmt.Errorf("distributed batch_tokens must be positive and divisible by seq_len")
	}
	b := Batch{X: make([]int, batchTokens), Y: make([]int, batchTokens), SeqLen: l.seqLen, BatchSize: batchTokens / l.seqLen, ExampleCount: batchTokens / l.seqLen}
	if l.manifest != nil {
		b.LossMask = make([]float32, batchTokens)
		b.MaskEligible = make([]uint8, batchTokens)
	}
	for row := 0; row < b.BatchSize; row++ {
		key := uint64(l.state.Seed) ^ (l.state.Epoch+1)*0x9e3779b97f4a7c15
		sample := int(distributedSamplePermutation(l.state.GlobalCursor+uint64(l.rank), uint64(l.ends[len(l.ends)-1]), key))
		shard := sort.SearchInts(l.ends, sample+1)
		if l.cached != shard {
			var err error
			if l.manifest != nil {
				l.records, err = LoadSequenceShard(l.files[shard])
			} else {
				l.tokens, err = LoadDataShard(l.files[shard])
			}
			if err != nil {
				return Batch{}, err
			}
			l.cached = shard
		}
		base := 0
		if shard > 0 {
			base = l.ends[shard-1]
		}
		if err := l.fillRow(&b, row, sample-base); err != nil {
			return Batch{}, err
		}
		l.state.ChunkOffset += l.seqLen
		if l.state.ChunkOffset == l.state.ChunkTokens {
			l.state.ChunkOffset = 0
			l.state.GlobalCursor += uint64(l.state.PartitionSize)
			if l.state.GlobalCursor == uint64(l.usable) {
				l.state.GlobalCursor = 0
				l.state.Epoch++
			}
		}
	}
	return b, nil
}

func (l *DistributedLoader) fillRow(b *Batch, row, sample int) error {
	if l.manifest != nil {
		if sample >= len(l.records) {
			return fmt.Errorf("distributed shard changed during training")
		}
		ids := l.manifest.SpecialTokenIDs
		return frameRecordRow(b, row, l.records[sample], 0, l.seqLen, ids["pad"], ids["bos"], ids["eos"], false)
	}
	pos := sample*l.state.ChunkTokens + l.state.ChunkOffset
	if pos+l.seqLen >= len(l.tokens) {
		return fmt.Errorf("distributed shard changed during training")
	}
	for j := 0; j < l.seqLen; j++ {
		x, y := int(l.tokens[pos+j]), int(l.tokens[pos+j+1])
		if x >= l.vocab || y >= l.vocab {
			return fmt.Errorf("distributed shard token out of bounds")
		}
		b.X[row*l.seqLen+j], b.Y[row*l.seqLen+j] = x, y
	}
	return nil
}
