// Package data provides binary shard data loading for training and evaluation.
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
	shardMagic               = 20240520
	shardVersion             = 1
	headerInts               = 256
	legacyShuffleChunkTokens = 2048
)

// LoaderOptions controls training/eval shard traversal.
type LoaderOptions struct {
	ChunkSize      int
	NoShardShuffle bool
	Framing        ExampleFraming
}

// ExampleFraming configures loader-side BOS/EOS wrapping for raw token
// streams. A zero ContentLen disables framing.
type ExampleFraming struct {
	ContentLen int
	BosID      int
	EosID      int
}

func (f ExampleFraming) Enabled() bool {
	return f.ContentLen > 0
}

// tokenStream reads tokens sequentially from binary shards.
type tokenStream struct {
	files     []string
	idx       int
	pos       int
	tok       []uint16
	rng       *rand.Rand
	chunkSize int
}

// newTokenStream opens shards matching pattern, applies default shard shuffling,
// and loads the first shard.
func newTokenStream(pattern string, seed int64) (*tokenStream, error) {
	return newTokenStreamWithOptions(pattern, seed, loaderOptionsFromChunkSize())
}

func newTokenStreamWithOptions(pattern string, seed int64, opts LoaderOptions) (*tokenStream, error) {
	opts = normalizeLoaderOptions(opts, legacyShuffleChunkTokens)
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no shard files matched %q; check the --train/--val glob and that prepare wrote .bin shards", pattern)
	}
	rng := rand.New(rand.NewSource(seed))
	if !opts.NoShardShuffle {
		rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	}

	toks, err := LoadDataShard(files[0])
	if err != nil {
		return nil, err
	}
	shuffleChunks(toks, opts.ChunkSize, rng)
	s := &tokenStream{files: files, tok: toks, rng: rng, chunkSize: opts.ChunkSize}
	if !opts.Framing.Enabled() && len(toks) > 10000 {
		s.pos = rng.Intn(len(toks) / 2)
	}
	return s, nil
}

func (s *tokenStream) advance() error {
	s.idx = (s.idx + 1) % len(s.files)
	t, err := LoadDataShard(s.files[s.idx])
	if err != nil {
		return err
	}
	shuffleChunks(t, s.chunkSize, s.rng)
	s.tok = t
	s.pos = 0
	return nil
}

// shuffleChunks shuffles contiguous blocks of chunkSize tokens.
func shuffleChunks(toks []uint16, chunkSize int, rng *rand.Rand) {
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}
	n := len(toks) / chunkSize
	if n <= 1 {
		return
	}
	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		for k := 0; k < chunkSize; k++ {
			toks[i*chunkSize+k], toks[j*chunkSize+k] = toks[j*chunkSize+k], toks[i*chunkSize+k]
		}
	}
}

// Take returns the next n tokens from the stream, advancing shards as needed.
func (s *tokenStream) Take(n int) ([]uint16, error) {
	if n <= 0 {
		return nil, fmt.Errorf("token count must be > 0; pass a positive batch size")
	}
	out := make([]uint16, 0, n)
	for len(out) < n {
		if s.pos >= len(s.tok) {
			if err := s.advance(); err != nil {
				return nil, err
			}
		}
		k := n - len(out)
		avail := len(s.tok) - s.pos
		if k > avail {
			k = avail
		}
		out = append(out, s.tok[s.pos:s.pos+k]...)
		s.pos += k
	}
	return out, nil
}

// TakeAlignedChunk returns the next complete n-token chunk. It skips ragged
// shard tails and never joins content across shard boundaries.
func (s *tokenStream) TakeAlignedChunk(n int) ([]uint16, error) {
	if n <= 0 {
		return nil, fmt.Errorf("token count must be > 0; pass a positive chunk size")
	}
	for attempts := 0; attempts <= len(s.files); attempts++ {
		usable := (len(s.tok) / n) * n
		if s.pos+n <= usable {
			out := append([]uint16(nil), s.tok[s.pos:s.pos+n]...)
			s.pos += n
			return out, nil
		}
		if err := s.advance(); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("no shard contains a complete %d-token framed content chunk", n)
}

// Loader wraps a TokenStream to produce batches for training.
type Loader struct {
	stream    *tokenStream
	sequences *sequenceStream
	framing   ExampleFraming
	packing   *sequencePacking
}

type sequencePacking struct {
	padID        int
	bosID        int
	eosID        int
	layout       string
	recordSeqLen int
}

// Batch is a fixed-shape loader result. Optional masks are nil for legacy flat
// token streams, preserving the historical data path exactly.
type Batch struct {
	X            []int
	Y            []int
	LossMask     []float32
	SegmentIDs   []int32
	MaskEligible []uint8
	Labels       []int32
	ValidMask    []float32
}

// NewLoader creates a Loader from a glob pattern for binary shard files.
// chunkSize controls the token-block shuffle granularity. Direct callers that
// pass a nonpositive chunkSize keep the historical 2048-token shuffle blocks;
// config-driven callers should pass seq_len when no explicit override is set.
func NewLoader(pattern string, seed int64, chunkSizeOpt ...int) (*Loader, error) {
	return NewLoaderWithOptions(pattern, seed, loaderOptionsFromChunkSize(chunkSizeOpt...))
}

func loaderOptionsFromChunkSize(chunkSizeOpt ...int) LoaderOptions {
	opts := LoaderOptions{ChunkSize: legacyShuffleChunkTokens}
	if len(chunkSizeOpt) > 0 && chunkSizeOpt[0] > 0 {
		opts.ChunkSize = chunkSizeOpt[0]
	}
	return opts
}

func normalizeLoaderOptions(opts LoaderOptions, defaultChunkSize int) LoaderOptions {
	if opts.ChunkSize <= 0 {
		opts.ChunkSize = defaultChunkSize
	}
	if opts.Framing.Enabled() {
		opts.ChunkSize = opts.Framing.ContentLen
	}
	return opts
}

// NewLoaderWithOptions creates a Loader from a glob pattern for binary shard
// files with explicit data-loader options.
func NewLoaderWithOptions(pattern string, seed int64, opts LoaderOptions) (*Loader, error) {
	manifest, _, found, err := FindDatasetManifest(pattern)
	if err != nil {
		return nil, err
	}
	if found && (manifest.ShardFormat == DatasetShardFormatSequenceV1 || manifest.ShardFormat == DatasetShardFormatLabeledSequenceV1) {
		pad, padOK := manifest.SpecialTokenIDs["pad"]
		bos, bosOK := manifest.SpecialTokenIDs["bos"]
		eos, eosOK := manifest.SpecialTokenIDs["eos"]
		if !padOK || !bosOK || !eosOK {
			return nil, fmt.Errorf("sequence dataset manifest requires semantic special_token_ids pad, bos, and eos")
		}
		sequences, err := newSequenceStreamWithFormat(pattern, seed, opts.NoShardShuffle, manifest.ShardFormat)
		if err != nil {
			return nil, err
		}
		return &Loader{sequences: sequences, packing: &sequencePacking{
			padID: pad, bosID: bos, eosID: eos,
			layout: manifest.EffectiveSequenceLayout(), recordSeqLen: manifest.RecordSeqLen,
		}}, nil
	}
	s, err := newTokenStreamWithOptions(pattern, seed, opts)
	if err != nil {
		return nil, err
	}
	return &Loader{stream: s, framing: opts.Framing}, nil
}

// NextBatch returns x (input tokens) and y (target tokens, shifted by 1).
// batchTokens must be positive and divisible by seqLen.
func (l *Loader) NextBatch(batchTokens, seqLen int) (x []int, y []int, err error) {
	batch, err := l.NextBatchDetailed(batchTokens, seqLen)
	if err != nil {
		return nil, nil, err
	}
	return batch.X, batch.Y, nil
}

// NextBatchDetailed returns token rows plus optional representation metadata.
func (l *Loader) NextBatchDetailed(batchTokens, seqLen int) (Batch, error) {
	if batchTokens <= 0 || seqLen <= 0 || batchTokens%seqLen != 0 {
		return Batch{}, fmt.Errorf("invalid batch shape: batchTokens=%d seqLen=%d; pass positive values with batchTokens divisible by seqLen", batchTokens, seqLen)
	}
	if l.sequences != nil {
		if l.packing.layout == DatasetSequenceLayoutOneRecordRow {
			return l.nextRecordBatch(batchTokens, seqLen)
		}
		return l.nextSequenceBatch(batchTokens, seqLen)
	}
	if l.framing.Enabled() {
		x, y, err := l.nextFramedBatch(batchTokens, seqLen)
		return Batch{X: x, Y: y, LossMask: FramedCausalLossMask(batchTokens, seqLen)}, err
	}
	tok, err := l.stream.Take(batchTokens + 1)
	if err != nil {
		return Batch{}, err
	}
	x := make([]int, batchTokens)
	y := make([]int, batchTokens)
	for i := 0; i < batchTokens; i++ {
		x[i] = int(tok[i])
		y[i] = int(tok[i+1])
	}
	return Batch{X: x, Y: y}, nil
}

func (l *Loader) nextRecordBatch(batchTokens, seqLen int) (Batch, error) {
	if seqLen != l.packing.recordSeqLen {
		return Batch{}, fmt.Errorf("one-record-per-row dataset requires seq_len=%d, got %d", l.packing.recordSeqLen, seqLen)
	}
	x := make([]int, batchTokens)
	y := make([]int, batchTokens)
	lossMask := make([]float32, batchTokens)
	maskEligible := make([]uint8, batchTokens)
	var validMask []float32
	var segmentIDs []int32
	var labels []int32
	if l.sequences.labeled {
		validMask = make([]float32, batchTokens)
		segmentIDs = make([]int32, batchTokens)
		labels = make([]int32, batchTokens/seqLen)
	}
	for rowStart := 0; rowStart < batchTokens; rowStart += seqLen {
		record, label, err := l.sequences.takeLabeledRecord()
		if err != nil {
			return Batch{}, err
		}
		if len(record) > seqLen-2 {
			return Batch{}, fmt.Errorf("record has %d tokens but seq_len=%d permits at most %d; re-run prepare with an explicit record overflow policy", len(record), seqLen, seqLen-2)
		}
		rowEnd := rowStart + seqLen
		for i := rowStart; i < rowEnd; i++ {
			x[i], y[i] = l.packing.padID, l.packing.padID
		}
		x[rowStart] = l.packing.bosID
		if l.sequences.labeled {
			validMask[rowStart] = 1
		}
		for i, token := range record {
			x[rowStart+1+i] = int(token)
			maskEligible[rowStart+1+i] = 1
			if l.sequences.labeled {
				validMask[rowStart+1+i] = 1
			}
		}
		eosInput := rowStart + len(record) + 1
		x[eosInput] = l.packing.eosID
		if l.sequences.labeled {
			validMask[eosInput] = 1
			labels[rowStart/seqLen] = label
		}
		for i := rowStart; i < eosInput; i++ {
			y[i] = x[i+1]
			lossMask[i] = 1
		}
		if l.sequences.labeled {
			for i := eosInput + 1; i < rowEnd; i++ {
				segmentIDs[i] = 1
			}
		}
	}
	return Batch{
		X: x, Y: y, LossMask: lossMask, SegmentIDs: segmentIDs,
		MaskEligible: maskEligible, Labels: labels, ValidMask: validMask,
	}, nil
}

func (l *Loader) nextSequenceBatch(batchTokens, seqLen int) (Batch, error) {
	if seqLen < 3 {
		return Batch{}, fmt.Errorf("record-oriented sequence datasets require seq_len >= 3 for BOS/base/EOS framing")
	}
	x := make([]int, batchTokens)
	y := make([]int, batchTokens)
	lossMask := make([]float32, batchTokens)
	segmentIDs := make([]int32, batchTokens)
	maskEligible := make([]uint8, batchTokens)
	for rowStart := 0; rowStart < batchTokens; rowStart += seqLen {
		cursor := rowStart
		rowEnd := rowStart + seqLen
		segment := int32(0)
		for cursor < rowEnd {
			remaining := rowEnd - cursor
			if remaining == 1 {
				x[cursor], y[cursor] = l.packing.padID, l.packing.padID
				segmentIDs[cursor] = segment
				cursor++
				continue
			}
			content, err := l.sequences.takeChunk(remaining - 2)
			if err != nil {
				return Batch{}, err
			}
			start := cursor
			x[cursor] = l.packing.bosID
			segmentIDs[cursor] = segment
			cursor++
			for _, token := range content {
				x[cursor] = int(token)
				segmentIDs[cursor] = segment
				maskEligible[cursor] = 1
				cursor++
			}
			x[cursor] = l.packing.eosID
			segmentIDs[cursor] = segment
			cursor++
			if len(content) > 0 {
				for i := start; i < cursor-1; i++ {
					y[i] = x[i+1]
					lossMask[i] = 1
				}
			}
			y[cursor-1] = l.packing.eosID
			segment++
		}
	}
	return Batch{X: x, Y: y, LossMask: lossMask, SegmentIDs: segmentIDs, MaskEligible: maskEligible}, nil
}

func (l *Loader) nextFramedBatch(batchTokens, seqLen int) (x []int, y []int, err error) {
	f := l.framing
	if f.ContentLen <= 0 {
		return nil, nil, fmt.Errorf("invalid framed content length=%d", f.ContentLen)
	}
	if seqLen != f.ContentLen+2 {
		return nil, nil, fmt.Errorf("invalid framed batch shape: seqLen=%d content_len=%d; seqLen must equal content_len+2", seqLen, f.ContentLen)
	}
	batchSize := batchTokens / seqLen
	x = make([]int, batchTokens)
	y = make([]int, batchTokens)
	for b := 0; b < batchSize; b++ {
		content, err := l.stream.TakeAlignedChunk(f.ContentLen)
		if err != nil {
			return nil, nil, err
		}
		row := b * seqLen
		x[row] = f.BosID
		for i, tok := range content {
			x[row+1+i] = int(tok)
		}
		x[row+seqLen-1] = f.EosID
		for i := 0; i < seqLen-1; i++ {
			y[row+i] = x[row+i+1]
		}
		y[row+seqLen-1] = f.EosID
	}
	return x, y, nil
}

// ValBatch holds a single validation batch.
type ValBatch struct {
	X            []int
	Y            []int
	LossMask     []float32
	SegmentIDs   []int32
	MaskEligible []uint8
	Labels       []int32
	ValidMask    []float32
}

// ValSet holds fixed batches for repeatable evaluation.
type ValSet struct {
	Batches []ValBatch
}

// NewValSet loads nBatches fixed batches from a validation shard.
func NewValSet(pattern string, seed int64, nBatches, batchTokens, seqLen int, chunkSizeOpt ...int) (*ValSet, error) {
	opts := LoaderOptions{ChunkSize: seqLen}
	if len(chunkSizeOpt) > 0 && chunkSizeOpt[0] > 0 {
		opts.ChunkSize = chunkSizeOpt[0]
	}
	return NewValSetWithOptions(pattern, seed, nBatches, batchTokens, seqLen, opts)
}

// NewValSetWithOptions loads nBatches fixed validation batches using explicit
// loader options.
func NewValSetWithOptions(pattern string, seed int64, nBatches, batchTokens, seqLen int, opts LoaderOptions) (*ValSet, error) {
	opts = normalizeLoaderOptions(opts, seqLen)
	loader, err := NewLoaderWithOptions(pattern, seed, opts)
	if err != nil {
		return nil, err
	}
	vs := &ValSet{Batches: make([]ValBatch, 0, nBatches)}
	for i := 0; i < nBatches; i++ {
		batch, err := loader.NextBatchDetailed(batchTokens, seqLen)
		if err != nil {
			break
		}
		vs.Batches = append(vs.Batches, ValBatch(batch))
	}
	if len(vs.Batches) == 0 {
		return nil, fmt.Errorf("no validation batches loaded from %q; check the validation glob or reduce seq_len/batch_tokens", pattern)
	}
	return vs, nil
}

// FramedCausalLossMask masks the final input position of every example row,
// where no within-example next-token target exists.
func FramedCausalLossMask(batchTokens, seqLen int) []float32 {
	if batchTokens <= 0 || seqLen <= 0 || batchTokens%seqLen != 0 {
		return nil
	}
	mask := make([]float32, batchTokens)
	for rowStart := 0; rowStart < batchTokens; rowStart += seqLen {
		for pos := 0; pos < seqLen-1; pos++ {
			mask[rowStart+pos] = 1
		}
	}
	return mask
}

// LoadDataShard reads a single binary shard file.
// Format: 256 int32 header (magic, version, nTok, ...) + nTok uint16 tokens.
func LoadDataShard(path string) ([]uint16, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	headerBytes := headerInts * 4
	if len(blob) < headerBytes {
		return nil, fmt.Errorf("shard %q is too small to contain a mixlab header; regenerate it with `mixlab -mode prepare`", path)
	}
	head := make([]int32, headerInts)
	for i := 0; i < headerInts; i++ {
		head[i] = int32(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	if int(head[0]) != shardMagic || int(head[1]) != shardVersion {
		return nil, fmt.Errorf("shard %q has an unexpected header; make sure it was produced by the current `mixlab -mode prepare` format", path)
	}
	nTok := int(head[2])
	expected := headerBytes + nTok*2
	if len(blob) != expected {
		return nil, fmt.Errorf("shard %q size mismatch: got=%d bytes want=%d bytes; re-run `mixlab -mode prepare` for this dataset", path, len(blob), expected)
	}
	toks := make([]uint16, nTok)
	off := headerBytes
	for i := 0; i < nTok; i++ {
		toks[i] = binary.LittleEndian.Uint16(blob[off+i*2:])
	}
	return toks, nil
}

// LoadValidationTokens loads all shards matching pattern in sorted order and
// trims to a length compatible with next-token evaluation at seqLen.
func LoadValidationTokens(pattern string, seqLen int) ([]uint16, error) {
	if seqLen <= 0 {
		return nil, fmt.Errorf("seqLen must be > 0; pass the same positive seq_len used for training/eval")
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no validation shard files matched %q; check the --val glob and shard directory", pattern)
	}
	all := make([]uint16, 0, 1024)
	for _, file := range files {
		toks, err := LoadDataShard(file)
		if err != nil {
			return nil, err
		}
		all = append(all, toks...)
	}
	usable := ((len(all) - 1) / seqLen) * seqLen
	if usable <= 0 {
		return nil, fmt.Errorf("validation split is too short for seqLen=%d; add more validation tokens or lower seq_len", seqLen)
	}
	return all[:usable+1], nil
}

// BPBFromNats converts average negative log-likelihood (nats) to bits-per-byte.
func BPBFromNats(nll float64) float64 {
	const ln2 = 0.6931471805599453
	return nll / ln2
}
