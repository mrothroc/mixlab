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

// tokenStream reads tokens sequentially from shuffled binary shards.
type tokenStream struct {
	files     []string
	idx       int
	pos       int
	tok       []uint16
	rng       *rand.Rand
	chunkSize int
}

// newTokenStream opens shards matching pattern, shuffles them, and loads the first.
func newTokenStream(pattern string, seed int64, chunkSizeOpt ...int) (*tokenStream, error) {
	chunkSize := legacyShuffleChunkTokens
	if len(chunkSizeOpt) > 0 && chunkSizeOpt[0] > 0 {
		chunkSize = chunkSizeOpt[0]
	}
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no shard files matched %q; check the --train/--val glob and that prepare wrote .bin shards", pattern)
	}
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })

	toks, err := LoadDataShard(files[0])
	if err != nil {
		return nil, err
	}
	shuffleChunks(toks, chunkSize, rng)
	s := &tokenStream{files: files, tok: toks, rng: rng, chunkSize: chunkSize}
	if len(toks) > 10000 {
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

// Loader wraps a TokenStream to produce batches for training.
type Loader struct {
	stream *tokenStream
}

// NewLoader creates a Loader from a glob pattern for binary shard files.
// chunkSize controls the token-block shuffle granularity. Direct callers that
// pass a nonpositive chunkSize keep the historical 2048-token shuffle blocks;
// config-driven callers should pass seq_len when no explicit override is set.
func NewLoader(pattern string, seed int64, chunkSizeOpt ...int) (*Loader, error) {
	chunkSize := legacyShuffleChunkTokens
	if len(chunkSizeOpt) > 0 && chunkSizeOpt[0] > 0 {
		chunkSize = chunkSizeOpt[0]
	}
	s, err := newTokenStream(pattern, seed, chunkSize)
	if err != nil {
		return nil, err
	}
	return &Loader{stream: s}, nil
}

// NextBatch returns x (input tokens) and y (target tokens, shifted by 1).
// batchTokens must be positive and divisible by seqLen.
func (l *Loader) NextBatch(batchTokens, seqLen int) (x []int, y []int, err error) {
	if batchTokens <= 0 || seqLen <= 0 || batchTokens%seqLen != 0 {
		return nil, nil, fmt.Errorf("invalid batch shape: batchTokens=%d seqLen=%d; pass positive values with batchTokens divisible by seqLen", batchTokens, seqLen)
	}
	tok, err := l.stream.Take(batchTokens + 1)
	if err != nil {
		return nil, nil, err
	}
	x = make([]int, batchTokens)
	y = make([]int, batchTokens)
	for i := 0; i < batchTokens; i++ {
		x[i] = int(tok[i])
		y[i] = int(tok[i+1])
	}
	return x, y, nil
}

// ValBatch holds a single validation batch.
type ValBatch struct {
	X []int
	Y []int
}

// ValSet holds fixed batches for repeatable evaluation.
type ValSet struct {
	Batches []ValBatch
}

// NewValSet loads nBatches fixed batches from a validation shard.
func NewValSet(pattern string, seed int64, nBatches, batchTokens, seqLen int, chunkSizeOpt ...int) (*ValSet, error) {
	chunkSize := seqLen
	if len(chunkSizeOpt) > 0 && chunkSizeOpt[0] > 0 {
		chunkSize = chunkSizeOpt[0]
	}
	loader, err := NewLoader(pattern, seed, chunkSize)
	if err != nil {
		return nil, err
	}
	vs := &ValSet{Batches: make([]ValBatch, 0, nBatches)}
	for i := 0; i < nBatches; i++ {
		x, y, err := loader.NextBatch(batchTokens, seqLen)
		if err != nil {
			break
		}
		vs.Batches = append(vs.Batches, ValBatch{X: x, Y: y})
	}
	if len(vs.Batches) == 0 {
		return nil, fmt.Errorf("no validation batches loaded from %q; check the validation glob or reduce seq_len/batch_tokens", pattern)
	}
	return vs, nil
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
