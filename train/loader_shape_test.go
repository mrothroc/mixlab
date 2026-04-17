package train

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

const (
	testShardMagic   = 20240520
	testShardVersion = 1
	testHeaderInts   = 256
)

func writeTestShard(t *testing.T, dir, name string, tokens []uint16) {
	t.Helper()

	path := filepath.Join(dir, name)
	buf := make([]byte, testHeaderInts*4+len(tokens)*2)
	header := make([]int32, testHeaderInts)
	header[0] = testShardMagic
	header[1] = testShardVersion
	header[2] = int32(len(tokens))
	for i, v := range header {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(v))
	}
	off := testHeaderInts * 4
	for i, tok := range tokens {
		binary.LittleEndian.PutUint16(buf[off+i*2:], tok)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestNextBatchShapeBatchTokens2048SeqLen2048(t *testing.T) {
	loader := newShapeTestLoader(t)

	const batchTokens = 2048
	const seqLen = 2048
	x, y, err := loader.NextBatch(batchTokens, seqLen)
	if err != nil {
		t.Fatalf("NextBatch: %v", err)
	}
	if got, want := len(x)/seqLen, 1; got != want {
		t.Fatalf("sequence count = %d, want %d", got, want)
	}
	if len(x) != batchTokens || len(y) != batchTokens {
		t.Fatalf("len(x)=%d len(y)=%d, want %d", len(x), len(y), batchTokens)
	}
}

func TestNextBatchShapeBatchTokens2048SeqLen1024(t *testing.T) {
	loader := newShapeTestLoader(t)

	const batchTokens = 2048
	const seqLen = 1024
	x, y, err := loader.NextBatch(batchTokens, seqLen)
	if err != nil {
		t.Fatalf("NextBatch: %v", err)
	}
	if got, want := len(x)/seqLen, 2; got != want {
		t.Fatalf("sequence count = %d, want %d", got, want)
	}
	if len(x) != batchTokens || len(y) != batchTokens {
		t.Fatalf("len(x)=%d len(y)=%d, want %d", len(x), len(y), batchTokens)
	}
}

func newShapeTestLoader(t *testing.T) *data.Loader {
	t.Helper()

	dir := t.TempDir()
	tokens := make([]uint16, 2049)
	for i := range tokens {
		tokens[i] = uint16(i % 1024)
	}
	writeTestShard(t, dir, "shard_00.bin", tokens)

	loader, err := data.NewLoader(filepath.Join(dir, "shard_*.bin"), 0, 2048)
	if err != nil {
		t.Fatalf("NewLoader: %v", err)
	}
	return loader
}

func TestLoaderShuffleChunkSizeChangesTokenOrder(t *testing.T) {
	dir := t.TempDir()
	tokens := make([]uint16, 8192)
	for i := range tokens {
		tokens[i] = uint16(i)
	}
	pattern := filepath.Join(dir, "shard_*.bin")
	writeTestShard(t, dir, "shard_00.bin", tokens)

	loader1024, err := data.NewLoader(pattern, 7, 1024)
	if err != nil {
		t.Fatalf("NewLoader chunk 1024: %v", err)
	}
	loader2048, err := data.NewLoader(pattern, 7, 2048)
	if err != nil {
		t.Fatalf("NewLoader chunk 2048: %v", err)
	}

	x1024, _, err := loader1024.NextBatch(4096, 1024)
	if err != nil {
		t.Fatalf("NextBatch chunk 1024: %v", err)
	}
	x2048, _, err := loader2048.NextBatch(4096, 1024)
	if err != nil {
		t.Fatalf("NextBatch chunk 2048: %v", err)
	}
	if equalIntSlices(x1024, x2048) {
		t.Fatal("token order matched for chunk_size=1024 and chunk_size=2048; shuffle granularity did not affect the stream")
	}
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
