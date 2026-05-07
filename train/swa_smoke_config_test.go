package train

import (
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestSWAWindow128SmokeConfigCoversExampleTokens(t *testing.T) {
	cfg, err := LoadArchConfig(filepath.Join("experiments", "swa_test", "window128_smoke.json"))
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	files, err := filepath.Glob(filepath.Join("..", "data", "example", "train_*.bin"))
	if err != nil {
		t.Fatalf("Glob train shards: %v", err)
	}
	if len(files) == 0 {
		t.Skip("no example train shards found (data/example is gitignored; skipping in environments without local data)")
	}

	maxToken := 0
	for _, file := range files {
		tokens, err := data.LoadDataShard(file)
		if err != nil {
			t.Fatalf("LoadDataShard(%s): %v", file, err)
		}
		for _, token := range tokens {
			if int(token) > maxToken {
				maxToken = int(token)
			}
		}
	}
	if maxToken >= cfg.VocabSize {
		t.Fatalf("example train token id %d exceeds config vocab_size=%d", maxToken, cfg.VocabSize)
	}
}
