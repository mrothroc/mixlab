package train

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIdentityFirstByteMaskValid_UTF8Classes(t *testing.T) {
	valid := identityFirstByteMaskValid(256)
	cases := []struct {
		id   int
		want int32
	}{
		{0x00, 1},
		{0x7f, 1},
		{0x80, 0},
		{0xbf, 0},
		{0xc0, 0},
		{0xc2, 1},
		{0xf4, 1},
		{0xf5, 0},
	}
	for _, tc := range cases {
		if got := valid[tc.id]; got != tc.want {
			t.Fatalf("valid[%#x] = %d, want %d", tc.id, got, tc.want)
		}
	}
}

func TestFirstByteMaskValidFromTokenizer_ByteLevelVocab(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	blob := []byte(`{
		"added_tokens": [{"id": 0, "content": "<|pad|>", "special": true}],
		"model": {
			"vocab": {
				"<|pad|>": 0,
				"A": 1,
				"Ġ": 2,
				"Ģ": 3,
				"Ã": 4
			}
		}
	}`)
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatalf("Write tokenizer: %v", err)
	}

	valid, err := firstByteMaskValidFromTokenizer(path, 5)
	if err != nil {
		t.Fatalf("firstByteMaskValidFromTokenizer: %v", err)
	}
	want := []int32{0, 1, 1, 0, 1}
	for i := range want {
		if valid[i] != want[i] {
			t.Fatalf("valid[%d] = %d, want %d", i, valid[i], want[i])
		}
	}
}
