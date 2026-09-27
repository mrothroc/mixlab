package workercontrol

import (
	"crypto/sha256"
	"errors"
	"testing"
)

func streamFor(t *testing.T, data string, limits StreamLimits) *StreamVerifier {
	t.Helper()
	s, err := NewStreamVerifier(StreamMetadata{ID: "stream-1", Size: uint64(len(data)), SHA256: sha256.Sum256([]byte(data))}, limits)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestStreamExactBoundsAndDigest(t *testing.T) {
	s := streamFor(t, "abcdef", StreamLimits{MaxBytes: 6, MaxChunkBytes: 3, MaxChunks: 2})
	chunk := []byte("abc")
	if err := s.Add("stream-1", 0, chunk); err != nil {
		t.Fatal(err)
	}
	chunk[0] = 'z' // No input slice is retained.
	if err := s.Add("stream-1", 3, []byte("def")); err != nil {
		t.Fatal(err)
	}
	if err := s.Finish(); err != nil {
		t.Fatal(err)
	}
	if err := s.Finish(); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
	if err := s.Add("stream-1", 6, []byte("g")); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
	empty := streamFor(t, "", StreamLimits{MaxBytes: 1, MaxChunkBytes: 1, MaxChunks: 1})
	if err := empty.Finish(); err != nil {
		t.Fatal(err)
	}
}

func TestStreamRejectsBoundsIdentityAndReplay(t *testing.T) {
	for _, tc := range []struct {
		name, id string
		offset   uint64
		data     string
	}{
		{"wrong stream", "other", 0, "a"},
		{"out of order", "stream-1", 1, "a"},
		{"overflow offset", "stream-1", ^uint64(0), "a"},
		{"empty chunk", "stream-1", 0, ""},
		{"oversized chunk", "stream-1", 0, "abcd"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := streamFor(t, "abcdef", StreamLimits{MaxBytes: 6, MaxChunkBytes: 3, MaxChunks: 2})
			if err := s.Add(tc.id, tc.offset, []byte(tc.data)); !errors.Is(err, ErrStream) {
				t.Fatal(err)
			}
			if err := s.Add("stream-1", 0, []byte("abc")); !errors.Is(err, ErrStream) {
				t.Fatal("error did not invalidate stream")
			}
			if err := s.Finish(); !errors.Is(err, ErrStream) {
				t.Fatal(err)
			}
		})
	}
	s := streamFor(t, "ab", StreamLimits{MaxBytes: 2, MaxChunkBytes: 3, MaxChunks: 2})
	if err := s.Add("stream-1", 0, []byte("abc")); !errors.Is(err, ErrStream) {
		t.Fatal("exceeded declared size")
	}
	s = streamFor(t, "ab", StreamLimits{MaxBytes: 2, MaxChunkBytes: 2, MaxChunks: 1})
	if err := s.Add("stream-1", 0, []byte("a")); err != nil {
		t.Fatal(err)
	}
	if err := s.Add("stream-1", 1, []byte("b")); !errors.Is(err, ErrStream) {
		t.Fatal("exceeded chunk count")
	}
	s = streamFor(t, "ab", StreamLimits{MaxBytes: 2, MaxChunkBytes: 2, MaxChunks: 2})
	if err := s.Add("stream-1", 0, []byte("a")); err != nil {
		t.Fatal(err)
	}
	if err := s.Add("stream-1", 0, []byte("a")); !errors.Is(err, ErrStream) {
		t.Fatal("accepted replayed chunk")
	}
}

func TestStreamFinishRejectsIncompleteAndCorrupt(t *testing.T) {
	for _, data := range []string{"", "ab", "abd"} {
		s := streamFor(t, "abc", StreamLimits{MaxBytes: 3, MaxChunkBytes: 3, MaxChunks: 1})
		if data != "" {
			if err := s.Add("stream-1", 0, []byte(data)); err != nil {
				t.Fatal(err)
			}
		}
		if err := s.Finish(); !errors.Is(err, ErrStream) {
			t.Fatal("accepted incomplete/corrupt stream")
		}
		if err := s.Finish(); !errors.Is(err, ErrStream) {
			t.Fatal(err)
		}
	}
	var zero StreamVerifier
	if err := zero.Finish(); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
	if err := zero.Add("stream-1", 0, []byte("a")); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
}

func TestInvalidStreamConfiguration(t *testing.T) {
	metadata := StreamMetadata{ID: "stream-1", Size: 3, SHA256: sha256.Sum256([]byte("abc"))}
	for _, limits := range []StreamLimits{
		{}, {MaxBytes: 2, MaxChunkBytes: 3, MaxChunks: 1},
		{MaxBytes: 3, MaxChunkBytes: 0, MaxChunks: 1},
		{MaxBytes: 3, MaxChunkBytes: 3, MaxChunks: 0},
		{MaxBytes: 3, MaxChunkBytes: MaxFrameBytes + 1, MaxChunks: 1},
	} {
		if _, err := NewStreamVerifier(metadata, limits); !errors.Is(err, ErrStream) {
			t.Fatal(err)
		}
	}
	metadata.Size = ^uint64(0)
	if _, err := NewStreamVerifier(metadata, StreamLimits{MaxBytes: 3, MaxChunkBytes: 3, MaxChunks: 1}); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
	metadata.ID = ""
	metadata.Size = 0
	if _, err := NewStreamVerifier(metadata, StreamLimits{MaxBytes: 3, MaxChunkBytes: 3, MaxChunks: 1}); !errors.Is(err, ErrStream) {
		t.Fatal(err)
	}
}
