package workercontrol

import (
	"crypto/sha256"
	"crypto/subtle"
	"hash"
)

// StreamMetadata describes framing integrity only, not an ArtifactRef or grant.
// The owning context must authorize the bytes, size, and expected digest.
type StreamMetadata struct {
	ID     string            `json:"id"`
	Size   uint64            `json:"size"`
	SHA256 [sha256.Size]byte `json:"sha256"`
}

// StreamLimits are mandatory per-stream bounds. Adapters must additionally
// limit concurrent streams, total resources, and elapsed/idle time.
type StreamLimits struct {
	MaxBytes      uint64
	MaxChunkBytes uint32
	MaxChunks     uint64
}

// StreamVerifier hashes ordered chunks without retaining artifact bytes.
// It is single-owner, not concurrent-safe. Errors permanently invalidate it.
// There is no stream IO or artifact publication in this foundation.
type StreamVerifier struct {
	metadata StreamMetadata
	limits   StreamLimits
	hash     hash.Hash
	received uint64
	chunks   uint64
	closed   bool
}

func NewStreamVerifier(metadata StreamMetadata, limits StreamLimits) (*StreamVerifier, error) {
	if !validID(metadata.ID) || limits.MaxBytes == 0 || limits.MaxChunkBytes == 0 ||
		limits.MaxChunkBytes > MaxFrameBytes || limits.MaxChunks == 0 || metadata.Size > limits.MaxBytes {
		return nil, ErrStream
	}
	return &StreamVerifier{metadata: metadata, limits: limits, hash: sha256.New()}, nil
}

// Add checks stream identity and contiguous byte offsets. Empty chunks are
// forbidden; empty streams instead call Finish directly. Hashing does not
// imply that these bytes are safe to publish before Finish.
func (s *StreamVerifier) Add(id string, offset uint64, chunk []byte) error {
	if s.closed || s.hash == nil {
		return ErrStream
	}
	if id != s.metadata.ID || offset != s.received || len(chunk) == 0 ||
		uint64(len(chunk)) > uint64(s.limits.MaxChunkBytes) || s.chunks >= s.limits.MaxChunks ||
		uint64(len(chunk)) > s.metadata.Size-s.received {
		s.closed = true
		return ErrStream
	}
	_, _ = s.hash.Write(chunk)
	s.received += uint64(len(chunk))
	s.chunks++
	return nil
}

// Finish is one-use and requires exact length and SHA-256 digest agreement.
func (s *StreamVerifier) Finish() error {
	if s.closed || s.hash == nil {
		return ErrStream
	}
	s.closed = true
	if s.received != s.metadata.Size || subtle.ConstantTimeCompare(s.hash.Sum(nil), s.metadata.SHA256[:]) != 1 {
		return ErrStream
	}
	return nil
}
