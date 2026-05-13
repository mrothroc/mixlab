// Package ranks defines the binary file format for per-token target ranks
// emitted by mixlab eval mode (-ranks-out).
//
// Each record stores the target token ID at an evaluation position together
// with the rank of that target under the model's softmax distribution. Rank
// is 0-indexed: 0 means the target was the model's argmax (top-1 correct);
// vocab-1 means the target had the lowest logit. When read alongside the
// matching logprobs.bin produced in the same eval pass, records align
// position-by-position with logprobs.Record.
package ranks

import (
	"encoding/binary"
	"fmt"
	"io"
)

const (
	// Magic is the 4-byte file identifier: ASCII "RANK".
	Magic uint32 = 0x52414E4B
	// Version is the on-disk format version.
	Version uint32 = 1
)

// Header is the fixed 16-byte file header.
type Header struct {
	Magic       uint32
	Version     uint32
	VocabSize   uint32
	TotalTokens uint32
}

// Record is a single eval position: the target token ID and its rank under
// the model's softmax distribution.
type Record struct {
	TokenID uint16
	Rank    uint16
}

// Writer is a streaming writer that emits a header followed by records and
// validates that the record count matches the header on Close.
type Writer struct {
	w       io.Writer
	header  Header
	written uint32
}

// NewWriter writes the header and returns a Writer ready to accept exactly
// totalTokens records.
func NewWriter(w io.Writer, vocabSize, totalTokens uint32) (*Writer, error) {
	if w == nil {
		return nil, fmt.Errorf("nil writer")
	}
	header := Header{
		Magic:       Magic,
		Version:     Version,
		VocabSize:   vocabSize,
		TotalTokens: totalTokens,
	}
	if err := binary.Write(w, binary.LittleEndian, header); err != nil {
		return nil, err
	}
	return &Writer{w: w, header: header}, nil
}

// Append writes one record.
func (w *Writer) Append(tokenID, rank uint16) error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written >= w.header.TotalTokens {
		return fmt.Errorf("too many records: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	if err := binary.Write(w.w, binary.LittleEndian, tokenID); err != nil {
		return err
	}
	if err := binary.Write(w.w, binary.LittleEndian, rank); err != nil {
		return err
	}
	w.written++
	return nil
}

// AppendBatch writes one record per (tokenID, rank) pair.
func (w *Writer) AppendBatch(tokenIDs, ranksOut []uint16) error {
	if len(tokenIDs) != len(ranksOut) {
		return fmt.Errorf("token/rank length mismatch: %d != %d", len(tokenIDs), len(ranksOut))
	}
	for i := range tokenIDs {
		if err := w.Append(tokenIDs[i], ranksOut[i]); err != nil {
			return err
		}
	}
	return nil
}

// Close validates that the writer received the promised number of records.
func (w *Writer) Close() error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written != w.header.TotalTokens {
		return fmt.Errorf("incomplete record count: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	return nil
}

// Write writes a complete ranks file in one call.
func Write(w io.Writer, vocabSize uint32, tokenIDs, ranksOut []uint16) error {
	if len(tokenIDs) != len(ranksOut) {
		return fmt.Errorf("token/rank length mismatch: %d != %d", len(tokenIDs), len(ranksOut))
	}
	rw, err := NewWriter(w, vocabSize, uint32(len(tokenIDs)))
	if err != nil {
		return err
	}
	if err := rw.AppendBatch(tokenIDs, ranksOut); err != nil {
		return err
	}
	return rw.Close()
}

// Read parses a ranks file produced by this package.
func Read(r io.Reader) (Header, []Record, error) {
	var header Header
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return Header{}, nil, err
	}
	if header.Magic != Magic {
		return Header{}, nil, fmt.Errorf("invalid magic: got=%#x want=%#x", header.Magic, Magic)
	}
	if header.Version != Version {
		return Header{}, nil, fmt.Errorf("unsupported version: %d", header.Version)
	}
	records := make([]Record, int(header.TotalTokens))
	for i := range records {
		if err := binary.Read(r, binary.LittleEndian, &records[i].TokenID); err != nil {
			return Header{}, nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &records[i].Rank); err != nil {
			return Header{}, nil, err
		}
	}
	var extra [1]byte
	n, err := r.Read(extra[:])
	if err != nil && err != io.EOF {
		return Header{}, nil, err
	}
	if n != 0 {
		return Header{}, nil, fmt.Errorf("trailing data after %d records", len(records))
	}
	return header, records, nil
}

// HitAtK returns the fraction of records whose rank is strictly less than k
// (top-K accuracy). Returns 0 for empty input.
func HitAtK(records []Record, k int) float64 {
	if len(records) == 0 || k <= 0 {
		return 0
	}
	hits := 0
	for _, rec := range records {
		if int(rec.Rank) < k {
			hits++
		}
	}
	return float64(hits) / float64(len(records))
}

// MRR is the mean reciprocal rank: average of 1/(rank+1). Returns 0 for
// empty input.
func MRR(records []Record) float64 {
	if len(records) == 0 {
		return 0
	}
	sum := 0.0
	for _, rec := range records {
		sum += 1.0 / (float64(rec.Rank) + 1.0)
	}
	return sum / float64(len(records))
}
