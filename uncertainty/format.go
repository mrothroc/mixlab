// Package uncertainty defines the binary file format for per-token candidate
// uncertainty metrics emitted by mixlab eval mode (-uncertainty-out).
//
// Each record stores the target token ID at an evaluation position together
// with quantities derived from the model's own next-token softmax
// distribution: top-1 probability, entropy, and top-1/top-2 probability
// margin. Records align position-by-position with logprobs.Record and
// ranks.Record when produced in the same eval pass.
package uncertainty

import (
	"encoding/binary"
	"fmt"
	"io"
)

const (
	// Magic is the 4-byte file identifier: ASCII "UNCT".
	Magic uint32 = 0x554E4354
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

// Record is a single eval position: the target token ID plus candidate-side
// uncertainty metrics from the model's softmax distribution.
type Record struct {
	TokenID  uint16
	Top1Prob float32
	Entropy  float32
	Margin   float32
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
func (w *Writer) Append(tokenID uint16, top1Prob, entropy, margin float32) error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written >= w.header.TotalTokens {
		return fmt.Errorf("too many records: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	if err := binary.Write(w.w, binary.LittleEndian, tokenID); err != nil {
		return err
	}
	if err := binary.Write(w.w, binary.LittleEndian, top1Prob); err != nil {
		return err
	}
	if err := binary.Write(w.w, binary.LittleEndian, entropy); err != nil {
		return err
	}
	if err := binary.Write(w.w, binary.LittleEndian, margin); err != nil {
		return err
	}
	w.written++
	return nil
}

// AppendBatch writes one record per position.
func (w *Writer) AppendBatch(tokenIDs []uint16, top1Probs, entropies, margins []float32) error {
	if len(tokenIDs) != len(top1Probs) {
		return fmt.Errorf("token/top1 length mismatch: %d != %d", len(tokenIDs), len(top1Probs))
	}
	if len(tokenIDs) != len(entropies) {
		return fmt.Errorf("token/entropy length mismatch: %d != %d", len(tokenIDs), len(entropies))
	}
	if len(tokenIDs) != len(margins) {
		return fmt.Errorf("token/margin length mismatch: %d != %d", len(tokenIDs), len(margins))
	}
	for i := range tokenIDs {
		if err := w.Append(tokenIDs[i], top1Probs[i], entropies[i], margins[i]); err != nil {
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

// Write writes a complete uncertainty file in one call.
func Write(w io.Writer, vocabSize uint32, tokenIDs []uint16, top1Probs, entropies, margins []float32) error {
	if len(tokenIDs) != len(top1Probs) {
		return fmt.Errorf("token/top1 length mismatch: %d != %d", len(tokenIDs), len(top1Probs))
	}
	if len(tokenIDs) != len(entropies) {
		return fmt.Errorf("token/entropy length mismatch: %d != %d", len(tokenIDs), len(entropies))
	}
	if len(tokenIDs) != len(margins) {
		return fmt.Errorf("token/margin length mismatch: %d != %d", len(tokenIDs), len(margins))
	}
	uw, err := NewWriter(w, vocabSize, uint32(len(tokenIDs)))
	if err != nil {
		return err
	}
	if err := uw.AppendBatch(tokenIDs, top1Probs, entropies, margins); err != nil {
		return err
	}
	return uw.Close()
}

// Read parses an uncertainty file produced by this package.
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
		if err := binary.Read(r, binary.LittleEndian, &records[i].Top1Prob); err != nil {
			return Header{}, nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &records[i].Entropy); err != nil {
			return Header{}, nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &records[i].Margin); err != nil {
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
