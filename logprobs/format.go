package logprobs

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	Magic   uint32 = 0x4C504C42
	Version uint32 = 1
)

type Header struct {
	Magic       uint32
	Version     uint32
	VocabSize   uint32
	TotalTokens uint32
}

type Record struct {
	TokenID uint16
	NLL     float32
}

type Writer struct {
	w       io.Writer
	header  Header
	written uint32
}

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

func (w *Writer) Append(tokenID uint16, nll float32) error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written >= w.header.TotalTokens {
		return fmt.Errorf("too many records: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	if err := binary.Write(w.w, binary.LittleEndian, tokenID); err != nil {
		return err
	}
	if err := binary.Write(w.w, binary.LittleEndian, nll); err != nil {
		return err
	}
	w.written++
	return nil
}

func (w *Writer) AppendBatch(tokenIDs []uint16, nlls []float32) error {
	if len(tokenIDs) != len(nlls) {
		return fmt.Errorf("token/nll length mismatch: %d != %d", len(tokenIDs), len(nlls))
	}
	for i := range tokenIDs {
		if err := w.Append(tokenIDs[i], nlls[i]); err != nil {
			return err
		}
	}
	return nil
}

func (w *Writer) Close() error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written != w.header.TotalTokens {
		return fmt.Errorf("incomplete record count: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	return nil
}

func Write(w io.Writer, vocabSize uint32, tokenIDs []uint16, nlls []float32) error {
	if len(tokenIDs) != len(nlls) {
		return fmt.Errorf("token/nll length mismatch: %d != %d", len(tokenIDs), len(nlls))
	}
	lw, err := NewWriter(w, vocabSize, uint32(len(tokenIDs)))
	if err != nil {
		return err
	}
	if err := lw.AppendBatch(tokenIDs, nlls); err != nil {
		return err
	}
	return lw.Close()
}

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
		if err := binary.Read(r, binary.LittleEndian, &records[i].NLL); err != nil {
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

func MeanNLL(records []Record) float64 {
	if len(records) == 0 {
		return 0
	}
	sum := 0.0
	for _, rec := range records {
		sum += float64(rec.NLL)
	}
	return sum / float64(len(records))
}

func IsFinite(nll float32) bool {
	return !math.IsNaN(float64(nll)) && !math.IsInf(float64(nll), 0)
}
