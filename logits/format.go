// Package logits defines the binary file format for per-token full-distribution
// model output emitted by mixlab eval mode (-logits-out).
//
// Each record stores the target token ID at an evaluation position together
// with the full vocab-sized vector for that position. The vector is either the
// raw logits ("raw" form) or the log-softmax of those logits ("logprobs"
// form), stored as either IEEE 754 half-precision floats (DTypeFloat16) or
// single-precision floats (DTypeFloat32). Records align position-by-position
// with logprobs.Record, ranks.Record, and uncertainty.Record when produced in
// the same eval pass.
package logits

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	// Magic is the 4-byte file identifier (ASCII "LOGT").
	Magic uint32 = 0x4C4F4754
	// Version is the on-disk format version.
	Version uint32 = 1
)

// DType selects the on-disk numeric encoding of each logits row.
type DType uint8

const (
	DTypeFloat16 DType = 0
	DTypeFloat32 DType = 1
)

// String returns a human-readable name for the dtype.
func (d DType) String() string {
	switch d {
	case DTypeFloat16:
		return "float16"
	case DTypeFloat32:
		return "float32"
	default:
		return fmt.Sprintf("dtype(%d)", uint8(d))
	}
}

// ParseDType parses a dtype name accepted by the CLI (-logits-dtype).
func ParseDType(s string) (DType, error) {
	switch s {
	case "float16", "fp16", "f16", "half":
		return DTypeFloat16, nil
	case "float32", "fp32", "f32", "single":
		return DTypeFloat32, nil
	default:
		return 0, fmt.Errorf("unsupported logits dtype %q (want \"float16\" or \"float32\")", s)
	}
}

// bytesPerElement returns the on-disk size in bytes of one logit value at this
// dtype.
func (d DType) bytesPerElement() int {
	switch d {
	case DTypeFloat16:
		return 2
	case DTypeFloat32:
		return 4
	default:
		return 0
	}
}

// Form selects how each logits row is encoded relative to the model output.
type Form uint8

const (
	// FormRaw stores the raw model logits as produced by EvalLogits.
	FormRaw Form = 0
	// FormLogprobs stores log_softmax(logits) — i.e. raw logits with logsumexp
	// subtracted per position. Useful when downstream consumers only need
	// log-probabilities and want to skip the renormalisation step.
	FormLogprobs Form = 1
)

// String returns a human-readable name for the form.
func (f Form) String() string {
	switch f {
	case FormRaw:
		return "raw"
	case FormLogprobs:
		return "logprobs"
	default:
		return fmt.Sprintf("form(%d)", uint8(f))
	}
}

// ParseForm parses a form name accepted by the CLI (-logits-form).
func ParseForm(s string) (Form, error) {
	switch s {
	case "raw", "logits":
		return FormRaw, nil
	case "logprobs", "log_softmax", "logsoftmax":
		return FormLogprobs, nil
	default:
		return 0, fmt.Errorf("unsupported logits form %q (want \"raw\" or \"logprobs\")", s)
	}
}

// Header is the fixed 20-byte file header. The trailing reserved bytes are
// zero in version 1 files and must be ignored by readers.
type Header struct {
	Magic       uint32
	Version     uint32
	VocabSize   uint32
	TotalTokens uint32
	DType       DType
	Form        Form
	Reserved    [2]byte
}

// HeaderSize is the on-disk size of Header.
const HeaderSize = 20

// Record is a single eval position decoded back from disk: the target token ID
// followed by a vocab-sized vector of float32 values. Float16 files are
// promoted to float32 on read so callers can compute on them directly.
type Record struct {
	TokenID uint16
	Values  []float32
}

// Writer is a streaming writer that emits a header followed by records and
// validates that the record count matches the header on Close.
type Writer struct {
	w          io.Writer
	header     Header
	written    uint32
	rowScratch []byte
}

// NewWriter writes the header and returns a Writer ready to accept exactly
// totalTokens records.
func NewWriter(w io.Writer, vocabSize, totalTokens uint32, dtype DType, form Form) (*Writer, error) {
	if w == nil {
		return nil, fmt.Errorf("nil writer")
	}
	if dtype.bytesPerElement() == 0 {
		return nil, fmt.Errorf("unsupported logits dtype: %s", dtype)
	}
	if form != FormRaw && form != FormLogprobs {
		return nil, fmt.Errorf("unsupported logits form: %s", form)
	}
	header := Header{
		Magic:       Magic,
		Version:     Version,
		VocabSize:   vocabSize,
		TotalTokens: totalTokens,
		DType:       dtype,
		Form:        form,
	}
	if err := writeHeader(w, header); err != nil {
		return nil, err
	}
	rowBytes := 2 + int(vocabSize)*dtype.bytesPerElement()
	return &Writer{w: w, header: header, rowScratch: make([]byte, rowBytes)}, nil
}

// Append writes one record. row must have at least vocabSize entries.
func (w *Writer) Append(tokenID uint16, row []float32) error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	if w.written >= w.header.TotalTokens {
		return fmt.Errorf("too many records: wrote=%d total=%d", w.written, w.header.TotalTokens)
	}
	vocab := int(w.header.VocabSize)
	if len(row) < vocab {
		return fmt.Errorf("logits row too short: got=%d want>=%d", len(row), vocab)
	}
	buf := w.rowScratch
	binary.LittleEndian.PutUint16(buf[0:2], tokenID)
	switch w.header.DType {
	case DTypeFloat16:
		for j := range vocab {
			bits := float32ToFloat16Bits(row[j])
			binary.LittleEndian.PutUint16(buf[2+j*2:4+j*2], bits)
		}
	case DTypeFloat32:
		for j := range vocab {
			binary.LittleEndian.PutUint32(buf[2+j*4:6+j*4], math.Float32bits(row[j]))
		}
	default:
		return fmt.Errorf("unsupported logits dtype: %s", w.header.DType)
	}
	n, err := w.w.Write(buf)
	if err != nil {
		return err
	}
	if n != len(buf) {
		return io.ErrShortWrite
	}
	w.written++
	return nil
}

// AppendBatch writes one record per position. logitsFlat must hold
// len(tokenIDs)*vocab entries in row-major order.
func (w *Writer) AppendBatch(tokenIDs []uint16, logitsFlat []float32) error {
	if w == nil {
		return fmt.Errorf("nil writer")
	}
	vocab := int(w.header.VocabSize)
	if len(logitsFlat) != len(tokenIDs)*vocab {
		return fmt.Errorf("logits length mismatch: got=%d want=%d", len(logitsFlat), len(tokenIDs)*vocab)
	}
	for i, tok := range tokenIDs {
		row := logitsFlat[i*vocab : (i+1)*vocab]
		if err := w.Append(tok, row); err != nil {
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

// Write writes a complete logits file in one call.
func Write(w io.Writer, vocabSize uint32, dtype DType, form Form, tokenIDs []uint16, logitsFlat []float32) error {
	vocab := int(vocabSize)
	if len(logitsFlat) != len(tokenIDs)*vocab {
		return fmt.Errorf("logits length mismatch: got=%d want=%d", len(logitsFlat), len(tokenIDs)*vocab)
	}
	lw, err := NewWriter(w, vocabSize, uint32(len(tokenIDs)), dtype, form)
	if err != nil {
		return err
	}
	if err := lw.AppendBatch(tokenIDs, logitsFlat); err != nil {
		return err
	}
	return lw.Close()
}

func writeHeader(w io.Writer, h Header) error {
	var buf [HeaderSize]byte
	binary.LittleEndian.PutUint32(buf[0:4], h.Magic)
	binary.LittleEndian.PutUint32(buf[4:8], h.Version)
	binary.LittleEndian.PutUint32(buf[8:12], h.VocabSize)
	binary.LittleEndian.PutUint32(buf[12:16], h.TotalTokens)
	buf[16] = uint8(h.DType)
	buf[17] = uint8(h.Form)
	buf[18] = h.Reserved[0]
	buf[19] = h.Reserved[1]
	n, err := w.Write(buf[:])
	if err != nil {
		return err
	}
	if n != len(buf) {
		return io.ErrShortWrite
	}
	return nil
}

func readHeader(r io.Reader) (Header, error) {
	var buf [HeaderSize]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return Header{}, err
	}
	h := Header{
		Magic:       binary.LittleEndian.Uint32(buf[0:4]),
		Version:     binary.LittleEndian.Uint32(buf[4:8]),
		VocabSize:   binary.LittleEndian.Uint32(buf[8:12]),
		TotalTokens: binary.LittleEndian.Uint32(buf[12:16]),
		DType:       DType(buf[16]),
		Form:        Form(buf[17]),
	}
	h.Reserved[0] = buf[18]
	h.Reserved[1] = buf[19]
	return h, nil
}

// Read parses a logits file produced by this package. Float16 records are
// promoted to float32 in the returned records so callers can operate on them
// uniformly.
func Read(r io.Reader) (Header, []Record, error) {
	header, err := readHeader(r)
	if err != nil {
		return Header{}, nil, err
	}
	if header.Magic != Magic {
		return Header{}, nil, fmt.Errorf("invalid magic: got=%#x want=%#x", header.Magic, Magic)
	}
	if header.Version != Version {
		return Header{}, nil, fmt.Errorf("unsupported version: %d", header.Version)
	}
	bpe := header.DType.bytesPerElement()
	if bpe == 0 {
		return Header{}, nil, fmt.Errorf("unsupported dtype on disk: %d", header.DType)
	}
	if header.Form != FormRaw && header.Form != FormLogprobs {
		return Header{}, nil, fmt.Errorf("unsupported form on disk: %d", header.Form)
	}
	vocab := int(header.VocabSize)
	rowBytes := 2 + vocab*bpe
	rowBuf := make([]byte, rowBytes)
	records := make([]Record, int(header.TotalTokens))
	for i := range records {
		if _, err := io.ReadFull(r, rowBuf); err != nil {
			return Header{}, nil, err
		}
		records[i].TokenID = binary.LittleEndian.Uint16(rowBuf[0:2])
		records[i].Values = make([]float32, vocab)
		switch header.DType {
		case DTypeFloat16:
			for j := range vocab {
				bits := binary.LittleEndian.Uint16(rowBuf[2+j*2 : 4+j*2])
				records[i].Values[j] = float16BitsToFloat32(bits)
			}
		case DTypeFloat32:
			for j := range vocab {
				bits := binary.LittleEndian.Uint32(rowBuf[2+j*4 : 6+j*4])
				records[i].Values[j] = math.Float32frombits(bits)
			}
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

// float32ToFloat16Bits converts a float32 to its IEEE 754 half-precision
// (binary16) bit pattern using round-to-nearest-even. NaN, infinity, and
// subnormal half values are handled. Values outside the half representable
// range saturate to ±Inf.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits >> 23) & 0xff)
	mant := bits & 0x7fffff

	// NaN / Inf
	if exp == 0xff {
		if mant != 0 {
			// Quiet NaN: keep top mantissa bit set; preserve some payload.
			payload := uint16(mant >> 13)
			if payload == 0 {
				payload = 1
			}
			return sign | 0x7c00 | payload | 0x200
		}
		return sign | 0x7c00
	}

	// Normalise exponent into half bias (15).
	e := exp - 127 + 15
	if e >= 0x1f {
		// Overflow -> infinity
		return sign | 0x7c00
	}
	if e <= 0 {
		// Subnormal half (or zero)
		if e < -10 {
			return sign
		}
		// Restore implicit leading bit and shift into half mantissa range.
		m := mant | 0x800000
		shift := uint(14 - e) // shift right by (1-e) + 13 = 14-e
		halfMant := m >> shift
		// Rounding: examine the bit just below the kept ones (round-to-nearest-even).
		if shift > 0 {
			rb := (m >> (shift - 1)) & 1
			rest := m & ((1 << (shift - 1)) - 1)
			if rb == 1 && (rest != 0 || (halfMant&1) == 1) {
				halfMant++
			}
		}
		return sign | uint16(halfMant)
	}
	// Normal half: round mantissa
	halfMant := mant >> 13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (halfMant&1) == 1) {
		halfMant++
		if halfMant == 0x400 {
			halfMant = 0
			e++
			if e >= 0x1f {
				return sign | 0x7c00
			}
		}
	}
	return sign | (uint16(e) << 10) | uint16(halfMant&0x3ff)
}

// float16BitsToFloat32 converts an IEEE 754 half-precision bit pattern to
// float32. Used by Read so callers can operate on float32 uniformly.
func float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	var bits uint32
	switch exp {
	case 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			// Subnormal half -> normalised float32
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x3ff
			bits = (sign << 31) | (e << 23) | (mant << 13)
		}
	case 0x1f:
		if mant == 0 {
			bits = (sign << 31) | 0x7f800000
		} else {
			bits = (sign << 31) | 0x7f800000 | (mant << 13)
		}
	default:
		bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}
