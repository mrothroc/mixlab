package logprobs

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	tokenIDs := []uint16{1, 2, 513}
	nlls := []float32{0.25, 1.5, 3.75}
	if err := Write(&buf, 1024, tokenIDs, nlls); err != nil {
		t.Fatalf("Write: %v", err)
	}

	header, records, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if header.Magic != Magic || header.Version != Version || header.VocabSize != 1024 || header.TotalTokens != 3 {
		t.Fatalf("unexpected header: %+v", header)
	}
	if len(records) != len(tokenIDs) {
		t.Fatalf("record count = %d, want %d", len(records), len(tokenIDs))
	}
	for i := range records {
		if records[i].TokenID != tokenIDs[i] {
			t.Fatalf("record[%d].TokenID = %d, want %d", i, records[i].TokenID, tokenIDs[i])
		}
		if math.Abs(float64(records[i].NLL-nlls[i])) > 1e-6 {
			t.Fatalf("record[%d].NLL = %g, want %g", i, records[i].NLL, nlls[i])
		}
	}
}

func TestWriteReadEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, 1024, nil, nil); err != nil {
		t.Fatalf("Write: %v", err)
	}

	header, records, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if header.TotalTokens != 0 {
		t.Fatalf("header.TotalTokens = %d, want 0", header.TotalTokens)
	}
	if len(records) != 0 {
		t.Fatalf("len(records) = %d, want 0", len(records))
	}
}

func TestReadInvalidMagic(t *testing.T) {
	var buf bytes.Buffer
	header := Header{
		Magic:       0xDEADBEEF,
		Version:     Version,
		VocabSize:   16,
		TotalTokens: 0,
	}
	if err := binary.Write(&buf, binary.LittleEndian, header); err != nil {
		t.Fatalf("binary.Write(header): %v", err)
	}

	_, _, err := Read(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("Read succeeded, want error")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("magic")) {
		t.Fatalf("Read error = %q, want mention of magic", err)
	}
}

func TestReadTruncated(t *testing.T) {
	var buf bytes.Buffer
	header := Header{
		Magic:       Magic,
		Version:     Version,
		VocabSize:   32,
		TotalTokens: 2,
	}
	if err := binary.Write(&buf, binary.LittleEndian, header); err != nil {
		t.Fatalf("binary.Write(header): %v", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint16(7)); err != nil {
		t.Fatalf("binary.Write(tokenID): %v", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, float32(1.25)); err != nil {
		t.Fatalf("binary.Write(nll): %v", err)
	}

	_, _, err := Read(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("Read succeeded, want error")
	}
}

func TestWriteTooManyRecords(t *testing.T) {
	var buf bytes.Buffer
	writer, err := NewWriter(&buf, 32, 2)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := writer.Append(1, 0.5); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := writer.Append(2, 1.5); err != nil {
		t.Fatalf("Append #2: %v", err)
	}
	err = writer.Append(3, 2.5)
	if err == nil {
		t.Fatal("Append #3 succeeded, want error")
	}
}

func TestWriterCloseIncomplete(t *testing.T) {
	var buf bytes.Buffer
	writer, err := NewWriter(&buf, 32, 3)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := writer.Append(1, 0.5); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := writer.Append(2, 1.5); err != nil {
		t.Fatalf("Append #2: %v", err)
	}
	err = writer.Close()
	if err == nil {
		t.Fatal("Close succeeded, want error")
	}
}

func TestAppendBatchLengthMismatch(t *testing.T) {
	var buf bytes.Buffer
	writer, err := NewWriter(&buf, 32, 2)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	err = writer.AppendBatch([]uint16{1, 2}, []float32{0.5})
	if err == nil {
		t.Fatal("AppendBatch succeeded, want error")
	}
}

func TestMeanNLL(t *testing.T) {
	records := []Record{
		{TokenID: 1, NLL: 1.0},
		{TokenID: 2, NLL: 2.0},
		{TokenID: 3, NLL: 3.0},
	}
	if got := MeanNLL(records); got != 2.0 {
		t.Fatalf("MeanNLL = %g, want 2.0", got)
	}
}

func TestMeanNLLEmpty(t *testing.T) {
	if got := MeanNLL(nil); got != 0 {
		t.Fatalf("MeanNLL(nil) = %g, want 0", got)
	}
}

func TestIsFinite(t *testing.T) {
	cases := []struct {
		name string
		nll  float32
		want bool
	}{
		{name: "zero", nll: 0, want: true},
		{name: "normal", nll: 1.25, want: true},
		{name: "negative", nll: -3.5, want: true},
		{name: "nan", nll: float32(math.NaN()), want: false},
		{name: "pos_inf", nll: float32(math.Inf(1)), want: false},
		{name: "neg_inf", nll: float32(math.Inf(-1)), want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsFinite(tc.nll); got != tc.want {
				t.Fatalf("IsFinite(%v) = %v, want %v", tc.nll, got, tc.want)
			}
		})
	}
}
