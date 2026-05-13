package uncertainty

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	tokenIDs := []uint16{1, 2, 513}
	top1 := []float32{0.5, 0.75, 0.25}
	entropies := []float32{0.7, 0.3, 1.25}
	margins := []float32{0.2, 0.5, 0.0}
	if err := Write(&buf, 1024, tokenIDs, top1, entropies, margins); err != nil {
		t.Fatalf("Write: %v", err)
	}

	const wantBytes = 16 + 3*14
	if buf.Len() != wantBytes {
		t.Fatalf("encoded size = %d, want %d", buf.Len(), wantBytes)
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
		if math.Abs(float64(records[i].Top1Prob-top1[i])) > 1e-6 {
			t.Fatalf("record[%d].Top1Prob = %g, want %g", i, records[i].Top1Prob, top1[i])
		}
		if math.Abs(float64(records[i].Entropy-entropies[i])) > 1e-6 {
			t.Fatalf("record[%d].Entropy = %g, want %g", i, records[i].Entropy, entropies[i])
		}
		if math.Abs(float64(records[i].Margin-margins[i])) > 1e-6 {
			t.Fatalf("record[%d].Margin = %g, want %g", i, records[i].Margin, margins[i])
		}
	}
}

func TestWriteReadEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, 1024, nil, nil, nil, nil); err != nil {
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

func TestReadUnsupportedVersion(t *testing.T) {
	var buf bytes.Buffer
	header := Header{
		Magic:       Magic,
		Version:     Version + 1,
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
	if err := binary.Write(&buf, binary.LittleEndian, float32(0.5)); err != nil {
		t.Fatalf("binary.Write(top1): %v", err)
	}

	_, _, err := Read(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("Read succeeded, want error")
	}
}

func TestReadTrailingData(t *testing.T) {
	var buf bytes.Buffer
	header := Header{
		Magic:       Magic,
		Version:     Version,
		VocabSize:   32,
		TotalTokens: 1,
	}
	if err := binary.Write(&buf, binary.LittleEndian, header); err != nil {
		t.Fatalf("binary.Write(header): %v", err)
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint16(7)); err != nil {
		t.Fatalf("binary.Write(tokenID): %v", err)
	}
	for _, v := range []float32{0.5, 0.7, 0.2} {
		if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
			t.Fatalf("binary.Write(record float): %v", err)
		}
	}
	buf.WriteByte(0xff)

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
	if err := writer.Append(1, 0.5, 0.7, 0.2); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := writer.Append(2, 0.75, 0.4, 0.5); err != nil {
		t.Fatalf("Append #2: %v", err)
	}
	err = writer.Append(3, 0.25, 1.0, 0.0)
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
	if err := writer.Append(1, 0.5, 0.7, 0.2); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := writer.Append(2, 0.75, 0.4, 0.5); err != nil {
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
	err = writer.AppendBatch([]uint16{1, 2}, []float32{0.5}, []float32{0.7, 0.4}, []float32{0.2, 0.5})
	if err == nil {
		t.Fatal("AppendBatch succeeded, want error")
	}
}
