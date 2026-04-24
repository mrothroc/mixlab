package logprobs

import (
	"bytes"
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
