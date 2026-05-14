package logits

import (
	"bytes"
	"encoding/binary"
	"math"
	"slices"
	"testing"
)

func TestWriteReadRoundTripFloat32Raw(t *testing.T) {
	const vocab = 5
	tokenIDs := []uint16{0, 3, 4}
	flat := []float32{
		1.0, -2.5, 0.0, 0.5, 3.25,
		-1.0, 2.0, 0.125, -0.5, 1.5,
		0.0, 0.0, 7.0, -7.0, 0.5,
	}

	var buf bytes.Buffer
	if err := Write(&buf, vocab, DTypeFloat32, FormRaw, tokenIDs, flat); err != nil {
		t.Fatalf("Write: %v", err)
	}

	header, records, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if header.Magic != Magic || header.Version != Version || header.VocabSize != vocab || header.TotalTokens != uint32(len(tokenIDs)) {
		t.Fatalf("unexpected header: %+v", header)
	}
	if header.DType != DTypeFloat32 || header.Form != FormRaw {
		t.Fatalf("unexpected dtype/form: dtype=%s form=%s", header.DType, header.Form)
	}
	if len(records) != len(tokenIDs) {
		t.Fatalf("len(records) = %d, want %d", len(records), len(tokenIDs))
	}
	for i, rec := range records {
		if rec.TokenID != tokenIDs[i] {
			t.Fatalf("record[%d] tokenID = %d, want %d", i, rec.TokenID, tokenIDs[i])
		}
		if len(rec.Values) != vocab {
			t.Fatalf("record[%d] len(values) = %d, want %d", i, len(rec.Values), vocab)
		}
		for j, v := range rec.Values {
			want := flat[i*vocab+j]
			if v != want {
				t.Fatalf("record[%d][%d] = %g, want %g", i, j, v, want)
			}
		}
	}
}

func TestWriteReadRoundTripFloat16Raw(t *testing.T) {
	const vocab = 4
	tokenIDs := []uint16{1, 7}
	flat := []float32{
		0.0, 1.0, -1.0, 65504.0, // 65504 is the max finite float16
		2.5, -3.5, 0.5, -0.5,
	}

	var buf bytes.Buffer
	if err := Write(&buf, vocab, DTypeFloat16, FormRaw, tokenIDs, flat); err != nil {
		t.Fatalf("Write: %v", err)
	}

	header, records, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if header.DType != DTypeFloat16 {
		t.Fatalf("dtype = %s, want float16", header.DType)
	}
	for i, rec := range records {
		if rec.TokenID != tokenIDs[i] {
			t.Fatalf("record[%d] tokenID = %d, want %d", i, rec.TokenID, tokenIDs[i])
		}
		for j, v := range rec.Values {
			want := flat[i*vocab+j]
			// All test values above are exactly representable in float16.
			if v != want {
				t.Fatalf("record[%d][%d] = %g, want %g", i, j, v, want)
			}
		}
	}
}

func TestWriteReadLogprobsForm(t *testing.T) {
	const vocab = 3
	tokenIDs := []uint16{0}
	// Use already log-softmaxed values: log(softmax([1,2,3])).
	row := []float32{1, 2, 3}
	maxV := row[2]
	sumExp := math.Exp(float64(row[0]-maxV)) + math.Exp(float64(row[1]-maxV)) + math.Exp(float64(row[2]-maxV))
	logNorm := float64(maxV) + math.Log(sumExp)
	lp := []float32{
		float32(float64(row[0]) - logNorm),
		float32(float64(row[1]) - logNorm),
		float32(float64(row[2]) - logNorm),
	}

	var buf bytes.Buffer
	if err := Write(&buf, vocab, DTypeFloat32, FormLogprobs, tokenIDs, lp); err != nil {
		t.Fatalf("Write: %v", err)
	}

	header, records, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if header.Form != FormLogprobs {
		t.Fatalf("form = %s, want logprobs", header.Form)
	}
	// Sanity: exp(logprobs) should sum to 1.
	probs := records[0].Values
	sum := 0.0
	for _, v := range probs {
		sum += math.Exp(float64(v))
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Fatalf("probabilities do not sum to 1: %g", sum)
	}
}

func TestReadInvalidMagic(t *testing.T) {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint32(0xDEADBEEF)); err != nil {
		t.Fatalf("write magic: %v", err)
	}
	// Pad to header size.
	for buf.Len() < HeaderSize {
		buf.WriteByte(0)
	}
	if _, _, err := Read(bytes.NewReader(buf.Bytes())); err == nil {
		t.Fatal("Read with bad magic succeeded, want error")
	}
}

func TestReadUnsupportedDType(t *testing.T) {
	// Write a header with dtype byte=99 by hand.
	var buf [HeaderSize]byte
	binary.LittleEndian.PutUint32(buf[0:4], Magic)
	binary.LittleEndian.PutUint32(buf[4:8], Version)
	binary.LittleEndian.PutUint32(buf[8:12], 4)
	binary.LittleEndian.PutUint32(buf[12:16], 0)
	buf[16] = 99
	if _, _, err := Read(bytes.NewReader(buf[:])); err == nil {
		t.Fatal("Read with bad dtype succeeded, want error")
	}
}

func TestWriteTooManyRecords(t *testing.T) {
	var buf bytes.Buffer
	w, err := NewWriter(&buf, 2, 1, DTypeFloat32, FormRaw)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Append(0, []float32{0, 0}); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := w.Append(0, []float32{0, 0}); err == nil {
		t.Fatal("Append #2 succeeded, want error")
	}
}

func TestCloseIncomplete(t *testing.T) {
	var buf bytes.Buffer
	w, err := NewWriter(&buf, 2, 3, DTypeFloat32, FormRaw)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Append(0, []float32{0, 0}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := w.Close(); err == nil {
		t.Fatal("Close succeeded with missing records, want error")
	}
}

func TestParseDType(t *testing.T) {
	cases := []struct {
		in   string
		want DType
	}{
		{"float16", DTypeFloat16},
		{"fp16", DTypeFloat16},
		{"f16", DTypeFloat16},
		{"half", DTypeFloat16},
		{"float32", DTypeFloat32},
		{"fp32", DTypeFloat32},
		{"f32", DTypeFloat32},
	}
	for _, tc := range cases {
		got, err := ParseDType(tc.in)
		if err != nil {
			t.Errorf("ParseDType(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ParseDType(%q) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if _, err := ParseDType("int8"); err == nil {
		t.Fatal("ParseDType(int8) succeeded, want error")
	}
}

func TestParseForm(t *testing.T) {
	cases := []struct {
		in   string
		want Form
	}{
		{"raw", FormRaw},
		{"logits", FormRaw},
		{"logprobs", FormLogprobs},
		{"log_softmax", FormLogprobs},
	}
	for _, tc := range cases {
		got, err := ParseForm(tc.in)
		if err != nil {
			t.Errorf("ParseForm(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ParseForm(%q) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if _, err := ParseForm("probs"); err == nil {
		t.Fatal("ParseForm(probs) succeeded, want error")
	}
}

func TestFloat16RoundTripValues(t *testing.T) {
	cases := []float32{
		0, -0, 1, -1, 0.5, -0.5, 0.25, 2.0, 16.0, -16.0,
		1.0 / 3.0, math.Pi, -math.Pi, 65504, -65504,
	}
	for _, v := range cases {
		bits := float32ToFloat16Bits(v)
		got := float16BitsToFloat32(bits)
		// Half-precision tolerance: ~10-bit mantissa gives ~1e-3 relative.
		if math.Abs(float64(got-v))/(math.Abs(float64(v))+1e-9) > 1e-3 {
			t.Errorf("float16 round-trip(%g) = %g (relative err too large)", v, got)
		}
	}
}

func TestFloat16OverflowSaturates(t *testing.T) {
	bits := float32ToFloat16Bits(1e6)
	got := float16BitsToFloat32(bits)
	if !math.IsInf(float64(got), 1) {
		t.Fatalf("float32->float16(1e6) = %g, want +Inf", got)
	}
	bits = float32ToFloat16Bits(-1e6)
	got = float16BitsToFloat32(bits)
	if !math.IsInf(float64(got), -1) {
		t.Fatalf("float32->float16(-1e6) = %g, want -Inf", got)
	}
}

func TestFloat16NaN(t *testing.T) {
	bits := float32ToFloat16Bits(float32(math.NaN()))
	got := float16BitsToFloat32(bits)
	if !math.IsNaN(float64(got)) {
		t.Fatalf("float32->float16(NaN) round-trip = %g, want NaN", got)
	}
	// NaN must be encoded as quiet (non-zero mantissa under exp=0x1f).
	if (bits & 0x7c00) != 0x7c00 {
		t.Fatalf("float16 NaN exp bits = %#x, want 0x7c00", bits&0x7c00)
	}
	if (bits & 0x03ff) == 0 {
		t.Fatalf("float16 NaN mantissa is zero (would alias Inf): bits=%#x", bits)
	}
}

// TestFloat16NegativeZeroSignPreserved exercises true IEEE 754 -0 (the literal
// -0 is folded to +0 by the Go compiler, so this test uses math.Copysign).
func TestFloat16NegativeZeroSignPreserved(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	if !math.Signbit(float64(negZero)) {
		t.Fatal("math.Copysign(0,-1) didn't produce a negative zero — test setup is wrong")
	}
	bits := float32ToFloat16Bits(negZero)
	if bits != 0x8000 {
		t.Fatalf("float16(-0) bits = %#x, want 0x8000", bits)
	}
	got := float16BitsToFloat32(bits)
	if got != 0 {
		t.Fatalf("float16(-0) round-trip = %g, want 0", got)
	}
	if !math.Signbit(float64(got)) {
		t.Fatal("float16(-0) round-trip lost the sign bit (became +0)")
	}
	// And +0 must round-trip to 0x0000 (i.e. signs are not silently flipped).
	if bits := float32ToFloat16Bits(0); bits != 0x0000 {
		t.Fatalf("float16(+0) bits = %#x, want 0x0000", bits)
	}
}

// TestFloat16SubnormalBoundaries checks the smallest representable subnormal,
// the boundary between subnormal and normal, and round-to-zero / tie-to-even
// behavior just below the smallest subnormal.
func TestFloat16SubnormalBoundaries(t *testing.T) {
	cases := []struct {
		name string
		in   float32
		want uint16
	}{
		// 2^-24 = smallest positive half subnormal; bits 0x0001.
		{name: "min_subnormal", in: float32(math.Ldexp(1, -24)), want: 0x0001},
		// 2^-14 = smallest positive half normal; bits 0x0400.
		{name: "min_normal", in: float32(math.Ldexp(1, -14)), want: 0x0400},
		// 2^-25 = exactly halfway between 0 and 2^-24. Round-to-even -> 0
		// (kept LSB is 0, even).
		{name: "subnormal_tie_to_zero", in: float32(math.Ldexp(1, -25)), want: 0x0000},
		// 3 * 2^-25 = halfway between 2^-24 (bits 0x0001) and 2*2^-24 (bits
		// 0x0002). Truncation gives halfMant=1 (odd), so tie rounds UP to
		// 0x0002.
		{name: "subnormal_tie_to_even_up", in: float32(3 * math.Ldexp(1, -25)), want: 0x0002},
		// Way below smallest subnormal -> +0.
		{name: "underflow_to_zero", in: float32(math.Ldexp(1, -30)), want: 0x0000},
		// Largest finite half: (2 - 2^-10) * 2^15 = 65504; bits 0x7bff.
		{name: "max_finite", in: 65504, want: 0x7bff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := float32ToFloat16Bits(tc.in)
			if got != tc.want {
				t.Fatalf("float16(%g) bits = %#x, want %#x", tc.in, got, tc.want)
			}
		})
	}
}

// TestFloat16NormalMantissaTieToEven constructs float32 values whose lower 13
// mantissa bits equal exactly 0x1000 — the tie point for round-to-nearest in
// the normal range. The half-precision mantissa LSB (parity) decides whether
// the tie rounds up or down.
func TestFloat16NormalMantissaTieToEven(t *testing.T) {
	// In the normal range with float32 biased exponent 127 (i.e. values in
	// [1.0, 2.0)), float32 mantissa contributes (mant/2^23) to the value.
	// Half mantissa = mant >> 13; tie is when rem == 0x1000.

	// halfMant after truncation = 0 (even) -> tie rounds DOWN to halfMant 0.
	// Float32 value = 1 + 0x1000/2^23.
	fEvenTie := math.Float32frombits(0x3f800000 | 0x1000)
	gotEven := float32ToFloat16Bits(fEvenTie)
	wantEven := uint16(0x3c00) // half 1.0 = sign=0, exp=15, mant=0
	if gotEven != wantEven {
		t.Fatalf("tie-to-even (halfMant=0): float16(%g) = %#x, want %#x", fEvenTie, gotEven, wantEven)
	}

	// halfMant after truncation = 1 (odd) -> tie rounds UP to halfMant 2.
	// Float32 value = 1 + 0x3000/2^23.
	fOddTie := math.Float32frombits(0x3f800000 | 0x3000)
	gotOdd := float32ToFloat16Bits(fOddTie)
	// half 1 + 2*2^-10: exp=15, mant=2 -> 0x3c00 | 0x002 = 0x3c02
	wantOdd := uint16(0x3c02)
	if gotOdd != wantOdd {
		t.Fatalf("tie-to-even (halfMant=1): float16(%g) = %#x, want %#x", fOddTie, gotOdd, wantOdd)
	}

	// Non-tie above the midpoint (rem > 0x1000): always rounds up regardless
	// of parity.
	fAbove := math.Float32frombits(0x3f800000 | 0x1001)
	gotAbove := float32ToFloat16Bits(fAbove)
	if gotAbove != 0x3c01 {
		t.Fatalf("round-up (rem>0x1000): float16(%g) = %#x, want 0x3c01", fAbove, gotAbove)
	}
	// Non-tie below the midpoint (rem < 0x1000): always rounds down.
	fBelow := math.Float32frombits(0x3f800000 | 0x0fff)
	gotBelow := float32ToFloat16Bits(fBelow)
	if gotBelow != 0x3c00 {
		t.Fatalf("round-down (rem<0x1000): float16(%g) = %#x, want 0x3c00", fBelow, gotBelow)
	}
}

// TestFloat16OverflowAtRoundingBoundary verifies that values past the largest
// finite half saturate to ±Inf rather than wrapping or producing a NaN.
func TestFloat16OverflowAtRoundingBoundary(t *testing.T) {
	// 65520 = 65504 + 16; the float16 "next" step above 65504 would be Inf,
	// and 65520 is the round-to-nearest boundary that promotes to Inf.
	if bits := float32ToFloat16Bits(65520); bits != 0x7c00 {
		t.Fatalf("float16(65520) bits = %#x, want 0x7c00 (+Inf)", bits)
	}
	if bits := float32ToFloat16Bits(-65520); bits != 0xfc00 {
		t.Fatalf("float16(-65520) bits = %#x, want 0xfc00 (-Inf)", bits)
	}
}

func TestFloat16Inf(t *testing.T) {
	bits := float32ToFloat16Bits(float32(math.Inf(1)))
	got := float16BitsToFloat32(bits)
	if !math.IsInf(float64(got), 1) {
		t.Fatalf("float32->float16(+Inf) round-trip = %g, want +Inf", got)
	}
	bits = float32ToFloat16Bits(float32(math.Inf(-1)))
	got = float16BitsToFloat32(bits)
	if !math.IsInf(float64(got), -1) {
		t.Fatalf("float32->float16(-Inf) round-trip = %g, want -Inf", got)
	}
}

// TestNumericalConsistencyWithLogprobsRecovers verifies that the acceptance
// criterion logsumexp(logits[i]) - logits[i, target] recovers the NLL for the
// same row, both for raw-form and logprobs-form storage.
func TestNumericalConsistencyWithLogprobsRecovers(t *testing.T) {
	row := []float32{1.5, -2.0, 3.0, 0.25, -0.5}
	const tgt = uint16(2)
	vocab := uint32(len(row))

	for _, dtype := range []DType{DTypeFloat32, DTypeFloat16} {
		for _, form := range []Form{FormRaw, FormLogprobs} {
			data := slices.Clone(row)
			if form == FormLogprobs {
				maxV := data[0]
				for _, v := range data[1:] {
					if v > maxV {
						maxV = v
					}
				}
				sumExp := 0.0
				for _, v := range data {
					sumExp += math.Exp(float64(v - maxV))
				}
				logNorm := float64(maxV) + math.Log(sumExp)
				for i := range data {
					data[i] = float32(float64(data[i]) - logNorm)
				}
			}

			var buf bytes.Buffer
			if err := Write(&buf, vocab, dtype, form, []uint16{tgt}, data); err != nil {
				t.Fatalf("Write(%s, %s): %v", dtype, form, err)
			}
			_, records, err := Read(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatalf("Read(%s, %s): %v", dtype, form, err)
			}
			values := records[0].Values
			maxV := values[0]
			for _, v := range values[1:] {
				if v > maxV {
					maxV = v
				}
			}
			sumExp := 0.0
			for _, v := range values {
				sumExp += math.Exp(float64(v - maxV))
			}
			logNorm := float64(maxV) + math.Log(sumExp)
			recoveredNLL := logNorm - float64(values[tgt])

			// Ground truth NLL from the raw row.
			rawMax := row[0]
			for _, v := range row[1:] {
				if v > rawMax {
					rawMax = v
				}
			}
			rawSum := 0.0
			for _, v := range row {
				rawSum += math.Exp(float64(v - rawMax))
			}
			rawLogNorm := float64(rawMax) + math.Log(rawSum)
			wantNLL := rawLogNorm - float64(row[tgt])

			tol := 1e-5
			if dtype == DTypeFloat16 {
				tol = 5e-3
			}
			if math.Abs(recoveredNLL-wantNLL) > tol {
				t.Fatalf("recovered NLL(%s,%s) = %g, want %g (tol=%g)", dtype, form, recoveredNLL, wantNLL, tol)
			}
		}
	}
}
