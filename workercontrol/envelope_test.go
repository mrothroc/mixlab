package workercontrol

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func testEnvelope() Envelope {
	return Envelope{Version: Version, JobID: "job-1", AttemptID: "attempt-1", Sequence: 1,
		CorrelationID: "correlation-1", Kind: KindAssignment, PayloadKind: "owner.assignment",
		PayloadVersion: 7, Payload: json.RawMessage(`{"owner_field":[1,true,"value"]}`)}
}

func rawFrame(body string) []byte {
	frame := make([]byte, 4+len(body))
	binary.BigEndian.PutUint32(frame, uint32(len(body)))
	copy(frame[4:], body)
	return frame
}

func TestFrameRoundTripAndAdjacentFrames(t *testing.T) {
	var b bytes.Buffer
	e := testEnvelope()
	for range 2 {
		if err := WriteFrame(&b, e, MaxFrameBytes); err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		got, err := ReadFrame(&b, MaxFrameBytes)
		if err != nil {
			t.Fatal(err)
		}
		if got.Version != e.Version || got.JobID != e.JobID || got.AttemptID != e.AttemptID ||
			got.Sequence != e.Sequence || got.CorrelationID != e.CorrelationID || got.Kind != e.Kind ||
			got.PayloadKind != e.PayloadKind || got.PayloadVersion != e.PayloadVersion || !bytes.Equal(got.Payload, e.Payload) {
			t.Fatalf("round trip mismatch: %+v", got)
		}
	}
	if _, err := ReadFrame(&b, MaxFrameBytes); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
}

func TestClosedKindsAndOpaquePayload(t *testing.T) {
	for _, kind := range []Kind{KindAssignment, KindReadiness, KindHeartbeat, KindProgress, KindCancellation,
		KindRoundRequest, KindRoundResponse, KindArtifactRead, KindArtifactWrite, KindUpdateSubmission,
		KindUpdateAcknowledgment, KindTerminalOutcome} {
		e := testEnvelope()
		e.Kind = kind
		// These values are deliberately not validated as any domain schema.
		for _, payload := range []string{`null`, `42`, `[]`, `{"future_domain_field":true}`} {
			e.Payload = json.RawMessage(payload)
			var b bytes.Buffer
			if err := WriteFrame(&b, e, MaxFrameBytes); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadFrame(&b, MaxFrameBytes); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestStrictEnvelopeRejection(t *testing.T) {
	b, err := json.Marshal(testEnvelope())
	if err != nil {
		t.Fatal(err)
	}
	base := string(b)
	replace := func(old, new string) string { return strings.Replace(base, old, new, 1) }
	cases := map[string]string{
		"empty": "", "null": "null", "array": "[]", "malformed": "{",
		"trailing value": base + "{}", "trailing junk": base + "x",
		"unknown field":            strings.TrimSuffix(base, "}") + `,"future":1}`,
		"duplicate":                strings.TrimSuffix(base, "}") + `,"job_id":"job-1"}`,
		"escaped duplicate":        strings.TrimSuffix(base, "}") + `,"job_\u0069d":"job-1"}`,
		"conflicting duplicate":    strings.TrimSuffix(base, "}") + `,"job_id":"other"}`,
		"case alias":               replace(`"job_id"`, `"Job_ID"`),
		"missing":                  replace(`"job_id":"job-1",`, ""),
		"null identity":            replace(`"job-1"`, `null`),
		"empty identity":           replace(`"job-1"`, `""`),
		"space identity":           replace(`"job-1"`, `"job 1"`),
		"control identity":         replace(`"job-1"`, `"job\n1"`),
		"nonascii identity":        replace(`"job-1"`, `"job\u00e9"`),
		"long identity":            replace(`"job-1"`, `"`+strings.Repeat("x", MaxIdentifierBytes+1)+`"`),
		"zero sequence":            replace(`"sequence":1`, `"sequence":0`),
		"negative sequence":        replace(`"sequence":1`, `"sequence":-1`),
		"float sequence":           replace(`"sequence":1`, `"sequence":1.0`),
		"string sequence":          replace(`"sequence":1`, `"sequence":"1"`),
		"overflow sequence":        replace(`"sequence":1`, `"sequence":18446744073709551616`),
		"zero payload version":     replace(`"payload_version":7`, `"payload_version":0`),
		"overflow payload version": replace(`"payload_version":7`, `"payload_version":4294967296`),
		"null payload version":     replace(`"payload_version":7`, `"payload_version":null`),
		"unknown version":          replace(Version, "mixlab_worker_control_v2"),
		"unknown kind":             replace(`"kind":"assignment"`, `"kind":"execute"`),
		"duplicate payload field":  replace(string(testEnvelope().Payload), `{"a":1,"a":2}`),
		"nested duplicate":         replace(string(testEnvelope().Payload), `[{"a":1,"\u0061":2}]`),
		"deep payload":             replace(string(testEnvelope().Payload), strings.Repeat("[", maxJSONDepth+1)+"0"+strings.Repeat("]", maxJSONDepth+1)),
		"invalid utf8":             replace("value", string([]byte{0xff})),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ReadFrame(bytes.NewReader(rawFrame(body)), MaxFrameBytes); err == nil {
				t.Fatal("accepted invalid envelope")
			}
		})
	}
}

func TestFrameBoundsAndTruncation(t *testing.T) {
	body, _ := json.Marshal(testEnvelope())
	frame := rawFrame(string(body))
	for _, cut := range []int{1, 2, 3, 4, len(frame) - 1} {
		if _, err := ReadFrame(bytes.NewReader(frame[:cut]), MaxFrameBytes); !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
			t.Fatalf("cut %d: %v", cut, err)
		}
	}
	for _, size := range []uint32{0, MaxFrameBytes + 1, ^uint32(0)} {
		var h [4]byte
		binary.BigEndian.PutUint32(h[:], size)
		// No body exists: size validation must happen before a body read.
		if _, err := ReadFrame(bytes.NewReader(h[:]), MaxFrameBytes); !errors.Is(err, ErrFrameSize) {
			t.Fatal(err)
		}
	}
	for _, limit := range []uint32{0, MaxFrameBytes + 1, uint32(len(body) - 1)} {
		if _, err := ReadFrame(bytes.NewReader(frame), limit); !errors.Is(err, ErrFrameSize) {
			t.Fatal(err)
		}
		var dst bytes.Buffer
		if err := WriteFrame(&dst, testEnvelope(), limit); !errors.Is(err, ErrFrameSize) || dst.Len() != 0 {
			t.Fatalf("write: %v, %d bytes", err, dst.Len())
		}
	}
	var dst bytes.Buffer
	if err := WriteFrame(&dst, testEnvelope(), uint32(len(body))); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadFrame(&dst, uint32(len(body))); err != nil {
		t.Fatal(err)
	}
	e := testEnvelope()
	e.Payload = json.RawMessage(`"` + strings.Repeat("x", int(MaxFrameBytes)) + `"`)
	if err := WriteFrame(&dst, e, MaxFrameBytes); !errors.Is(err, ErrFrameSize) {
		t.Fatal(err)
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

type brokenIO struct{ err error }

func (b brokenIO) Write([]byte) (int, error) { return 0, b.err }
func (b brokenIO) Read([]byte) (int, error)  { return 0, b.err }

func TestFrameIOErrors(t *testing.T) {
	if err := WriteFrame(shortWriter{}, testEnvelope(), MaxFrameBytes); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	want := errors.New("connection failed")
	if err := WriteFrame(brokenIO{want}, testEnvelope(), MaxFrameBytes); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if _, err := ReadFrame(brokenIO{want}, MaxFrameBytes); !errors.Is(err, want) {
		t.Fatal(err)
	}
	e := testEnvelope()
	e.Kind = "unknown"
	var b bytes.Buffer
	if err := WriteFrame(&b, e, MaxFrameBytes); !errors.Is(err, ErrKind) || b.Len() != 0 {
		t.Fatal(err)
	}
}

func TestEnvelopeValidationAndMaximumFrame(t *testing.T) {
	for _, mutate := range []func(*Envelope){
		func(e *Envelope) { e.JobID = "" },
		func(e *Envelope) { e.AttemptID = "" },
		func(e *Envelope) { e.CorrelationID = "" },
		func(e *Envelope) { e.PayloadKind = "" },
		func(e *Envelope) { e.PayloadVersion = 0 },
		func(e *Envelope) { e.Sequence = 0 },
		func(e *Envelope) { e.Payload = nil },
		func(e *Envelope) { e.Payload = []byte(`{"truncated":`) },
		func(e *Envelope) { e.Payload = []byte{'"', 0xff, '"'} },
	} {
		e := testEnvelope()
		mutate(&e)
		if err := e.Validate(); !errors.Is(err, ErrEnvelope) {
			t.Fatal(err)
		}
	}
	e := testEnvelope()
	e.Payload = []byte(`""`)
	base, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	e.Payload = []byte(`"` + strings.Repeat("x", int(MaxFrameBytes)-len(base)) + `"`)
	var out bytes.Buffer
	if err := WriteFrame(&out, e, MaxFrameBytes); err != nil {
		t.Fatal(err)
	}
	if out.Len() != int(MaxFrameBytes)+4 {
		t.Fatal("did not exercise exact maximum")
	}
	if _, err := ReadFrame(&out, MaxFrameBytes); err != nil {
		t.Fatal(err)
	}
	e.Payload = append(e.Payload[:len(e.Payload)-1], 'x', '"')
	if err := WriteFrame(&out, e, MaxFrameBytes); !errors.Is(err, ErrFrameSize) {
		t.Fatal(err)
	}
}

func TestReadPreservesRawPayload(t *testing.T) {
	base, _ := json.Marshal(testEnvelope())
	payload := `{ "owner_field" : [ 1, true, "value" ] }`
	body := strings.Replace(string(base), string(testEnvelope().Payload), payload, 1)
	e, err := ReadFrame(bytes.NewReader(rawFrame(body)), MaxFrameBytes)
	if err != nil {
		t.Fatal(err)
	}
	if string(e.Payload) != payload {
		t.Fatal("decoder altered raw payload")
	}
}

func FuzzReadFrame(f *testing.F) {
	b, _ := json.Marshal(testEnvelope())
	f.Add(rawFrame(string(b)))
	f.Add([]byte{255, 255, 255, 255})
	f.Add(rawFrame(`{"job_id":null}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		e, err := ReadFrame(bytes.NewReader(input), MaxFrameBytes)
		if err != nil {
			return
		}
		var out bytes.Buffer
		if err := WriteFrame(&out, e, MaxFrameBytes); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadFrame(&out, MaxFrameBytes); err != nil {
			t.Fatal(err)
		}
	})
}
