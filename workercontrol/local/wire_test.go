package local

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"testing"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

func fixtureMetadata() metadata {
	return metadata{Binding: wc.Binding{JobID: "job", AttemptID: "attempt", BinaryID: "mixlab", BuildID: "build", AssignmentDigest: [32]byte{1}},
		Agent: wc.PeerIdentity{PID: 42, UID: 501}}
}

func TestHelloRoundTripAndMalformed(t *testing.T) {
	m := fixtureMetadata()
	var buf bytes.Buffer
	proof := [wc.CapabilityBytes]byte{1, 2, 3}
	if err := writeHello(&buf, m, proof[:]); err != nil {
		t.Fatal(err)
	}
	original := bytes.Clone(buf.Bytes())
	var decoded [wc.CapabilityBytes]byte
	got, err := readHello(&buf, &decoded)
	if err != nil || got != m || proof != decoded || buf.Len() != 0 {
		t.Fatal("handshake roundtrip", err)
	}
	for _, size := range []int{0, 3, len(original) - 1} {
		if _, err := readHello(bytes.NewReader(original[:size]), &decoded); err == nil {
			t.Fatalf("accepted truncated %d", size)
		}
	}
	if err := writeHello(&buf, m, proof[:31]); !errors.Is(err, ErrHandshake) {
		t.Fatal(err)
	}
	if err := writeHello(shortWriter{}, m, proof[:]); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	for _, mode := range []string{"unknown", "case-alias", "wrong-kind", "wrong-job", "invalid-agent"} {
		t.Run(mode, func(t *testing.T) {
			e, err := wc.ReadFrame(bytes.NewReader(original), handshakeLimit)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "unknown":
				e.Payload = append([]byte(`{"extra":1,`), e.Payload[1:]...)
			case "case-alias":
				e.Payload = bytes.Replace(e.Payload, []byte(`"agent"`), []byte(`"Agent"`), 1)
			case "wrong-kind":
				e.Kind = wc.KindAssignment
			case "wrong-job":
				e.JobID = "other"
			case "invalid-agent":
				bad := m
				bad.Agent.PID = 0
				e.Payload, err = json.Marshal(bad)
				if err != nil {
					t.Fatal(err)
				}
			}
			var frame bytes.Buffer
			if err := wc.WriteFrame(&frame, e, handshakeLimit); err != nil {
				t.Fatal(err)
			}
			frame.Write(proof[:])
			if _, err := readHello(&frame, &decoded); !errors.Is(err, ErrHandshake) {
				t.Fatal(err)
			}
		})
	}
	var size [4]byte
	binary.BigEndian.PutUint32(size[:], handshakeLimit+1)
	if _, err := readHello(bytes.NewReader(size[:]), &decoded); !errors.Is(err, wc.ErrFrameSize) {
		t.Fatal(err)
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func FuzzHello(f *testing.F) {
	var fixture bytes.Buffer
	_ = writeHello(&fixture, fixtureMetadata(), make([]byte, wc.CapabilityBytes))
	f.Add(fixture.Bytes())
	f.Add([]byte("junk"))
	f.Fuzz(func(t *testing.T, raw []byte) {
		var proof [wc.CapabilityBytes]byte
		m, err := readHello(bytes.NewReader(raw), &proof)
		if err == nil && (m.Binding.Validate() != nil || wc.ValidatePeerIdentity(m.Agent, m.Agent) != nil) {
			t.Fatal("accepted invalid hello")
		}
	})
}
