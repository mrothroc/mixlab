package workerprobe

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/workercontrol"
)

func TestProbeReportStrictCapabilities(t *testing.T) {
	r := Report{Version: Version, BuildID: strings.Repeat("a", 64), BuildVersion: "test", WorkerProtocol: workercontrol.Version, OS: "darwin", Arch: "amd64", DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Report){func(r *Report) { r.BuildID = "bad" }, func(r *Report) { r.Available = true }, func(r *Report) { r.DTypes = []string{"fp32"} }, func(r *Report) { r.Backends = []string{"ring", "ring"} }, func(r *Report) { r.CustomOps = nil }} {
		copy := r
		change(&copy)
		if err := copy.Validate(); err == nil {
			t.Fatal("accepted invalid probe", copy)
		}
	}
	for _, invalid := range [][]byte{append(b, []byte("{}")...), []byte(strings.Replace(string(b), `"available":false`, `"available":false,"available":false`, 1)), []byte(strings.Repeat("x", MaxBytes+1))} {
		if _, err := Decode(invalid); err == nil {
			t.Fatal("accepted ambiguous/oversized response")
		}
	}
	r.Available = true
	r.DeviceKind = "metal"
	r.DeviceName = "test device"
	r.MLXVersion = "0.32.1"
	r.MLXSupported = true
	r.DTypes = []string{"bf16", "fp32"}
	total, free := uint64(100), uint64(50)
	r.MemoryBytes = &total
	r.FreeMemoryBytes = &free
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	free = 101
	if err := r.Validate(); err == nil {
		t.Fatal("invalid free memory")
	}
}
