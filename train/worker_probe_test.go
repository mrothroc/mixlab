package train

import (
	"bytes"
	"testing"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestWorkerProbeContract(t *testing.T) {
	var out bytes.Buffer
	if err := RunWorkerProbe(&out); err != nil {
		t.Fatal(err)
	}
	r, err := workerprobe.Decode(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if r.BuildID == "" || r.BuildVersion == "" {
		t.Fatal("missing executable identity")
	}
	t.Logf("device_available=%t kind=%s mlx=%s supported=%t", r.Available, r.DeviceKind, r.MLXVersion, r.MLXSupported)
}
