// Package workerprobe defines the approved executable's read-only device probe
// contract. It is shared by the training CLI and MLX-free hosting adapter.
package workerprobe

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/mrothroc/mixlab/workercontrol"
)

const Version = "mixlab_worker_probe_v1"
const MaxBytes = 64 << 10

type Report struct {
	Version         string   `json:"version"`
	BuildID         string   `json:"build_id"`
	BuildVersion    string   `json:"build_version"`
	WorkerProtocol  string   `json:"worker_protocol"`
	OS              string   `json:"os"`
	Arch            string   `json:"arch"`
	Available       bool     `json:"available"`
	MLXVersion      string   `json:"mlx_version"`
	MLXSupported    bool     `json:"mlx_supported"`
	DeviceKind      string   `json:"device_kind"`
	DeviceName      string   `json:"device_name"`
	MemoryBytes     *uint64  `json:"memory_bytes"`
	FreeMemoryBytes *uint64  `json:"free_memory_bytes"`
	Backends        []string `json:"backends"`
	DTypes          []string `json:"dtypes"`
	CustomOps       []string `json:"custom_ops"`
}

func (r Report) Validate() error {
	h, e := hex.DecodeString(r.BuildID)
	if e != nil || len(h) != 32 || hex.EncodeToString(h) != r.BuildID || r.Version != Version || r.WorkerProtocol != workercontrol.Version || len(r.BuildVersion) == 0 || len(r.BuildVersion) > 512 || (r.OS != "darwin" && r.OS != "linux") || (r.Arch != "arm64" && r.Arch != "amd64") || len(r.MLXVersion) > 128 || len(r.DeviceName) > 256 {
		return fmt.Errorf("invalid worker probe identity/profile")
	}
	if r.Backends == nil || r.DTypes == nil || r.CustomOps == nil {
		return fmt.Errorf("probe capability lists must be explicit")
	}
	for _, list := range [][]string{r.Backends, r.DTypes, r.CustomOps} {
		if len(list) > 256 || !slices.IsSorted(list) {
			return fmt.Errorf("unordered/oversized probe capabilities")
		}
		for i, s := range list {
			if s == "" || len(s) > 128 || (i > 0 && s == list[i-1]) {
				return fmt.Errorf("invalid/duplicate probe capability")
			}
		}
	}
	for _, b := range r.Backends {
		if b != "ring" && b != "nccl" {
			return fmt.Errorf("unknown distributed backend")
		}
	}
	for _, d := range r.DTypes {
		if d != "fp32" && d != "bf16" {
			return fmt.Errorf("unknown compute dtype")
		}
	}
	if !r.Available {
		if r.DeviceKind != "none" || r.DeviceName != "" || r.MemoryBytes != nil || r.FreeMemoryBytes != nil || len(r.DTypes) > 0 || len(r.CustomOps) > 0 {
			return fmt.Errorf("unavailable device claims compute capabilities")
		}
		return nil
	}
	if (r.DeviceKind != "metal" && r.DeviceKind != "cuda") || r.DeviceName == "" || r.MLXVersion == "" || len(r.DTypes) == 0 {
		return fmt.Errorf("missing live device capabilities")
	}
	if r.MemoryBytes != nil && *r.MemoryBytes == 0 {
		return fmt.Errorf("unknown memory must be null, not zero")
	}
	if r.FreeMemoryBytes != nil && (r.MemoryBytes == nil || *r.FreeMemoryBytes > *r.MemoryBytes) {
		return fmt.Errorf("invalid free device memory")
	}
	return nil
}
func Decode(b []byte) (Report, error) {
	var r Report
	if len(b) > MaxBytes {
		return r, fmt.Errorf("worker probe response too large")
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return r, err
	}
	canonical, err := json.Marshal(r)
	if err != nil || !bytes.Equal(bytes.TrimSuffix(b, []byte("\n")), canonical) {
		return r, fmt.Errorf("noncanonical worker probe response")
	}
	return r, r.Validate()
}
