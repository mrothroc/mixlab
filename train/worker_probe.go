package train

import (
	"encoding/json"
	"io"
	"os"
	"runtime"
	"slices"

	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/buildinfo"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

// RunWorkerProbe reports the installed executable/device without starting a
// process group, reading datasets, constructing a trainer or loading keys.
func RunWorkerProbe(out io.Writer) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	hash, err := workerjob.FileDigest(exe)
	if err != nil {
		return err
	}
	info := gpu.RuntimeInfo()
	r := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: buildinfo.Report("mixlab"), WorkerProtocol: buildinfo.WorkerProtocol, OS: runtime.GOOS, Arch: runtime.GOARCH, Available: info.Available, MLXVersion: info.Version, MLXSupported: info.VersionSupported, DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	if info.Ring {
		r.Backends = append(r.Backends, "ring")
	}
	if info.NCCL {
		r.Backends = append(r.Backends, "nccl")
	}
	slices.Sort(r.Backends)
	if info.Available {
		r.DeviceKind = "cuda"
		if runtime.GOOS == "darwin" {
			r.DeviceKind = "metal"
		}
		r.DeviceName = info.Device
		r.DTypes = []string{"bf16", "fp32"}
		// This names the linked IR implementation, not the presence of optional
		// fast kernels. Exact build equality remains a recruitment requirement.
		r.CustomOps = []string{"mixlab-ir-v1"}
		if memory, ok := gpu.DeviceMemoryInfo(); ok && memory.TotalBytes > 0 {
			r.MemoryBytes = &memory.TotalBytes
			if memory.FreeBytes > 0 {
				r.FreeMemoryBytes = &memory.FreeBytes
			}
		}
	}
	if err := r.Validate(); err != nil {
		return err
	}
	return json.NewEncoder(out).Encode(r)
}
