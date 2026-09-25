package train

import (
	"fmt"
	"math"
)

func distributedBackend(request, platform string) (string, error) {
	if request == "" || request == "auto" {
		switch platform {
		case "darwin":
			request = "ring"
		case "linux":
			request = "nccl"
		}
	}
	if (request == "ring" && platform == "darwin") || (request == "nccl" && platform == "linux") {
		return request, nil
	}
	return "", fmt.Errorf("DDP backend %q is unsupported on %s; ring requires Metal/macOS and nccl requires CUDA/Linux", request, platform)
}

func distributedRootDecision(group distributedCheckpointControl, stop bool, localErr error) (bool, error) {
	values := []int32{0, 0}
	if stop {
		values[0] = 1
	}
	if localErr != nil {
		values[1] = 1
	}
	observed, err := group.BroadcastControl(0, values)
	if err != nil {
		return false, err
	}
	if len(observed) != 2 {
		return false, fmt.Errorf("invalid distributed root decision")
	}
	if observed[1] != 0 {
		if localErr != nil {
			return false, localErr
		}
		return false, fmt.Errorf("rank zero failed during validation, telemetry, or artifact publication")
	}
	return observed[0] != 0, nil
}

func distributedMeanLoss(group distributedCheckpointControl, local float64) (float64, error) {
	sum := 0.0
	bits := math.Float64bits(local)
	for root := 0; root < group.WorldSize(); root++ {
		v, err := group.BroadcastControl(root, []int32{int32(uint32(bits)), int32(uint32(bits >> 32))})
		if err != nil {
			return 0, err
		}
		if len(v) != 2 {
			return 0, fmt.Errorf("invalid distributed loss control")
		}
		sum += math.Float64frombits(uint64(uint32(v[0])) | uint64(uint32(v[1]))<<32)
	}
	return sum / float64(group.WorldSize()), nil
}
