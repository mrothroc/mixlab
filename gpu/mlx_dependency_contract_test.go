package gpu

import (
	"os"
	"strings"
	"testing"
)

const pinnedMLXCommit = "7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"

func TestCUDAImagePinsDistributedCapableMLX(t *testing.T) {
	base := readRepositoryFile(t, "../docker/base.Dockerfile")
	for _, required := range []string{
		"ARG MLX_VERSION=v0.32.0",
		"ARG MLX_COMMIT=" + pinnedMLXCommit,
		"test -f /usr/include/nccl.h",
		"libnccl\\.so",
		"NCCL_LIBRARIES:FILEPATH",
	} {
		if !strings.Contains(base, required) {
			t.Errorf("docker/base.Dockerfile is missing %q", required)
		}
	}

	addarch := readRepositoryFile(t, "../docker/addarch.Dockerfile")
	for _, required := range []string{
		"ARG MLX_VERSION=v0.32.0",
		"ARG MLX_COMMIT=" + pinnedMLXCommit,
		"MIXLAB_MLX_BUILD_VERSION",
		"NCCL_LIBRARIES:FILEPATH",
	} {
		if !strings.Contains(addarch, required) {
			t.Errorf("docker/addarch.Dockerfile is missing %q", required)
		}
	}

	baseBuild := readRepositoryFile(t, "../docker/cloudbuild-mlx-cuda-base.yaml")
	for _, required := range []string{
		`_MLX_VERSION: "v0.32.0"`,
		`_MLX_COMMIT: "` + pinnedMLXCommit + `"`,
		"golf-mlx-cuda-base:mlx-0.32.0",
	} {
		if !strings.Contains(baseBuild, required) {
			t.Errorf("docker/cloudbuild-mlx-cuda-base.yaml is missing %q", required)
		}
	}

	archBuild := readRepositoryFile(t, "../docker/cloudbuild-golf-mlx-cuda.yaml")
	for _, required := range []string{
		`_IMAGE_TAG: "mlx-0.32.0"`,
		"${_REGISTRY_PREFIX}/golf-mlx-cuda:${_IMAGE_TAG}",
	} {
		if !strings.Contains(archBuild, required) {
			t.Errorf("docker/cloudbuild-golf-mlx-cuda.yaml is missing %q", required)
		}
	}

	appBuild := readRepositoryFile(t, "../docker/cloudbuild-ci.yaml")
	for _, required := range []string{
		"_IMAGE_TAG: 'latest'",
		"_RUNPOD_TAG: 'runpod'",
		"${_REGISTRY_PREFIX}/mixlab:${_IMAGE_TAG}",
		"${_REGISTRY_PREFIX}/mixlab:${_RUNPOD_TAG}",
	} {
		if !strings.Contains(appBuild, required) {
			t.Errorf("docker/cloudbuild-ci.yaml is missing %q", required)
		}
	}
}

func TestCUDAJITDispatchUsesBackendDevice(t *testing.T) {
	dispatch := readRepositoryFile(t, "cuda_kernel_dispatch.cpp")
	if !strings.Contains(dispatch, "mx::cu::device(stream.device)") {
		t.Fatal("CUDA JIT dispatch must resolve the generic stream device through mx::cu::device")
	}
	if strings.Contains(dispatch, "get_jit_module(\n          stream.device") ||
		strings.Contains(dispatch, "get_jit_module(\n        stream.device") {
		t.Fatal("CUDA JIT dispatch passes generic stream.device directly to get_jit_module")
	}
}

func readRepositoryFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}
