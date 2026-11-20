package gpu

import (
	"os"
	"strings"
	"testing"
)

const pinnedMLXCommit = "36969c094354086e15d2a327a9c50feb20b9cf11"

func TestCUDAImagePinsDistributedCapableMLX(t *testing.T) {
	base := readRepositoryFile(t, "../docker/base.Dockerfile")
	for _, required := range []string{
		"ARG MLX_VERSION=v0.32.4",
		"ARG MLX_COMMIT=" + pinnedMLXCommit,
		"test -f /usr/include/nccl.h",
		"libnccl\\.so",
		"NCCL_LIBRARIES:FILEPATH",
		"MIXLAB_MLX_CUDA_WORKER_FIX=1",
		"MIXLAB_MLX_CONV_TRANSPOSE_GRAD_FIX=1",
		"MIXLAB_MLX_CUDA_ALLOCATOR_FIX=2",
		"MIXLAB_MLX_CUDA_CONV_PLAN_FIX=1",
		"MIXLAB_MLX_CUDA_JIT_CCCL_FIX=1",
		"python3 /tmp/test_mlx_jit_cccl.py --source /opt/mlx --expect-broken",
		"git -C /opt/mlx apply --check /tmp/mlx-cuda-jit-cccl-install-dir.patch",
		"git -C /opt/mlx apply /tmp/mlx-cuda-jit-cccl-install-dir.patch",
		"/tmp/test_mlx_jit_cccl_probe.cpp",
		"git -C /opt/mlx apply --check /tmp/mlx-cuda-conv-plan-fallback.patch",
		"python3 /tmp/test_mlx_conv_fallback.py --source /opt/mlx/mlx/backend/cuda/conv.cpp --expect-broken",
		"python3 /tmp/test_mlx_allocator_retry.py --source /opt/mlx/mlx/backend/cuda/allocator.cpp",
		"git -C /opt/mlx apply --check /tmp/mlx-cuda-allocator-reclaim.patch",
		"python3 /tmp/test_mlx_allocator.py --source /opt/mlx/mlx/backend/cuda/allocator.cpp --expect-broken",
		"git -C /opt/mlx apply --check /tmp/mlx-conv-transpose-weight-grad.patch",
		"git -C /opt/mlx apply /tmp/mlx-conv-transpose-weight-grad.patch",
		"python3 /tmp/test_mlx_worker.py --source /opt/mlx/mlx/backend/cuda/worker.cpp",
	} {
		if !strings.Contains(base, required) {
			t.Errorf("docker/base.Dockerfile is missing %q", required)
		}
	}
	// v0.32.4 fixed the idle worker spin upstream; the local patch must stay gone.
	if strings.Contains(base, "mlx-cuda-worker-wait.patch") || strings.Contains(base, "--expect-spin") {
		t.Error("docker/base.Dockerfile still applies the worker patch that MLX v0.32.4 made redundant")
	}
	if _, err := os.Stat("../docker/patches/mlx-cuda-worker-wait.patch"); err == nil {
		t.Error("docker/patches/mlx-cuda-worker-wait.patch is redundant on MLX v0.32.4")
	}

	addarch := readRepositoryFile(t, "../docker/addarch.Dockerfile")
	for _, required := range []string{
		"ARG MLX_VERSION=v0.32.4",
		"ARG MLX_COMMIT=" + pinnedMLXCommit,
		"MIXLAB_MLX_BUILD_VERSION",
		"MIXLAB_MLX_CUDA_WORKER_FIX",
		"MIXLAB_MLX_CONV_TRANSPOSE_GRAD_FIX",
		"MIXLAB_MLX_CUDA_ALLOCATOR_FIX",
		"MIXLAB_MLX_CUDA_CONV_PLAN_FIX",
		`test "${MIXLAB_MLX_CUDA_JIT_CCCL_FIX}" = "1"`,
		"/tmp/test_mlx_jit_cccl_probe.cpp",
		"NCCL_LIBRARIES:FILEPATH",
	} {
		if !strings.Contains(addarch, required) {
			t.Errorf("docker/addarch.Dockerfile is missing %q", required)
		}
	}

	baseBuild := readRepositoryFile(t, "../docker/cloudbuild-mlx-cuda-base.yaml")
	for _, required := range []string{
		`_MLX_VERSION: "v0.32.4"`,
		`_MLX_COMMIT: "` + pinnedMLXCommit + `"`,
		"golf-mlx-cuda-base:mlx-0.32.4",
	} {
		if !strings.Contains(baseBuild, required) {
			t.Errorf("docker/cloudbuild-mlx-cuda-base.yaml is missing %q", required)
		}
	}

	archBuild := readRepositoryFile(t, "../docker/cloudbuild-golf-mlx-cuda.yaml")
	for _, required := range []string{
		// The tag also carries the Go version; cmd/mixlab's toolchain test owns that part.
		`_IMAGE_TAG: "mlx-0.32.4-go`,
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
	app := readRepositoryFile(t, "../docker/app.Dockerfile")
	// The encoder-recovery and consecutive-miss thrash fixes come from the MLX
	// release itself, not a marker-carrying patch, so the app checks the version.
	for _, required := range []string{
		"ARG MLX_VERSION=v0.32.4",
		`test "${MIXLAB_MLX_BUILD_VERSION}" = "${MLX_VERSION}"`,
	} {
		if !strings.Contains(app, required) {
			t.Errorf("docker/app.Dockerfile is missing %q", required)
		}
	}
	if !strings.Contains(app, `test "${MIXLAB_MLX_CUDA_JIT_CCCL_FIX}" = "1"`) {
		t.Fatal("app image must reject an MLX base whose JIT can fall back to the toolkit's CCCL")
	}
	if !strings.Contains(app, `test "${MIXLAB_MLX_CUDA_CONV_PLAN_FIX}" = "1"`) {
		t.Fatal("app image must reject an MLX base without strict convolution fallback")
	}
	if !strings.Contains(app, `test "${MIXLAB_MLX_CUDA_ALLOCATOR_FIX}" = "2"`) {
		t.Fatal("app image must reject an MLX CUDA base without allocator reclamation")
	}
	if !strings.Contains(app, `test "${MIXLAB_MLX_CONV_TRANSPOSE_GRAD_FIX}" = "1"`) {
		t.Fatal("app image must reject an MLX CUDA base without the convolution gradient fix")
	}
	if !strings.Contains(app, `test "${MIXLAB_MLX_CUDA_WORKER_FIX}" = "1"`) {
		t.Fatal("app image must reject an unpatched MLX CUDA base")
	}
}

func TestCUDAAllocatorPatchIsCoveredInCI(t *testing.T) {
	ci := readRepositoryFile(t, "../.github/workflows/ci.yml")
	for _, required := range []string{"mlx-cuda-allocator-reclaim.patch", "test_mlx_allocator.py", "test_mlx_allocator_retry.py", "--expect-broken", "allocator.cpp allocator.h eval.cpp cudnn_utils.cpp conv/gemm_conv.cu conv/gemm_grouped_conv.cu", pinnedMLXCommit + "/mlx/backend/cuda/$FILE"} {
		if !strings.Contains(ci, required) {
			t.Errorf("CI missing %s", required)
		}
	}
}

func TestCUDAConvolutionFallbackPatchIsCoveredInCI(t *testing.T) {
	ci := readRepositoryFile(t, "../.github/workflows/ci.yml")
	for _, required := range []string{"mlx-cuda-conv-plan-fallback.patch", "test_mlx_conv_fallback.py", pinnedMLXCommit + "/mlx/backend/cuda/conv.cpp"} {
		if !strings.Contains(ci, required) {
			t.Errorf("CI missing %s", required)
		}
	}
}

func TestCUDAJITCCCLPatchIsCoveredInCI(t *testing.T) {
	ci := readRepositoryFile(t, "../.github/workflows/ci.yml")
	for _, required := range []string{"mlx-cuda-jit-cccl-install-dir.patch", "test_mlx_jit_cccl.py", "--expect-broken", pinnedMLXCommit + "/mlx/backend/cuda/$FILE", "CMakeLists.txt dirs.cpp"} {
		if !strings.Contains(ci, required) {
			t.Errorf("CI missing %s", required)
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
