// GPU-free check of the installed static libmlx: its compiled-in CCCL fallback
// must name headers that NVRTC can use for cuda::std::rint, and the CUDA
// toolkit's own CCCL alone must not (the control that proves the check bites).
//
// Link only the archive member under test; the rest of libmlx needs the CUDA
// runtime. Build: ar x /usr/local/lib/libmlx.a dirs.cpp.o && g++ -std=c++17 \
//   test_mlx_jit_cccl_probe.cpp dirs.cpp.o -I/usr/local/cuda/include \
//   -L/usr/local/cuda/lib64 -lnvrtc
// Run:   probe <expected cccl dir> <cuda include dir>

#include <nvrtc.h>

#include <cstdio>
#include <string>
#include <vector>

namespace mlx::core::cu {
const char* cccl_dir();
}

namespace {

const char* kKernel = R"(
#include <cuda/std/cmath>
#if !defined(CCCL_VERSION) || CCCL_VERSION < 3000000
#error "CCCL older than 3.0 has no cuda::std::rint under NVRTC"
#endif
extern "C" __global__ void round_kernel(float* x) { x[0] = cuda::std::rint(x[0]); }
)";

bool compiles(const std::vector<std::string>& include_dirs, std::string& log) {
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, kKernel, "probe.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
    log = "nvrtcCreateProgram failed";
    return false;
  }
  std::vector<std::string> args = {"--gpu-architecture=compute_80", "-std=c++17"};
  for (const auto& dir : include_dirs) {
    args.push_back("--include-path=" + dir);
  }
  std::vector<const char*> argv;
  for (const auto& arg : args) {
    argv.push_back(arg.c_str());
  }
  const bool ok = nvrtcCompileProgram(prog, static_cast<int>(argv.size()), argv.data()) == NVRTC_SUCCESS;
  size_t size = 0;
  nvrtcGetProgramLogSize(prog, &size);
  log.assign(size, '\0');
  if (size > 0) {
    nvrtcGetProgramLog(prog, log.data());
  }
  nvrtcDestroyProgram(&prog);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <expected cccl dir> <cuda include dir>\n", argv[0]);
    return 2;
  }
  const std::string expected = argv[1];
  const std::string toolkit = argv[2];
  const char* dir = mlx::core::cu::cccl_dir();
  if (!dir || expected != dir) {
    std::fprintf(stderr, "libmlx cccl_dir() = %s, want %s\n", dir ? dir : "(null)", expected.c_str());
    return 1;
  }
  std::string log;
  // MLX passes the CCCL directory before the toolkit include directory.
  if (!compiles({dir, toolkit}, log)) {
    std::fprintf(stderr, "cuda::std::rint failed with libmlx's CCCL fallback:\n%s\n", log.c_str());
    return 1;
  }
  if (compiles({toolkit}, log)) {
    std::fprintf(stderr, "control compiled with toolkit CCCL only; the probe no longer detects the defect\n");
    return 1;
  }
  std::printf("libmlx CCCL fallback %s compiles cuda::std::rint; toolkit-only control fails as expected\n", dir);
  return 0;
}
