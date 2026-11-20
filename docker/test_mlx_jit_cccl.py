"""CPU-only test of where MLX's CUDA JIT looks for CCCL when it cannot find headers
next to the executable.

Configures the pinned mlx/backend/cuda/CMakeLists.txt block that defines
MLX_CCCL_DIR into a scratch project, builds MLX's real dirs.cpp, and prints
cccl_dir(). Upstream defines it only for test builds, so a binary outside the
install prefix (a static libmlx resolves headers relative to the executable)
compiles JIT kernels against the CUDA toolkit's older CCCL.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile


START = "# The binary of C++ tests will not be installed"
END = "# Use fixed version of NVTX."

# dirs.cpp includes MLX's common utils header for current_binary_dir(), which
# pulls in the array headers. Declare just that function so the real dirs.cpp
# builds without the rest of MLX.
UTILS_STUB = """#pragma once
#include <filesystem>
namespace mlx::core {
std::filesystem::path current_binary_dir();
}
"""

MAIN = r"""
#include <cstdio>
#include <filesystem>
namespace mlx::core {
std::filesystem::path current_binary_dir() { return "/nonexistent"; }
}
namespace mlx::core::cu {
const char* cccl_dir();
}
int main() {
  const char* dir = mlx::core::cu::cccl_dir();
  std::printf("%s\n", dir ? dir : "(null)");
}
"""


def extract_block(cmake):
    text = cmake.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(f"{cmake}: expected exactly one MLX_CCCL_DIR block between fixed markers")
    block = text[text.index(START):text.index(END)]
    if "MLX_CCCL_DIR" not in block:
        raise SystemExit(f"{cmake}: marker block no longer defines MLX_CCCL_DIR")
    return block


def cccl_dir(source, block, build_tests, work):
    project = work / ("tests-on" if build_tests else "tests-off")
    project.mkdir()
    (project / "main.cpp").write_text(MAIN)
    stub = project / "stub/mlx/backend/common"
    stub.mkdir(parents=True)
    (stub / "utils.h").write_text(UTILS_STUB)
    (project / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(mlx_jit_cccl_probe CXX)
set(CMAKE_CXX_STANDARD 17)
set(MLX_BUILD_TESTS {"ON" if build_tests else "OFF"})
set(cccl_SOURCE_DIR "/mlx-build/_deps/cccl-src")
add_library(mlx_dirs OBJECT "{source / 'mlx/backend/cuda/dirs.cpp'}")
target_include_directories(mlx_dirs PRIVATE "{project / 'stub'}")
{block}
add_executable(probe main.cpp $<TARGET_OBJECTS:mlx_dirs>)
""")
    build = project / "build"
    prefix = work / "prefix"
    for cmd in (["cmake", "-S", str(project), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={prefix}"],
                ["cmake", "--build", str(build)]):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} failed:\n{result.stdout}{result.stderr}")
    out = subprocess.run([str(build / "probe")], check=True, capture_output=True, text=True)
    return out.stdout.strip(), f"{prefix}/include/cccl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="MLX source root")
    parser.add_argument("--expect-broken", action="store_true",
                        help="assert the unpatched behaviour: no fallback outside test builds")
    args = parser.parse_args()
    source = args.source.resolve()
    block = extract_block(source / "mlx/backend/cuda/CMakeLists.txt")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        release, installed = cccl_dir(source, block, False, work)
        tests, _ = cccl_dir(source, block, True, work)
    # Test builds keep upstream's source-tree path either way.
    if tests != "/mlx-build/_deps/cccl-src/include":
        raise SystemExit(f"test build cccl_dir() = {tests!r}, want the CCCL source include dir")
    if args.expect_broken:
        if release != "(null)":
            raise SystemExit(f"expected no fallback outside test builds, got {release!r}")
        print("unpatched: non-test builds have no CCCL fallback (JIT can use the toolkit's CCCL)")
        return
    if release != installed:
        raise SystemExit(f"non-test cccl_dir() = {release!r}, want installed headers {installed!r}")
    print(f"patched: non-test builds fall back to {release}")


if __name__ == "__main__":
    main()
