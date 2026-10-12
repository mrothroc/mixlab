"""Fault-inject the actual patched CUDA allocation retry methods without a GPU."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from test_mlx_allocator import method


HEADER = r"""
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <vector>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaStreamCaptureStatus = int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorInvalidValue=3;
constexpr int cudaStreamCaptureStatusNone=0;
constexpr int cudaMemPoolAttrReservedMemCurrent=1, cudaMemPoolAttrUsedMemCurrent=2;
std::vector<int> results;
int allocations, reclaims, syncs, capture, sync_error, diagnostics_error;
int cudaMallocAsync(void** p, size_t, void*) {
  auto result = results.at(allocations++);
  if (!result) *p = reinterpret_cast<void*>(42);
  return result;
}
int cudaMalloc(void** p, size_t n) { return cudaMallocAsync(p,n,nullptr); }
int cudaGetLastError() { return 0; }
int cudaStreamIsCapturing(void*, int* result) { *result=capture; return 0; }
int cudaStreamSynchronize(void*) { ++syncs; return sync_error; }
int cudaMemGetInfo(size_t* f, size_t* t) { *f=100; *t=200; return diagnostics_error; }
int cudaMemPoolGetAttribute(int, int, uint64_t* out) { *out=40; return diagnostics_error; }
const char* cudaGetErrorString(int s) { return s ? "failure" : "success"; }
#define CHECK_CUDA_ERROR(x) do { int error=(x); if(error) throw error; } while(0)
namespace cu {
struct Device { void make_current() {} };
Device device(int) { return {}; }
}
class CudaAllocator {
 public:
  std::vector<int> mem_pools_{1};
  void clear_cache() { ++reclaims; }
  void log_pressure(size_t,int,void*,const char*,int);
  void* allocate_device(size_t,int,void*);
};
"""

MAIN = r"""
int main() {
  const int oom=cudaErrorMemoryAllocation, other=cudaErrorInvalidValue;
  for (bool pool : {false,true}) {
    for (int scenario=0; scenario<9; ++scenario) {
      allocations=reclaims=syncs=capture=sync_error=diagnostics_error=0;
      results={0}; int expected_error=0, expected_alloc=1, expected_reclaim=0, expected_sync=0;
      switch(scenario) {
        case 1: results={other}; expected_error=other; break;
        case 2: results={oom,0}; expected_alloc=2; expected_reclaim=1; break;
        case 3: results={oom,oom,0}; expected_alloc=3; expected_reclaim=2; expected_sync=1; break;
        case 4: results={oom,oom,oom}; expected_alloc=3; expected_reclaim=2; expected_sync=1; expected_error=oom; break;
        case 5: results={oom}; capture=1; expected_error=oom; break;
        case 6: results={oom,other}; expected_alloc=2; expected_reclaim=1; expected_error=other; break;
        case 7: results={oom,oom}; expected_alloc=2; expected_reclaim=1; expected_sync=1; sync_error=other; expected_error=other; break;
        case 8: results={oom,0}; diagnostics_error=other; expected_alloc=2; expected_reclaim=1; break;
      }
      CudaAllocator a; a.mem_pools_[0]=pool;
      int error=0;
      try { assert(a.allocate_device(3460300800ULL,0,reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(42)); }
      catch (int e) { error=e; }
      assert(error==expected_error);
      assert(allocations==expected_alloc && reclaims==expected_reclaim && syncs==expected_sync);
    }
  }
  assert(std::string(allocation_op)=="unknown");
  try {
    AllocationScope op("Convolution",nullptr);
    { AllocationScope workspace(nullptr,"cudnn-workspace");
      assert(std::string(allocation_op)=="Convolution");
      assert(std::string(allocation_kind)=="cudnn-workspace");
    }
    assert(std::string(allocation_kind)=="tensor");
    throw 1;
  } catch(int) {}
  assert(std::string(allocation_op)=="unknown" && std::string(allocation_kind)=="tensor");
  std::puts("allocator_retry_contract=passed");
}
"""


def run(source):
    text = source.read_text()
    header = source.with_suffix(".h").read_text()
    scope = header[header.index("inline thread_local"):header.index("// Stores cuda-managed")]
    bodies = "\n".join(method(text, name) for name in (
        "void CudaAllocator::log_pressure(", "void* CudaAllocator::allocate_device("))
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("C++ compiler required")
    with tempfile.TemporaryDirectory(prefix="mixlab-retry-probe-") as tmp:
        cpp, binary = Path(tmp)/"probe.cpp", Path(tmp)/"probe"
        cpp.write_text(HEADER + scope + bodies + MAIN)
        subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(binary)], check=True, timeout=60)
        result = subprocess.run([str(binary)], check=True, timeout=10, capture_output=True, text=True)
        for field in ("bytes=3460300800", "stage=initial", "stage=reclaim-retry-1",
                      "stage=stream-sync-reclaim-retry-2", "stage=capture-no-retry",
                      "stage=stream-sync-failed", "memory_valid=0", "pool_valid=0"):
            if field not in result.stderr:
                raise AssertionError(f"missing allocation diagnostic: {field}")
        print(result.stdout, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    run(parser.parse_args().source)
