"""CPU regression of pinned MLX allocator cache/reclaim methods with CUDA stubs.

Extracts the actual method bodies, not a reimplementation. GPU pressure/order
acceptance lives separately in test_mlx_allocator_cuda.cpp.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def method(source, name):
    start = source.index(name)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


HEADER = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>
int current_device = 7;
std::vector<int> events;
struct Mutex {
  bool held = false;
  void lock() { assert(!held); held = true; }
  void unlock() { held = false; }
};
Mutex* allocator_mutex;
bool unlocked_wait = true;
int cudaGetDevice(int* d) { *d = current_device; return 0; }
int cudaSetDevice(int d) { current_device = d; return 0; }
int cudaStreamSynchronize(int s) {
  if (allocator_mutex->held) unlocked_wait = false;
  events.push_back(s); return 0;
}
int cudaMemPoolTrimTo(int p, int) { events.push_back(p); return 0; }
#define CHECK_CUDA_ERROR(x) assert((x) == 0)
namespace cu {
struct Device { int id; void make_current() { current_device = id; } };
Device device(int id) { return Device{id}; }
}
struct CudaBuffer { size_t size; };
struct Buffer { CudaBuffer* p; void* ptr() { return p; } };
struct Cache {
  size_t bytes = 0;
  void recycle_to_cache(CudaBuffer* b) { bytes += b->size; delete b; }
  void release_cached_buffers(size_t n) { bytes -= std::min(bytes,n); }
  void clear() { bytes = 0; }
};
class CudaAllocator {
 public:
  Mutex mutex_;
  Cache buffer_cache_;
  size_t max_pool_size_ = 100, active_memory_ = 0, freed = 0;
  std::vector<int> mem_pools_{10,20}, free_streams_{1,2};
  size_t get_cache_memory() { return buffer_cache_.bytes; }
  void free_cuda_buffer(CudaBuffer* b) { freed += b->size; delete b; }
  void free(Buffer);
  size_t set_cache_limit(size_t);
  void clear_cache();
};
"""

MAIN = r"""
int main(int argc, char**) {
  int failures = 0;
  // Last buffer must fit, zero disables caching, exact fit is permitted.
  for (auto [cached,size,limit,want] : std::vector<std::array<size_t,4>>{
      {90,20,100,90}, {0,120,100,0}, {0,1,0,0}, {50,50,100,100},
      {120,1,100,120}}) {
    CudaAllocator a;
    a.buffer_cache_.bytes = cached; a.max_pool_size_ = limit;
    a.active_memory_ = size;
    a.free(Buffer{new CudaBuffer{size}});
    failures += a.get_cache_memory() != want || a.active_memory_ != 0;
  }
  CudaAllocator a;
  a.buffer_cache_.bytes = 80;
  failures += a.set_cache_limit(20) != 100 || a.get_cache_memory() > 20;
  failures += a.set_cache_limit(0) != 20 || a.get_cache_memory() != 0;
  allocator_mutex = &a.mutex_;
  a.clear_cache();
  failures += events != std::vector<int>{1,10,2,20};
  failures += !unlocked_wait || current_device != 7;
  a.free(Buffer{nullptr});
  a.free(Buffer{new CudaBuffer{0}});
  std::cout << "allocator_contract_failures=" << failures << std::endl;
  return (argc > 1 ? failures > 0 : failures == 0) ? 0 : 1;
}
"""


def run_probe(source, expect_broken=False):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("C++ compiler required")
    text = source.read_text()
    bodies = "\n".join(method(text, name) for name in (
        "void CudaAllocator::free(Buffer", "size_t CudaAllocator::set_cache_limit(",
        "void CudaAllocator::clear_cache("))
    with tempfile.TemporaryDirectory(prefix="mixlab-allocator-probe-") as tmp:
        cpp, binary = Path(tmp)/"probe.cpp", Path(tmp)/"probe"
        cpp.write_text(HEADER + bodies + MAIN)
        subprocess.run([compiler, "-std=c++20", "-O2", "-pthread", str(cpp), "-o", str(binary)], check=True, timeout=60)
        return subprocess.run([str(binary)] + (["broken"] if expect_broken else []), check=True, timeout=10, capture_output=True, text=True).stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-broken", action="store_true")
    args = parser.parse_args()
    print(run_probe(args.source, args.expect_broken), end="")
