"""Compile the pinned MLX patch-size policy without a GPU; native VJP is separate."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from test_mlx_allocator import method


def run(source, expect_broken):
    text = source.read_text()
    name = "int conv_weight_patch_batch("
    if expect_broken:
        assert name not in text, "baseline unexpectedly already tiled"
        print("untiled baseline confirmed")
        return
    body = method(text, name)
    caller = method(text, "array conv_weight_backward_patches(")
    assert "conv_weight_patch_batch(in, wt, cotan)" in caller
    assert "begin = end;" in caller and "slice(cotan, starts, cotan_ends, s)" in caller
    header = r'''
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <vector>
enum DType { float32, float16 };
struct array {
  std::vector<int> dims;
  DType type = float32;
  int shape(int i) const { return dims.at(i); }
  int ndim() const { return dims.size(); }
  DType dtype() const { return type; }
};
'''
    tests = r'''
int main() {
  assert(conv_weight_patch_batch({{31,256,256,32}}, {{2,3,3,32}}, {{31,256,256,2}}) == 3);
  assert(conv_weight_patch_batch({{5,256,256,32}}, {{2,3,3,32}}, {{5,256,256,2}}) == 5);
  assert(conv_weight_patch_batch({{16,256,256,32}}, {{32,3,3,32}}, {{16,256,256,32}}) == 16);
  assert(conv_weight_patch_batch({{16,1024,363,32}}, {{32,9,1,32}}, {{16,1024,363,32}}) == 1);
  assert(conv_weight_patch_batch({{16,8,8,2}}, {{2,3,3,2}}, {{16,8,8,2}}) == 16);
  assert(conv_weight_patch_batch({{33,1024,1024,16}}, {{2,1,1,16}}, {{33,1024,1024,2}}) == 4);
  assert(conv_weight_patch_batch({{32,1024,1024,16}}, {{2,1,1,16}}, {{32,1024,1024,2}}) == 32);
  assert(conv_weight_patch_batch({{64,1048576,16}}, {{2,3,16}}, {{64,1048576,2}}) == 1);
  assert(conv_weight_patch_batch({{16,64,64,64,32}}, {{2,3,3,3,32}}, {{16,64,64,64,2}}) == 1);
  assert(conv_weight_patch_batch({{31,256,256,32},float16}, {{2,3,3,32}}, {{31,256,256,2}}) == 31);
  assert(conv_weight_patch_batch({{1,1024,1024,1024}}, {{2,3,3,1024}}, {{1,1024,1024,2}}) == 1);
  assert(conv_weight_patch_batch({{0,256,256,32}}, {{2,3,3,32}}, {{0,256,256,2}}) == 0);
  assert(conv_weight_patch_batch({{5,0,256,32}}, {{2,3,3,32}}, {{5,0,256,2}}) == 5);
  assert(conv_weight_patch_batch({{5,256,256,0}}, {{2,3,3,0}}, {{5,256,256,2}}) == 5);
  assert(conv_weight_patch_batch({{INT_MAX,INT_MAX,INT_MAX,INT_MAX}},
      {{INT_MAX,INT_MAX,INT_MAX,INT_MAX}}, {{INT_MAX,INT_MAX,INT_MAX,INT_MAX}}) == 1);
}
'''
    with tempfile.TemporaryDirectory(prefix="mixlab-conv-tiles-") as tmp:
        cpp, binary = Path(tmp)/"test.cpp", Path(tmp)/"test"
        cpp.write_text(header + body + tests)
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-O2",
                        str(cpp), "-o", str(binary)], check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=10)
    print("convolution patch tile policy passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-broken", action="store_true")
    args = parser.parse_args()
    run(args.source, args.expect_broken)
