"""Exercise the pinned cuDNN plan-decision body with CPU-only status stubs.

Numerical/fallback execution acceptance is in test_mlx_dilated_conv.cpp.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expect-broken', action='store_true')
    args = parser.parse_args()
    source = args.source.read_text()
    if 'auto status = graph.build();' in source:
        start = source.index('auto status = graph.build();')
    else:
        start = source.index('CHECK_CUDNN_ERROR(graph.build());')
    body = source[start:source.index('return graph;', start)]
    header = r'''
#include <optional>
#include <stdexcept>
namespace fe {
enum class error_code_t { OK, GRAPH_NOT_SUPPORTED,
 GRAPH_EXECUTION_PLAN_CREATION_FAILED, GRAPH_EXECUTION_FAILED, CUDA_API_FAILED };
}
using Code = fe::error_code_t;
struct Status { Code code; Code get_code() const { return code; } };
void check(Status s) { if (s.code != Code::OK) throw std::runtime_error("fatal"); }
#define CHECK_CUDNN_ERROR(s) check(s)
namespace env { bool tf32; bool enable_tf32() { return tf32; } }
enum Dtype { float32, float16 };
struct Graph { Status status; Status build() { return status; } };
std::optional<int> decide(Dtype dtype, Graph graph) {
'''
    main_source = r'''
 return 1;
}
int main(int argc, char**) {
 bool broken = argc > 1;
 for (auto dtype : {float32,float16}) for (bool tf32 : {false,true}) {
  env::tf32 = tf32;
  for (auto code : {Code::OK, Code::GRAPH_NOT_SUPPORTED,
       Code::GRAPH_EXECUTION_PLAN_CREATION_FAILED, Code::GRAPH_EXECUTION_FAILED,
       Code::CUDA_API_FAILED}) {
   bool fallback = !broken && dtype == float32 && !tf32 &&
      (code == Code::GRAPH_NOT_SUPPORTED || code == Code::GRAPH_EXECUTION_PLAN_CREATION_FAILED);
   try {
    auto result = decide(dtype,Graph{Status{code}});
    if (code != Code::OK && !fallback) return 1;
    if (result.has_value() == fallback) return 2;
   } catch (...) { if (code == Code::OK || fallback) return 3; }
  }
 }
}
'''
    with tempfile.TemporaryDirectory() as temp:
        cpp, exe = Path(temp)/'test.cpp', Path(temp)/'test'
        cpp.write_text(header + body + main_source)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', str(cpp), '-o', str(exe)], check=True)
        subprocess.run([str(exe)] + (['broken'] if args.expect_broken else []), check=True)
        if not args.expect_broken:
            start = source.index('// Under strict FP32, do not reinterpret')
            retry = source[start:source.index('\n  }\n\n  if (graph)', start)]
            cpp.write_text(r'''
enum Dtype { float32, float16 };
namespace env { bool tf32; bool enable_tf32() { return tf32; } }
int attempts(Dtype dtype) {
  for (int attempt=0; attempt<2; attempt++) {
    if (attempt==1) return 2; // Alternate interpretation claims support.
''' + retry + r'''
  }
  return 1;
}
int main() {
  env::tf32=false;
  if(attempts(float32)!=1 || attempts(float16)!=2) return 1;
  env::tf32=true;
  if(attempts(float32)!=2 || attempts(float16)!=2) return 2;
}
''')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
            start = source.index('void log_strict_conv_fallback(')
            logger = source[start:source.index('// Transpose from', start)]
            cpp.write_text(r'''
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <vector>
enum Dtype { float32, float16 };
namespace env { bool tf32=false; bool enable_tf32() { return tf32; } }
struct array {
  int n; Dtype type=float32;
  Dtype dtype() const { return type; }
  std::vector<int> shape() const { return {2,n,n,192}; }
};
''' + logger + r'''
int main() {
 auto log=[](int n, Dtype dtype=float32) {
   log_strict_conv_fallback(array{n},array{3},array{n,dtype},{1,1},{2,2},{2,2},{1,1},false);
 };
 env::tf32=true; log(9000); env::tf32=false; log(9001,float16);
 log(32); log(32);
 for(int n=100;n<400;n++) log(n);
}
''')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', str(cpp), '-o', str(exe)], check=True)
            output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stderr
            lines = output.splitlines()
            assert len(lines) == 257 and lines[-1].endswith('suppressed'), output
            assert output.count('input=[2,32,32,192]') == 1, output
            assert '9000' not in output and '9001' not in output, output
            for field in ('operation=Convolution', 'precision=strict-fp32', 'reason=', 'weight=', 'output=', 'dilation=', 'input_dilation=', 'stride=', 'padding='):
                assert field in lines[0], field
    print('convolution plan fallback policy passed')


if __name__ == '__main__':
    main()
