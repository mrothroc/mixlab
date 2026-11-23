// Opt-in native back-map timing; no Python implementation or coordinate cache.
#include "grid_polar.h"
#include <mlx/device.h>
#include <mlx/memory.h>
#include <mlx/transforms.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected 512, 1024, or --smoke");
    const std::string argument = argv[1];
    const bool smoke = argument == "--smoke";
    if (!smoke && argument != "512" && argument != "1024")
      throw std::runtime_error("benchmark rays must be 512 or 1024");
    const int k = smoke ? 16 : std::stoi(argument), b = smoke ? 1 : 16;
    const int h = smoke ? 8 : 256, w = h, c = smoke ? 2 : 32;
    const int j = int(std::ceil(std::hypot(h-1, w-1))) + 2;
    const int warmup = smoke ? 1 : 5, measured = smoke ? 2 : 20;
    mx::set_default_device(mx::Device::gpu);
    mx::set_memory_limit(8ULL << 30);
    mx::set_cache_limit(256ULL << 20);
    mlx_ir::IRop op;
    op.type = mlx_ir::OP_POLAR_TO_CARTESIAN;
    op.n_inputs = 2; op.n_outputs = 1; op.n_int_params = 6; op.n_float_params = 1;
    const int params[] = {k, j, h, w, 0, 1};
    std::copy(params, params + 6, op.int_params);
    op.float_params[0] = 1;
    auto base = mx::ones({b, k, j, c});
    auto cotangent = mx::ones({b, h, w, c});
    mx::eval({base, cotangent});
    for (bool backward : {false, true}) {
      mx::clear_cache();
      mx::reset_peak_memory();
      std::vector<double> samples;
      for (int step = 0; step < warmup + measured; ++step) {
        std::vector<float> tx(2*b);
        for (int n = 0; n < b; ++n) {
          tx[2*n] = float((n*17+step)%97)/97 * (h-1);
          tx[2*n+1] = float((n*31+step)%89)/89 * (w-1);
        }
        auto v = mx::array(tx.data(), {b, 2});
        auto p = base + float(step+1)*1e-5f;
        auto u = cotangent + float(step+1)*1e-5f;
        mx::eval({p, v, u});
        const auto start = Clock::now();
        auto result = backward
            ? mx::vjp([&](const mx::array& x) { return mlx_ir::polar_resample(op, x, v); }, p, u).second
            : mlx_ir::polar_resample(op, p, v);
        result.eval();
        const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
        if (!mx::all(mx::isfinite(result)).item<bool>())
          throw std::runtime_error("nonfinite polar timing output");
        if (step >= warmup) samples.push_back(seconds);
      }
      auto sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      const double median = (sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])/2;
      std::cout << std::setprecision(10)
                << "{\"format\":\"mixlab.polar_benchmark.v1\",\"status\":\"pass\",\"phase\":\""
                << (backward ? "gradient_only_vjp" : "forward")
                << "\",\"batch\":" << b << ",\"height\":" << h << ",\"width\":" << w
                << ",\"rays\":" << k << ",\"radius\":" << j << ",\"channels\":" << c
                << ",\"warmup\":" << warmup << ",\"measured_steps\":" << measured
                << ",\"coordinate_cache\":false,\"median_seconds\":" << median
                << ",\"peak_bytes\":" << mx::get_peak_memory()
                << ",\"active_bytes\":" << mx::get_active_memory()
                << ",\"cache_bytes\":" << mx::get_cache_memory()
                << ",\"memory_limit_bytes\":" << mx::get_memory_limit()
                << ",\"samples_seconds\":[";
      for (size_t n = 0; n < samples.size(); ++n) std::cout << (n ? "," : "") << samples[n];
      std::cout << "]}" << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "polar benchmark failed: " << error.what() << std::endl;
    return 1;
  }
}
