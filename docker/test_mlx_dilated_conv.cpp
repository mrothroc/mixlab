// Real forward and adjoint acceptance with independent FP64 scalar oracles.
// Samples span the complete tensors; MLX executes each full production shape.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mlx/mlx.h"
#include "mlx/memory.h"

namespace mx = mlx::core;

std::vector<size_t> samples(size_t n) {
  std::vector<size_t> result{0,n-1,n/2};
  for (size_t i=0;i<96;i++) result.push_back((i*104729+17)%n);
  return result;
}

void check(const char* name, const std::vector<double>& reference,
           const std::vector<double>& actual, double tolerance) {
  double error=0, scale=0;
  for (size_t i=0;i<actual.size();i++) {
    if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite result");
    error=std::max(error,std::abs(actual[i]-reference[i]));
    scale=std::max(scale,std::abs(reference[i]));
  }
  std::cout << " " << name << "_max_abs=" << error
            << " normalized=" << error/std::max(scale,1e-20) << std::flush;
  if (error > tolerance*std::max(scale,1e-20)) throw std::runtime_error(name);
}

int main(int argc, char** argv) {
  const bool forward_only = argc > 1 && std::string(argv[1]) == "--forward-only";
  const bool tf32 = !std::getenv("MLX_ENABLE_TF32") || std::string(std::getenv("MLX_ENABLE_TF32")) != "0";
  mx::set_default_device(mx::Device::gpu);
  try {
    for (int h : {32,64}) for (int dilation : {2,4}) {
      const int b=2, ci=192, co=192, k=3, pad=dilation;
      std::vector<float> x(size_t(b)*h*h*ci), w(size_t(co)*k*k*ci), cot(size_t(b)*h*h*co);
      uint32_t state=83;
      auto next=[&] { state=state*1664525u+1013904223u; return float((state>>8)*(2.0/16777216.0)-1.0); };
      for(auto& v:x) v=next();
      for(auto& v:w) v=next()/std::sqrt(float(k*k*ci));
      for(auto& v:cot) v=next()/std::sqrt(float(b*h*h));
      auto ix=[&](int n,int y,int z,int c) {return ((size_t(n)*h+y)*h+z)*ci+c;};
      auto iw=[&](int o,int y,int z,int c) {return ((size_t(o)*k+y)*k+z)*ci+c;};
      auto io=[&](int n,int y,int z,int o) {return ((size_t(n)*h+y)*h+z)*co+o;};
      auto input=mx::array(x.data(),{b,h,h,ci});
      auto weight=mx::array(w.data(),{co,k,k,ci});
      auto cotangent=mx::array(cot.data(),{b,h,h,co});
      auto forward=[&](const std::vector<mx::array>& a) {
        return std::vector<mx::array>{mx::conv2d(a[0],a[1],{1,1},{pad,pad},{dilation,dilation})};
      };
      for(int repeat=0;repeat<2;repeat++) {
        std::cout << "h=" << h << " dilation=" << dilation << " tf32=" << tf32 << " repeat=" << repeat << std::flush;
        // Forward is deliberately executed before constructing any gradient.
        auto y=forward({input,weight})[0]; mx::eval(y);
        std::vector<double> ref, got;
        for(auto index:samples(cot.size())) {
          int o=index%co, z=(index/co)%h, yy=(index/co/h)%h, n=index/co/h/h;
          double sum=0;
          for(int ky=0;ky<k;ky++) for(int kz=0;kz<k;kz++) {
            int iy=yy-pad+ky*dilation, iz=z-pad+kz*dilation;
            if(iy<0 || iy>=h || iz<0 || iz>=h) continue;
            for(int c=0;c<ci;c++) sum+=double(x[ix(n,iy,iz,c)])*w[iw(o,ky,kz,c)];
          }
          ref.push_back(sum); got.push_back(y.data<float>()[index]);
        }
        check("forward",ref,got,tf32?.02:2e-5);
        if(!forward_only) {
          auto gradients=mx::vjp(forward,{input,weight},{cotangent}).second;
          for (auto& gradient : gradients) gradient=mx::contiguous(gradient);
          mx::eval(gradients);
          ref.clear(); got.clear();
          for(auto index:samples(x.size())) {
            int c=index%ci,z=(index/ci)%h,yy=(index/ci/h)%h,n=index/ci/h/h;
            double sum=0;
            for(int ky=0;ky<k;ky++) for(int kz=0;kz<k;kz++) {
              int oy=yy+pad-ky*dilation,oz=z+pad-kz*dilation;
              if(oy<0 || oy>=h || oz<0 || oz>=h) continue;
              for(int o=0;o<co;o++) sum+=double(cot[io(n,oy,oz,o)])*w[iw(o,ky,kz,c)];
            }
            ref.push_back(sum); got.push_back(gradients[0].data<float>()[index]);
          }
          check("dx",ref,got,tf32?.02:2e-5);
          ref.clear(); got.clear();
          for(auto index:samples(w.size())) {
            int c=index%ci,kz=(index/ci)%k,ky=(index/ci/k)%k,o=index/ci/k/k;
            double sum=0;
            for(int n=0;n<b;n++) for(int yy=0;yy<h;yy++) for(int z=0;z<h;z++) {
              int iy=yy-pad+ky*dilation,iz=z-pad+kz*dilation;
              if(iy>=0 && iy<h && iz>=0 && iz<h) sum+=double(x[ix(n,iy,iz,c)])*cot[io(n,yy,z,o)];
            }
            ref.push_back(sum); got.push_back(gradients[1].data<float>()[index]);
          }
          check("dw",ref,got,tf32?.02:2e-5);
        }
        std::cout << " peak_MiB=" << mx::get_peak_memory()/double(1<<20) << " PASS" << std::endl;
      }
    }
  } catch(const std::exception& e) {
    std::cerr << "\nFAIL: " << e.what() << std::endl;
    return 1;
  }
}
