// Deterministic synthetic CUDA convolution acceptance. No dataset/checkpoint.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
struct Case { bool transpose; int h, ci, co, k, stride, pad; };

int main(int argc, char** argv) {
  const int batch = argc == 2 ? std::atoi(argv[1]) : 1;
  if (argc > 2 || (batch != 1 && batch != 16)) {
    std::cerr << "usage: probe [1|16]" << std::endl;
    return 2;
  }
  mx::set_default_device(mx::Device::gpu);
  const bool tf32 = !std::getenv("MLX_ENABLE_TF32") ||
      std::string(std::getenv("MLX_ENABLE_TF32")) != "0";
  bool passed = true;
  for (auto c : std::vector<Case>{
      {false,32,2,6,3,1,1}, {false,32,6,40,5,1,2},
      {false,16,40,50,5,2,2}, {false,16,100,100,3,1,1},
      {true,8,100,60,4,2,1}, {true,16,60,40,6,2,2},
      {false,8,300,500,5,1,2}, {true,4,500,300,4,2,1},
      {false,256,2,6,3,1,1}, {false,256,6,40,5,1,2},
      {false,128,40,50,5,1,2}, {false,64,50,60,5,1,2},
      {false,32,60,100,5,1,2}, {false,16,100,150,5,1,2},
      {false,8,150,300,5,1,2}, {false,4,300,500,5,1,2},
      {true,8,600,150,4,2,1}, {true,16,300,100,4,2,1},
      {false,32,200,100,3,1,1}, {true,32,200,60,6,2,2},
      {false,64,120,50,5,1,2}, {true,64,100,40,6,2,2},
      {true,128,80,20,6,2,2}, {false,256,28,20,5,1,2},
      {false,256,22,1,5,1,2}}) {
    std::vector<float> x(size_t(c.h)*c.h*c.ci), w(size_t(c.co)*c.k*c.k*c.ci);
    // Integer PRNG makes the fixture identical across standard libraries.
    uint32_t state=83;
    auto next = [&] {
      state = state*1664525u + 1013904223u;
      return float((state>>8)*(2.0/16777216.0)-1.0);
    };
    for (auto& v:x) v=next();
    for (auto& v:w) v=next()/std::sqrt(float(c.k*c.k*c.ci));
    int oh=c.transpose ? (c.h-1)*c.stride-2*c.pad+c.k : (c.h+2*c.pad-c.k)/c.stride+1;
    std::vector<double> reference(size_t(oh)*oh*c.co,0);
    // Large production shapes use evenly distributed spatial samples; GPU
    // computation still evaluates the full tensor and selects its real plan.
    std::vector<int> pixels;
    for(int i=0;i<std::min(64,oh*oh);i++) pixels.push_back((i*104729)%(oh*oh));
    pixels.push_back(oh*oh-1);
    pixels.push_back((oh/2)*oh+oh/2);
    std::sort(pixels.begin(),pixels.end());
    pixels.erase(std::unique(pixels.begin(),pixels.end()),pixels.end());
    for(auto pixel:pixels) { int y=pixel/oh,z=pixel%oh;
      for(int ky=0;ky<c.k;ky++) for(int kz=0;kz<c.k;kz++) {
        int iy=y*c.stride-c.pad+ky, iz=z*c.stride-c.pad+kz;
        if(c.transpose) {
          iy=y+c.pad-ky; iz=z+c.pad-kz;
          if(iy%c.stride || iz%c.stride) continue;
          iy/=c.stride; iz/=c.stride;
        }
        if(iy<0 || iy>=c.h || iz<0 || iz>=c.h) continue;
        for(int ci=0;ci<c.ci;ci++) for(int co=0;co<c.co;co++) {
          int wi=c.transpose ? ((ci*c.k+ky)*c.k+kz)*c.co+co : ((co*c.k+ky)*c.k+kz)*c.ci+ci;
          reference[(y*oh+z)*c.co+co]+=double(x[(iy*c.h+iz)*c.ci+ci])*w[wi];
        }
      }
    }
    // Equal records let the independent oracle check both ends of a larger
    // batch while cuDNN selects a plan for the actual batch shape.
    std::vector<float> batched(size_t(batch)*x.size());
    for(int row=0;row<batch;row++) std::copy(x.begin(),x.end(),batched.begin()+row*x.size());
    auto input=mx::array(batched.data(),{batch,c.h,c.h,c.ci});
    auto weight=mx::array(w.data(), c.transpose ? mx::Shape{c.ci,c.k,c.k,c.co} : mx::Shape{c.co,c.k,c.k,c.ci});
    if(c.transpose) weight=mx::transpose(weight,{3,1,2,0});
    for(int repeat=0;repeat<2;repeat++) {
      auto output=c.transpose ? mx::conv_transpose2d(input,weight,{c.stride,c.stride},{c.pad,c.pad}) : mx::conv2d(input,weight,{c.stride,c.stride},{c.pad,c.pad});
      mx::eval(output);
      auto values=output.data<float>();
      double error=0,scale=0;
      for(int row : std::vector<int>{0,batch-1}) for(auto pixel:pixels) for(int co=0;co<c.co;co++) {
        size_t i=size_t(pixel)*c.co+co;
        size_t offset=size_t(row)*oh*oh*c.co+i;
        if(!std::isfinite(values[offset])) { passed=false; error=INFINITY; }
        error=std::max(error,std::abs(values[offset]-reference[i]));
        scale=std::max(scale,std::abs(reference[i]));
      }
      double relative=error/std::max(scale,1e-20);
      passed &= relative <= (tf32 ? .02 : 2e-5);
      std::cout << "tf32=" << tf32 << " batch=" << batch << " transpose=" << c.transpose
          << " h=" << c.h << " c=" << c.ci << ":" << c.co << " kernel=" << c.k
          << " repeat=" << repeat << " abs=" << error << " relative=" << relative << std::endl;
    }
  }
  return passed ? 0 : 1;
}
