// Large synthetic convolutions exercise batch tiling and a non-divisible tail.
// Spatially constant inputs/cotangents permit an independent closed-form VJP.
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
namespace mx = mlx::core;
namespace mlx::core {
int conv_weight_patch_batch(const array&, const array&, const array&);
}

int main() {
  mx::set_default_device(mx::Device::gpu);
  mx::set_memory_limit(4ULL << 30);
  mx::set_cache_limit(256ULL << 20);
  constexpr int n=31,h=256,w=256,k=3;
  bool pass=true;
  for (bool trans : {false,true}) {
    int ci=trans?2:32, co=trans?32:2;
    std::vector<float> x(size_t(n)*h*w*ci), wt(size_t(co)*k*k*ci), dy(size_t(n)*h*w*co);
    auto input_scale=[](int b,int c) { return float(b+1)*float(c%3-1)*.125f; };
    auto output_scale=[](int c) { return float(c%3-1)*.25f; };
    for(int b=0;b<n;++b) for(int p=0;p<h*w;++p) {
      for(int c=0;c<ci;++c) x[(size_t(b)*h*w+p)*ci+c]=input_scale(b,c);
      for(int c=0;c<co;++c) dy[(size_t(b)*h*w+p)*co+c]=output_scale(c);
    }
    for(size_t i=0;i<wt.size();++i) wt[i]=float(int(i*7%17)-8)/1024;
    auto input=mx::array(x.data(),{n,h,w,ci});
    auto weight=mx::array(wt.data(),{co,k,k,ci});
    auto upstream=mx::array(dy.data(),{n,h,w,co});
    const int tile = trans
        ? mx::conv_weight_patch_batch(upstream, mx::swapaxes(weight,0,-1), input)
        : mx::conv_weight_patch_batch(input, weight, upstream);
    if (tile != 3) return 2; // Proves the patched native policy is linked/live.
    auto result=mx::vjp([&](const std::vector<mx::array>& a) {
      return std::vector<mx::array>{trans ? mx::conv_transpose2d(a[0],a[1],{1,1},{1,1})
                                             : mx::conv2d(a[0],a[1],{1,1},{1,1})};
    }, {input,weight}, {upstream});
    result.first[0]=mx::contiguous(result.first[0]);
    result.second[0]=mx::contiguous(result.second[0]);
    result.second[1]=mx::contiguous(result.second[1]);
    mx::eval({result.first[0],result.second[0],result.second[1]});
    double errors[3]={}, scales[3]={};
    auto check=[&](int which,float actual,double expected) {
      if(!std::isfinite(actual)) { pass=false; errors[which]=INFINITY; }
      const double error=std::abs(actual-expected);
      if(error > 1e-6 && error > 2e-5*std::abs(expected)) pass=false;
      errors[which]=std::max(errors[which],std::abs(actual-expected));
      scales[which]=std::max(scales[which],std::abs(expected));
    };
    const float* y=result.first[0].data<float>();
    const float* dx=result.second[0].data<float>();
    const float* dw=result.second[1].data<float>();
    // Check all weight gradients, and all spatial boundary classes for outputs
    // and input gradients, including both ends of every batch record.
    for(int o=0;o<co;++o) for(int ky=0;ky<k;++ky) for(int kx=0;kx<k;++kx) for(int c=0;c<ci;++c) {
      double sum=0;
      for(int b=0;b<n;++b) sum+=double(input_scale(b,c))*output_scale(o);
      size_t i=((o*k+ky)*k+kx)*ci+c;
      check(2,dw[i],sum*(h-std::abs(ky-1))*(w-std::abs(kx-1)));
    }
    for(int b=0;b<n;++b) for(int r : {0,1,127,254,255}) for(int col : {0,1,127,254,255}) {
      for(int o=0;o<co;++o) {
        double sum=0;
        for(int ky=0;ky<k;++ky) for(int kx=0;kx<k;++kx) {
          int ir=trans?r+1-ky:r-1+ky, ic=trans?col+1-kx:col-1+kx;
          if(ir<0||ir>=h||ic<0||ic>=w) continue;
          for(int c=0;c<ci;++c) sum+=double(input_scale(b,c))*wt[((o*k+ky)*k+kx)*ci+c];
        }
        check(0,y[((size_t(b)*h+r)*w+col)*co+o],sum);
      }
      for(int c=0;c<ci;++c) {
        double sum=0;
        for(int ky=0;ky<k;++ky) for(int kx=0;kx<k;++kx) {
          int orow=trans?r-1+ky:r+1-ky, ocol=trans?col-1+kx:col+1-kx;
          if(orow<0||orow>=h||ocol<0||ocol>=w) continue;
          for(int o=0;o<co;++o) sum+=double(output_scale(o))*wt[((o*k+ky)*k+kx)*ci+c];
        }
        check(1,dx[((size_t(b)*h+r)*w+col)*ci+c],sum);
      }
    }
    for(int i=0;i<3;++i) {
      const double relative=errors[i]/std::max(scales[i],1e-20);
      pass &= errors[i] <= 1e-6 || relative <= 2e-5;
      std::cout<<"transpose="<<trans<<" component="<<i<<" abs="<<errors[i]<<" relative="<<relative<<std::endl;
    }
  }
  return pass?0:1;
}
