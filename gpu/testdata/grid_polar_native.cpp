// Independent double-precision geometry and scatter oracle for native helpers.
// Compiled/run by scripts/test_grid_polar_native.sh; no requestor data required.
#include "grid_polar.h"
#include <mlx/device.h>
#include <mlx/memory.h>
#include <mlx/transforms.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using namespace mlx_ir;
constexpr double tau=6.283185307179586476925286766559;
struct OraclePlan {
  std::array<std::vector<int>,4> index;
  std::array<std::vector<float>,4> rounded;
  std::array<std::vector<double>,4> weight;
};
OraclePlan oracle_plan(const std::vector<double>& row,const std::vector<double>& col,int h,int w,bool wrap) {
  OraclePlan p;
  for(size_t i=0;i<row.size();++i) {
    double r=wrap?row[i]:std::clamp(row[i],0.,double(h-1));
    double c=wrap?col[i]:std::clamp(col[i],0.,double(w-1));
    int r0=std::floor(r),c0=std::floor(c),r1=wrap?(r0+1)%h:std::min(r0+1,h-1),c1=wrap?c0+1:std::min(c0+1,w-1);
    double fr=r-r0,fc=c-c0;
    int idx[]={r0*w+c0,r0*w+c1,r1*w+c0,r1*w+c1};
    double weights[]={(1-fr)*(1-fc),(1-fr)*fc,fr*(1-fc),fr*fc};
    for(int n=0;n<4;++n) {p.index[n].push_back(idx[n]);p.weight[n].push_back(weights[n]);p.rounded[n].push_back(weights[n]);}
  }
  return p;
}
std::vector<double> sample(const std::vector<float>& input,const OraclePlan& p,int batch,int source_size) {
  std::vector<double> y(p.index[0].size(),0);
  int per=y.size()/batch;
  for(size_t i=0;i<y.size();++i) for(int n=0;n<4;++n) y[i]+=input[(i/per)*source_size+p.index[n][i]]*p.weight[n][i];
  return y;
}
std::vector<double> adjoint(const std::vector<float>& cot,const OraclePlan& p,int batch,int source_size,bool rounded) {
  std::vector<double> g(batch*source_size,0);
  int per=cot.size()/batch;
  for(size_t i=0;i<cot.size();++i) for(int n=0;n<4;++n)
    g[(i/per)*source_size+p.index[n][i]]+=cot[i]*(rounded?double(p.rounded[n][i]):p.weight[n][i]);
  return g;
}
double error(mx::array x,const std::vector<double>& ref,bool circular=false,int k=0) {
  x.eval(); if(x.size()!=ref.size()) throw std::runtime_error("oracle size mismatch");
  auto p=x.data<float>(); double e=0;
  for(size_t i=0;i<ref.size();++i) {
    if(!std::isfinite(p[i])) throw std::runtime_error("nonfinite result");
    double d=std::abs(double(p[i])-ref[i]);
    if(circular) {d=std::fmod(d,k);d=std::min(d,k-d);}
    e=std::max(e,d);
  }
  return e;
}
void require(double value,double limit,const char* name) {
  if(!std::isfinite(value)||value>limit) throw std::runtime_error(std::string(name)+" error="+std::to_string(value));
}
void run_case(int h,int w,int k,float dr,const std::vector<float>& tx) {
  int b=tx.size()/2,j=std::ceil(std::hypot(double(h-1),double(w-1))/dr)+2;
  PolarGeometry g{k,j,h,w,0,1,dr};
  std::vector<double> rows,cols,angles,radii;
  std::vector<float> field(b*h*w),polar(b*k*j),cot(b*h*w),noise(b*k*j);
  std::mt19937 rng(412);std::normal_distribution<float> normal;
  for(int n=0;n<b;++n) {
    for(int a=0;a<k;++a) for(int r=0;r<j;++r) {
      rows.push_back(tx[2*n]+r*double(dr)*std::sin(tau*a/k));
      cols.push_back(tx[2*n+1]+r*double(dr)*std::cos(tau*a/k));
      polar[(n*k+a)*j+r]=1+.2*std::cos(tau*a/k)+.2*std::sin(r*.01);
      noise[(n*k+a)*j+r]=normal(rng);
    }
    for(int r=0;r<h;++r) for(int c=0;c<w;++c) {
      double dy=r-double(tx[2*n]),dx=c-double(tx[2*n+1]);
      double a=(dy==0&&dx==0)?0:std::atan2(dy,dx);
      if(a<0) a+=tau;
      angles.push_back(a*k/tau);radii.push_back(std::hypot(dy,dx)/dr);
      field[(n*h+r)*w+c]=1+.2*std::sin(r*.01)+.2*std::cos(c*.013);
      cot[(n*h+r)*w+c]=normal(rng);
    }
  }
  auto v=mx::array(tx.data(),{b,2});
  auto xy=polar_coordinates(g,v,false), ar=polar_coordinates(g,v,true);
  double ce=std::max({error(xy[0],rows),error(xy[1],cols),error(ar[0],angles,true,k),error(ar[1],radii)});
  require(ce,1e-3,"coordinates");
  auto cp=oracle_plan(rows,cols,h,w,false),bp=oracle_plan(angles,radii,k,j,true);
  auto x=mx::array(field.data(),{b,h,w,1}),p=mx::array(polar.data(),{b,k,j,1}),u=mx::array(cot.data(),{b,h,w,1});
  auto device_cp=polar_bilinear_plan(xy[0],xy[1],h,w,false),device_bp=polar_bilinear_plan(ar[0],ar[1],k,j,true);
  IRop op;op.type=OP_CARTESIAN_TO_POLAR;op.n_inputs=2;op.n_outputs=1;op.n_int_params=6;op.n_float_params=1;
  int ip[]={k,j,h,w,0,1};std::copy(ip,ip+6,op.int_params);op.float_params[0]=dr;
  auto forward=polar_resample(op,x,v);
  op.type=OP_POLAR_TO_CARTESIAN;
  auto back=polar_resample(op,p,v);
  double fe=std::max(error(forward,sample(field,cp,b,h*w)),error(back,sample(polar,bp,b,k*j)));
  require(fe,2e-5,"smooth forward");
  // The fixed-plan test exercises the exact gather helper used by production.
  auto shared=device_bp;
  for(int n=0;n<4;++n) {
    shared.indices[n]=mx::array(bp.index[n].data(),{b,h,w});
    shared.weights[n]=mx::array(bp.rounded[n].data(),{b,h,w});
  }
  auto grad=mx::vjp([&](const mx::array& z){return polar_sample(z,shared);},p,u).second;
  double ge=error(grad,adjoint(cot,bp,b,k*j,true)); require(ge,1e-5,"shared-plan adjoint");
  auto e2e=mx::vjp([&](const mx::array& z){return polar_sample(z,device_bp);},p,u).second;
  auto white=mx::array(noise.data(),{b,k,j,1});
  double ne=error(polar_sample(white,device_bp),sample(noise,bp,b,k*j));
  double eg=error(e2e,adjoint(cot,bp,b,k*j,false));
  auto constant=polar_sample(polar_sample(mx::ones({b,h,w,1}),device_cp),device_bp);
  require(error(constant,std::vector<double>(b*h*w,1)),1e-6,"constant roundtrip");
  // Coordinate input is detached even when used as an autodiff argument.
  auto vg=mx::vjp([&](const mx::array& vv){auto co=polar_coordinates(g,vv,true);return polar_sample(p,polar_bilinear_plan(co[0],co[1],k,j,true));},v,u).second;
  require(error(vg,std::vector<double>(b*2,0)),0,"vector gradient");
  if(h<=7&&w<=7) {
    auto fg=adjoint(cot,bp,b,k*j,true); auto sy=sample(noise,bp,b,k*j);
    double left=0,right=0;for(size_t i=0;i<sy.size();++i)left+=sy[i]*cot[i];
    // Use double weights for independent CPU dot-product equality.
    fg=adjoint(cot,bp,b,k*j,false);
    for(size_t i=0;i<fg.size();++i)right+=fg[i]*noise[i];
    require(std::abs(left-right),1e-10,"dot product");
    auto delta=white*1e-3f;
    auto fd=mx::sum((polar_sample(p+delta,shared)-polar_sample(p-delta,shared))*u)/.002f;
    auto analytic=mx::sum(grad*white);fd.eval();analytic.eval();
    require(std::abs(fd.item<float>()-analytic.item<float>()),.01,"finite difference");
  }
  std::cout<<"h="<<h<<" w="<<w<<" k="<<k<<" dr="<<dr<<" b="<<b<<" coordinate="<<ce<<" smooth="<<fe<<" shared_adjoint="<<ge<<" noise_observed="<<ne<<" e2e_grad_observed="<<eg<<std::endl;
}
int main() {
  try {
    mx::set_default_device(mx::Device::gpu); mx::set_memory_limit(8ULL<<30);mx::set_cache_limit(128ULL<<20);
    // Exercise index conversion at the int32 shape limit without a huge grid.
    auto large=mx::array(float(INT_MAX));
    for(bool row_axis:{false,true}) {
      auto bounds=polar_bilinear_plan(row_axis?large:mx::array(0.f),row_axis?mx::array(0.f):large,
        row_axis?INT_MAX:1,row_axis?1:INT_MAX,false);
      for(auto index:bounds.indices) {
        index.eval();
        if(index.item<int>()!=INT_MAX-1) throw std::runtime_error("large-axis border index overflow");
      }
    }
    for(auto hw: {std::pair{256,256},std::pair{192,256}}) for(auto kd:{std::pair{512,1.f},std::pair{1024,1.f},std::pair{1024,.5f}}) {
      int h=hw.first,w=hw.second;
      std::vector<float> tx={(h-1)/2.f,(w-1)/2.f,0,0,float(h-1),(w-1)/2.f,17,31,37.25,101.5};
      for(int n=0;n<5;++n)run_case(h,w,kd.first,kd.second,{tx[2*n],tx[2*n+1]});
      tx.resize(8);run_case(h,w,kd.first,kd.second,tx);
    }
    run_case(1,1,1,1,{0,0});run_case(1,7,1,1,{0,2.5});run_case(7,1,1,1,{2.5,0});run_case(4,5,8,1,{1.25,2.5});
    std::cout<<"PASS 40 native geometry/adjoint cases"<<std::endl;
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
