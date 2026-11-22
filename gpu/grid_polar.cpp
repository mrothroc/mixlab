#include "grid_polar.h"
#include <mlx/transforms.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mx = mlx::core;
namespace mlx_ir {
namespace {
constexpr double tau = 6.283185307179586476925286766559;
void check_shape(const mx::Shape& shape) {
  int64_t size = 1;
  for (auto n : shape) {
    if (n <= 0 || size > INT32_MAX / int64_t(n))
      throw std::runtime_error("polar shape exceeds positive int32 bounds");
    size *= n;
  }
}
mx::array clip(const mx::array& x, float hi) {
  return mx::minimum(mx::maximum(x, mx::array(0.f)), mx::array(hi));
}
}

PolarGeometry polar_geometry(const IRop& op) {
  if (op.n_inputs != 2 || op.n_outputs != 1 || op.n_int_params != 6 || op.n_float_params != 1)
    throw std::runtime_error("polar op requires two inputs, one output, six integer params and dr");
  const int* p = op.int_params;
  PolarGeometry g{p[0],p[1],p[2],p[3],p[4],p[5],op.float_params[0]};
  check_shape({g.k,g.j}); check_shape({g.h,g.w});
  if (!std::isfinite(g.dr) || g.dr <= 0 || !std::isfinite((g.j-1)*g.dr) ||
      g.row_column < 0 || g.col_column < 0 || g.row_column == g.col_column)
    throw std::runtime_error("invalid polar dr or tx columns");
  if (op.type == OP_POLAR_TO_CARTESIAN &&
      g.j < std::ceil(std::hypot(double(g.h-1),double(g.w-1))/g.dr)+2)
    throw std::runtime_error("polar radius too short for Cartesian output");
  return g;
}

void validate_polar_inputs(const IRProgram& p, const TensorMap& inputs) {
  for (const auto& op : p.ops) {
    if (op.type != OP_CARTESIAN_TO_POLAR && op.type != OP_POLAR_TO_CARTESIAN) continue;
    auto g = polar_geometry(op);
    auto it = inputs.find(op.inputs[1]);
    if (it == inputs.end()) throw std::runtime_error("polar vectors must be a direct host input");
    const auto& v = it->second;
    if (v.dtype != TensorDesc::FLOAT32 || v.shape.size() != 2 ||
        v.shape[1] <= std::max(g.row_column,g.col_column))
      throw std::runtime_error("invalid polar vector shape/dtype");
    check_shape({v.shape[0],v.shape[1]});
    if (!v.data || v.size_bytes != size_t(v.shape[0])*v.shape[1]*sizeof(float))
      throw std::runtime_error("invalid polar vector payload");
    const float* data = static_cast<const float*>(v.data);
    for (int b=0; b<v.shape[0]; ++b) {
      float r=data[b*v.shape[1]+g.row_column], c=data[b*v.shape[1]+g.col_column];
      if (!std::isfinite(r) || !std::isfinite(c) || r<0 || c<0 || double(r)>g.h-1 || double(c)>g.w-1)
        throw std::runtime_error("invalid polar transmitter at batch row "+std::to_string(b));
    }
  }
}

std::array<mx::array,2> polar_coordinates(const PolarGeometry& g, const mx::array& vectors, bool inverse) {
  auto v = mx::stop_gradient(vectors);
  int b=v.shape(0);
  auto tr=mx::reshape(mx::slice(v,{0,g.row_column},{b,g.row_column+1}),{b,1,1});
  auto tc=mx::reshape(mx::slice(v,{0,g.col_column},{b,g.col_column+1}),{b,1,1});
  if (!inverse) {
    auto angle=mx::reshape(mx::arange(g.k,mx::float32)*float(tau/g.k),{1,g.k,1});
    auto radius=mx::reshape(mx::arange(g.j,mx::float32)*g.dr,{1,1,g.j});
    return {tr+mx::sin(angle)*radius,tc+mx::cos(angle)*radius};
  }
  auto dy=mx::reshape(mx::arange(g.h,mx::float32),{1,g.h,1})-tr;
  auto dx=mx::reshape(mx::arange(g.w,mx::float32),{1,1,g.w})-tc;
  auto angle=mx::arctan2(dy,dx);
  angle=mx::where(mx::logical_and(mx::equal(dy,mx::array(0.f)),mx::equal(dx,mx::array(0.f))),mx::array(0.f),angle);
  // atan2 lies in [-pi,pi]; avoid modulo and CUDA integer fmod JIT paths.
  angle=mx::where(mx::less(angle,mx::array(0.f)),angle+float(tau),angle);
  auto a=angle*float(g.k/tau);
  a=mx::where(mx::greater_equal(a,mx::array(float(g.k))),a-float(g.k),a);
  return {a,mx::sqrt(dy*dy+dx*dx)/g.dr};
}

BilinearPlan polar_bilinear_plan(const mx::array& row, const mx::array& col, int h, int w, bool wrap) {
  auto r=wrap ? row : clip(row,float(h-1));
  auto c=wrap ? col : clip(col,float(w-1));
  auto rf=mx::floor(r), cf=mx::floor(c);
  auto fr=r-rf, fc=c-cf;
  // A valid last index can round up to 2^31 in FP32. Bound in uint32 before
  // converting to signed gather indices, avoiding an overflowing int32 cast.
  auto r0=mx::astype(mx::minimum(mx::astype(rf,mx::uint32),mx::array(uint32_t(h-1))),mx::int32);
  auto c0=mx::astype(mx::minimum(mx::astype(cf,mx::uint32),mx::array(uint32_t(w-1))),mx::int32);
  auto r1=r0+1, c1=c0+1;
  if (wrap) {
    r1=mx::where(mx::greater_equal(r1,mx::array(h,mx::int32)),mx::array(0,mx::int32),r1);
    c1=mx::minimum(c1,mx::array(w-1,mx::int32));
  } else {
    r1=mx::minimum(r1,mx::array(h-1,mx::int32));
    c1=mx::minimum(c1,mx::array(w-1,mx::int32));
  }
  return {{r0*w+c0,r0*w+c1,r1*w+c0,r1*w+c1},
          {(1-fr)*(1-fc),(1-fr)*fc,fr*(1-fc),fr*fc}};
}

mx::array polar_sample(const mx::array& input, const BilinearPlan& plan) {
  const int b=input.shape(0),h=input.shape(1),w=input.shape(2),c=input.shape(3);
  auto flat=mx::reshape(input,{b*h*w,c});
  auto offsets=mx::reshape(mx::arange(b,mx::int32)*(h*w),{b,1,1});
  auto corner=[&](int n) {
    return mx::take(flat,mx::stop_gradient(plan.indices[n]+offsets),0)*mx::expand_dims(mx::stop_gradient(plan.weights[n]),-1);
  };
  return corner(0)+corner(1)+corner(2)+corner(3);
}

mx::array polar_resample(const IRop& op, const mx::array& input, const mx::array& vectors) {
  auto g=polar_geometry(op);
  const bool inverse=op.type==OP_POLAR_TO_CARTESIAN;
  if (input.ndim()!=4 || vectors.ndim()!=2 || input.shape(0)!=vectors.shape(0) ||
      vectors.shape(1)<=std::max(g.row_column,g.col_column) || input.dtype()!=mx::float32 || vectors.dtype()!=mx::float32)
    throw std::runtime_error("polar requires float32 NHWC input and [B,S] vectors");
  if (input.shape(1)!=(inverse?g.k:g.h) || input.shape(2)!=(inverse?g.j:g.w))
    throw std::runtime_error("polar input does not match resolved geometry");
  check_shape(input.shape());
  check_shape({input.shape(0),inverse?g.h:g.k,inverse?g.w:g.j,input.shape(3)});
  auto coords=polar_coordinates(g,vectors,inverse);
  auto plan=polar_bilinear_plan(coords[0],coords[1],input.shape(1),input.shape(2),inverse);
  auto out=polar_sample(inverse?input:mx::stop_gradient(input),plan);
  return inverse?out:mx::stop_gradient(out);
}
}
