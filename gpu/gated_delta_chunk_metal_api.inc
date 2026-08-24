bool gated_delta_chunk_metal_primitive_available(
    int d_k,
    int d_v,
    int chunk_size) {
  if (env_is_one("MIXLAB_DISABLE_GATED_DELTA_CHUNK_METAL")) {
    return false;
  }
  if (d_k <= 0 || d_k > kMaxDK || d_v <= 0 || d_v > kMaxDV ||
      chunk_size <= 0 || chunk_size > kMaxChunk) {
    return false;
  }
#ifdef __APPLE__
  return mx::metal::is_available();
#else
  return false;
#endif
}

mx::array gated_delta_chunk_metal_forward(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& beta,
    const mx::array& gate,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
#ifdef __APPLE__
  validate_shape(B, T, H, Dk, Dv, chunk_size);
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true)) {
    std::cerr
        << "[mlx_ir] GatedDeltaNet scan using chunk-parallel Metal forward/backward"
        << " (chunk=" << chunk_size
        << " backward_window=" << backward_window(Dk, chunk_size)
        << "; set MIXLAB_DISABLE_GATED_DELTA_CHUNK_METAL=1"
        << " to use the recurrent Metal fallback)"
        << std::endl;
  }
  const int chunks = (T + chunk_size - 1) / chunk_size;
  auto primitive = std::make_shared<ChunkForwardPrimitive>(
      mx::default_stream(mx::default_device()),
      B, T, H, Dk, Dv, chunk_size);
  auto outputs = mx::array::make_arrays(
      {
          mx::Shape{B * T * H, Dv},
          mx::Shape{B * H * chunks * Dk, Dk},
          mx::Shape{B * H * chunks * Dk, Dv},
          mx::Shape{B * H * chunks * Dk, Dv}},
      std::vector<mx::Dtype>(4, mx::float32),
      primitive,
      contiguous_inputs(q, k, v, beta, gate));
  return outputs[0];
#else
  (void)q;
  (void)k;
  (void)v;
  (void)beta;
  (void)gate;
  (void)B;
  (void)T;
  (void)H;
  (void)Dk;
  (void)Dv;
  (void)chunk_size;
  throw std::runtime_error("chunk-parallel Gated DeltaNet requires Metal");
#endif
}

std::vector<mx::array> gated_delta_chunk_metal_vjp(
    const std::vector<mx::array>& args,
    const std::vector<mx::array>& cotangents,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int chunk_size) {
#ifdef __APPLE__
  if (args.size() != 5 || cotangents.size() != 1) {
    throw std::runtime_error(
        "gated_delta_chunk_metal_vjp expects 5 args and 1 cotangent");
  }
  validate_shape(B, T, H, Dk, Dv, chunk_size);
  auto inputs = contiguous_inputs(args[0], args[1], args[2], args[3], args[4]);
  inputs.push_back(mx::contiguous(mx::astype(cotangents[0], mx::float32)));
  const int chunks = (T + chunk_size - 1) / chunk_size;
  const int window = backward_window(Dk, chunk_size);
  const int micros = (chunk_size + window - 1) / window;
  const mx::Shape chunk_state{B * H * chunks * Dk, Dv};
  auto primitive = std::make_shared<ChunkBackwardPrimitive>(
      mx::default_stream(mx::default_device()),
      B, T, H, Dk, Dv, chunk_size);
  auto outputs = mx::array::make_arrays(
      {
          mx::Shape{B * H * chunks * Dk, Dk},
          chunk_state,
          chunk_state,
          chunk_state,
          chunk_state,
          mx::Shape{B * H * chunks * micros * Dk, Dv},
          mx::Shape{B, T, H, Dk},
          mx::Shape{B, T, H, Dk},
          mx::Shape{B, T, H, Dv},
          mx::Shape{B, T, H},
          mx::Shape{B, T, H}},
      std::vector<mx::Dtype>(11, mx::float32),
      primitive,
      inputs);
  return {outputs[6], outputs[7], outputs[8], outputs[9], outputs[10]};
#else
  (void)args;
  (void)cotangents;
  (void)B;
  (void)T;
  (void)H;
  (void)Dk;
  (void)Dv;
  (void)chunk_size;
  throw std::runtime_error("chunk-parallel Gated DeltaNet requires Metal");
#endif
}
