#include "group_runtime.h"

#include "mlx_bridge.h"
#include "mlx_bridge_internal.h"

#include <mlx/distributed/ops.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_ir {

GroupRuntime::GroupRuntime(mx::distributed::Group group, std::string backend)
    : group_(std::move(group)),
      backend_(std::move(backend)),
      rank_(group_.rank()),
      world_size_(group_.size()) {}

const mx::distributed::Group& GroupRuntime::group() const {
  return group_;
}

const std::string& GroupRuntime::backend() const {
  return backend_;
}

int GroupRuntime::rank() const {
  return rank_;
}

int GroupRuntime::world_size() const {
  return world_size_;
}

GroupRuntime* group_runtime_from_handle(int64_t handle) {
  if (handle == 0) {
    return nullptr;
  }
  return reinterpret_cast<GroupRuntime*>(static_cast<intptr_t>(handle));
}

} // namespace mlx_ir

namespace {

constexpr int kDigestWords = 8;
constexpr int kIdentityWords = 2 + kDigestWords * 3;

uint32_t backend_digest(const std::string& backend) {
  uint32_t hash = 2166136261u;
  for (unsigned char value : backend) {
    hash ^= value;
    hash *= 16777619u;
  }
  return hash;
}

int validate_identity_exchange(
    const mlx_ir::GroupRuntime& runtime,
    uint64_t generation,
    const uint32_t* membership_digest,
    const uint32_t* expected_member_digests,
    int n_members,
    const uint32_t* local_member_digest) {
  if (!membership_digest || !expected_member_digests || !local_member_digest ||
      n_members <= 0) {
    return -1;
  }
  if (runtime.world_size() != n_members) {
    return -2;
  }

  std::array<int32_t, kIdentityWords> local{};
  local[0] = static_cast<int32_t>(static_cast<uint32_t>(runtime.rank()));
  local[1] = static_cast<int32_t>(backend_digest(runtime.backend()));
  for (int i = 0; i < kDigestWords; ++i) {
    local[2 + i] = static_cast<int32_t>(membership_digest[i]);
    local[2 + kDigestWords + i] = static_cast<int32_t>(
        i < 2 ? static_cast<uint32_t>(generation >> (i * 32)) : 0u);
    local[2 + kDigestWords * 2 + i] =
        static_cast<int32_t>(local_member_digest[i]);
  }

  auto local_array = mx::array(
      local.data(),
      {kIdentityWords},
      mx::int32);
  auto gathered = mx::distributed::all_gather(local_array, runtime.group());
  mx::eval(gathered);
  if (gathered.size() != runtime.world_size() * kIdentityWords) {
    return -3;
  }
  const auto* values = gathered.data<int32_t>();
  const auto expected_backend = static_cast<int32_t>(backend_digest(runtime.backend()));
  for (int rank = 0; rank < runtime.world_size(); ++rank) {
    const auto* row = values + rank * kIdentityWords;
    if (row[0] != rank || row[1] != expected_backend) {
      return -4;
    }
    for (int i = 0; i < kDigestWords; ++i) {
      if (row[2 + i] != static_cast<int32_t>(membership_digest[i])) {
        return -5;
      }
      const uint32_t expected_generation_word =
          i < 2 ? static_cast<uint32_t>(generation >> (i * 32)) : 0u;
      if (row[2 + kDigestWords + i] != static_cast<int32_t>(expected_generation_word)) {
        return -6;
      }
      if (row[2 + kDigestWords * 2 + i] !=
          static_cast<int32_t>(expected_member_digests[rank * kDigestWords + i])) {
        return -7;
      }
    }
  }
  return 0;
}

} // namespace

extern "C" {

int64_t mlx_group_runtime_create(const char* backend, int strict) {
  if (!backend || backend[0] == '\0') {
    return 0;
  }
  try {
    if (mlx_init() != 0) {
      return 0;
    }
    auto group = mx::distributed::init(strict != 0, std::string(backend));
    auto runtime = std::make_unique<mlx_ir::GroupRuntime>(
        std::move(group), std::string(backend));
    return static_cast<int64_t>(
        reinterpret_cast<intptr_t>(runtime.release()));
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_group_runtime_create", e);
    return 0;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_group_runtime_create unknown exception" << std::endl;
    return 0;
  }
}

int mlx_group_runtime_rank(int64_t handle) {
  auto* runtime = mlx_ir::group_runtime_from_handle(handle);
  return runtime ? runtime->rank() : -1;
}

int mlx_group_runtime_world_size(int64_t handle) {
  auto* runtime = mlx_ir::group_runtime_from_handle(handle);
  return runtime ? runtime->world_size() : -1;
}

int mlx_group_runtime_validate_identity(
    int64_t handle,
    uint64_t generation,
    const uint32_t* membership_digest,
    const uint32_t* expected_member_digests,
    int n_members,
    const uint32_t* local_member_digest) {
  auto* runtime = mlx_ir::group_runtime_from_handle(handle);
  if (!runtime) {
    return -1;
  }
  try {
    return validate_identity_exchange(
        *runtime,
        generation,
        membership_digest,
        expected_member_digests,
        n_members,
        local_member_digest);
  } catch (const std::exception& e) {
    log_bridge_exception("mlx_group_runtime_validate_identity", e);
    return -8;
  } catch (...) {
    std::cerr << "[mlx_bridge] mlx_group_runtime_validate_identity unknown exception" << std::endl;
    return -8;
  }
}

void mlx_group_runtime_destroy(int64_t handle) {
  delete mlx_ir::group_runtime_from_handle(handle);
}

} // extern "C"
