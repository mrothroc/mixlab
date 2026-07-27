#ifndef MIXLAB_GROUP_RUNTIME_H
#define MIXLAB_GROUP_RUNTIME_H

#include <mlx/distributed/distributed.h>

#include <cstdint>
#include <string>

namespace mlx_ir {

class GroupRuntime {
 public:
  GroupRuntime(mlx::core::distributed::Group group, std::string backend);

  const mlx::core::distributed::Group& group() const;
  const std::string& backend() const;
  int rank() const;
  int world_size() const;

 private:
  const mlx::core::distributed::Group group_;
  const std::string backend_;
  const int rank_;
  const int world_size_;
};

GroupRuntime* group_runtime_from_handle(int64_t handle);

} // namespace mlx_ir

#endif
