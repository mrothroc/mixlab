#ifndef MLX_BACKWARD_TRACE_H
#define MLX_BACKWARD_TRACE_H

#include "ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlx_ir {

struct BackwardTraceSummary {
  uint64_t bad_edges = 0;
  int first_forward_bad_op_index = -1;
  int first_forward_bad_op_type = 0;
  int first_forward_bad_output_index = -1;
  int first_bad_op_index = -1;
  int first_bad_op_type = 0;
  int first_bad_input_index = -1;
};

class BackwardTraceCollector {
 public:
  explicit BackwardTraceCollector(int training_step);

  mlx::core::array wrap_input(
      const mlx::core::array& value,
      size_t op_index,
      int op_type,
      int input_index,
      std::string input_name);
  mlx::core::array wrap_output(
      const mlx::core::array& value,
      size_t op_index,
      int op_type,
      int output_index,
      std::string output_name);
  void append_evaluation_arrays(std::vector<mlx::core::array>& arrays) const;
  BackwardTraceSummary summarize_and_log() const;

 private:
  struct Record {
    bool is_output;
    size_t op_index;
    int op_type;
    int input_index;
    std::string input_name;
    mlx::core::array forward_nonfinite_count;
    mlx::core::array forward_max_abs_finite;
    mlx::core::array nonfinite_count;
    mlx::core::array max_abs_finite;
  };

  int training_step_;
  std::vector<Record> records_;
};

class BackwardTraceScope {
 public:
  explicit BackwardTraceScope(BackwardTraceCollector* collector);
  ~BackwardTraceScope();

  BackwardTraceScope(const BackwardTraceScope&) = delete;
  BackwardTraceScope& operator=(const BackwardTraceScope&) = delete;

 private:
  BackwardTraceCollector* previous_;
};

bool backward_trace_enabled_for_step(int training_step);
bool backward_trace_active();
mlx::core::array trace_backward_input(
    const mlx::core::array& value,
    size_t op_index,
    int op_type,
    int input_index,
    const std::string& input_name);
mlx::core::array trace_backward_output(
    const mlx::core::array& value,
    size_t op_index,
    int op_type,
    int output_index,
    const std::string& output_name);
const char* ir_op_type_name(int op_type);

} // namespace mlx_ir

#endif
