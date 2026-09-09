"""CPU-only test of the actual MLX CUDA worker.cpp, with CUDA calls stubbed.

The completion scheduler/loop is unmodified; only CUDA event delivery is replaced.
This isolates idle spinning, not CUDA ordering or the long-run training hang.
"""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


HEADER = r"""
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
using cudaStream_t = int;
constexpr int cudaEventDisableTiming = 1, cudaEventBlockingSync = 2;
#define CHECK_CUDA_ERROR(call) (call)
inline int cudaLaunchHostFunc(int, void (*fn)(void*), void* data) {
  fn(data);
  return 0;
}
namespace mlx::core::cu {
struct Device {};
struct Stream {
  explicit Stream(Device&) {}
  operator int() { return 0; }
};
struct Event {
  Event(Device&, int) {}
  void record(int) {}
  void wait(int) {}
};
struct CountedCondition {
  std::condition_variable cv;
  std::atomic<uint64_t> visits{0};
  template<class Predicate>
  void wait(std::unique_lock<std::mutex>& lock, Predicate predicate) {
    ++visits;
    cv.wait(lock, predicate);
  }
  void notify_one() { cv.notify_one(); }
};
class Worker : public std::enable_shared_from_this<Worker> {
 public:
  explicit Worker(Device&);
  ~Worker();
  void start();
  void stop();
  void add_task(std::function<void()>);
  static void signal(void*);
  void commit(int);
  void thread_fn();
  using Tasks = std::vector<std::function<void()>>;
  Stream signal_stream_;
  Event signal_event_;
  std::thread worker_;
  std::mutex mtx_;
  CountedCondition cond_;
  bool stop_{false};
  uint64_t committed_batch_{0}, signaled_batch_{0};
  Tasks pending_tasks_;
  std::map<uint64_t, Tasks> worker_tasks_;
};
}
"""

MAIN = r"""
#include "mlx/backend/cuda/worker.h"
using namespace mlx::core::cu;
int main(int argc, char**) {
  Device device;
  auto worker = std::make_shared<Worker>(device);
  // Join explicitly so shutdown is also tested, unlike production's detach.
  std::thread thread([worker] { worker->thread_fn(); });
  bool delivered = true;
  for (int i = 0; i < 3; ++i) {
    auto done = std::make_shared<std::promise<void>>();
    auto result = done->get_future();
    worker->add_task([done] { done->set_value(); });
    worker->commit(0);
    if (result.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      delivered = false;
      break;
    }
  }
  auto before = worker->cond_.visits.load();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto idle_visits = worker->cond_.visits.load() - before;
  auto after_idle = std::make_shared<std::promise<void>>();
  auto resumed = after_idle->get_future();
  worker->add_task([after_idle] { after_idle->set_value(); });
  worker->commit(0);
  delivered &= resumed.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  worker->stop();
  thread.join();
  std::cout << "callbacks_delivered=" << delivered
            << " idle_loop_visits=" << idle_visits << std::endl;
  bool spin = idle_visits > 100;
  return delivered && (argc > 1 ? spin : !spin) ? 0 : 1;
}
"""


def run_probe(source, expect_spin=False):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("a C++ compiler is required for the MLX worker probe")
    with tempfile.TemporaryDirectory(prefix="mixlab-worker-probe-") as tmp:
        root = Path(tmp)
        include = root / "mlx/backend/cuda"
        include.mkdir(parents=True)
        (include / "worker.h").write_text(HEADER)
        (include / "device.h").write_text('#include "mlx/backend/cuda/worker.h"\n')
        (root / "main.cpp").write_text(MAIN)
        binary = root / "probe"
        subprocess.run([compiler, "-std=c++20", "-O2", "-pthread", "-I", tmp,
                        str(source), str(root / "main.cpp"), "-o", str(binary)],
                       check=True, timeout=60)
        return subprocess.run([str(binary)] + (["expect-spin"] if expect_spin else []),
                              check=True, timeout=5, text=True, capture_output=True).stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-spin", action="store_true")
    args = parser.parse_args()
    print(run_probe(args.source, args.expect_spin), end="")
