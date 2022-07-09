#include "tensorflow/compiler/plugin/aluminum_shark/utils/parallel.h"

#include <thread>
#include <vector>

namespace {
const uint n_threads = std::thread::hardware_concurrency();
}

namespace aluminum_shark {

void run_parallel(size_t start, size_t stop, std::function<void(size_t)> func) {
  // number of threads to use
  const uint nt = stop - start < n_threads ? stop - start : n_threads;

  // create threads
  std::vector<std::thread> threads;
  threads.reserve(nt);
  for (size_t thread_id = 0; thread_id < nt; ++thread_id) {
    threads.push_back(std::thread([thread_id, nt, start, stop, &func] {
      // iterate over all indecies and execute all where i % nt == thread_id;
      for (size_t i = start; i < stop; ++i) {
        if (i % nt == thread_id) {
          func(i);
        }
      }
    }));
  }
  // wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace aluminum_shark