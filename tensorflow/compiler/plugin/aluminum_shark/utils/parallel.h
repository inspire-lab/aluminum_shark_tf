#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_PARALLEL_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_PARALLEL_H

#include <functional>

namespace aluminum_shark {

/*
Runs `func` for every value in [start, stop). Distributes the workload on as
many threads as possible.
*/
void run_parallel(size_t start, size_t stop, std::function<void(size_t)> func);

/*
Runs `func` with as many threads as possible until each thread has returned
false
*/
void run_parallel(std::function<bool()> func);

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_PARALLEL_H \
        */
