#include "util.h"

#include <benchmark/benchmark.h>


Counter make_time_per_n_counter(int64_t n) {
return benchmark::Counter(
(double) n,
Counter::kAvgThreadsRate |
Counter::kIsIterationInvariantRate |
Counter::kInvert,
Counter::kIs1000);
}
