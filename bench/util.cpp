#include "util.h"

#include <benchmark/benchmark.h>

namespace vxsort_bench {
Counter make_time_per_n_counter(int64_t n) {
  return benchmark::Counter((double)n,
                            Counter::kAvgThreadsRate |
                            Counter::kIsIterationInvariantRate |
                            Counter::kInvert,
                            Counter::kIs1000);
}

Counter make_cycle_per_n_counter(double n) {
  return benchmark::Counter(n,
                            Counter::kIsRate,
                            Counter::kIs1000);
}

}
