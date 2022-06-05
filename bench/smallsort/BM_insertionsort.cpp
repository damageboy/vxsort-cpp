#include "BM_smallsort.h"

namespace vxsort_bench {
using namespace vxsort::types;
using benchmark::TimeUnit;

BENCHMARK_TEMPLATE(BM_insertion_sort, i16)->DenseRange(8, 128, 8)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_insertion_sort, i32)->DenseRange(8, 128, 8)->Unit(kNanosecond);
BENCHMARK_TEMPLATE(BM_insertion_sort, i64)->DenseRange(8, 128, 8)->Unit(kNanosecond);

}