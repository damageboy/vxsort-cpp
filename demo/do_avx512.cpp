#include "vxsort_targets_enable_avx512.h"

#include "vxsort.avx512.h"

using namespace vxsort::types;

void do_avx512(int *begin, int *end) {
  auto sorter = vxsort::vxsort<i32, vxsort::vector_machine::AVX512, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
