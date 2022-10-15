#include "vxsort_targets_enable_avx2.h"

#include "vxsort.avx2.h"

using namespace vxsort::types;

void do_avx2(int *begin, int *end) {
  auto sorter = vxsort::vxsort<i32, vxsort::vector_machine::AVX2, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
