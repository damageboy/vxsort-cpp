#include "vxsort_targets_enable_avx2.h"

#include "vxsort.avx2.h"

using namespace vxsort::types;

void do_avx2(i64 *begin, i64 *end) {
  auto sorter = vxsort::vxsort<i64, vxsort::vector_machine::AVX2, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
