#include "vxsort_targets_enable_avx512.h"

#include "vxsort.avx512.h"

using namespace vxsort::types;

void do_avx512(i64 *begin, i64 *end) {
  auto sorter = vxsort::vxsort<i64, vxsort::vector_machine::AVX512, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
