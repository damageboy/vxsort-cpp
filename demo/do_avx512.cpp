#include "vxsort_targets_enable_avx512.h"

#include "vxsort.h"
#include "machine_traits.avx512.h"
#include "smallsort/bitonic_sort.AVX512.int32_t.generated.h"

void do_avx512(int *begin, int *end) {
  auto sorter = vxsort::vxsort<int, vxsort::vector_machine::AVX512, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
