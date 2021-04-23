
#include "vxsort_targets_enable_avx2.h"

#include "smallsort/avx2/bitonic_machine.AVX2.i32.generated.h"
#include "smallsort/bitonic_sort.h"
#include "vector_machine/machine_traits.avx2.h"
#include "vxsort.h"

void do_avx2(int *begin, int *end) {
  auto sorter = vxsort::vxsort<int, vxsort::vector_machine::AVX2, 8>();
  sorter.sort(begin, end);
}
#include "vxsort_targets_disable.h"
