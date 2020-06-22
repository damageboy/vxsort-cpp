#include "vxsort_targets_enable_avx512.h"

#include <vector>
#include "vxsort.h"
#include "machine_traits.avx512.h"
#include "smallsort/bitonic_sort.AVX512.int32_t.generated.h"

namespace vxsort_demo {
void do_avx512(std::vector<int>& V) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = vxsort::vxsort<int, vxsort::vector_machine::AVX512, 8>();
  sorter.sort(begin, end);
}
}
#include "vxsort_targets_disable.h"
