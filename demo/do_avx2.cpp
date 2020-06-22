
#include "vxsort_targets_enable_avx2.h"

#include <vector>
#include "vxsort.h"
#include "machine_traits.avx2.h"
#include "smallsort/bitonic_sort.AVX2.int32_t.generated.h"

namespace vxsort_demo {
void do_avx2(std::vector<int>& V) {
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;

  auto sorter = vxsort::vxsort<int, vxsort::vector_machine::AVX2, 8>();
  sorter.sort(begin, end);
}
}
#include "vxsort_targets_disable.h"
