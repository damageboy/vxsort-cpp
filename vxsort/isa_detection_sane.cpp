#include "isa_detection.h"

#if _MSC_VER
#else
namespace vxsort {

  bool init_isa_detection() {
    __builtin_cpu_init();
    return true;
  }

  extern bool supports_vector_machine(vector_machine m)
  {
    switch (m) {
      case NONE:
        return true;
      case AVX2:
        return __builtin_cpu_supports("avx2");
      case AVX512:
        return __builtin_cpu_supports("avx512f");
        break;
      case SVE:
        return false;
    }
    return false;
  }
}  // namespace gcsort
#endif
