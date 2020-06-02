#include "isa_detection.h"

#if _MSC_VER
#else
namespace gcsort {

  void init_isa_detection() {
    __builtin_cpu_init();
  }
  bool __builtin_has_cpufeature_avx2() {
    return __builtin_cpu_supports("avx2");
  }
  bool __builtin_has_cpufeature_avx512f() {
    return __builtin_cpu_supports("avx512f");
  }
}  // namespace gcsort
#endif
