#ifndef GCSORT_ISA_DETECTION_H
#define GCSORT_ISA_DETECTION_H

#include "machine_traits.h"

namespace gcsort {
  extern void init_isa_detection();
  extern bool __builtin_has_cpufeature_avx2();
  extern bool __builtin_has_cpufeature_avx512f();
}

#endif  // GCSORT_ISA_DETECTION_H
