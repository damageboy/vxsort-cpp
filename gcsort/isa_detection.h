#ifndef VXSORT_ISA_DETECTION_H
#define VXSORT_ISA_DETECTION_H

#include "machine_traits.h"

namespace vxsort {
  extern bool init_isa_detection();
  extern bool supports_vector_machine(vector_machine m);
}

#endif  // VXSORT_ISA_DETECTION_H
