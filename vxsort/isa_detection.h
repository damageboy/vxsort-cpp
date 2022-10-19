#ifndef VXSORT_ISA_DETECTION_H
#define VXSORT_ISA_DETECTION_H

#include "vector_machine/machine_traits.h"
#include "cpu_features_macros.h"

namespace vxsort {

extern bool init_isa_detection();
extern bool supports_vector_machine(vector_machine m);

template <vector_machine M>
bool supports_vector_machine(usize width);

} // namespace vxsort

#endif  // VXSORT_ISA_DETECTION_H
