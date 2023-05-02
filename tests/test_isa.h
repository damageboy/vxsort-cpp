#ifndef VXSORT_TEST_ISA_H
#define VXSORT_TEST_ISA_H

#include "isa_detection.h"

#define VXSORT_TEST_ISA() \
    if (!::vxsort::supports_vector_machine<M>(sizeof(T))) { \
        GTEST_SKIP_("Current CPU does not support the minimal features for this test"); \
        return; \
    }

#endif //VXSORT_TEST_ISA_H
