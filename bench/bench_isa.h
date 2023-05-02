#ifndef VXSORT_BENCH_ISA_H
#define VXSORT_BENCH_ISA_H

#include <isa_detection.h>

#define VXSORT_BENCH_ISA() \
    if (!::vxsort::supports_vector_machine<M>(sizeof(Q))) { \
                state.SkipWithError("Current CPU does not support the minimal features for this benchmark"); \
        return; \
    }

#endif //VXSORT_BENCH_ISA_H
