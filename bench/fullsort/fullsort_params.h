#ifndef VXSORT_FULLSORT_PARAMS_H
#define VXSORT_FULLSORT_PARAMS_H

#include <vxsort.h>

namespace vxsort_bench {

using namespace vxsort::types;
using vxsort::vector_machine;

const auto processor_count = 1;

static const i32 MIN_SORT = 256;
static const i32 MAX_SORT = 1 << 24;

static const i32 MIN_STRIDE = 1 << 3;
static const i32 MAX_STRIDE = 1 << 27;
}

#endif //VXSORT_FULLSORT_PARAMS_H
