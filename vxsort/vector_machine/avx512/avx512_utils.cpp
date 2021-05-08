#include <cstdint>

#include "../machine_traits.avx512.h"

namespace vxsort {

__m512_dbg __m512_dbg::make(__m512_dbg::TV v) {
    return __m512_dbg(v);
}

}
