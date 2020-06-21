#include "bitonic_sort.AVX512.int64_t.generated.h"

using namespace vxsort;

void vxsort::smallsort::bitonic<int64_t, vector_machine::AVX512 >::sort(int64_t *ptr, size_t length) {
    const int N = 8;

    switch(length / N) {
        case 1: sort_01v(ptr); break;
        case 2: sort_02v(ptr); break;
        case 3: sort_03v(ptr); break;
        case 4: sort_04v(ptr); break;
        case 5: sort_05v(ptr); break;
        case 6: sort_06v(ptr); break;
        case 7: sort_07v(ptr); break;
        case 8: sort_08v(ptr); break;
        case 9: sort_09v(ptr); break;
        case 10: sort_10v(ptr); break;
        case 11: sort_11v(ptr); break;
        case 12: sort_12v(ptr); break;
        case 13: sort_13v(ptr); break;
        case 14: sort_14v(ptr); break;
        case 15: sort_15v(ptr); break;
        case 16: sort_16v(ptr); break;
        case 17: sort_17v(ptr); break;
        case 18: sort_18v(ptr); break;
        case 19: sort_19v(ptr); break;
        case 20: sort_20v(ptr); break;
        case 21: sort_21v(ptr); break;
        case 22: sort_22v(ptr); break;
        case 23: sort_23v(ptr); break;
        case 24: sort_24v(ptr); break;
        case 25: sort_25v(ptr); break;
        case 26: sort_26v(ptr); break;
        case 27: sort_27v(ptr); break;
        case 28: sort_28v(ptr); break;
        case 29: sort_29v(ptr); break;
        case 30: sort_30v(ptr); break;
        case 31: sort_31v(ptr); break;
        case 32: sort_32v(ptr); break;
    }
}
