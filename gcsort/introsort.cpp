#include "introsort.h"


#ifdef _MSC_VER
    #define NOINLINE __declspec(noinline)
#else
    #define NOINLINE __attribute__((noinline))
#endif


extern "C" NOINLINE void sort_introsort(uint8_t** begin,
                                        uint8_t** end) {
  gcsort::introsort::sort(begin, end, 0);
}

extern "C" NOINLINE void sort_insertionsort(uint8_t** begin,
                                            uint8_t** end) {
  gcsort::introsort::sort(begin, end, 0);
}
