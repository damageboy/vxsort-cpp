#include <algorithm>
#include <numeric>
#include <random>
#include <fcntl.h>
#include <ctime>

using namespace std;

#include "isa_detection.h"

using vxsort::vector_machine;

extern void do_avx2(int *begin, int *end);
extern void do_avx512(int *begin, int *end);

int main(int argc, char** argv) {
    time_t t;
    if (argc != 2) {
        fprintf(stderr, "demo array size must be sepcificied\n");
        return -1;
    }

    const size_t vector_size = atoi(argv[1]);
    srand((unsigned)time(&t));

    auto* data = static_cast<int32_t*>(malloc(sizeof(int32_t) * vector_size));

    for (size_t i = 0; i < vector_size; i++) {
            data[i] = rand();
    }

    const auto begin = data;
    const auto end = begin + vector_size - 1;

#if defined(CPU_FEATURES_ARCH_X86)
    if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX512)) {
        fprintf(stderr, "Sorting with AVX512...");
        do_avx512(begin, end);
        fprintf(stderr, "...done!\n");
    } else if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX2)) {
        fprintf(stderr, "Sorting with AVX2...");
        do_avx2(begin, end);
        fprintf(stderr, "...done!\n");
    } else
#endif
#if defined(CPU_FEATURES_ARCH_AARCH64)
    if (vxsort::supports_vector_machine(vxsort::vector_machine::NEON)) {
    }
#endif

    {
        fprintf(stderr, "CPU doesn't seem to support any vectorized ISA, bye-bye\n");
        return -2;
    }

    return 0;
}
