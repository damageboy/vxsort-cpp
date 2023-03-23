#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

#include "isa_detection.h"

using vxsort::vector_machine;
using namespace vxsort::types;

extern void do_avx2(i64 *begin, i64 *end);
extern void do_avx512(i64 *begin, i64 *end);

std::vector<i64> generate_random_garbage(const usize size) {

    auto vec = std::vector<i64>(size);
    std::iota(vec.begin(), vec.end(), 666);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(vec.begin(), vec.end(), g);
    return vec;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "demo array size must be specified\n");
        return -1;
    }

    const size_t vector_size = atoi(argv[1]);
    auto v = generate_random_garbage(vector_size);

    const auto begin = v.data();
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
    } else
#endif

    {
        fprintf(stderr, "CPU doesn't seem to support any vectorized ISA, bye-bye\n");
        return -2;
    }

    return 0;
}
