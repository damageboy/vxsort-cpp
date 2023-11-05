#include "benchmark/benchmark.h"

namespace vxsort_bench {

void register_fullsort_avx2_i_benchmarks();
void register_fullsort_avx2_u_benchmarks();
void register_fullsort_avx2_f_benchmarks();
void register_fullsort_avx512_i_benchmarks();
void register_fullsort_avx512_u_benchmarks();
void register_fullsort_avx512_f_benchmarks();

void register_benchmarks() {
    register_fullsort_avx2_i_benchmarks();
    register_fullsort_avx2_u_benchmarks();
    register_fullsort_avx2_f_benchmarks();
    register_fullsort_avx512_i_benchmarks();
    register_fullsort_avx512_u_benchmarks();
    register_fullsort_avx512_f_benchmarks();
}
}  // namespace vxsort_bench

using namespace std;

int main(int argc, char** argv) {
    vxsort_bench::register_benchmarks();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
