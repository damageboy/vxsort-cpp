#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

#include "isa_detection.h"

namespace vxsort_demo {

template <typename T>
extern void generate_unique_ptrs_vec(std::vector<T>& vec, size_t n) {
  std::iota(vec.begin(), vec.end(), (T)0x1000);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(vec.begin(), vec.end(), g);
}


using vxsort::vector_machine;

template <class T, int Unroll, vector_machine M>
void perform_vxsort_test(std::vector<T> V) {

}

extern void do_avx2(std::vector<int>& V);
extern void do_avx512(std::vector<int>& V);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    fprintf(stderr, "demo array size must be sepcificied\n");
    return -1;
  }

  size_t vector_size = atoi(argv[1]);
  auto data = std::vector<int>(vector_size);

  if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX512)) {
    fprintf(stderr, "Sorting with AVX512...");
    vxsort_demo::do_avx512(data);
    fprintf(stderr, "...done!\n");
  } else if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX2)) {
    fprintf(stderr, "Sorting with AVX2...");
    vxsort_demo::do_avx2(data);
    fprintf(stderr, "...done!\n");
  } else {
    fprintf(stderr, "CPU doesn't seem to support any vectorized ISA, bye-bye\n");
    return -2;
  }

  return 0;
}


