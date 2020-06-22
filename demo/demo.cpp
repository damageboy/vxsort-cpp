#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

#include "isa_detection.h"

using vxsort::vector_machine;

template <typename T>
extern void generate_unique_ptrs_vec(std::vector<T>& vec) {
  std::iota(vec.begin(), vec.end(), (T)0x1000);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(vec.begin(), vec.end(), g);
}

extern void do_avx2(int *begin, int *end);
extern void do_avx512(int *begin, int *end);

int main(int argc, char** argv)
{
  if (argc != 2) {
    fprintf(stderr, "demo array size must be sepcificied\n");
    return -1;
  }

  size_t vector_size = atoi(argv[1]);
  auto V = std::vector<int>(vector_size);
  generate_unique_ptrs_vec(V);
  auto begin = V.data();
  auto end = V.data() + V.size() - 1;


  if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX512)) {
    fprintf(stderr, "Sorting with AVX512...");
    do_avx512(begin, end);
    fprintf(stderr, "...done!\n");
  } else if (vxsort::supports_vector_machine(vxsort::vector_machine::AVX2)) {
    fprintf(stderr, "Sorting with AVX2...");
    do_avx2(begin, end);
    fprintf(stderr, "...done!\n");
  } else {
    fprintf(stderr, "CPU doesn't seem to support any vectorized ISA, bye-bye\n");
    return -2;
  }

  return 0;
}


