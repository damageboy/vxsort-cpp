#ifndef VXSORT_BENCH_UTIL_H
#define VXSORT_BENCH_UTIL_H

#include <benchmark/benchmark.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace benchmark;

namespace vxsort_bench {

Counter make_time_per_n_counter(int64_t n);

Counter make_cycle_per_n_counter(double n);


template <typename T>
extern void generate_unique_ptrs_vec(std::vector<T>& vec, size_t n) {
  std::iota(vec.begin(), vec.end(), (T) 0x1000);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(vec.begin(), vec.end(), g);
}

template <typename T>
extern std::vector<T *> generate_array_beginnings(std::vector<std::vector<T>> &copies) {
  const auto num_copies = copies.size();
  std::vector<T *> begins(num_copies);
  for (size_t i = 0; i < num_copies; i++)
    begins[i] = copies[i].data();
  return begins;
}
template <typename T>
extern void refresh_copies(std::vector<std::vector<T>> &copies,
                           std::vector<T>& orig) {
  const auto begin = orig.begin();
  const auto end = orig.end();
  const auto num_copies = copies.size();
  for (size_t i = 0; i < num_copies; i++)
    copies[i].assign(begin, end);
}
template <typename T>
extern std::vector<std::vector<T>> generate_copies(size_t num_copies,  int64_t n, std::vector<T>& orig) {
  std::vector<std::vector<T>> copies(num_copies);
  for (size_t i = 0; i < num_copies; i++)
    copies[i] = std::vector<T>(n);
  refresh_copies(copies, orig);
  return copies;
}

}

#endif //VXSORT_BENCH_UTIL_H
