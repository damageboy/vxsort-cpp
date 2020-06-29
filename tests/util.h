#ifndef VXSORT_TEST_UTIL_H
#define VXSORT_TEST_UTIL_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

template <typename T>
void generate_unique_ptrs_vec(std::vector<T>& vec, T start, T stride=0x1, bool randomize = true) {
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = start;
    start += stride;
  }

  if (!randomize)
    return;

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(vec.begin(), vec.end(), g);
}

template <typename IntType>
std::vector<IntType> range(IntType start, IntType stop, IntType step) {
  if (step == IntType(0)) {
    throw std::invalid_argument("step for range must be non-zero");
  }

  std::vector<IntType> result;
  IntType i = start;
  while ((step > 0) ? (i <= stop) : (i > stop)) {
    result.push_back(i);
    i += step;
  }

  return result;
}

template <typename IntType>
std::vector<IntType> multiply_range(IntType start, IntType stop, IntType step) {
  if (step == IntType(0)) {
    throw std::invalid_argument("step for range must be non-zero");
  }

  std::vector<IntType> result;
  IntType i = start;
  while ((step > 0) ? (i <= stop) : (i > stop)) {
    result.push_back(i);
    i *= step;
  }

  return result;
}

#endif
