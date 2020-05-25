#ifndef GCSORT_TEST_UTIL_H
#define GCSORT_TEST_UTIL_H

#include <vector>
#include <algorithm>
#include <random>

template <typename T>
void generate_unique_ptrs_vec(std::vector<T>& vec) {
  std::iota(vec.begin(), vec.end(), (T)0x1000);

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
