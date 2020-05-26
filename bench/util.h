#ifndef GCSORT_BENCH_UTIL_H
#define GCSORT_BENCH_UTIL_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

template <typename T>
extern void generate_unique_ptrs_vec(std::vector<T>& vec, size_t n) {
  std::iota(vec.begin(), vec.end(), (T) 0x1000);

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(vec.begin(), vec.end(), g);
}

template <typename T>
extern std::vector<T *> generate_array_beginnings(const std::vector<std::vector<T> *> &copies) {
  const auto num_copies = copies.size();
  std::vector<T *> begins(num_copies);
  for (auto i = 0; i < num_copies; i++)
    begins[i] = copies[i]->data();
  return begins;
}
template <typename T>
extern void refresh_copies(const std::vector<std::vector<T> *> &copies,
                           std::vector<T>& orig) {
  const auto begin = orig.begin();
  const auto end = orig.end();
  const auto num_copies = copies.size();
  for (auto i = 0; i < num_copies; i++)
    copies[i]->assign(begin, end);
}
template <typename T>
extern std::vector<std::vector<T> *> generate_copies(int num_copies,  int64_t n, std::vector<T>& orig) {
  std::vector<std::vector<T>*> copies(num_copies);
  for (auto i = 0; i < num_copies; i++)
    copies[i] = new std::vector<T>(n);
  refresh_copies(copies, orig);
  return copies;
}
template <typename T> extern void delete_copies(std::vector<std::vector<T> *> &copies)
{
  for (auto c : copies)
    delete c;
}

#endif
