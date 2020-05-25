#ifndef GCSORT_BENCH_UTIL_H
#define GCSORT_BENCH_UTIL_H

#include <vector>
#include <algorithm>
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
  std::vector<T *> begins(copies.size());
  for (auto i = 0; i < copies.size(); i++)
    begins[i] = copies[i]->data();
  return begins;
}
template <typename T>
extern void refresh_copies(const std::vector<std::vector<T> *> &copies,
                           std::vector<T>& orig) {
  auto begin = orig.begin();
  auto end = orig.end();
  for (auto i = 0; i < copies.size(); i++)
    copies[i]->assign(begin, end);
}
template <typename T>
extern std::vector<std::vector<T> *> generate_copies(int num_copies,  int32_t n, std::vector<T>& orig) {
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
