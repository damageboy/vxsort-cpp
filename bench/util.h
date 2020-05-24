//
// Created by dans on 5/24/20.
//

#ifndef GCSORT_BENCH_UTIL_H
#define GCSORT_BENCH_UTIL_H
#include <algorithm>
#include <random>

extern void generate_unique_ptrs_vec(std::vector<uint8_t *>& vec, size_t n);
extern void generate_unique_ptrs_vec(std::vector<int64_t>& vec, size_t n);
extern void generate_unique_ptrs_vec(std::vector<int32_t>& vec, size_t n);
template <typename T>
extern std::vector<T *> generate_array_beginnings(const std::vector<std::vector<T> *> &copies);
template <typename T>
extern void refresh_copies(const std::vector<std::vector<T> *> &copies,
                    std::vector<T>& orig);
template <typename T>
extern std::vector<std::vector<T> *> generate_copies(int num_copies,  int32_t n, std::vector<T>& orig);

template <typename T> extern void delete_copies(std::vector<std::vector<T> *> &copies);

#endif
