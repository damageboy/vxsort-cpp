#include "introsort.h"

extern "C" __attribute__((noinline)) void sort_introsort(uint8_t** begin, uint8_t** end)
{
  gcsort::introsort::sort(begin, end, 0);
}


extern "C" __attribute__((noinline)) void sort_insertionsort(uint8_t** begin, uint8_t** end)
{
  gcsort::introsort::sort(begin, end, 0);
}
