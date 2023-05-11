# vxsort-cpp

## Tests

[![Build and Test](https://github.com/damageboy/vxsort-cpp/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/damageboy/vxsort-cpp/actions/workflows/build-and-test.yml)
![Latest Test Status](https://gist.githubusercontent.com/damageboy/dfd9d01f2c710f96b444532b92539321/raw/vxsort-suites-badge.svg)
![Latest Test Status](https://gist.githubusercontent.com/damageboy/dfd9d01f2c710f96b444532b92539321/raw/vxsort-tests-badge.svg)
![Latest Test Status](https://gist.githubusercontent.com/damageboy/dfd9d01f2c710f96b444532b92539321/raw/vxsort-runs-badge.svg)

## What

vxsort is a fast, somewhat novel, hybrid, vectorized quicksort+bitonic primitive sorter implemented in C++.  
The name vxsort stands for <ins>v</ins>ectorized 10<ins>x</ins> <ins>sort</ins>.
It currently supports the following combination of vector ISA and primitive types:

|        | i64 | i32 | i16           | u64 | u32 | u16           | f64 | f32 | f16 |
|--------|-----|-----|---------------|-----|-----|---------------|-----|-----|-----|
| AVX2   | ✅   | ✅   | ✅             | ✅   | ✅   | ✅             | ✅   | ✅   | ❌   |
| AVX512 | ✅   | ✅   | ✅<sup>1</sup> | ✅   | ✅   | ✅<sup>1</sup> | ✅   | ✅   | ❌   |
| ARM-Neon|  ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| ARM-SVE2|  ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |
| RiscV-V 1.0 |  ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   |

<sup>1</sup> - Requires AVX512/VBMI2 support which is available for all Intel AVX512 CPUs post Icelake, AMD post Zen4 


## Benchmark Results

## Building

```shell
mkdir build
cd build
# For better code-gen
export CC=clang CXX=clang++
cmake .. -G Ninja
ninja
```

## Testing

To run tests, use ctest, preferrably with `-J $(nproc)` to avoid waiting for a long time: 

```bash
ctest -J $(nproc)
```

Tests are built into 3 executables (signed integers, unsigned integer, floating-point) per supported vector ISA.
This allows for easy to expolit parallelization both whne building and executing the tests.

## Benchmarking

1. Plug in to a power-source if on laptop
2. Try to kill CPU hogs
 -or-
3. Alternatively, rely on the supplied `run.sh` to `kill -STOP / -CONT` well known CPU hogs

```bash
# This script will kill -STOP a bunch of CPU hogs like chrome,
# Then executes
# ./bench/vxsort_bench --benchmark_counters_tabular
# and resume them after it's done
./bench/run.sh
```


