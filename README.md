# gcsort

## What

This is a port of the C# [VxSort](https://github.com/damageboy/VxSort/) to high-perf C++.

## Building

```bash
mkdir build-release
cd build-release
# For better code-gen
export CC=clang
export CXX=clang++
cmake ..
make -j 4
```

## Testing

```bash
./test/gcsort_test
```

## Benchmarking

1. Plug in to a power-source if on laptop
2. Try to kill CPU hogs
 -or-
3. Alternatively, rely on the supplied `run.sh` to `kill -STOP / -CONT` well known CPU hogs

```bash
# This script will kill -STOP a bunch of CPU hogs like chrome,
# Then executes
# ./bench/gcsort_bench --benchmark_counters_tabular
# and resume them after it's done
./bench/run.sh
```

## Results (Ryzen 3950X, 3.8Ghz)

### int64

Compared to Introspective Sort, we can hit:
* For 1M `int64` elements, roughly 4.8x improvement with 8x unroll (`55ns` per element -> `11.5ns`)
* For 128K `int64` elements, roughly 4.5x improvement with 8x unroll (`45ns` per element -> `10.5ns`)

#### Introspective Sort (Scalar Baseline):

```bash
-----------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations     Time/N
-----------------------------------------------------------------------------------------
BM_full_introsort/4096                    1.18 ms         1.18 ms          586  28.8719ns
BM_full_introsort/8192                    2.78 ms         2.77 ms          262  33.8639ns
BM_full_introsort/16384                   5.86 ms         5.86 ms          120  35.7374ns
BM_full_introsort/32768                   12.4 ms         12.4 ms           54   37.948ns
BM_full_introsort/65536                   27.7 ms         27.6 ms           25  42.1669ns
BM_full_introsort/131072                  59.3 ms         59.3 ms           12  45.2203ns
BM_full_introsort/262144                   124 ms          124 ms            6   47.261ns
BM_full_introsort/524288                   268 ms          268 ms            3  51.0427ns
BM_full_introsort/1048576                  557 ms          557 ms            1  53.0751ns
```

#### VxSort No Unroll, Bitonic Sort 64-elements

```bash
-----------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations     Time/N
-----------------------------------------------------------------------------------------
BM_full_vxsort<int64_t, 1>/4096          0.486 ms        0.485 ms         1457  11.8505ns
BM_full_vxsort<int64_t, 1>/8192           1.29 ms         1.29 ms          561  15.7416ns
BM_full_vxsort<int64_t, 1>/16384          2.76 ms         2.75 ms          253  16.8082ns
BM_full_vxsort<int64_t, 1>/32768          6.03 ms         6.03 ms          116  18.3878ns
BM_full_vxsort<int64_t, 1>/65536          13.1 ms         13.1 ms           53   19.927ns
BM_full_vxsort<int64_t, 1>/131072         27.7 ms         27.7 ms           25  21.1131ns
BM_full_vxsort<int64_t, 1>/262144         60.1 ms         60.1 ms           12  22.9112ns
BM_full_vxsort<int64_t, 1>/524288          127 ms          126 ms            5  24.1048ns
BM_full_vxsort<int64_t, 1>/1048576         269 ms          269 ms            3  25.6178ns
```

#### VxSort Unroll x 4, Bitonic Sort 64-elements

```bash
-----------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations     Time/N
-----------------------------------------------------------------------------------------
BM_full_vxsort<int64_t, 4>/4096          0.279 ms        0.279 ms         2462  6.79957ns
BM_full_vxsort<int64_t, 4>/8192          0.673 ms        0.672 ms         1000  8.20411ns
BM_full_vxsort<int64_t, 4>/16384          1.52 ms         1.52 ms          455  9.25887ns
BM_full_vxsort<int64_t, 4>/32768          3.37 ms         3.36 ms          210  10.2602ns
BM_full_vxsort<int64_t, 4>/65536          7.20 ms         7.20 ms           96   10.982ns
BM_full_vxsort<int64_t, 4>/131072         15.1 ms         15.1 ms           46  11.4838ns
BM_full_vxsort<int64_t, 4>/262144         32.5 ms         32.5 ms           21  12.3887ns
BM_full_vxsort<int64_t, 4>/524288         67.4 ms         67.3 ms           10  12.8354ns
BM_full_vxsort<int64_t, 4>/1048576         144 ms          144 ms            5   13.689ns
```

#### VxSort Unroll x 8, Bitonic Sort 64-elements

```bash
-----------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations     Time/N
-----------------------------------------------------------------------------------------
BM_full_vxsort<int64_t, 8>/4096          0.271 ms        0.271 ms         2601  6.61364ns
BM_full_vxsort<int64_t, 8>/8192          0.603 ms        0.603 ms         1190  7.35612ns
BM_full_vxsort<int64_t, 8>/16384          1.35 ms         1.35 ms          517  8.23185ns
BM_full_vxsort<int64_t, 8>/32768          2.96 ms         2.96 ms          232  9.02835ns
BM_full_vxsort<int64_t, 8>/65536          6.32 ms         6.31 ms          111    9.634ns
BM_full_vxsort<int64_t, 8>/131072         13.3 ms         13.3 ms           52  10.1321ns
BM_full_vxsort<int64_t, 8>/262144         27.8 ms         27.8 ms           25    10.61ns
BM_full_vxsort<int64_t, 8>/524288         59.7 ms         59.6 ms           12  11.3669ns
BM_full_vxsort<int64_t, 8>/1048576         121 ms          121 ms            6  11.5373ns
```

#### VxSort Unroll x 12, Bitonic Sort 64-elements

```bash
-----------------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations     Time/N
-----------------------------------------------------------------------------------------
BM_full_vxsort<int64_t, 12>/4096         0.275 ms        0.275 ms         2504  6.70556ns
BM_full_vxsort<int64_t, 12>/8192         0.593 ms        0.593 ms         1140  7.23399ns
BM_full_vxsort<int64_t, 12>/16384         1.38 ms         1.37 ms          496  8.38849ns
BM_full_vxsort<int64_t, 12>/32768         2.95 ms         2.95 ms          235   8.9886ns
BM_full_vxsort<int64_t, 12>/65536         6.39 ms         6.38 ms          111  9.73922ns
BM_full_vxsort<int64_t, 12>/131072        13.2 ms         13.2 ms           53  10.0404ns
BM_full_vxsort<int64_t, 12>/262144        28.4 ms         28.4 ms           25   10.833ns
BM_full_vxsort<int64_t, 12>/524288        58.9 ms         58.8 ms           12  11.2206ns
BM_full_vxsort<int64_t, 12>/1048576        125 ms          124 ms            6  11.8665ns
```

### int32 

#### VxSort No Unroll, Bitonic Sort 128-elements

```
```bash
----------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations     Time/N
----------------------------------------------------------------------------------------
BM_full_vxsort<int32_t, 1>/4096         0.169 ms        0.169 ms         4261  4.11785ns
BM_full_vxsort<int32_t, 1>/8192         0.459 ms        0.459 ms         1516  5.60347ns
BM_full_vxsort<int32_t, 1>/16384         1.19 ms         1.19 ms          596  7.27952ns
BM_full_vxsort<int32_t, 1>/32768         2.61 ms         2.61 ms          269  7.96259ns
BM_full_vxsort<int32_t, 1>/65536         5.70 ms         5.69 ms          120  8.68616ns
BM_full_vxsort<int32_t, 1>/131072        12.6 ms         12.6 ms           57  9.62647ns
BM_full_vxsort<int32_t, 1>/262144        26.4 ms         26.4 ms           26  10.0556ns
BM_full_vxsort<int32_t, 1>/524288        56.9 ms         56.8 ms           12  10.8417ns
BM_full_vxsort<int32_t, 1>/1048576        120 ms          120 ms            6   11.407ns
```

#### VxSort Unroll x 4, Bitonic Sort 128-elements

```bash
----------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations     Time/N
----------------------------------------------------------------------------------------
BM_full_vxsort<int32_t, 4>/4096         0.135 ms        0.135 ms         4836  3.29119ns
BM_full_vxsort<int32_t, 4>/8192         0.291 ms        0.291 ms         2372  3.55463ns
BM_full_vxsort<int32_t, 4>/16384        0.657 ms        0.656 ms         1061  4.00521ns
BM_full_vxsort<int32_t, 4>/32768         1.57 ms         1.57 ms          462  4.79389ns
BM_full_vxsort<int32_t, 4>/65536         3.35 ms         3.34 ms          205  5.10211ns
BM_full_vxsort<int32_t, 4>/131072        7.20 ms         7.19 ms           98  5.48511ns
BM_full_vxsort<int32_t, 4>/262144        15.4 ms         15.4 ms           45  5.87159ns
BM_full_vxsort<int32_t, 4>/524288        34.0 ms         34.0 ms           22  6.48067ns
BM_full_vxsort<int32_t, 4>/1048576       68.6 ms         68.5 ms           10  6.53424ns
```

#### VxSort Unroll x 8, Bitonic Sort 128-elements

```bash
----------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations     Time/N
----------------------------------------------------------------------------------------
BM_full_vxsort<int32_t, 8>/4096         0.132 ms        0.132 ms         5341  3.21375ns
BM_full_vxsort<int32_t, 8>/8192         0.292 ms        0.292 ms         2495  3.56416ns
BM_full_vxsort<int32_t, 8>/16384        0.631 ms        0.631 ms         1145  3.84954ns
BM_full_vxsort<int32_t, 8>/32768         1.43 ms         1.43 ms          524  4.35009ns
BM_full_vxsort<int32_t, 8>/65536         3.21 ms         3.21 ms          232  4.89271ns
BM_full_vxsort<int32_t, 8>/131072        6.41 ms         6.40 ms          108  4.88355ns
BM_full_vxsort<int32_t, 8>/262144        13.8 ms         13.8 ms           51  5.26214ns
BM_full_vxsort<int32_t, 8>/524288        29.1 ms         29.0 ms           24  5.53438ns
BM_full_vxsort<int32_t, 8>/1048576       59.8 ms         59.7 ms           11  5.69466ns
```
