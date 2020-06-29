#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push (__attribute__((target("avx512f,avx512dq,avx512bw,avx512vbmi2"))), apply_to = any(function))
#else
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vbmi2")
#endif
#endif
