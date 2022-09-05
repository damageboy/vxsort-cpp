#ifndef VXSORT_DEFS_H
#define VXSORT_DEFS_H

#if _MSC_VER
#ifdef _M_X86
#define ARCH_X86
#endif
#ifdef _M_X64
#define ARCH_X64
#endif
#ifdef _M_ARM64
#define ARCH_ARM
#endif
#else
#ifdef __i386__
#define ARCH_X86
#endif
#ifdef __amd64__
#define ARCH_X64
#endif
#ifdef __arm__
#define ARCH_ARM
#endif
#endif

#ifdef _MSC_VER
#ifdef __clang__
#define mess_up_cmov()
#define INLINE __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))
#else
// MSVC
#include <intrin.h>
#define mess_up_cmov() _ReadBarrier();
#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define VXSORT_COMPILER_MSVC 1
#endif
#else
// GCC + Clang
#define mess_up_cmov()
#define INLINE __attribute__((always_inline))
#define NOINLINE __attribute__((noinline))
#endif

#include <cstdint>
#include <sys/types.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace vxsort {

template <class... E>
constexpr bool always_false = false;
constexpr bool is_powerof2(int v) {
    return v && ((v & (v - 1)) == 0);
}

namespace types {
    using i8  = int8_t;
    using u8  = uint8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using f32 = float;
    using f64 = double;
    using isize = ssize_t;
    using usize = size_t;
}

} // namespace vxsort

#endif  // VXSORT_DEFS_H
