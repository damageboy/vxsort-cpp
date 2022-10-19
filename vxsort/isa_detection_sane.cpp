#include "isa_detection.h"

#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
using namespace cpu_features;
static const X86Features features = GetX86Info().features;
static const bool has_avx2 = CPU_FEATURES_COMPILED_X86_AVX2 || (features.avx2 && features.avx && features.popcnt && features.bmi2);
static const bool has_avx512_32_64 = CPU_FEATURES_COMPILED_X86_AVX2 || (features.avx512f && features.avx512dq && features.avx512bw && features.popcnt);
static const bool has_avx512_16 = has_avx512_32_64 && features.avx512vbmi2;
//static const bool has_avx512_16_fp16 = has_avx512_16 && features.avx512_fp16;
#elif defined(CPU_FEATURES_ARCH_ARM)
#include "cpuinfo_arm.h"
using namespace cpu_features;
static const ArmFeatures features = GetArmInfo().features;
static const bool has_neon = CPU_FEATURES_COMPILED_ANY_ARM_NEON || features.neon;
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#include "cpuinfo_aarch64.h"
using namespace cpu_features;
static const Aarch64Features features = GetAarch64Info().features;
static const bool has_neon = CPU_FEATURES_COMPILED_ANY_ARM_NEON || features.asimd;
static const bool has_sve = CPU_FEATURES_COMPILED_ANY_ARM_NEON || features.sve;

#elif defined(CPU_FEATURES_ARCH_MIPS)
#include "cpuinfo_mips.h"
#elif defined(CPU_FEATURES_ARCH_PPC)
#include "cpuinfo_ppc.h"
#endif

namespace vxsort {

bool init_isa_detection() {
    return true;
}

extern bool supports_vector_machine(vector_machine m)
{
    switch (m) {
        case NONE:
            return true;
#if defined(CPU_FEATURES_ARCH_X86)
        case AVX2:
            return has_avx2;
        case AVX512:
            return has_avx512_32_64;
#endif
#if defined(CPU_FEATURES_ARCH_ANY_ARM)
        case NEON:
            return has_neon;
#endif
#if defined(CPU_FEATURES_ARCH_AARCH64)
        case SVE:
            return has_sve;
#endif
        default:
            break;
    }
    return false;
}

template<>
bool supports_vector_machine<AVX512>(usize width) {
    switch (width) {
        case 16:
            // We require AVX512VBMI2 for 16-bit partitioning
            // since we use the _mm512_mask_compressstoreu_epi16 intrinsic
            return has_avx512_16;
        case 32:
        case 64:
            return has_avx512_32_64;
        default:
            break;
    }
    return false;
}
} // namespace vxsort
