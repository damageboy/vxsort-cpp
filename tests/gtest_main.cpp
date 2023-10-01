#include <cstdio>
#include <backward.hpp>

#include "gtest/gtest.h"

namespace vxsort_tests {


    void register_fullsort_avx2_i_tests();
    void register_fullsort_avx512_i_tests();
    void register_fullsort_avx2_u_tests();
    void register_fullsort_avx2_f_tests();
    void register_fullsort_avx512_u_tests();
    void register_fullsort_avx512_f_tests();

    void register_fullsort_test_matrix() {

#ifdef VXSORT_TEST_AVX2_I
        register_fullsort_avx2_i_tests();
#endif
#ifdef VXSORT_TEST_AVX2_U
        register_fullsort_avx2_u_tests();
#endif
#ifdef VXSORT_TEST_AVX2_F
        register_fullsort_avx2_f_tests();
#endif
#ifdef VXSORT_TEST_AVX512_I
        register_fullsort_avx512_i_tests();
#endif
#ifdef VXSORT_TEST_AVX512_U
        register_fullsort_avx512_u_tests();
#endif
#ifdef VXSORT_TEST_AVX512_F
        register_fullsort_avx512_f_tests();
#endif
    }
}  // namespace vxsort_tests

GTEST_API_ int main(int argc, char **argv) {
    backward::SignalHandling sh;

    testing::InitGoogleTest(&argc, argv);

    vxsort_tests::register_fullsort_test_matrix();

    return RUN_ALL_TESTS();
}
