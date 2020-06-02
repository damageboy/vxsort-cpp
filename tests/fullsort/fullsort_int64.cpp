#include "gtest/gtest.h"
#include "../fixtures.h"
#include "fullsort.h"

#include "vxsort_targets_enable.h"

#include <vxsort.int64_t.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

using gcsort::vector_machine;

static int64_t peters_pointers_of_doom[] = {
/* 1a8ff59e050 */ 0x1a8df0f4bf8, 0x1a8df0f4c60, 0x1a8df7bfb68, 0x1a8df7c0440, 0x1a8df7c0d00, 0x1a8df7bbdf8, 0x1a8df7bc6d0, 0x1a8df7bcf90, 0x1a8df7bd850, 0x1a8df7be128, 0x1a8df7be9e8, 0x1a8df7bf2a8,
/* 1a8ff59e0b0 */ 0x1a8df0f4cc8, 0x1a8df7bfc48, 0x1a8df7c0520, 0x1a8df7c0de0, 0x1a8df7bbed8, 0x1a8df7bc7b0, 0x1a8df7bd070, 0x1a8df7bd930, 0x1a8df7be208, 0x1a8df7beac8, 0x1a8df7bf388, 0x1a8df0f4d30,
/* 1a8ff59e110 */ 0x1a8df7bfd28, 0x1a8df7c0600, 0x1a8df7c0ec0, 0x1a8df7bbfd0, 0x1a8df7bc890, 0x1a8df7bd150, 0x1a8df7bda10, 0x1a8df7be2e8, 0x1a8df7beba8, 0x1a8df7bf468, 0x1a8df0f4d98, 0x1a8df7bfe08,
/* 1a8ff59e170 */ 0x1a8df7c06e0, 0x1a8df7c0fb8, 0x1a8df7bc0b0, 0x1a8df7bc970, 0x1a8df7bd230, 0x1a8df7bdaf0, 0x1a8df7be3c8, 0x1a8df7bec88, 0x1a8df7bf548, 0x1a8df0f4e00, 0x1a8df7bfee8, 0x1a8df7c07c0,
/* 1a8ff59e1d0 */ 0x1a8df7bb8b8, 0x1a8df7bc190, 0x1a8df7bca50, 0x1a8df7bd310, 0x1a8df7bdbd0, 0x1a8df7be4a8, 0x1a8df7bed68, 0x1a8df7bf628, 0x1a8df0f4e68, 0x1a8df7bffe0, 0x1a8df7c08a0, 0x1a8df7bb998,
/* 1a8ff59e230 */ 0x1a8df7bc270, 0x1a8df7bcb30, 0x1a8df7bd3f0, 0x1a8df7bdcb0, 0x1a8df7be588, 0x1a8df7bee48, 0x1a8df7bf708, 0x1a8df0f4ed0, 0x1a8df7c00c0, 0x1a8df7c0980, 0x1a8df7bba78, 0x1a8df7bc350,
/* 1a8ff59e290 */ 0x1a8df7bcc10, 0x1a8df7bd4d0, 0x1a8df7bdd90, 0x1a8df7be668, 0x1a8df7bef28, 0x1a8df7bf7e8, 0x1a8df0f4f38, 0x1a8df7c01a0, 0x1a8df7c0a60, 0x1a8df7bbb58, 0x1a8df7bc430, 0x1a8df7bccf0,
/* 1a8ff59e2f0 */ 0x1a8df7bd5b0, 0x1a8df7bde70, 0x1a8df7be748, 0x1a8df7bf008, 0x1a8df7bf8c8, 0x1a8df0f4fa0, 0x1a8df7c0280, 0x1a8df7c0b40, 0x1a8df7bbc38, 0x1a8df7bc510, 0x1a8df7bcdd0, 0x1a8df7bd690,
/* 1a8ff59e350 */ 0x1a8df7bdf68, 0x1a8df7be828, 0x1a8df7bf0e8, 0x1a8df7bf9a8, 0x1a8df0f5008, 0x1a8df7c0360, 0x1a8df7c0c20, 0x1a8df7bbd18, 0x1a8df7bc5f0, 0x1a8df7bceb0, 0x1a8df7bd770, 0x1a8df7be048,
/* 1a8ff59e3b0 */ 0x1a8df7be908, 0x1a8df7bf1c8, 0x1a8df7bfa88, 0x1a8df0f3360, 0x1a8df0e13a8, 0x1a8df0f4860, 0x1a8df0f2e68, 0x1a8df0f2eb0, 0x1a8df0f2e88, 0x1a8df0f46d0, 0x1a8df0f4af8, 0x1a8df0f3f98,
/* 1a8ff59e410 */ 0x1a8df0f3820, 0x1a8df0f37c8, 0x1a8df0f2e48, 0x1a8df0e1328, 0x1a8df0e12a8, 0x1a8df0e1228, 0x1a8df0e11a8, 0x1a8df0e1128, 0x1a8df0e10a8, 0x1a8df0e1048, 0x1a8df0f32a0, 0x1a8df0f32c0,
/* 1a8ff59e470 */ 0x1a8df0f2d08, 0x1a8df0f2d58, 0x1a8df0e1730, 0x1a8df0f2d80, 0x1a8df0e13c0, 0x1a8df0e1470, 0x1a8df0e1638, 0x1a8df0e1670, 0x1a8df0e1900, 0x1a8df0e1968, 0x1a8df0f2da8, 0x1a8df0f2dd0,
/* 1a8ff59e4d0 */ 0x1a8df0f2f20, 0x1a8df0f2f50, 0x1a8df0f3030, 0x1a8df0f3080, 0x1a8df0f3190, 0x1a8df0f31e8, 0x1a8df0f3220, 0x1a8df0f3240, 0x1a8df0f3280, 0x1a8df0e1758, 0x1a8df0f4808, 0x1a8df0f3700,
/* 1a8ff59e530 */ 0x1a8df0f3780, 0x1a8df0f37f0, 0x1a8df0f37b0, 0x1a8df0f3808, 0x1a8df0e1770, 0x1a8df0f3720, 0x1a8df0f4780, 0x1a8df0f47a0, 0x1a8df0f4848, 0x1a8df0f4b20, 0x1a8df0f4420, 0x1a8df0f44d8,
/* 1a8ff59e590 */ 0x1a8df0f4608, 0x1a8df0f4678, 0x1a8df0f4690, 0x1a8df0f4530, 0x1a8df0f46f8, 0x1a8df0f4768, 0x1a8df0f47c8, 0x1a8df0f45c8, 0x1a8df0f4548, 0x1a8df0f4560, 0x1a8df0f33c8, 0x1a8df0f3518,
/* 1a8ff59e5f0 */ 0x1a8df0f42d0, 0x1a8df0f4340, 0x1a8df0f4358, 0x1a8df0f3550, 0x1a8df0f4398, 0x1a8df0f4408, 0x1a8df0f4ab8, 0x1a8df0f3f68, 0x1a8df0f3568, 0x1a8df0f3580, 0x1a8df0e14c0, 0x1a8df0e13d8,
/* 1a8ff59e650 */ 0x1a8df0f2b70, 0x1a8df0f2bc8, 0x1a8df0f2c48, 0x1a8df0f2c90, 0x1a8df0f3378, 0x1a8df0f33a8, 0x1a8df0f35e8, 0x1a8df0f3608, 0x1a8df0f3680, 0x1a8df0f3740, 0x1a8df0f3760, 0x1a8df0f3848,
/* 1a8ff59e6b0 */ 0x1a8df0f38c0, 0x1a8df0f3950, 0x1a8df0f39b0, 0x1a8df0f3a40, 0x1a8df0f3ae0, 0x1a8df0f3b80, 0x1a8df0f3c08, 0x1a8df0f3c58, 0x1a8df0f3ca8, 0x1a8df0f3d00, 0x1a8df0f3d50, 0x1a8df0f3dd8,
/* 1a8ff59e710 */ 0x1a8df0f3e50, 0x1a8df0f3ed8, 0x1a8df0f3fd8, 0x1a8df0f4040, 0x1a8df0f40e8, 0x1a8df0f4178, 0x1a8df0f41c8, 0x1a8df0f4238, 0x1a8df0f48d0, 0x1a8df0f4920, 0x1a8df0f49a0, 0x1a8df0f4ba0,
/* 1a8ff59e770 */ 0x1a8df0e1860, 0x1a8df0e1888, 0x1a8df0e18b0, 0x1a8df0e18d8, 0x1a8df0f2d30, 0x1a8df0f4a50, 0x1a8df0f4a90, };

TEST(PetersCrash, Crash1) {
    const auto n = sizeof(peters_pointers_of_doom) / sizeof(int64_t);
    auto begin = peters_pointers_of_doom;
    auto end = begin + n - 1;
    auto sorter = gcsort::vxsort<int64_t, vector_machine::AVX2>();
    sorter.sort(begin, end);

    EXPECT_THAT(peters_pointers_of_doom, WhenSorted(ElementsAreArray(peters_pointers_of_doom)));
}

#ifdef _WIN32
TEST(PetersCrash, Crash2) {
  const auto ALIGN_PAGE = 1 << 12;
  const auto desired_pointer = 0x1a8ff59e050;
  VirtualAlloc(reinterpret_cast<void*>(desired_pointer & ~(ALIGN_PAGE - 1)), 1 << 12, MEM_COMMIT | MEM_RESERVE,
               PAGE_READWRITE);
  memcpy((void*)desired_pointer, peters_pointers_of_doom,
         sizeof(peters_pointers_of_doom));
  const auto n = sizeof(peters_pointers_of_doom) / sizeof(int64_t);
  auto begin = reinterpret_cast<int64_t*>(desired_pointer);
  auto end = begin + n - 1;
  auto sorter = gcsort::vxsort<int64_t, vector_machine::AVX2>();
  sorter.sort(begin, end);

  auto result = std::vector<int64_t>(begin, end+1);

  EXPECT_THAT(result,
              WhenSorted(ElementsAreArray(result)));
}
#endif

struct FullSortTest_i64 : public SortWithSlackTest<int64_t> {};

INSTANTIATE_TEST_SUITE_P(FullSortSizes,
                         FullSortTest_i64,
                         ValuesIn(SizeAndSlack::generate(10, 1000000, 10, FullSortTest_i64::VectorElements*2)),
                         PrintSizeAndSlack());

TEST_P(FullSortTest_i64, VxSortAVX2_1)  { perform_vxsort_test<int64_t,  1, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_2)  { perform_vxsort_test<int64_t,  2, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_4)  { perform_vxsort_test<int64_t,  4, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_8)  { perform_vxsort_test<int64_t,  8, vector_machine::AVX2>(V); }
TEST_P(FullSortTest_i64, VxSortAVX2_12) { perform_vxsort_test<int64_t, 12, vector_machine::AVX2>(V); }

TEST_P(FullSortTest_i64, VxSortAVX512_1)  { perform_vxsort_test<int64_t,  1, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i64, VxSortAVX512_2)  { perform_vxsort_test<int64_t,  2, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i64, VxSortAVX512_4)  { perform_vxsort_test<int64_t,  4, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i64, VxSortAVX512_8)  { perform_vxsort_test<int64_t,  8, vector_machine::AVX512>(V); }
TEST_P(FullSortTest_i64, VxSortAVX512_12) { perform_vxsort_test<int64_t, 12, vector_machine::AVX512>(V); }

}

#include "vxsort_targets_disable.h"