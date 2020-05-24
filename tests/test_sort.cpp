 #include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <gcsort.h>
#include <introsort.h>
#include <bitonic_sort.h>

#include <random>
#include <algorithm>
#include <iterator>

namespace gcsort_tests {
    using testing::WhenSorted;
    using testing::ElementsAreArray;
    using testing::ValuesIn;


    static std::vector<uint8_t *> generate_unique_ptrs_vec(size_t n) {
        std::vector<uint8_t *> pvec(n);

        std::iota(pvec.begin(), pvec.end(), (uint8_t *) 0x1000);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(pvec.begin(), pvec.end(), g);
        return pvec;
    }


    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop, IntType step)
    {
      if (step == IntType(0))
      {
        throw std::invalid_argument("step for range must be non-zero");
      }

      std::vector<IntType> result;
      IntType i = start;
      while ((step > 0) ? (i <= stop) : (i > stop))
      {
        result.push_back(i);
        i += step;
      }

      return result;
    }

    TEST(Sort, IntroSort) {
        auto v = generate_unique_ptrs_vec(1000);

        auto begin = v.data();
        auto end = v.data() + v.size() - 1;

        sort_introsort(begin, end);
        EXPECT_THAT(v, WhenSorted(ElementsAreArray(v)));
    }



    // A new one of these is create for each test
    class SortTest : public testing::TestWithParam<int>
    {
    public:
      virtual void SetUp(){}
      virtual void TearDown(){}
    };

  struct PrintValue {
    template <class ParamType>
    std::string operator()( const testing::TestParamInfo<ParamType>& info ) const
    {
      auto v = static_cast<int>(info.param);
      return std::to_string(v);
    }
  };

    INSTANTIATE_TEST_SUITE_P(
        BitonicSizes,
        SortTest,
        ValuesIn(range(4, 128, 4)),
                           PrintValue());
    
    TEST_P(SortTest, BitonicSort) {
      auto v = generate_unique_ptrs_vec(GetParam());
      auto begin = v.data();

      gcsort::smallsort::bitonic_sort_int64_t((int64_t *) begin, GetParam());

      EXPECT_THAT(v, WhenSorted(ElementsAreArray(v)));
    }

}