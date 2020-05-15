#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <gcsort.h>

#include <random>
#include <algorithm>
#include <iterator>

namespace gcsort_tests {
    using testing::WhenSorted;
    using testing::ElementsAreArray;


    static std::vector<uint8_t *> generate_unique_ptrs_vec(size_t n) {
        std::vector<uint8_t *> pvec(1000);

        std::iota(pvec.begin(), pvec.end(), (uint8_t *) 0x1000);

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(pvec.begin(), pvec.end(), g);
        return pvec;
    }

    TEST(Sort, IntroSort) {

        auto v = generate_unique_ptrs_vec(1000);

        auto begin = v.data();
        auto end = v.data() + v.size() - 1;

        sort_introsort(begin, end);
        EXPECT_THAT(v, WhenSorted(ElementsAreArray(v)));
    }

}