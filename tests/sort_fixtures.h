#ifndef VXSORT_SORT_FIXTURES_H
#define VXSORT_SORT_FIXTURES_H

#include "gtest/gtest.h"
#include "stats/vxsort_stats.h"
#include "util.h"

#include <array>
#include <algorithm>
#include <iterator>
#include <random>
#include <stdlib.h>

namespace vxsort_tests {
using namespace vxsort::types;
using testing::ValuesIn;
using testing::Types;

/// @brief This sort fixture 
/// @tparam T 
/// @tparam AlignTo 
template <typename T>
struct SortTestParams {
public:
    SortPattern Pattern;
    usize Size;
    i32 Slack;
    T FirstValue;
    T ValueStride;


    SortTestParams(SortPattern pattern, size_t size, int slack, T first_value, T value_stride)
        : Pattern(pattern), Size(size), Slack(slack), FirstValue(first_value), ValueStride(value_stride) {}

    /**
     * Generate sorting problems "descriptions"
     * @param patterns - the sort patterns to test with
     * @param start - start value for the size parameter
     * @param stop - stop value for the size paraameter
     * @param step - the step/multiplier for the size parameter
     * @param slack - the slack parameter used to generate ranges of problem sized around a base value
     * @param first_value - the smallest value in each test array
     * @param value_stride - the minimal jump between array elements
     * @return
     */
    static std::vector<SortTestParams> gen_mult(std::vector<SortPattern> patterns, usize start, usize stop, usize step, i32 slack, T first_value, T value_stride) {
        if (step == 0) {
            throw std::invalid_argument("step for range must be non-zero");
        }

        std::vector<SortTestParams> result;
        size_t i = start;
        for (auto p : patterns) {
            while ((step > 0) ? (i <= stop) : (i > stop)) {
                for (auto j : range<int>(-slack, slack, 1)) {
                    if ((i64)i + j <= 0)
                        continue;
                    result.push_back(SortTestParams(p, i, j, first_value, value_stride));
                }
                i *= step;
            }
        }
        return result;
    }

    /**
     * Generate sorting problems "descriptions"
     * @param pattern - the sort pattern to test with
     * @param start - start value for the size parameter
     * @param stop - stop value for the size paraameter
     * @param step - the step/multiplier for the size parameter
     * @param slack - the slack parameter used to generate ranges of problem sized around a base value
     * @param first_value - the smallest value in each test array
     * @param value_stride - the minimal jump between array elements
     * @return
     */
    static auto gen_mult(SortPattern pattern, usize start, usize stop, usize step, i32 slack, T first_value, T value_stride) {
        return gen_mult(std::vector<SortPattern>{pattern}, start, stop, step, slack,
                                 first_value, value_stride);
    }

    /**
     * Generate sorting problems "descriptions"
     * @param patterns - the sort patterns to test with
     * @param start - start value for the size parameter
     * @param stop - stop value for the size paraameter
     * @param step - the step/multiplier for the size parameter
     * @param slack - the slack parameter used to generate ranges of problem sized around a base value
     * @param first_value - the smallest value in each test array
     * @param value_stride - the minimal jump between array elements
     * @return
     */
    static std::vector<SortTestParams> gen_step(std::vector<SortPattern> patterns, usize start, usize stop, usize step, i32 slack, T first_value, T value_stride) {
        if (step == 0) {
            throw std::invalid_argument("step for range must be non-zero");
        }

        std::vector<SortTestParams> result;
        size_t i = start;
        for (auto p : patterns) {
            while ((step > 0) ? (i <= stop) : (i > stop)) {
                for (auto j : range<int>(-slack, slack, 1)) {
                    if ((i64)i + j <= 0)
                        continue;
                    result.push_back(SortTestParams(p, i, j, first_value, value_stride));
                }
                i += step;
            }
        }
        return result;
    }

    /**
     * Generate sorting problems "descriptions"
     * @param pattern - the sort pattern to test with
     * @param start - start value for the size parameter
     * @param stop - stop value for the size paraameter
     * @param step - the step for the size parameter
     * @param slack - the slack parameter used to generate ranges of problem sized around a base value
     * @param first_value - the smallest value in each test array
     * @param value_stride - the minimal jump between array elements
     * @return
     */
    static auto gen_step(SortPattern pattern, usize start, usize stop, usize step, i32 slack, T first_value, T value_stride) {
        return gen_step(std::vector<SortPattern>{pattern}, start, stop, step, slack,
                        first_value, value_stride);
    }
};

template <typename T, int AlignTo = 0>
struct ParametrizedSortFixture : public testing::TestWithParam<SortTestParams<T>> {
protected:
    std::vector<T> V;

public:
    virtual void SetUp() {
        testing::TestWithParam<SortTestParams<T>>::SetUp();
        auto p = this->GetParam();
        auto v = unique_values(p.Size + p.Slack, p.FirstValue, p.ValueStride);
    }
    virtual void TearDown() {
#ifdef VXSORT_STATS
        vxsort::print_all_stats();
        vxsort::reset_all_stats();
#endif
    }
};

template <typename T>
struct PrintSortTestParams {
    std::string operator()(const testing::TestParamInfo<SortTestParams<T>>& info) const {
        return std::to_string(info.param.Size + info.param.Slack);
    }
};

}

#endif  // VXSORT_SORT_FIXTURES_H
