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

template <typename T, int AlignTo = 0>
struct SortFixture : public testing::TestWithParam<int> {
protected:
    std::vector<T> V;

public:
    virtual void SetUp() {
        V = std::vector<T>(GetParam());
        generate_unique_values_vec(V, (T)0x1000, (T)0x1);
    }
    virtual void TearDown() {
    }
};

struct PrintValue {
    template <class ParamType>
    std::string operator()(const testing::TestParamInfo<ParamType>& info) const {
        auto v = static_cast<int>(info.param);
        return std::to_string(v);
    }
};

template <typename T>
struct SizeAndSlack {
public:
    usize Size;
    i32 Slack;
    T FirstValue;
    T ValueStride;
    bool Randomize;

    SizeAndSlack(size_t size, int slack, T first_value, T value_stride, bool randomize)
        : Size(size), Slack(slack), FirstValue(first_value), ValueStride(value_stride), Randomize(randomize) {}

    /**
     * Generate sorting problems "descriptions"
     * @param start
     * @param stop
     * @param step
     * @param slack
     * @param first_value - the smallest value in each test array
     * @param value_stride - the minimal jump between array elements
     * @param randomize - should the problem array contents be randomized, defaults to true
     * @return
     */
    static std::vector<SizeAndSlack> generate(size_t start, size_t stop, size_t step, int slack, T first_value, T value_stride, bool randomize = true) {
        if (step == 0) {
            throw std::invalid_argument("step for range must be non-zero");
        }

        std::vector<SizeAndSlack> result;
        size_t i = start;
        while ((step > 0) ? (i <= stop) : (i > stop)) {
            for (auto j : range<int>(-slack, slack, 1)) {
                if ((i64)i + j <= 0)
                    continue;
                result.push_back(SizeAndSlack(i, j, first_value, value_stride, randomize));
            }
            i *= step;
        }
        return result;
    }
};

template <typename T, int AlignTo = 0>
struct SortWithSlackFixture : public testing::TestWithParam<SizeAndSlack<T>> {
protected:
    std::vector<T> V;

public:
    virtual void SetUp() {
        testing::TestWithParam<SizeAndSlack<T>>::SetUp();
        auto p = this->GetParam();
        V = std::vector<T>(p.Size + p.Slack);
        generate_unique_values_vec(V, p.FirstValue, p.ValueStride, p.Randomize);
    }
    virtual void TearDown() {
#ifdef VXSORT_STATS
        vxsort::print_all_stats();
        vxsort::reset_all_stats();
#endif
    }
};

template <typename T>
struct PrintSizeAndSlack {
    std::string operator()(const testing::TestParamInfo<SizeAndSlack<T>>& info) const {
        return std::to_string(info.param.Size + info.param.Slack);
    }
};

template <typename T>
struct SizeAndStride {
public:
    usize Size;
    T FirstValue;
    T ValueStride;
    bool Randomize;

    SizeAndStride(size_t size, T first_value, T value_stride, bool randomize)
        : Size(size), FirstValue(first_value), ValueStride(value_stride), Randomize(randomize) {}

    static std::vector<SizeAndStride> generate(size_t size, T stride_start, T stride_stop, T first_value, bool randomize = true) {
        std::vector<SizeAndStride> result;
        for (auto j : multiply_range<T>(stride_start, stride_stop, 2)) {
            result.push_back(SizeAndStride(size, first_value, j, randomize));
        }
        return result;
    }
};

template <typename T, i32 AlignTo = 0>
struct SortWithStrideFixture : public testing::TestWithParam<SizeAndStride<T>> {
protected:
    std::vector<T> V;
    T MinValue;
    T MaxValue;

public:
    virtual void SetUp() {
        testing::TestWithParam<SizeAndStride<T>>::SetUp();
        auto p = this->GetParam();
        V = std::vector<T>(p.Size);
        generate_unique_values_vec(V, p.FirstValue, p.ValueStride, p.Randomize);
        MinValue = p.FirstValue;
        MaxValue = MinValue + p.Size * p.ValueStride;
        if (MinValue > MaxValue)
            throw std::invalid_argument("stride is generating an overflow");
    }
    virtual void TearDown() {
#ifdef VXSORT_STATS
        vxsort::print_all_stats();
        vxsort::reset_all_stats();
#endif
    }
};

template <typename T>
struct PrintSizeAndStride {
    std::string operator()(const testing::TestParamInfo<SizeAndStride<T>>& info) const {
        return std::to_string(info.param.ValueStride);
    }
};
}

#endif  // VXSORT_SORT_FIXTURES_H
