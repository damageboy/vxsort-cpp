#ifndef VXSORT_FULLSORT_TEST_H
#define VXSORT_FULLSORT_TEST_H

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include "../util.h"
#include "../sort_fixtures.h"
#include "../test_isa.h"
#include "vxsort.h"

namespace vxsort_tests {
using namespace vxsort::types;
using ::vxsort::vector_machine;

template <typename T, i32 Unroll, vector_machine M>
void vxsort_pattern_test(SortPattern, usize size, T first_value, T stride) {
    VXSORT_TEST_ISA();

    auto V = unique_values<T>(size, first_value, stride);

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll>();
    sorter.sort(begin, end);

    std::sort(v_copy.begin(), v_copy.end());
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

template <typename T, i32 Unroll, int Shift, vector_machine M>
void vxsort_hinted_test(std::vector<T>& V, T min_value, T max_value) {
    VXSORT_TEST_ISA();

    auto v_copy = std::vector<T>(V);
    auto begin = V.data();
    auto end = V.data() + V.size() - 1;

    auto sorter = ::vxsort::vxsort<T, M, Unroll, Shift>();
    sorter.sort(begin, end, min_value, max_value);

    std::sort(v_copy.begin(), v_copy.end());
    usize size = v_copy.size();
    for (usize i = 0; i < size; ++i) {
        if (v_copy[i] != V[i]) {
            GTEST_FAIL() << fmt::format("value at idx #{}  {} != {}", i, v_copy[i], V[i]);
        }
    }
}

static inline std::vector<SortPattern> test_patterns() {
    return {
            SortPattern::unique_values,
            SortPattern::shuffled_16_values,
            SortPattern::all_equal,
    };
}

template <typename T>
struct SortTestParams2 {
public:
    SortTestParams2(SortPattern pattern, usize size, i32 slack, T first_value, T value_stride)
            : Pattern(pattern), Size(size), Slack(slack), FirstValue(first_value), ValueStride(value_stride) {}
    SortPattern Pattern;
    usize Size;
    i32 Slack;
    T FirstValue;
    T ValueStride;
};

class VxSortFixture : public testing::Test {
public:
    using FunctionType = std::function<void()>;
    explicit VxSortFixture(FunctionType fn) : _fn(std::move(fn)) {}

    VxSortFixture(VxSortFixture const&) = delete;

    void TestBody() override {
        _fn();
    }

private:
    FunctionType _fn;
};

template <class Lambda, class... Args>
void RegisterSingleTest(const char* test_suite_name, const char* test_name,
                        const char* type_param, const char* value_param,
                        const char* file, int line,
                        Lambda&& fn, Args&&... args) {

    testing::RegisterTest(
            test_suite_name, test_name, type_param, value_param,
            file, line,
            [=]() mutable -> testing::Test* { return new VxSortFixture(
                    [=]() mutable { fn(args...); });
            });
}


template <vector_machine M, i32 U, typename T>
void register_fullsort_benchmarks(usize start, usize stop, usize step, T first_value, T value_stride) {
    if (step == 0) {
        throw std::invalid_argument("step for range must be non-zero");
    }

    if constexpr (U >= 2) {
        register_fullsort_benchmarks<M, U / 2, T>(start, stop, step, first_value, value_stride);
    }

    using VM = vxsort::vxsort_machine_traits<T, M>;

    // Test "slacks" are defined in terms of number of elements in the primitive size (T)
    // up to the number of such elements contained in one vector type (VM::TV)
    constexpr i32 slack = sizeof(typename VM::TV) /  sizeof(T);
    static_assert(slack > 1);

    std::vector<SortTestParams2<T>> tests;
    size_t i = start;
    for (auto p : test_patterns()) {
        while ((step > 0) ? (i <= stop) : (i > stop)) {
            for (auto j : range<int>(-slack, slack, 1)) {
                if ((i64)i + j <= 0)
                    continue;
                tests.push_back(SortTestParams2<T>(p, i, j, first_value, value_stride));
            }
            i *= step;
        }
    }

    for (auto p : tests) {
        auto *test_type = get_canonical_typename<T>();

        auto test_size = p.Size + p.Slack;
        auto test_name = fmt::format("vxsort_pattern_test<{}, {}, {}>/{}/{}", test_type, U,
                                     magic_enum::enum_name(M), magic_enum::enum_name(p.Pattern), test_size);

        RegisterSingleTest(
                "fullsort", test_name.c_str(), nullptr,
                std::to_string(p.Size).c_str(),
                __FILE__, __LINE__,
                vxsort_pattern_test<T, U, M>, p.Pattern, test_size, p.FirstValue, p.ValueStride);
    }
}

}

#endif  // VXSORT_FULLSORT_TEST_H
