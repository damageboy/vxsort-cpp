#ifndef VXSORT_SORT_FIXTURES_H
#define VXSORT_SORT_FIXTURES_H

#include "gtest/gtest.h"
#include "stats/vxsort_stats.h"
#include "test_vectors.h"

#include <array>
#include <algorithm>
#include <iterator>
#include <random>
#include <stdlib.h>

namespace vxsort_tests {
using namespace vxsort::types;
using testing::ValuesIn;
using testing::Types;

class VxSortLambdaFixture : public testing::Test {
public:
    using FunctionType = std::function<void()>;
    explicit VxSortLambdaFixture(FunctionType fn) : _fn(std::move(fn)) {}

    VxSortLambdaFixture(VxSortLambdaFixture const&) = delete;

    void TestBody() override {
        _fn();
    }

private:
    FunctionType _fn;
};

template <class Lambda, class... Args>
void RegisterSingleLambdaTest(const char* test_suite_name, const char* test_name,
                              const char* type_param, const char* value_param,
                              const char* file, int line,
                              Lambda&& fn, Args&&... args) {

    testing::RegisterTest(
            test_suite_name, test_name, type_param, value_param,
            file, line,
            [=]() mutable -> testing::Test* { return new VxSortLambdaFixture(
                    [=]() mutable { fn(args...); });
            });
}
}

#endif  // VXSORT_SORT_FIXTURES_H
