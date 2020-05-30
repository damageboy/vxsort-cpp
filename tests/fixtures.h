#ifndef GCSORT_FIXTURES_H
#define GCSORT_FIXTURES_H

#include "util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util.h"

#include <algorithm>
#include <iterator>
#include <random>

namespace gcsort_tests {
using testing::ElementsAreArray;
using testing::ValuesIn;
using testing::WhenSorted;
using testing::Types;

template <typename T>
struct SortTest : public testing::TestWithParam<int> {
 protected:
  std::vector<T> V;

 public:
  static constexpr int VectorElements = 32 / sizeof(T);
  virtual void SetUp() {
    V = std::vector<T>(GetParam());
    generate_unique_ptrs_vec(V);
  }
  virtual void TearDown() {}
};

struct PrintValue {
  template <class ParamType>
  std::string operator()(const testing::TestParamInfo<ParamType>& info) const {
    auto v = static_cast<int>(info.param);
    return std::to_string(v);
  }
};

struct SizeAndSlack {
 public:
  size_t Size;
  int Slack;

  SizeAndSlack(size_t size, int slack) : Size(size), Slack(slack) {}
  static std::vector<SizeAndSlack> generate(size_t start,
                                            size_t stop,
                                            size_t step,
                                            int slack) {
    if (step == 0) {
      throw std::invalid_argument("step for range must be non-zero");
    }

    std::vector<SizeAndSlack> result;
    size_t i = start;
    while ((step > 0) ? (i <= stop) : (i > stop)) {
      for (auto j : range<int>(-slack, slack, 1)) {
        if ((int64_t) i + j <= 0)
          continue;
        result.push_back(SizeAndSlack(i, j));
      }
      i *= step;
    }
    return result;
  }
};

template <typename T>
struct SortWithSlackTest : public testing::TestWithParam<SizeAndSlack> {
 protected:
  std::vector<T> V;

 public:
  static constexpr int VectorElements = 32 / sizeof(T);
  virtual void SetUp() {
    auto p = GetParam();
    V = std::vector<T>(p.Size + p.Slack);
    generate_unique_ptrs_vec(V);
  }
  virtual void TearDown() {}
};

struct PrintSizeAndSlack {
  std::string operator()(
      const testing::TestParamInfo<SizeAndSlack>& info) const {
    return std::to_string(info.param.Size + info.param.Slack);
  }
};
}

#endif  // GCSORT_FIXTURES_H
