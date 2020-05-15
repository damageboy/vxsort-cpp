#include "benchmark/benchmark.h"

using namespace std;


int main(int argc, char** argv)
{
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}