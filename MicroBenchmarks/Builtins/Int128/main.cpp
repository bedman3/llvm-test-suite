#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>


template <typename IntT = uint64_t, typename IntSqT = __uint128_t>
class integer_trade_size_variance {
public:
    IntT _sum_trade_size;
    IntT _trade_count;
    IntSqT _sum_sq_trade_size;
    integer_trade_size_variance() {
        reset();
    }

    void reset() {
        _sum_trade_size = 0;
        _trade_count = 0;
        _sum_sq_trade_size = 0;
    }

    void on_trade_update(IntT trade_size) {
        _sum_trade_size += trade_size;
        IntSqT ts = trade_size;
        _sum_sq_trade_size += ts * ts;
        ++_trade_count;
    }

    double get_variance() {
        if (_trade_count == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double trade_count_d = _trade_count;
        IntSqT sum_pow_2 = _sum_trade_size;
        sum_pow_2 *= sum_pow_2;
        IntSqT sum_pow_2_whole = (sum_pow_2 / _trade_count);
        double sum_pow_2_decimal = (sum_pow_2 - sum_pow_2_whole * _trade_count) / trade_count_d;
        return (_sum_sq_trade_size - sum_pow_2_whole - sum_pow_2_decimal) / trade_count_d;
    }

    void on_trade_expire(IntT trade_size) {
        _sum_trade_size -= trade_size;
        IntSqT ts = trade_size;
        _sum_sq_trade_size -= ts * ts;
        --_trade_count;
    }
};

template <typename FloatT>
class fp_variance {
    uint64_t _input_count;
    FloatT _mean;
    FloatT _second_order_stats;
public:
    fp_variance() {
        reset();
    }

    void reset() {
        _input_count = 0;
        _mean = 0;
        _second_order_stats = 0;
    }

    void on_update(FloatT val) {
        ++_input_count;
        FloatT delta = val - _mean;
        _mean += delta / _input_count;
        FloatT delta_2 = val - _mean;
        _second_order_stats += delta * delta_2;
    }

    void on_expire(FloatT val) {
        if (_input_count == 1) {
            reset();
        }
        --_input_count;
        FloatT delta = val - _mean;
        _mean -= delta / _input_count;
        FloatT delta_2 = val - _mean;
        _second_order_stats -= delta * delta_2;
    }

    FloatT get_variance() {
        return _second_order_stats / _input_count;   
    }
};

#include "benchmark/benchmark.h"

#if defined(__SIZEOF_INT128__)

namespace {

constexpr size_t kSampleSize = 1 << 20;

// Some implementations of <random> do not support __int128_t when it is
// available, so we make our own uniform_int_distribution-like type.
template <typename T>
class UniformIntDistribution128 {
 public:
  T operator()(std::mt19937& generator) {
    return (static_cast<T>(dist64_(generator)) << 64) | dist64_(generator);
  }

 private:
  using H = typename std::conditional<
              std::is_same<T, __int128_t>::value, int64_t, uint64_t>::type;
  std::uniform_int_distribution<H> dist64_;
};

// Generates uniformly pairs (a, b) of 128-bit integers such as a >= b.
template <typename T>
std::vector<std::pair<T, T>> GetRandomIntrinsic128SampleUniformDivisor() {
  std::vector<std::pair<T, T>> values;
  std::mt19937 random;
  UniformIntDistribution128<T> uniform_128;
  values.reserve(kSampleSize);
  for (size_t i = 0; i < kSampleSize; ++i) {
    T a = uniform_128(random);
    T b = uniform_128(random);
    values.emplace_back(std::max(static_cast<T>(2), std::max(a, b)),
                        std::max(static_cast<T>(2), std::min(a, b)));
  }
  return values;
}

template <typename T>
void BM_DivideIntrinsic128UniformDivisor(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleUniformDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize(values[i].first / values[i].second);
    i = (i + 1) % kSampleSize;
  }
}
// BENCHMARK_TEMPLATE(BM_DivideIntrinsic128UniformDivisor, __uint128_t);
// BENCHMARK_TEMPLATE(BM_DivideIntrinsic128UniformDivisor, __int128_t);

template <typename T>
void BM_RemainderIntrinsic128UniformDivisor(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleUniformDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize(values[i].first % values[i].second);
    i = (i + 1) % kSampleSize;
  }
}
// BENCHMARK_TEMPLATE(BM_RemainderIntrinsic128UniformDivisor, __uint128_t);
// BENCHMARK_TEMPLATE(BM_RemainderIntrinsic128UniformDivisor, __int128_t);

// Generates random pairs of (a, b) where a >= b, a is 128-bit and
// b is 64-bit.
template <typename T, typename H = typename std::conditional<
              std::is_same<T, __int128_t>::value, int64_t, uint64_t>::type>
std::vector<std::pair<T, H>> GetRandomIntrinsic128SampleSmallDivisor() {
  std::vector<std::pair<T, H>> values;
  std::mt19937 random;
  UniformIntDistribution128<T> uniform_int128;
  std::uniform_int_distribution<H> uniform_int64;
  values.reserve(kSampleSize);
  for (size_t i = 0; i < kSampleSize; ++i) {
    T a = uniform_int128(random);
    H b = std::max(H{2}, uniform_int64(random));
    values.emplace_back(std::max(a, static_cast<T>(b)), b);
  }
  return values;
}

template <typename T>
void BM_DivideIntrinsic128SmallDivisor(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize(values[i].first / values[i].second);
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK_TEMPLATE(BM_DivideIntrinsic128SmallDivisor, __uint128_t);
BENCHMARK_TEMPLATE(BM_DivideIntrinsic128SmallDivisor, __int128_t);

template <typename T>
void BM_RemainderIntrinsic128SmallDivisor(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize(values[i].first % values[i].second);
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK_TEMPLATE(BM_RemainderIntrinsic128SmallDivisor, __uint128_t);
BENCHMARK_TEMPLATE(BM_RemainderIntrinsic128SmallDivisor, __int128_t);

template <typename T>
void BM_128IntegerToDoubleConversion(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize((double)values[i].first);
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK_TEMPLATE(BM_128IntegerToDoubleConversion, __uint128_t);
BENCHMARK_TEMPLATE(BM_128IntegerToDoubleConversion, __int128_t);

template <typename T>
void BM_IntegerAggregatorUpdate(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  integer_trade_size_variance<uint64_t, T> aggregator;
  for (const auto _ : state) {
    aggregator.on_trade_update(values[i].second);
    benchmark::DoNotOptimize(aggregator._sum_trade_size);
    benchmark::DoNotOptimize(aggregator._trade_count);
    benchmark::DoNotOptimize(aggregator._sum_sq_trade_size);
    i = (i + 1) % kSampleSize;
  }
  benchmark::DoNotOptimize(aggregator.get_variance());
}
BENCHMARK_TEMPLATE(BM_IntegerAggregatorUpdate, __uint128_t);

template <typename T>
void BM_IntegerAggregatorUpdateAndGetVariance(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  integer_trade_size_variance<uint64_t, T> aggregator;
  for (const auto _ : state) {
    aggregator.on_trade_update(values[i].second);
    benchmark::DoNotOptimize(aggregator.get_variance());
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK_TEMPLATE(BM_IntegerAggregatorUpdateAndGetVariance, __uint128_t);

void BM_FloatingPointAggregatorUpdate(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<__uint128_t>();
  auto double_array = std::vector<double>();
  for (auto& pair : values) {
    double_array.push_back((double)pair.first);
  }
  size_t i = 0;
  fp_variance<double> aggregator;
  for (const auto _ : state) {
    aggregator.on_update(double_array[i]);
    i = (i + 1) % kSampleSize;
  }
  benchmark::DoNotOptimize(aggregator.get_variance());
}
BENCHMARK(BM_FloatingPointAggregatorUpdate);

void BM_FloatingPointAggregatorUpdateAndGetVariance(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<__uint128_t>();
  auto double_array = std::vector<double>();
  fp_variance<double> aggregator;
  for (auto& pair : values) {
    double_array.push_back((double)pair.first);
    aggregator.on_update((double)pair.first);
  }
  size_t i = 0;
  for (const auto _ : state) {
    benchmark::DoNotOptimize(aggregator.get_variance());
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK(BM_FloatingPointAggregatorUpdateAndGetVariance);

template <typename T>
void BM_BaseCase(benchmark::State& state) {
  auto values = GetRandomIntrinsic128SampleSmallDivisor<T>();
  size_t i = 0;
  for (const auto _ : state) {
    i = (i + 1) % kSampleSize;
  }
}
BENCHMARK_TEMPLATE(BM_BaseCase, __uint128_t);

}

// namespace

#endif

BENCHMARK_MAIN();
