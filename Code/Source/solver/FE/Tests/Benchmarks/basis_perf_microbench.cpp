/*
 * Standalone Basis microbenchmark with allocation counting.
 *
 * Usage:
 *   basis_perf_microbench [iterations]
 *
 * The benchmark warms each case before enabling allocation counting, so
 * allocation_per_call highlights repeated hidden allocations in hot paths.
 */

#include "FE/Assembly/BatchedProjection.h"
#include "FE/Basis/BatchEvaluator.h"
#include "FE/Basis/BernsteinBasis.h"
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/BubbleBasis.h"
#include "FE/Basis/CompatibleTensorVectorBasis.h"
#include "FE/Basis/HermiteBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/NURBSTensorBasis.h"
#include "FE/Basis/SpectralBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/TensorBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

#define SVMP_STRINGIFY_DETAIL(x) #x
#define SVMP_STRINGIFY(x) SVMP_STRINGIFY_DETAIL(x)

using Clock = std::chrono::steady_clock;
using svmp::FE::ElementType;
using svmp::FE::Real;
using svmp::FE::basis::BatchEvaluator;
using svmp::FE::basis::BernsteinBasis;
using svmp::FE::basis::BasisCache;
using svmp::FE::basis::BSplineBasis;
using svmp::FE::basis::BDMBasis;
using svmp::FE::basis::BubbleBasis;
using svmp::FE::basis::CompatibleTensorVectorBasis;
using svmp::FE::basis::Gradient;
using svmp::FE::basis::Hessian;
using svmp::FE::basis::HermiteBasis;
using svmp::FE::basis::HierarchicalBasis;
using svmp::FE::basis::LagrangeBasis;
using svmp::FE::basis::NedelecBasis;
using svmp::FE::basis::NURBSTensorBasis;
using svmp::FE::basis::RaviartThomasBasis;
using svmp::FE::basis::SerendipityBasis;
using svmp::FE::basis::SpectralBasis;
using svmp::FE::basis::TensorProductBasis;
using svmp::FE::basis::VectorJacobian;
using svmp::FE::math::Vector;

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocations{0};
volatile double g_sink = 0.0;

const char* compiler_id() noexcept {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

const char* compiler_version() noexcept {
#if defined(__clang__)
    return SVMP_STRINGIFY(__clang_major__) "."
           SVMP_STRINGIFY(__clang_minor__) "."
           SVMP_STRINGIFY(__clang_patchlevel__);
#elif defined(__GNUC__)
    return SVMP_STRINGIFY(__GNUC__) "."
           SVMP_STRINGIFY(__GNUC_MINOR__) "."
           SVMP_STRINGIFY(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return SVMP_STRINGIFY(_MSC_VER);
#else
    return "unknown";
#endif
}

std::size_t simd_width_bytes() noexcept {
#if defined(__AVX512F__)
    return 64u;
#elif defined(__AVX__)
    return 32u;
#elif defined(__SSE2__) || defined(__ARM_NEON)
    return 16u;
#else
    return sizeof(Real);
#endif
}

std::string csv_token(std::string value) {
    for (char& ch : value) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }
    return value.empty() ? std::string{"unknown"} : value;
}

std::string build_flags_token() {
    if (const char* env = std::getenv("SVMP_BASIS_BENCH_FLAGS")) {
        return csv_token(env);
    }
#if defined(__OPTIMIZE__)
    return "optimized";
#else
    return "unoptimized";
#endif
}

std::string cpu_model_token() {
    static const std::string model = []() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            constexpr const char* key = "model name";
            if (line.rfind(key, 0) == 0) {
                const auto colon = line.find(':');
                if (colon != std::string::npos) {
                    return csv_token(line.substr(colon + 2u));
                }
            }
        }
        return std::string{"unknown"};
    }();
    return model;
}

std::string memory_bandwidth_token() {
    if (const char* env = std::getenv("SVMP_BASIS_BENCH_STREAM_GBPS")) {
        return csv_token(env);
    }
    return "unmeasured";
}

double positive_env_double(const char* name) noexcept {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return 0.0;
    }
    char* end = nullptr;
    const double value = std::strtod(env, &end);
    if (end == env || value <= 0.0) {
        return 0.0;
    }
    return value;
}

double peak_gflops() noexcept {
    return positive_env_double("SVMP_BASIS_BENCH_PEAK_GFLOPS");
}

double stream_gbps() noexcept {
    return positive_env_double("SVMP_BASIS_BENCH_STREAM_GBPS");
}

double machine_balance_flop_per_byte() noexcept {
    if (const double explicit_balance =
            positive_env_double("SVMP_BASIS_BENCH_MACHINE_BALANCE_FLOP_PER_BYTE");
        explicit_balance > 0.0) {
        return explicit_balance;
    }
    const double peak = peak_gflops();
    const double stream = stream_gbps();
    if (peak > 0.0 && stream > 0.0) {
        return peak / stream;
    }
    return 8.0;
}

std::string vector_efficiency_token() {
    if (const char* env = std::getenv("SVMP_BASIS_BENCH_VECTOR_EFFICIENCY")) {
        return csv_token(env);
    }
    return "unmeasured";
}

void consume(double value) noexcept {
    g_sink = static_cast<double>(g_sink) + value;
}

void note_allocation() noexcept {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocations.fetch_add(1, std::memory_order_relaxed);
    }
}

std::size_t reset_allocation_counter() noexcept {
    return g_allocations.exchange(0, std::memory_order_relaxed);
}

struct CountingScope {
    CountingScope() {
        reset_allocation_counter();
        g_count_allocations.store(true, std::memory_order_relaxed);
    }

    ~CountingScope() {
        g_count_allocations.store(false, std::memory_order_relaxed);
    }
};

struct Result {
    const char* name;
    const char* category;
    std::size_t iterations;
    double seconds;
    std::size_t allocations;
    std::size_t estimated_bytes_per_call;
    double modeled_flops_per_call{0.0};
    std::size_t modeled_min_bytes_per_call{0u};
    std::size_t repeats;
    double min_seconds;
    double max_seconds;
    std::size_t worker_threads{0u};
};

class SpinBarrier {
public:
    explicit SpinBarrier(std::size_t participants)
        : participants_(participants) {}

    void arrive_and_wait() noexcept {
        const std::size_t observed_generation = generation_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1u == participants_) {
            count_.store(0, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        while (generation_.load(std::memory_order_acquire) == observed_generation) {
            std::this_thread::yield();
        }
    }

private:
    const std::size_t participants_;
    std::atomic<std::size_t> count_{0};
    std::atomic<std::size_t> generation_{0};
};

std::size_t benchmark_thread_count() {
    const auto reported = std::thread::hardware_concurrency();
    if (reported == 0u) {
        return 4u;
    }
    return std::max<std::size_t>(2u, std::min<std::size_t>(4u, reported));
}

std::size_t benchmark_repeats() {
    const char* env = std::getenv("SVMP_BASIS_BENCH_REPEATS");
    if (env == nullptr || *env == '\0') {
        return 1u;
    }
    return std::max<std::size_t>(
        1u,
        static_cast<std::size_t>(std::strtoull(env, nullptr, 10)));
}

struct TimedSample {
    double seconds{0.0};
    std::size_t allocations{0};
};

template <typename MeasureFn>
Result make_repeated_result(const char* name,
                            const char* category,
                            std::size_t iterations,
                            std::size_t estimated_bytes_per_call,
                            MeasureFn&& measure) {
    const std::size_t repeats = benchmark_repeats();
    std::vector<TimedSample> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        samples.push_back(measure());
    }
    std::sort(samples.begin(), samples.end(),
              [](const TimedSample& a, const TimedSample& b) {
                  return a.seconds < b.seconds;
              });
    const auto& median = samples[samples.size() / 2u];

    return Result{
        name,
        category,
        iterations,
        median.seconds,
        median.allocations,
        estimated_bytes_per_call,
        0.0,
        0u,
        repeats,
        samples.front().seconds,
        samples.back().seconds
    };
}

template <typename Fn>
Result run_case(const char* name,
                const char* category,
                std::size_t iterations,
                std::size_t warmup,
                std::size_t estimated_bytes_per_call,
                Fn&& fn) {
    for (std::size_t i = 0; i < warmup; ++i) {
        fn();
    }

    return make_repeated_result(
        name,
        category,
        iterations,
        estimated_bytes_per_call,
        [&]() {
            auto t0 = Clock::now();
            std::size_t allocations = 0;
            {
                CountingScope counting;
                for (std::size_t i = 0; i < iterations; ++i) {
                    fn();
                }
                allocations = g_allocations.load(std::memory_order_relaxed);
            }
            auto t1 = Clock::now();
            return TimedSample{
                std::chrono::duration<double>(t1 - t0).count(),
                allocations
            };
        });
}

template <typename Fn>
Result run_threaded_case(const char* name,
                         const char* category,
                         std::size_t iterations,
                         std::size_t warmup,
                         std::size_t estimated_bytes_per_call,
                         Fn&& fn) {
    const std::size_t thread_count = benchmark_thread_count();
    for (std::size_t i = 0; i < warmup; ++i) {
        consume(fn(i % thread_count));
    }

    const std::size_t iterations_per_thread =
        std::max<std::size_t>(1u, (iterations + thread_count - 1u) / thread_count);
    const std::size_t total_iterations = iterations_per_thread * thread_count;
    return make_repeated_result(
        name,
        category,
        total_iterations,
        estimated_bytes_per_call,
        [&]() {
            std::atomic<std::size_t> ready{0};
            std::atomic<bool> start{false};
            std::vector<double> local_sums(thread_count, 0.0);
            std::vector<std::thread> threads;
            threads.reserve(thread_count);

            for (std::size_t thread = 0; thread < thread_count; ++thread) {
                threads.emplace_back([&, thread]() {
                    ready.fetch_add(1, std::memory_order_release);
                    while (!start.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    double local = 0.0;
                    for (std::size_t i = 0; i < iterations_per_thread; ++i) {
                        local += fn(thread);
                    }
                    local_sums[thread] = local;
                });
            }

            while (ready.load(std::memory_order_acquire) != thread_count) {
                std::this_thread::yield();
            }

            auto t0 = Clock::now();
            std::size_t allocations = 0;
            {
                CountingScope counting;
                start.store(true, std::memory_order_release);
                for (auto& thread : threads) {
                    thread.join();
                }
                allocations = g_allocations.load(std::memory_order_relaxed);
            }
            auto t1 = Clock::now();

            for (double value : local_sums) {
                consume(value);
            }

            return TimedSample{
                std::chrono::duration<double>(t1 - t0).count(),
                allocations
            };
        });
}

template <typename Fn>
Result run_cold_race_case(const char* name,
                          const char* category,
                          std::size_t rounds,
                          std::size_t warmup_rounds,
                          std::size_t estimated_bytes_per_call,
                          Fn&& fn) {
    const std::size_t thread_count = benchmark_thread_count();
    rounds = std::max<std::size_t>(1u, rounds);

    for (std::size_t round = 0; round < warmup_rounds; ++round) {
        BasisCache::instance().clear();
        for (std::size_t thread = 0; thread < thread_count; ++thread) {
            consume(fn(thread));
        }
    }
    BasisCache::instance().clear();

    return make_repeated_result(
        name,
        category,
        rounds * thread_count,
        estimated_bytes_per_call,
        [&]() {
            SpinBarrier start_round(thread_count + 1u);
            SpinBarrier finish_round(thread_count + 1u);
            std::vector<double> local_sums(thread_count, 0.0);
            std::vector<std::thread> threads;
            threads.reserve(thread_count);

            for (std::size_t thread = 0; thread < thread_count; ++thread) {
                threads.emplace_back([&, thread]() {
                    double local = 0.0;
                    for (std::size_t round = 0; round < rounds; ++round) {
                        start_round.arrive_and_wait();
                        local += fn(thread);
                        finish_round.arrive_and_wait();
                    }
                    local_sums[thread] = local;
                });
            }

            auto t0 = Clock::now();
            std::size_t allocations = 0;
            {
                CountingScope counting;
                for (std::size_t round = 0; round < rounds; ++round) {
                    BasisCache::instance().clear();
                    start_round.arrive_and_wait();
                    finish_round.arrive_and_wait();
                }
                allocations = g_allocations.load(std::memory_order_relaxed);
            }
            auto t1 = Clock::now();

            for (auto& thread : threads) {
                thread.join();
            }
            BasisCache::instance().clear();
            for (double value : local_sums) {
                consume(value);
            }

            return TimedSample{
                std::chrono::duration<double>(t1 - t0).count(),
                allocations
            };
        });
}

void print_result(const Result& r) {
    const double ns_per_call = r.seconds * 1.0e9 / static_cast<double>(r.iterations);
    const double min_ns_per_call = r.min_seconds * 1.0e9 / static_cast<double>(r.iterations);
    const double max_ns_per_call = r.max_seconds * 1.0e9 / static_cast<double>(r.iterations);
    const double allocs_per_call = static_cast<double>(r.allocations) /
                                   static_cast<double>(r.iterations);
    const double arithmetic_intensity = r.modeled_min_bytes_per_call == 0u
        ? 0.0
        : r.modeled_flops_per_call / static_cast<double>(r.modeled_min_bytes_per_call);
    const double machine_balance = machine_balance_flop_per_byte();
    const char* bound_class = r.modeled_flops_per_call <= 0.0 ||
                                      r.modeled_min_bytes_per_call == 0u
                                  ? "unmodeled"
                                  : (arithmetic_intensity < machine_balance
                                         ? "memory"
                                         : "compute");
    const double peak = peak_gflops();
    const double stream = stream_gbps();
    double model_lower_bound_ns = 0.0;
    if (r.modeled_flops_per_call > 0.0 &&
        r.modeled_min_bytes_per_call > 0u &&
        peak > 0.0 &&
        stream > 0.0) {
        const double compute_ns = r.modeled_flops_per_call / peak;
        const double memory_ns =
            static_cast<double>(r.modeled_min_bytes_per_call) / stream;
        model_lower_bound_ns = std::max(compute_ns, memory_ns);
    }
    const double measured_to_model_bound =
        model_lower_bound_ns > 0.0 ? ns_per_call / model_lower_bound_ns : 0.0;
    const std::size_t row_thread_count =
        r.worker_threads != 0u ? r.worker_threads : benchmark_thread_count();
    std::cout << r.name << ','
              << r.category << ','
              << compiler_id() << ','
              << compiler_version() << ','
              << r.iterations << ','
              << std::setprecision(12) << r.seconds << ','
              << ns_per_call << ','
              << r.allocations << ','
              << allocs_per_call << ','
              << r.estimated_bytes_per_call << ','
              << r.modeled_flops_per_call << ','
              << r.modeled_min_bytes_per_call << ','
              << arithmetic_intensity << ','
              << bound_class << ','
              << model_lower_bound_ns << ','
              << measured_to_model_bound << ','
              << machine_balance << ','
              << vector_efficiency_token() << ','
              << static_cast<double>(g_sink) << ','
              << r.repeats << ','
              << min_ns_per_call << ','
              << max_ns_per_call << ','
              << build_flags_token() << ','
              << cpu_model_token() << ','
              << std::thread::hardware_concurrency() << ','
              << row_thread_count << ','
              << simd_width_bytes() << ','
              << memory_bandwidth_token() << '\n';
}

std::size_t scaled_iterations(std::size_t base, std::size_t requested) {
    if (requested == 0) {
        return base;
    }
    return requested;
}

std::vector<Real> cubic_open_knots() {
    return {Real(0), Real(0), Real(0), Real(0),
            Real(0.5),
            Real(1), Real(1), Real(1), Real(1)};
}

std::vector<Real> tensor_nurbs_weights(std::size_t count) {
    std::vector<Real> weights(count, Real(1));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] += Real(0.01) * static_cast<Real>(i % 5u);
    }
    return weights;
}

bool lagrange_peak_scope_enabled() {
    const char* env = std::getenv("SVMP_BASIS_BENCH_LAGRANGE_PEAK");
    return env != nullptr && std::string(env) == "1";
}

bool lagrange_parallel_scope_enabled() {
    const char* env = std::getenv("SVMP_BASIS_BENCH_LAGRANGE_PARALLEL");
    return env != nullptr && std::string(env) == "1";
}

int lagrange_peak_max_order() {
    const char* env = std::getenv("SVMP_BASIS_BENCH_LAGRANGE_PEAK_MAX_ORDER");
    if (env == nullptr || *env == '\0') {
        return 8;
    }
    const int requested = static_cast<int>(std::strtol(env, nullptr, 10));
    return std::clamp(requested, 0, 8);
}

std::size_t scaled_lagrange_iterations(std::size_t base,
                                       std::size_t dofs,
                                       std::size_t requested) {
    if (requested != 0u) {
        return requested;
    }
    const std::size_t divisor = std::max<std::size_t>(1u, dofs / 8u);
    return std::max<std::size_t>(5u, base / divisor);
}

std::size_t scaled_lagrange_parallel_iterations(std::size_t requested) {
    if (requested != 0u) {
        return requested;
    }
    return 200u;
}

struct LagrangePeakTopology {
    const char* name;
    ElementType type;
    Vector<Real, 3> point;
    std::array<Vector<Real, 3>, 4> batch_points;
};

struct KernelModel {
    double flops_per_call{0.0};
    std::size_t min_bytes_per_call{0u};
};

int reference_dimension(ElementType type) noexcept {
    switch (type) {
        case ElementType::Line2:
            return 1;
        case ElementType::Triangle3:
        case ElementType::Quad4:
            return 2;
        case ElementType::Tetra4:
        case ElementType::Hex8:
        case ElementType::Wedge6:
        case ElementType::Pyramid5:
            return 3;
        default:
            return 3;
    }
}

bool is_tensor_lagrange_topology(ElementType type) noexcept {
    return type == ElementType::Line2 ||
           type == ElementType::Quad4 ||
           type == ElementType::Hex8;
}

bool is_simplex_lagrange_topology(ElementType type) noexcept {
    return type == ElementType::Triangle3 ||
           type == ElementType::Tetra4;
}

std::size_t output_components(bool values,
                              bool gradients,
                              bool hessians) noexcept {
    return (values ? 1u : 0u) +
           (gradients ? 3u : 0u) +
           (hessians ? 9u : 0u);
}

double active_components_for_flops(int dimension,
                                   bool values,
                                   bool gradients,
                                   bool hessians) noexcept {
    return (values ? 1.0 : 0.0) +
           (gradients ? static_cast<double>(dimension) : 0.0) +
           (hessians ? static_cast<double>(dimension * dimension) : 0.0);
}

double pyramid_fast_combination_nonzeros(int order) noexcept {
    if (order == 1) {
        return 21.0;
    }
    if (order == 2) {
        return 84.0;
    }
    return 0.0;
}

std::size_t lagrange_tensor_axis_table_bytes(ElementType type,
                                             int order,
                                             std::size_t qpts,
                                             bool values,
                                             bool gradients,
                                             bool hessians) noexcept {
    if (!is_tensor_lagrange_topology(type) || order <= 3) {
        return 0u;
    }

    const std::size_t n_axis = static_cast<std::size_t>(order + 1);
    const std::size_t dim = static_cast<std::size_t>(reference_dimension(type));
    const bool q4_product_axis_path =
        qpts == 4u &&
        (type == ElementType::Line2 ||
         type == ElementType::Quad4 ||
         (type == ElementType::Hex8 &&
          gradients &&
          !values &&
          !hessians));
    if (q4_product_axis_path) {
        return dim * qpts * n_axis * sizeof(Real);
    }

    std::size_t entries_per_axis = n_axis * n_axis;
    if (gradients || hessians) {
        entries_per_axis += n_axis * (n_axis - 1u);
    }
    if (hessians) {
        entries_per_axis += n_axis * (n_axis - 2u);
    }
    (void)values;
    return dim * qpts * entries_per_axis * sizeof(Real);
}

KernelModel lagrange_kernel_model(ElementType type,
                                  int order,
                                  std::size_t dofs,
                                  std::size_t qpts,
                                  bool values,
                                  bool gradients,
                                  bool hessians) {
    const int dim = reference_dimension(type);
    const double components =
        active_components_for_flops(dim, values, gradients, hessians);
    const std::size_t output_bytes =
        dofs * output_components(values, gradients, hessians) * qpts * sizeof(Real);
    const std::size_t point_bytes =
        qpts * static_cast<std::size_t>(std::max(dim, 1)) * sizeof(Real);
    const std::size_t table_bytes =
        lagrange_tensor_axis_table_bytes(type, order, qpts, values, gradients, hessians);
    const std::size_t bytes = output_bytes + point_bytes + table_bytes;

    double flops = 0.0;
    if (components > 0.0) {
        const double q = static_cast<double>(qpts);
        const double n = static_cast<double>(dofs);
        const double p = static_cast<double>(std::max(order, 1));

        if (is_tensor_lagrange_topology(type)) {
            const double setup_per_axis = hessians ? 12.0 : (gradients ? 8.0 : 4.0);
            const double product_flops =
                static_cast<double>(std::max(dim - 1, 1));
            flops = q * (static_cast<double>(dim * (order + 1)) * setup_per_axis +
                         n * components * product_flops);
        } else if (is_simplex_lagrange_topology(type)) {
            flops = q * n * components * (2.0 * p + static_cast<double>(dim));
        } else if (type == ElementType::Wedge6) {
            const double setup =
                static_cast<double>((order + 1) * (order + 1)) *
                (hessians ? 14.0 : (gradients ? 9.0 : 5.0));
            flops = q * (setup + n * components * 3.0);
        } else if (type == ElementType::Pyramid5) {
            const double modal_eval =
                q * n * components * (4.0 * p + 2.0);
            const double fast_nnz = pyramid_fast_combination_nonzeros(order);
            if (fast_nnz > 0.0) {
                flops = modal_eval + q * fast_nnz * components * 2.0;
            } else {
                flops = modal_eval + q * n * n * components * 2.0;
            }
        } else {
            flops = q * n * components * (2.0 * p + static_cast<double>(dim));
        }
    }

    return KernelModel{flops, bytes};
}

void attach_model(Result& result, const KernelModel& model) noexcept {
    result.modeled_flops_per_call = model.flops_per_call;
    result.modeled_min_bytes_per_call = model.min_bytes_per_call;
}

std::array<LagrangePeakTopology, 7> lagrange_peak_topologies() {
    return {{
        {"line", ElementType::Line2,
         Vector<Real, 3>{Real(0.125), Real(0), Real(0)},
         {Vector<Real, 3>{Real(-0.65), Real(0), Real(0)},
          Vector<Real, 3>{Real(-0.15), Real(0), Real(0)},
          Vector<Real, 3>{Real(0.35), Real(0), Real(0)},
          Vector<Real, 3>{Real(0.75), Real(0), Real(0)}}},
        {"triangle", ElementType::Triangle3,
         Vector<Real, 3>{Real(0.2), Real(0.3), Real(0)},
         {Vector<Real, 3>{Real(0.15), Real(0.2), Real(0)},
          Vector<Real, 3>{Real(0.55), Real(0.1), Real(0)},
          Vector<Real, 3>{Real(0.2), Real(0.55), Real(0)},
          Vector<Real, 3>{Real(0.3), Real(0.25), Real(0)}}},
        {"quad", ElementType::Quad4,
         Vector<Real, 3>{Real(0.125), Real(-0.25), Real(0)},
         {Vector<Real, 3>{Real(-0.7), Real(-0.6), Real(0)},
          Vector<Real, 3>{Real(0.4), Real(-0.35), Real(0)},
          Vector<Real, 3>{Real(-0.2), Real(0.45), Real(0)},
          Vector<Real, 3>{Real(0.65), Real(0.55), Real(0)}}},
        {"tet", ElementType::Tetra4,
         Vector<Real, 3>{Real(0.2), Real(0.2), Real(0.2)},
         {Vector<Real, 3>{Real(0.12), Real(0.18), Real(0.22)},
          Vector<Real, 3>{Real(0.45), Real(0.12), Real(0.16)},
          Vector<Real, 3>{Real(0.16), Real(0.44), Real(0.14)},
          Vector<Real, 3>{Real(0.18), Real(0.16), Real(0.42)}}},
        {"hex", ElementType::Hex8,
         Vector<Real, 3>{Real(0.125), Real(-0.25), Real(0.375)},
         {Vector<Real, 3>{Real(-0.55), Real(-0.45), Real(-0.35)},
          Vector<Real, 3>{Real(0.45), Real(-0.25), Real(0.2)},
          Vector<Real, 3>{Real(-0.25), Real(0.5), Real(-0.15)},
          Vector<Real, 3>{Real(0.6), Real(0.35), Real(0.55)}}},
        {"wedge", ElementType::Wedge6,
         Vector<Real, 3>{Real(0.2), Real(0.2), Real(0.1)},
         {Vector<Real, 3>{Real(0.15), Real(0.2), Real(-0.55)},
          Vector<Real, 3>{Real(0.45), Real(0.15), Real(-0.1)},
          Vector<Real, 3>{Real(0.2), Real(0.45), Real(0.25)},
          Vector<Real, 3>{Real(0.3), Real(0.25), Real(0.65)}}},
        {"pyramid", ElementType::Pyramid5,
         Vector<Real, 3>{Real(0.1), Real(-0.2), Real(0.25)},
         {Vector<Real, 3>{Real(-0.35), Real(-0.25), Real(0.1)},
          Vector<Real, 3>{Real(0.25), Real(-0.2), Real(0.25)},
          Vector<Real, 3>{Real(-0.12), Real(0.22), Real(0.35)},
          Vector<Real, 3>{Real(0.08), Real(0.05), Real(0.7)}}},
    }};
}

void run_lagrange_peak_scope(std::size_t requested_iterations) {
    const int max_order = lagrange_peak_max_order();
    const auto topologies = lagrange_peak_topologies();
    for (const auto& topology : topologies) {
        for (int order = 0; order <= max_order; ++order) {
            const std::string construction_name =
                std::string("lagrange_") + topology.name + "_order" +
                std::to_string(order) + "_construction";
            print_result(run_case(
                construction_name.c_str(),
                "lagrange_construction",
                scaled_lagrange_iterations(20, 1u, requested_iterations),
                0,
                0,
                [&]() {
                    LagrangeBasis basis(topology.type, order);
                    consume(static_cast<double>(basis.size()));
                }));

            LagrangeBasis basis(topology.type, order);
            const std::size_t dofs = basis.size();
            const std::size_t scalar_iterations =
                scaled_lagrange_iterations(5000, dofs, requested_iterations);
            const std::size_t strided_iterations =
                scaled_lagrange_iterations(1000, dofs, requested_iterations);
            const std::size_t stride = topology.batch_points.size();
            const std::vector<Vector<Real, 3>> points(topology.batch_points.begin(),
                                                      topology.batch_points.end());
            svmp::FE::basis::prewarm_lagrange_basis_scratch(order, points.size());

            std::vector<Real> values(dofs);
            std::vector<Gradient> gradients(dofs);
            std::vector<Hessian> hessians(dofs);
            std::vector<Real> raw_values(dofs);
            std::vector<Real> raw_gradients(dofs * 3u);
            std::vector<Real> raw_hessians(dofs * 9u);
            std::vector<Real> values_strided(dofs * stride, Real(0));
            std::vector<Real> gradients_strided(dofs * 3u * stride, Real(0));
            std::vector<Real> hessians_strided(dofs * 9u * stride, Real(0));
            std::size_t scalar_point_cursor = 0;
            auto next_scalar_point = [&]() -> const Vector<Real, 3>& {
                const Vector<Real, 3>& point = points[scalar_point_cursor];
                scalar_point_cursor = (scalar_point_cursor + 1u) & 3u;
                return point;
            };

            auto run_scalar = [&](const char* op,
                                  bool need_values,
                                  bool need_gradients,
                                  bool need_hessians,
                                  std::size_t bytes,
                                  auto&& fn) {
                const std::string name =
                    std::string("lagrange_") + topology.name + "_order" +
                    std::to_string(order) + "_point_" + op;
                auto result = run_case(
                    name.c_str(),
                    "lagrange_scalar_point",
                    scalar_iterations,
                    25,
                    bytes,
                    [&]() {
                        fn();
                    });
                attach_model(result,
                             lagrange_kernel_model(topology.type,
                                                   order,
                                                   dofs,
                                                   1u,
                                                   need_values,
                                                   need_gradients,
                                                   need_hessians));
                print_result(result);
            };

            run_scalar("values", true, false, false, dofs * sizeof(Real), [&]() {
                basis.evaluate_values(next_scalar_point(), values);
                consume(values[0]);
            });
            run_scalar("gradients", false, true, false, dofs * 3u * sizeof(Real), [&]() {
                basis.evaluate_gradients(next_scalar_point(), gradients);
                consume(gradients[0][0]);
            });
            run_scalar("hessians", false, false, true, dofs * 9u * sizeof(Real), [&]() {
                basis.evaluate_hessians(next_scalar_point(), hessians);
                consume(hessians[0](0, 0));
            });
            run_scalar("all", true, true, true, dofs * 13u * sizeof(Real), [&]() {
                basis.evaluate_all(next_scalar_point(), values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            });

            auto run_raw_to = [&](const char* op,
                                  bool need_values,
                                  bool need_gradients,
                                  bool need_hessians,
                                  std::size_t bytes,
                                  auto&& fn) {
                const std::string name =
                    std::string("lagrange_") + topology.name + "_order" +
                    std::to_string(order) + "_to_" + op;
                auto result = run_case(
                    name.c_str(),
                    "lagrange_raw_to_point",
                    scalar_iterations,
                    25,
                    bytes,
                    [&]() {
                        fn();
                    });
                attach_model(result,
                             lagrange_kernel_model(topology.type,
                                                   order,
                                                   dofs,
                                                   1u,
                                                   need_values,
                                                   need_gradients,
                                                   need_hessians));
                print_result(result);
            };

            run_raw_to("values", true, false, false, dofs * sizeof(Real), [&]() {
                basis.evaluate_values_to(next_scalar_point(), raw_values.data());
                consume(raw_values[0]);
            });
            run_raw_to("gradients", false, true, false, dofs * 3u * sizeof(Real), [&]() {
                basis.evaluate_gradients_to(next_scalar_point(), raw_gradients.data());
                consume(raw_gradients[0]);
            });
            run_raw_to("hessians", false, false, true, dofs * 9u * sizeof(Real), [&]() {
                basis.evaluate_hessians_to(next_scalar_point(), raw_hessians.data());
                consume(raw_hessians[0]);
            });
            run_raw_to("all", true, true, true, dofs * 13u * sizeof(Real), [&]() {
                const auto& point = next_scalar_point();
                basis.evaluate_values_to(point, raw_values.data());
                basis.evaluate_gradients_to(point, raw_gradients.data());
                basis.evaluate_hessians_to(point, raw_hessians.data());
                consume(raw_values[0] + raw_gradients[0] + raw_hessians[0]);
            });

            auto run_strided = [&](const char* op,
                                   bool need_values,
                                   bool need_gradients,
                                   bool need_hessians,
                                   std::size_t components) {
                const std::string name =
                    std::string("lagrange_") + topology.name + "_order" +
                    std::to_string(order) + "_strided_" + op;
                auto result = run_case(
                    name.c_str(),
                    "lagrange_strided_batch",
                    strided_iterations,
                    10,
                    dofs * components * stride * sizeof(Real),
                    [&]() {
                        basis.evaluate_at_quadrature_points_strided(
                            points,
                            stride,
                            need_values ? values_strided.data() : nullptr,
                            need_gradients ? gradients_strided.data() : nullptr,
                            need_hessians ? hessians_strided.data() : nullptr);
                        Real sample = Real(0);
                        if (need_values) sample += values_strided[0];
                        if (need_gradients) sample += gradients_strided[0];
                        if (need_hessians) sample += hessians_strided[0];
                        consume(sample);
                    });
                attach_model(result,
                             lagrange_kernel_model(topology.type,
                                                   order,
                                                   dofs,
                                                   points.size(),
                                                   need_values,
                                                   need_gradients,
                                                   need_hessians));
                print_result(result);
            };

            run_strided("values", true, false, false, 1u);
            run_strided("gradients", false, true, false, 3u);
            run_strided("hessians", false, false, true, 9u);
            run_strided("all", true, true, true, 13u);
        }
    }
}

void run_lagrange_parallel_scope(std::size_t requested_iterations) {
    constexpr int order = 4;
    constexpr ElementType element_type = ElementType::Hex8;
    const std::array<Vector<Real, 3>, 8> batch_points{{
        Vector<Real, 3>{Real(-0.55), Real(-0.45), Real(-0.35)},
        Vector<Real, 3>{Real(0.45), Real(-0.25), Real(0.2)},
        Vector<Real, 3>{Real(-0.25), Real(0.5), Real(-0.15)},
        Vector<Real, 3>{Real(0.6), Real(0.35), Real(0.55)},
        Vector<Real, 3>{Real(-0.4), Real(0.15), Real(0.3)},
        Vector<Real, 3>{Real(0.25), Real(-0.55), Real(-0.2)},
        Vector<Real, 3>{Real(-0.1), Real(-0.2), Real(0.65)},
        Vector<Real, 3>{Real(0.15), Real(0.25), Real(-0.55)},
    }};
    const std::vector<Vector<Real, 3>> points(batch_points.begin(), batch_points.end());
    const std::size_t stride = points.size() + 1u;
    const std::size_t iterations_per_thread = scaled_lagrange_parallel_iterations(requested_iterations);
    const LagrangeBasis size_probe(element_type, order);
    const std::size_t dofs = size_probe.size();
    const KernelModel model =
        lagrange_kernel_model(element_type, order, dofs, points.size(), true, true, true);

    const std::array<std::size_t, 5> thread_counts{{1u, 2u, 4u, 8u, 16u}};
    for (const std::size_t thread_count : thread_counts) {
        for (const bool schedule_only : {true, false}) {
            const std::string name =
                std::string("lagrange_parallel_hex_order4_") +
                (schedule_only ? "schedule_only" : "strided_all") +
                "_threads" + std::to_string(thread_count);
            const char* category =
                schedule_only ? "lagrange_parallel_schedule" : "lagrange_parallel_eval";
            const std::size_t total_iterations = iterations_per_thread * thread_count;
            const std::size_t bytes = schedule_only
                ? 0u
                : dofs * 13u * stride * sizeof(Real);

            auto result = make_repeated_result(
                name.c_str(),
                category,
                total_iterations,
                bytes,
                [&]() {
                    std::atomic<std::size_t> ready{0};
                    std::atomic<bool> start{false};
                    std::vector<double> local_sums(thread_count, 0.0);
                    std::vector<std::thread> threads;
                    threads.reserve(thread_count);

                    for (std::size_t thread = 0; thread < thread_count; ++thread) {
                        threads.emplace_back([&, thread]() {
                            LagrangeBasis basis(element_type, order);
                            svmp::FE::basis::prewarm_lagrange_basis_scratch(order, points.size());
                            std::vector<Real> values(dofs * stride, Real(0));
                            std::vector<Real> gradients(dofs * 3u * stride, Real(0));
                            std::vector<Real> hessians(dofs * 9u * stride, Real(0));

                            ready.fetch_add(1, std::memory_order_release);
                            while (!start.load(std::memory_order_acquire)) {
                                std::this_thread::yield();
                            }

                            double local = 0.0;
                            for (std::size_t iter = 0; iter < iterations_per_thread; ++iter) {
                                if (schedule_only) {
                                    local += static_cast<double>((thread + 1u) * (iter + 1u)) *
                                             1.0e-12;
                                } else {
                                    basis.evaluate_at_quadrature_points_strided(
                                        points,
                                        stride,
                                        values.data(),
                                        gradients.data(),
                                        hessians.data());
                                    local += values[0] + gradients[0] + hessians[0];
                                }
                            }
                            local_sums[thread] = local;
                        });
                    }

                    while (ready.load(std::memory_order_acquire) != thread_count) {
                        std::this_thread::yield();
                    }

                    auto t0 = Clock::now();
                    std::size_t allocations = 0;
                    {
                        CountingScope counting;
                        start.store(true, std::memory_order_release);
                        for (auto& thread : threads) {
                            thread.join();
                        }
                        allocations = g_allocations.load(std::memory_order_relaxed);
                    }
                    auto t1 = Clock::now();

                    for (double value : local_sums) {
                        consume(value);
                    }

                    return TimedSample{
                        std::chrono::duration<double>(t1 - t0).count(),
                        allocations
                    };
                });

            result.worker_threads = thread_count;
            if (!schedule_only) {
                attach_model(result, model);
            }
            print_result(result);
        }
    }
}

} // namespace

#undef SVMP_STRINGIFY
#undef SVMP_STRINGIFY_DETAIL

void* operator new(std::size_t size) {
    note_allocation();
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    note_allocation();
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}

void* operator new(std::size_t size, std::align_val_t align) {
    note_allocation();
    void* p = nullptr;
    const auto alignment = static_cast<std::size_t>(align);
    if (posix_memalign(&p, alignment, size) == 0 && p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t align) {
    note_allocation();
    void* p = nullptr;
    const auto alignment = static_cast<std::size_t>(align);
    if (posix_memalign(&p, alignment, size) == 0 && p != nullptr) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

int main(int argc, char** argv) {
    std::size_t requested_iterations = 0;
    if (argc > 1) {
        requested_iterations = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }

    std::cout << std::unitbuf;
    std::cout << "case,category,compiler_id,compiler_version,iterations,"
                 "seconds,ns_per_call,allocations,"
                 "allocations_per_call,estimated_bytes_per_call,"
                 "modeled_flops_per_call,modeled_min_bytes_per_call,"
                 "arithmetic_intensity_flop_per_byte,bound_class,"
                 "model_lower_bound_ns,measured_to_model_bound,"
                 "machine_balance_flop_per_byte,vector_efficiency,sink,"
                 "repeats,min_ns_per_call,max_ns_per_call,"
                 "build_flags,cpu_model,hardware_threads,bench_threads,"
                 "simd_width_bytes,stream_gbps\n";

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        Vector<Real, 3> xi{Real(0.125), Real(-0.25), Real(0.375)};
        std::vector<Real> values(basis.size());
        const std::size_t bytes = basis.size() * sizeof(Real);
        print_result(run_case(
            "lagrange_hex_order2_values", "scalar_point", scaled_iterations(200000, requested_iterations), 1000, bytes,
            [&]() {
                basis.evaluate_values(xi, values);
                consume(values[0]);
            }));
    }

    auto run_lagrange_pyramid_strided_case =
        [&](int order,
            int quadrature_order,
            const char* name,
            bool need_values,
            bool need_gradients,
            bool need_hessians,
            std::size_t base_iterations) {
            LagrangeBasis basis(ElementType::Pyramid5, order);
            auto quad = svmp::FE::quadrature::QuadratureFactory::create(
                ElementType::Pyramid5, quadrature_order);
            const std::size_t stride = quad->num_points() + 1u;
            std::vector<Real> values(need_values ? basis.size() * stride : 0u, Real(0));
            std::vector<Real> gradients(need_gradients ? basis.size() * 3u * stride : 0u, Real(0));
            std::vector<Real> hessians(need_hessians ? basis.size() * 9u * stride : 0u, Real(0));
            const std::size_t components =
                (need_values ? 1u : 0u) +
                (need_gradients ? 3u : 0u) +
                (need_hessians ? 9u : 0u);
            const std::size_t bytes = basis.size() * components * stride * sizeof(Real);
            print_result(run_case(
                name,
                "lagrange_pyramid_strided",
                scaled_iterations(base_iterations, requested_iterations),
                50,
                bytes,
                [&]() {
                    basis.evaluate_at_quadrature_points_strided(
                        quad->points(),
                        stride,
                        need_values ? values.data() : nullptr,
                        need_gradients ? gradients.data() : nullptr,
                        need_hessians ? hessians.data() : nullptr);
                    Real sample = Real(0);
                    if (need_values) {
                        sample += values[0];
                    }
                    if (need_gradients) {
                        sample += gradients[0];
                    }
                    if (need_hessians) {
                        sample += hessians[0];
                    }
                    consume(sample);
                }));
        };

    run_lagrange_pyramid_strided_case(
        2, 4, "lagrange_pyramid_special_order2_strided_values", true, false, false, 4000);
    run_lagrange_pyramid_strided_case(
        2, 4, "lagrange_pyramid_special_order2_strided_values_gradients", true, true, false, 3000);
    run_lagrange_pyramid_strided_case(
        2, 4, "lagrange_pyramid_special_order2_strided_hessians", false, false, true, 2000);
    run_lagrange_pyramid_strided_case(
        5, 6, "lagrange_pyramid_special_order5_strided_all", true, true, true, 500);

    if (lagrange_peak_scope_enabled()) {
        run_lagrange_peak_scope(requested_iterations);
    }
    if (lagrange_parallel_scope_enabled()) {
        run_lagrange_parallel_scope(requested_iterations);
    }

    {
        BernsteinBasis basis(ElementType::Hex8, 5);
        Vector<Real, 3> xi{Real(0.125), Real(-0.25), Real(0.375)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "bernstein_hex_order5_all", "bernstein_recurrence",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    auto run_bernstein_strided_case =
        [&](ElementType element_type,
            int order,
            int quadrature_order,
            const char* name,
            std::size_t base_iterations) {
            BernsteinBasis basis(element_type, order);
            auto quad = svmp::FE::quadrature::QuadratureFactory::create(
                element_type, quadrature_order);
            const std::size_t stride = quad->num_points() + 1u;
            std::vector<Real> values(basis.size() * stride, Real(0));
            std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
            std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
            const std::size_t bytes = basis.size() * 13u * stride * sizeof(Real);
            print_result(run_case(
                name,
                "bernstein_strided",
                scaled_iterations(base_iterations, requested_iterations),
                50,
                bytes,
                [&]() {
                    basis.evaluate_at_quadrature_points_strided(
                        quad->points(),
                        stride,
                        values.data(),
                        gradients.data(),
                        hessians.data());
                    consume(values[0] + gradients[0] + hessians[0]);
                }));
        };

    run_bernstein_strided_case(
        ElementType::Hex8, 5, 6, "bernstein_hex_order5_strided_all", 2000);
    run_bernstein_strided_case(
        ElementType::Pyramid5, 4, 6, "bernstein_pyramid_order4_strided_all", 1000);

    {
        SerendipityBasis basis(ElementType::Hex20, 2);
        Vector<Real, 3> xi{Real(0.2), Real(-0.1), Real(0.3)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "serendipity_hex20_all", "serendipity_tensor_modal",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_values(xi, values);
                basis.evaluate_gradients(xi, gradients);
                basis.evaluate_hessians(xi, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    {
        HierarchicalBasis basis(ElementType::Hex8, 5);
        Vector<Real, 3> xi{Real(0.125), Real(-0.25), Real(0.375)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "hierarchical_hex_order5_all", "hierarchical_legendre_scratch",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    auto run_scalar_strided_all_case =
        [&](auto& basis,
            const std::vector<Vector<Real, 3>>& points,
            const char* name,
            std::size_t base_iterations) {
            const std::size_t stride = points.size() + 1u;
            std::vector<Real> values(basis.size() * stride, Real(0));
            std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
            std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
            const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
            print_result(run_case(
                name,
                "generic_scalar_fallback_strided",
                scaled_iterations(base_iterations, requested_iterations),
                100,
                bytes,
                [&]() {
                    basis.evaluate_at_quadrature_points_strided(
                        points, stride, values.data(), gradients.data(), hessians.data());
                    consume(values[0] + gradients[0] + hessians[0]);
                }));
        };

    {
        HermiteBasis basis(ElementType::Hex8, 3);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 3);
        run_scalar_strided_all_case(basis, quad->points(), "hermite_hex_cubic_strided_all", 2000);
    }

    {
        HierarchicalBasis basis(ElementType::Hex8, 5);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 5);
        run_scalar_strided_all_case(basis, quad->points(), "hierarchical_hex_order5_strided_all", 1000);
    }

    {
        SerendipityBasis basis(ElementType::Hex20, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        run_scalar_strided_all_case(basis, quad->points(), "serendipity_hex20_strided_all", 4000);
    }

    {
        BubbleBasis basis(ElementType::Pyramid5);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Pyramid5, 4);
        run_scalar_strided_all_case(basis, quad->points(), "bubble_pyramid_strided_all", 10000);
    }

    {
        BSplineBasis basis(3, cubic_open_knots());
        Vector<Real, 3> xi{Real(0.15), Real(0), Real(0)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "bspline_line_degree3_all", "spline_flat_scratch",
            scaled_iterations(200000, requested_iterations), 1000, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    {
        BSplineBasis bx(3, cubic_open_knots());
        BSplineBasis by(3, cubic_open_knots());
        auto weights = tensor_nurbs_weights(25u);
        NURBSTensorBasis basis(std::move(bx), std::move(by), std::move(weights));
        Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "nurbs_tensor_quad_degree3_all", "nurbs_active_support",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    {
        TensorProductBasis<BSplineBasis> basis(
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()));
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Quad4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "tensor_bspline_quad_degree3_strided_all", "tensor_spline_strided",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    {
        TensorProductBasis<BSplineBasis, 1> basis(BSplineBasis(3, cubic_open_knots()));
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Line2, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "tensor_bspline_line_degree3_static_dim1_strided_all", "tensor_spline_strided_static",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    {
        TensorProductBasis<BSplineBasis, 2> basis(
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()));
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Quad4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "tensor_bspline_quad_degree3_static_dim2_strided_all", "tensor_spline_strided_static",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    {
        TensorProductBasis<BSplineBasis, 3> basis(
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()));
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "tensor_bspline_hex_degree3_static_dim3_strided_all", "tensor_spline_strided_static",
            scaled_iterations(50000, requested_iterations), 250, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    {
        BSplineBasis bx(3, cubic_open_knots());
        BSplineBasis by(3, cubic_open_knots());
        NURBSTensorBasis basis(std::move(bx), std::move(by), tensor_nurbs_weights(25u));
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Quad4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "nurbs_tensor_quad_degree3_strided_all", "nurbs_strided",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        std::vector<Real> coeffs(basis.size(), Real(0.25));
        std::vector<Real> weights(quad->num_points(), Real(1));
        std::vector<Real> result(quad->num_points(), Real(0));
        const std::size_t bytes = (basis.size() + 2u * quad->num_points()) * sizeof(Real);
        print_result(run_case(
            "batch_hex_order2_weighted_sum", "batched_quadrature",
            scaled_iterations(200000, requested_iterations), 1000, bytes,
            [&]() {
                BatchEvaluator batch(basis, *quad, true, false);
                svmp::FE::assembly::weighted_sum(batch, coeffs.data(), weights.data(), result.data());
                consume(result[0]);
            }));
    }

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        const std::size_t bytes = basis.size() * quad->num_points() * 4u * sizeof(Real);
        print_result(run_case(
            "cache_hex_order2_uncached", "cache_construction", scaled_iterations(2000, requested_iterations), 20, bytes,
            [&]() {
                const auto entry = BasisCache::instance().compute_uncached(basis, *quad, true, false);
                consume(entry.scalarValue(0, 0));
            }));
    }

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        BasisCache::instance().clear();
        const auto handle = BasisCache::instance().prewarm_handle(basis, *quad, true, false);
        print_result(run_case(
            "cache_hex_order2_reuse", "cache_reuse", scaled_iterations(200000, requested_iterations), 1000, 0,
            [&]() {
                const auto& entry = handle.entry();
                consume(entry.scalarValue(0, 0));
            }));
    }

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        BasisCache::instance().clear();
        const auto handle = BasisCache::instance().prewarm_handle(basis, *quad, true, false);
        print_result(run_threaded_case(
            "cache_hex_order2_reuse_threaded", "cache_reuse_threaded",
            scaled_iterations(200000, requested_iterations), 1000, 0,
            [&](std::size_t) {
                const auto& entry = handle.entry();
                return entry.scalarValue(0, 0);
            }));
    }

    {
        LagrangeBasis basis(ElementType::Hex8, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 4);
        const std::size_t bytes = basis.size() * quad->num_points() * 4u * sizeof(Real);
        print_result(run_cold_race_case(
            "cache_hex_order2_cold_race_threaded", "cache_cold_race",
            scaled_iterations(1000, requested_iterations), 5, bytes,
            [&](std::size_t) {
                const auto& entry = BasisCache::instance().get_or_compute(basis, *quad, true, false);
                return entry.scalarValue(0, 0);
            }));
    }

    {
        RaviartThomasBasis basis(ElementType::Wedge6, 3);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Wedge6, 4);
        const std::size_t bytes = basis.size() * quad->num_points() * (3u + 9u + 1u) * sizeof(Real);
        print_result(run_case(
            "cache_rt_wedge_order3_uncached", "vector_cache_construction",
            scaled_iterations(200, requested_iterations), 5, bytes,
            [&]() {
                const auto entry = BasisCache::instance().compute_uncached(basis, *quad, true, false);
                consume(entry.vectorValue(0, 0, 0) + entry.vectorDivergenceValue(0, 0));
            }));
    }

    {
        RaviartThomasBasis basis(ElementType::Wedge6, 3);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Wedge6, 4);
        BasisCache::instance().clear();
        const auto handle = BasisCache::instance().prewarm_handle(basis, *quad, true, false);
        print_result(run_case(
            "cache_rt_wedge_order3_reuse", "vector_cache_reuse",
            scaled_iterations(100000, requested_iterations), 1000, 0,
            [&]() {
                const auto& entry = handle.entry();
                consume(entry.vectorValue(0, 0, 0) + entry.vectorDivergenceValue(0, 0));
            }));
    }

    {
        RaviartThomasBasis basis(ElementType::Wedge6, 3);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Wedge6, 4);
        const std::size_t bytes = basis.size() * quad->num_points() * (3u + 9u + 1u) * sizeof(Real);
        print_result(run_cold_race_case(
            "cache_rt_wedge_order3_cold_race_threaded", "vector_cache_cold_race",
            scaled_iterations(50, requested_iterations), 2, bytes,
            [&](std::size_t) {
                const auto& entry = BasisCache::instance().get_or_compute(basis, *quad, true, false);
                return entry.vectorValue(0, 0, 0) + entry.vectorDivergenceValue(0, 0);
            }));
    }

    {
        SpectralBasis basis(ElementType::Hex8, 6);
        Vector<Real, 3> xi{Real(0.2), Real(-0.15), Real(0.05)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "spectral_hex_order6_all", "spectral_high_order", scaled_iterations(50000, requested_iterations), 200, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    {
        SpectralBasis basis(ElementType::Hex8, 6);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Hex8, 6);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * stride, Real(0));
        std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
        std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
        print_result(run_case(
            "spectral_hex_order6_strided_all", "spectral_strided",
            scaled_iterations(1000, requested_iterations), 50, bytes,
            [&]() {
                basis.evaluate_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), gradients.data(), hessians.data());
                consume(values[0] + gradients[0] + hessians[0]);
            }));
    }

    auto run_spectral_simplex_point_cases =
        [&](ElementType type,
            int order,
            const Vector<Real, 3>& xi,
            const char* values_name,
            const char* gradients_name,
            const char* hessians_name,
            const char* all_name,
            std::size_t base_iterations) {
            SpectralBasis basis(type, order);
            std::vector<Real> values(basis.size());
            std::vector<Gradient> gradients(basis.size());
            std::vector<Hessian> hessians(basis.size());
            const std::size_t value_bytes = basis.size() * sizeof(Real);
            const std::size_t gradient_bytes = basis.size() * 3u * sizeof(Real);
            const std::size_t hessian_bytes = basis.size() * 9u * sizeof(Real);
            const std::size_t all_bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);

            print_result(run_case(
                values_name, "spectral_simplex_values",
                scaled_iterations(base_iterations, requested_iterations), 100, value_bytes,
                [&]() {
                    basis.evaluate_values(xi, values);
                    consume(values[0]);
                }));
            print_result(run_case(
                gradients_name, "spectral_simplex_gradients",
                scaled_iterations(base_iterations, requested_iterations), 100, gradient_bytes,
                [&]() {
                    basis.evaluate_gradients(xi, gradients);
                    consume(gradients[0][0]);
                }));
            print_result(run_case(
                hessians_name, "spectral_simplex_hessians",
                scaled_iterations(base_iterations / 2u, requested_iterations), 50, hessian_bytes,
                [&]() {
                    basis.evaluate_hessians(xi, hessians);
                    consume(hessians[0](0, 0));
                }));
            print_result(run_case(
                all_name, "spectral_simplex_all",
                scaled_iterations(base_iterations / 2u, requested_iterations), 50, all_bytes,
                [&]() {
                    basis.evaluate_all(xi, values, gradients, hessians);
                    consume(values[0] + gradients[0][0] + hessians[0](0, 0));
                }));
        };

    run_spectral_simplex_point_cases(
        ElementType::Triangle3,
        8,
        Vector<Real, 3>{Real(0.24), Real(0.31), Real(0)},
        "spectral_triangle_order8_values",
        "spectral_triangle_order8_gradients",
        "spectral_triangle_order8_hessians",
        "spectral_triangle_order8_all",
        20000);
    run_spectral_simplex_point_cases(
        ElementType::Triangle3,
        10,
        Vector<Real, 3>{Real(0.22), Real(0.27), Real(0)},
        "spectral_triangle_order10_values",
        "spectral_triangle_order10_gradients",
        "spectral_triangle_order10_hessians",
        "spectral_triangle_order10_all",
        10000);
    run_spectral_simplex_point_cases(
        ElementType::Tetra4,
        6,
        Vector<Real, 3>{Real(0.18), Real(0.21), Real(0.16)},
        "spectral_tetra_order6_values",
        "spectral_tetra_order6_gradients",
        "spectral_tetra_order6_hessians",
        "spectral_tetra_order6_all",
        8000);
    run_spectral_simplex_point_cases(
        ElementType::Tetra4,
        8,
        Vector<Real, 3>{Real(0.16), Real(0.19), Real(0.14)},
        "spectral_tetra_order8_values",
        "spectral_tetra_order8_gradients",
        "spectral_tetra_order8_hessians",
        "spectral_tetra_order8_all",
        4000);

    auto run_spectral_strided_case =
        [&](ElementType element_type,
            int order,
            int quadrature_order,
            const char* name,
            std::size_t base_iterations) {
            SpectralBasis basis(element_type, order);
            auto quad = svmp::FE::quadrature::QuadratureFactory::create(
                element_type, quadrature_order);
            const std::size_t stride = quad->num_points() + 1u;
            std::vector<Real> values(basis.size() * stride, Real(0));
            std::vector<Real> gradients(basis.size() * 3u * stride, Real(0));
            std::vector<Real> hessians(basis.size() * 9u * stride, Real(0));
            const std::size_t bytes = basis.size() * (1u + 3u + 9u) * stride * sizeof(Real);
            print_result(run_case(
                name,
                "spectral_strided",
                scaled_iterations(base_iterations, requested_iterations),
                5,
                bytes,
                [&]() {
                    basis.evaluate_at_quadrature_points_strided(
                        quad->points(), stride, values.data(), gradients.data(), hessians.data());
                    consume(values[0] + gradients[0] + hessians[0]);
                }));
        };

    run_spectral_strided_case(
        ElementType::Triangle3, 8, 10, "spectral_triangle_order8_strided_all", 20);
    run_spectral_strided_case(
        ElementType::Triangle3, 10, 12, "spectral_triangle_order10_strided_all", 10);
    run_spectral_strided_case(
        ElementType::Tetra4, 6, 8, "spectral_tetra_order6_strided_all", 10);
    run_spectral_strided_case(
        ElementType::Tetra4, 8, 10, "spectral_tetra_order8_strided_all", 5);

    {
        SpectralBasis basis(ElementType::Pyramid5, 4);
        Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.35)};
        std::vector<Real> values(basis.size());
        std::vector<Gradient> gradients(basis.size());
        std::vector<Hessian> hessians(basis.size());
        const std::size_t bytes = basis.size() * (1u + 3u + 9u) * sizeof(Real);
        print_result(run_case(
            "spectral_pyramid_order4_all", "pyramid_modal_to_nodal",
            scaled_iterations(50000, requested_iterations), 200, bytes,
            [&]() {
                basis.evaluate_all(xi, values, gradients, hessians);
                consume(values[0] + gradients[0][0] + hessians[0](0, 0));
            }));
    }

    run_spectral_strided_case(
        ElementType::Pyramid5, 4, 5, "spectral_pyramid_order4_strided_all", 20);
    run_spectral_strided_case(
        ElementType::Pyramid5, 5, 6, "spectral_pyramid_order5_strided_all", 10);

    {
        RaviartThomasBasis basis(ElementType::Wedge6, 3);
        Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(-0.1)};
        std::vector<Vector<Real, 3>> values(basis.size());
        std::vector<VectorJacobian> jacobians(basis.size());
        std::vector<Real> divergence(basis.size());
        const std::size_t bytes = basis.size() * (3u + 9u + 1u) * sizeof(Real);
        print_result(run_case(
            "rt_wedge_order3_values_jac_div", "vector_rt_generated",
            scaled_iterations(10000, requested_iterations), 100, bytes,
            [&]() {
                basis.evaluate_vector_values(xi, values);
                basis.evaluate_vector_jacobians(xi, jacobians);
                basis.evaluate_divergence(xi, divergence);
                consume(values[0][0] + jacobians[0](0, 0) + divergence[0]);
            }));
    }

    auto run_rt_strided_case =
        [&](ElementType element_type,
            int order,
            int quadrature_order,
            const char* name,
            std::size_t base_iterations) {
            RaviartThomasBasis basis(element_type, order);
            auto quad = svmp::FE::quadrature::QuadratureFactory::create(
                element_type, quadrature_order);
            const std::size_t stride = quad->num_points() + 1u;
            std::vector<Real> values(basis.size() * 3u * stride, Real(0));
            std::vector<Real> jacobians(basis.size() * 9u * stride, Real(0));
            std::vector<Real> divergence(basis.size() * stride, Real(0));
            const std::size_t bytes = basis.size() * (3u + 9u + 1u) * stride * sizeof(Real);
            print_result(run_case(
                name,
                "vector_rt_strided",
                scaled_iterations(base_iterations, requested_iterations),
                5,
                bytes,
                [&]() {
                    basis.evaluate_vector_at_quadrature_points_strided(
                        quad->points(),
                        stride,
                        values.data(),
                        jacobians.data(),
                        nullptr,
                        divergence.data());
                    consume(values[0] + jacobians[0] + divergence[0]);
                }));
        };

    run_rt_strided_case(
        ElementType::Wedge6, 3, 5, "rt_wedge_order3_strided_values_jac_div", 20);
    run_rt_strided_case(
        ElementType::Tetra4, 3, 5, "rt_tetra_order3_strided_values_jac_div", 50);
    run_rt_strided_case(
        ElementType::Pyramid5, 3, 5, "rt_pyramid_order3_strided_values_jac_div", 20);
    run_rt_strided_case(
        ElementType::Hex8, 3, 5, "rt_hex_order3_strided_values_jac_div", 50);

    {
        RaviartThomasBasis basis(ElementType::Tetra4, 2);
        Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(0.15)};
        std::vector<Vector<Real, 3>> values(basis.size());
        std::vector<VectorJacobian> jacobians(basis.size());
        std::vector<Real> divergence(basis.size());
        const std::size_t bytes = basis.size() * (3u + 9u + 1u) * sizeof(Real);
        print_result(run_case(
            "rt_tetra_order2_values_jac_div", "vector_rt_nodal",
            scaled_iterations(20000, requested_iterations), 100, bytes,
            [&]() {
                basis.evaluate_vector_values(xi, values);
                basis.evaluate_vector_jacobians(xi, jacobians);
                basis.evaluate_divergence(xi, divergence);
                consume(values[0][0] + jacobians[0](0, 0) + divergence[0]);
            }));
    }

    {
        NedelecBasis basis(ElementType::Wedge6, 3);
        Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(-0.1)};
        std::vector<Vector<Real, 3>> values(basis.size());
        std::vector<VectorJacobian> jacobians(basis.size());
        std::vector<Vector<Real, 3>> curl(basis.size());
        const std::size_t bytes = basis.size() * (3u + 9u + 3u) * sizeof(Real);
        print_result(run_case(
            "nedelec_wedge_order3_values_jac_curl", "vector_nedelec_generated",
            scaled_iterations(3000, requested_iterations), 50, bytes,
            [&]() {
                basis.evaluate_vector_values(xi, values);
                basis.evaluate_vector_jacobians(xi, jacobians);
                basis.evaluate_curl(xi, curl);
                consume(values[0][0] + jacobians[0](0, 0) + curl[0][0]);
            }));
    }

    auto run_nedelec_strided_case =
        [&](ElementType element_type,
            int order,
            int quadrature_order,
            const char* name,
            std::size_t base_iterations) {
            NedelecBasis basis(element_type, order);
            auto quad = svmp::FE::quadrature::QuadratureFactory::create(
                element_type, quadrature_order);
            const std::size_t stride = quad->num_points() + 1u;
            std::vector<Real> values(basis.size() * 3u * stride, Real(0));
            std::vector<Real> jacobians(basis.size() * 9u * stride, Real(0));
            std::vector<Real> curls(basis.size() * 3u * stride, Real(0));
            const std::size_t bytes = basis.size() * (3u + 9u + 3u) * stride * sizeof(Real);
            print_result(run_case(
                name,
                "vector_nedelec_strided",
                scaled_iterations(base_iterations, requested_iterations),
                5,
                bytes,
                [&]() {
                    basis.evaluate_vector_at_quadrature_points_strided(
                        quad->points(),
                        stride,
                        values.data(),
                        jacobians.data(),
                        curls.data(),
                        nullptr);
                    consume(values[0] + jacobians[0] + curls[0]);
                }));
        };

    run_nedelec_strided_case(
        ElementType::Wedge6, 3, 5, "nedelec_wedge_order3_strided_values_jac_curl", 10);
    run_nedelec_strided_case(
        ElementType::Tetra4, 3, 5, "nedelec_tetra_order3_strided_values_jac_curl", 20);
    run_nedelec_strided_case(
        ElementType::Pyramid5, 3, 5, "nedelec_pyramid_order3_strided_values_jac_curl", 10);
    run_nedelec_strided_case(
        ElementType::Hex8, 0, 3, "nedelec_hex_order0_strided_values_jac_curl", 50);

    {
        NedelecBasis basis(ElementType::Pyramid5, 2);
        Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.35)};
        std::vector<Vector<Real, 3>> values(basis.size());
        std::vector<VectorJacobian> jacobians(basis.size());
        std::vector<Vector<Real, 3>> curl(basis.size());
        const std::size_t bytes = basis.size() * (3u + 9u + 3u) * sizeof(Real);
        print_result(run_case(
            "nedelec_pyramid_order2_values_jac_curl", "vector_nedelec_pyramid",
            scaled_iterations(5000, requested_iterations), 50, bytes,
            [&]() {
                basis.evaluate_vector_values(xi, values);
                basis.evaluate_vector_jacobians(xi, jacobians);
                basis.evaluate_curl(xi, curl);
                consume(values[0][0] + jacobians[0](0, 0) + curl[0][0]);
            }));
    }

    run_nedelec_strided_case(
        ElementType::Pyramid5, 2, 4, "nedelec_pyramid_order2_strided_values_jac_curl", 20);

    {
        BDMBasis basis(ElementType::Tetra4, 2);
        Vector<Real, 3> xi{Real(0.2), Real(0.25), Real(0.15)};
        std::vector<Vector<Real, 3>> values(basis.size());
        std::vector<VectorJacobian> jacobians(basis.size());
        std::vector<Real> divergence(basis.size());
        const std::size_t bytes = basis.size() * (3u + 9u + 1u) * sizeof(Real);
        print_result(run_case(
            "bdm_tetra_order2_values_jac_div", "vector_bdm",
            scaled_iterations(20000, requested_iterations), 100, bytes,
            [&]() {
                basis.evaluate_vector_values(xi, values);
                basis.evaluate_vector_jacobians(xi, jacobians);
                basis.evaluate_divergence(xi, divergence);
                consume(values[0][0] + jacobians[0](0, 0) + divergence[0]);
            }));
    }

    {
        BDMBasis basis(ElementType::Tetra4, 2);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Tetra4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * 3u * stride, Real(0));
        std::vector<Real> jacobians(basis.size() * 9u * stride, Real(0));
        std::vector<Real> divergence(basis.size() * stride, Real(0));
        const std::size_t bytes = basis.size() * (3u + 9u + 1u) * stride * sizeof(Real);
        print_result(run_case(
            "bdm_tetra_order2_strided_values_jac_div",
            "vector_bdm_strided",
            scaled_iterations(50, requested_iterations),
            5,
            bytes,
            [&]() {
                basis.evaluate_vector_at_quadrature_points_strided(
                    quad->points(),
                    stride,
                    values.data(),
                    jacobians.data(),
                    nullptr,
                    divergence.data());
                consume(values[0] + jacobians[0] + divergence[0]);
            }));
    }

    {
        using SplineTensorBasis = TensorProductBasis<BSplineBasis>;
        auto first = std::make_shared<SplineTensorBasis>(
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()));
        auto second = std::make_shared<SplineTensorBasis>(
            BSplineBasis(3, cubic_open_knots()),
            BSplineBasis(3, cubic_open_knots()));
        std::vector<svmp::FE::basis::DofAssociation> associations(first->size() + second->size());
        CompatibleTensorVectorBasis basis(CompatibleTensorVectorBasis::Family::HDiv,
                                          svmp::FE::BasisType::BSpline,
                                          first,
                                          second,
                                          associations,
                                          3,
                                          ElementType::Quad4);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Quad4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * 3u * stride, Real(0));
        std::vector<Real> jacobians(basis.size() * 9u * stride, Real(0));
        std::vector<Real> divergence(basis.size() * stride, Real(0));
        const std::size_t bytes = basis.size() * (3u + 9u + 1u) * stride * sizeof(Real);
        print_result(run_case(
            "compatible_bspline_hdiv_quad_strided", "compatible_tensor_vector",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_vector_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), jacobians.data(), nullptr, divergence.data());
                consume(values[0] + jacobians[0] + divergence[0]);
            }));
    }

    {
        auto make_component = []() {
            return std::make_shared<NURBSTensorBasis>(
                BSplineBasis(3, cubic_open_knots()),
                BSplineBasis(3, cubic_open_knots()),
                tensor_nurbs_weights(25u));
        };
        auto first = make_component();
        auto second = make_component();
        std::vector<svmp::FE::basis::DofAssociation> associations(first->size() + second->size());
        CompatibleTensorVectorBasis basis(CompatibleTensorVectorBasis::Family::HCurl,
                                          svmp::FE::BasisType::NURBS,
                                          first,
                                          second,
                                          associations,
                                          3,
                                          ElementType::Quad4);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ElementType::Quad4, 4);
        const std::size_t stride = quad->num_points() + 1u;
        std::vector<Real> values(basis.size() * 3u * stride, Real(0));
        std::vector<Real> jacobians(basis.size() * 9u * stride, Real(0));
        std::vector<Real> curl(basis.size() * 3u * stride, Real(0));
        const std::size_t bytes = basis.size() * (3u + 9u + 3u) * stride * sizeof(Real);
        print_result(run_case(
            "compatible_nurbs_hcurl_quad_strided", "compatible_tensor_vector",
            scaled_iterations(100000, requested_iterations), 500, bytes,
            [&]() {
                basis.evaluate_vector_at_quadrature_points_strided(
                    quad->points(), stride, values.data(), jacobians.data(), curl.data(), nullptr);
                consume(values[0] + jacobians[0] + curl[0]);
            }));
    }

    return 0;
}
