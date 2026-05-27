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
#include <chrono>
#include <cstddef>
#include <cstdlib>
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
    std::size_t repeats;
    double min_seconds;
    double max_seconds;
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
              << static_cast<double>(g_sink) << ','
              << r.repeats << ','
              << min_ns_per_call << ','
              << max_ns_per_call << '\n';
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
                 "allocations_per_call,estimated_bytes_per_call,sink,"
                 "repeats,min_ns_per_call,max_ns_per_call\n";

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
        2, 4, "lagrange_pyramid_order2_strided_values", true, false, false, 4000);
    run_lagrange_pyramid_strided_case(
        2, 4, "lagrange_pyramid_order2_strided_values_gradients", true, true, false, 3000);
    run_lagrange_pyramid_strided_case(
        2, 4, "lagrange_pyramid_order2_strided_hessians", false, false, true, 2000);
    run_lagrange_pyramid_strided_case(
        5, 6, "lagrange_pyramid_order5_strided_all", true, true, true, 500);

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
