// Performance comparison harness: legacy nn:: vs OOP svmp::FE::basis::LagrangeBasis.
//
// Sections (mirrors the migration plan):
//   3.1 MicrobenchmarkPointwise   - single-point evaluate_{values,gradients,hessians}
//   3.2 SetupCost                 - all-QP basis data fill (legacy mesh init / OOP cache)
//   3.3 SteadyStateAccess         - mock element loop with 4 access patterns
//   3.4 FusedKernels              - mass / stiffness / convection element matrices
//   3.6 CacheEffectiveness        - OOP-only: cache hit/miss latency
//   3.7 ParallelScaling           - std::thread scaling (OpenMP unavailable in this build)
//
// Section 3.5 (affine vs curved) is intentionally out of scope: that is a geometry
// property (Jacobian recomputation), not a basis property. Both basis APIs are
// pure reference-space evaluators; AssemblyContext owns the geometry transform.
//
// All tests gate on SVMP_FE_RUN_PERF_TESTS=1 so they don't run under the normal
// regression suite. Output goes to $SVMP_BASIS_COMPARE_OUT/perf/ (default
// ./basis_comparison_output/perf/).

#include <gtest/gtest.h>

#include "nn.h"
#include "Array.h"
#include "Array3.h"
#include "Vector.h"
#include "consts.h"

#include "FE/Basis/BasisCache.h"
#include "FE/Basis/BatchEvaluator.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Assembly/BatchedStiffness.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/QuadratureRule.h"
#include "FE/Core/Types.h"
#include "Math/Vector.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// Timing utilities (kept local to avoid pulling in PerfTestHelpers.h which
// lives in a different test target).
// -----------------------------------------------------------------------------
[[nodiscard]] inline bool perfEnabled() {
    const char* v = std::getenv("SVMP_FE_RUN_PERF_TESTS");
    return v != nullptr && std::string(v) == "1";
}

using Clock = std::chrono::steady_clock;

template <class Fn>
[[nodiscard]] inline double timeSeconds(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

template <class Fn>
[[nodiscard]] inline double bestOfSeconds(int repeats, Fn&& fn) {
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < repeats; ++r) best = std::min(best, timeSeconds(fn));
    return best;
}

std::filesystem::path output_dir() {
    if (const char* env = std::getenv("SVMP_BASIS_COMPARE_OUT")) {
        return std::filesystem::path{env} / "perf";
    }
    return std::filesystem::path{"basis_comparison_output/perf"};
}

// Volatile sinks to defeat dead-code elimination.
volatile double g_sink_d = 0.0;
volatile std::int64_t g_sink_i = 0;

struct ElementCase {
    consts::ElementType legacy_type;
    svmp::FE::ElementType oop_type;
    int eNoN;
    int insd;
    int order;
    int legacy_nG;     // legacy quadrature point count
    int quad_degree;   // OOP quadrature exactness
    bool legacy_has_hessian;
    const char* name;
};

// legacy_has_hessian: only QUD8/QUD9/TET10/TRI6 are present in nn_elem_gnnxx.h.
// legacy_in_meshfree_gip: nn::get_gip mesh-free path covers TRI3/TET4/HEX8/HEX20/HEX27/WDG;
//                         TRI6 and TET10 require the mesh-bound API (out of scope for these benches).
const std::vector<ElementCase> kCases = {
    {consts::ElementType::TRI3,  svmp::FE::ElementType::Triangle3,  3,  2, 1,  3,  2, false, "TRI3"},
    {consts::ElementType::TRI6,  svmp::FE::ElementType::Triangle6,  6,  2, 2,  7,  4, true,  "TRI6"},
    {consts::ElementType::TET4,  svmp::FE::ElementType::Tetra4,     4,  3, 1,  4,  2, false, "TET4"},
    {consts::ElementType::TET10, svmp::FE::ElementType::Tetra10,   10,  3, 2, 15,  4, true,  "TET10"},
    {consts::ElementType::HEX8,  svmp::FE::ElementType::Hex8,       8,  3, 1,  8,  2, false, "HEX8"},
    {consts::ElementType::HEX27, svmp::FE::ElementType::Hex27,     27,  3, 2, 27,  4, false, "HEX27"},
};

// True iff nn::get_gip mesh-free path supports this element (TRI3/TET4/HEX8/HEX27).
// TRI6 and TET10 only have entries in the mesh-bound set_element_gauss_int_data.
inline bool legacy_meshfree_gip(consts::ElementType t) {
    return t == consts::ElementType::TRI3 ||
           t == consts::ElementType::TET4 ||
           t == consts::ElementType::HEX8 ||
           t == consts::ElementType::HEX27;
}

// Return a representative interior reference point for each element.
std::array<double, 3> centroid(const ElementCase& ec) {
    if (ec.insd == 2 && ec.legacy_type == consts::ElementType::TRI3) return {1.0/3, 1.0/3, 0};
    if (ec.insd == 2 && ec.legacy_type == consts::ElementType::TRI6) return {1.0/3, 1.0/3, 0};
    if (ec.insd == 3 && (ec.legacy_type == consts::ElementType::TET4 ||
                          ec.legacy_type == consts::ElementType::TET10))
        return {0.25, 0.25, 0.25};
    return {0.0, 0.0, 0.0};
}

// Pack a single xi point into a legacy Array<double>(insd, 1).
Array<double> pack_xi(int insd, const std::array<double, 3>& xi) {
    Array<double> out(insd, 1);
    for (int d = 0; d < insd; ++d) out(d, 0) = xi[d];
    return out;
}

// Build a legacy xi array from an OOP quadrature rule.
Array<double> legacy_xi_from_quad(const svmp::FE::quadrature::QuadratureRule& quad,
                                  int insd) {
    int nQ = static_cast<int>(quad.num_points());
    Array<double> xi(insd, nQ);
    for (int q = 0; q < nQ; ++q) {
        const auto& p = quad.point(q);
        for (int d = 0; d < insd; ++d) xi(d, q) = p[d];
    }
    return xi;
}

class LagrangeBasisPerformance : public ::testing::Test {
protected:
    void SetUp() override {
        if (!perfEnabled())
            GTEST_SKIP() << "set SVMP_FE_RUN_PERF_TESTS=1 to enable performance harness.";
        std::filesystem::create_directories(output_dir());
    }
};

}  // namespace

// =============================================================================
// 3.1 MicrobenchmarkPointwise
//
// Time a single-point evaluation of {values, gradients, hessians} on each
// implementation, repeated in a tight loop to amortize timer overhead.
// Reports ns / call per (element, operation, implementation).
// =============================================================================
TEST_F(LagrangeBasisPerformance, MicrobenchmarkPointwise) {
    std::ofstream csv(output_dir() / "perf_microbench_pointwise.csv");
    csv << std::setprecision(10);
    csv << "elem_type,operation,implementation,iterations,seconds,ns_per_call\n";

    const int iters = 500000;
    const int repeats = 5;

    for (const auto& ec : kCases) {
        svmp::FE::basis::LagrangeBasis oop(ec.oop_type, ec.order);
        auto pt = centroid(ec);

        // --- Legacy values + gradients (combined in get_gnn) -----------------
        {
            Array<double> xi = pack_xi(ec.insd, pt);
            Array<double> N(ec.eNoN, 1);
            Array3<double> Nx(ec.insd, ec.eNoN, 1);
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi, N, Nx);
                    acc += N(0, 0) + Nx(0, 0, 0);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",values_and_gradients,legacy," << iters << ","
                << s << "," << (s * 1e9 / iters) << "\n";
        }

        // --- OOP values only -------------------------------------------------
        {
            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
            std::vector<svmp::FE::Real> values;
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    oop.evaluate_values(xi_oop, values);
                    acc += static_cast<double>(values[0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",values,oop," << iters << ","
                << s << "," << (s * 1e9 / iters) << "\n";
        }

        // --- OOP gradients ---------------------------------------------------
        {
            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
            std::vector<svmp::FE::basis::Gradient> grads;
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    oop.evaluate_gradients(xi_oop, grads);
                    acc += static_cast<double>(grads[0][0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",gradients,oop," << iters << ","
                << s << "," << (s * 1e9 / iters) << "\n";
        }

        // --- OOP hessians ----------------------------------------------------
        {
            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
            std::vector<svmp::FE::basis::Hessian> hess;
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    oop.evaluate_hessians(xi_oop, hess);
                    acc += static_cast<double>(hess[0](0, 0));
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",hessians,oop," << iters << ","
                << s << "," << (s * 1e9 / iters) << "\n";
        }

        // --- Legacy hessians (only some element types) -----------------------
        if (ec.legacy_has_hessian) {
            const int ind2 = (ec.insd == 2) ? 3 : 6;
            Array<double> xi = pack_xi(ec.insd, pt);
            Array3<double> Nxx(ind2, ec.eNoN, 1);
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    nn::get_gn_nxx(ec.insd, ind2, ec.legacy_type, ec.eNoN, 0, xi, Nxx);
                    acc += Nxx(0, 0, 0);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",hessians,legacy," << iters << ","
                << s << "," << (s * 1e9 / iters) << "\n";
        }

        std::cout << "[3.1 " << ec.name << "] microbench done\n";
    }
}

// =============================================================================
// 3.2 SetupCost
//
// Time the all-QP basis data fill on each implementation.
//   Legacy: nn::get_gip + per-QP nn::get_gnn (matches what mesh init does).
//   OOP cold: BasisCache::clear() then get_or_compute(...).
//   OOP warm: repeated get_or_compute(...) (cache hit, should be ~free).
// =============================================================================
TEST_F(LagrangeBasisPerformance, SetupCost) {
    std::ofstream csv(output_dir() / "perf_setup_cost.csv");
    csv << std::setprecision(10);
    csv << "elem_type,eNoN,nG,implementation,iterations,seconds,ns_per_setup\n";

    const int repeats = 7;

    for (const auto& ec : kCases) {
        // Build OOP basis + quadrature once
        auto oop = std::make_shared<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(
            ec.oop_type, ec.quad_degree);

        // Number of legacy QPs we'll fill (use the legacy default for the element)
        const int nG = ec.legacy_nG;

        // Iterations chosen so total wall time is ~1 second per setup pattern
        const int iters_legacy = 5000;
        const int iters_oop_cold = 200;  // each iter clears the cache
        const int iters_oop_warm = 200000;

        // --- Legacy: full mesh-init shape data fill --------------------------
        // Skip TRI6/TET10: they require the mesh-bound API which needs a mshType.
        if (legacy_meshfree_gip(ec.legacy_type)) {
            Vector<double> w(nG);
            Array<double> xi_a(ec.insd, nG);
            Array<double> N(ec.eNoN, nG);
            Array3<double> Nx(ec.insd, ec.eNoN, nG);
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters_legacy; ++i) {
                    nn::get_gip(ec.insd, ec.legacy_type, nG, w, xi_a);
                    for (int g = 0; g < nG; ++g) {
                        nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, g, xi_a, N, Nx);
                    }
                    acc += w(0) + N(0, 0) + Nx(0, 0, 0);
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << nG << ",legacy_full_fill,"
                << iters_legacy << "," << s << ","
                << (s * 1e9 / iters_legacy) << "\n";
        }

        // --- OOP cold (clear cache each iteration) ---------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters_oop_cold; ++i) {
                    svmp::FE::basis::BasisCache::instance().clear();
                    const auto& entry =
                        svmp::FE::basis::BasisCache::instance().get_or_compute(
                            *oop, *quad, /*grads=*/true, /*hess=*/false);
                    acc += static_cast<double>(entry.scalar_values[0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << nG << ",oop_cache_cold,"
                << iters_oop_cold << "," << s << ","
                << (s * 1e9 / iters_oop_cold) << "\n";
        }

        // --- OOP warm (cache hit) -------------------------------------------
        {
            // Prime the cache once
            (void)svmp::FE::basis::BasisCache::instance().get_or_compute(
                *oop, *quad, /*grads=*/true, /*hess=*/false);
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters_oop_warm; ++i) {
                    const auto& entry =
                        svmp::FE::basis::BasisCache::instance().get_or_compute(
                            *oop, *quad, /*grads=*/true, /*hess=*/false);
                    acc += static_cast<double>(entry.scalar_values[0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << nG << ",oop_cache_warm,"
                << iters_oop_warm << "," << s << ","
                << (s * 1e9 / iters_oop_warm) << "\n";
        }

        std::cout << "[3.2 " << ec.name << "] setup-cost done\n";
    }
}

// =============================================================================
// 3.3 SteadyStateAccess
//
// Mock assembly hot loop. For N elements x nG QPs x eNoN basis functions,
// accumulate sum N_i(q) * coeff[i] into a scalar sink. Compare 4 access
// patterns:
//   A. legacy: read from precomputed (N, Nx) arrays (mesh-style storage)
//   B. OOP via BasisCache: random access scalarValue(dof, qp)
//   C. OOP via BasisCache: contiguous span scalarValuesForDof(dof)
//   D. OOP via BatchEvaluator: SIMD-aligned values_for_basis(dof)
// =============================================================================
TEST_F(LagrangeBasisPerformance, SteadyStateAccess) {
    std::ofstream csv(output_dir() / "perf_steady_state_access.csv");
    csv << std::setprecision(10);
    csv << "elem_type,eNoN,nG,access_pattern,n_elements,seconds,ns_per_element\n";

    const int n_elem = 20000;
    const int repeats = 5;

    for (const auto& ec : kCases) {
        // Use OOP quadrature for both implementations so the QP set matches and
        // we can run all elements (legacy mesh-free gip lacks TRI6, TET10).
        auto oop = std::make_shared<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(
            ec.oop_type, ec.quad_degree);
        const int nG = static_cast<int>(quad->num_points());

        // Per-element coefficients (same for every element to simplify)
        std::vector<double> coeffs(ec.eNoN);
        for (int i = 0; i < ec.eNoN; ++i)
            coeffs[i] = 0.5 + 0.1 * std::sin(double(i));

        // ---------- Legacy storage (precomputed mesh.N, mesh.Nx style) -------
        Array<double> xi_leg = legacy_xi_from_quad(*quad, ec.insd);
        Array<double> N_leg_all(ec.eNoN, nG);
        Array3<double> Nx_leg_all(ec.insd, ec.eNoN, nG);
        for (int g = 0; g < nG; ++g) {
            nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, g, xi_leg, N_leg_all, Nx_leg_all);
        }

        // ---------- OOP cache + batch ----------------------------------------
        const auto& cache_entry =
            svmp::FE::basis::BasisCache::instance().get_or_compute(
                *oop, *quad, /*grads=*/true, /*hess=*/false);
        svmp::FE::basis::BatchEvaluator batch(*oop, *quad, /*grads=*/true);
        const auto& batch_data = batch.data();

        const std::size_t cache_nQ = cache_entry.num_qpts;
        const std::size_t cache_nD = cache_entry.num_dofs;

        // ----- A. Legacy random access (mesh.N(a, g)) ------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    for (int g = 0; g < nG; ++g) {
                        for (int a = 0; a < ec.eNoN; ++a) {
                            acc += coeffs[a] * N_leg_all(a, g);
                        }
                    }
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << nG << ",legacy_array,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ----- B. OOP cache, random access scalarValue(dof, qp) --------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    for (std::size_t g = 0; g < cache_nQ; ++g) {
                        for (std::size_t a = 0; a < cache_nD; ++a) {
                            acc += coeffs[a] *
                                   static_cast<double>(cache_entry.scalarValue(a, g));
                        }
                    }
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << cache_nQ
                << ",oop_cache_random,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ----- C. OOP cache, contiguous span scalarValuesForDof --------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    for (std::size_t a = 0; a < cache_nD; ++a) {
                        auto span = cache_entry.scalarValuesForDof(a);
                        double cf = coeffs[a];
                        for (std::size_t g = 0; g < span.size(); ++g) {
                            acc += cf * static_cast<double>(span[g]);
                        }
                    }
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << cache_nQ
                << ",oop_cache_span,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ----- D. OOP BatchEvaluator: SIMD-aligned values_for_basis ----------
        {
            const std::size_t bnQ = batch_data.num_quad_points;
            const std::size_t bnD = batch_data.num_basis;
            double s = bestOfSeconds(repeats, [&]() {
                double acc = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    for (std::size_t a = 0; a < bnD; ++a) {
                        const svmp::FE::Real* vals = batch_data.values_for_basis(a);
                        double cf = coeffs[a];
                        for (std::size_t g = 0; g < bnQ; ++g) {
                            acc += cf * static_cast<double>(vals[g]);
                        }
                    }
                }
                g_sink_d = acc;
            });
            csv << ec.name << "," << ec.eNoN << "," << bnQ
                << ",oop_batch_aligned,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        std::cout << "[3.3 " << ec.name << "] steady-state done\n";
    }
}

// =============================================================================
// 3.4 FusedKernels
//
// Element-level matrix assembly for three operators:
//   - Mass:        M_ij = sum_q w_q * N_i(q) * N_j(q)
//   - Stiffness:   K_ij = sum_q w_q * grad N_i(q) . grad N_j(q)
//   - Convection:  C_ij = sum_q w_q * N_i(q) * (b . grad N_j(q))
//
// Implementations compared:
//   * legacy_manual : pre-computed legacy N/Nx arrays + handwritten triple loop
//   * oop_manual    : OOP BasisCache values + handwritten triple loop
//   * oop_fused     : OOP assembly::assemble_stiffness_contribution (stiffness only)
//
// Both legacy and OOP-manual paths use the SAME quadrature (built from the OOP rule)
// so that the comparison is on basis-evaluation/storage cost, not on integration
// accuracy.
// =============================================================================
TEST_F(LagrangeBasisPerformance, FusedKernels) {
    std::ofstream csv(output_dir() / "perf_fused_kernels.csv");
    csv << std::setprecision(10);
    csv << "elem_type,eNoN,nG,operator,implementation,n_elements,seconds,ns_per_element\n";

    const int n_elem = 5000;
    const int repeats = 5;

    for (const auto& ec : kCases) {
        // Use the OOP quadrature so legacy + OOP do equivalent work.
        auto oop = std::make_shared<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(
            ec.oop_type, ec.quad_degree);
        const int nQ = static_cast<int>(quad->num_points());

        std::vector<double> w(nQ);
        for (int q = 0; q < nQ; ++q) w[q] = static_cast<double>(quad->weight(q));
        Array<double> xi_leg = legacy_xi_from_quad(*quad, ec.insd);

        // Pre-compute legacy N, Nx arrays at all OOP-quadrature points
        Array<double>  N_leg(ec.eNoN, nQ);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, nQ);
        for (int g = 0; g < nQ; ++g) {
            nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, g, xi_leg, N_leg, Nx_leg);
        }

        // OOP cache + batch
        const auto& cache_entry =
            svmp::FE::basis::BasisCache::instance().get_or_compute(
                *oop, *quad, /*grads=*/true, /*hess=*/false);
        svmp::FE::basis::BatchEvaluator batch(*oop, *quad, /*grads=*/true);
        const auto& bd = batch.data();

        const int n = ec.eNoN;
        const int dim = ec.insd;

        // Convective velocity vector (constant per element, fine for benchmarking)
        std::array<double, 3> b_vec{1.2, -0.7, 0.4};
        std::array<svmp::FE::Real, 9> D_iden{1, 0, 0, 0, 1, 0, 0, 0, 1};

        std::vector<double> M(n * n);
        std::vector<double> K(n * n);
        std::vector<double> C(n * n);
        std::vector<svmp::FE::Real> K_oop(n * n);

        // ------- MASS, legacy_manual ---------------------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(M.begin(), M.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            double Na = N_leg(a, q);
                            for (int bb = 0; bb < n; ++bb) {
                                M[a * n + bb] += wq * Na * N_leg(bb, q);
                            }
                        }
                    }
                    sink += M[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",mass,legacy_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- MASS, oop_manual (using cache) ----------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(M.begin(), M.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            double Na = static_cast<double>(cache_entry.scalarValue(a, q));
                            for (int bb = 0; bb < n; ++bb) {
                                double Nb = static_cast<double>(cache_entry.scalarValue(bb, q));
                                M[a * n + bb] += wq * Na * Nb;
                            }
                        }
                    }
                    sink += M[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",mass,oop_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- STIFFNESS, legacy_manual ----------------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(K.begin(), K.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            for (int bb = 0; bb < n; ++bb) {
                                double dot = 0.0;
                                for (int d = 0; d < dim; ++d) {
                                    dot += Nx_leg(d, a, q) * Nx_leg(d, bb, q);
                                }
                                K[a * n + bb] += wq * dot;
                            }
                        }
                    }
                    sink += K[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",stiffness,legacy_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- STIFFNESS, oop_manual (cache) -----------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(K.begin(), K.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            for (int bb = 0; bb < n; ++bb) {
                                double dot = 0.0;
                                for (int d = 0; d < dim; ++d)
                                    dot += static_cast<double>(cache_entry.gradientValue(
                                                static_cast<std::size_t>(a),
                                                static_cast<std::size_t>(d),
                                                static_cast<std::size_t>(q))) *
                                           static_cast<double>(cache_entry.gradientValue(
                                                static_cast<std::size_t>(bb),
                                                static_cast<std::size_t>(d),
                                                static_cast<std::size_t>(q)));
                                K[a * n + bb] += wq * dot;
                            }
                        }
                    }
                    sink += K[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",stiffness,oop_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- STIFFNESS, oop_fused (BatchEvaluator) ----------------------
        {
            std::vector<svmp::FE::Real> wrt(nQ);
            for (int q = 0; q < nQ; ++q) wrt[q] = static_cast<svmp::FE::Real>(w[q]);
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(K_oop.begin(), K_oop.end(), svmp::FE::Real(0));
                    svmp::FE::assembly::assemble_stiffness_contribution(
                        batch, D_iden.data(), wrt.data(), K_oop.data());
                    sink += static_cast<double>(K_oop[0]);
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",stiffness,oop_fused,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- CONVECTION, legacy_manual ---------------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(C.begin(), C.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            double Na = N_leg(a, q);
                            for (int bb = 0; bb < n; ++bb) {
                                double bdotg = 0.0;
                                for (int d = 0; d < dim; ++d) {
                                    bdotg += b_vec[d] * Nx_leg(d, bb, q);
                                }
                                C[a * n + bb] += wq * Na * bdotg;
                            }
                        }
                    }
                    sink += C[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",convection,legacy_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        // ------- CONVECTION, oop_manual (cache) ----------------------------
        {
            double s = bestOfSeconds(repeats, [&]() {
                double sink = 0.0;
                for (int e = 0; e < n_elem; ++e) {
                    std::fill(C.begin(), C.end(), 0.0);
                    for (int q = 0; q < nQ; ++q) {
                        double wq = w[q];
                        for (int a = 0; a < n; ++a) {
                            double Na = static_cast<double>(cache_entry.scalarValue(a, q));
                            for (int bb = 0; bb < n; ++bb) {
                                double bdotg = 0.0;
                                for (int d = 0; d < dim; ++d)
                                    bdotg += b_vec[d] * static_cast<double>(
                                        cache_entry.gradientValue(static_cast<std::size_t>(bb),
                                                                  static_cast<std::size_t>(d),
                                                                  static_cast<std::size_t>(q)));
                                C[a * n + bb] += wq * Na * bdotg;
                            }
                        }
                    }
                    sink += C[0];
                }
                g_sink_d = sink;
            });
            csv << ec.name << "," << n << "," << nQ << ",convection,oop_manual,"
                << n_elem << "," << s << "," << (s * 1e9 / n_elem) << "\n";
        }

        std::cout << "[3.4 " << ec.name << "] fused kernels done\n";
    }
}

// =============================================================================
// 3.6 CacheEffectiveness   (OOP-only)
//
// Measures BasisCache hit/miss latency:
//   - cold: clear cache, time first get_or_compute()
//   - warm: time subsequent get_or_compute() lookups (lock-free read after compute)
//   - mixed-mesh: cache holds entries for {Triangle3, Hex8, Tetra4} simultaneously;
//                 round-robin lookups should all be hits.
// =============================================================================
TEST_F(LagrangeBasisPerformance, CacheEffectiveness) {
    std::ofstream csv(output_dir() / "perf_cache_effectiveness.csv");
    csv << std::setprecision(10);
    csv << "elem_type,scenario,iterations,seconds,ns_per_call\n";

    auto& cache = svmp::FE::basis::BasisCache::instance();

    for (const auto& ec : kCases) {
        auto oop = std::make_shared<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(
            ec.oop_type, ec.quad_degree);

        // --- cold: cache cleared each iteration ---
        {
            const int iters = 200;
            double s = bestOfSeconds(5, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    cache.clear();
                    const auto& e = cache.get_or_compute(*oop, *quad, true, false);
                    acc += static_cast<double>(e.scalar_values[0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",cold," << iters << "," << s << ","
                << (s * 1e9 / iters) << "\n";
        }

        // --- warm: prime once, then time many hits ---
        {
            (void)cache.get_or_compute(*oop, *quad, true, false);
            const int iters = 500000;
            double s = bestOfSeconds(5, [&]() {
                double acc = 0.0;
                for (int i = 0; i < iters; ++i) {
                    const auto& e = cache.get_or_compute(*oop, *quad, true, false);
                    acc += static_cast<double>(e.scalar_values[0]);
                }
                g_sink_d = acc;
            });
            csv << ec.name << ",warm," << iters << "," << s << ","
                << (s * 1e9 / iters) << "\n";
        }
    }

    // --- mixed-mesh: prime cache with 3 distinct (basis, quad) entries,
    //                 then round-robin hits.
    {
        cache.clear();
        std::vector<std::shared_ptr<svmp::FE::basis::BasisFunction>> bases;
        std::vector<std::shared_ptr<const svmp::FE::quadrature::QuadratureRule>> quads;
        bases.push_back(std::make_shared<svmp::FE::basis::LagrangeBasis>(
            svmp::FE::ElementType::Triangle3, 1));
        quads.push_back(svmp::FE::quadrature::QuadratureFactory::create(
            svmp::FE::ElementType::Triangle3, 2));
        bases.push_back(std::make_shared<svmp::FE::basis::LagrangeBasis>(
            svmp::FE::ElementType::Tetra4, 1));
        quads.push_back(svmp::FE::quadrature::QuadratureFactory::create(
            svmp::FE::ElementType::Tetra4, 2));
        bases.push_back(std::make_shared<svmp::FE::basis::LagrangeBasis>(
            svmp::FE::ElementType::Hex8, 1));
        quads.push_back(svmp::FE::quadrature::QuadratureFactory::create(
            svmp::FE::ElementType::Hex8, 2));
        for (std::size_t i = 0; i < bases.size(); ++i)
            (void)cache.get_or_compute(*bases[i], *quads[i], true, false);

        const int iters = 300000;
        double s = bestOfSeconds(5, [&]() {
            double acc = 0.0;
            for (int i = 0; i < iters; ++i) {
                const auto& e = cache.get_or_compute(
                    *bases[i % bases.size()], *quads[i % quads.size()],
                    true, false);
                acc += static_cast<double>(e.scalar_values[0]);
            }
            g_sink_d = acc;
        });
        csv << "MIXED,mixed_round_robin," << iters << "," << s << ","
            << (s * 1e9 / iters) << "\n";
    }

    std::cout << "[3.6] cache-effectiveness done\n";
}

// =============================================================================
// 3.7 ParallelScaling
//
// Each thread runs the same steady-state element loop independently. We
// measure aggregate throughput vs thread count.
//
// We use std::thread directly because OpenMP is not enabled in this build
// (per cmake configure: "FE: OpenMP not found").
// =============================================================================
TEST_F(LagrangeBasisPerformance, ParallelScaling) {
    std::ofstream csv(output_dir() / "perf_parallel_scaling.csv");
    csv << std::setprecision(10);
    csv << "elem_type,implementation,n_threads,n_elements_per_thread,seconds,"
           "elements_per_second_total\n";

    const int n_elem = 10000;
    const int repeats = 5;
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    for (const auto& ec : kCases) {
        // OOP basis + matching quadrature (used for both legacy and OOP paths)
        auto oop = std::make_shared<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(
            ec.oop_type, ec.quad_degree);
        const int nG = static_cast<int>(quad->num_points());

        // Legacy precomputed N at OOP quadrature points
        Array<double> xi_leg = legacy_xi_from_quad(*quad, ec.insd);
        Array<double> N_leg(ec.eNoN, nG);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, nG);
        for (int g = 0; g < nG; ++g)
            nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, g, xi_leg, N_leg, Nx_leg);

        // OOP cache (read-only after first compute is lock-free)
        const auto& cache_entry =
            svmp::FE::basis::BasisCache::instance().get_or_compute(
                *oop, *quad, /*grads=*/true, /*hess=*/false);
        const std::size_t cache_nQ = cache_entry.num_qpts;
        const std::size_t cache_nD = cache_entry.num_dofs;

        std::vector<double> coeffs(ec.eNoN);
        for (int i = 0; i < ec.eNoN; ++i)
            coeffs[i] = 0.5 + 0.1 * std::sin(double(i));

        for (int n_threads : thread_counts) {
            // ---- Legacy ----
            {
                double s = bestOfSeconds(repeats, [&]() {
                    std::vector<std::thread> ths;
                    std::vector<double> sinks(n_threads, 0.0);
                    for (int t = 0; t < n_threads; ++t) {
                        ths.emplace_back([&, t]() {
                            double acc = 0.0;
                            for (int e = 0; e < n_elem; ++e) {
                                for (int g = 0; g < nG; ++g) {
                                    for (int a = 0; a < ec.eNoN; ++a) {
                                        acc += coeffs[a] * N_leg(a, g);
                                    }
                                }
                            }
                            sinks[t] = acc;
                        });
                    }
                    for (auto& th : ths) th.join();
                    double sum = 0.0;
                    for (auto x : sinks) sum += x;
                    g_sink_d = sum;
                });
                double total_elem = double(n_elem) * n_threads;
                csv << ec.name << ",legacy," << n_threads << "," << n_elem
                    << "," << s << "," << (total_elem / s) << "\n";
            }

            // ---- OOP cache (span access) ----
            {
                double s = bestOfSeconds(repeats, [&]() {
                    std::vector<std::thread> ths;
                    std::vector<double> sinks(n_threads, 0.0);
                    for (int t = 0; t < n_threads; ++t) {
                        ths.emplace_back([&, t]() {
                            double acc = 0.0;
                            for (int e = 0; e < n_elem; ++e) {
                                for (std::size_t a = 0; a < cache_nD; ++a) {
                                    auto span = cache_entry.scalarValuesForDof(a);
                                    double cf = coeffs[a];
                                    for (std::size_t g = 0; g < span.size(); ++g) {
                                        acc += cf * static_cast<double>(span[g]);
                                    }
                                }
                            }
                            sinks[t] = acc;
                        });
                    }
                    for (auto& th : ths) th.join();
                    double sum = 0.0;
                    for (auto x : sinks) sum += x;
                    g_sink_d = sum;
                });
                double total_elem = double(n_elem) * n_threads;
                csv << ec.name << ",oop_cache_span," << n_threads << "," << n_elem
                    << "," << s << "," << (total_elem / s) << "\n";
            }
        }

        std::cout << "[3.7 " << ec.name << "] parallel scaling done\n";
    }
}

// -----------------------------------------------------------------------------
// Pζ1: evaluate_all (fused values+gradients+hessians) vs three separate calls.
// Quantifies the B4 fused-evaluator savings explicitly.
// -----------------------------------------------------------------------------

TEST_F(LagrangeBasisPerformance, EvaluateAllVsSeparate) {
    std::ofstream csv(output_dir() / "perf_evaluate_all_vs_separate.csv");
    csv << "elem_type,path,iterations,seconds,ns_per_call\n";

    constexpr int kIterations = 200000;
    constexpr int kRepeats = 7;

    for (const auto& ec : kCases) {
        auto basis = std::make_unique<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        const auto pt_arr = centroid(ec);
        svmp::FE::math::Vector<svmp::FE::Real, 3> pt;
        pt[0] = pt_arr[0]; pt[1] = pt_arr[1]; pt[2] = pt_arr[2];

        std::vector<svmp::FE::Real> values(basis->size());
        std::vector<svmp::FE::basis::Gradient> gradients(basis->size());
        std::vector<svmp::FE::basis::Hessian> hessians(basis->size());

        // Path A: three separate virtual calls.
        double sec_separate = bestOfSeconds(kRepeats, [&]() {
            for (int i = 0; i < kIterations; ++i) {
                basis->evaluate_values(pt, values);
                basis->evaluate_gradients(pt, gradients);
                basis->evaluate_hessians(pt, hessians);
                g_sink_d += values[0] + gradients[0][0] + hessians[0](0, 0);
            }
        });

        // Path B: single fused evaluate_all call.
        double sec_fused = bestOfSeconds(kRepeats, [&]() {
            for (int i = 0; i < kIterations; ++i) {
                basis->evaluate_all(pt, values, gradients, hessians);
                g_sink_d += values[0] + gradients[0][0] + hessians[0](0, 0);
            }
        });

        csv << ec.name << ",separate," << kIterations << "," << sec_separate
            << "," << (sec_separate * 1e9 / kIterations) << "\n";
        csv << ec.name << ",evaluate_all," << kIterations << "," << sec_fused
            << "," << (sec_fused * 1e9 / kIterations) << "\n";

        std::cout << "[Pzeta1 " << ec.name << "] separate=" << (sec_separate * 1e9 / kIterations)
                  << " ns, evaluate_all=" << (sec_fused * 1e9 / kIterations) << " ns\n";
    }
}

// -----------------------------------------------------------------------------
// Pζ2: evaluate_at_quadrature_points (multi-QP entry) vs per-QP loop.
// Quantifies the E3-partial multi-QP amortization on top of evaluate_all.
// -----------------------------------------------------------------------------

TEST_F(LagrangeBasisPerformance, MultiQpVsPerQpLoop) {
    std::ofstream csv(output_dir() / "perf_multi_qp_vs_per_qp.csv");
    csv << "elem_type,path,n_qpts,iterations,seconds,ns_per_qp\n";

    constexpr int kIterations = 5000;
    constexpr int kRepeats = 7;

    for (const auto& ec : kCases) {
        auto basis = std::make_unique<svmp::FE::basis::LagrangeBasis>(ec.oop_type, ec.order);
        auto quad = svmp::FE::quadrature::QuadratureFactory::create(ec.oop_type, ec.quad_degree);
        const auto& points = quad->points();
        const std::size_t n_qpts = points.size();
        const std::size_t n_dofs = basis->size();

        // Path A: per-QP loop with evaluate_all.
        std::vector<svmp::FE::Real> values_tmp(n_dofs);
        std::vector<svmp::FE::basis::Gradient> gradients_tmp(n_dofs);
        std::vector<svmp::FE::basis::Hessian> hessians_tmp(n_dofs);

        std::vector<svmp::FE::Real> values_soa(n_dofs * n_qpts);
        std::vector<svmp::FE::Real> grads_soa(n_dofs * 3 * n_qpts);
        std::vector<svmp::FE::Real> hess_soa(n_dofs * 9 * n_qpts);

        double sec_per_qp = bestOfSeconds(kRepeats, [&]() {
            for (int it = 0; it < kIterations; ++it) {
                for (std::size_t q = 0; q < n_qpts; ++q) {
                    basis->evaluate_all(points[q], values_tmp, gradients_tmp, hessians_tmp);
                    for (std::size_t d = 0; d < n_dofs; ++d) {
                        values_soa[d * n_qpts + q] = values_tmp[d];
                        for (int c = 0; c < 3; ++c) {
                            grads_soa[(d * 3 + c) * n_qpts + q] = gradients_tmp[d][c];
                        }
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                hess_soa[(d * 9 + r * 3 + c) * n_qpts + q] =
                                    hessians_tmp[d](r, c);
                            }
                        }
                    }
                }
                g_sink_d += values_soa[0] + grads_soa[0] + hess_soa[0];
            }
        });

        // Path B: single multi-QP entry call.
        double sec_multi = bestOfSeconds(kRepeats, [&]() {
            for (int it = 0; it < kIterations; ++it) {
                basis->evaluate_at_quadrature_points(points,
                                                     values_soa.data(),
                                                     grads_soa.data(),
                                                     hess_soa.data());
                g_sink_d += values_soa[0] + grads_soa[0] + hess_soa[0];
            }
        });

        const double total_qpts = double(kIterations) * double(n_qpts);
        csv << ec.name << ",per_qp_loop," << n_qpts << "," << kIterations << ","
            << sec_per_qp << "," << (sec_per_qp * 1e9 / total_qpts) << "\n";
        csv << ec.name << ",multi_qp_entry," << n_qpts << "," << kIterations << ","
            << sec_multi << "," << (sec_multi * 1e9 / total_qpts) << "\n";

        std::cout << "[Pzeta2 " << ec.name << "] per_qp=" << (sec_per_qp * 1e9 / total_qpts)
                  << " ns/qp, multi_qp=" << (sec_multi * 1e9 / total_qpts) << " ns/qp\n";
    }
}
