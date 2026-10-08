/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsDirectSolver.h"

#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsShared.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#if defined(FE_HAS_EIGEN)
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include <mpi.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

static_assert(std::is_same_v<Real, double>, "FsilsGatheredDirectSolver exchanges values as MPI_DOUBLE");

namespace {

constexpr int kRoot = 0;
constexpr int kMaxRefinementSteps = 3;
constexpr std::uint64_t kSummaryInterval = 200;
/// Diagonal pivot threshold of the threshold partial pivoting: the diagonal
/// is kept as pivot while |a_jj| >= kPivotThreshold * max_i |a_ij|, which
/// preserves the fill-reducing symmetric ordering (MUMPS uses 0.01 as well).
constexpr double kPivotThreshold = 0.01;
/// The factored matrix is symmetrically permuted, so SparseLU runs in its
/// symmetric mode (no column postorder, which would move pivots off the
/// diagonal).
constexpr bool kSymmetricMode = true;

[[nodiscard]] double wallSeconds() noexcept
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[nodiscard]] double peakRssMegabytes() noexcept
{
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // ru_maxrss is in kB on Linux
}

/// Exact power-of-two approximation of 1 / value (value > 0).
[[nodiscard]] double powerOfTwoReciprocal(double value) noexcept
{
    if (!(value > 0.0) || !std::isfinite(value)) {
        return 1.0;
    }
    int exponent = 0;
    (void)std::frexp(value, &exponent);  // value = m * 2^exponent, m in [0.5, 1)
    return std::ldexp(1.0, 1 - exponent);
}

[[nodiscard]] int checkedInt(std::size_t value, const char* what)
{
    FE_THROW_IF(value > static_cast<std::size_t>(std::numeric_limits<int>::max()), InvalidArgumentException,
                std::string("FsilsGatheredDirectSolver: ") + what + " exceeds the MPI count range");
    return static_cast<int>(value);
}

/// Report fields decided on the root and broadcast to every rank.
struct RootDecision {
    double initial_residual_norm{0.0};
    double final_residual_norm{0.0};
    double relative_residual{0.0};
    int status{0};       // 0 solved, 1 factorization failed, 2 non-finite result, 3 root error
    int converged{0};
    int iterations{0};
    int analyzed{0};
    int ordered{0};
    int refinement_steps{0};
};

/// Rows gathered on the root, in arrival order (rank by rank, owned rows in
/// each rank's internal order); columns sorted by global node within a row.
struct GatheredRows {
    std::vector<int> rows;
    std::vector<int> len;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> rhs;
    std::vector<std::int64_t> dirichlet;
};

} // namespace

struct FsilsGatheredDirectSolver::Impl {
#if defined(FE_HAS_EIGEN)
    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using LU = Eigen::SparseLU<SpMat, Eigen::NaturalOrdering<int>>;
#endif

    FsilsDirectSolveStats stats{};
    int log_rank{-1};

    // ---- root-side state --------------------------------------------------
    int dof{0};
    int gnNo{0};
    // Global node pattern (rows by global node, columns sorted).
    std::vector<std::int64_t> g_row_ptr{};
    std::vector<int> g_cols{};
    bool pattern_valid{false};
    // Fill-reducing symmetric ordering of the stored structure (scalar old -> new).
    std::vector<int> inv{};
    // Dirichlet scalar DOFs (backend numbering), sorted unique, and flags.
    std::vector<std::int64_t> dirichlet{};
    std::vector<char> is_dirichlet{};
    // Structural entries of the factored matrix, per global block entry and
    // component pair: diagonals, plus every entry outside Dirichlet rows and
    // columns that has been nonzero since the pattern or the Dirichlet set
    // last changed.
    std::vector<char> keep{};
    std::vector<char> keep_scratch{};
    // Factored structure (permuted numbering).
    bool structure_valid{false};
    std::vector<std::int32_t> slot{};  // per global block entry and component pair, -1 if not stored
    std::vector<int> unit_diagonal{};  // CSC positions of Dirichlet unit diagonals
    std::vector<double> row_scale{};   // power-of-two equilibration (permuted numbering)
    std::vector<double> col_scale{};
#if defined(FE_HAS_EIGEN)
    SpMat csc{};
    std::unique_ptr<LU> lu{};
    Eigen::VectorXd bp{};
    Eigen::VectorXd y{};
    Eigen::VectorXd rp{};
#endif

    void log(const std::string& msg) const
    {
        if (log_rank == kRoot) {
            FE_LOG_INFO(msg);
        }
    }

#if defined(FE_HAS_EIGEN)
    /// Approximate minimum degree on the stored structure (A + A^T, scalar
    /// unknowns).  Recomputed with every new stored structure.
    void computeOrdering()
    {
        const int d = dof;
        const std::size_t d2 = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
        const int n = gnNo * d;
        std::vector<int> col_count(static_cast<std::size_t>(n) + 1u, 0);
        for (int i = 0; i < gnNo; ++i) {
            for (auto p = g_row_ptr[static_cast<std::size_t>(i)]; p < g_row_ptr[static_cast<std::size_t>(i) + 1u];
                 ++p) {
                const int j = g_cols[static_cast<std::size_t>(p)];
                const char* kp = keep.data() + static_cast<std::size_t>(p) * d2;
                for (int r = 0; r < d; ++r) {
                    for (int c = 0; c < d; ++c) {
                        if (kp[r * d + c]) {
                            ++col_count[static_cast<std::size_t>(j * d + c) + 1u];
                        }
                    }
                }
            }
        }
        for (int k = 0; k < n; ++k) {
            col_count[static_cast<std::size_t>(k) + 1u] += col_count[static_cast<std::size_t>(k)];
        }
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> pattern(n, n);
        pattern.resizeNonZeros(static_cast<Eigen::Index>(col_count[static_cast<std::size_t>(n)]));
        std::vector<int> next(col_count.begin(), col_count.end() - 1);
        int* outer = pattern.outerIndexPtr();
        int* inner = pattern.innerIndexPtr();
        double* values = pattern.valuePtr();
        for (int k = 0; k <= n; ++k) {
            outer[k] = col_count[static_cast<std::size_t>(k)];
        }
        for (int i = 0; i < gnNo; ++i) {
            for (auto p = g_row_ptr[static_cast<std::size_t>(i)]; p < g_row_ptr[static_cast<std::size_t>(i) + 1u];
                 ++p) {
                const int j = g_cols[static_cast<std::size_t>(p)];
                const char* kp = keep.data() + static_cast<std::size_t>(p) * d2;
                for (int r = 0; r < d; ++r) {
                    for (int c = 0; c < d; ++c) {
                        if (kp[r * d + c]) {
                            const int pos = next[static_cast<std::size_t>(j * d + c)]++;
                            inner[pos] = i * d + r;  // rows ascend within a column (row nodes ascend)
                            values[pos] = 1.0;
                        }
                    }
                }
            }
        }
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> order;
        Eigen::AMDOrdering<int> amd;
        amd(pattern, order);
        // Eigen's AMDOrdering returns the elimination order: indices()(new) = old.
        FE_THROW_IF(order.size() != n, FEException, "FsilsGatheredDirectSolver: AMD ordering size mismatch");
        inv.assign(static_cast<std::size_t>(n), -1);
        for (int new_index = 0; new_index < n; ++new_index) {
            const int old_index = order.indices()(new_index);
            FE_THROW_IF(old_index < 0 || old_index >= n || inv[static_cast<std::size_t>(old_index)] >= 0,
                        FEException, "FsilsGatheredDirectSolver: invalid AMD ordering");
            inv[static_cast<std::size_t>(old_index)] = new_index;
        }
    }

    /// Build the CSC structure (permuted numbering) and the slot map from the
    /// node pattern, the Dirichlet flags and the structural entries.
    void buildStructure()
    {
        const int d = dof;
        const std::size_t d2 = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
        const int n = gnNo * d;
        const std::size_t n_entries = g_cols.size() * d2;
        FE_THROW_IF(keep.size() != n_entries, FEException, "FsilsGatheredDirectSolver: structure size mismatch");

        std::vector<int> col_count(static_cast<std::size_t>(n) + 1u, 0);
        std::vector<char> has_diagonal(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < gnNo; ++i) {
            for (auto p = g_row_ptr[static_cast<std::size_t>(i)]; p < g_row_ptr[static_cast<std::size_t>(i) + 1u];
                 ++p) {
                const int j = g_cols[static_cast<std::size_t>(p)];
                for (int c = 0; c < d; ++c) {
                    const int col = j * d + c;
                    for (int r = 0; r < d; ++r) {
                        const std::size_t id = static_cast<std::size_t>(p) * d2 + static_cast<std::size_t>(r * d + c);
                        if (i * d + r == col) {
                            has_diagonal[static_cast<std::size_t>(col)] = 1;
                        }
                        if (keep[id]) {
                            ++col_count[static_cast<std::size_t>(inv[static_cast<std::size_t>(col)]) + 1u];
                        }
                    }
                }
            }
        }
        for (int row = 0; row < n; ++row) {
            FE_THROW_IF(has_diagonal[static_cast<std::size_t>(row)] == 0, FEException,
                        "FsilsGatheredDirectSolver: global pattern lacks a diagonal block");
        }
        for (int k = 0; k < n; ++k) {
            col_count[static_cast<std::size_t>(k) + 1u] += col_count[static_cast<std::size_t>(k)];
        }
        const std::size_t nnz = static_cast<std::size_t>(col_count[static_cast<std::size_t>(n)]);
        FE_THROW_IF(nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()), InvalidArgumentException,
                    "FsilsGatheredDirectSolver: factored matrix exceeds the 32-bit index range");

        // Place (new row, entry id) column by column, then sort each column.
        std::vector<std::pair<int, std::int64_t>> placed(nnz);
        std::vector<int> next(col_count.begin(), col_count.end() - 1);
        for (int i = 0; i < gnNo; ++i) {
            for (auto p = g_row_ptr[static_cast<std::size_t>(i)]; p < g_row_ptr[static_cast<std::size_t>(i) + 1u];
                 ++p) {
                const int j = g_cols[static_cast<std::size_t>(p)];
                for (int r = 0; r < d; ++r) {
                    const int new_row = inv[static_cast<std::size_t>(i * d + r)];
                    for (int c = 0; c < d; ++c) {
                        const std::size_t id = static_cast<std::size_t>(p) * d2 + static_cast<std::size_t>(r * d + c);
                        if (!keep[id]) {
                            continue;
                        }
                        const int new_col = inv[static_cast<std::size_t>(j * d + c)];
                        placed[static_cast<std::size_t>(next[static_cast<std::size_t>(new_col)]++)] = {
                            new_row, static_cast<std::int64_t>(id)};
                    }
                }
            }
        }

        csc.resize(n, n);
        csc.resizeNonZeros(static_cast<Eigen::Index>(nnz));
        int* outer = csc.outerIndexPtr();
        int* inner = csc.innerIndexPtr();
        double* values = csc.valuePtr();
        for (int k = 0; k <= n; ++k) {
            outer[k] = col_count[static_cast<std::size_t>(k)];
        }
        for (int k = 0; k < n; ++k) {
            std::sort(placed.begin() + col_count[static_cast<std::size_t>(k)],
                      placed.begin() + col_count[static_cast<std::size_t>(k) + 1u],
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        }
        slot.assign(n_entries, -1);
        unit_diagonal.clear();
        for (std::size_t pos = 0; pos < nnz; ++pos) {
            inner[pos] = placed[pos].first;
            values[pos] = 0.0;
            slot[static_cast<std::size_t>(placed[pos].second)] = static_cast<std::int32_t>(pos);
        }
        for (const auto dd : dirichlet) {
            // The diagonal entry of a Dirichlet row in permuted numbering.
            const int nd = inv[static_cast<std::size_t>(dd)];
            const int col_begin = outer[nd];
            const int col_end = outer[nd + 1];
            const auto it = std::lower_bound(inner + col_begin, inner + col_end, nd);
            FE_THROW_IF(it == inner + col_end || *it != nd, FEException,
                        "FsilsGatheredDirectSolver: missing Dirichlet diagonal");
            unit_diagonal.push_back(static_cast<int>(it - inner));
        }
        csc.makeCompressed();
        structure_valid = true;
        stats.n = n;
        stats.nnz_matrix = static_cast<std::int64_t>(nnz);
    }

    /// Assemble, factor and solve the gathered system on the root.
    RootDecision rootSolve(GatheredRows& in, int gn, int d, const SolverOptions& options,
                           std::vector<double>& x_global, std::string& error, double timing[4])
    {
        RootDecision decision{};
        const std::size_t d2 = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
        const int n = gn * d;
        const double ta0 = wallSeconds();

        // Global node pattern from the gathered rows.
        std::vector<std::int64_t> row_ptr(static_cast<std::size_t>(gn) + 1u, 0);
        std::vector<std::int64_t> row_src(static_cast<std::size_t>(gn), -1);
        {
            std::int64_t offset = 0;
            for (std::size_t k = 0; k < in.rows.size(); ++k) {
                const int g = in.rows[k];
                FE_THROW_IF(g < 0 || g >= gn, FEException, "FsilsGatheredDirectSolver: gathered row out of range");
                FE_THROW_IF(row_src[static_cast<std::size_t>(g)] >= 0, FEException,
                            "FsilsGatheredDirectSolver: global row owned by two ranks");
                row_src[static_cast<std::size_t>(g)] = offset;
                row_ptr[static_cast<std::size_t>(g) + 1u] = in.len[k];
                offset += in.len[k];
            }
            FE_THROW_IF(in.rows.size() != static_cast<std::size_t>(gn), FEException,
                        "FsilsGatheredDirectSolver: gathered rows do not cover the global operator");
            for (int g = 0; g < gn; ++g) {
                row_ptr[static_cast<std::size_t>(g) + 1u] += row_ptr[static_cast<std::size_t>(g)];
            }
        }
        std::vector<int> cols(in.cols.size());
        std::vector<std::int64_t> src_of_entry(in.cols.size());  // global block entry -> gathered entry
        for (int g = 0; g < gn; ++g) {
            const auto first = row_ptr[static_cast<std::size_t>(g)];
            const auto len = row_ptr[static_cast<std::size_t>(g) + 1u] - first;
            const auto src = row_src[static_cast<std::size_t>(g)];
            for (std::int64_t e = 0; e < len; ++e) {
                cols[static_cast<std::size_t>(first + e)] = in.cols[static_cast<std::size_t>(src + e)];
                src_of_entry[static_cast<std::size_t>(first + e)] = src + e;
            }
        }
        std::sort(in.dirichlet.begin(), in.dirichlet.end());
        in.dirichlet.erase(std::unique(in.dirichlet.begin(), in.dirichlet.end()), in.dirichlet.end());

        std::string reason;
        auto addReason = [&](const char* r) { reason += reason.empty() ? r : (std::string("+") + r); };
        if (!pattern_valid || dof != d || gnNo != gn || g_row_ptr != row_ptr || g_cols != cols) {
            addReason(pattern_valid ? "pattern" : "first");
            dof = d;
            gnNo = gn;
            g_row_ptr = std::move(row_ptr);
            g_cols = std::move(cols);
            pattern_valid = true;
            structure_valid = false;
            keep.clear();
        }
        if (is_dirichlet.size() != static_cast<std::size_t>(n) || dirichlet != in.dirichlet) {
            if (structure_valid) {
                addReason("dirichlet");
            }
            dirichlet = in.dirichlet;
            is_dirichlet.assign(static_cast<std::size_t>(n), 0);
            for (const auto dd : dirichlet) {
                is_dirichlet[static_cast<std::size_t>(dd)] = 1;
            }
            structure_valid = false;
            keep.clear();
        }

        // Structural entries of this matrix; grow the stored structure when a
        // new nonzero appears (zeros in stored entries are kept explicitly).
        const std::size_t n_blocks = g_cols.size();
        keep_scratch.assign(n_blocks * d2, 0);
        for (int i = 0; i < gn; ++i) {
            for (auto p = g_row_ptr[static_cast<std::size_t>(i)]; p < g_row_ptr[static_cast<std::size_t>(i) + 1u];
                 ++p) {
                const int j = g_cols[static_cast<std::size_t>(p)];
                const double* blk = in.vals.data() + static_cast<std::size_t>(src_of_entry[static_cast<std::size_t>(p)]) * d2;
                char* kp = keep_scratch.data() + static_cast<std::size_t>(p) * d2;
                for (int r = 0; r < d; ++r) {
                    const int row = i * d + r;
                    const bool dir_row = is_dirichlet[static_cast<std::size_t>(row)] != 0;
                    for (int c = 0; c < d; ++c) {
                        const int col = j * d + c;
                        const std::size_t k = static_cast<std::size_t>(r * d + c);
                        if (row == col) {
                            kp[k] = 1;
                        } else if (!dir_row && !is_dirichlet[static_cast<std::size_t>(col)] && blk[k] != 0.0) {
                            kp[k] = 1;
                        }
                    }
                }
            }
        }
        if (keep.empty()) {
            keep.swap(keep_scratch);
            structure_valid = false;
        } else {
            bool grew = false;
            for (std::size_t k = 0; k < keep.size(); ++k) {
                if (keep_scratch[k] && !keep[k]) {
                    keep[k] = 1;
                    grew = true;
                }
            }
            if (grew) {
                addReason("nonzeros");
                structure_valid = false;
            }
        }

        if (!structure_valid) {
            computeOrdering();
            decision.ordered = 1;
            buildStructure();
            lu.reset();
        }

        // Values, Dirichlet unit diagonals and power-of-two equilibration.
        {
            double* out = csc.valuePtr();
            for (std::size_t p = 0; p < n_blocks; ++p) {
                const double* blk = in.vals.data() + static_cast<std::size_t>(src_of_entry[p]) * d2;
                const std::int32_t* sl = slot.data() + p * d2;
                for (std::size_t k = 0; k < d2; ++k) {
                    if (sl[k] >= 0) {
                        out[sl[k]] = blk[k];
                    }
                }
            }
            for (const int pos : unit_diagonal) {
                out[pos] = 1.0;
            }
            const int* outer = csc.outerIndexPtr();
            const int* inner = csc.innerIndexPtr();
            row_scale.assign(static_cast<std::size_t>(n), 0.0);
            for (int col = 0; col < n; ++col) {
                for (int k = outer[col]; k < outer[col + 1]; ++k) {
                    auto& m = row_scale[static_cast<std::size_t>(inner[k])];
                    m = std::max(m, std::abs(out[k]));
                }
            }
            for (auto& v : row_scale) {
                v = powerOfTwoReciprocal(v);
            }
            col_scale.assign(static_cast<std::size_t>(n), 1.0);
            for (int col = 0; col < n; ++col) {
                double m = 0.0;
                for (int k = outer[col]; k < outer[col + 1]; ++k) {
                    m = std::max(m, std::abs(out[k]) * row_scale[static_cast<std::size_t>(inner[k])]);
                }
                const double cs = powerOfTwoReciprocal(m);
                col_scale[static_cast<std::size_t>(col)] = cs;
                for (int k = outer[col]; k < outer[col + 1]; ++k) {
                    out[k] *= row_scale[static_cast<std::size_t>(inner[k])] * cs;
                }
            }
        }
        timing[0] = wallSeconds() - ta0;

        if (!lu && reason.empty()) {
            addReason("retry");
        }
        if (!lu) {
            const double tz0 = wallSeconds();
            lu = std::make_unique<LU>();
            lu->setPivotThreshold(kPivotThreshold);
            lu->isSymmetric(kSymmetricMode);
            lu->analyzePattern(csc);
            timing[1] = wallSeconds() - tz0;
            decision.analyzed = 1;
        }
        const double tf0 = wallSeconds();
        lu->factorize(csc);
        timing[2] = wallSeconds() - tf0;

        const double ts0 = wallSeconds();
        std::vector<double> b_old(static_cast<std::size_t>(n), 0.0);
        for (std::size_t k = 0; k < in.rows.size(); ++k) {
            const int g = in.rows[k];
            for (int c = 0; c < d; ++c) {
                b_old[static_cast<std::size_t>(g) * d + c] = in.rhs[k * static_cast<std::size_t>(d) + c];
            }
        }
        for (const auto dd : dirichlet) {
            b_old[static_cast<std::size_t>(dd)] = 0.0;
        }
        double b_norm_sq = 0.0;
        for (const double v : b_old) {
            b_norm_sq += v * v;
        }
        const double b_norm = std::sqrt(b_norm_sq);
        decision.initial_residual_norm = b_norm;

        if (lu->info() != Eigen::Success) {
            decision.status = 1;
            error = lu->lastErrorMessage();
            lu.reset();  // analyze again next time
        } else {
            bp.resize(n);
            for (int i = 0; i < n; ++i) {
                const int ni = inv[static_cast<std::size_t>(i)];
                bp(ni) = b_old[static_cast<std::size_t>(i)] * row_scale[static_cast<std::size_t>(ni)];
            }
            y = lu->solve(bp);
            // Residual of the unscaled system: r = R^-1 (b~ - A~ y).
            auto residualNorm = [&]() -> double {
                rp = bp - csc * y;
                double sq = 0.0;
                for (int ni = 0; ni < n; ++ni) {
                    const double r = rp(ni) / row_scale[static_cast<std::size_t>(ni)];
                    sq += r * r;
                }
                return std::sqrt(sq);
            };
            double r_norm = residualNorm();
            const double target = std::max<double>(options.abs_tol, options.rel_tol * b_norm);
            int steps = 0;
            while (std::isfinite(r_norm) && r_norm > target && steps < kMaxRefinementSteps) {
                const Eigen::VectorXd dy = lu->solve(rp);
                y += dy;
                const double next_norm = residualNorm();
                ++steps;
                if (!(next_norm < r_norm)) {
                    y -= dy;
                    rp = bp - csc * y;
                    break;
                }
                r_norm = next_norm;
            }
            decision.refinement_steps = steps;
            x_global.assign(static_cast<std::size_t>(n), 0.0);
            bool finite = std::isfinite(r_norm);
            for (int i = 0; i < n; ++i) {
                const int ni = inv[static_cast<std::size_t>(i)];
                const double v = y(ni) * col_scale[static_cast<std::size_t>(ni)];
                finite = finite && std::isfinite(v);
                x_global[static_cast<std::size_t>(i)] = v;
            }
            for (const auto dd : dirichlet) {
                x_global[static_cast<std::size_t>(dd)] = 0.0;
            }
            decision.final_residual_norm = r_norm;
            decision.relative_residual = b_norm > 0.0 ? r_norm / b_norm : 0.0;
            decision.iterations = 1 + steps;
            decision.converged = (finite && r_norm <= target) ? 1 : 0;
            decision.status = finite ? 0 : 2;
            stats.nnz_factors = static_cast<std::int64_t>(lu->nnzL() + lu->nnzU());
            // L and U values (8 bytes) with their row indices (4 bytes).
            stats.factor_megabytes = static_cast<double>(stats.nnz_factors) * 12.0 / 1048576.0;
        }
        timing[3] = wallSeconds() - ts0;

        if (decision.analyzed) {
            std::ostringstream oss;
            oss << "FsilsDirectSolve: analysis diagnostic=fsils_direct_analysis reason=" << reason
                << " ordering=" << (decision.ordered ? "new" : "kept") << " n=" << n << " nnz=" << csc.nonZeros()
                << " dirichlet=" << dirichlet.size() << " nnz_factors=" << stats.nnz_factors
                << " factor_MB=" << stats.factor_megabytes << " assemble_s=" << timing[0]
                << " analyze_s=" << timing[1] << " factor_s=" << timing[2] << " solve_s=" << timing[3]
                << " peak_rss_MB=" << peakRssMegabytes();
            if (decision.status != 0) {
                oss << " status=" << decision.status << " error='" << error << "'";
            }
            log(oss.str());
        }
        return decision;
    }
#endif
};

FsilsGatheredDirectSolver::FsilsGatheredDirectSolver()
    : impl_(std::make_unique<Impl>())
{
}

FsilsGatheredDirectSolver::~FsilsGatheredDirectSolver()
{
    if (impl_ && impl_->stats.solves > 0) {
        impl_->log("FsilsDirectSolve: final " + summary());
    }
}

const FsilsDirectSolveStats& FsilsGatheredDirectSolver::stats() const noexcept
{
    return impl_->stats;
}

std::string FsilsGatheredDirectSolver::summary() const
{
    const auto& s = impl_->stats;
    std::ostringstream oss;
    oss << "summary diagnostic=fsils_direct_summary solves=" << s.solves << " orderings=" << s.orderings
        << " analyses=" << s.analyses
        << " factorizations=" << s.factorizations << " refinement_steps=" << s.refinement_steps
        << " failures=" << s.failures << " n=" << s.n << " nnz=" << s.nnz_matrix << " nnz_factors=" << s.nnz_factors
        << " factor_MB=" << s.factor_megabytes << " gather_s=" << s.gather_seconds
        << " assemble_s=" << s.assemble_seconds << " analyze_s=" << s.analyze_seconds
        << " factor_s=" << s.factor_seconds << " solve_s=" << s.solve_seconds
        << " peak_rss_MB=" << s.peak_rss_megabytes;
    return oss.str();
}

SolverReport FsilsGatheredDirectSolver::solve(const FsilsMatrix& A,
                                              FsilsVector& x,
                                              const FsilsVector& b,
                                              std::span<const GlobalIndex> dirichlet_fe_dofs,
                                              const SolverOptions& options)
{
    auto& st = *impl_;
    const auto shared = A.operatorShared();
    FE_CHECK_NOT_NULL(shared.get(), "FsilsGatheredDirectSolver: shared layout");
    const auto& lhs = shared->lhs;
    const MPI_Comm comm = lhs.commu.comm;
    const int n_ranks = std::max(1, lhs.commu.nTasks);
    int rank = 0;
    if (n_ranks > 1) {
        MPI_Comm_rank(comm, &rank);
    }
    st.log_rank = rank;
    const bool root = (rank == kRoot);

    const int d = A.fsilsDof();
    const std::size_t d2 = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    const int nNo = lhs.nNo;
    const int owned_rows = (n_ranks > 1) ? lhs.mynNo : nNo;
    FE_THROW_IF(d <= 0 || nNo <= 0, FEException, "FsilsGatheredDirectSolver: empty FSILS operator");
    FE_THROW_IF(n_ranks > 1 && !lhs.owned_row_operator, NotImplementedException,
                "FsilsGatheredDirectSolver: distributed solves need the owned-row FSILS operator");
    FE_THROW_IF(owned_rows != shared->owned_node_count, FEException,
                "FsilsGatheredDirectSolver: owned FSILS rows do not match the owned node count");
    FE_THROW_IF(x.data().size() != static_cast<std::size_t>(nNo) * static_cast<std::size_t>(d) ||
                    b.data().size() != x.data().size(),
                InvalidArgumentException, "FsilsGatheredDirectSolver: vector size mismatch");

    const double t_start = wallSeconds();

    // ---- local contribution: owned rows in global node numbering ------------
    std::vector<int> old_of_internal(static_cast<std::size_t>(nNo), -1);
    if (static_cast<int>(shared->old_of_internal.size()) == nNo) {
        old_of_internal = shared->old_of_internal;
    } else {
        for (int old = 0; old < nNo; ++old) {
            const int internal = lhs.map(old);
            FE_THROW_IF(internal < 0 || internal >= nNo, FEException, "FsilsGatheredDirectSolver: invalid FSILS map");
            old_of_internal[static_cast<std::size_t>(internal)] = old;
        }
    }
    std::vector<int> global_of_internal(static_cast<std::size_t>(nNo), -1);
    for (int internal = 0; internal < nNo; ++internal) {
        const int g = shared->oldToGlobalNode(old_of_internal[static_cast<std::size_t>(internal)]);
        FE_THROW_IF(g < 0 || g >= shared->gnNo, FEException, "FsilsGatheredDirectSolver: invalid global node");
        global_of_internal[static_cast<std::size_t>(internal)] = g;
    }

    const Real* values = A.fsilsValuesPtr();
    std::vector<int> send_rows(static_cast<std::size_t>(owned_rows));
    std::vector<int> send_len(static_cast<std::size_t>(owned_rows));
    std::vector<int> send_cols;
    std::vector<double> send_vals;
    std::vector<double> send_rhs(static_cast<std::size_t>(owned_rows) * static_cast<std::size_t>(d));
    {
        std::size_t local_entries = 0;
        for (int r = 0; r < owned_rows; ++r) {
            local_entries += static_cast<std::size_t>(lhs.rowPtr(1, r) - lhs.rowPtr(0, r) + 1);
        }
        send_cols.reserve(local_entries);
        send_vals.reserve(local_entries * d2);
        std::vector<std::pair<int, int>> row_entries;
        const auto& b_data = b.data();
        for (int r = 0; r < owned_rows; ++r) {
            send_rows[static_cast<std::size_t>(r)] = global_of_internal[static_cast<std::size_t>(r)];
            row_entries.clear();
            for (auto nz = lhs.rowPtr(0, r); nz <= lhs.rowPtr(1, r); ++nz) {
                const int c = lhs.colPtr(nz);
                FE_THROW_IF(c < 0 || c >= nNo, FEException, "FsilsGatheredDirectSolver: invalid FSILS column");
                row_entries.emplace_back(global_of_internal[static_cast<std::size_t>(c)], static_cast<int>(nz));
            }
            std::sort(row_entries.begin(), row_entries.end());
            send_len[static_cast<std::size_t>(r)] = static_cast<int>(row_entries.size());
            for (const auto& [gc, nz] : row_entries) {
                send_cols.push_back(gc);
                const Real* src = values + static_cast<std::size_t>(nz) * d2;
                send_vals.insert(send_vals.end(), src, src + d2);
            }
            const int old = old_of_internal[static_cast<std::size_t>(r)];
            for (int c = 0; c < d; ++c) {
                send_rhs[static_cast<std::size_t>(r) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)] =
                    b_data[static_cast<std::size_t>(old) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)];
            }
        }
    }
    std::vector<std::int64_t> send_dir;
    send_dir.reserve(dirichlet_fe_dofs.size());
    const std::int64_t n_scalar = static_cast<std::int64_t>(shared->gnNo) * d;
    for (const auto fe_dof : dirichlet_fe_dofs) {
        GlobalIndex backend = fe_dof;
        if (shared->dof_permutation && !shared->dof_permutation->forward.empty()) {
            const auto idx = static_cast<std::size_t>(fe_dof);
            backend = (fe_dof >= 0 && idx < shared->dof_permutation->forward.size())
                          ? shared->dof_permutation->forward[idx]
                          : INVALID_GLOBAL_INDEX;
        }
        if (backend >= 0 && backend < n_scalar) {
            send_dir.push_back(static_cast<std::int64_t>(backend));
        }
    }

    // ---- gather to the root ----------------------------------------------------
    const int local_counts[3] = {owned_rows, checkedInt(send_cols.size(), "local block count"),
                                 checkedInt(send_dir.size(), "local Dirichlet count")};
    std::vector<int> all_counts(root ? static_cast<std::size_t>(3 * n_ranks) : 0u);
    std::vector<int> recv_rows, recv_len, recv_cols;
    std::vector<double> recv_vals, recv_rhs;
    std::vector<std::int64_t> recv_dir;
    std::vector<int> cnt_rows, dsp_rows, cnt_cols, dsp_cols, cnt_vals, dsp_vals, cnt_rhs, dsp_rhs, cnt_dir, dsp_dir;
    if (n_ranks > 1) {
        MPI_Gather(local_counts, 3, MPI_INT, all_counts.data(), 3, MPI_INT, kRoot, comm);
    } else {
        all_counts.assign(local_counts, local_counts + 3);
    }
    if (root) {
        auto layout = [&](int field, std::size_t scale, std::vector<int>& cnt, std::vector<int>& dsp,
                          const char* what) -> std::size_t {
            cnt.resize(static_cast<std::size_t>(n_ranks));
            dsp.resize(static_cast<std::size_t>(n_ranks));
            std::size_t total = 0;
            for (int q = 0; q < n_ranks; ++q) {
                const std::size_t c = static_cast<std::size_t>(all_counts[static_cast<std::size_t>(3 * q + field)]) * scale;
                cnt[static_cast<std::size_t>(q)] = checkedInt(c, what);
                dsp[static_cast<std::size_t>(q)] = checkedInt(total, what);
                total += c;
            }
            return total;
        };
        recv_rows.resize(layout(0, 1u, cnt_rows, dsp_rows, "gathered rows"));
        recv_len.resize(recv_rows.size());
        recv_cols.resize(layout(1, 1u, cnt_cols, dsp_cols, "gathered blocks"));
        recv_vals.resize(layout(1, d2, cnt_vals, dsp_vals, "gathered values"));
        recv_rhs.resize(layout(0, static_cast<std::size_t>(d), cnt_rhs, dsp_rhs, "gathered right-hand side"));
        recv_dir.resize(layout(2, 1u, cnt_dir, dsp_dir, "gathered Dirichlet DOFs"));
    }
    if (n_ranks > 1) {
        MPI_Gatherv(send_rows.data(), owned_rows, MPI_INT, recv_rows.data(), cnt_rows.data(), dsp_rows.data(), MPI_INT,
                    kRoot, comm);
        MPI_Gatherv(send_len.data(), owned_rows, MPI_INT, recv_len.data(), cnt_rows.data(), dsp_rows.data(), MPI_INT,
                    kRoot, comm);
        MPI_Gatherv(send_cols.data(), local_counts[1], MPI_INT, recv_cols.data(), cnt_cols.data(), dsp_cols.data(),
                    MPI_INT, kRoot, comm);
        MPI_Gatherv(send_vals.data(), checkedInt(send_vals.size(), "local values"), MPI_DOUBLE, recv_vals.data(),
                    cnt_vals.data(), dsp_vals.data(), MPI_DOUBLE, kRoot, comm);
        MPI_Gatherv(send_rhs.data(), checkedInt(send_rhs.size(), "local right-hand side"), MPI_DOUBLE,
                    recv_rhs.data(), cnt_rhs.data(), dsp_rhs.data(), MPI_DOUBLE, kRoot, comm);
        MPI_Gatherv(send_dir.data(), local_counts[2], MPI_INT64_T, recv_dir.data(), cnt_dir.data(), dsp_dir.data(),
                    MPI_INT64_T, kRoot, comm);
    } else {
        recv_rows = std::move(send_rows);
        recv_len = std::move(send_len);
        recv_cols = std::move(send_cols);
        recv_vals = std::move(send_vals);
        recv_rhs = std::move(send_rhs);
        recv_dir = std::move(send_dir);
    }
    const double t_gathered = wallSeconds();

    // ---- root: assemble, factor, solve ----------------------------------------
    RootDecision decision{};
    std::vector<double> x_global;
    std::string root_error;
    double timing[4] = {0.0, 0.0, 0.0, 0.0};  // assemble, analyze, factor, solve
    if (root) {
#if defined(FE_HAS_EIGEN)
        try {
            GatheredRows gathered;
            gathered.rows = std::move(recv_rows);
            gathered.len = std::move(recv_len);
            gathered.cols = std::move(recv_cols);
            gathered.vals = std::move(recv_vals);
            gathered.rhs = std::move(recv_rhs);
            gathered.dirichlet = std::move(recv_dir);
            decision = st.rootSolve(gathered, shared->gnNo, d, options, x_global, root_error, timing);
            recv_rows = std::move(gathered.rows);
        } catch (const std::exception& e) {
            decision = RootDecision{};
            decision.status = 3;
            root_error = e.what();
            st.pattern_valid = false;
            st.structure_valid = false;
            st.keep.clear();
            st.lu.reset();
        }
#else
        decision.status = 3;
        root_error = "the FE Eigen backend is required (FE_ENABLE_EIGEN=ON)";
#endif
    }
    const double t_assemble = timing[0], t_analyze = timing[1], t_factor = timing[2], t_solve = timing[3];

    // ---- broadcast the decision and scatter the owned solution ------------------
    if (n_ranks > 1) {
        MPI_Bcast(&decision, static_cast<int>(sizeof(RootDecision)), MPI_BYTE, kRoot, comm);
    }
    if (decision.status == 3) {
        if (root) {
            FE_LOG_ERROR("FsilsDirectSolve: root error: " + root_error);
        }
        FE_THROW(FEException, "FsilsGatheredDirectSolver: direct solve failed on the root rank" +
                                  (root ? (": " + root_error) : std::string(" (see rank 0 log)")));
    }

    auto& x_data = x.data();
    std::fill(x_data.begin(), x_data.end(), Real(0.0));
    if (decision.status == 0) {
        std::vector<double> recv_x(static_cast<std::size_t>(owned_rows) * static_cast<std::size_t>(d));
        std::vector<double> send_x;
        if (root) {
            send_x.resize(recv_rows.size() * static_cast<std::size_t>(d));
            for (std::size_t k = 0; k < recv_rows.size(); ++k) {
                const int g = recv_rows[k];
                for (int c = 0; c < d; ++c) {
                    send_x[k * static_cast<std::size_t>(d) + c] = x_global[static_cast<std::size_t>(g) * d + c];
                }
            }
        }
        if (n_ranks > 1) {
            MPI_Scatterv(send_x.data(), cnt_rhs.data(), dsp_rhs.data(), MPI_DOUBLE, recv_x.data(),
                         checkedInt(recv_x.size(), "local solution"), MPI_DOUBLE, kRoot, comm);
        } else {
            recv_x = std::move(send_x);
        }
        for (int r = 0; r < owned_rows; ++r) {
            const int old = old_of_internal[static_cast<std::size_t>(r)];
            for (int c = 0; c < d; ++c) {
                x_data[static_cast<std::size_t>(old) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)] =
                    recv_x[static_cast<std::size_t>(r) * static_cast<std::size_t>(d) + static_cast<std::size_t>(c)];
            }
        }
    }
    x.updateGhosts();
    const double t_end = wallSeconds();

    // ---- report and counters ---------------------------------------------------
    auto& s = st.stats;
    ++s.solves;
    s.analyses += static_cast<std::uint64_t>(decision.analyzed);
    s.orderings += static_cast<std::uint64_t>(decision.ordered);
    s.refinement_steps += static_cast<std::uint64_t>(decision.refinement_steps);
    if (decision.status == 0) {
        ++s.factorizations;
    } else {
        ++s.failures;
    }
    if (root) {
        s.gather_seconds += (t_gathered - t_start) + (t_end - t_gathered - t_assemble - t_analyze - t_factor - t_solve);
        s.assemble_seconds += t_assemble;
        s.analyze_seconds += t_analyze;
        s.factor_seconds += t_factor;
        s.solve_seconds += t_solve;
        s.peak_rss_megabytes = peakRssMegabytes();
        if (s.solves % kSummaryInterval == 0) {
            st.log("FsilsDirectSolve: " + summary());
        }
    }

    SolverReport report;
    report.iterations = decision.iterations;
    report.initial_residual_norm = static_cast<Real>(decision.initial_residual_norm);
    if (decision.status == 0) {
        report.converged = decision.converged != 0;
        report.final_residual_norm = static_cast<Real>(decision.final_residual_norm);
        report.relative_residual = static_cast<Real>(decision.relative_residual);
        report.message = report.converged ? "fsils direct (gathered sparse LU)"
                                          : "fsils direct (residual above target)";
    } else {
        report.converged = false;
        report.numerical_breakdown = true;
        report.final_residual_norm = std::numeric_limits<Real>::infinity();
        report.relative_residual = std::numeric_limits<Real>::infinity();
        report.message = decision.status == 1 ? "fsils direct (numerical breakdown: singular factorization)"
                                              : "fsils direct (numerical breakdown: non-finite solution)";
        if (root) {
            FE_LOG_WARNING("FsilsDirectSolve: " + report.message +
                           (root_error.empty() ? std::string() : (" error='" + root_error + "'")));
        }
    }
    report.setup_time_seconds = root ? (t_assemble + t_analyze + t_factor) : 0.0;
    return report;
}

} // namespace backends
} // namespace FE
} // namespace svmp
