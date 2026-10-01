/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsBlockPreconditioners.h"

#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include "Array.h"
#include "Vector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace backends {

namespace {

using fe_fsi_linear_solver::fsils_int;

template <int M>
[[nodiscard]] inline int dimension(int runtime_m) noexcept
{
    if constexpr (M > 0) {
        return M;
    } else {
        return runtime_m;
    }
}

/// C = A * B for row-major m x m blocks.
template <int M>
inline void blockMul(int m_rt, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C)
{
    const int m = dimension<M>(m_rt);
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
            C[r * m + c] = 0.0;
        }
        for (int k = 0; k < m; ++k) {
            const double a = A[r * m + k];
            for (int c = 0; c < m; ++c) {
                C[r * m + c] += a * B[k * m + c];
            }
        }
    }
}

/// C -= A * B for row-major m x m blocks.
template <int M>
inline void blockSubMul(int m_rt, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C)
{
    const int m = dimension<M>(m_rt);
    for (int r = 0; r < m; ++r) {
        for (int k = 0; k < m; ++k) {
            const double a = A[r * m + k];
            for (int c = 0; c < m; ++c) {
                C[r * m + c] -= a * B[k * m + c];
            }
        }
    }
}

/// y -= A x.
template <int M>
inline void blockSubMatVec(int m_rt, const double* __restrict__ A, const double* __restrict__ x, double* __restrict__ y)
{
    const int m = dimension<M>(m_rt);
    for (int r = 0; r < m; ++r) {
        double s = 0.0;
        for (int c = 0; c < m; ++c) {
            s += A[r * m + c] * x[c];
        }
        y[r] -= s;
    }
}

/// y = A x.
template <int M>
inline void blockMatVec(int m_rt, const double* __restrict__ A, const double* __restrict__ x, double* __restrict__ y)
{
    const int m = dimension<M>(m_rt);
    for (int r = 0; r < m; ++r) {
        double s = 0.0;
        for (int c = 0; c < m; ++c) {
            s += A[r * m + c] * x[c];
        }
        y[r] = s;
    }
}

template <int M>
void factorImpl(const FsilsOwnedBlockGraph& g,
                int m_rt,
                std::vector<double>& lu,
                std::vector<double>& dinv,
                double& flops,
                int& regularized)
{
    const int m = dimension<M>(m_rt);
    const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    const int n = g.n;
    std::vector<int> pos(static_cast<std::size_t>(n), -1);
    std::vector<double> tmp(mm, 0.0);
    dinv.assign(static_cast<std::size_t>(n) * mm, 0.0);
    double count = 0.0;
    int reg = 0;
    const double block_flops = 2.0 * static_cast<double>(mm) * static_cast<double>(m);

    for (int i = 0; i < n; ++i) {
        const int row_begin = g.row_ptr[static_cast<std::size_t>(i)];
        const int row_end = g.row_ptr[static_cast<std::size_t>(i) + 1u];
        const int di = g.diag[static_cast<std::size_t>(i)];
        for (int p = row_begin; p < row_end; ++p) {
            pos[static_cast<std::size_t>(g.cols[static_cast<std::size_t>(p)])] = p;
        }
        for (int p = row_begin; p < di; ++p) {
            const int k = g.cols[static_cast<std::size_t>(p)];
            double* lik = lu.data() + static_cast<std::size_t>(p) * mm;
            blockMul<M>(m, lik, dinv.data() + static_cast<std::size_t>(k) * mm, tmp.data());
            std::copy(tmp.begin(), tmp.end(), lik);
            count += block_flops;
            const int k_end = g.row_ptr[static_cast<std::size_t>(k) + 1u];
            for (int q = g.diag[static_cast<std::size_t>(k)] + 1; q < k_end; ++q) {
                const int pj = pos[static_cast<std::size_t>(g.cols[static_cast<std::size_t>(q)])];
                if (pj < 0) {
                    continue;
                }
                blockSubMul<M>(m, lik, lu.data() + static_cast<std::size_t>(q) * mm,
                               lu.data() + static_cast<std::size_t>(pj) * mm);
                count += block_flops;
            }
        }
        reg += invertDenseBlock(m, lu.data() + static_cast<std::size_t>(di) * mm,
                                dinv.data() + static_cast<std::size_t>(i) * mm);
        count += block_flops;
        for (int p = row_begin; p < row_end; ++p) {
            pos[static_cast<std::size_t>(g.cols[static_cast<std::size_t>(p)])] = -1;
        }
    }
    flops = count;
    regularized = reg;
}

template <int M>
void solveImpl(const FsilsOwnedBlockGraph& g,
               int m_rt,
               const std::vector<double>& lu,
               const std::vector<double>& dinv,
               const double* in,
               double* out)
{
    const int m = dimension<M>(m_rt);
    const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    const int n = g.n;
    if (out != in) {
        std::copy(in, in + static_cast<std::size_t>(n) * static_cast<std::size_t>(m), out);
    }
    // Forward substitution with unit lower blocks.
    for (int i = 0; i < n; ++i) {
        double* yi = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        const int row_begin = g.row_ptr[static_cast<std::size_t>(i)];
        const int di = g.diag[static_cast<std::size_t>(i)];
        for (int p = row_begin; p < di; ++p) {
            const int k = g.cols[static_cast<std::size_t>(p)];
            blockSubMatVec<M>(m, lu.data() + static_cast<std::size_t>(p) * mm,
                              out + static_cast<std::size_t>(k) * static_cast<std::size_t>(m), yi);
        }
    }
    // Backward substitution with the inverted diagonal blocks.
    double t_stack[16];
    std::vector<double> t_heap;
    double* t = t_stack;
    if (m > 16) {
        t_heap.resize(static_cast<std::size_t>(m));
        t = t_heap.data();
    }
    for (int i = n - 1; i >= 0; --i) {
        double* zi = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        for (int c = 0; c < m; ++c) {
            t[c] = zi[c];
        }
        const int di = g.diag[static_cast<std::size_t>(i)];
        const int row_end = g.row_ptr[static_cast<std::size_t>(i) + 1u];
        for (int p = di + 1; p < row_end; ++p) {
            const int j = g.cols[static_cast<std::size_t>(p)];
            blockSubMatVec<M>(m, lu.data() + static_cast<std::size_t>(p) * mm,
                              out + static_cast<std::size_t>(j) * static_cast<std::size_t>(m), t);
        }
        blockMatVec<M>(m, dinv.data() + static_cast<std::size_t>(i) * mm, t, zi);
    }
}

template <typename F>
void dispatchBlockSize(int m, F&& f)
{
    switch (m) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: f(std::integral_constant<int, 0>{}); break;
    }
}

[[nodiscard]] std::uint64_t fnv1a(std::uint64_t h, const void* data, std::size_t bytes) noexcept
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

// ---------------------------------------------------------------------------
// Dense block inverse
// ---------------------------------------------------------------------------

int invertDenseBlock(int d, const double* block, double* inverse)
{
    const std::size_t dd = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    std::vector<double> a(block, block + dd);
    std::vector<double> inv(dd, 0.0);
    for (int i = 0; i < d; ++i) {
        inv[static_cast<std::size_t>(i * d + i)] = 1.0;
    }
    double scale = 0.0;
    for (const double v : a) {
        scale = std::max(scale, std::abs(v));
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        // A vanishing (or nonfinite) block carries no information: use identity.
        std::copy(inv.begin(), inv.end(), inverse);
        return d;
    }
    const double tiny = std::numeric_limits<double>::epsilon() * scale * static_cast<double>(d);
    int regularized = 0;
    for (int col = 0; col < d; ++col) {
        int piv = col;
        double best = std::abs(a[static_cast<std::size_t>(col * d + col)]);
        for (int r = col + 1; r < d; ++r) {
            const double v = std::abs(a[static_cast<std::size_t>(r * d + col)]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (int c = 0; c < d; ++c) {
                std::swap(a[static_cast<std::size_t>(col * d + c)], a[static_cast<std::size_t>(piv * d + c)]);
                std::swap(inv[static_cast<std::size_t>(col * d + c)], inv[static_cast<std::size_t>(piv * d + c)]);
            }
        }
        double pivot = a[static_cast<std::size_t>(col * d + col)];
        if (!(std::abs(pivot) > tiny)) {
            pivot = (pivot < 0.0) ? -tiny : tiny;
            a[static_cast<std::size_t>(col * d + col)] = pivot;
            ++regularized;
        }
        const double inv_pivot = 1.0 / pivot;
        for (int c = 0; c < d; ++c) {
            a[static_cast<std::size_t>(col * d + c)] *= inv_pivot;
            inv[static_cast<std::size_t>(col * d + c)] *= inv_pivot;
        }
        for (int r = 0; r < d; ++r) {
            if (r == col) {
                continue;
            }
            const double f = a[static_cast<std::size_t>(r * d + col)];
            if (f == 0.0) {
                continue;
            }
            for (int c = 0; c < d; ++c) {
                a[static_cast<std::size_t>(r * d + c)] -= f * a[static_cast<std::size_t>(col * d + c)];
                inv[static_cast<std::size_t>(r * d + c)] -= f * inv[static_cast<std::size_t>(col * d + c)];
            }
        }
    }
    std::copy(inv.begin(), inv.end(), inverse);
    return regularized;
}

// ---------------------------------------------------------------------------
// Owned block graph
// ---------------------------------------------------------------------------

std::uint64_t FsilsOwnedBlockGraph::computeSignature(const fe_fsi_linear_solver::FSILS_lhsType& lhs)
{
    std::uint64_t h = 1469598103934665603ULL;
    const std::int64_t header[3] = {static_cast<std::int64_t>(lhs.nNo),
                                    static_cast<std::int64_t>(lhs.mynNo),
                                    static_cast<std::int64_t>(lhs.nnz)};
    h = fnv1a(h, header, sizeof(header));
    if (lhs.rowPtr.size() > 0) {
        h = fnv1a(h, lhs.rowPtr.data(), static_cast<std::size_t>(lhs.rowPtr.size()) * sizeof(fsils_int));
    }
    if (lhs.colPtr.size() > 0) {
        h = fnv1a(h, lhs.colPtr.data(), static_cast<std::size_t>(lhs.colPtr.size()) * sizeof(fsils_int));
    }
    return h;
}

void FsilsOwnedBlockGraph::build(const fe_fsi_linear_solver::FSILS_lhsType& lhs)
{
    FE_THROW_IF(lhs.mynNo < 0 || lhs.mynNo > lhs.nNo, FEException,
                "FsilsOwnedBlockGraph: invalid owned node count");
    FE_THROW_IF(lhs.nnz > static_cast<fsils_int>(std::numeric_limits<int>::max()), FEException,
                "FsilsOwnedBlockGraph: nnz exceeds int range");
    n = static_cast<int>(lhs.mynNo);
    row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
    cols.clear();
    src.clear();
    diag.assign(static_cast<std::size_t>(n), -1);
    cols.reserve(static_cast<std::size_t>(lhs.nnz));
    src.reserve(static_cast<std::size_t>(lhs.nnz));
    std::vector<std::pair<int, int>> row;
    for (int r = 0; r < n; ++r) {
        row.clear();
        for (fsils_int nz = lhs.rowPtr(0, r); nz <= lhs.rowPtr(1, r); ++nz) {
            const fsils_int c = lhs.colPtr(static_cast<int>(nz));
            if (c < 0 || c >= static_cast<fsils_int>(n)) {
                continue;
            }
            row.emplace_back(static_cast<int>(c), static_cast<int>(nz));
        }
        std::sort(row.begin(), row.end());
        for (const auto& [c, nz] : row) {
            if (c == r) {
                diag[static_cast<std::size_t>(r)] = static_cast<int>(cols.size());
            }
            cols.push_back(c);
            src.push_back(nz);
        }
        row_ptr[static_cast<std::size_t>(r) + 1u] = static_cast<int>(cols.size());
        FE_THROW_IF(diag[static_cast<std::size_t>(r)] < 0, FEException,
                    "FsilsOwnedBlockGraph: operator row without a diagonal block");
    }
    signature = computeSignature(lhs);
}

// ---------------------------------------------------------------------------
// Block ILU(0)
// ---------------------------------------------------------------------------

void BlockIlu0Factorization::factor(const FsilsOwnedBlockGraph& graph, int m, std::vector<double> blocks)
{
    FE_THROW_IF(m <= 0, InvalidArgumentException, "BlockIlu0Factorization: block size must be positive");
    const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    FE_THROW_IF(blocks.size() != graph.nnz() * mm, InvalidArgumentException,
                "BlockIlu0Factorization: block array size mismatch");
    graph_ = &graph;
    m_ = m;
    lu_ = std::move(blocks);
    dispatchBlockSize(m, [&](auto tag) {
        constexpr int M = decltype(tag)::value;
        factorImpl<M>(graph, m, lu_, dinv_, factor_flops_, regularized_pivots_);
    });
    apply_flops_ = 2.0 * static_cast<double>(mm) *
                   (static_cast<double>(graph.nnz()));
}

void BlockIlu0Factorization::solve(const double* in, double* out) const
{
    FE_THROW_IF(graph_ == nullptr, FEException, "BlockIlu0Factorization: not factored");
    dispatchBlockSize(m_, [&](auto tag) {
        constexpr int M = decltype(tag)::value;
        solveImpl<M>(*graph_, m_, lu_, dinv_, in, out);
    });
}

// ---------------------------------------------------------------------------
// Krylov right preconditioner with reuse
// ---------------------------------------------------------------------------

class FsilsKrylovPreconditioner::Applier final : public fe_fsi_linear_solver::FSILS_rightPreconditioner {
public:
    explicit Applier(const FsilsKrylovPreconditioner& owner) : owner_(owner) {}
    void apply(const Array<double>& in, Array<double>& out) const override { owner_.apply(in, out); }

private:
    const FsilsKrylovPreconditioner& owner_;
};

FsilsKrylovPreconditioner::FsilsKrylovPreconditioner()
    : applier_(std::make_unique<Applier>(*this))
{
}

FsilsKrylovPreconditioner::~FsilsKrylovPreconditioner() = default;

std::string FsilsKrylovPreconditioner::kindName(Kind kind)
{
    switch (kind) {
        case Kind::BlockIlu0: return "block-ilu0";
        case Kind::Simple: return "simple";
    }
    return "unknown";
}

void FsilsKrylovPreconditioner::configure(Kind kind, bool reuse, int constraint_component, int krylov_dim)
{
    const bool changed = !configured_ || kind != kind_ || reuse != reuse_ ||
                         constraint_component != constraint_component_ || krylov_dim != krylov_dim_;
    kind_ = kind;
    reuse_ = reuse;
    constraint_component_ = constraint_component;
    krylov_dim_ = krylov_dim;
    configured_ = true;
    if (changed) {
        invalidate();
    }
}

void FsilsKrylovPreconditioner::invalidate()
{
    policy_.invalidate();
    graph_.signature = 0;
    scaling_identity_ = true;
}

fe_fsi_linear_solver::FSILS_rightPreconditionerHook FsilsKrylovPreconditioner::makeHook()
{
    fe_fsi_linear_solver::FSILS_rightPreconditionerHook hook;
    hook.prepare = [this](const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                          int dof,
                          const Array<double>& Val,
                          const Array<double>* row_scale,
                          const Array<double>* col_scale,
                          bool force_refresh,
                          bool& fresh) {
        return prepare(lhs, dof, Val, row_scale, col_scale, force_refresh, fresh);
    };
    hook.finish = [this](int iterations, bool converged, bool fresh, bool retried) {
        finish(iterations, converged, fresh, retried);
    };
    return hook;
}

const fe_fsi_linear_solver::FSILS_rightPreconditioner*
FsilsKrylovPreconditioner::prepare(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                   int dof,
                                   const Array<double>& Val,
                                   const Array<double>* row_scale,
                                   const Array<double>* col_scale,
                                   bool force_refresh,
                                   bool& fresh)
{
    fresh = false;
    if (dof <= 0 || lhs.nNo <= 0) {
        return nullptr;
    }
    const std::uint64_t signature = FsilsOwnedBlockGraph::computeSignature(lhs);
    const bool local_structure_changed = signature != graph_.signature || dof != dof_ ||
                                         static_cast<int>(lhs.nnz) != lhs_nnz_ ||
                                         static_cast<int>(lhs.nNo) != nNo_;
    n_tasks_ = lhs.commu.nTasks;
    comm_ = lhs.commu.comm;
    int changed = local_structure_changed ? 1 : 0;
    if (n_tasks_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_LOR, comm_);
    }
    const bool structure_changed = changed != 0;
    auto decision = policy_.beforeSolve(reuse_, structure_changed);
    if (force_refresh) {
        decision = {true, PreconditionerReusePolicy::Reason::StaleFailure};
    }

    if (decision.refresh) {
        const auto t0 = std::chrono::steady_clock::now();
        if (structure_changed || graph_.n != static_cast<int>(lhs.mynNo)) {
            graph_.build(lhs);
        }
        dof_ = dof;
        lhs_nnz_ = static_cast<int>(lhs.nnz);
        nNo_ = static_cast<int>(lhs.nNo);
        refresh(lhs, dof, Val, row_scale, col_scale);
        policy_.recordRefresh();
        stats_.last_setup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        stats_.setup_seconds += stats_.last_setup_seconds;
        ++stats_.refreshes;
        fresh = true;
    } else {
        updateScalingRatios(dof, row_scale, col_scale);
        stats_.last_setup_seconds = 0.0;
    }
    stats_.last_reason = decision.reason;
    stats_.last_fresh = fresh;
    return applier_.get();
}

void FsilsKrylovPreconditioner::refresh(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                        int dof,
                                        const Array<double>& Val,
                                        const Array<double>* row_scale,
                                        const Array<double>* col_scale)
{
    (void)lhs;
    const std::size_t dd = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t nnz = graph_.nnz();
    const double* val = Val.data();
    const int n = graph_.n;

    Kind effective = kind_;
    if (effective == Kind::Simple && (dof < 2 || constraint_component_ < 0 || constraint_component_ >= dof)) {
        effective = Kind::BlockIlu0;
    }

    if (effective == Kind::BlockIlu0) {
        std::vector<double> blocks(nnz * dd);
        for (std::size_t p = 0; p < nnz; ++p) {
            const double* srcb = val + static_cast<std::size_t>(graph_.src[p]) * dd;
            std::copy(srcb, srcb + dd, blocks.data() + p * dd);
        }
        factor_.factor(graph_, dof, std::move(blocks));
        schur_factor_ = BlockIlu0Factorization{};
        d_blocks_.clear();
        g_blocks_.clear();
        dk_inv_.clear();
        setup_flops_ = factor_.factorFlops();
        apply_flops_ = factor_.applyFlops() + 2.0 * static_cast<double>(dd) * static_cast<double>(n);
        stats_.regularized_pivots = factor_.regularizedPivots();
    } else {
        const int pc = constraint_component_;
        const int m = dof - 1;
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        std::vector<int> vcomp;
        vcomp.reserve(static_cast<std::size_t>(m));
        for (int c = 0; c < dof; ++c) {
            if (c != pc) {
                vcomp.push_back(c);
            }
        }
        std::vector<double> k_blocks(nnz * mm);
        std::vector<double> s_values(nnz);
        d_blocks_.assign(nnz * static_cast<std::size_t>(m), 0.0);
        g_blocks_.assign(nnz * static_cast<std::size_t>(m), 0.0);
        for (std::size_t p = 0; p < nnz; ++p) {
            const double* b = val + static_cast<std::size_t>(graph_.src[p]) * dd;
            double* kb = k_blocks.data() + p * mm;
            for (int r = 0; r < m; ++r) {
                const int rr = vcomp[static_cast<std::size_t>(r)];
                for (int c = 0; c < m; ++c) {
                    kb[static_cast<std::size_t>(r * m + c)] =
                        b[static_cast<std::size_t>(rr * dof + vcomp[static_cast<std::size_t>(c)])];
                }
                g_blocks_[p * static_cast<std::size_t>(m) + static_cast<std::size_t>(r)] =
                    b[static_cast<std::size_t>(rr * dof + pc)];
                d_blocks_[p * static_cast<std::size_t>(m) + static_cast<std::size_t>(r)] =
                    b[static_cast<std::size_t>(pc * dof + rr)];
            }
            s_values[p] = b[static_cast<std::size_t>(pc * dof + pc)];
        }
        // Nodal diagonal blocks of K and their inverses.
        dk_inv_.assign(static_cast<std::size_t>(n) * mm, 0.0);
        int regularized = 0;
        for (int i = 0; i < n; ++i) {
            const int di = graph_.diag[static_cast<std::size_t>(i)];
            regularized += invertDenseBlock(m, k_blocks.data() + static_cast<std::size_t>(di) * mm,
                                            dk_inv_.data() + static_cast<std::size_t>(i) * mm);
        }
        // S = C - D D_K^{-1} G on the operator graph.
        std::vector<int> pos(static_cast<std::size_t>(n), -1);
        std::vector<double> w(static_cast<std::size_t>(m), 0.0);
        double schur_flops = 0.0;
        for (int i = 0; i < n; ++i) {
            const int rb = graph_.row_ptr[static_cast<std::size_t>(i)];
            const int re = graph_.row_ptr[static_cast<std::size_t>(i) + 1u];
            for (int p = rb; p < re; ++p) {
                pos[static_cast<std::size_t>(graph_.cols[static_cast<std::size_t>(p)])] = p;
            }
            for (int pk = rb; pk < re; ++pk) {
                const int k = graph_.cols[static_cast<std::size_t>(pk)];
                const double* dik = d_blocks_.data() + static_cast<std::size_t>(pk) * static_cast<std::size_t>(m);
                const double* dkinv = dk_inv_.data() + static_cast<std::size_t>(k) * mm;
                for (int c = 0; c < m; ++c) {
                    double s = 0.0;
                    for (int r = 0; r < m; ++r) {
                        s += dik[r] * dkinv[static_cast<std::size_t>(r * m + c)];
                    }
                    w[static_cast<std::size_t>(c)] = s;
                }
                schur_flops += 2.0 * static_cast<double>(mm);
                const int kb = graph_.row_ptr[static_cast<std::size_t>(k)];
                const int ke = graph_.row_ptr[static_cast<std::size_t>(k) + 1u];
                for (int q = kb; q < ke; ++q) {
                    const int pj = pos[static_cast<std::size_t>(graph_.cols[static_cast<std::size_t>(q)])];
                    if (pj < 0) {
                        continue;
                    }
                    const double* gkj = g_blocks_.data() + static_cast<std::size_t>(q) * static_cast<std::size_t>(m);
                    double s = 0.0;
                    for (int c = 0; c < m; ++c) {
                        s += w[static_cast<std::size_t>(c)] * gkj[c];
                    }
                    s_values[static_cast<std::size_t>(pj)] -= s;
                    schur_flops += 2.0 * static_cast<double>(m);
                }
            }
            for (int p = rb; p < re; ++p) {
                pos[static_cast<std::size_t>(graph_.cols[static_cast<std::size_t>(p)])] = -1;
            }
        }
        factor_.factor(graph_, m, std::move(k_blocks));
        schur_factor_.factor(graph_, 1, std::move(s_values));
        setup_flops_ = factor_.factorFlops() + schur_factor_.factorFlops() + schur_flops +
                       2.0 * static_cast<double>(mm) * static_cast<double>(m) * static_cast<double>(n);
        apply_flops_ = factor_.applyFlops() + schur_factor_.applyFlops() +
                       4.0 * static_cast<double>(m) * static_cast<double>(nnz) +
                       2.0 * static_cast<double>(mm) * static_cast<double>(n);
        stats_.regularized_pivots =
            regularized + factor_.regularizedPivots() + schur_factor_.regularizedPivots();
    }

    // Remember the scalings the factorization was built with.
    const std::size_t owned = static_cast<std::size_t>(n) * static_cast<std::size_t>(dof);
    row_scale_at_factor_.clear();
    col_scale_at_factor_.clear();
    if (row_scale != nullptr && static_cast<std::size_t>(row_scale->size()) >= owned &&
        col_scale != nullptr && static_cast<std::size_t>(col_scale->size()) >= owned) {
        row_scale_at_factor_.assign(row_scale->data(), row_scale->data() + owned);
        col_scale_at_factor_.assign(col_scale->data(), col_scale->data() + owned);
    }
    scaling_identity_ = true;
    ratio_in_.clear();
    ratio_out_.clear();
}

void FsilsKrylovPreconditioner::updateScalingRatios(int dof,
                                                    const Array<double>* row_scale,
                                                    const Array<double>* col_scale)
{
    scaling_identity_ = true;
    const std::size_t owned = static_cast<std::size_t>(graph_.n) * static_cast<std::size_t>(dof);
    if (row_scale_at_factor_.size() != owned || col_scale_at_factor_.size() != owned ||
        row_scale == nullptr || col_scale == nullptr ||
        static_cast<std::size_t>(row_scale->size()) < owned ||
        static_cast<std::size_t>(col_scale->size()) < owned) {
        return;
    }
    ratio_in_.resize(owned);
    ratio_out_.resize(owned);
    const double* w1 = row_scale->data();
    const double* w2 = col_scale->data();
    bool identity = true;
    for (std::size_t i = 0; i < owned; ++i) {
        const double r1 = (w1[i] != 0.0) ? row_scale_at_factor_[i] / w1[i] : 1.0;
        const double r2 = (w2[i] != 0.0) ? col_scale_at_factor_[i] / w2[i] : 1.0;
        ratio_in_[i] = std::isfinite(r1) ? r1 : 1.0;
        ratio_out_[i] = std::isfinite(r2) ? r2 : 1.0;
        identity = identity && ratio_in_[i] == 1.0 && ratio_out_[i] == 1.0;
    }
    scaling_identity_ = identity;
}

void FsilsKrylovPreconditioner::apply(const Array<double>& in, Array<double>& out) const
{
    const int n = graph_.n;
    const std::size_t owned = static_cast<std::size_t>(n) * static_cast<std::size_t>(dof_);
    const double* src = in.data();
    double* dst = out.data();
    work_a_.resize(owned);
    if (scaling_identity_) {
        std::copy(src, src + owned, work_a_.data());
    } else {
        for (std::size_t i = 0; i < owned; ++i) {
            work_a_[i] = src[i] * ratio_in_[i];
        }
    }
    if (!d_blocks_.empty()) {
        applySimple(work_a_.data(), dst);
    } else {
        applyBlockIlu0(work_a_.data(), dst);
    }
    if (!scaling_identity_) {
        for (std::size_t i = 0; i < owned; ++i) {
            dst[i] *= ratio_out_[i];
        }
    }
    const std::size_t total = static_cast<std::size_t>(out.size());
    for (std::size_t i = owned; i < total; ++i) {
        dst[i] = 0.0;
    }
}

void FsilsKrylovPreconditioner::applyForTesting(const Array<double>& in, Array<double>& out) const
{
    apply(in, out);
}

void FsilsKrylovPreconditioner::applyBlockIlu0(const double* in, double* out) const
{
    factor_.solve(in, out);
}

void FsilsKrylovPreconditioner::applySimple(const double* in, double* out) const
{
    const int n = graph_.n;
    const int dof = dof_;
    const int pc = constraint_component_;
    const int m = dof - 1;
    const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
    work_b_.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    work_c_.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double* vi = in + static_cast<std::size_t>(i) * static_cast<std::size_t>(dof);
        double* ui = work_b_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        int r = 0;
        for (int c = 0; c < dof; ++c) {
            if (c == pc) {
                continue;
            }
            ui[r++] = vi[c];
        }
        work_c_[static_cast<std::size_t>(i)] = vi[pc];
    }
    // u* = K~^{-1} r_u
    factor_.solve(work_b_.data(), work_b_.data());
    // r_p - D u*
    for (int i = 0; i < n; ++i) {
        double s = work_c_[static_cast<std::size_t>(i)];
        const int rb = graph_.row_ptr[static_cast<std::size_t>(i)];
        const int re = graph_.row_ptr[static_cast<std::size_t>(i) + 1u];
        for (int p = rb; p < re; ++p) {
            const int j = graph_.cols[static_cast<std::size_t>(p)];
            const double* dij = d_blocks_.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
            const double* uj = work_b_.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
            for (int c = 0; c < m; ++c) {
                s -= dij[c] * uj[c];
            }
        }
        work_c_[static_cast<std::size_t>(i)] = s;
    }
    // p = S~^{-1} (r_p - D u*)
    schur_factor_.solve(work_c_.data(), work_c_.data());
    // u = u* - D_K^{-1} G p, written with p into the output.
    std::vector<double> g(static_cast<std::size_t>(m), 0.0);
    std::vector<double> corr(static_cast<std::size_t>(m), 0.0);
    for (int i = 0; i < n; ++i) {
        std::fill(g.begin(), g.end(), 0.0);
        const int rb = graph_.row_ptr[static_cast<std::size_t>(i)];
        const int re = graph_.row_ptr[static_cast<std::size_t>(i) + 1u];
        for (int p = rb; p < re; ++p) {
            const int j = graph_.cols[static_cast<std::size_t>(p)];
            const double pj = work_c_[static_cast<std::size_t>(j)];
            const double* gij = g_blocks_.data() + static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
            for (int c = 0; c < m; ++c) {
                g[static_cast<std::size_t>(c)] += gij[c] * pj;
            }
        }
        const double* dkinv = dk_inv_.data() + static_cast<std::size_t>(i) * mm;
        for (int r = 0; r < m; ++r) {
            double s = 0.0;
            for (int c = 0; c < m; ++c) {
                s += dkinv[static_cast<std::size_t>(r * m + c)] * g[static_cast<std::size_t>(c)];
            }
            corr[static_cast<std::size_t>(r)] = s;
        }
        const double* ui = work_b_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        double* oi = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(dof);
        int r = 0;
        for (int c = 0; c < dof; ++c) {
            if (c == pc) {
                oi[c] = work_c_[static_cast<std::size_t>(i)];
                continue;
            }
            oi[c] = ui[r] - corr[static_cast<std::size_t>(r)];
            ++r;
        }
    }
}

void FsilsKrylovPreconditioner::finish(int iterations, bool converged, bool fresh, bool retried)
{
    ++stats_.solves;
    stats_.last_iterations = iterations;
    if (!fresh) {
        ++stats_.reuses;
    }
    if (retried) {
        ++stats_.stale_retries;
    }
    double refresh_cost_iterations = 0.0;
    if (fresh) {
        const double owned = static_cast<double>(graph_.n) * static_cast<double>(dof_);
        const double basis = static_cast<double>(
            std::min(std::max(iterations, 1), std::max(krylov_dim_, 1)));
        const double matvec = 2.0 * static_cast<double>(lhs_nnz_) * static_cast<double>(dof_) *
                              static_cast<double>(dof_);
        const double orthogonalization = 4.0 * owned * (0.5 * basis + 1.0);
        double flops[2] = {setup_flops_, matvec + apply_flops_ + orthogonalization};
        if (n_tasks_ > 1) {
            MPI_Allreduce(MPI_IN_PLACE, flops, 2, MPI_DOUBLE, MPI_SUM, comm_);
        }
        refresh_cost_iterations = flops[1] > 0.0 ? flops[0] / flops[1] : 0.0;
    }
    policy_.recordSolve(iterations, fresh, refresh_cost_iterations);

    std::ostringstream oss;
    oss << "FsilsKrylovPreconditioner: diagnostic=fsils_right_preconditioner"
        << " kind=" << kindName(d_blocks_.empty() ? Kind::BlockIlu0 : Kind::Simple)
        << " reuse=" << (reuse_ ? 1 : 0)
        << " action=" << (fresh ? "refresh" : "reuse")
        << " reason=" << PreconditionerReusePolicy::reasonName(stats_.last_reason)
        << " retried=" << (retried ? 1 : 0)
        << " iterations=" << iterations
        << " converged=" << (converged ? 1 : 0)
        << " fresh_iterations=" << policy_.freshIterations()
        << " excess_iterations=" << policy_.excessIterations()
        << " refresh_cost_iterations=" << policy_.refreshCostIterations()
        << " setup_s=" << stats_.last_setup_seconds
        << " setup_flops=" << setup_flops_
        << " apply_flops=" << apply_flops_
        << " regularized_pivots=" << stats_.regularized_pivots
        << " refreshes=" << stats_.refreshes
        << " reuses=" << stats_.reuses;
    FE_LOG_INFO(oss.str());
}

} // namespace backends
} // namespace FE
} // namespace svmp
