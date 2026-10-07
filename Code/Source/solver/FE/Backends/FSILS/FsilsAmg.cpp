/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsAmg.h"

#include "Backends/FSILS/FsilsBlockPreconditioners.h"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"
#include "Core/FEException.h"

#include "Array.h"
#include "Vector.h"

#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace backends {

std::uint64_t amgHashKey(std::uint64_t key) noexcept
{
    std::uint64_t z = key + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

namespace {

using fe_fsi_linear_solver::fsils_int;
using Clock = std::chrono::steady_clock;

constexpr int kOwnerShift = 40;
constexpr long long kIndexMask = (1LL << kOwnerShift) - 1;

[[nodiscard]] inline long long makeGid(int owner, long long index) noexcept
{
    return (static_cast<long long>(owner) << kOwnerShift) | index;
}
[[nodiscard]] inline int gidOwner(long long gid) noexcept
{
    return static_cast<int>(gid >> kOwnerShift);
}
[[nodiscard]] inline long long gidIndex(long long gid) noexcept
{
    return gid & kIndexMask;
}

[[nodiscard]] double seconds(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Owner -> ghost exchange plan of node-block vectors.
// ---------------------------------------------------------------------------
struct Halo {
    MPI_Comm comm{MPI_COMM_NULL};
    std::vector<int> ranks{};                 // ascending
    std::vector<std::vector<int>> send{};     // owned local nodes sent to ranks[k]
    std::vector<std::vector<int>> recv{};     // ghost local nodes received from ranks[k]
    mutable std::vector<std::vector<unsigned char>> sbuf{};
    mutable std::vector<std::vector<unsigned char>> rbuf{};
    mutable std::vector<MPI_Request> reqs{};

    void sortByRank()
    {
        std::vector<std::size_t> order(ranks.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ranks[a] < ranks[b]; });
        std::vector<int> r2;
        std::vector<std::vector<int>> s2, v2;
        for (const auto k : order) {
            if (send[k].empty() && recv[k].empty()) {
                continue;
            }
            r2.push_back(ranks[k]);
            s2.push_back(std::move(send[k]));
            v2.push_back(std::move(recv[k]));
        }
        ranks = std::move(r2);
        send = std::move(s2);
        recv = std::move(v2);
        sbuf.assign(ranks.size(), {});
        rbuf.assign(ranks.size(), {});
    }

    /// v[ghost] = v[owner copy]; `bs` values of type T per node.
    template <class T>
    void forward(T* v, int bs, int tag) const
    {
        const std::size_t nk = ranks.size();
        if (nk == 0) {
            return;
        }
        reqs.assign(2 * nk, MPI_REQUEST_NULL);
        const std::size_t w = sizeof(T) * static_cast<std::size_t>(bs);
        for (std::size_t k = 0; k < nk; ++k) {
            rbuf[k].resize(recv[k].size() * w);
            MPI_Irecv(rbuf[k].data(), static_cast<int>(rbuf[k].size()), MPI_BYTE, ranks[k], tag, comm, &reqs[k]);
        }
        for (std::size_t k = 0; k < nk; ++k) {
            sbuf[k].resize(send[k].size() * w);
            unsigned char* p = sbuf[k].data();
            for (const int i : send[k]) {
                std::memcpy(p, v + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs), w);
                p += w;
            }
            MPI_Isend(sbuf[k].data(), static_cast<int>(sbuf[k].size()), MPI_BYTE, ranks[k], tag, comm,
                      &reqs[nk + k]);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        for (std::size_t k = 0; k < nk; ++k) {
            const unsigned char* p = rbuf[k].data();
            for (const int i : recv[k]) {
                std::memcpy(v + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs), p, w);
                p += w;
            }
        }
    }

    /// v[owner] += v[ghost copies] in ascending neighbour-rank order.
    void reverseAdd(double* v, int bs, int tag) const
    {
        const std::size_t nk = ranks.size();
        if (nk == 0) {
            return;
        }
        reqs.assign(2 * nk, MPI_REQUEST_NULL);
        const std::size_t w = sizeof(double) * static_cast<std::size_t>(bs);
        for (std::size_t k = 0; k < nk; ++k) {
            rbuf[k].resize(send[k].size() * w);
            MPI_Irecv(rbuf[k].data(), static_cast<int>(rbuf[k].size()), MPI_BYTE, ranks[k], tag, comm, &reqs[k]);
        }
        for (std::size_t k = 0; k < nk; ++k) {
            sbuf[k].resize(recv[k].size() * w);
            unsigned char* p = sbuf[k].data();
            for (const int i : recv[k]) {
                std::memcpy(p, v + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs), w);
                p += w;
            }
            MPI_Isend(sbuf[k].data(), static_cast<int>(sbuf[k].size()), MPI_BYTE, ranks[k], tag, comm,
                      &reqs[nk + k]);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        for (std::size_t k = 0; k < nk; ++k) {
            const double* p = reinterpret_cast<const double*>(rbuf[k].data());
            for (const int i : send[k]) {
                double* dst = v + static_cast<std::size_t>(i) * static_cast<std::size_t>(bs);
                for (int c = 0; c < bs; ++c) {
                    dst[c] += p[c];
                }
                p += bs;
            }
        }
    }
};

constexpr int kTagForward = 7101;
constexpr int kTagReverse = 7102;
constexpr int kTagSetup = 7103;

// ---------------------------------------------------------------------------
// Small dense block kernels (row-major).
// ---------------------------------------------------------------------------
/// C = A * B (m x m).
inline void blockMul(int m, const double* A, const double* B, double* C)
{
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
            C[r * m + c] = 0.0;
        }
        for (int k = 0; k < m; ++k) {
            const double a = A[r * m + k];
            if (a == 0.0) {
                continue;
            }
            for (int c = 0; c < m; ++c) {
                C[r * m + c] += a * B[k * m + c];
            }
        }
    }
}
/// C += A * B.
inline void blockMulAdd(int m, const double* A, const double* B, double* C)
{
    for (int r = 0; r < m; ++r) {
        for (int k = 0; k < m; ++k) {
            const double a = A[r * m + k];
            if (a == 0.0) {
                continue;
            }
            for (int c = 0; c < m; ++c) {
                C[r * m + c] += a * B[k * m + c];
            }
        }
    }
}
/// C += A^T * B.
inline void blockTMulAdd(int m, const double* A, const double* B, double* C)
{
    for (int k = 0; k < m; ++k) {
        for (int r = 0; r < m; ++r) {
            const double a = A[k * m + r];
            if (a == 0.0) {
                continue;
            }
            for (int c = 0; c < m; ++c) {
                C[r * m + c] += a * B[k * m + c];
            }
        }
    }
}
/// y += A x.
inline void blockVecAdd(int m, const double* A, const double* x, double* y)
{
    for (int r = 0; r < m; ++r) {
        double s = 0.0;
        for (int c = 0; c < m; ++c) {
            s += A[r * m + c] * x[c];
        }
        y[r] += s;
    }
}
/// y += A^T x.
inline void blockTVecAdd(int m, const double* A, const double* x, double* y)
{
    for (int k = 0; k < m; ++k) {
        const double xk = x[k];
        if (xk == 0.0) {
            continue;
        }
        for (int c = 0; c < m; ++c) {
            y[c] += A[k * m + c] * xk;
        }
    }
}
[[nodiscard]] inline bool blockIsZero(int m, const double* A)
{
    for (int i = 0; i < m * m; ++i) {
        if (A[i] != 0.0) {
            return false;
        }
    }
    return true;
}

/// Calls f(std::integral_constant<int, M>) with M = bs for small block sizes
/// and M = 0 (run-time size) otherwise.
template <class F>
inline void dispatchBlockSize(int bs, F&& f)
{
    switch (bs) {
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

/// Sparse row of blocks keyed by a 64-bit column id, kept sorted by id.
struct BlockRow {
    std::vector<long long> ids{};
    std::vector<double> vals{};  // ids.size() * m * m

    double* slot(long long id, int m)
    {
        const auto it = std::lower_bound(ids.begin(), ids.end(), id);
        const auto pos = static_cast<std::size_t>(it - ids.begin());
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        if (it == ids.end() || *it != id) {
            ids.insert(it, id);
            vals.insert(vals.begin() + static_cast<std::ptrdiff_t>(pos * mm), mm, 0.0);
        }
        return vals.data() + pos * mm;
    }
    void clear()
    {
        ids.clear();
        vals.clear();
    }
};

// ---------------------------------------------------------------------------
// One level: owned rows with ghost columns.
// ---------------------------------------------------------------------------
struct Level {
    int bs{0};
    int n_owned{0};
    int n_local{0};
    std::vector<int> row_ptr{};
    std::vector<int> cols{};
    std::vector<int> diag{};
    const double* ext{nullptr};
    std::vector<std::int64_t> slot{};
    std::vector<double> vals{};
    const double* vbase{nullptr};   // contiguous block storage, set by finalizeStorage()
    Halo halo{};
    std::vector<std::uint64_t> key{};
    std::vector<long long> gid{};      // (owner, owner index) of every local node
    std::vector<int> agg_local{};      // owned nodes: local index of their aggregate on the next level
    std::vector<double> dinv{};
    double lambda{1.0};

    // Prolongator from the next level: rows = owned nodes here, columns =
    // local nodes of the next level, bs x bs blocks.
    std::vector<int> p_ptr{};
    std::vector<int> p_col{};
    std::vector<double> p_val{};

    mutable std::vector<double> b{}, x{}, r{}, z{}, d{};

    [[nodiscard]] std::size_t bb() const noexcept { return static_cast<std::size_t>(bs) * static_cast<std::size_t>(bs); }
    [[nodiscard]] const double* block(int p) const noexcept
    {
        if (vbase != nullptr) {
            return vbase + static_cast<std::size_t>(p) * bb();
        }
        return ext != nullptr ? ext + static_cast<std::size_t>(slot[static_cast<std::size_t>(p)]) * bb()
                              : vals.data() + static_cast<std::size_t>(p) * bb();
    }

    /// Use one base pointer when the blocks are stored contiguously in entry order.
    void finalizeStorage()
    {
        vbase = nullptr;
        if (ext == nullptr) {
            vbase = vals.data();
            return;
        }
        bool contiguous = !slot.empty();
        for (std::size_t p = 1; p < slot.size() && contiguous; ++p) {
            contiguous = slot[p] == slot[0] + static_cast<std::int64_t>(p);
        }
        if (contiguous) {
            vbase = ext + static_cast<std::size_t>(slot[0]) * bb();
        }
    }

    void allocateWork() const
    {
        const std::size_t n = static_cast<std::size_t>(n_local) * static_cast<std::size_t>(bs);
        b.assign(n, 0.0);
        x.assign(n, 0.0);
        r.assign(n, 0.0);
        z.assign(n, 0.0);
        d.assign(n, 0.0);
    }

    template <int M>
    void spmvImpl(const double* v, double* y) const
    {
        const int m = M > 0 ? M : bs;
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        double acc[16];
        for (int i = 0; i < n_owned; ++i) {
            for (int r = 0; r < m; ++r) {
                acc[r] = 0.0;
            }
            const int pe = row_ptr[static_cast<std::size_t>(i) + 1];
            for (int p = row_ptr[static_cast<std::size_t>(i)]; p < pe; ++p) {
                const double* B = vbase + static_cast<std::size_t>(p) * mm;
                const double* vj = v + static_cast<std::size_t>(cols[static_cast<std::size_t>(p)]) * static_cast<std::size_t>(m);
                for (int r = 0; r < m; ++r) {
                    double s = 0.0;
                    for (int c = 0; c < m; ++c) {
                        s += B[r * m + c] * vj[c];
                    }
                    acc[r] += s;
                }
            }
            double* yi = y + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int r = 0; r < m; ++r) {
                yi[r] = acc[r];
            }
        }
    }

    /// y = A v on owned rows; v must have current ghost values.
    void spmv(const double* v, double* y) const
    {
        if (vbase != nullptr && bs <= 16) {
            dispatchBlockSize(bs, [&](auto tag) { spmvImpl<decltype(tag)::value>(v, y); });
            return;
        }
        const std::size_t m = static_cast<std::size_t>(bs);
        for (int i = 0; i < n_owned; ++i) {
            double* yi = y + static_cast<std::size_t>(i) * m;
            std::fill(yi, yi + m, 0.0);
            for (int p = row_ptr[static_cast<std::size_t>(i)]; p < row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                blockVecAdd(bs, block(p), v + static_cast<std::size_t>(cols[static_cast<std::size_t>(p)]) * m, yi);
            }
        }
    }

    template <int M>
    void jacobiImpl(const double* rr, double* zz) const
    {
        const int m = M > 0 ? M : bs;
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        for (int i = 0; i < n_owned; ++i) {
            const double* D = dinv.data() + static_cast<std::size_t>(i) * mm;
            const double* ri = rr + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            double* zi = zz + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int r = 0; r < m; ++r) {
                double s = 0.0;
                for (int c = 0; c < m; ++c) {
                    s += D[r * m + c] * ri[c];
                }
                zi[r] = s;
            }
        }
    }

    /// z = D^{-1} r on owned rows (z must not alias r).
    void jacobi(const double* rr, double* zz) const
    {
        dispatchBlockSize(bs, [&](auto tag) { jacobiImpl<decltype(tag)::value>(rr, zz); });
    }

    /// x_fine += P x_coarse on owned rows (x_coarse with current ghost values).
    template <int M>
    void prolongImpl(const double* xc, double* x) const
    {
        const int m = M > 0 ? M : bs;
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        for (int i = 0; i < n_owned; ++i) {
            double* xi = x + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int p = p_ptr[static_cast<std::size_t>(i)]; p < p_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                const double* B = p_val.data() + static_cast<std::size_t>(p) * mm;
                const double* xj = xc + static_cast<std::size_t>(p_col[static_cast<std::size_t>(p)]) * static_cast<std::size_t>(m);
                for (int r = 0; r < m; ++r) {
                    double s = 0.0;
                    for (int c = 0; c < m; ++c) {
                        s += B[r * m + c] * xj[c];
                    }
                    xi[r] += s;
                }
            }
        }
    }

    /// b_coarse += P^T r on owned rows (b_coarse local, ghosts included).
    template <int M>
    void restrictImpl(const double* rf, double* bc) const
    {
        const int m = M > 0 ? M : bs;
        const std::size_t mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
        for (int i = 0; i < n_owned; ++i) {
            const double* ri = rf + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int p = p_ptr[static_cast<std::size_t>(i)]; p < p_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                const double* B = p_val.data() + static_cast<std::size_t>(p) * mm;
                double* bj = bc + static_cast<std::size_t>(p_col[static_cast<std::size_t>(p)]) * static_cast<std::size_t>(m);
                for (int k = 0; k < m; ++k) {
                    const double rk = ri[k];
                    if (rk == 0.0) {
                        continue;
                    }
                    for (int c = 0; c < m; ++c) {
                        bj[c] += B[k * m + c] * rk;
                    }
                }
            }
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
struct FsilsAmgHierarchy::Impl {
    MPI_Comm comm{MPI_COMM_NULL};
    int rank{0};
    int size{1};
    FsilsAmgOptions opt{};
    std::vector<Level> levels{};
    Stats stats{};
    std::vector<long long> finest_aggregates{};

    // Coarsest level, gathered on rank 0 in key order.
    struct Coarse {
        int bs{0};
        long long n_nodes{0};
        std::vector<int> counts{};       // owned nodes per rank (scalars = counts*bs)
        std::vector<int> displs{};
        std::vector<int> order{};        // root: gathered position (rank-major) -> key-sorted node
        using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
        std::unique_ptr<Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>>> lu{};
        bool ok{false};
        mutable std::vector<double> gathered{};
        mutable std::vector<double> sorted{};
        mutable Eigen::VectorXd rhs{};
    } coarse{};

    ~Impl()
    {
        if (comm != MPI_COMM_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Comm_free(&comm);
            }
        }
    }

    long long sumAll(long long v) const
    {
        long long out = v;
        MPI_Allreduce(&v, &out, 1, MPI_LONG_LONG, MPI_SUM, comm);
        return out;
    }
    double maxAll(double v) const
    {
        double out = v;
        MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
        return out;
    }

    void buildFinest(const fe_fsi_linear_solver::FSILS_lhsType& lhs, int dof, const double* val,
                     std::span<const std::uint64_t> owned_keys, bool copy_values);
    void setupSmoother(Level& L);
    long long aggregate(Level& L, std::vector<long long>& agg, std::vector<std::uint64_t>& coarse_keys);
    void buildProlongator(const Level& L, const std::vector<long long>& agg, std::vector<BlockRow>& P) const;
    Level galerkin(Level& L, const std::vector<long long>& agg, std::vector<BlockRow>& P,
                   std::vector<std::uint64_t>& coarse_keys, long long n_coarse_owned);
    void setupCoarse(const Level& L);
    void coarseSolve(const Level& L, const double* b, double* x) const;
    void chebyshev(const Level& L, const double* b, double* x, bool zero_guess) const;
    void vcycle(std::size_t l) const;
};

void FsilsAmgHierarchy::Impl::buildFinest(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                                          int dof,
                                          const double* val,
                                          std::span<const std::uint64_t> owned_keys,
                                          bool copy_values)
{
    Level L;
    L.bs = dof;
    L.n_owned = static_cast<int>(lhs.mynNo);
    L.n_local = static_cast<int>(lhs.nNo);
    const std::size_t dd = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    L.row_ptr.assign(static_cast<std::size_t>(L.n_owned) + 1u, 0);
    for (int i = 0; i < L.n_owned; ++i) {
        const auto s = lhs.rowPtr(0, i);
        const auto e = lhs.rowPtr(1, i);
        L.row_ptr[static_cast<std::size_t>(i) + 1u] =
            L.row_ptr[static_cast<std::size_t>(i)] + static_cast<int>(e >= s ? e - s + 1 : 0);
    }
    const std::size_t nnz = static_cast<std::size_t>(L.row_ptr.back());
    L.cols.resize(nnz);
    L.slot.resize(nnz);
    L.diag.assign(static_cast<std::size_t>(L.n_owned), -1);
    for (int i = 0; i < L.n_owned; ++i) {
        int p = L.row_ptr[static_cast<std::size_t>(i)];
        for (fsils_int j = lhs.rowPtr(0, i); j <= lhs.rowPtr(1, i); ++j, ++p) {
            const int c = static_cast<int>(lhs.colPtr(j));
            L.cols[static_cast<std::size_t>(p)] = c;
            L.slot[static_cast<std::size_t>(p)] = static_cast<std::int64_t>(j);
            if (c == i) {
                L.diag[static_cast<std::size_t>(i)] = p;
            }
        }
    }
    if (copy_values) {
        L.vals.resize(nnz * dd);
        for (std::size_t p = 0; p < nnz; ++p) {
            std::memcpy(L.vals.data() + p * dd, val + static_cast<std::size_t>(L.slot[p]) * dd, dd * sizeof(double));
        }
        L.slot.clear();
        L.ext = nullptr;
    } else {
        L.ext = val;
    }

    L.halo.comm = comm;
    const std::size_t nk = lhs.owned_halo_neighbor_ranks.size();
    for (std::size_t k = 0; k < nk; ++k) {
        L.halo.ranks.push_back(lhs.owned_halo_neighbor_ranks[k]);
        std::vector<int> s, r;
        if (k < lhs.owned_halo_send_nodes.size()) {
            for (const auto v : lhs.owned_halo_send_nodes[k]) {
                s.push_back(static_cast<int>(v));
            }
        }
        if (k < lhs.owned_halo_recv_nodes.size()) {
            for (const auto v : lhs.owned_halo_recv_nodes[k]) {
                r.push_back(static_cast<int>(v));
            }
        }
        L.halo.send.push_back(std::move(s));
        L.halo.recv.push_back(std::move(r));
    }
    L.halo.sortByRank();

    L.key.assign(static_cast<std::size_t>(L.n_local), 0u);
    const bool have_keys = owned_keys.size() >= static_cast<std::size_t>(L.n_owned);
    for (int i = 0; i < L.n_owned; ++i) {
        if (have_keys) {
            L.key[static_cast<std::size_t>(i)] = owned_keys[static_cast<std::size_t>(i)];
        } else {
            const int g = (static_cast<fsils_int>(i) < lhs.gNodes.size()) ? lhs.gNodes(i) : i;
            L.key[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(g);
        }
    }
    L.halo.forward(L.key.data(), 1, kTagSetup);
    L.gid.assign(static_cast<std::size_t>(L.n_local), -1);
    for (int i = 0; i < L.n_owned; ++i) {
        L.gid[static_cast<std::size_t>(i)] = makeGid(rank, i);
    }
    L.halo.forward(L.gid.data(), 1, kTagSetup);
    levels.push_back(std::move(L));
}

void FsilsAmgHierarchy::Impl::setupSmoother(Level& L)
{
    const int m = L.bs;
    const std::size_t mm = L.bb();
    L.dinv.assign(static_cast<std::size_t>(L.n_owned) * mm, 0.0);
    std::vector<double> ident(mm, 0.0);
    for (int c = 0; c < m; ++c) {
        ident[static_cast<std::size_t>(c * m + c)] = 1.0;
    }
    double local_max = 0.0;
    std::vector<double> w(mm, 0.0);
    std::vector<double> rowsum(static_cast<std::size_t>(m), 0.0);
    for (int i = 0; i < L.n_owned; ++i) {
        const int pd = L.diag[static_cast<std::size_t>(i)];
        double* dinv = L.dinv.data() + static_cast<std::size_t>(i) * mm;
        if (pd < 0) {
            std::copy(ident.begin(), ident.end(), dinv);
        } else {
            stats.regularized_pivots += invertDenseBlock(m, L.block(pd), dinv);
        }
        std::fill(rowsum.begin(), rowsum.end(), 0.0);
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            blockMul(m, dinv, L.block(p), w.data());
            for (int r = 0; r < m; ++r) {
                double s = 0.0;
                for (int c = 0; c < m; ++c) {
                    s += std::abs(w[static_cast<std::size_t>(r * m + c)]);
                }
                rowsum[static_cast<std::size_t>(r)] += s;
            }
        }
        for (int r = 0; r < m; ++r) {
            local_max = std::max(local_max, rowsum[static_cast<std::size_t>(r)]);
        }
    }
    L.lambda = maxAll(local_max);
    if (!(L.lambda > 0.0) || !std::isfinite(L.lambda)) {
        L.lambda = 1.0;
    }
    if (opt.lambda_iterations > 0) {
        const std::size_t nl = static_cast<std::size_t>(L.n_local) * static_cast<std::size_t>(m);
        const std::size_t no = static_cast<std::size_t>(L.n_owned) * static_cast<std::size_t>(m);
        std::vector<double> v(nl, 0.0), av(nl, 0.0), dv(nl, 0.0);
        for (int i = 0; i < L.n_owned; ++i) {
            for (int c = 0; c < m; ++c) {
                const std::uint64_t hk = amgHashKey(L.key[static_cast<std::size_t>(i)] * 31u + static_cast<std::uint64_t>(c));
                v[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(c)] =
                    1.0 + 0.5 * static_cast<double>(hk >> 11) * 0x1.0p-53;
            }
        }
        double estimate = 0.0;
        for (int k = 0; k < opt.lambda_iterations; ++k) {
            double sv = 0.0;
            for (std::size_t q = 0; q < no; ++q) {
                sv += v[q] * v[q];
            }
            L.halo.forward(v.data(), m, kTagSetup);
            L.spmv(v.data(), av.data());
            L.jacobi(av.data(), dv.data());
            double sd = 0.0;
            for (std::size_t q = 0; q < no; ++q) {
                sd += dv[q] * dv[q];
            }
            double sums[2] = {sv, sd};
            MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
            if (!(sums[1] > 0.0) || !(sums[0] > 0.0)) {
                break;
            }
            estimate = std::sqrt(sums[1] / sums[0]);
            const double scale = 1.0 / std::sqrt(sums[1]);
            for (std::size_t q = 0; q < no; ++q) {
                v[q] = dv[q] * scale;
            }
        }
        if (estimate > 0.0 && std::isfinite(estimate)) {
            L.lambda = std::min(L.lambda, 1.1 * estimate);
        }
    }
}

long long FsilsAmgHierarchy::Impl::aggregate(Level& L,
                                             std::vector<long long>& agg,
                                             std::vector<std::uint64_t>& coarse_keys)
{
    const int n = L.n_owned;
    const int nl = L.n_local;
    const int m = L.bs;
    // Couplings: off-diagonal blocks of owned rows that are nonzero and, with
    // a strength threshold, strong relative to the two diagonal blocks.
    auto frob = [m](const double* B) {
        double s = 0.0;
        for (int q = 0; q < m * m; ++q) {
            s += B[q] * B[q];
        }
        return std::sqrt(s);
    };
    const double theta = std::max(0.0, opt.strength_threshold);
    std::vector<double> dnorm(static_cast<std::size_t>(nl), 0.0);
    if (theta > 0.0) {
        for (int i = 0; i < n; ++i) {
            const int pd = L.diag[static_cast<std::size_t>(i)];
            dnorm[static_cast<std::size_t>(i)] = pd >= 0 ? frob(L.block(pd)) : 0.0;
        }
        L.halo.forward(dnorm.data(), 1, kTagSetup);
    }
    std::vector<unsigned char> conn(L.cols.size(), 0);
    std::vector<unsigned char> isolated(static_cast<std::size_t>(n), 1);
    for (int i = 0; i < n; ++i) {
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            const int j = L.cols[static_cast<std::size_t>(p)];
            if (j == i) {
                continue;
            }
            const double* B = L.block(p);
            if (blockIsZero(m, B)) {
                continue;
            }
            if (theta > 0.0 &&
                !(frob(B) > theta * std::sqrt(dnorm[static_cast<std::size_t>(i)] * dnorm[static_cast<std::size_t>(j)]))) {
                continue;
            }
            conn[static_cast<std::size_t>(p)] = 1;
            isolated[static_cast<std::size_t>(i)] = 0;
        }
    }

    // Priorities: 62-bit hashes of the keys.
    std::vector<std::uint64_t> h(static_cast<std::size_t>(nl));
    for (int i = 0; i < nl; ++i) {
        h[static_cast<std::size_t>(i)] = amgHashKey(L.key[static_cast<std::size_t>(i)]) >> 2;
    }
    // state: 0 out, 1 undecided, 2 root.
    std::vector<std::uint64_t> state(static_cast<std::size_t>(nl), 0u);
    for (int i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = isolated[static_cast<std::size_t>(i)] ? 0u : 1u;
    }
    auto tuple = [&](int i) -> std::uint64_t {
        const auto s = state[static_cast<std::size_t>(i)];
        return s == 0u ? 0u : ((s << 62) | h[static_cast<std::size_t>(i)]);
    };
    std::vector<std::uint64_t> t(static_cast<std::size_t>(nl), 0u), t1(static_cast<std::size_t>(nl), 0u),
        t2(static_cast<std::size_t>(nl), 0u);
    int rounds = 0;
    for (;; ++rounds) {
        long long undecided = 0;
        for (int i = 0; i < n; ++i) {
            undecided += state[static_cast<std::size_t>(i)] == 1u ? 1 : 0;
        }
        if (sumAll(undecided) == 0 || rounds > 200) {
            break;
        }
        for (int i = 0; i < n; ++i) {
            t[static_cast<std::size_t>(i)] = tuple(i);
        }
        L.halo.forward(t.data(), 1, kTagSetup);
        for (int i = 0; i < n; ++i) {
            std::uint64_t v = t[static_cast<std::size_t>(i)];
            for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                if (conn[static_cast<std::size_t>(p)]) {
                    v = std::max(v, t[static_cast<std::size_t>(L.cols[static_cast<std::size_t>(p)])]);
                }
            }
            t1[static_cast<std::size_t>(i)] = v;
        }
        L.halo.forward(t1.data(), 1, kTagSetup);
        for (int i = 0; i < n; ++i) {
            std::uint64_t v = t1[static_cast<std::size_t>(i)];
            for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                if (conn[static_cast<std::size_t>(p)]) {
                    v = std::max(v, t1[static_cast<std::size_t>(L.cols[static_cast<std::size_t>(p)])]);
                }
            }
            t2[static_cast<std::size_t>(i)] = v;
        }
        for (int i = 0; i < n; ++i) {
            if (state[static_cast<std::size_t>(i)] != 1u) {
                continue;
            }
            if (t2[static_cast<std::size_t>(i)] == t[static_cast<std::size_t>(i)]) {
                state[static_cast<std::size_t>(i)] = 2u;
            } else if ((t2[static_cast<std::size_t>(i)] >> 62) == 2u) {
                state[static_cast<std::size_t>(i)] = 0u;
            }
        }
    }
    stats.mis_rounds += rounds;
    // Remaining undecided nodes (only after the round cap) become roots.
    for (int i = 0; i < n; ++i) {
        if (state[static_cast<std::size_t>(i)] == 1u) {
            state[static_cast<std::size_t>(i)] = 2u;
        }
    }
    L.halo.forward(state.data(), 1, kTagSetup);

    agg.assign(static_cast<std::size_t>(nl), -1);
    coarse_keys.clear();
    long long n_coarse = 0;
    for (int i = 0; i < n; ++i) {
        if (state[static_cast<std::size_t>(i)] == 2u) {
            agg[static_cast<std::size_t>(i)] = makeGid(rank, n_coarse++);
            coarse_keys.push_back(L.key[static_cast<std::size_t>(i)]);
        }
    }
    L.halo.forward(agg.data(), 1, kTagSetup);
    // Phase 1: distance-1 neighbours of roots.
    for (int i = 0; i < n; ++i) {
        if (state[static_cast<std::size_t>(i)] == 2u || isolated[static_cast<std::size_t>(i)]) {
            continue;
        }
        std::uint64_t best = 0u;
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            const int j = L.cols[static_cast<std::size_t>(p)];
            if (!conn[static_cast<std::size_t>(p)] || state[static_cast<std::size_t>(j)] != 2u) {
                continue;
            }
            const std::uint64_t tj = h[static_cast<std::size_t>(j)] + 1u;
            if (tj > best) {
                best = tj;
                agg[static_cast<std::size_t>(i)] = agg[static_cast<std::size_t>(j)];
            }
        }
    }
    L.halo.forward(agg.data(), 1, kTagSetup);
    // Phase 2: remaining nodes join the neighbouring aggregate of highest priority.
    const std::vector<long long> agg1 = agg;
    for (int i = 0; i < n; ++i) {
        if (agg1[static_cast<std::size_t>(i)] >= 0 || isolated[static_cast<std::size_t>(i)]) {
            continue;
        }
        std::uint64_t best = 0u;
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            const int j = L.cols[static_cast<std::size_t>(p)];
            if (!conn[static_cast<std::size_t>(p)] || agg1[static_cast<std::size_t>(j)] < 0) {
                continue;
            }
            const std::uint64_t tj = h[static_cast<std::size_t>(j)] + 1u;
            if (tj > best) {
                best = tj;
                agg[static_cast<std::size_t>(i)] = agg1[static_cast<std::size_t>(j)];
            }
        }
    }
    // Phase 3: nodes without an aggregated neighbour form their own aggregate.
    for (int i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] < 0 && !isolated[static_cast<std::size_t>(i)]) {
            agg[static_cast<std::size_t>(i)] = makeGid(rank, n_coarse++);
            coarse_keys.push_back(L.key[static_cast<std::size_t>(i)]);
        }
    }
    L.halo.forward(agg.data(), 1, kTagSetup);
    return n_coarse;
}

void FsilsAmgHierarchy::Impl::buildProlongator(const Level& L,
                                               const std::vector<long long>& agg,
                                               std::vector<BlockRow>& P) const
{
    const int n = L.n_owned;
    const int m = L.bs;
    const std::size_t mm = L.bb();
    // Unknowns with an off-diagonal coupling (others are left to the smoother).
    std::vector<double> mask(static_cast<std::size_t>(L.n_local) * static_cast<std::size_t>(m), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            const double* B = L.block(p);
            const bool on_diag = L.cols[static_cast<std::size_t>(p)] == i;
            for (int r = 0; r < m; ++r) {
                for (int c = 0; c < m; ++c) {
                    if ((on_diag && r == c) || B[r * m + c] == 0.0) {
                        continue;
                    }
                    mask[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(r)] = 1.0;
                }
            }
        }
    }
    L.halo.forward(mask.data(), m, kTagSetup);

    const double omega = opt.smooth_prolongator ? opt.prolongator_omega / L.lambda : 0.0;
    P.assign(static_cast<std::size_t>(n), BlockRow{});
    std::vector<double> w(mm, 0.0);
    for (int i = 0; i < n; ++i) {
        BlockRow& row = P[static_cast<std::size_t>(i)];
        const double* mi = mask.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
        if (agg[static_cast<std::size_t>(i)] >= 0) {
            double* blk = row.slot(agg[static_cast<std::size_t>(i)], m);
            for (int c = 0; c < m; ++c) {
                blk[c * m + c] += mi[c];
            }
        }
        if (omega != 0.0) {
            const double* dinv = L.dinv.data() + static_cast<std::size_t>(i) * mm;
            for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1];
                 ++p) {
                const int j = L.cols[static_cast<std::size_t>(p)];
                const long long a = agg[static_cast<std::size_t>(j)];
                if (a < 0) {
                    continue;
                }
                const double* B = L.block(p);
                if (blockIsZero(m, B)) {
                    continue;
                }
                blockMul(m, dinv, B, w.data());
                const double* mj = mask.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                double* blk = row.slot(a, m);
                for (int r = 0; r < m; ++r) {
                    for (int c = 0; c < m; ++c) {
                        blk[r * m + c] -= omega * w[static_cast<std::size_t>(r * m + c)] * mj[c];
                    }
                }
            }
        }
        // Rows of unknowns without couplings stay zero: those unknowns are
        // solved by the smoother.
        for (int r = 0; r < m; ++r) {
            if (mi[r] != 0.0) {
                continue;
            }
            for (std::size_t e = 0; e < row.ids.size(); ++e) {
                double* blk = row.vals.data() + e * mm;
                for (int c = 0; c < m; ++c) {
                    blk[r * m + c] = 0.0;
                }
            }
        }
    }
}

Level FsilsAmgHierarchy::Impl::galerkin(Level& L,
                                        const std::vector<long long>& agg,
                                        std::vector<BlockRow>& P,
                                        std::vector<std::uint64_t>& coarse_keys,
                                        long long n_coarse_owned)
{
    const int n = L.n_owned;
    const int m = L.bs;
    const std::size_t mm = L.bb();

    // ---- P rows of ghost nodes from their owners --------------------------
    std::vector<BlockRow> Pg(static_cast<std::size_t>(L.n_local - n));
    {
        std::vector<int> len(static_cast<std::size_t>(L.n_local), 0);
        for (int i = 0; i < n; ++i) {
            len[static_cast<std::size_t>(i)] = static_cast<int>(P[static_cast<std::size_t>(i)].ids.size());
        }
        L.halo.forward(len.data(), 1, kTagSetup);
        const std::size_t nk = L.halo.ranks.size();
        std::vector<std::vector<long long>> sid(nk), rid(nk);
        std::vector<std::vector<double>> sval(nk), rval(nk);
        std::vector<MPI_Request> reqs(4 * nk, MPI_REQUEST_NULL);
        for (std::size_t k = 0; k < nk; ++k) {
            std::size_t cnt = 0;
            for (const int g : L.halo.recv[k]) {
                cnt += static_cast<std::size_t>(len[static_cast<std::size_t>(g)]);
            }
            rid[k].resize(cnt);
            rval[k].resize(cnt * mm);
            MPI_Irecv(rid[k].data(), static_cast<int>(cnt), MPI_LONG_LONG, L.halo.ranks[k], kTagSetup, comm,
                      &reqs[4 * k]);
            MPI_Irecv(rval[k].data(), static_cast<int>(cnt * mm), MPI_DOUBLE, L.halo.ranks[k], kTagSetup + 1, comm,
                      &reqs[4 * k + 1]);
        }
        for (std::size_t k = 0; k < nk; ++k) {
            for (const int s : L.halo.send[k]) {
                const BlockRow& row = P[static_cast<std::size_t>(s)];
                sid[k].insert(sid[k].end(), row.ids.begin(), row.ids.end());
                sval[k].insert(sval[k].end(), row.vals.begin(), row.vals.end());
            }
            MPI_Isend(sid[k].data(), static_cast<int>(sid[k].size()), MPI_LONG_LONG, L.halo.ranks[k], kTagSetup,
                      comm, &reqs[4 * k + 2]);
            MPI_Isend(sval[k].data(), static_cast<int>(sval[k].size()), MPI_DOUBLE, L.halo.ranks[k],
                      kTagSetup + 1, comm, &reqs[4 * k + 3]);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        for (std::size_t k = 0; k < nk; ++k) {
            std::size_t pos = 0;
            for (const int g : L.halo.recv[k]) {
                BlockRow& row = Pg[static_cast<std::size_t>(g - n)];
                const auto c = static_cast<std::size_t>(len[static_cast<std::size_t>(g)]);
                row.ids.assign(rid[k].begin() + static_cast<std::ptrdiff_t>(pos),
                               rid[k].begin() + static_cast<std::ptrdiff_t>(pos + c));
                row.vals.assign(rval[k].begin() + static_cast<std::ptrdiff_t>(pos * mm),
                                rval[k].begin() + static_cast<std::ptrdiff_t>((pos + c) * mm));
                pos += c;
            }
        }
    }
    auto prow = [&](int j) -> const BlockRow& {
        return j < n ? P[static_cast<std::size_t>(j)] : Pg[static_cast<std::size_t>(j - n)];
    };

    // ---- Galerkin contributions C(a,b) += P_ia^T (A P)_ib ------------------
    // Owned coarse rows accumulate locally; others go to their owners.
    std::vector<BlockRow> crow(static_cast<std::size_t>(n_coarse_owned));
    std::vector<std::vector<long long>> out_ids(static_cast<std::size_t>(size));   // pairs (row index, column id)
    std::vector<std::vector<double>> out_vals(static_cast<std::size_t>(size));
    BlockRow ap;
    std::vector<double> w(mm, 0.0);
    for (int i = 0; i < n; ++i) {
        const BlockRow& pi = P[static_cast<std::size_t>(i)];
        if (pi.ids.empty()) {
            continue;
        }
        ap.clear();
        for (int p = L.row_ptr[static_cast<std::size_t>(i)]; p < L.row_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
            const double* A = L.block(p);
            if (blockIsZero(m, A)) {
                continue;
            }
            const BlockRow& pj = prow(L.cols[static_cast<std::size_t>(p)]);
            for (std::size_t e = 0; e < pj.ids.size(); ++e) {
                blockMulAdd(m, A, pj.vals.data() + e * mm, ap.slot(pj.ids[e], m));
            }
        }
        for (std::size_t ea = 0; ea < pi.ids.size(); ++ea) {
            const long long a = pi.ids[ea];
            const double* Pa = pi.vals.data() + ea * mm;
            const int owner = gidOwner(a);
            for (std::size_t eb = 0; eb < ap.ids.size(); ++eb) {
                std::fill(w.begin(), w.end(), 0.0);
                blockTMulAdd(m, Pa, ap.vals.data() + eb * mm, w.data());
                if (owner == rank) {
                    double* dst = crow[static_cast<std::size_t>(gidIndex(a))].slot(ap.ids[eb], m);
                    for (std::size_t q = 0; q < mm; ++q) {
                        dst[q] += w[q];
                    }
                } else {
                    out_ids[static_cast<std::size_t>(owner)].push_back(gidIndex(a));
                    out_ids[static_cast<std::size_t>(owner)].push_back(ap.ids[eb]);
                    out_vals[static_cast<std::size_t>(owner)].insert(out_vals[static_cast<std::size_t>(owner)].end(),
                                                                     w.begin(), w.end());
                }
            }
        }
    }
    // Exchange contributions (counts first), added in ascending source rank order.
    {
        std::vector<int> scount(static_cast<std::size_t>(size), 0), rcount(static_cast<std::size_t>(size), 0);
        for (int r = 0; r < size; ++r) {
            scount[static_cast<std::size_t>(r)] = static_cast<int>(out_ids[static_cast<std::size_t>(r)].size() / 2);
        }
        MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
        std::vector<MPI_Request> reqs;
        std::vector<std::vector<long long>> in_ids(static_cast<std::size_t>(size));
        std::vector<std::vector<double>> in_vals(static_cast<std::size_t>(size));
        for (int r = 0; r < size; ++r) {
            const auto c = static_cast<std::size_t>(rcount[static_cast<std::size_t>(r)]);
            if (c == 0 || r == rank) {
                continue;
            }
            in_ids[static_cast<std::size_t>(r)].resize(2 * c);
            in_vals[static_cast<std::size_t>(r)].resize(c * mm);
            reqs.emplace_back();
            MPI_Irecv(in_ids[static_cast<std::size_t>(r)].data(), static_cast<int>(2 * c), MPI_LONG_LONG, r,
                      kTagSetup + 2, comm, &reqs.back());
            reqs.emplace_back();
            MPI_Irecv(in_vals[static_cast<std::size_t>(r)].data(), static_cast<int>(c * mm), MPI_DOUBLE, r,
                      kTagSetup + 3, comm, &reqs.back());
        }
        for (int r = 0; r < size; ++r) {
            if (scount[static_cast<std::size_t>(r)] == 0 || r == rank) {
                continue;
            }
            reqs.emplace_back();
            MPI_Isend(out_ids[static_cast<std::size_t>(r)].data(),
                      static_cast<int>(out_ids[static_cast<std::size_t>(r)].size()), MPI_LONG_LONG, r, kTagSetup + 2,
                      comm, &reqs.back());
            reqs.emplace_back();
            MPI_Isend(out_vals[static_cast<std::size_t>(r)].data(),
                      static_cast<int>(out_vals[static_cast<std::size_t>(r)].size()), MPI_DOUBLE, r, kTagSetup + 3,
                      comm, &reqs.back());
        }
        if (!reqs.empty()) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        }
        for (int r = 0; r < size; ++r) {
            const auto c = in_ids[static_cast<std::size_t>(r)].size() / 2;
            for (std::size_t e = 0; e < c; ++e) {
                const long long a = in_ids[static_cast<std::size_t>(r)][2 * e];
                const long long b = in_ids[static_cast<std::size_t>(r)][2 * e + 1];
                double* dst = crow[static_cast<std::size_t>(a)].slot(b, m);
                const double* src = in_vals[static_cast<std::size_t>(r)].data() + e * mm;
                for (std::size_t q = 0; q < mm; ++q) {
                    dst[q] += src[q];
                }
            }
        }
    }

    // ---- Coarse level numbering: owned, then ghosts sorted by id -----------
    Level C;
    C.bs = m;
    C.n_owned = static_cast<int>(n_coarse_owned);
    std::vector<long long> ghosts;
    for (const auto& row : crow) {
        for (const long long b : row.ids) {
            if (gidOwner(b) != rank) {
                ghosts.push_back(b);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (const long long a : P[static_cast<std::size_t>(i)].ids) {
            if (gidOwner(a) != rank) {
                ghosts.push_back(a);
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    C.n_local = C.n_owned + static_cast<int>(ghosts.size());
    auto local_of = [&](long long gid) -> int {
        if (gidOwner(gid) == rank) {
            return static_cast<int>(gidIndex(gid));
        }
        const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), gid);
        return C.n_owned + static_cast<int>(it - ghosts.begin());
    };

    // Coarse rows; empty unknowns get a unit diagonal.
    C.row_ptr.assign(static_cast<std::size_t>(C.n_owned) + 1u, 0);
    C.diag.assign(static_cast<std::size_t>(C.n_owned), -1);
    for (int a = 0; a < C.n_owned; ++a) {
        BlockRow& row = crow[static_cast<std::size_t>(a)];
        double* dblk = row.slot(makeGid(rank, a), m);
        for (int r = 0; r < m; ++r) {
            bool empty = true;
            for (std::size_t e = 0; e < row.ids.size() && empty; ++e) {
                const double* blk = row.vals.data() + e * mm;
                for (int c = 0; c < m; ++c) {
                    if (blk[r * m + c] != 0.0) {
                        empty = false;
                        break;
                    }
                }
            }
            if (empty) {
                dblk[r * m + r] = 1.0;
            }
        }
        C.row_ptr[static_cast<std::size_t>(a) + 1u] =
            C.row_ptr[static_cast<std::size_t>(a)] + static_cast<int>(row.ids.size());
    }
    const std::size_t cnnz = static_cast<std::size_t>(C.row_ptr.back());
    C.cols.resize(cnnz);
    C.vals.resize(cnnz * mm);
    for (int a = 0; a < C.n_owned; ++a) {
        const BlockRow& row = crow[static_cast<std::size_t>(a)];
        std::size_t p = static_cast<std::size_t>(C.row_ptr[static_cast<std::size_t>(a)]);
        for (std::size_t e = 0; e < row.ids.size(); ++e, ++p) {
            const int c = local_of(row.ids[e]);
            C.cols[p] = c;
            if (c == a) {
                C.diag[static_cast<std::size_t>(a)] = static_cast<int>(p);
            }
            std::memcpy(C.vals.data() + p * mm, row.vals.data() + e * mm, mm * sizeof(double));
        }
    }

    // ---- Coarse halo from requests to the owners ---------------------------
    C.halo.comm = comm;
    {
        std::vector<int> scount(static_cast<std::size_t>(size), 0), rcount(static_cast<std::size_t>(size), 0);
        for (const long long g : ghosts) {
            ++scount[static_cast<std::size_t>(gidOwner(g))];
        }
        MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);
        std::vector<int> sdispl(static_cast<std::size_t>(size) + 1u, 0), rdispl(static_cast<std::size_t>(size) + 1u, 0);
        for (int r = 0; r < size; ++r) {
            sdispl[static_cast<std::size_t>(r) + 1u] = sdispl[static_cast<std::size_t>(r)] + scount[static_cast<std::size_t>(r)];
            rdispl[static_cast<std::size_t>(r) + 1u] = rdispl[static_cast<std::size_t>(r)] + rcount[static_cast<std::size_t>(r)];
        }
        std::vector<long long> req(ghosts.size());
        for (std::size_t g = 0; g < ghosts.size(); ++g) {
            req[g] = gidIndex(ghosts[g]);  // ghosts are sorted by owner, then index
        }
        std::vector<long long> got(static_cast<std::size_t>(rdispl.back()));
        MPI_Alltoallv(req.data(), scount.data(), sdispl.data(), MPI_LONG_LONG, got.data(), rcount.data(),
                      rdispl.data(), MPI_LONG_LONG, comm);
        for (int r = 0; r < size; ++r) {
            if (r == rank || (scount[static_cast<std::size_t>(r)] == 0 && rcount[static_cast<std::size_t>(r)] == 0)) {
                continue;
            }
            C.halo.ranks.push_back(r);
            std::vector<int> s, v;
            for (int e = rdispl[static_cast<std::size_t>(r)]; e < rdispl[static_cast<std::size_t>(r) + 1]; ++e) {
                s.push_back(static_cast<int>(got[static_cast<std::size_t>(e)]));
            }
            for (int e = sdispl[static_cast<std::size_t>(r)]; e < sdispl[static_cast<std::size_t>(r) + 1]; ++e) {
                v.push_back(C.n_owned + e);
            }
            C.halo.send.push_back(std::move(s));
            C.halo.recv.push_back(std::move(v));
        }
        C.halo.sortByRank();
    }
    C.key.assign(static_cast<std::size_t>(C.n_local), 0u);
    std::copy(coarse_keys.begin(), coarse_keys.end(), C.key.begin());
    C.halo.forward(C.key.data(), 1, kTagSetup);
    C.gid.resize(static_cast<std::size_t>(C.n_local));
    for (int a = 0; a < C.n_owned; ++a) {
        C.gid[static_cast<std::size_t>(a)] = makeGid(rank, a);
    }
    std::copy(ghosts.begin(), ghosts.end(), C.gid.begin() + C.n_owned);
    L.agg_local.assign(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i) {
        if (agg[static_cast<std::size_t>(i)] >= 0) {
            L.agg_local[static_cast<std::size_t>(i)] = local_of(agg[static_cast<std::size_t>(i)]);
        }
    }

    // ---- Prolongator columns in coarse local numbering ---------------------
    L.p_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);
    for (int i = 0; i < n; ++i) {
        L.p_ptr[static_cast<std::size_t>(i) + 1u] =
            L.p_ptr[static_cast<std::size_t>(i)] + static_cast<int>(P[static_cast<std::size_t>(i)].ids.size());
    }
    L.p_col.resize(static_cast<std::size_t>(L.p_ptr.back()));
    L.p_val.resize(static_cast<std::size_t>(L.p_ptr.back()) * mm);
    for (int i = 0; i < n; ++i) {
        const BlockRow& row = P[static_cast<std::size_t>(i)];
        std::size_t p = static_cast<std::size_t>(L.p_ptr[static_cast<std::size_t>(i)]);
        for (std::size_t e = 0; e < row.ids.size(); ++e, ++p) {
            L.p_col[p] = local_of(row.ids[e]);
            std::memcpy(L.p_val.data() + p * mm, row.vals.data() + e * mm, mm * sizeof(double));
        }
    }
    return C;
}

void FsilsAmgHierarchy::Impl::setupCoarse(const Level& L)
{
    const int m = L.bs;
    const std::size_t mm = L.bb();
    coarse.lu.reset();
    coarse.ok = false;
    coarse.bs = m;
    coarse.counts.assign(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&L.n_owned, 1, MPI_INT, coarse.counts.data(), 1, MPI_INT, comm);
    coarse.displs.assign(static_cast<std::size_t>(size) + 1u, 0);
    for (int r = 0; r < size; ++r) {
        coarse.displs[static_cast<std::size_t>(r) + 1u] =
            coarse.displs[static_cast<std::size_t>(r)] + coarse.counts[static_cast<std::size_t>(r)];
    }
    const int total = coarse.displs.back();
    coarse.n_nodes = total;
    // Every rank learns all keys (rank-major) and the key order.
    std::vector<std::uint64_t> all_keys(static_cast<std::size_t>(total));
    MPI_Allgatherv(L.key.data(), L.n_owned, MPI_UINT64_T, all_keys.data(), coarse.counts.data(),
                   coarse.displs.data(), MPI_UINT64_T, comm);
    std::vector<int> by_key(static_cast<std::size_t>(total));
    std::iota(by_key.begin(), by_key.end(), 0);
    // Key order; equal keys (not expected) fall back to the rank-major order.
    std::stable_sort(by_key.begin(), by_key.end(), [&](int a, int b) {
        return all_keys[static_cast<std::size_t>(a)] < all_keys[static_cast<std::size_t>(b)];
    });
    std::vector<int> pos_of(static_cast<std::size_t>(total));  // rank-major index -> sorted position
    for (int s = 0; s < total; ++s) {
        pos_of[static_cast<std::size_t>(by_key[static_cast<std::size_t>(s)])] = s;
    }
    coarse.order = pos_of;
    // Rows in sorted positions: (row, col) pairs and blocks, gathered on rank 0.
    std::vector<int> ij;
    std::vector<double> vals;
    for (int a = 0; a < L.n_owned; ++a) {
        const int ra = pos_of[static_cast<std::size_t>(coarse.displs[static_cast<std::size_t>(rank)] + a)];
        for (int p = L.row_ptr[static_cast<std::size_t>(a)]; p < L.row_ptr[static_cast<std::size_t>(a) + 1]; ++p) {
            const long long g = L.gid[static_cast<std::size_t>(L.cols[static_cast<std::size_t>(p)])];
            const int owner = gidOwner(g);
            FE_THROW_IF(owner < 0 || owner >= size, FEException, "FsilsAmgHierarchy: invalid coarse column owner");
            const long long rm = coarse.displs[static_cast<std::size_t>(owner)] + gidIndex(g);
            FE_THROW_IF(rm < 0 || rm >= total, FEException, "FsilsAmgHierarchy: invalid coarse column index");
            ij.push_back(ra);
            ij.push_back(pos_of[static_cast<std::size_t>(rm)]);
            const double* blk = L.block(p);
            vals.insert(vals.end(), blk, blk + mm);
        }
    }
    const int my_pairs = static_cast<int>(ij.size() / 2);
    std::vector<int> pair_counts(static_cast<std::size_t>(size), 0);
    MPI_Gather(&my_pairs, 1, MPI_INT, pair_counts.data(), 1, MPI_INT, 0, comm);
    std::vector<int> ij_counts(static_cast<std::size_t>(size)), ij_displs(static_cast<std::size_t>(size) + 1u, 0);
    std::vector<int> v_counts(static_cast<std::size_t>(size)), v_displs(static_cast<std::size_t>(size) + 1u, 0);
    for (int r = 0; r < size; ++r) {
        ij_counts[static_cast<std::size_t>(r)] = 2 * pair_counts[static_cast<std::size_t>(r)];
        v_counts[static_cast<std::size_t>(r)] = pair_counts[static_cast<std::size_t>(r)] * static_cast<int>(mm);
        ij_displs[static_cast<std::size_t>(r) + 1u] = ij_displs[static_cast<std::size_t>(r)] + ij_counts[static_cast<std::size_t>(r)];
        v_displs[static_cast<std::size_t>(r) + 1u] = v_displs[static_cast<std::size_t>(r)] + v_counts[static_cast<std::size_t>(r)];
    }
    std::vector<int> all_ij(rank == 0 ? static_cast<std::size_t>(ij_displs.back()) : 0u);
    std::vector<double> all_vals(rank == 0 ? static_cast<std::size_t>(v_displs.back()) : 0u);
    MPI_Gatherv(ij.data(), 2 * my_pairs, MPI_INT, all_ij.data(), ij_counts.data(), ij_displs.data(), MPI_INT, 0,
                comm);
    MPI_Gatherv(vals.data(), my_pairs * static_cast<int>(mm), MPI_DOUBLE, all_vals.data(), v_counts.data(),
                v_displs.data(), MPI_DOUBLE, 0, comm);
    int ok = 1;
    if (rank == 0) {
        const auto t0 = Clock::now();
        const int ns = total * m;
        // Scalar triplets in (row, col) key order so that the factored matrix
        // does not depend on the partition.
        const std::size_t npairs = all_ij.size() / 2;
        std::vector<std::size_t> order(npairs);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const int ra = all_ij[2 * a], rb = all_ij[2 * b];
            return ra != rb ? ra < rb : all_ij[2 * a + 1] < all_ij[2 * b + 1];
        });
        std::vector<Eigen::Triplet<double, int>> trip;
        trip.reserve(npairs * mm);
        for (const auto e : order) {
            const int R = all_ij[2 * e];
            const int Cc = all_ij[2 * e + 1];
            const double* blk = all_vals.data() + e * mm;
            for (int r = 0; r < m; ++r) {
                for (int c = 0; c < m; ++c) {
                    const double v = blk[r * m + c];
                    if (v != 0.0) {
                        trip.emplace_back(R * m + r, Cc * m + c, v);
                    }
                }
            }
        }
        Coarse::SpMat A(ns, ns);
        A.setFromTriplets(trip.begin(), trip.end());
        A.makeCompressed();
        coarse.lu = std::make_unique<Eigen::SparseLU<Coarse::SpMat, Eigen::COLAMDOrdering<int>>>();
        coarse.lu->analyzePattern(A);
        coarse.lu->factorize(A);
        ok = coarse.lu->info() == Eigen::Success ? 1 : 0;
        stats.coarse_factor_seconds += seconds(t0);
        coarse.rhs.resize(ns);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    coarse.ok = ok != 0;
    FE_THROW_IF(!coarse.ok, FEException, "FsilsAmgHierarchy: coarse factorization failed");
}

void FsilsAmgHierarchy::Impl::coarseSolve(const Level& L, const double* b, double* x) const
{
    const int m = coarse.bs;
    std::vector<int> counts(static_cast<std::size_t>(size)), displs(static_cast<std::size_t>(size));
    for (int r = 0; r < size; ++r) {
        counts[static_cast<std::size_t>(r)] = coarse.counts[static_cast<std::size_t>(r)] * m;
        displs[static_cast<std::size_t>(r)] = coarse.displs[static_cast<std::size_t>(r)] * m;
    }
    const int ns = static_cast<int>(coarse.n_nodes) * m;
    coarse.gathered.resize(rank == 0 ? static_cast<std::size_t>(ns) : 0u);
    MPI_Gatherv(b, L.n_owned * m, MPI_DOUBLE, coarse.gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                comm);
    if (rank == 0) {
        for (int g = 0; g < static_cast<int>(coarse.n_nodes); ++g) {
            const int s = coarse.order[static_cast<std::size_t>(g)];
            for (int c = 0; c < m; ++c) {
                coarse.rhs[s * m + c] = coarse.gathered[static_cast<std::size_t>(g * m + c)];
            }
        }
        const Eigen::VectorXd sol = coarse.lu->solve(coarse.rhs);
        for (int g = 0; g < static_cast<int>(coarse.n_nodes); ++g) {
            const int s = coarse.order[static_cast<std::size_t>(g)];
            for (int c = 0; c < m; ++c) {
                coarse.gathered[static_cast<std::size_t>(g * m + c)] = sol[s * m + c];
            }
        }
    }
    MPI_Scatterv(coarse.gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, x, L.n_owned * m, MPI_DOUBLE, 0,
                 comm);
}

void FsilsAmgHierarchy::Impl::chebyshev(const Level& L, const double* b, double* x, bool zero_guess) const
{
    const std::size_t no = static_cast<std::size_t>(L.n_owned) * static_cast<std::size_t>(L.bs);
    const double lmax = L.lambda;
    const double lmin = lmax / std::max(opt.smoother_ratio, 1.000001);
    const double theta = 0.5 * (lmax + lmin);
    const double delta = 0.5 * (lmax - lmin);
    const double sigma = theta / delta;
    double* r = L.r.data();
    double* z = L.z.data();
    double* d = L.d.data();
    if (zero_guess) {
        std::copy(b, b + no, r);
        std::fill(x, x + no, 0.0);
    } else {
        L.halo.forward(x, L.bs, kTagForward);
        L.spmv(x, r);
        for (std::size_t i = 0; i < no; ++i) {
            r[i] = b[i] - r[i];
        }
    }
    L.jacobi(r, z);
    double rho = 1.0 / sigma;
    for (std::size_t i = 0; i < no; ++i) {
        d[i] = z[i] / theta;
    }
    const int degree = std::max(1, opt.smoother_degree);
    for (int k = 0; k < degree; ++k) {
        for (std::size_t i = 0; i < no; ++i) {
            x[i] += d[i];
        }
        if (k == degree - 1) {
            break;
        }
        L.halo.forward(d, L.bs, kTagForward);
        L.spmv(d, z);
        for (std::size_t i = 0; i < no; ++i) {
            r[i] -= z[i];
        }
        L.jacobi(r, z);
        const double rho_new = 1.0 / (2.0 * sigma - rho);
        const double c1 = rho_new * rho;
        const double c2 = 2.0 * rho_new / delta;
        for (std::size_t i = 0; i < no; ++i) {
            d[i] = c1 * d[i] + c2 * z[i];
        }
        rho = rho_new;
    }
}

void FsilsAmgHierarchy::Impl::vcycle(std::size_t l) const
{
    const Level& L = levels[l];
    if (l + 1 == levels.size()) {
        coarseSolve(L, L.b.data(), L.x.data());
        return;
    }
    const Level& C = levels[l + 1];
    const int m = L.bs;
    const std::size_t mm = L.bb();
    const std::size_t no = static_cast<std::size_t>(L.n_owned) * static_cast<std::size_t>(m);
    chebyshev(L, L.b.data(), L.x.data(), true);
    // Residual r = b - A x (z as scratch).
    L.halo.forward(L.x.data(), m, kTagForward);
    L.spmv(L.x.data(), L.z.data());
    for (std::size_t i = 0; i < no; ++i) {
        L.r[i] = L.b[i] - L.z[i];
    }
    // Restriction b_c = P^T r.
    std::fill(C.b.begin(), C.b.end(), 0.0);
    if (m <= 16) {
        dispatchBlockSize(m, [&](auto tag) { L.restrictImpl<decltype(tag)::value>(L.r.data(), C.b.data()); });
    } else {
        for (int i = 0; i < L.n_owned; ++i) {
            const double* ri = L.r.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int p = L.p_ptr[static_cast<std::size_t>(i)]; p < L.p_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                blockTVecAdd(m, L.p_val.data() + static_cast<std::size_t>(p) * mm, ri,
                             C.b.data() + static_cast<std::size_t>(L.p_col[static_cast<std::size_t>(p)]) * static_cast<std::size_t>(m));
            }
        }
    }
    C.halo.reverseAdd(C.b.data(), m, kTagReverse);
    std::fill(C.b.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(C.n_owned) * static_cast<std::size_t>(m)),
              C.b.end(), 0.0);
    vcycle(l + 1);
    // Prolongation x += P x_c.
    C.halo.forward(C.x.data(), m, kTagForward);
    if (m <= 16) {
        dispatchBlockSize(m, [&](auto tag) { L.prolongImpl<decltype(tag)::value>(C.x.data(), L.x.data()); });
    } else {
        for (int i = 0; i < L.n_owned; ++i) {
            double* xi = L.x.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int p = L.p_ptr[static_cast<std::size_t>(i)]; p < L.p_ptr[static_cast<std::size_t>(i) + 1]; ++p) {
                blockVecAdd(m, L.p_val.data() + static_cast<std::size_t>(p) * mm,
                            C.x.data() + static_cast<std::size_t>(L.p_col[static_cast<std::size_t>(p)]) * static_cast<std::size_t>(m), xi);
            }
        }
    }
    chebyshev(L, L.b.data(), L.x.data(), false);
}

// ---------------------------------------------------------------------------
FsilsAmgHierarchy::FsilsAmgHierarchy() : impl_(std::make_unique<Impl>()) {}
FsilsAmgHierarchy::~FsilsAmgHierarchy() = default;

void FsilsAmgHierarchy::build(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                              int dof,
                              const double* val,
                              std::span<const std::uint64_t> owned_keys,
                              const FsilsAmgOptions& options,
                              bool copy_values)
{
    const auto t0 = Clock::now();
    Impl& I = *impl_;
    if (I.comm == MPI_COMM_NULL) {
        MPI_Comm_dup(lhs.commu.comm, &I.comm);
    }
    MPI_Comm_rank(I.comm, &I.rank);
    MPI_Comm_size(I.comm, &I.size);
    FE_THROW_IF(I.size > (1 << 22), FEException, "FsilsAmgHierarchy: too many ranks for the coarse id encoding");
    I.opt = options;
    I.levels.clear();
    I.stats = Stats{};
    I.finest_aggregates.clear();

    I.buildFinest(lhs, dof, val, owned_keys, copy_values);
    for (std::size_t l = 0;; ++l) {
        Level& L = I.levels[l];
        I.setupSmoother(L);
        const long long nodes = I.sumAll(L.n_owned);
        const long long blocks = I.sumAll(static_cast<long long>(L.cols.size()));
        I.stats.levels.push_back(LevelInfo{nodes, blocks, L.bs, L.lambda});
        const bool last = nodes <= I.opt.coarse_nodes || static_cast<int>(l) + 1 >= I.opt.max_levels;
        if (last) {
            I.setupCoarse(L);
            break;
        }
        const auto ta = Clock::now();
        std::vector<long long> agg;
        std::vector<std::uint64_t> coarse_keys;
        const long long nc = I.aggregate(L, agg, coarse_keys);
        const long long nc_global = I.sumAll(nc);
        std::vector<BlockRow> P;
        I.buildProlongator(L, agg, P);
        I.stats.aggregation_seconds += seconds(ta);
        if (nc_global == 0 || static_cast<double>(nc_global) > 0.85 * static_cast<double>(nodes)) {
            I.setupCoarse(L);
            break;
        }
        const auto tg = Clock::now();
        Level C = I.galerkin(L, agg, P, coarse_keys, nc);
        I.stats.galerkin_seconds += seconds(tg);
        if (l == 0) {
            I.finest_aggregates.assign(static_cast<std::size_t>(L.n_owned), 0);
            for (int i = 0; i < L.n_owned; ++i) {
                const int a = L.agg_local[static_cast<std::size_t>(i)];
                I.finest_aggregates[static_cast<std::size_t>(i)] =
                    a >= 0 ? static_cast<long long>(C.key[static_cast<std::size_t>(a)]) : 0;
            }
        }
        I.levels.push_back(std::move(C));
    }
    for (auto& L : I.levels) {
        L.finalizeStorage();
        L.allocateWork();
    }
    I.stats.setup_seconds = seconds(t0);
}

void FsilsAmgHierarchy::apply(const double* in, double* out) const
{
    const auto t0 = Clock::now();
    const Impl& I = *impl_;
    FE_THROW_IF(I.levels.empty(), FEException, "FsilsAmgHierarchy::apply: not built");
    const Level& L = I.levels.front();
    const std::size_t no = static_cast<std::size_t>(L.n_owned) * static_cast<std::size_t>(L.bs);
    std::copy(in, in + no, L.b.begin());
    I.vcycle(0);
    std::copy(L.x.begin(), L.x.begin() + static_cast<std::ptrdiff_t>(no), out);
    ++impl_->stats.applies;
    impl_->stats.apply_seconds += seconds(t0);
}

bool FsilsAmgHierarchy::empty() const noexcept
{
    return impl_->levels.empty();
}

const FsilsAmgHierarchy::Stats& FsilsAmgHierarchy::stats() const noexcept
{
    return impl_->stats;
}

std::string FsilsAmgHierarchy::summary() const
{
    const auto& s = impl_->stats;
    std::ostringstream oss;
    oss << "levels=" << s.levels.size() << " nodes=";
    for (std::size_t l = 0; l < s.levels.size(); ++l) {
        oss << (l ? "/" : "") << s.levels[l].nodes;
    }
    oss << " blocks=";
    for (std::size_t l = 0; l < s.levels.size(); ++l) {
        oss << (l ? "/" : "") << s.levels[l].blocks;
    }
    oss << " setup_s=" << s.setup_seconds << " aggregation_s=" << s.aggregation_seconds
        << " galerkin_s=" << s.galerkin_seconds << " coarse_factor_s=" << s.coarse_factor_seconds
        << " mis_rounds=" << s.mis_rounds << " regularized_pivots=" << s.regularized_pivots;
    return oss.str();
}

std::vector<long long> FsilsAmgHierarchy::finestAggregatesForTesting() const
{
    return impl_->finest_aggregates;
}

} // namespace backends
} // namespace FE
} // namespace svmp
