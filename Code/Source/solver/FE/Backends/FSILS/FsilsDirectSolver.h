/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_DIRECT_SOLVER_H
#define SVMP_FE_BACKENDS_FSILS_DIRECT_SOLVER_H

#include "Backends/Utils/BackendOptions.h"
#include "Core/Types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace svmp {
namespace FE {
namespace backends {

class FsilsMatrix;
class FsilsVector;

/// Counters and timings of the gathered direct solve.  Counters are identical
/// on every rank; timings, sizes and memory are measured on the root rank.
struct FsilsDirectSolveStats {
    std::uint64_t solves{0};
    std::uint64_t orderings{0};         ///< fill-reducing orderings (node pattern changes)
    std::uint64_t analyses{0};          ///< symbolic analyses (stored structure changes)
    std::uint64_t factorizations{0};    ///< numeric factorizations
    std::uint64_t refinement_steps{0};  ///< iterative-refinement corrections
    std::uint64_t failures{0};          ///< singular or non-finite factorizations
    double gather_seconds{0.0};         ///< local packing, gather and scatter
    double assemble_seconds{0.0};       ///< global pattern and value placement on root
    double analyze_seconds{0.0};        ///< ordering + symbolic analysis on root
    double factor_seconds{0.0};         ///< numeric factorization on root
    double solve_seconds{0.0};          ///< triangular solves, residual and refinement on root
    std::int64_t n{0};                  ///< global scalar unknowns
    std::int64_t nnz_matrix{0};         ///< stored entries of the factored matrix
    std::int64_t nnz_factors{0};        ///< nnz(L) + nnz(U) of the last factorization
    double factor_megabytes{0.0};       ///< estimated L + U storage of the last factorization
    double peak_rss_megabytes{0.0};     ///< peak resident set size of the root process
};

/**
 * @brief Sparse direct solve of an FSILS operator by gathering it to one rank.
 *
 * Selected with SolverMethod::Direct on the FSILS backend (`<LS type="Direct">`).
 * Every rank sends its owned rows (in global backend node numbering, columns
 * sorted), its right-hand side and its Dirichlet DOFs to rank 0, which
 * assembles the global matrix, factors it with a sparse LU and returns each
 * rank its owned solution entries; ghost entries are refreshed from their
 * owners.  Only rank 0 factors, so the solve does not depend on how the rows
 * are distributed: the rank count enters only through the round-off of the
 * distributed assembly and through the global node numbering (an
 * owner-contiguous numbering relabels the nodes, which changes ties in the
 * ordering), and results agree across rank counts at round-off level.
 *
 * Dirichlet DOFs get x = 0: their rows and columns are removed and replaced
 * by a unit diagonal, which is the operator the FSILS Krylov path solves
 * through its zero preconditioner weights on Dirichlet faces.
 *
 * Reuse: the fill-reducing ordering (approximate minimum degree on the node
 * graph) is kept while the node pattern is unchanged.  The stored structure
 * holds the diagonal and every entry that has been nonzero since the node
 * pattern or the Dirichlet set last changed (exact zeros of the node blocks
 * are not stored until they become nonzero); its symbolic analysis is kept
 * until that structure changes.  The numeric factorization (threshold
 * partial pivoting with diagonal preference) runs for every solve.
 */
class FsilsGatheredDirectSolver {
public:
    FsilsGatheredDirectSolver();
    ~FsilsGatheredDirectSolver();

    FsilsGatheredDirectSolver(const FsilsGatheredDirectSolver&) = delete;
    FsilsGatheredDirectSolver& operator=(const FsilsGatheredDirectSolver&) = delete;

    /// Collective over the operator's communicator.
    [[nodiscard]] SolverReport solve(const FsilsMatrix& A,
                                     FsilsVector& x,
                                     const FsilsVector& b,
                                     std::span<const GlobalIndex> dirichlet_fe_dofs,
                                     const SolverOptions& options);

    [[nodiscard]] const FsilsDirectSolveStats& stats() const noexcept;

    /// One-line summary of the counters (logged periodically on rank 0).
    [[nodiscard]] std::string summary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_DIRECT_SOLVER_H
