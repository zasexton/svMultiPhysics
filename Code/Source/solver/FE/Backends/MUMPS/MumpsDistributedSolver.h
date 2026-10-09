/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_MUMPS_DISTRIBUTED_SOLVER_H
#define SVMP_FE_BACKENDS_MUMPS_DISTRIBUTED_SOLVER_H

#include "Core/Types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(FE_HAS_MPI) && FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace backends {

/// True when FE was built with FE_ENABLE_MUMPS (an external MUMPS build).
[[nodiscard]] bool mumpsAvailable() noexcept;

/**
 * @brief Sparse direct solve distributed over an MPI communicator (MUMPS).
 *
 * Every rank passes a disjoint share of the matrix entries as triplets in
 * global 0-based indices (duplicates are summed by MUMPS); the analysis
 * (ordering, symbolic factorization, mapping) and the numeric factorization
 * run on all ranks of the communicator.  The analysis is kept while the
 * entry pattern of every rank is unchanged and redone otherwise; the numeric
 * factorization runs on every call of factorize().
 *
 * Symmetric modes take one triangle (entries with row <= col or row >= col,
 * as given; the mirrored entry must not be passed as well).
 *
 * All calls are collective.  Errors are reported on every rank (MUMPS
 * INFOG(1) is global): factorize() and solve() return false and set
 * lastError(); a failure leaves the instance ready for a fresh analysis.
 * Requires FE_HAS_MUMPS; without it every call throws.
 */
class MumpsDistributedSolver {
public:
    enum class Symmetry : int {
        Unsymmetric = 0,
        SymmetricPositiveDefinite = 1,
        GeneralSymmetric = 2,
    };

    enum class Ordering : int {
        Automatic = 7,
        Amd = 0,
        Amf = 2,
        Scotch = 3,
        Pord = 4,
        Metis = 5,
        Qamd = 6,
    };

    struct Statistics {
        std::uint64_t analyses{0};
        std::uint64_t factorizations{0};
        std::uint64_t solves{0};
        double analyze_seconds{0.0};
        double factor_seconds{0.0};
        double solve_seconds{0.0};
        std::int64_t n{0};
        std::int64_t entries{0};             ///< matrix entries over all ranks
        std::int64_t factor_entries{0};      ///< INFOG(29): entries in the factors
        double peak_memory_mb_max_rank{0.0}; ///< INFOG(21): factorization memory, largest rank
        double peak_memory_mb_total{0.0};    ///< INFOG(22): factorization memory, all ranks
        int memory_relaxation_percent{0};    ///< ICNTL(14) in use
    };

#if defined(FE_HAS_MPI) && FE_HAS_MPI
    MumpsDistributedSolver(MPI_Comm comm, Symmetry symmetry, Ordering ordering = Ordering::Metis);
#endif
    ~MumpsDistributedSolver();

    MumpsDistributedSolver(const MumpsDistributedSolver&) = delete;
    MumpsDistributedSolver& operator=(const MumpsDistributedSolver&) = delete;

    /// Collective: analysis (when the pattern changed) and factorization.
    [[nodiscard]] bool factorize(GlobalIndex n,
                                 std::span<const GlobalIndex> rows,
                                 std::span<const GlobalIndex> cols,
                                 std::span<const Real> values);

    /// Collective: solves A x = b with the last factorization.  `rhs` holds b
    /// on rank 0 (ignored elsewhere); on return `solution` holds x on every rank.
    [[nodiscard]] bool solveReplicated(std::span<const Real> rhs, std::vector<Real>& solution);

    [[nodiscard]] const Statistics& statistics() const noexcept;
    [[nodiscard]] const std::string& lastError() const noexcept;
    /// Approximate memory of the factors held by this rank (bytes), from INFO(16).
    [[nodiscard]] std::size_t localFactorBytes() const noexcept;
    [[nodiscard]] bool factorized() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_MUMPS_DISTRIBUTED_SOLVER_H
