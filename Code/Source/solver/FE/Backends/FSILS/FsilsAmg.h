/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_AMG_H
#define SVMP_FE_BACKENDS_FSILS_AMG_H

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fe_fsi_linear_solver {
class FSILS_lhsType;
}

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Settings of the partition-independent aggregation multigrid.
 */
struct FsilsAmgOptions {
    int max_levels{10};             ///< Levels including the finest.
    long long coarse_nodes{600};    ///< Stop coarsening at or below this global node count.
    int smoother_degree{3};         ///< Chebyshev degree of each pre- and post-smoothing.
    double smoother_ratio{30.0};    ///< lambda_max / lambda_min of the Chebyshev interval.
    bool smooth_prolongator{false}; ///< Smoothed (true) or plain (false) aggregation.
    double prolongator_omega{4.0 / 3.0};  ///< Prolongator smoothing weight times 1 / lambda_max.
    /// Power iterations for lambda_max of D^{-1} A (start vector from the node
    /// keys, estimate times 1.1, capped by ||D^{-1} A||_inf); 0 uses the bound.
    int lambda_iterations{0};
    /// Aggregation couples nodes i and j only when ||A_ij||_F > theta *
    /// sqrt(||A_ii||_F ||A_jj||_F) (Frobenius norms of the nodal blocks of
    /// row i); 0 keeps every nonzero block.
    double strength_threshold{0.0};
};

/**
 * @brief Aggregation multigrid V-cycle on the nodal blocks of an FSILS operator
 *        whose aggregates and smoothers do not depend on the partition.
 *
 * Every level is a block matrix with complete owned rows and ghost columns,
 * like the FSILS owned-row operator it starts from.  The construction uses only
 * quantities that are independent of how the nodes are split across ranks:
 *
 *  - Aggregates come from a distance-2 maximal independent set whose
 *    priorities are hashes of caller-supplied node keys (for example
 *    DofPermutation::node_key, derived from mesh vertex coordinates).  The set is computed in synchronous rounds
 *    (each node compares its own priority with the largest one in its
 *    distance-2 neighbourhood), so it is a function of the operator graph and
 *    the keys only.  Roots take their distance-1 neighbours; remaining nodes
 *    join the neighbouring aggregate of highest priority.
 *  - The smoother is Chebyshev iteration on D^{-1} A with D the nodal diagonal
 *    blocks; its upper bound is ||D^{-1} A||_inf (an exact maximum reduction).
 *  - The prolongator is the tentative one (one identity block per node,
 *    unknowns without off-diagonal couplings removed), optionally smoothed by
 *    one damped block-Jacobi step; coarse operators are Galerkin products.
 *  - The coarsest operator is gathered in key order and factored by sparse LU.
 *
 * Results therefore differ across rank counts only by the summation order of
 * reductions and of the Galerkin sums (round-off).
 *
 * All member functions except the accessors are collective over the
 * communicator of the operator passed to build().
 */
class FsilsAmgHierarchy {
public:
    struct LevelInfo {
        long long nodes{0};       ///< Global nodes.
        long long blocks{0};      ///< Global stored blocks.
        int block_size{0};
        double lambda_max{0.0};
    };

    struct Stats {
        std::vector<LevelInfo> levels{};
        double setup_seconds{0.0};
        double aggregation_seconds{0.0};
        double galerkin_seconds{0.0};
        double coarse_factor_seconds{0.0};
        int mis_rounds{0};
        int regularized_pivots{0};
        std::uint64_t applies{0};
        double apply_seconds{0.0};
    };

    FsilsAmgHierarchy();
    ~FsilsAmgHierarchy();
    FsilsAmgHierarchy(const FsilsAmgHierarchy&) = delete;
    FsilsAmgHierarchy& operator=(const FsilsAmgHierarchy&) = delete;

    /**
     * @brief Build the hierarchy for the owned rows of `lhs` with values `val`
     *        (dof*dof row-major blocks in FSILS storage order).
     *
     * `owned_keys` holds one key per owned node (lhs.mynNo entries, FSILS
     * internal order); equal nodes must have equal keys on every partition and
     * distinct nodes distinct keys.  When it is empty, the backend global node
     * ids are used, which makes the aggregates partition dependent.
     *
     * When `copy_values` is false the finest level refers to `val`, which must
     * stay valid and unchanged until the next build() (one GMRES solve).
     */
    void build(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
               int dof,
               const double* val,
               std::span<const std::uint64_t> owned_keys,
               const FsilsAmgOptions& options,
               bool copy_values);

    /// out = M^{-1} in on owned entries (node-major, dof per node); out may alias in.
    void apply(const double* in, double* out) const;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const Stats& stats() const noexcept;
    [[nodiscard]] std::string summary() const;

    /// Key of the root of the aggregate of each owned node of the finest
    /// level, 0 for nodes without aggregate.  Exposed for tests.
    [[nodiscard]] std::vector<long long> finestAggregatesForTesting() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// 64-bit mix of a key (splitmix64 finalizer); exposed for tests.
[[nodiscard]] std::uint64_t amgHashKey(std::uint64_t key) noexcept;

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_AMG_H
