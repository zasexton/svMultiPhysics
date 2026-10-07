/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_BLOCK_PRECONDITIONERS_H
#define SVMP_FE_BACKENDS_FSILS_BLOCK_PRECONDITIONERS_H

#include "Backends/FSILS/FsilsAmg.h"
#include "Backends/FSILS/liner_solver/right_precond.h"
#include "Backends/Utils/BackendOptions.h"
#include "Backends/Utils/PreconditionerReusePolicy.h"

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fe_fsi_linear_solver {
class FSILS_lhsType;
}

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Owned-node block graph of an FSILS operator with sorted columns.
 *
 * Rows and columns are the owned nodes of this rank (FSILS internal ordering
 * places them first).  Couplings to ghost nodes are dropped, so in parallel
 * the factorizations built on this graph are block-Jacobi across ranks.
 */
struct FsilsOwnedBlockGraph {
    int n{0};                       ///< Owned nodes.
    std::vector<int> row_ptr{};     ///< n + 1
    std::vector<int> cols{};        ///< Sorted columns per row (< n).
    std::vector<int> diag{};        ///< Position of the diagonal entry of each row.
    std::vector<int> src{};         ///< FSILS nnz slot of each entry.
    std::uint64_t signature{0};     ///< Structure hash of the source operator.

    void build(const fe_fsi_linear_solver::FSILS_lhsType& lhs);
    [[nodiscard]] static std::uint64_t computeSignature(const fe_fsi_linear_solver::FSILS_lhsType& lhs);
    [[nodiscard]] std::size_t nnz() const noexcept { return cols.size(); }
};

/**
 * @brief Block ILU(0) on an owned block graph with dense row-major m x m blocks.
 *
 * L has identity diagonal blocks, U keeps the factored diagonal blocks, whose
 * inverses are stored separately.  A singular diagonal block is regularized by
 * replacing vanishing pivots with machine-precision multiples of the block
 * scale (counted in regularizedPivots()).
 */
class BlockIlu0Factorization {
public:
    /// Factor `blocks` given in graph order (m*m values per entry).
    void factor(const FsilsOwnedBlockGraph& graph, int m, std::vector<double> blocks);

    /// out = (L U)^{-1} in for node-major arrays of graph.n * m values; in-place allowed.
    void solve(const double* in, double* out) const;

    [[nodiscard]] int blockSize() const noexcept { return m_; }
    [[nodiscard]] double factorFlops() const noexcept { return factor_flops_; }
    [[nodiscard]] double applyFlops() const noexcept { return apply_flops_; }
    [[nodiscard]] int regularizedPivots() const noexcept { return regularized_pivots_; }
    [[nodiscard]] bool empty() const noexcept { return graph_ == nullptr; }
    [[nodiscard]] const std::vector<double>& diagonalInverses() const noexcept { return dinv_; }

private:
    const FsilsOwnedBlockGraph* graph_{nullptr};
    int m_{0};
    std::vector<double> lu_{};
    std::vector<double> dinv_{};
    double factor_flops_{0.0};
    double apply_flops_{0.0};
    int regularized_pivots_{0};
};

/// Invert a dense row-major d x d block with partial pivoting (in place on a copy).
/// Returns the number of regularized pivots.  Exposed for unit tests.
int invertDenseBlock(int d, const double* block, double* inverse);

/**
 * @brief Right preconditioner for the FSILS GMRES kernel with optional reuse.
 *
 * Kinds:
 *  - BlockIlu0: block ILU(0) of the scaled monolithic operator (nodal blocks).
 *  - Simple: SIMPLE splitting for a scalar constraint component p
 *    (pressure) and the remaining components u (velocity and any nodal
 *    auxiliary fields):
 *        u* = K~^{-1} r_u,  p = S~^{-1} (r_p - D u*),  u = u* - D_K^{-1} G p,
 *    with K~ the block ILU(0) of K, D_K the nodal diagonal blocks of K,
 *    S = C - D D_K^{-1} G restricted to the operator graph and S~ its ILU(0).
 *    No relaxation or other coefficients.
 *  - Amg: aggregation multigrid V-cycle on the nodal blocks of the scaled
 *    operator (FsilsAmgHierarchy).  Its aggregates come from node keys that do
 *    not depend on the partition (setNodeKeys) and it keeps every coupling
 *    between ranks, so results differ across rank counts only by round-off.
 *
 * Reuse follows PreconditionerReusePolicy.  A reused preconditioner is
 * combined with the current diagonal scalings, so that for the current scaled
 * operator D1 A D2 it approximates D1 A_old D2 rather than the old scaling.
 * Refresh decisions are made collectively (structure changes are OR-reduced
 * and the setup cost uses global operation counts), so every rank refreshes
 * on the same solves.
 */
class FsilsKrylovPreconditioner {
public:
    enum class Kind : std::uint8_t { BlockIlu0, Simple, Amg };

    struct Stats {
        std::uint64_t solves{0};
        std::uint64_t refreshes{0};
        std::uint64_t reuses{0};
        std::uint64_t stale_retries{0};
        double setup_seconds{0.0};
        double last_setup_seconds{0.0};
        int last_iterations{0};
        int regularized_pivots{0};
        PreconditionerReusePolicy::Reason last_reason{PreconditionerReusePolicy::Reason::Initial};
        bool last_fresh{true};
    };

    FsilsKrylovPreconditioner();
    ~FsilsKrylovPreconditioner();
    FsilsKrylovPreconditioner(const FsilsKrylovPreconditioner&) = delete;
    FsilsKrylovPreconditioner& operator=(const FsilsKrylovPreconditioner&) = delete;

    /// Configure kind, reuse and the scalar constraint component (Simple only).
    /// A change of any setting invalidates the current preconditioner.
    void configure(Kind kind, bool reuse, int constraint_component, int krylov_dim);

    /// FSILS hook entry point (see FSILS_rightPreconditionerHook::prepare).
    const fe_fsi_linear_solver::FSILS_rightPreconditioner*
    prepare(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
            int dof,
            const Array<double>& Val,
            const Array<double>* row_scale,
            const Array<double>* col_scale,
            bool force_refresh,
            bool& fresh);

    /// FSILS hook completion (see FSILS_rightPreconditionerHook::finish).
    void finish(int iterations, double residual_reduction, bool converged, bool fresh, bool retried);

    /// Iteration budget of a solve with the reused preconditioner (see
    /// PreconditionerReusePolicy::staleIterationBudget); 0 when none applies.
    [[nodiscard]] int staleIterationCap(double target_reduction) const noexcept;

    /// Drop the current preconditioner (next solve refreshes).
    void invalidate();

    /// Settings of the Amg kind; a change invalidates the current preconditioner.
    void setAmgOptions(const FsilsAmgOptions& options);

    /// Partition-independent keys of the owned nodes (FSILS internal order)
    /// for the Amg kind; empty to use the backend node ids.
    void setNodeKeys(std::vector<std::uint64_t> owned_keys);

    /// The Amg hierarchy, or nullptr for the other kinds (exposed for tests and logs).
    [[nodiscard]] const FsilsAmgHierarchy* amgHierarchy() const noexcept { return amg_.get(); }

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool reuseEnabled() const noexcept { return reuse_; }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const PreconditionerReusePolicy& policy() const noexcept { return policy_; }

    /// Fill the hook for FSILS_lsType::right_pc_hook.
    [[nodiscard]] fe_fsi_linear_solver::FSILS_rightPreconditionerHook makeHook();

    /// Apply the current preconditioner (exposed for tests): out = M^{-1} in.
    void applyForTesting(const Array<double>& in, Array<double>& out) const;

    [[nodiscard]] static std::string kindName(Kind kind);

private:
    class Applier;

    void refresh(const fe_fsi_linear_solver::FSILS_lhsType& lhs,
                 int dof,
                 const Array<double>& Val,
                 const Array<double>* row_scale,
                 const Array<double>* col_scale);
    void updateScalingRatios(int dof,
                             const Array<double>* row_scale,
                             const Array<double>* col_scale);
    void apply(const Array<double>& in, Array<double>& out) const;
    void applyBlockIlu0(const double* in, double* out) const;
    void applySimple(const double* in, double* out) const;
    void refreshAmg(const fe_fsi_linear_solver::FSILS_lhsType& lhs, int dof, const double* val);

    Kind kind_{Kind::BlockIlu0};
    bool reuse_{false};
    int constraint_component_{-1};
    int krylov_dim_{0};
    bool configured_{false};

    int dof_{0};
    int n_tasks_{1};
    MPI_Comm comm_{MPI_COMM_SELF};
    int lhs_nnz_{0};
    int nNo_{0};
    FsilsOwnedBlockGraph graph_{};
    BlockIlu0Factorization factor_{};      // full operator (BlockIlu0) or K block (Simple)
    BlockIlu0Factorization schur_factor_{}; // scalar S (Simple)
    std::vector<double> d_blocks_{};       // Simple: D (1 x m) per entry
    std::vector<double> g_blocks_{};       // Simple: G (m x 1) per entry
    std::vector<double> dk_inv_{};         // Simple: inverse nodal K diagonal blocks (m x m)
    std::unique_ptr<FsilsAmgHierarchy> amg_{};  // Amg
    FsilsAmgOptions amg_options_{};
    std::vector<std::uint64_t> node_keys_{};
    double setup_flops_{0.0};
    double apply_flops_{0.0};

    std::vector<double> row_scale_at_factor_{};
    std::vector<double> col_scale_at_factor_{};
    std::vector<double> ratio_in_{};
    std::vector<double> ratio_out_{};
    bool scaling_identity_{true};

    mutable std::vector<double> work_a_{};
    mutable std::vector<double> work_b_{};
    mutable std::vector<double> work_c_{};

    PreconditionerReusePolicy policy_{};
    PreconditionerReusePolicy::Reason pending_reason_{PreconditionerReusePolicy::Reason::Initial};
    Stats stats_{};
    std::unique_ptr<Applier> applier_;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_BLOCK_PRECONDITIONERS_H
