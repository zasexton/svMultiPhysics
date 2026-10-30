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

#include <cstdint>
#include <memory>
#include <vector>

namespace fe_fsi_linear_solver {
class FSILS_lhsType;
}

namespace svmp {
namespace FE {
namespace backends {

/// Invert a dense row-major d x d block with partial pivoting (in place on a copy).
/// Returns the number of regularized pivots.  Exposed for unit tests.
int invertDenseBlock(int d, const double* block, double* inverse);

/**
 * @brief Aggregation-multigrid right preconditioner for the FSILS GMRES kernel.
 *
 * One V-cycle of FsilsAmgHierarchy on the nodal blocks of the scaled
 * operator, rebuilt for every solve.  Its aggregates come from node keys that
 * do not depend on the partition (setNodeKeys) and it keeps every coupling
 * between ranks, so results differ across rank counts only by round-off.
 */
class FsilsKrylovPreconditioner {
public:
    struct Stats {
        std::uint64_t solves{0};
        double setup_seconds{0.0};
        double last_setup_seconds{0.0};
        int last_iterations{0};
        int regularized_pivots{0};
    };

    FsilsKrylovPreconditioner();
    ~FsilsKrylovPreconditioner();
    FsilsKrylovPreconditioner(const FsilsKrylovPreconditioner&) = delete;
    FsilsKrylovPreconditioner& operator=(const FsilsKrylovPreconditioner&) = delete;

    /// FSILS hook entry point (see FSILS_rightPreconditionerHook::prepare).
    const fe_fsi_linear_solver::FSILS_rightPreconditioner*
    prepare(const fe_fsi_linear_solver::FSILS_lhsType& lhs, int dof, const Array<double>& Val);

    /// FSILS hook completion (see FSILS_rightPreconditionerHook::finish).
    void finish(int iterations, double residual_reduction, bool converged);

    /// Settings of the hierarchy.
    void setAmgOptions(const FsilsAmgOptions& options);

    /// Partition-independent keys of the owned nodes (FSILS internal order);
    /// empty to use the backend node ids.
    void setNodeKeys(std::vector<std::uint64_t> owned_keys);

    /// The hierarchy of the last solve (exposed for tests and logs).
    [[nodiscard]] const FsilsAmgHierarchy* amgHierarchy() const noexcept { return amg_.get(); }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    /// Fill the hook for FSILS_lsType::right_pc_hook.
    [[nodiscard]] fe_fsi_linear_solver::FSILS_rightPreconditionerHook makeHook();

    /// Apply the current preconditioner (exposed for tests): out = M^{-1} in.
    void applyForTesting(const Array<double>& in, Array<double>& out) const;

private:
    class Applier;

    void apply(const Array<double>& in, Array<double>& out) const;

    int dof_{0};
    int n_owned_{0};
    std::unique_ptr<FsilsAmgHierarchy> amg_{};
    FsilsAmgOptions amg_options_{};
    std::vector<std::uint64_t> node_keys_{};
    mutable std::vector<double> work_{};
    Stats stats_{};
    std::unique_ptr<Applier> applier_;
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_BLOCK_PRECONDITIONERS_H
