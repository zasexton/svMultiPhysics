/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_LINEAR_SOLVER_H
#define SVMP_FE_BACKENDS_FSILS_LINEAR_SOLVER_H

#include "Backends/Interfaces/LinearSolver.h"

namespace svmp {
namespace FE {
namespace backends {

class FsilsLinearSolver final : public LinearSolver {
public:
    explicit FsilsLinearSolver(const SolverOptions& options);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::FSILS; }

    void setOptions(const SolverOptions& options) override;
    [[nodiscard]] const SolverOptions& getOptions() const noexcept override { return options_; }

    [[nodiscard]] SolverReport solve(const GenericMatrix& A,
                                     GenericVector& x,
                                     const GenericVector& b) override;

    void setRankOneUpdates(std::span<const RankOneUpdate> updates) override;
    void setDirichletDofs(std::span<const GlobalIndex> dofs) override;
    [[nodiscard]] bool supportsNativeRankOneUpdates() const noexcept override { return true; }
    void setEffectiveTimeStep(double dt_eff) override;

private:
    SolverOptions options_{};
    std::vector<RankOneUpdate> rank_one_updates_{};
    std::vector<GlobalIndex> dirichlet_dofs_{};
    double dt_eff_{1.0};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_LINEAR_SOLVER_H
