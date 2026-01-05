/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_MULTI_STAGE_SCHEME_H
#define SVMP_FE_TIMESTEPPING_MULTI_STAGE_SCHEME_H

#include "Backends/Interfaces/GenericVector.h"
#include "Core/Types.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include <memory>
#include <optional>

namespace svmp {
namespace FE {
namespace timestepping {

struct StageWeights {
    Real time_derivative{1.0};
    Real non_time_derivative{1.0};
};

struct ResidualAdditionSpec {
    std::shared_ptr<const systems::TimeIntegrator> integrator{};
    StageWeights weights{};
    systems::SystemStateView state{};
};

struct ImplicitStageSpec {
    std::shared_ptr<const systems::TimeIntegrator> integrator{};
    StageWeights weights{};
    double solve_time{0.0};
    std::optional<ResidualAdditionSpec> residual_addition{};
};

/**
 * @brief Wrapper around an existing Systems time integrator to scale dt-containing vs dt-free terms.
 *
 * This supports operator-splitting style schemes (e.g. θ-method, trapezoidal stages)
 * without requiring multiple kernel compilations.
 */
class WeightedIntegrator final : public systems::TimeIntegrator {
public:
    WeightedIntegrator(std::shared_ptr<const systems::TimeIntegrator> base,
                       Real time_derivative_weight,
                       Real non_time_derivative_weight);

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] int maxSupportedDerivativeOrder() const noexcept override;

    [[nodiscard]] assembly::TimeIntegrationContext
    buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const override;

private:
    std::shared_ptr<const systems::TimeIntegrator> base_{};
    Real time_derivative_weight_{1.0};
    Real non_time_derivative_weight_{1.0};
};

/**
 * @brief Helper that solves a sequence of implicit stages using Newton + Systems transient assembly.
 */
class MultiStageSolver {
public:
    explicit MultiStageSolver(const NewtonSolver& newton);

    [[nodiscard]] NewtonReport solveImplicitStage(systems::FESystem& system,
                                                 backends::LinearSolver& linear,
                                                 TimeHistory& history,
                                                 NewtonWorkspace& workspace,
                                                 const ImplicitStageSpec& stage,
                                                 backends::GenericVector* scratch_residual_addition) const;

private:
    [[nodiscard]] std::shared_ptr<const systems::TimeIntegrator>
    wrapIntegrator(std::shared_ptr<const systems::TimeIntegrator> base, const StageWeights& weights) const;

    const NewtonSolver& newton_;
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_MULTI_STAGE_SCHEME_H
