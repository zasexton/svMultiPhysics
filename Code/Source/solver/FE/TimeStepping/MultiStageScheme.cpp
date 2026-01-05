/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/MultiStageScheme.h"

#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

namespace svmp {
namespace FE {
namespace timestepping {

WeightedIntegrator::WeightedIntegrator(std::shared_ptr<const systems::TimeIntegrator> base,
                                       Real time_derivative_weight,
                                       Real non_time_derivative_weight)
    : base_(std::move(base))
    , time_derivative_weight_(time_derivative_weight)
    , non_time_derivative_weight_(non_time_derivative_weight)
{
    FE_CHECK_NOT_NULL(base_.get(), "WeightedIntegrator::base");
}

std::string WeightedIntegrator::name() const
{
    return base_->name();
}

int WeightedIntegrator::maxSupportedDerivativeOrder() const noexcept
{
    return base_->maxSupportedDerivativeOrder();
}

assembly::TimeIntegrationContext
WeightedIntegrator::buildContext(int max_time_derivative_order, const systems::SystemStateView& state) const
{
    auto ctx = base_->buildContext(max_time_derivative_order, state);
    ctx.time_derivative_term_weight = time_derivative_weight_;
    ctx.non_time_derivative_term_weight = non_time_derivative_weight_;
    return ctx;
}

MultiStageSolver::MultiStageSolver(const NewtonSolver& newton)
    : newton_(newton)
{
}

std::shared_ptr<const systems::TimeIntegrator>
MultiStageSolver::wrapIntegrator(std::shared_ptr<const systems::TimeIntegrator> base, const StageWeights& weights) const
{
    FE_CHECK_NOT_NULL(base.get(), "MultiStageSolver::wrapIntegrator base");
    if (weights.time_derivative == static_cast<Real>(1.0) &&
        weights.non_time_derivative == static_cast<Real>(1.0)) {
        return base;
    }
    return std::make_shared<const WeightedIntegrator>(std::move(base), weights.time_derivative, weights.non_time_derivative);
}

NewtonReport MultiStageSolver::solveImplicitStage(systems::FESystem& system,
                                                 backends::LinearSolver& linear,
                                                 TimeHistory& history,
                                                 NewtonWorkspace& workspace,
                                                 const ImplicitStageSpec& stage,
                                                 backends::GenericVector* scratch_residual_addition) const
{
    FE_CHECK_NOT_NULL(stage.integrator.get(), "MultiStageSolver stage.integrator");

    auto current_integrator = wrapIntegrator(stage.integrator, stage.weights);
    systems::TransientSystem transient_current(system, current_integrator);

    const backends::GenericVector* addition = nullptr;
    if (stage.residual_addition.has_value()) {
        FE_THROW_IF(scratch_residual_addition == nullptr, systems::InvalidStateException,
                    "MultiStageSolver: residual addition requires scratch vector");

        scratch_residual_addition->zero();

        const auto& add = *stage.residual_addition;
        FE_CHECK_NOT_NULL(add.integrator.get(), "MultiStageSolver residual_addition.integrator");
        auto add_integrator = wrapIntegrator(add.integrator, add.weights);
        systems::TransientSystem transient_add(system, add_integrator);

        systems::AssemblyRequest req;
        req.op = newton_.options().residual_op;
        req.want_vector = true;

        transient_add.system().beginTimeStep();
        auto view = scratch_residual_addition->createAssemblyView();
        FE_CHECK_NOT_NULL(view.get(), "MultiStageSolver: residual addition assembly view");
        (void)transient_add.assemble(req, add.state, nullptr, view.get());
        addition = scratch_residual_addition;
    }

    return newton_.solveStep(transient_current, linear, stage.solve_time, history, workspace, addition);
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
