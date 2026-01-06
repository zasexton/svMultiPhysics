/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Core/FEException.h"
#include "Core/Types.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <memory>
#include <vector>

namespace svmp::FE::timestepping::test {

namespace {

struct KernelCallCounts {
    int total{0};
    int matrix_only{0};
    int vector_only{0};
    int matrix_and_vector{0};
};

class CountingKernel final : public assembly::AssemblyKernel {
public:
    CountingKernel(std::shared_ptr<assembly::AssemblyKernel> inner, KernelCallCounts* counts)
        : inner_(std::move(inner))
        , counts_(counts)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return inner_->getRequiredData();
    }

    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override
    {
        return inner_->materialStateSpec();
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return inner_->parameterSpecs();
    }

    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override
    {
        return inner_->maxTemporalDerivativeOrder();
    }

    [[nodiscard]] bool hasCell() const noexcept override { return inner_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return inner_->hasBoundaryFace(); }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return inner_->hasInteriorFace(); }

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override
    {
        inner_->computeCell(ctx, output);
        counts_->total += 1;
        if (output.has_matrix && output.has_vector) {
            counts_->matrix_and_vector += 1;
        } else if (output.has_matrix) {
            counts_->matrix_only += 1;
        } else if (output.has_vector) {
            counts_->vector_only += 1;
        }
    }

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override
    {
        inner_->computeBoundaryFace(ctx, boundary_marker, output);
    }

    void computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                             const assembly::AssemblyContext& ctx_plus,
                             assembly::KernelOutput& output_minus,
                             assembly::KernelOutput& output_plus,
                             assembly::KernelOutput& coupling_minus_plus,
                             assembly::KernelOutput& coupling_plus_minus) override
    {
        inner_->computeInteriorFace(ctx_minus, ctx_plus, output_minus, output_plus, coupling_minus_plus, coupling_plus_minus);
    }

    [[nodiscard]] std::string name() const override
    {
        return "Counting(" + inner_->name() + ")";
    }

private:
    std::shared_ptr<assembly::AssemblyKernel> inner_{};
    KernelCallCounts* counts_{nullptr};
};

KernelCallCounts runLinearReaction(SchemeKind scheme, int steps)
{
    KernelCallCounts counts;
    
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(1.0)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto inner_kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    auto kernel = std::make_shared<CountingKernel>(inner_kernel, &counts);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>(); // Placeholder
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) return counts; // Skip if no backend
    auto linear = factory->createLinearSolver(directSolve());

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();
    history.setDt(0.1);
    history.setPrevDt(0.1);

    TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.1 * steps;
    opts.dt = 0.1;
    opts.scheme = scheme;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 10;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0; // Force convergence based on abs tolerance (machine precision essentially for linear)

    TimeLoop loop(opts);
    loop.run(transient, *factory, *linear, history);
    
    return counts;
}

} // namespace

TEST(TimeLoopPerformance, TRBDF2_CostsTwiceAsMuchAsBackwardEuler)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    const int steps = 5;
    auto counts_be = runLinearReaction(SchemeKind::BackwardEuler, steps);
    auto counts_tr = runLinearReaction(SchemeKind::TRBDF2, steps);
    
    // Backward Euler: 1 solve per step.
    // For linear problem: 
    //   It 0: Residual eval.
    //   It 1: Jacobian eval + Residual eval (if separate assembly).
    //   Check: Residual eval.
    // Minimally we expect 1 Jacobian per step.
    
    EXPECT_GT(counts_be.matrix_only + counts_be.matrix_and_vector, steps);
    
    // TRBDF2: 2 solves per step.
    // Expect roughly 2x Jacobian evaluations.
    
    int jac_be = counts_be.matrix_only + counts_be.matrix_and_vector;
    int jac_tr = counts_tr.matrix_only + counts_tr.matrix_and_vector;
    
    // TRBDF2 performs two implicit solves per step; depending on how residual
    // additions and Jacobian reuse are implemented, we expect between 2x and 3x
    // the Jacobian assembly count of Backward Euler.
    EXPECT_GE(jac_tr, 2 * jac_be);
    EXPECT_LE(jac_tr, 3 * jac_be);
}

TEST(TimeLoopPerformance, BDF2_CostsSameAsBackwardEuler)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    const int steps = 5;
    auto counts_be = runLinearReaction(SchemeKind::BackwardEuler, steps);
    auto counts_bdf2 = runLinearReaction(SchemeKind::BDF2, steps);
    
    int jac_be = counts_be.matrix_only + counts_be.matrix_and_vector;
    int jac_bdf2 = counts_bdf2.matrix_only + counts_bdf2.matrix_and_vector;
    
    // BDF2 might take an extra step for startup (Theta method), but generally 1 solve per step.
    EXPECT_NEAR(static_cast<double>(jac_bdf2), static_cast<double>(jac_be), static_cast<double>(jac_be) * 0.2);
}

} // namespace svmp::FE::timestepping::test
