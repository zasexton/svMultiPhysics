/**
 * @file test_MixedFormPerformance.cpp
 * @brief Phase 5 performance verification: mixed vs manual compilation and assembly
 *
 * Verifies that:
 * 1. Mixed-compiled forms produce identical assembly results as manual blocks
 * 2. installMixedFormIR creates MonolithicCellKernel for mixed cell execution when JIT is enabled
 * 3. JIT monolithic assembly and exact per-block fallback produce identical matrix/vector results
 * 4. installFormulation (residual path) uses the explicit mixed kernel plan end-to-end
 *
 * Performance micro-benchmarks are DISABLED by default (prefix with DISABLED_).
 * Enable with --gtest_also_run_disabled_tests.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FormsInstallerDetail.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Dofs/DofMap.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/TimeSteppingUtils.h"

#include "Forms/BlockForm.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/MixedFormIR.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/Vocabulary.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/SpaceFactory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <chrono>
#include <iostream>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

using Clock = std::chrono::steady_clock;

template <class Fn>
double timeSeconds(Fn&& fn) {
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

const svmp::FE::forms::SymbolicNonlinearFormKernel* unwrapSymbolicKernel(
    const std::shared_ptr<svmp::FE::assembly::AssemblyKernel>& kernel)
{
    if (!kernel) {
        return nullptr;
    }
    if (const auto* jit = dynamic_cast<const svmp::FE::forms::jit::JITKernelWrapper*>(kernel.get())) {
        return dynamic_cast<const svmp::FE::forms::SymbolicNonlinearFormKernel*>(&jit->fallbackKernel());
    }
    return dynamic_cast<const svmp::FE::forms::SymbolicNonlinearFormKernel*>(kernel.get());
}

struct AssemblySnapshot {
    std::vector<Real> matrix;
    std::vector<Real> vector;
    bool monolithic{false};
};

svmp::FE::forms::FormExpr makeTransientNavierStokesResidual(
    const svmp::FE::spaces::FunctionSpace& velocity_space,
    const svmp::FE::spaces::FunctionSpace& pressure_space,
    FieldId velocity_field,
    FieldId pressure_field,
    int dim,
    bool enable_convection,
    bool enable_vms)
{
    using namespace svmp::FE::forms;

    const auto u = FormExpr::stateField(velocity_field, velocity_space, "u");
    const auto p = FormExpr::stateField(pressure_field, pressure_space, "p");
    const auto v = FormExpr::testFunction(velocity_space, "v");
    const auto q = FormExpr::testFunction(pressure_space, "q");

    const auto rho = FormExpr::constant(1.0);
    const auto mu = FormExpr::constant(0.01);
    const auto eps = FormExpr::constant(1.0e-12);
    const auto ct_m = FormExpr::constant(1.0);
    const auto ct_c = FormExpr::constant(36.0);

    std::vector<FormExpr> zero_components;
    zero_components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        zero_components.push_back(FormExpr::constant(0.0));
    }
    const auto f = FormExpr::asVector(zero_components);

    FormExpr a = f;
    if (enable_convection) {
        a = u;
    }

    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    const auto inertia = rho * inner(dt(u), v);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    auto momentum = (inertia + convection + viscous + pressure + forcing).dx();
    auto continuity = (q * div(u)).dx();

    if (!enable_vms) {
        return momentum + continuity;
    }

    const auto dt_step = FormExpr::effectiveTimeStep();
    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;

    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
    const auto kU = inner(a, K * a);
    const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
    const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

    const auto u_sub = -tau_m * r_m;
    const auto p_sub = -tau_c * div(u);

    const auto u_adv = enable_convection ? (u + u_sub) : a;
    const auto p_adv = p + p_sub;
    const auto convection_adv = rho * inner(grad(u) * u_adv, v);
    const auto pressure_adv = -p_adv * div(v);
    const auto supg = -rho * inner(grad(v) * u_adv, u_sub);

    FormExpr cross_stress = FormExpr::constant(0.0);
    if (enable_convection) {
        const auto tau_b = rho / sqrt(inner(u_sub, K * u_sub) + eps);
        const auto rV_tau = tau_b * (grad(u) * u_sub);
        cross_stress = inner(grad(v) * u_sub, rV_tau);
    }

    momentum = (inertia + convection_adv + viscous + pressure_adv + forcing + supg + cross_stress).dx();
    continuity = (q * div(u) - inner(grad(q), u_sub)).dx();
    return momentum + continuity;
}

template <class BuildResidual>
auto assembleTransientMixedResidual(BuildResidual&& build_residual, bool enable_jit)
{
    constexpr int dim = 3;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto u_space = svmp::FE::spaces::VectorSpace(
        svmp::FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/dim);
    auto p_space = svmp::FE::spaces::Space(
        svmp::FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField({.name = "u", .space = u_space, .components = dim});
    const auto p_field = sys.addField({.name = "p", .space = p_space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = enable_jit;
    opts.compiler_options.use_symbolic_tangent = true;

    const auto installed = svmp::FE::systems::installFormulation(
        sys,
        "op",
        {u_field, p_field},
        build_residual(*u_space, *p_space, u_field, p_field),
        opts);
    EXPECT_NE(installed.mixed_plan, nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> u(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> u_prev(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        u[static_cast<std::size_t>(i)] = static_cast<Real>(0.02) * static_cast<Real>(i + 1);
        u_prev[static_cast<std::size_t>(i)] = static_cast<Real>(-0.01) * static_cast<Real>(i + 2);
    }

    svmp::FE::systems::SystemStateView state;
    state.time = 0.1;
    state.dt = 0.05;
    state.u = u;
    state.u_prev = u_prev;

    svmp::FE::systems::TransientSystem transient(
        sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    const auto ar = transient.assemble(req, state, &out, &out);
    EXPECT_TRUE(ar.success);

    return AssemblySnapshot{
        .matrix = std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
        .vector = std::vector<Real>(out.vectorData().begin(), out.vectorData().end()),
        .monolithic = installed.mixed_plan && installed.mixed_plan->usesMonolithicCellKernel(),
    };
}

template <class BuildResidual>
auto assembleTransientMixedResidualGeneralizedAlpha(BuildResidual&& build_residual, bool enable_jit)
{
    constexpr int dim = 3;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto u_space = svmp::FE::spaces::VectorSpace(
        svmp::FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/dim);
    auto p_space = svmp::FE::spaces::Space(
        svmp::FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField({.name = "u", .space = u_space, .components = dim});
    const auto p_field = sys.addField({.name = "p", .space = p_space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = enable_jit;
    opts.compiler_options.use_symbolic_tangent = true;

    const auto installed = svmp::FE::systems::installFormulation(
        sys,
        "op",
        {u_field, p_field},
        build_residual(*u_space, *p_space, u_field, p_field),
        opts);
    EXPECT_NE(installed.mixed_plan, nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> u_stage(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> u_prev(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> u_prev2(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        u_stage[static_cast<std::size_t>(i)] = static_cast<Real>(0.03) * static_cast<Real>(i + 1);
        u_prev[static_cast<std::size_t>(i)] = static_cast<Real>(-0.02) * static_cast<Real>(i + 2);
        u_prev2[static_cast<std::size_t>(i)] = static_cast<Real>(0.015) * static_cast<Real>(i + 3);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    svmp::FE::systems::SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    svmp::FE::systems::TransientSystem transient(sys, integrator);

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    const auto ar = transient.assemble(req, state, &out, &out);
    EXPECT_TRUE(ar.success);

    return AssemblySnapshot{
        .matrix = std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
        .vector = std::vector<Real>(out.vectorData().begin(), out.vectorData().end()),
        .monolithic = installed.mixed_plan && installed.mixed_plan->usesMonolithicCellKernel(),
    };
}

} // namespace

// ============================================================================
// Functional: MonolithicCellKernel created by installMixedFormIR when JIT enabled
// ============================================================================

TEST(MixedFormPerformance, InstallMixedFormIR_CreatesMonolithicCellKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    auto mixed = (u * v).dx() + (p * v).dx() + (u * q).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto mir = compiler.compileMixed(mixed, svmp::FE::forms::FormKind::Bilinear);
    ASSERT_EQ(mir.numActiveBlocks(), 3u);

    // Install with JIT enabled
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;

    const std::array fields = {u_f, p_f};
    svmp::FE::systems::installMixedFormIR(
        sys, "op",
        std::span<const FieldId>(fields),
        std::span<const FieldId>(fields),
        mir, opts);

    const auto& def = sys.operatorDefinition("op");
    ASSERT_EQ(def.cells.size(), 1u);
    ASSERT_NE(def.cells[0].kernel, nullptr);
    EXPECT_EQ(
        def.cells[0].kernel->semanticKernelKind(),
        svmp::FE::assembly::SemanticKernelKind::MonolithicCell);

    const auto* monolithic =
        dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(def.cells[0].kernel.get());
    ASSERT_NE(monolithic, nullptr);
    EXPECT_EQ(monolithic->numBlocks(), 3u);
}

// ============================================================================
// Functional: installFormulation residual path gets fused execution
// ============================================================================

TEST(MixedFormPerformance, InstallFormulation_MixedResidual_AssemblesCorrectly)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");

    // Mixed residual: momentum + continuity
    auto residual = (u_state * v + p_state * v).dx() + (u_state * q).dx();

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    opts.compiler_options.use_symbolic_tangent = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual, opts);

    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());
    ASSERT_EQ(installed.mixed_plan->blocks.size(), 3u);

    std::size_t matrix_blocks = 0;
    std::size_t vector_blocks = 0;
    for (const auto& block : installed.mixed_plan->blocks) {
        if (block.want_matrix) {
            ++matrix_blocks;
        }
        if (block.want_vector) {
            ++vector_blocks;
        }
    }
    EXPECT_EQ(matrix_blocks, 3u);
    EXPECT_EQ(vector_blocks, 2u);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto& def = sys.operatorDefinition("op");
    ASSERT_EQ(def.cells.size(), 1u);
    EXPECT_NE(
        dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(def.cells[0].kernel.get()),
        nullptr);

    const auto n = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.1 * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    // Assemble both matrix and vector
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);

    // Basic sanity: residual vector should be non-zero
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0) << "Residual vector should be non-zero";

    // Jacobian should be non-zero
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0) << "Jacobian matrix should be non-zero";

    // Verify finite differences: dR/du ≈ (R(u+eps*e_j) - R(u)) / eps
    const Real eps = 1e-7;
    std::vector<Real> R0(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 1e-5)
                << "FD mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(MixedFormPerformance, InstallFormulation_VectorScalarCrossFieldComponentBlockMatchesFD)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto velocity_space = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = velocity_space, .components = 2});
    const auto p_f = sys.addField({.name = "p", .space = scalar_space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *velocity_space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *scalar_space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *velocity_space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *scalar_space, "q");

    const auto momentum = (u_state.component(1) * p_state * v.component(0)).dx();
    const auto continuity = (q * div(u_state)).dx();
    const auto residual = momentum + continuity;

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.use_symbolic_tangent = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual, opts);

    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_FALSE(installed.mixed_plan->blocks.empty());

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.05 * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    EXPECT_NO_THROW((void)sys.assemble(req, state, &out, &out));

    std::vector<Real> R0(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n);
        Rp.zero();
        EXPECT_NO_THROW((void)sys.assemble(req_vec, state_plus, nullptr, &Rp));

        for (GlobalIndex i = 0; i < n; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 2e-5)
                << "FD mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(MixedFormPerformance, InstallFormulation_MonolithicJITParity_VersusPerBlockFallback)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    struct AssemblySnapshot {
        std::vector<Real> matrix;
        std::vector<Real> vector;
        bool monolithic{false};
    };

    auto assemble = [&](bool enable_jit) {
        svmp::FE::systems::FESystem sys(mesh);
        const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
        const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
        sys.addOperator("op");

        auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
        auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
        auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
        auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");
        auto residual = (u_state * v + p_state * v).dx() + (u_state * q).dx();

        svmp::FE::systems::FormInstallOptions opts;
        opts.compiler_options.jit.enable = enable_jit;
        const auto installed =
            svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual, opts);
        EXPECT_NE(installed.mixed_plan, nullptr);

        svmp::FE::systems::SetupInputs inputs;
        inputs.topology_override = singleTetraTopology();
        sys.setup({}, inputs);

        const auto n = sys.dofHandler().getNumDofs();
        std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
        for (GlobalIndex i = 0; i < n; ++i) {
            U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
        }

        svmp::FE::systems::SystemStateView state;
        state.u = U;

        svmp::FE::systems::AssemblyRequest req;
        req.op = "op";
        req.want_matrix = true;
        req.want_vector = true;

        svmp::FE::assembly::DenseSystemView out(n);
        out.zero();
        (void)sys.assemble(req, state, &out, &out);

        return AssemblySnapshot{
            .matrix = std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
            .vector = std::vector<Real>(out.vectorData().begin(), out.vectorData().end()),
            .monolithic = installed.mixed_plan && installed.mixed_plan->usesMonolithicCellKernel(),
        };
    };

    const auto jit = assemble(true);
    const auto fallback = assemble(false);

    EXPECT_TRUE(jit.monolithic);
    EXPECT_FALSE(fallback.monolithic);
    ASSERT_EQ(jit.matrix.size(), fallback.matrix.size());
    ASSERT_EQ(jit.vector.size(), fallback.vector.size());

    for (std::size_t i = 0; i < jit.matrix.size(); ++i) {
        EXPECT_NEAR(jit.matrix[i], fallback.matrix[i], 1e-12) << "matrix[" << i << "]";
    }
    for (std::size_t i = 0; i < jit.vector.size(); ++i) {
        EXPECT_NEAR(jit.vector[i], fallback.vector[i], 1e-12) << "vector[" << i << "]";
    }
}

TEST(MixedFormPerformance, Setup_ResolvesNestedSymbolicFallbacks_ForMonolithicCellKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto alpha = svmp::FE::forms::FormExpr::parameter("alpha");
    auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");
    auto residual = (alpha * u_state * v + p_state * v).dx() + (u_state * q).dx();

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    opts.compiler_options.use_symbolic_tangent = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual, opts);
    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.parameterRegistry().slotOf("alpha").has_value());

    const auto& def = sys.operatorDefinition("op");
    ASSERT_EQ(def.cells.size(), 1u);
    const auto* monolithic =
        dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(def.cells[0].kernel.get());
    ASSERT_NE(monolithic, nullptr);
    if (!monolithic->hasCompiledDispatch()) {
        EXPECT_FALSE(monolithic->compileMessage().empty());
    }

    for (std::size_t bi = 0; bi < monolithic->numBlocks(); ++bi) {
        const auto& block = monolithic->blockSpec(bi);
        const auto* symbolic = unwrapSymbolicKernel(block.fallback_kernel);
        ASSERT_NE(symbolic, nullptr) << "block " << bi;
        EXPECT_TRUE(symbolic->tangentIR().isCompiled()) << "block " << bi;
    }
}

TEST(MixedFormPerformance, Setup_ResolvesNestedSymbolicFallbacks_ForMixedBlockKernelSet)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");
    auto residual = (u_state * v + p_state * v).dx() + (u_state * q).dx();

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = false;
    opts.compiler_options.use_symbolic_tangent = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual, opts);
    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_FALSE(installed.mixed_plan->usesMonolithicCellKernel());

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto& def = sys.operatorDefinition("op");
    ASSERT_EQ(def.cells.size(), 1u);
    const auto* mixed =
        dynamic_cast<const svmp::FE::forms::MixedBlockKernelSet*>(def.cells[0].kernel.get());
    ASSERT_NE(mixed, nullptr);

    for (std::size_t bi = 0; bi < mixed->numBlocks(); ++bi) {
        const auto& block = mixed->blockSpec(bi);
        const auto* symbolic = unwrapSymbolicKernel(block.fallback_kernel);
        ASSERT_NE(symbolic, nullptr) << "block " << bi;
        EXPECT_TRUE(symbolic->tangentIR().isCompiled()) << "block " << bi;
    }
}

TEST(MixedFormPerformance, InstallFormulation_MonolithicJITParity_TransientMixedNoVMS)
{
    const auto build_residual = [](const auto& u_space, const auto& p_space, FieldId u_field, FieldId p_field) {
        return makeTransientNavierStokesResidual(u_space, p_space, u_field, p_field, /*dim=*/3,
                                                /*enable_convection=*/true,
                                                /*enable_vms=*/false);
    };
    const auto jit = assembleTransientMixedResidual(build_residual, /*enable_jit=*/true);
    const auto fallback = assembleTransientMixedResidual(build_residual, /*enable_jit=*/false);

    EXPECT_TRUE(jit.monolithic);
    EXPECT_FALSE(fallback.monolithic);
    ASSERT_EQ(jit.matrix.size(), fallback.matrix.size());
    ASSERT_EQ(jit.vector.size(), fallback.vector.size());

    for (std::size_t i = 0; i < jit.matrix.size(); ++i) {
        EXPECT_NEAR(jit.matrix[i], fallback.matrix[i], 1e-10) << "matrix[" << i << "]";
    }
    for (std::size_t i = 0; i < jit.vector.size(); ++i) {
        EXPECT_NEAR(jit.vector[i], fallback.vector[i], 1e-10) << "vector[" << i << "]";
    }
}

TEST(MixedFormPerformance, InstallFormulation_MonolithicJITParity_TransientVMSResidual)
{
    const auto build_residual = [](const auto& u_space, const auto& p_space, FieldId u_field, FieldId p_field) {
        return makeTransientNavierStokesResidual(u_space, p_space, u_field, p_field, /*dim=*/3,
                                                /*enable_convection=*/true,
                                                /*enable_vms=*/true);
    };
    const auto jit = assembleTransientMixedResidual(build_residual, /*enable_jit=*/true);
    const auto fallback = assembleTransientMixedResidual(build_residual, /*enable_jit=*/false);

    EXPECT_TRUE(jit.monolithic);
    EXPECT_FALSE(fallback.monolithic);
    ASSERT_EQ(jit.matrix.size(), fallback.matrix.size());
    ASSERT_EQ(jit.vector.size(), fallback.vector.size());

    for (std::size_t i = 0; i < jit.matrix.size(); ++i) {
        EXPECT_NEAR(jit.matrix[i], fallback.matrix[i], 1e-10) << "matrix[" << i << "]";
    }
    for (std::size_t i = 0; i < jit.vector.size(); ++i) {
        EXPECT_NEAR(jit.vector[i], fallback.vector[i], 1e-10) << "vector[" << i << "]";
    }
}

TEST(MixedFormPerformance, InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientMixedNoVMS)
{
    const auto build_residual = [](const auto& u_space, const auto& p_space, FieldId u_field, FieldId p_field) {
        return makeTransientNavierStokesResidual(u_space, p_space, u_field, p_field, /*dim=*/3,
                                                /*enable_convection=*/true,
                                                /*enable_vms=*/false);
    };
    const auto jit = assembleTransientMixedResidualGeneralizedAlpha(build_residual, /*enable_jit=*/true);
    const auto fallback = assembleTransientMixedResidualGeneralizedAlpha(build_residual, /*enable_jit=*/false);

    EXPECT_TRUE(jit.monolithic);
    EXPECT_FALSE(fallback.monolithic);
    ASSERT_EQ(jit.matrix.size(), fallback.matrix.size());
    ASSERT_EQ(jit.vector.size(), fallback.vector.size());

    for (std::size_t i = 0; i < jit.matrix.size(); ++i) {
        EXPECT_NEAR(jit.matrix[i], fallback.matrix[i], 1e-10) << "matrix[" << i << "]";
    }
    for (std::size_t i = 0; i < jit.vector.size(); ++i) {
        EXPECT_NEAR(jit.vector[i], fallback.vector[i], 1e-10) << "vector[" << i << "]";
    }
}

TEST(MixedFormPerformance, InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual)
{
    const auto build_residual = [](const auto& u_space, const auto& p_space, FieldId u_field, FieldId p_field) {
        return makeTransientNavierStokesResidual(u_space, p_space, u_field, p_field, /*dim=*/3,
                                                /*enable_convection=*/true,
                                                /*enable_vms=*/true);
    };
    const auto jit = assembleTransientMixedResidualGeneralizedAlpha(build_residual, /*enable_jit=*/true);
    const auto fallback = assembleTransientMixedResidualGeneralizedAlpha(build_residual, /*enable_jit=*/false);

    EXPECT_TRUE(jit.monolithic);
    EXPECT_FALSE(fallback.monolithic);
    ASSERT_EQ(jit.matrix.size(), fallback.matrix.size());
    ASSERT_EQ(jit.vector.size(), fallback.vector.size());

    for (std::size_t i = 0; i < jit.matrix.size(); ++i) {
        EXPECT_NEAR(jit.matrix[i], fallback.matrix[i], 1e-10) << "matrix[" << i << "]";
    }
    for (std::size_t i = 0; i < jit.vector.size(); ++i) {
        EXPECT_NEAR(jit.vector[i], fallback.vector[i], 1e-10) << "vector[" << i << "]";
    }
}

// ============================================================================
// Functional: per-block JIT specialization works with mixed-compiled forms
// ============================================================================

TEST(MixedFormPerformance, PerBlockJIT_MixedBilinear)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    // Use grad-grad coupling (all scalar, no div needed)
    auto mixed = (inner(grad(u), grad(v))).dx() + (p * v).dx() + (u * q).dx();

    // Install with JIT and assemble
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;

    const std::array fields = {u_f, p_f};
    svmp::FE::systems::installMixedBilinear(
        sys, "op",
        std::span<const FieldId>(fields),
        std::span<const FieldId>(fields),
        mixed, opts);

    svmp::FE::systems::SetupInputs setup_inputs;
    setup_inputs.topology_override = singleTetraTopology();
    sys.setup({}, setup_inputs);

    const auto n = sys.dofHandler().getNumDofs();

    svmp::FE::assembly::DenseMatrixView mat(n);
    mat.zero();

    svmp::FE::systems::SystemStateView state;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &mat, nullptr);

    // The assembled matrix should be non-trivial
    double norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            norm += mat.getMatrixEntry(i, j) * mat.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(norm, 0.0) << "Assembled matrix from JIT mixed bilinear should be non-zero";
}

TEST(MonolithicCellKernelBlockSpec, PreservesResolvedMixedBlockMetadata)
{
    svmp::FE::spaces::H1Space scalar_space(ElementType::Tetra4, 1);
    svmp::FE::dofs::DofMap row_map(1, 4, 4);
    svmp::FE::dofs::DofMap col_map(1, 4, 4);
    const std::vector<GlobalIndex> cell_dofs{0, 1, 2, 3};
    row_map.setCellDofs(0, cell_dofs);
    row_map.setNumDofs(4);
    row_map.setNumLocalDofs(4);
    row_map.finalize();
    col_map.setCellDofs(0, cell_dofs);
    col_map.setNumDofs(4);
    col_map.setNumLocalDofs(4);
    col_map.finalize();

    svmp::FE::forms::FormCompiler compiler;
    const auto u = svmp::FE::forms::FormExpr::trialFunction(scalar_space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(scalar_space, "v");
    const auto mass = (u * v).dx();

    auto matrix_fallback = std::make_shared<svmp::FE::forms::FormKernel>(compiler.compileBilinear(mass));
    auto matrix_ir = compiler.compileBilinear(mass);
    auto residual_ir = compiler.compileResidual(mass);

    constexpr FieldId velocity_field = 11;
    constexpr FieldId pressure_field = 12;

    std::vector<svmp::FE::forms::MonolithicCellKernel::BlockSpec> blocks;
    blocks.push_back(svmp::FE::forms::MonolithicCellKernel::BlockSpec{
        .test_field = velocity_field,
        .trial_field = pressure_field,
        .want_matrix = true,
        .want_vector = false,
        .fallback_kernel = matrix_fallback,
        .tangent_ir = std::move(matrix_ir),
        .test_space = &scalar_space,
        .trial_space = &scalar_space,
        .row_dof_map = &row_map,
        .col_dof_map = &col_map,
        .row_dof_offset = 8,
        .col_dof_offset = 12,
    });
    blocks.push_back(svmp::FE::forms::MonolithicCellKernel::BlockSpec{
        .test_field = pressure_field,
        .trial_field = velocity_field,
        .want_matrix = false,
        .want_vector = true,
        .residual_ir = std::move(residual_ir),
        .test_space = &scalar_space,
        .trial_space = &scalar_space,
        .row_dof_map = &row_map,
        .col_dof_map = &col_map,
        .row_dof_offset = 16,
        .col_dof_offset = 20,
    });

    svmp::FE::forms::MonolithicCellKernel kernel(
        std::move(blocks),
        nullptr,
        svmp::FE::forms::JITOptions{});

    ASSERT_EQ(kernel.numBlocks(), 2u);
    EXPECT_TRUE(kernel.hasCell());
    EXPECT_FALSE(kernel.isMatrixOnly());
    EXPECT_FALSE(kernel.isVectorOnly());
    EXPECT_EQ(kernel.semanticKernelKind(), svmp::FE::assembly::SemanticKernelKind::MonolithicCell);
    EXPECT_FALSE(kernel.isResolved());

    const auto& matrix_block = kernel.blockSpec(0);
    EXPECT_EQ(matrix_block.test_field, velocity_field);
    EXPECT_EQ(matrix_block.trial_field, pressure_field);
    EXPECT_TRUE(matrix_block.want_matrix);
    EXPECT_FALSE(matrix_block.want_vector);
    EXPECT_EQ(matrix_block.fallback_kernel, matrix_fallback);
    EXPECT_TRUE(matrix_block.tangent_ir.has_value());
    EXPECT_FALSE(matrix_block.residual_ir.has_value());
    EXPECT_EQ(matrix_block.test_space, &scalar_space);
    EXPECT_EQ(matrix_block.trial_space, &scalar_space);
    EXPECT_EQ(matrix_block.row_dof_map, &row_map);
    EXPECT_EQ(matrix_block.col_dof_map, &col_map);
    EXPECT_EQ(matrix_block.row_dof_offset, 8);
    EXPECT_EQ(matrix_block.col_dof_offset, 12);

    const auto& vector_block = kernel.blockSpec(1);
    EXPECT_EQ(vector_block.test_field, pressure_field);
    EXPECT_EQ(vector_block.trial_field, velocity_field);
    EXPECT_FALSE(vector_block.want_matrix);
    EXPECT_TRUE(vector_block.want_vector);
    EXPECT_FALSE(vector_block.tangent_ir.has_value());
    EXPECT_TRUE(vector_block.residual_ir.has_value());
    EXPECT_EQ(vector_block.row_dof_offset, 16);
    EXPECT_EQ(vector_block.col_dof_offset, 20);

    kernel.mutableBlockSpec(1).row_dof_offset = 24;
    EXPECT_EQ(kernel.blockSpec(1).row_dof_offset, 24);
    kernel.setResolved();
    EXPECT_TRUE(kernel.isResolved());
}

TEST(MonolithicCellKernelBlockSpec, RoutesMatrixVectorAndMixedBlocksIndependently)
{
    svmp::FE::spaces::H1Space scalar_space(ElementType::Tetra4, 1);
    svmp::FE::dofs::DofMap row_map(1, 4, 4);
    svmp::FE::dofs::DofMap col_map(1, 4, 4);
    const std::vector<GlobalIndex> cell_dofs{0, 1, 2, 3};
    row_map.setCellDofs(0, cell_dofs);
    row_map.setNumDofs(4);
    row_map.setNumLocalDofs(4);
    row_map.finalize();
    col_map.setCellDofs(0, cell_dofs);
    col_map.setNumDofs(4);
    col_map.setNumLocalDofs(4);
    col_map.finalize();

    svmp::FE::forms::FormCompiler compiler;
    const auto u = svmp::FE::forms::FormExpr::trialFunction(scalar_space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(scalar_space, "v");

    auto matrix_fallback =
        std::make_shared<svmp::FE::forms::FormKernel>(compiler.compileBilinear((u * v).dx()));
    auto vector_fallback =
        std::make_shared<svmp::FE::forms::SymbolicNonlinearFormKernel>(
            compiler.compileResidual((u * u * v).dx()),
            svmp::FE::forms::NonlinearKernelOutput::Both);
    auto mixed_fallback =
        std::make_shared<svmp::FE::forms::SymbolicNonlinearFormKernel>(
            compiler.compileResidual((svmp::FE::forms::FormExpr::constant(2.0) * u * u * v).dx()),
            svmp::FE::forms::NonlinearKernelOutput::Both);

    constexpr FieldId velocity_field = 21;
    constexpr FieldId pressure_field = 22;
    constexpr FieldId temperature_field = 23;

    std::vector<svmp::FE::forms::MonolithicCellKernel::BlockSpec> blocks;
    blocks.push_back(svmp::FE::forms::MonolithicCellKernel::BlockSpec{
        .test_field = velocity_field,
        .trial_field = pressure_field,
        .want_matrix = true,
        .want_vector = false,
        .fallback_kernel = matrix_fallback,
        .tangent_ir = compiler.compileBilinear((u * v).dx()),
        .test_space = &scalar_space,
        .trial_space = &scalar_space,
        .row_dof_map = &row_map,
        .col_dof_map = &col_map,
        .row_dof_offset = 4,
        .col_dof_offset = 8,
    });
    blocks.push_back(svmp::FE::forms::MonolithicCellKernel::BlockSpec{
        .test_field = pressure_field,
        .trial_field = velocity_field,
        .want_matrix = false,
        .want_vector = true,
        .fallback_kernel = vector_fallback,
        .residual_ir = compiler.compileResidual((u * u * v).dx()),
        .test_space = &scalar_space,
        .trial_space = &scalar_space,
        .row_dof_map = &row_map,
        .col_dof_map = &col_map,
        .row_dof_offset = 12,
        .col_dof_offset = 16,
    });
    blocks.push_back(svmp::FE::forms::MonolithicCellKernel::BlockSpec{
        .test_field = temperature_field,
        .trial_field = temperature_field,
        .want_matrix = true,
        .want_vector = true,
        .fallback_kernel = mixed_fallback,
        .tangent_ir = compiler.compileBilinear((svmp::FE::forms::FormExpr::constant(3.0) * u * v).dx()),
        .residual_ir = compiler.compileResidual((svmp::FE::forms::FormExpr::constant(4.0) * u * u * v).dx()),
        .test_space = &scalar_space,
        .trial_space = &scalar_space,
        .row_dof_map = &row_map,
        .col_dof_map = &col_map,
        .row_dof_offset = 20,
        .col_dof_offset = 24,
    });

    svmp::FE::forms::MonolithicCellKernel kernel(
        std::move(blocks),
        nullptr,
        svmp::FE::forms::JITOptions{});

    ASSERT_EQ(kernel.numBlocks(), 3u);
    EXPECT_TRUE(kernel.hasCell());
    EXPECT_FALSE(kernel.isMatrixOnly());
    EXPECT_FALSE(kernel.isVectorOnly());
    EXPECT_EQ(kernel.name(), "MonolithicCellKernel[3 blocks]");

    const auto& matrix_only = kernel.blockSpec(0);
    EXPECT_EQ(matrix_only.test_field, velocity_field);
    EXPECT_EQ(matrix_only.trial_field, pressure_field);
    EXPECT_TRUE(matrix_only.want_matrix);
    EXPECT_FALSE(matrix_only.want_vector);
    EXPECT_EQ(matrix_only.fallback_kernel, matrix_fallback);
    EXPECT_TRUE(matrix_only.tangent_ir.has_value());
    EXPECT_FALSE(matrix_only.residual_ir.has_value());
    EXPECT_EQ(matrix_only.row_dof_map, &row_map);
    EXPECT_EQ(matrix_only.col_dof_map, &col_map);
    EXPECT_EQ(matrix_only.row_dof_offset, 4);
    EXPECT_EQ(matrix_only.col_dof_offset, 8);

    const auto& vector_only = kernel.blockSpec(1);
    EXPECT_EQ(vector_only.test_field, pressure_field);
    EXPECT_EQ(vector_only.trial_field, velocity_field);
    EXPECT_FALSE(vector_only.want_matrix);
    EXPECT_TRUE(vector_only.want_vector);
    EXPECT_EQ(vector_only.fallback_kernel, vector_fallback);
    EXPECT_FALSE(vector_only.tangent_ir.has_value());
    EXPECT_TRUE(vector_only.residual_ir.has_value());
    EXPECT_EQ(vector_only.row_dof_offset, 12);
    EXPECT_EQ(vector_only.col_dof_offset, 16);

    const auto& mixed = kernel.blockSpec(2);
    EXPECT_EQ(mixed.test_field, temperature_field);
    EXPECT_EQ(mixed.trial_field, temperature_field);
    EXPECT_TRUE(mixed.want_matrix);
    EXPECT_TRUE(mixed.want_vector);
    EXPECT_EQ(mixed.fallback_kernel, mixed_fallback);
    EXPECT_TRUE(mixed.tangent_ir.has_value());
    EXPECT_TRUE(mixed.residual_ir.has_value());
    EXPECT_EQ(mixed.row_dof_offset, 20);
    EXPECT_EQ(mixed.col_dof_offset, 24);

    kernel.mutableBlockSpec(2).col_dof_offset = 28;
    EXPECT_EQ(kernel.blockSpec(2).col_dof_offset, 28);
}

// ============================================================================
// DISABLED benchmark: compilation timing for mixed vs manual
// Enable with --gtest_also_run_disabled_tests
// ============================================================================

TEST(MixedFormPerformance, DISABLED_Benchmark_CompilationTiming)
{
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    constexpr int N = 100;

    // Benchmark: manual block compilation
    double manual_time = timeSeconds([&]() {
        svmp::FE::forms::FormCompiler compiler;
        for (int i = 0; i < N; ++i) {
            (void)compiler.compileBilinear((inner(grad(u), grad(v))).dx());
            (void)compiler.compileBilinear((p * v).dx());
            (void)compiler.compileBilinear((u * q).dx());
        }
    });

    // Benchmark: mixed compilation
    auto mixed = (inner(grad(u), grad(v))).dx() + (p * v).dx() + (u * q).dx();
    double mixed_time = timeSeconds([&]() {
        svmp::FE::forms::FormCompiler compiler;
        for (int i = 0; i < N; ++i) {
            (void)compiler.compileMixed(mixed, svmp::FE::forms::FormKind::Bilinear);
        }
    });

    std::cout << "\n=== Compilation Benchmark (" << N << " iterations) ===\n"
              << "  Manual (3 separate compileBilinear): " << manual_time * 1000.0 << " ms\n"
              << "  Mixed  (1 compileMixed):             " << mixed_time * 1000.0 << " ms\n"
              << "  Ratio (mixed/manual):                " << mixed_time / manual_time << "x\n"
              << std::endl;

    // Mixed should not be more than 3x slower than manual (generous margin)
    EXPECT_LT(mixed_time, manual_time * 3.0)
        << "Mixed compilation should not be significantly slower than manual block compilation";
}
