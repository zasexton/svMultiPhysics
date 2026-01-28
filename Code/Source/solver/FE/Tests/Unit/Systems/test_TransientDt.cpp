/**
 * @file test_TransientDt.cpp
 * @brief Unit tests for symbolic `dt(u,k)` wiring through Systems transient infrastructure
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"

#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <string>
#include <vector>

using svmp::FE::ElementType;
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

svmp::FE::assembly::DenseMatrixView assembleBilinear(
    const svmp::FE::forms::FormExpr& form,
    const svmp::FE::spaces::FunctionSpace& space,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear(form);
    svmp::FE::forms::FormKernel kernel(std::move(ir));

    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();
    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

} // namespace

TEST(TransientDtTest, HeatEquationDtPlusDiffusion_AssemblesScaledMassPlusStiffness)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form = (svmp::FE::forms::dt(u) * v + svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v))).dx();
    auto ir = compiler.compileBilinear(form);
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());
    EXPECT_EQ(sys.temporalOrder(), 1);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(4);
    A.zero();
    (void)transient.assemble(req, state, &A, nullptr);

    // Reference: assemble mass and stiffness and compare A == (1/dt)*M + K.
    const auto mass = assembleBilinear((u * v).dx(), *space, *mesh);
    const auto stiff = assembleBilinear(svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx(),
                                        *space, *mesh);

    const Real a0 = 1.0 / static_cast<Real>(state.dt);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = a0 * mass.getMatrixEntry(i, j) + stiff.getMatrixEntry(i, j);
            EXPECT_NEAR(A.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(TransientDtTest, WaveEquationDt2PlusDiffusion_AssemblesScaledMassPlusStiffness)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form = (svmp::FE::forms::dt(u, 2) * v + svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v))).dx();
    auto ir = compiler.compileBilinear(form);
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());
    EXPECT_EQ(sys.temporalOrder(), 2);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(4);
    A.zero();
    (void)transient.assemble(req, state, &A, nullptr);

    const auto mass = assembleBilinear((u * v).dx(), *space, *mesh);
    const auto stiff = assembleBilinear(svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx(),
                                        *space, *mesh);

    const Real dt = static_cast<Real>(state.dt);
    const Real a0 = 1.0 / (dt * dt);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = a0 * mass.getMatrixEntry(i, j) + stiff.getMatrixEntry(i, j);
            EXPECT_NEAR(A.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(TransientDtTest, HigherOrderDtCompilesButRequiresHistoryVector)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear((svmp::FE::forms::dt(u, 3) * v).dx());
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());
    EXPECT_EQ(sys.temporalOrder(), 3);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(4);
    A.zero();

    try {
        (void)transient.assemble(req, state, &A, nullptr);
        FAIL() << "Expected InvalidArgumentException";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("state.u_history"), std::string::npos);
    }
}

TEST(TransientDtTest, ResidualAndJacobianIncludeHistoryForDt)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto residual_form = (svmp::FE::forms::dt(u) * v + svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v))).dx();
    auto residual_ir = compiler.compileResidual(residual_form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(residual_ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n = {1.0, 2.0, 3.0, 4.0};
    const std::vector<Real> u_prev = {0.25, -0.5, 0.75, -1.25};

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    (void)transient.assemble(req, state, &out, &out);

    const auto mass = assembleBilinear((u * v).dx(), *space, *mesh);
    const auto stiff = assembleBilinear(svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx(),
                                        *space, *mesh);

    const Real dt = static_cast<Real>(state.dt);
    const Real a0 = 1.0 / dt;
    const Real a1 = -1.0 / dt;

    // Jacobian: a0*M + K
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = a0 * mass.getMatrixEntry(i, j) + stiff.getMatrixEntry(i, j);
            EXPECT_NEAR(out.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    // Residual: a0*M*u_n + a1*M*u_prev + K*u_n
    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += a0 * mass.getMatrixEntry(i, j) * u_n[static_cast<std::size_t>(j)];
            expected += a1 * mass.getMatrixEntry(i, j) * u_prev[static_cast<std::size_t>(j)];
            expected += stiff.getMatrixEntry(i, j) * u_n[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(TransientDtTest, Bdf2RequiresSecondHistoryState)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto residual_form = (svmp::FE::forms::dt(u) * v).dx();
    auto residual_ir = compiler.compileResidual(residual_form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(residual_ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n(4, 1.0);
    const std::vector<Real> u_prev(4, 0.0);

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.dt_prev = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BDF2Integrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();

    EXPECT_THROW((void)transient.assemble(req, state, &out, &out), svmp::FE::InvalidArgumentException);
}

TEST(TransientDtTest, ResidualAndJacobianIncludeTwoHistoryStatesForBdf2)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto residual_form =
        (svmp::FE::forms::dt(u) * v + svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v))).dx();
    auto residual_ir = compiler.compileResidual(residual_form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(residual_ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const std::vector<Real> u_n = {1.0, 2.0, 3.0, 4.0};
    const std::vector<Real> u_prev = {0.25, -0.5, 0.75, -1.25};
    const std::vector<Real> u_prev2 = {-0.125, 0.375, -0.625, 0.875};

    auto run_case = [&](double dt, double dt_prev) {
        svmp::FE::systems::SystemStateView state;
        state.dt = dt;
        state.dt_prev = dt_prev;
        state.u = u_n;
        state.u_prev = u_prev;
        state.u_prev2 = u_prev2;

        svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BDF2Integrator>());

        svmp::FE::systems::AssemblyRequest req;
        req.op = "op";
        req.want_matrix = true;
        req.want_vector = true;

        svmp::FE::assembly::DenseSystemView out(4);
        out.zero();
        (void)transient.assemble(req, state, &out, &out);

        const auto mass = assembleBilinear((u * v).dx(), *space, *mesh);
        const auto stiff = assembleBilinear(svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)).dx(),
                                            *space, *mesh);

        const Real r = static_cast<Real>(dt / dt_prev);
        const Real inv_dt = 1.0 / static_cast<Real>(dt);
        const Real a0 = ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt;
        const Real a1 = (-(1.0 + r)) * inv_dt;
        const Real a2 = ((r * r) / (1.0 + r)) * inv_dt;

        // Jacobian: a0*M + K
        for (GlobalIndex i = 0; i < 4; ++i) {
            for (GlobalIndex j = 0; j < 4; ++j) {
                const Real expected = a0 * mass.getMatrixEntry(i, j) + stiff.getMatrixEntry(i, j);
                EXPECT_NEAR(out.getMatrixEntry(i, j), expected, 1e-12);
            }
        }

        // Residual: a0*M*u_n + a1*M*u_prev + a2*M*u_prev2 + K*u_n
        for (GlobalIndex i = 0; i < 4; ++i) {
            Real expected = 0.0;
            for (GlobalIndex j = 0; j < 4; ++j) {
                expected += a0 * mass.getMatrixEntry(i, j) * u_n[static_cast<std::size_t>(j)];
                expected += a1 * mass.getMatrixEntry(i, j) * u_prev[static_cast<std::size_t>(j)];
                expected += a2 * mass.getMatrixEntry(i, j) * u_prev2[static_cast<std::size_t>(j)];
                expected += stiff.getMatrixEntry(i, j) * u_n[static_cast<std::size_t>(j)];
            }
            EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
        }
    };

    run_case(/*dt=*/0.5, /*dt_prev=*/0.5);
    run_case(/*dt=*/0.5, /*dt_prev=*/0.25);
}

TEST(TransientDtTest, SteadyAssemblyOfDtFormFailsWithClearDiagnostic)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear((svmp::FE::forms::dt(u) * v).dx());
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(4);
    A.zero();

    try {
        (void)sys.assemble(req, svmp::FE::systems::SystemStateView{}, &A, nullptr);
        FAIL() << "Expected steady assembly to fail for dt(...)";
    } catch (const svmp::FE::FEException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("dt(...) operator requires a transient time-integration context"), std::string::npos);
    }
}

TEST(TransientDtTest, VectorDtPlusDiffusion_AssemblesBlockDiagonalScaledMassPlusStiffness)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto vec_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, ElementType::Tetra4, 1, 3);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = vec_space, .components = 3});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*vec_space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*vec_space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form = (svmp::FE::forms::inner(svmp::FE::forms::dt(u), v) +
                       svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)))
                          .dx();

    auto ir = compiler.compileBilinear(form);
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());
    EXPECT_EQ(sys.temporalOrder(), 1);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 12);

    const std::vector<Real> u_n(static_cast<std::size_t>(n_dofs), 0.0);
    const std::vector<Real> u_prev(static_cast<std::size_t>(n_dofs), 0.0);

    svmp::FE::systems::SystemStateView state;
    state.dt = 0.5;
    state.u = u_n;
    state.u_prev = u_prev;

    svmp::FE::systems::TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(n_dofs);
    A.zero();
    (void)transient.assemble(req, state, &A, nullptr);

    const auto mass = assembleBilinear((svmp::FE::forms::FormExpr::trialFunction(*scalar_space, "u") *
                                        svmp::FE::forms::FormExpr::testFunction(*scalar_space, "v"))
                                           .dx(),
                                       *scalar_space, *mesh);
    const auto stiff = assembleBilinear(
        svmp::FE::forms::inner(svmp::FE::forms::grad(svmp::FE::forms::FormExpr::trialFunction(*scalar_space, "u")),
                               svmp::FE::forms::grad(svmp::FE::forms::FormExpr::testFunction(*scalar_space, "v")))
            .dx(),
        *scalar_space, *mesh);

    const Real a0 = 1.0 / static_cast<Real>(state.dt);
    const Real B00 = a0 * mass.getMatrixEntry(0, 0) + stiff.getMatrixEntry(0, 0);
    (void)B00; // silence -Wunused-but-set-variable under some toolchains

    const GlobalIndex dofs_per_comp = 4;
    for (GlobalIndex ci = 0; ci < 3; ++ci) {
        for (GlobalIndex cj = 0; cj < 3; ++cj) {
            for (GlobalIndex i = 0; i < dofs_per_comp; ++i) {
                for (GlobalIndex j = 0; j < dofs_per_comp; ++j) {
                    const GlobalIndex I = ci * dofs_per_comp + i;
                    const GlobalIndex J = cj * dofs_per_comp + j;
                    const Real expected = (ci == cj)
                        ? (a0 * mass.getMatrixEntry(i, j) + stiff.getMatrixEntry(i, j))
                        : 0.0;
                    EXPECT_NEAR(A.getMatrixEntry(I, J), expected, 1e-12);
                }
            }
        }
    }
}

TEST(TransientDtTest, SteadyAssemblyOfVectorDtFormFailsWithClearDiagnostic)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto vec_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, ElementType::Tetra4, 1, 3);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = vec_space, .components = 3});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*vec_space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*vec_space, "v");

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileBilinear(svmp::FE::forms::inner(svmp::FE::forms::dt(u), v).dx());
    auto kernel = std::make_shared<svmp::FE::forms::FormKernel>(std::move(ir));
    sys.addCellKernel("op", u_field, u_field, kernel);
    EXPECT_TRUE(sys.isTransient());

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView A(sys.dofHandler().getNumDofs());
    A.zero();

    try {
        (void)sys.assemble(req, svmp::FE::systems::SystemStateView{}, &A, nullptr);
        FAIL() << "Expected steady assembly to fail for dt(...) on a vector field";
    } catch (const svmp::FE::FEException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("dt(...) operator requires a transient time-integration context"), std::string::npos);
    }
}
