/**
 * @file test_FormsInstaller.cpp
 * @brief Unit tests for Systems FormsInstaller helpers
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include "Spaces/H1Space.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::INVALID_FIELD_ID;
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
    const svmp::FE::spaces::FunctionSpace& test_space,
    const svmp::FE::spaces::FunctionSpace& trial_space,
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
    (void)assembler.assembleMatrix(mesh, test_space, trial_space, kernel, mat);
    return mat;
}

svmp::FE::assembly::DenseVectorView assembleLinear(
    const svmp::FE::forms::FormExpr& form,
    const svmp::FE::spaces::FunctionSpace& test_space,
    const svmp::FE::assembly::IMeshAccess& mesh)
{
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileLinear(form);
    svmp::FE::forms::FormKernel kernel(std::move(ir));

    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();
    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseVectorView vec(4);
    vec.zero();
    (void)assembler.assembleVector(mesh, test_space, kernel, vec);
    return vec;
}

} // namespace

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_RegistersKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(sys, "op", u_field, u_field, residual_form);
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::LinearFormKernel*>(installed.get()), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    const auto mass = assembleBilinear((u * v).dx(), *space, *space, *mesh);

    // Matrix matches mass.
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), mass.getMatrixEntry(i, j), 1e-12);
        }
    }

    // Vector matches mass * U.
    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += mass.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_AffineWithRHS_UsesLinearKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto one = svmp::FE::forms::FormExpr::constant(1.0);

    // Residual: ∫ u v dx - ∫ 1 * v dx
    const auto residual_form = (u * v - one * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(sys, "op", u_field, u_field, residual_form);
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::LinearFormKernel*>(installed.get()), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(4);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    const auto mass = assembleBilinear((u * v).dx(), *space, *space, *mesh);
    const auto rhs = assembleLinear((-one * v).dx(), *space, *mesh);

    for (GlobalIndex i = 0; i < 4; ++i) {
        // Matrix matches mass.
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), mass.getMatrixEntry(i, j), 1e-12);
        }

        // Vector matches mass*U + rhs.
        Real expected = rhs.getVectorEntry(i);
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += mass.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_ADModeForward)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * u * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(
        sys, "op", u_field, u_field, residual_form,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Forward});
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::NonlinearFormKernel*>(installed.get()), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_ADModeReverse)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * u * v).dx();

    auto installed = svmp::FE::systems::installResidualForm(
        sys, "op", u_field, u_field, residual_form,
        svmp::FE::systems::FormInstallOptions{.ad_mode = svmp::FE::forms::ADMode::Reverse});
    ASSERT_NE(installed, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::NonlinearFormKernel*>(installed.get()), nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-7;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 5e-6);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallResidualForm_InvalidFieldId_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    EXPECT_THROW((void)svmp::FE::systems::installResidualForm(sys, "op", INVALID_FIELD_ID, u_field, residual_form),
                 svmp::FE::InvalidArgumentException);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_MultipleBlocksRegistered)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    const auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());
    blocks.setBlock(1, 1, (p * q).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    const auto kernels = svmp::FE::systems::installResidualBlocks(sys, "op", fields, fields, blocks);
    ASSERT_EQ(kernels.size(), 2u);
    ASSERT_EQ(kernels[0].size(), 2u);

    EXPECT_NE(kernels[0][0], nullptr);
    EXPECT_EQ(kernels[0][1], nullptr);
    EXPECT_EQ(kernels[1][0], nullptr);
    EXPECT_NE(kernels[1][1], nullptr);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_EmptyBlocksSkipped)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());

    const std::array<FieldId, 2> fields = {u_field, p_field};
    const auto kernels = svmp::FE::systems::installResidualBlocks(sys, "op", fields, fields, blocks);
    EXPECT_NE(kernels[0][0], nullptr);
    EXPECT_EQ(kernels[0][1], nullptr);
    EXPECT_EQ(kernels[1][0], nullptr);
    EXPECT_EQ(kernels[1][1], nullptr);
}

TEST(FormsInstaller, FormsInstaller_InstallResidualBlocks_InitializerListOverload)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::BlockBilinearForm blocks(/*tests=*/2, /*trials=*/2);
    blocks.setBlock(0, 0, (u * v).dx());

    // Span overload.
    svmp::FE::systems::FESystem sys_span(mesh);
    const auto u0 = sys_span.addField(svmp::FE::systems::FieldSpec{.name = "u0", .space = space, .components = 1});
    const auto u1 = sys_span.addField(svmp::FE::systems::FieldSpec{.name = "u1", .space = space, .components = 1});
    sys_span.addOperator("op");
    const std::array<FieldId, 2> fields_span = {u0, u1};
    const auto kernels_span = svmp::FE::systems::installResidualBlocks(sys_span, "op", fields_span, fields_span, blocks);

    // Initializer-list overload.
    svmp::FE::systems::FESystem sys_list(mesh);
    const auto v0 = sys_list.addField(svmp::FE::systems::FieldSpec{.name = "v0", .space = space, .components = 1});
    const auto v1 = sys_list.addField(svmp::FE::systems::FieldSpec{.name = "v1", .space = space, .components = 1});
    sys_list.addOperator("op");
    const auto kernels_list = svmp::FE::systems::installResidualBlocks(sys_list, "op", {v0, v1}, {v0, v1}, blocks);

    ASSERT_EQ(kernels_span.size(), kernels_list.size());
    for (std::size_t i = 0; i < kernels_span.size(); ++i) {
        ASSERT_EQ(kernels_span[i].size(), kernels_list[i].size());
        for (std::size_t j = 0; j < kernels_span[i].size(); ++j) {
            EXPECT_EQ(static_cast<bool>(kernels_span[i][j]), static_cast<bool>(kernels_list[i][j]));
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallCoupledResidual_SeparatesVectorAndMatrix)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    svmp::FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(0, (u_state * v + p_state * v).dx()); // depends on u and p
    residual.setBlock(1, (q * u_state).dx());               // depends on u only

    const std::array<FieldId, 2> fields = {u_field, p_field};
    (void)svmp::FE::systems::installCoupledResidual(sys, "op", fields, fields, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 8);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.1) * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    svmp::FE::systems::AssemblyRequest req_mat;
    req_mat.op = "op";
    req_mat.want_matrix = true;

    svmp::FE::systems::AssemblyRequest req_both;
    req_both.op = "op";
    req_both.want_matrix = true;
    req_both.want_vector = true;

    svmp::FE::assembly::DenseVectorView R_vec(n_dofs);
    svmp::FE::assembly::DenseMatrixView J_mat(n_dofs);
    svmp::FE::assembly::DenseSystemView both(n_dofs);
    R_vec.zero();
    J_mat.zero();
    both.zero();

    (void)sys.assemble(req_vec, state, nullptr, &R_vec);
    (void)sys.assemble(req_mat, state, &J_mat, nullptr);
    (void)sys.assemble(req_both, state, &both, &both);

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        EXPECT_NEAR(both.getVectorEntry(i), R_vec.getVectorEntry(i), 1e-12);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            EXPECT_NEAR(both.getMatrixEntry(i, j), J_mat.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(FormsInstaller, FormsInstaller_InstallCoupledResidual_StateFieldsTracked)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    svmp::FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(0, (u_state * v + p_state * v).dx()); // depends on u and p
    residual.setBlock(1, (q * u_state).dx());               // depends on u only

    const std::array<FieldId, 2> fields = {u_field, p_field};
    const auto installed = svmp::FE::systems::installCoupledResidual(sys, "op", fields, fields, residual);

    ASSERT_EQ(installed.jacobian_blocks.size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[0].size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[1].size(), 2u);

    EXPECT_NE(installed.jacobian_blocks[0][0], nullptr);
    EXPECT_NE(installed.jacobian_blocks[0][1], nullptr);
    EXPECT_NE(installed.jacobian_blocks[1][0], nullptr);
    EXPECT_EQ(installed.jacobian_blocks[1][1], nullptr); // dR_p / dp == 0
}

TEST(FormsInstaller, FormsInstaller_FormWithoutDx_Behavior)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    try {
        (void)svmp::FE::systems::installResidualForm(sys, "op", u_field, u_field, (u * v));
        FAIL() << "Expected installResidualForm to throw for residual missing dx()/ds()/dS()";
    } catch (const svmp::FE::InvalidArgumentException&) {
        SUCCEED();
    } catch (const std::invalid_argument&) {
        SUCCEED();
    }
}

TEST(FormsInstaller, FormsInstaller_MismatchedFieldSpaces_Behavior)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_field = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    auto space_form = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space_field, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space_form, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space_form, "v");
    const auto residual_form = (u * v).dx();

    EXPECT_THROW((void)svmp::FE::systems::installResidualForm(sys, "op", u_field, u_field, residual_form),
                 svmp::FE::InvalidArgumentException);
}
