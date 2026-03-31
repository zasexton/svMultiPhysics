/**
 * @file test_FormsInstaller.cpp
 * @brief Unit tests for Systems FormsInstaller helpers
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstallerDetail.h"
#include "Systems/AuxiliaryModelBuilder.h"
#include "Systems/AuxiliaryBindings.h"
#include "Systems/AuxiliaryInputRegistry.h"
#include "Systems/AuxiliaryOperatorBuilder.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Forms/WeakForm.h"

#include "Spaces/H1Space.h"
#include "Spaces/HCurlSpace.h"

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

const svmp::FE::forms::NonlinearFormKernel* unwrapNonlinearKernel(
    const std::shared_ptr<svmp::FE::assembly::AssemblyKernel>& kernel)
{
    if (!kernel) {
        return nullptr;
    }
    if (const auto* jit = dynamic_cast<const svmp::FE::forms::jit::JITKernelWrapper*>(kernel.get())) {
        return dynamic_cast<const svmp::FE::forms::NonlinearFormKernel*>(&jit->fallbackKernel());
    }
    return dynamic_cast<const svmp::FE::forms::NonlinearFormKernel*>(kernel.get());
}

} // namespace

TEST(FormsInstaller, FormsInstaller_InstallFormulation_RegistersKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    auto installed = svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

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

    // Use trialFunction for the verification helper (assembleBilinear requires it).
    const auto u_trial = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto mass = assembleBilinear((u_trial * v).dx(), *space, *space, *mesh);

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

TEST(FormsInstaller, FormsInstaller_InstallFormulation_AffineWithRHS)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto one = svmp::FE::forms::FormExpr::constant(1.0);

    // Residual: ∫ u v dx - ∫ 1 * v dx
    const auto residual_form = (u * v - one * v).dx();

    auto installed = svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    ASSERT_FALSE(installed.residual.empty());
    ASSERT_NE(installed.residual[0], nullptr);

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

    // Use trialFunction for the verification helper (assembleBilinear requires it).
    const auto u_trial = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto mass = assembleBilinear((u_trial * v).dx(), *space, *space, *mesh);
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

TEST(FormsInstaller, FormsInstaller_InstallFormulation_MultiOpInstallsConstraintsOnce)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual_form = (u * v).dx();

    const auto bc = svmp::FE::forms::bc::strongDirichlet(u_field, marker, svmp::FE::forms::FormExpr::constant(2.5), "u");
    svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    (void)svmp::FE::systems::installFormulation(sys, "op", {u_field}, residual_form);
    (void)svmp::FE::systems::installFormulation(sys, "op2", {u_field}, residual_form);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    for (svmp::FE::GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;

    svmp::FE::assembly::DenseMatrixView mat(sys.dofHandler().getNumDofs());
    mat.zero();
    (void)sys.assemble(req, state, &mat, nullptr);

    for (svmp::FE::GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(mat.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (svmp::FE::GlobalIndex j = 0; j < mat.numCols(); ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mat.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
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
    EXPECT_NE(unwrapNonlinearKernel(installed), nullptr);

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
    EXPECT_NE(unwrapNonlinearKernel(installed), nullptr);

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

TEST(FormsInstaller, FormsInstaller_InstallFormulation_CoupledSeparatesVectorAndMatrix)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto p_state = svmp::FE::forms::FormExpr::stateField(p_field, *space, "p");
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_field, *space, "q");

    const auto residual =
        (u_state * v + p_state * v).dx() +  // depends on u and p
        (q * u_state).dx();                  // depends on u only

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    const auto installed =
        svmp::FE::systems::installFormulation(sys, "op", {u_field, p_field}, residual, opts);

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
    const auto v = svmp::FE::forms::FormExpr::testFunction(u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(p_field, *space, "q");

    svmp::FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(0, (u_state * v + p_state * v).dx()); // depends on u and p
    residual.setBlock(1, (q * u_state).dx());               // depends on u only

    const std::array<FieldId, 2> fields = {u_field, p_field};
    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;
    const auto installed =
        svmp::FE::systems::installCoupledResidual(sys, "op", fields, fields, residual, opts);

    ASSERT_EQ(installed.jacobian_blocks.size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[0].size(), 2u);
    ASSERT_EQ(installed.jacobian_blocks[1].size(), 2u);

    EXPECT_NE(installed.jacobian_blocks[0][0], nullptr);
    EXPECT_NE(installed.jacobian_blocks[0][1], nullptr);
    EXPECT_NE(installed.jacobian_blocks[1][0], nullptr);
    EXPECT_EQ(installed.jacobian_blocks[1][1], nullptr); // dR_p / dp == 0

    ASSERT_NE(installed.mixed_plan, nullptr);
    EXPECT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());
    ASSERT_EQ(installed.mixed_plan->blocks.size(), 3u);

    std::size_t vector_blocks = 0;
    for (const auto& block : installed.mixed_plan->blocks) {
        if (block.want_vector) {
            ++vector_blocks;
        }
    }
    EXPECT_EQ(vector_blocks, 2u);
}

TEST(FormsInstaller, FormsInstaller_InstallFormulation_FormWithoutDx_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    try {
        (void)svmp::FE::systems::installFormulation(sys, "op", {u_field}, (u * v));
        FAIL() << "Expected installFormulation to throw for residual missing dx()/ds()/dS()";
    } catch (const svmp::FE::InvalidArgumentException&) {
        SUCCEED();
    } catch (const std::invalid_argument&) {
        SUCCEED();
    }
}

TEST(FormsInstaller, FormsInstaller_InstanceQualifiedAuxiliaryOutput)
{
    // Deploy two auxiliary models with same output name, then install a
    // form referencing AuxiliaryOutput(instance, name) to verify the
    // FormsInstaller resolves the instance-qualified path correctly.
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Two models both with output named "P_out".
    auto model_a = AuxiliaryModelBuilder("model_a")
        .state("x")
        .ode("x", -modelState("x"))
        .output("P_out", modelState("x") * forms::FormExpr::constant(2.0))
        .build();

    auto model_b = AuxiliaryModelBuilder("model_b")
        .state("y")
        .ode("y", -modelState("y"))
        .output("P_out", modelState("y") * forms::FormExpr::constant(3.0))
        .build();

    sys.deployAuxiliaryModel(
        use(model_a)
            .name("inst_a")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .initialize({1.0}));

    sys.deployAuxiliaryModel(
        use(model_b)
            .name("inst_b")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .initialize({1.0}));

    sys.finalizeAuxiliaryLayout();

    // Install a form referencing instance-qualified AuxiliaryOutput.
    // The form: (AuxiliaryOutput("inst_a", "P_out") * v).dx()
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto aux_a = forms::AuxiliaryOutput("inst_a", "P_out");
    const auto residual_form = (aux_a * v).dx();

    // This should NOT throw — the FormsInstaller should resolve
    // "inst_a/P_out" to a valid slot via auxiliaryOutputSlotOf("inst_a", "P_out").
    auto installed = installFormulation(sys, "op", {u_field}, residual_form);
    EXPECT_FALSE(installed.residual.empty());

    // Verify slot resolution correctness: inst_a and inst_b should get
    // different slots, and the form should reference inst_a's slot.
    auto slot_a = sys.auxiliaryOutputSlotOf("inst_a", "P_out");
    auto slot_b = sys.auxiliaryOutputSlotOf("inst_b", "P_out");
    EXPECT_NE(slot_a, slot_b);

    // Prepare auxiliary state: x=5 → inst_a output = 5*2 = 10,
    //                          y=7 → inst_b output = 7*3 = 21.
    sys.prepareAuxiliaryForAssembly({});

    auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), std::max(slot_a, slot_b));
    // x was initialized to 1.0, so P_out_a = 1.0*2 = 2.0
    // y was initialized to 1.0, so P_out_b = 1.0*3 = 3.0.
    EXPECT_DOUBLE_EQ(outputs[slot_a], 2.0);
    EXPECT_DOUBLE_EQ(outputs[slot_b], 3.0);
}

TEST(FormsInstaller, FormsInstaller_AuxiliaryInput_AutoResolution)
{
    // Deploy an auxiliary model that consumes an input, register the input,
    // then install a form referencing AuxiliaryInput("Q") to verify the
    // FormsInstaller auto-resolves the input symbol to a slot ref.
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Register an auxiliary input "Q".
    auto& reg = sys.auxiliaryInputRegistry();
    reg.registerInput({.name = "Q", .size = 1},
                      [](Real, Real, std::span<Real> out) { out[0] = 42.0; });

    // Deploy a model that uses input "Q".
    auto model = AuxiliaryModelBuilder("rcr")
        .state("P").input("Q")
        .ode("P", -modelState("P"))
        .output("P_out", modelState("P"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("rcr_inst")
            .scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned)
            .bind("Q", "Q")
            .initialize({0.0}));

    sys.finalizeAuxiliaryLayout();

    // Install a form referencing AuxiliaryInput("Q").
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto q = forms::AuxiliaryInput("Q");
    const auto residual_form = (q * v).dx();

    // This should NOT throw — "Q" is registered and should resolve.
    auto installed = installFormulation(sys, "op", {u_field}, residual_form);
    EXPECT_FALSE(installed.residual.empty());

    // Verify the slot was resolved: "Q" should have slot 0.
    auto input_slot = reg.slotOf("Q");
    EXPECT_EQ(input_slot, 0u);
}

TEST(FormsInstaller, RegisterSampledFieldInput_ReadsFromSolution)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Install a simple form so setup() can build DOF maps.
    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Register a sampled field input for "u" with 4 vertices.
    sys.registerSampledFieldInput("u_sampled", "u", 4);

    // Deploy an auxiliary model that uses the sampled input.
    auto model = AuxiliaryModelBuilder("sampler")
        .state("x")
        .input("Q")
        .ode("x", -modelState("x"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("samp_inst")
            .scope(AuxiliaryStateScope::Global)
            .entityCount(4)
            .bind("Q", "u_sampled")
            .initialize({0.0}));

    sys.finalizeAuxiliaryLayout();

    // Set solution vector and prepare assembly.
    std::vector<Real> U = {10.0, 20.0, 30.0, 40.0};
    SystemStateView state;
    state.u = U; state.time = 0.0; state.dt = 0.1;
    sys.prepareAuxiliaryForAssembly(state);

    // The sampled input should now contain the field values at each vertex.
    auto& reg = sys.auxiliaryInputRegistry();
    ASSERT_TRUE(reg.hasInput("u_sampled"));
    for (std::size_t i = 0; i < 4; ++i) {
        auto vals = reg.valuesOf("u_sampled", i);
        ASSERT_FALSE(vals.empty());
        EXPECT_NEAR(vals[0], U[i], 1e-12)
            << "Vertex " << i << " should have field value " << U[i];
    }
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_SumsCorrectNodes)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    // SingleTetraOneBoundaryFaceMeshAccess: 1 Tet4, boundary face 0 with marker 42.
    // Tet4 face 0 local vertices = {1,2,3}.
    // Global nodes: 0,1,2,3.  Face nodes = {1,2,3}.
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(42);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(u_field, *space, "p");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    installFormulation(sys, "op", {u_field}, (u * v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Register boundary nodal sum for boundary marker 42.
    sys.registerBoundaryNodalSumInput("p_bnd", "p", 42);

    // Deploy aux model using the boundary sum.
    auto model = AuxiliaryModelBuilder("bnd_test")
        .state("x")
        .input("Q")
        .ode("x", -modelState("x"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("bnd_inst")
            .bind("Q", "p_bnd")
            .initialize({0.0}));
    sys.finalizeAuxiliaryLayout();

    // Solution: p = {100, 200, 300, 400} at nodes 0,1,2,3.
    // Boundary face nodes = {1,2,3} → sum = 200 + 300 + 400 = 900.
    std::vector<Real> U = {100.0, 200.0, 300.0, 400.0};
    SystemStateView state;
    state.u = U; state.time = 0.0; state.dt = 0.1;
    sys.prepareAuxiliaryForAssembly(state);

    auto& reg = sys.auxiliaryInputRegistry();
    ASSERT_TRUE(reg.hasInput("p_bnd"));
    auto vals = reg.valuesOf("p_bnd");
    ASSERT_FALSE(vals.empty());
    // Node 0 is NOT on the boundary face, so only 200+300+400 = 900.
    EXPECT_NEAR(vals[0], 900.0, 1e-10);
}

// ---------------------------------------------------------------------------
//  Guardrail tests for FE-coupled helpers
// ---------------------------------------------------------------------------

TEST(FormsInstaller, RegisterSampledFieldInput_ThrowsBeforeSetup)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    // Do NOT call setup().

    EXPECT_THROW(
        sys.registerSampledFieldInput("u_sampled", "u", 4),
        InvalidStateException);
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_ThrowsBeforeSetup)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(1);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});

    EXPECT_THROW(
        sys.registerBoundaryNodalSumInput("p_bnd", "p", 1),
        InvalidStateException);
}

TEST(FormsInstaller, RegisterSampledFieldInput_RejectsNonVertexSpace)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    // HCurl space has edge DOFs, no vertex DOFs.
    auto hcurl = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto e_field = sys.addField(FieldSpec{.name = "E", .space = hcurl, .components = 3});
    sys.addOperator("op");

    // Install a dummy form so setup() builds DOF handlers.
    const auto u = forms::FormExpr::stateField(e_field, *hcurl, "E");
    const auto v = forms::FormExpr::testFunction(*hcurl, "v");
    installFormulation(sys, "op", {e_field}, forms::inner(u, v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // HCurl space should have no vertex DOFs → rejection.
    EXPECT_THROW(
        sys.registerSampledFieldInput("E_sampled", "E", 4),
        InvalidArgumentException);
}

TEST(FormsInstaller, RegisterBoundaryNodalSumInput_RejectsNonVertexSpace)
{
    using namespace svmp::FE;
    using namespace svmp::FE::systems;

    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(1);
    auto hcurl = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    const auto e_field = sys.addField(FieldSpec{.name = "E", .space = hcurl, .components = 3});
    sys.addOperator("op");

    const auto u = forms::FormExpr::stateField(e_field, *hcurl, "E");
    const auto v = forms::FormExpr::testFunction(*hcurl, "v");
    installFormulation(sys, "op", {e_field}, forms::inner(u, v).dx());

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_THROW(
        sys.registerBoundaryNodalSumInput("E_bnd", "E", 1),
        InvalidArgumentException);
}

TEST(FormsInstaller, FieldToFieldOperator_Rejected)
{
    using namespace svmp::FE::systems;

    // Different field names.
    EXPECT_THROW(
        AuxiliaryOperatorBuilder("bad_op")
            .source("field:velocity")
            .target("field:pressure")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build(),
        std::invalid_argument);

    // Same field name — also rejected (not misclassified as AuxSelf).
    EXPECT_THROW(
        AuxiliaryOperatorBuilder("bad_self")
            .source("field:u")
            .target("field:u")
            .residual([](const AuxiliaryOperatorContext&, std::span<Real>) {})
            .build(),
        std::invalid_argument);
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
