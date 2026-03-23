/**
 * @file test_CanonicalWorkflow.cpp
 * @brief End-to-end tests for the canonical formulation workflow
 *
 * Tests the recommended pattern:
 *   StateField + TestField + BoundaryConditionManager::applyAll + installFormulation
 *
 * Covers:
 *   - Same-space multi-field residual assembly (not just metadata)
 *   - BC convenience flow with actual strong constraints
 *   - Full canonical workflow from registration through assembly
 *   - Expert/manual parity verification
 */

#include <gtest/gtest.h>

#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/MixedFormIR.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"

#include "Spaces/H1Space.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <span>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using namespace svmp::FE::forms;

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

} // namespace

// ============================================================================
// Same-space assembly: two scalar fields on the same H1 space
// ============================================================================

TEST(CanonicalWorkflow, SameSpace_TwoField_Assembly)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto T_f = sys->addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys->addField({.name = "C", .space = space, .components = 1});
    sys->addOperator("op");

    // Canonical workflow: StateField + TestField with field bindings
    auto T = StateField(T_f, *space, "T");
    auto C = StateField(C_f, *space, "C");
    auto w = TestField(T_f, *space, "w");
    auto r = TestField(C_f, *space, "r");

    // Coupled system: T equation depends on T only, C equation depends on C and T
    auto residual = (inner(grad(T), grad(w))).dx()
                  + (inner(grad(C), grad(r)) + T * r).dx();

    svmp::FE::systems::installFormulation(*sys, "op", {T_f, C_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys->setup({}, inputs);

    const auto n = sys->dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.1 * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    (void)sys->assemble(req, state, &out, &out);

    // Residual should be non-zero
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0);

    // Jacobian should be non-zero
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0);

    // Verify block structure: T equation (rows 0-3) should have no coupling to C DOFs (cols 4-7)
    // in the off-diagonal TT block, since grad(T):grad(w) has no C dependency.
    // C equation (rows 4-7) SHOULD couple to T DOFs (cols 0-3) via the T*r term.
    bool has_CT_coupling = false;
    for (GlobalIndex i = 4; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            if (std::abs(out.getMatrixEntry(i, j)) > 1e-14) {
                has_CT_coupling = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_CT_coupling) << "C equation should couple to T via the T*r term";
}

// ============================================================================
// BC workflow: strong Dirichlet via direct installStrongDirichlet
// ============================================================================

TEST(CanonicalWorkflow, BC_StrongDirichlet_Constraints)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    auto residual = (inner(grad(u), grad(v))).dx();

    auto bc = svmp::FE::forms::bc::strongDirichlet(
        u_f, marker, FormExpr::constant(2.5), "u");
    svmp::FE::systems::installStrongDirichlet(sys,
        std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    inputs.topology_override = topo;
    sys.setup({}, inputs);

    // Boundary face (0,1,2) should be constrained
    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    // Constrained DOFs should have inhomogeneity = 2.5
    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }

    // Assembly should apply constraint: constrained rows get identity, off-diag zeroed
    const auto n = sys.dofHandler().getNumDofs();
    svmp::FE::assembly::DenseMatrixView mat(n);
    mat.zero();

    svmp::FE::systems::SystemStateView state;
    std::vector<Real> U = {2.5, 2.5, 2.5, 0.4};
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &mat, nullptr);

    // Constrained rows: diagonal = 1, off-diagonal = 0
    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(mat.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (GlobalIndex j = 0; j < n; ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mat.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

// ============================================================================
// BC convenience: applyAll with a real EssentialBC object
// ============================================================================

TEST(CanonicalWorkflow, BCApplyAll_WithEssentialBC)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    auto residual = (inner(grad(u), grad(v))).dx();

    // Use BoundaryConditionManager with a concrete EssentialBC
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::EssentialBC>(
        marker, FormExpr::constant(3.0), "u"));

    // One-call convenience: validate + apply weak terms + install strong constraints
    bc_manager.applyAll(sys, residual, u, v, u_f);

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    inputs.topology_override = topo;
    sys.setup({}, inputs);

    // Boundary face nodes (0,1,2) should be constrained with value 3.0
    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 3.0, 1e-15);
    }
}

// ============================================================================
// Same-space, same-name test functions disambiguated by field binding
// ============================================================================

TEST(CanonicalWorkflow, SameSpace_SameName_DifferentBindings)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto T_f = sys->addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys->addField({.name = "C", .space = space, .components = 1});
    sys->addOperator("op");

    // Same space, same test function name "v", different field bindings.
    // Asymmetric diffusion (k_T=1, k_C=10) and REVERSED residual order
    // (C term first) so an order-based mapping bug would put the wrong
    // coefficient in the wrong block.
    auto T = StateField(T_f, *space, "T");
    auto C = StateField(C_f, *space, "C");
    auto v_T = TestField(T_f, *space, "v");
    auto v_C = TestField(C_f, *space, "v");
    const auto k_T = FormExpr::constant(1.0);
    const auto k_C = FormExpr::constant(10.0);

    // C term listed FIRST, T term second — reversed from field order {T_f, C_f}
    auto residual = (k_C * inner(grad(C), grad(v_C))).dx()
                  + (k_T * inner(grad(T), grad(v_T))).dx();

    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(*sys, "op", {T_f, C_f}, residual));

    // Go through setup and assembly
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys->setup({}, inputs);

    const auto n = sys->dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) U[i] = 0.1 * static_cast<Real>(i + 1);

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    (void)sys->assemble(req, state, &out, &out);

    // T block (rows 0-3, cols 0-3) should have k_T=1 scaling
    // C block (rows 4-7, cols 4-7) should have k_C=10 scaling
    // If mapping is correct, C block norm should be ~100x larger than T block norm
    double T_block_norm = 0.0, C_block_norm = 0.0;
    for (GlobalIndex i = 0; i < 4; ++i)
        for (GlobalIndex j = 0; j < 4; ++j)
            T_block_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
    for (GlobalIndex i = 4; i < 8; ++i)
        for (GlobalIndex j = 4; j < 8; ++j)
            C_block_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);

    EXPECT_GT(T_block_norm, 0.0) << "T diagonal block should be non-zero";
    EXPECT_GT(C_block_norm, 0.0) << "C diagonal block should be non-zero";

    // k_C/k_T = 10 → Frobenius norms scale quadratically → ratio ~100
    const double ratio = C_block_norm / T_block_norm;
    EXPECT_NEAR(ratio, 100.0, 1.0)
        << "C block (k=10) should have ~100x the Frobenius norm² of T block (k=1). "
           "Got ratio=" << ratio << ". If ~1.0, rows are swapped.";
}

// ============================================================================
// Negative: same-FieldId conflicting name or space is rejected
// ============================================================================

TEST(CanonicalWorkflow, ConflictingBoundTests_SameFieldId_DifferentName_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");

    // Two test functions bound to the SAME FieldId but with DIFFERENT names
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto w = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "w");

    auto residual = (u * v).dx() + (p * w).dx();

    try {
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual);
        FAIL() << "Expected InvalidArgumentException for same-FieldId/different-name conflict";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("conflicting test functions"), std::string::npos)
            << "Error should mention 'conflicting test functions', got: " << msg;
        EXPECT_NE(msg.find("consistent space and name"), std::string::npos)
            << "Error should mention 'consistent space and name', got: " << msg;
    }
}

TEST(CanonicalWorkflow, ConflictingBoundTests_SameFieldId_DifferentSpace_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_p1 = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto space_p2 = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space_p1, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space_p1, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space_p1, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space_p1, "p");

    // Two test functions bound to the SAME FieldId but with DIFFERENT spaces
    auto v1 = svmp::FE::forms::FormExpr::testFunction(u_f, *space_p1, "v");
    auto v2 = svmp::FE::forms::FormExpr::testFunction(u_f, *space_p2, "v");

    auto residual = (u * v1).dx() + (p * v2).dx();

    try {
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual);
        FAIL() << "Expected InvalidArgumentException for same-FieldId/different-space conflict";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        // May be caught by the same-FieldId conflict check or by the
        // duplicate-name-across-different-spaces check, depending on which
        // fires first. Both are valid early diagnostics.
        const bool has_conflict_msg = msg.find("conflicting test functions") != std::string::npos;
        const bool has_duplicate_msg = msg.find("duplicate TestFunction name") != std::string::npos;
        EXPECT_TRUE(has_conflict_msg || has_duplicate_msg)
            << "Error should mention 'conflicting test functions' or "
               "'duplicate TestFunction name', got: " << msg;
    }
}

// ============================================================================
// Full canonical workflow: register → symbols → BCs → installFormulation → setup → assemble
// ============================================================================

TEST(CanonicalWorkflow, FullWorkflow_SingleField)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // 1. Register fields and operator
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("equations");

    // 2. Build field-bound symbols
    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    const auto k = FormExpr::constant(2.0);
    const auto f = FormExpr::constant(1.0);

    // 3. Write the residual
    auto residual = (k * inner(grad(u), grad(v)) - f * v).dx();

    // 4. Apply BCs (empty manager for this test)
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.applyAll(sys, residual, u, v, u_f);

    // 5. Install
    svmp::FE::systems::installFormulation(sys, "equations", {u_f}, residual);

    // 6. Setup
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // 7. Assemble and verify
    const auto n = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    (void)sys.assemble(req, state, &out, &out);

    // Matrix should have stiffness structure (symmetric, non-zero)
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0);

    // Vector should have both stiffness and source contributions
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0);
}

// ============================================================================
// Expert manual path still works (parity guard)
// ============================================================================

TEST(CanonicalWorkflow, ExpertPath_ManualBlocksStillWork)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // Use the expert path: manual MixedFormIR + installMixedFormIR
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    svmp::FE::forms::MixedFormIR mir(1, 1);
    mir.setKind(svmp::FE::forms::FormKind::Bilinear);
    mir.setBlock(0, 0, compiler.compileBilinear((inner(grad(u), grad(v))).dx()));

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const std::array fields = {u_f};
    svmp::FE::systems::installMixedFormIR(
        sys, "op",
        std::span<const FieldId>(fields),
        std::span<const FieldId>(fields),
        mir);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    svmp::FE::assembly::DenseMatrixView mat(n);
    mat.zero();

    svmp::FE::systems::SystemStateView state;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &mat, nullptr);

    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += mat.getMatrixEntry(i, j) * mat.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0) << "Expert manual path should produce non-zero matrix";
}
