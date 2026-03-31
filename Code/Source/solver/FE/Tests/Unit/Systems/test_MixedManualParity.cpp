/**
 * @file test_MixedManualParity.cpp
 * @brief Parity tests: mixed expression vs manual block decomposition
 *
 * Phase 3 acceptance criterion: an equivalent operator assembled from manual
 * block decomposition and one mixed source expression produces the same
 * registered block structure in Systems, and the same assembly results.
 */

#include <gtest/gtest.h>

#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FormsInstallerDetail.h"

#include "Assembly/AssemblyKernel.h"

#include "Forms/BlockForm.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MixedFormIR.h"
#include "Forms/MonolithicCellKernel.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace {

/// Count semantic cell blocks, expanding aggregate mixed-cell kernels into their owned blocks.
std::size_t countSemanticCellTerms(const svmp::FE::systems::OperatorDefinition& def) {
    std::size_t count = 0;
    for (const auto& ct : def.cells) {
        if (!ct.kernel) {
            continue;
        }
        if (const auto* monolithic =
                dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(ct.kernel.get())) {
            count += monolithic->numBlocks();
        } else if (const auto* mixed_block =
                       dynamic_cast<const svmp::FE::forms::MixedBlockKernelSet*>(ct.kernel.get())) {
            count += mixed_block->numBlocks();
        } else {
            ++count;
        }
    }
    return count;
}

/// Extract semantic (test_field, trial_field) pairs, expanding MonolithicCellKernel blocks.
std::vector<std::pair<int, int>> semanticCellPairs(const svmp::FE::systems::OperatorDefinition& def) {
    std::vector<std::pair<int, int>> pairs;
    for (const auto& ct : def.cells) {
        if (!ct.kernel) {
            continue;
        }
        if (const auto* monolithic =
                dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(ct.kernel.get())) {
            for (std::size_t i = 0; i < monolithic->numBlocks(); ++i) {
                const auto& block = monolithic->blockSpec(i);
                pairs.emplace_back(static_cast<int>(block.test_field), static_cast<int>(block.trial_field));
            }
        } else if (const auto* mixed_block =
                       dynamic_cast<const svmp::FE::forms::MixedBlockKernelSet*>(ct.kernel.get())) {
            for (std::size_t i = 0; i < mixed_block->numBlocks(); ++i) {
                const auto& block = mixed_block->blockSpec(i);
                pairs.emplace_back(static_cast<int>(block.test_field), static_cast<int>(block.trial_field));
            }
        } else {
            pairs.emplace_back(static_cast<int>(ct.test_field), static_cast<int>(ct.trial_field));
        }
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

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
// Structural parity: operator registry has identical terms
// ============================================================================

TEST(MixedManualParity, BilinearStructuralParity_CellTerms)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // --- Manual block decomposition ---
    svmp::FE::systems::FESystem sys_manual(mesh);
    const auto u_m = sys_manual.addField({.name = "u", .space = space, .components = 1});
    const auto p_m = sys_manual.addField({.name = "p", .space = space, .components = 1});
    sys_manual.addOperator("op");

    {
        auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
        auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
        auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
        auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

        svmp::FE::forms::BlockBilinearForm blocks(2, 2);
        blocks.setBlock(0, 0, (u * v).dx());
        blocks.setBlock(0, 1, (p * v).dx());
        blocks.setBlock(1, 0, (u * q).dx());

        const std::array fields_m = {u_m, p_m};
        svmp::FE::systems::installResidualBlocks(
            sys_manual, "op",
            std::span<const FieldId>(fields_m),
            std::span<const FieldId>(fields_m),
            blocks);
    }

    // --- Mixed expression ---
    svmp::FE::systems::FESystem sys_mixed(mesh);
    const auto u_x = sys_mixed.addField({.name = "u", .space = space, .components = 1});
    const auto p_x = sys_mixed.addField({.name = "p", .space = space, .components = 1});
    sys_mixed.addOperator("op");

    {
        auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
        auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
        auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
        auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

        auto mixed = (u * v).dx() + (p * v).dx() + (u * q).dx();
        const std::array fields_x = {u_x, p_x};
        svmp::FE::systems::installMixedBilinear(
            sys_mixed, "op",
            std::span<const FieldId>(fields_x),
            std::span<const FieldId>(fields_x),
            mixed);
    }

    // --- Compare operator structure ---
    // Compare semantic block structure, not the concrete runtime wrapper count.
    // The mixed path may use one MonolithicCellKernel instead of several per-block terms.
    const auto& def_m = sys_manual.operatorDefinition("op");
    const auto& def_x = sys_mixed.operatorDefinition("op");

    // Same number of semantic cell terms
    EXPECT_EQ(countSemanticCellTerms(def_m), countSemanticCellTerms(def_x));

    // Same number of boundary/interior/interface terms (all zero here)
    EXPECT_EQ(def_m.boundary.size(), def_x.boundary.size());
    EXPECT_EQ(def_m.interior.size(), def_x.interior.size());
    EXPECT_EQ(def_m.interface_faces.size(), def_x.interface_faces.size());

    // Same (test_field, trial_field) pairs for the semantic cell blocks
    const auto manual_pairs = semanticCellPairs(def_m);
    const auto mixed_pairs = semanticCellPairs(def_x);
    ASSERT_EQ(manual_pairs.size(), mixed_pairs.size());

    for (std::size_t i = 0; i < manual_pairs.size(); ++i) {
        EXPECT_EQ(manual_pairs[i], mixed_pairs[i])
            << "Cell term " << i << ": manual=("
            << manual_pairs[i].first << "," << manual_pairs[i].second
            << ") mixed=(" << mixed_pairs[i].first << "," << mixed_pairs[i].second << ")";
    }
}

TEST(MixedManualParity, BilinearStructuralParity_BoundaryTerms)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(5);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // Use two distinct test functions so compileMixed sees 2 test spaces
    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    // --- Manual ---
    svmp::FE::systems::FESystem sys_manual(mesh);
    const auto u_m = sys_manual.addField({.name = "u", .space = space, .components = 1});
    const auto p_m = sys_manual.addField({.name = "p", .space = space, .components = 1});
    sys_manual.addOperator("op");

    {
        svmp::FE::forms::BlockBilinearForm blocks(2, 2);
        blocks.setBlock(0, 0, (u * v).dx());
        blocks.setBlock(0, 1, (p * v).ds(5));
        blocks.setBlock(1, 0, (u * q).dx());

        const std::array fields_m = {u_m, p_m};
        svmp::FE::systems::installResidualBlocks(
            sys_manual, "op",
            std::span<const FieldId>(fields_m),
            std::span<const FieldId>(fields_m),
            blocks);
    }

    // --- Mixed ---
    svmp::FE::systems::FESystem sys_mixed(mesh);
    const auto u_x = sys_mixed.addField({.name = "u", .space = space, .components = 1});
    const auto p_x = sys_mixed.addField({.name = "p", .space = space, .components = 1});
    sys_mixed.addOperator("op");

    {
        // Mixed form with cell + boundary terms across 2 test functions
        auto mixed = (u * v).dx() + (p * v).ds(5) + (u * q).dx();
        const std::array fields_x = {u_x, p_x};
        svmp::FE::systems::installMixedBilinear(
            sys_mixed, "op",
            std::span<const FieldId>(fields_x),
            std::span<const FieldId>(fields_x),
            mixed);
    }

    const auto& def_m = sys_manual.operatorDefinition("op");
    const auto& def_x = sys_mixed.operatorDefinition("op");

    // Same per-block cell and boundary term counts
    EXPECT_EQ(countSemanticCellTerms(def_m), countSemanticCellTerms(def_x));
    EXPECT_EQ(def_m.boundary.size(), def_x.boundary.size());
    if (!def_m.boundary.empty() && !def_x.boundary.empty()) {
        EXPECT_EQ(def_m.boundary[0].marker, def_x.boundary[0].marker);
    }
}

TEST(MixedManualParity, BilinearStructuralParity_ZeroBlockElimination)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // --- Manual: only diagonal blocks ---
    svmp::FE::systems::FESystem sys_manual(mesh);
    const auto u_m = sys_manual.addField({.name = "u", .space = space, .components = 1});
    const auto p_m = sys_manual.addField({.name = "p", .space = space, .components = 1});
    sys_manual.addOperator("op");

    {
        auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
        auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
        auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
        auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

        svmp::FE::forms::BlockBilinearForm blocks(2, 2);
        blocks.setBlock(0, 0, (u * v).dx());
        blocks.setBlock(1, 1, (p * q).dx());

        const std::array fields_m = {u_m, p_m};
        svmp::FE::systems::installResidualBlocks(
            sys_manual, "op",
            std::span<const FieldId>(fields_m),
            std::span<const FieldId>(fields_m),
            blocks);
    }

    // --- Mixed: naturally zero off-diagonals ---
    svmp::FE::systems::FESystem sys_mixed(mesh);
    const auto u_x = sys_mixed.addField({.name = "u", .space = space, .components = 1});
    const auto p_x = sys_mixed.addField({.name = "p", .space = space, .components = 1});
    sys_mixed.addOperator("op");

    {
        auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
        auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
        auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
        auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

        auto mixed = (u * v).dx() + (p * q).dx();
        const std::array fields_x = {u_x, p_x};
        svmp::FE::systems::installMixedBilinear(
            sys_mixed, "op",
            std::span<const FieldId>(fields_x),
            std::span<const FieldId>(fields_x),
            mixed);
    }

    const auto& def_m = sys_manual.operatorDefinition("op");
    const auto& def_x = sys_mixed.operatorDefinition("op");

    // Semantic cell blocks: 2 each (VV and PP, no off-diagonals)
    EXPECT_EQ(countSemanticCellTerms(def_m), 2u);
    EXPECT_EQ(countSemanticCellTerms(def_x), 2u);
}

// ============================================================================
// Assembly value parity: assembled matrices match
// ============================================================================

TEST(MixedManualParity, BilinearAssemblyParity_MatrixValues)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    svmp::FE::forms::FormCompiler compiler;

    auto build_and_assemble = [&](auto install_fn) {
        svmp::FE::systems::FESystem sys(mesh);
        const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
        const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
        sys.addOperator("op");
        install_fn(sys, u_f, p_f);

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
        return mat;
    };

    // Manual: compile each block as bilinear, install via installMixedFormIR
    auto mat_manual = build_and_assemble([&](svmp::FE::systems::FESystem& sys,
                                              FieldId u_f, FieldId p_f) {
        svmp::FE::forms::MixedFormIR mir(2, 2);
        mir.setKind(svmp::FE::forms::FormKind::Bilinear);
        mir.setBlock(0, 0, compiler.compileBilinear((u * v).dx()));
        mir.setBlock(0, 1, compiler.compileBilinear((p * v).dx()));
        mir.setBlock(1, 0, compiler.compileBilinear((u * q).dx()));

        const std::array fields = {u_f, p_f};
        svmp::FE::systems::installMixedFormIR(
            sys, "op",
            std::span<const FieldId>(fields),
            std::span<const FieldId>(fields),
            mir);
    });

    // Mixed: single expression, compile + install
    auto mat_mixed = build_and_assemble([&](svmp::FE::systems::FESystem& sys,
                                             FieldId u_f, FieldId p_f) {
        auto mixed = (u * v).dx() + (p * v).dx() + (u * q).dx();
        const std::array fields = {u_f, p_f};
        svmp::FE::systems::installMixedBilinear(
            sys, "op",
            std::span<const FieldId>(fields),
            std::span<const FieldId>(fields),
            mixed);
    });

    // Compare all entries
    ASSERT_EQ(mat_manual.numRows(), mat_mixed.numRows());
    const auto n = mat_manual.numRows();
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            EXPECT_NEAR(mat_manual.getMatrixEntry(i, j), mat_mixed.getMatrixEntry(i, j), 1e-14)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// installMixedLinear end-to-end test
// ============================================================================

TEST(MixedManualParity, InstallMixedLinear_InstallsKernels)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);

    auto linear = (f * v).dx();

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const std::array fields = {u_f};
    auto kernels = svmp::FE::systems::installMixedLinear(
        sys, "op", std::span<const FieldId>(fields), linear);

    // Should have installed one kernel for the single test field
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_NE(kernels[0], nullptr);

    // Operator should have at least one cell term
    const auto& def = sys.operatorDefinition("op");
    EXPECT_GE(def.cells.size(), 1u);
}

// ============================================================================
// block_couplings after installFormulation on StateField-based mixed residual
// ============================================================================

TEST(MixedManualParity, BlockCouplings_StateFieldResidual_NotDense)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto u_f = sys->addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys->addField({.name = "p", .space = space, .components = 1});
    sys->addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");

    // Residual: momentum depends on u and p; continuity depends on u only
    auto residual = (u * v + p * v).dx() + (u * q).dx();

    svmp::FE::systems::installFormulation(*sys, "op", {u_f, p_f}, residual);

    // Check FormulationRecord block_couplings
    const auto& recs = sys->formulationRecords();
    ASSERT_FALSE(recs.empty());
    const auto& bc = recs[0].block_couplings;

    // Should NOT be dense 4 couplings (2x2). Continuity (q) doesn't depend on p,
    // so (p_f, p_f) coupling should be absent.
    bool has_pp = false;
    for (const auto& [test, trial] : bc) {
        if (test == p_f && trial == p_f) has_pp = true;
    }
    EXPECT_FALSE(has_pp) << "block_couplings should not include PP coupling when "
                            "continuity equation only depends on u";

    // VV, VP, PV should be present
    bool has_vv = false, has_vp = false, has_pv = false;
    for (const auto& [test, trial] : bc) {
        if (test == u_f && trial == u_f) has_vv = true;
        if (test == u_f && trial == p_f) has_vp = true;
        if (test == p_f && trial == u_f) has_pv = true;
    }
    EXPECT_TRUE(has_vv);
    EXPECT_TRUE(has_vp);
    EXPECT_TRUE(has_pv);
}

// ============================================================================
// Duplicate test names on multi-field residual path
// ============================================================================

// ============================================================================
// Multi-field linear installation
// ============================================================================

TEST(MixedManualParity, InstallMixedLinear_MultiField)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_a = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto space_b = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto v = svmp::FE::forms::FormExpr::testFunction(*space_a, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space_b, "q");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);
    auto g = svmp::FE::forms::FormExpr::constant(2.0);

    auto linear = (f * v).dx() + (g * q).dx();

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space_a, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space_b, .components = 1});
    sys.addOperator("op");

    const std::array fields = {u_f, p_f};
    auto kernels = svmp::FE::systems::installMixedLinear(
        sys, "op", std::span<const FieldId>(fields), linear);

    ASSERT_EQ(kernels.size(), 2u);
    EXPECT_NE(kernels[0], nullptr);
    EXPECT_NE(kernels[1], nullptr);

    // Both kernels should be vector-only (linear forms produce no matrix)
    EXPECT_TRUE(kernels[0]->isVectorOnly());
    EXPECT_TRUE(kernels[1]->isVectorOnly());
}

// ============================================================================
// Pure-source row produces no block_couplings
// ============================================================================

TEST(MixedManualParity, BlockCouplings_PartialDependency_Sparse)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto u_f = sys->addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys->addField({.name = "p", .space = space, .components = 1});
    sys->addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");

    // Momentum depends on u and p; continuity depends on u only (not p)
    auto residual = (u * v + p * v).dx() + (u * q).dx();

    svmp::FE::systems::installFormulation(*sys, "op", {u_f, p_f}, residual);

    const auto& recs = sys->formulationRecords();
    ASSERT_FALSE(recs.empty());
    const auto& bc = recs[0].block_couplings;

    // VV, VP should be present (momentum depends on u and p)
    // PV should be present (continuity depends on u)
    // PP should be ABSENT (continuity does NOT depend on p)
    bool has_vv = false, has_vp = false, has_pv = false, has_pp = false;
    for (const auto& [test, trial] : bc) {
        if (test == u_f && trial == u_f) has_vv = true;
        if (test == u_f && trial == p_f) has_vp = true;
        if (test == p_f && trial == u_f) has_pv = true;
        if (test == p_f && trial == p_f) has_pp = true;
    }
    EXPECT_TRUE(has_vv);
    EXPECT_TRUE(has_vp);
    EXPECT_TRUE(has_pv);
    EXPECT_FALSE(has_pp) << "block_couplings should not include PP when "
                            "continuity equation only depends on u";
}

// ============================================================================
// Duplicate-name rejection happens before FormulationRecord creation
// ============================================================================

// ============================================================================
// Pure-source residual row installs correctly (no crash, no state left behind)
// ============================================================================

// ============================================================================
// Single-field source-only residual installs as linear kernel
// ============================================================================

TEST(MixedManualParity, SingleFieldSourceOnly_Installs)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);

    // Source-only residual: no StateField, just f*v
    auto residual = (f * v).dx();

    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual));

    // Should have installed a cell term
    const auto& def = sys.operatorDefinition("op");
    EXPECT_GE(def.cells.size(), 1u);

    // The kernel should be vector-only (no matrix contribution)
    bool has_vector_only = false;
    for (const auto& ct : def.cells) {
        if (ct.kernel && ct.kernel->isVectorOnly()) has_vector_only = true;
    }
    EXPECT_TRUE(has_vector_only) << "Source-only residual should install a vector-only kernel";

    // FormulationRecord: block_couplings should be empty (no Jacobian)
    const auto& recs = sys.formulationRecords();
    ASSERT_FALSE(recs.empty());
    EXPECT_TRUE(recs[0].block_couplings.empty())
        << "Source-only residual should have no block couplings";
}

// ============================================================================
// Single-field TrialFunction residual (non-StateField) has correct metadata
// ============================================================================

TEST(MixedManualParity, SingleFieldTrialFunction_HasCoupling)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Residual authored with TrialFunction (not StateField)
    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    auto result = svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    // Should have a real Jacobian kernel
    ASSERT_EQ(result.jacobian_blocks.size(), 1u);
    ASSERT_EQ(result.jacobian_blocks[0].size(), 1u);
    EXPECT_NE(result.jacobian_blocks[0][0], nullptr)
        << "TrialFunction residual should have a Jacobian block";

    // FormulationRecord should have the self-coupling
    const auto& recs = sys.formulationRecords();
    ASSERT_FALSE(recs.empty());
    EXPECT_FALSE(recs[0].block_couplings.empty())
        << "TrialFunction residual should have block_couplings";
}

// ============================================================================

// ============================================================================
// End-to-end: installFormulation → analysis contributions for TrialFunction
// ============================================================================

TEST(MixedManualParity, EndToEnd_TrialFunction_AnalysisContributions)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto residual = (u * v).dx();

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    const auto& contribs = sys.contributionDescriptors();
    // Exactly one contribution: the (u, u) diagonal block
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_FALSE(contribs[0].trial_variables.empty())
        << "TrialFunction residual must produce a contribution with trial variables";
    // Must NOT be classified as source
    EXPECT_EQ(contribs[0].origin.find("source"), std::string::npos)
        << "TrialFunction residual must not be classified as source-only";
}

TEST(MixedManualParity, EndToEnd_SourceOnly_AnalysisContributions)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);
    auto residual = (f * v).dx();

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    const auto& contribs = sys.contributionDescriptors();
    // Exactly one contribution: the source term
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_TRUE(contribs[0].trial_variables.empty())
        << "Source-only residual must produce a contribution with empty trial variables";
    EXPECT_NE(contribs[0].origin.find("source"), std::string::npos)
        << "Source-only residual must be classified as source";
}

TEST(MixedManualParity, EndToEnd_MixedResidual_AnalysisContributions)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto u_f = sys->addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys->addField({.name = "p", .space = space, .components = 1});
    sys->addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");

    // Momentum (u,p) + continuity (u only)
    auto residual = (u * v + p * v).dx() + (u * q).dx();

    svmp::FE::systems::installFormulation(*sys, "op", {u_f, p_f}, residual);

    const auto& contribs = sys->contributionDescriptors();

    // Classify each contribution by (test, trial) field pair
    using svmp::FE::analysis::VariableKind;
    bool has_vv = false, has_vp = false, has_pv = false, has_pp = false;
    for (const auto& c : contribs) {
        FieldId test_fid = svmp::FE::INVALID_FIELD_ID;
        FieldId trial_fid = svmp::FE::INVALID_FIELD_ID;
        for (const auto& tv : c.test_variables) {
            if (tv.kind == VariableKind::FieldComponent) test_fid = tv.field_id;
        }
        for (const auto& tv : c.trial_variables) {
            if (tv.kind == VariableKind::FieldComponent) trial_fid = tv.field_id;
        }
        if (test_fid == u_f && trial_fid == u_f) has_vv = true;
        if (test_fid == u_f && trial_fid == p_f) has_vp = true;
        if (test_fid == p_f && trial_fid == u_f) has_pv = true;
        if (test_fid == p_f && trial_fid == p_f) has_pp = true;
    }

    EXPECT_TRUE(has_vv) << "Expected VV contribution";
    EXPECT_TRUE(has_vp) << "Expected VP contribution";
    EXPECT_TRUE(has_pv) << "Expected PV contribution";
    EXPECT_FALSE(has_pp) << "PP contribution should be absent (continuity does not depend on p)";

    // All contributions should have block_context with field names
    for (const auto& c : contribs) {
        EXPECT_FALSE(c.block_context.empty())
            << "Every contribution should have non-empty block_context";
        // block_context should contain a field name (u or p)
        EXPECT_TRUE(c.block_context.find("u") != std::string::npos ||
                     c.block_context.find("p") != std::string::npos)
            << "block_context should mention field names, got: " << c.block_context;
    }
}

// ============================================================================

TEST(MixedManualParity, SingleFieldSourceOnly_ReturnValue)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);
    auto residual = (f * v).dx();

    auto result = svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    // residual[0] should be the vector-only kernel
    ASSERT_EQ(result.residual.size(), 1u);
    EXPECT_NE(result.residual[0], nullptr);

    // jacobian_blocks[0][0] should be nullptr (no Jacobian for source-only)
    ASSERT_EQ(result.jacobian_blocks.size(), 1u);
    ASSERT_EQ(result.jacobian_blocks[0].size(), 1u);
    EXPECT_EQ(result.jacobian_blocks[0][0], nullptr)
        << "Source-only residual should not populate jacobian_blocks";
}

// ============================================================================
// Multi-field with one pure-source row
// ============================================================================

TEST(MixedManualParity, PureSourceResidualRow_Installs)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto u_f = sys->addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys->addField({.name = "p", .space = space, .components = 1});
    sys->addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");
    auto f = svmp::FE::forms::FormExpr::constant(1.0);

    // Momentum depends on u; continuity is pure source (f*q)
    auto residual = (u * v).dx() + (f * q).dx();

    // Should not throw — pure source row installed as linear kernel
    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(*sys, "op", {u_f, p_f}, residual));

    // FormulationRecord should be present (committed after success)
    EXPECT_FALSE(sys->formulationRecords().empty());

    // Operator should have cell terms (at least the momentum Jacobian block
    // plus the pure-source linear kernel for continuity)
    const auto& def = sys->operatorDefinition("op");
    EXPECT_GE(def.cells.size(), 1u);
}

// ============================================================================

TEST(MixedManualParity, DuplicateTestNames_NoSideEffects)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_a = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto space_b = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto f1 = sys.addField({.name = "u", .space = space_a, .components = 1});
    const auto f2 = sys.addField({.name = "p", .space = space_b, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(f1, *space_a, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(f2, *space_b, "p");
    auto v1 = svmp::FE::forms::FormExpr::testFunction(*space_a, "v");
    auto v2 = svmp::FE::forms::FormExpr::testFunction(*space_b, "v");

    auto residual = (u * v1).dx() + (p * v2).dx();

    // Should throw before any side effects
    EXPECT_THROW(
        svmp::FE::systems::installFormulation(sys, "op", {f1, f2}, residual),
        svmp::FE::InvalidArgumentException);

    // No FormulationRecord should have been added
    EXPECT_TRUE(sys.formulationRecords().empty())
        << "Failed installFormulation should not leave a FormulationRecord";
}

// ============================================================================

TEST(MixedManualParity, DuplicateTestNames_ResidualPath_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_a = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto space_b = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto f1 = sys.addField({.name = "u", .space = space_a, .components = 1});
    const auto f2 = sys.addField({.name = "p", .space = space_b, .components = 1});
    sys.addOperator("op");

    // Two test functions named "v" on DIFFERENT spaces
    auto u = svmp::FE::forms::FormExpr::stateField(f1, *space_a, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(f2, *space_b, "p");
    auto v1 = svmp::FE::forms::FormExpr::testFunction(*space_a, "v");
    auto v2 = svmp::FE::forms::FormExpr::testFunction(*space_b, "v");

    auto residual = (u * v1).dx() + (p * v2).dx();

    // installFormulation routes through installCoupledResidualMixed which should
    // reject the duplicate name
    EXPECT_THROW(
        svmp::FE::systems::installFormulation(sys, "op", {f1, f2}, residual),
        svmp::FE::InvalidArgumentException);
}

// ============================================================================

TEST(MixedManualParity, MixedFormIR_ProducesIdenticalStructure)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    auto mixed_expr = (u * v).dx() + (p * v).dx() + (u * q).dx();

    // Compile to MixedFormIR
    svmp::FE::forms::FormCompiler compiler;
    auto mir = compiler.compileMixed(mixed_expr, svmp::FE::forms::FormKind::Bilinear);

    EXPECT_EQ(mir.numTestFields(), 2u);
    EXPECT_EQ(mir.numTrialFields(), 2u);
    EXPECT_EQ(mir.numActiveBlocks(), 3u);

    // Install MixedFormIR directly
    svmp::FE::systems::FESystem sys_mir(mesh);
    const auto u_f = sys_mir.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys_mir.addField({.name = "p", .space = space, .components = 1});
    sys_mir.addOperator("op");

    {
        const std::array fields = {u_f, p_f};
        svmp::FE::systems::installMixedFormIR(
            sys_mir, "op",
            std::span<const FieldId>(fields),
            std::span<const FieldId>(fields),
            mir);
    }

    // Install via installMixedBilinear
    svmp::FE::systems::FESystem sys_bilinear(mesh);
    const auto u_f2 = sys_bilinear.addField({.name = "u", .space = space, .components = 1});
    const auto p_f2 = sys_bilinear.addField({.name = "p", .space = space, .components = 1});
    sys_bilinear.addOperator("op");

    {
        const std::array fields = {u_f2, p_f2};
        svmp::FE::systems::installMixedBilinear(
            sys_bilinear, "op",
            std::span<const FieldId>(fields),
            std::span<const FieldId>(fields),
            mixed_expr);
    }

    const auto& def_mir = sys_mir.operatorDefinition("op");
    const auto& def_bilinear = sys_bilinear.operatorDefinition("op");

    // Both paths should have the same semantic cell blocks.
    EXPECT_EQ(countSemanticCellTerms(def_mir), countSemanticCellTerms(def_bilinear));
    EXPECT_EQ(def_mir.boundary.size(), def_bilinear.boundary.size());
}

// ============================================================================
// Same-space multi-field residuals: field-bound test functions
// ============================================================================

// ============================================================================
// BoundaryConditionManager::applyAll convenience test
// ============================================================================

TEST(MixedManualParity, BCManager_ApplyAll_EmptyManager)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto residual = (u * v).dx();

    // Empty manager — no BCs to apply
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    EXPECT_NO_THROW(bc_manager.applyAll(sys, residual, u, v, u_f));

    // Should still be installable
    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // No constraints
    EXPECT_FALSE(sys.constraints().isConstrained(0));
}

// ============================================================================
// Same-space multi-field residuals: field-bound test functions
// ============================================================================

// ============================================================================
// Transactional installation: rollback on failure
// ============================================================================

TEST(MixedManualParity, Transactional_FailedInstall_NoPartialState)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    // Install a valid single-field residual first to establish a baseline
    {
        auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
        auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
        auto residual = (u * v).dx();
        svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);
    }

    const auto& def_before = sys.operatorDefinition("op");
    const auto cell_count_before = def_before.cells.size();
    EXPECT_GE(cell_count_before, 1u);

    // Now attempt a second install that FAILS (duplicate test names, same space)
    {
        auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
        auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
        // Two test functions named "v" on DIFFERENT order spaces → caught by
        // the early validation check. Use same-space without bindings instead.
        auto v = svmp::FE::forms::FormExpr::testFunction(*space, "w");
        auto q = svmp::FE::forms::FormExpr::testFunction(*space, "r");

        auto bad_residual = (u * v + p * v).dx() + (u * q).dx();

        // This should throw because two unbound test functions match two
        // same-space fields ambiguously.
        EXPECT_THROW(
            svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, bad_residual),
            svmp::FE::InvalidArgumentException);
    }

    // After the failed install, the operator should be unchanged
    const auto& def_after = sys.operatorDefinition("op");
    EXPECT_EQ(def_after.cells.size(), cell_count_before)
        << "Failed install should not leave partial operator state behind";

    // Analysis metadata should also be unchanged
    // (Only the first successful install should have a record)
    EXPECT_EQ(sys.formulationRecords().size(), 1u)
        << "Failed install should not add a FormulationRecord";
}

TEST(MixedManualParity, Transactional_MixedBilinear_Rollback)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto& def_before = sys.operatorDefinition("op");
    EXPECT_EQ(def_before.cells.size(), 0u);

    // Attempt installMixedBilinear with an invalid form (no test function)
    auto bad_form = svmp::FE::forms::FormExpr::constant(1.0).dx();
    const std::array fields = {u_f};
    EXPECT_THROW(
        svmp::FE::systems::installMixedBilinear(
            sys, "op",
            std::span<const FieldId>(fields),
            std::span<const FieldId>(fields),
            bad_form),
        std::invalid_argument);

    // After failure, operator should still be empty
    const auto& def_after = sys.operatorDefinition("op");
    EXPECT_EQ(def_after.cells.size(), 0u)
        << "Failed installMixedBilinear should leave operator unchanged";
}

// ============================================================================
// Same-space multi-field residuals: field-bound test functions
// ============================================================================

TEST(MixedManualParity, SameSpace_TwoField_WithFieldBindings)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto T_f = sys->addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys->addField({.name = "C", .space = space, .components = 1});
    sys->addOperator("op");

    // Two fields on the SAME space — use field-bound test functions
    auto T = svmp::FE::forms::FormExpr::stateField(T_f, *space, "T");
    auto C = svmp::FE::forms::FormExpr::stateField(C_f, *space, "C");
    auto w = svmp::FE::forms::FormExpr::testFunction(T_f, *space, "w");
    auto r = svmp::FE::forms::FormExpr::testFunction(C_f, *space, "r");

    auto residual = (inner(grad(T), grad(w))).dx() + (inner(grad(C), grad(r))).dx();

    // Should install without ambiguity
    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(*sys, "op", {T_f, C_f}, residual));

    // Both fields should have block couplings
    const auto& recs = sys->formulationRecords();
    ASSERT_FALSE(recs.empty());

    bool has_TT = false, has_CC = false;
    for (const auto& [test, trial] : recs[0].block_couplings) {
        if (test == T_f && trial == T_f) has_TT = true;
        if (test == C_f && trial == C_f) has_CC = true;
    }
    EXPECT_TRUE(has_TT) << "Expected TT coupling";
    EXPECT_TRUE(has_CC) << "Expected CC coupling";
}

TEST(MixedManualParity, SameSpace_TwoField_WithoutBindings_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto T_f = sys.addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys.addField({.name = "C", .space = space, .components = 1});
    sys.addOperator("op");

    // Two fields on the SAME space — WITHOUT field bindings
    auto T = svmp::FE::forms::FormExpr::stateField(T_f, *space, "T");
    auto C = svmp::FE::forms::FormExpr::stateField(C_f, *space, "C");
    auto w = svmp::FE::forms::FormExpr::testFunction(*space, "w");
    auto r = svmp::FE::forms::FormExpr::testFunction(*space, "r");

    auto residual = (inner(grad(T), grad(w))).dx() + (inner(grad(C), grad(r))).dx();

    // Should throw because the installer can't determine which test goes to which field
    EXPECT_THROW(
        svmp::FE::systems::installFormulation(sys, "op", {T_f, C_f}, residual),
        svmp::FE::InvalidArgumentException);
}

TEST(MixedManualParity, SameSpace_ThreeField_WithFieldBindings)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto a_f = sys->addField({.name = "a", .space = space, .components = 1});
    const auto b_f = sys->addField({.name = "b", .space = space, .components = 1});
    const auto c_f = sys->addField({.name = "c", .space = space, .components = 1});
    sys->addOperator("op");

    auto a = svmp::FE::forms::FormExpr::stateField(a_f, *space, "a");
    auto b = svmp::FE::forms::FormExpr::stateField(b_f, *space, "b");
    auto c = svmp::FE::forms::FormExpr::stateField(c_f, *space, "c");
    auto va = svmp::FE::forms::FormExpr::testFunction(a_f, *space, "va");
    auto vb = svmp::FE::forms::FormExpr::testFunction(b_f, *space, "vb");
    auto vc = svmp::FE::forms::FormExpr::testFunction(c_f, *space, "vc");

    auto residual = (a * va).dx() + (b * vb).dx() + (c * vc).dx();

    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(*sys, "op", {a_f, b_f, c_f}, residual));

    const auto& recs = sys->formulationRecords();
    ASSERT_FALSE(recs.empty());

    // All three diagonal couplings should be present
    bool has_aa = false, has_bb = false, has_cc = false;
    for (const auto& [test, trial] : recs[0].block_couplings) {
        if (test == a_f && trial == a_f) has_aa = true;
        if (test == b_f && trial == b_f) has_bb = true;
        if (test == c_f && trial == c_f) has_cc = true;
    }
    EXPECT_TRUE(has_aa);
    EXPECT_TRUE(has_bb);
    EXPECT_TRUE(has_cc);
}

// ============================================================================
// Expert manual path: MixedFormIR + installMixedFormIR (moved from
// test_CanonicalWorkflow.cpp to keep the canonical test file focused on
// the recommended StateField/TestField/installFormulation workflow)
// ============================================================================

TEST(MixedManualParity, ExpertPath_ManualBlocksStillWork)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // Use the expert path: manual MixedFormIR + installMixedFormIR
    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

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
