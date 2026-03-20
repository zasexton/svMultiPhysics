/**
 * @file test_GaugeIntegration.cpp
 * @brief End-to-end tests for gauge/nullspace detection through FESystem setup
 *
 * Tests the full pipeline: installFormulation → NullspaceAnalyzer → GaugeRegistry
 * → SystemSetup resolution → automatic constraint creation.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/BoundaryConditionManager.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/FormExpr.h"
#include "Forms/StandardBCs.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Constraints/GaugeRegistry.h"
#include "Constraints/DirichletBC.h"

#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;
using namespace svmp::FE::gauge;

namespace {

dofs::MeshTopologyInfo singleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
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

/// Two disconnected tetrahedra: vertices 0-3 form tet A, vertices 4-7 form tet B.
/// No shared vertices → two connected components.
dofs::MeshTopologyInfo twoDisconnectedTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 8;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

/// Mesh access for two disconnected tetrahedra
class TwoDisconnectedTetraMeshAccess final : public assembly::IMeshAccess {
public:
    TwoDisconnectedTetraMeshAccess()
    {
        // Tet A: vertices 0-3 at origin
        nodes_ = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            // Tet B: vertices 4-7 offset by (5,0,0)
            {5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, {5.0, 1.0, 0.0}, {5.0, 0.0, 1.0}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override {
        std::vector<GlobalIndex> nds;
        getCellNodes(cell_id, nds);
        coords.resize(nds.size());
        for (std::size_t i = 0; i < nds.size(); ++i)
            coords[i] = nodes_.at(static_cast<std::size_t>(nds[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
};

/// Two disconnected tetrahedra with one boundary face on tet A (marker 1).
/// Face 0 is on cell 0 (tet A), so Robin BC on marker 1 only touches region A.
class TwoTetraOneBoundaryMeshAccess final : public assembly::IMeshAccess {
public:
    TwoTetraOneBoundaryMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
            {5.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, {5.0, 1.0, 0.0}, {5.0, 0.0, 1.0}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        if (cell_id == 0) nodes = {0, 1, 2, 3};
        else              nodes = {4, 5, 6, 7};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override {
        std::vector<GlobalIndex> nds;
        getCellNodes(cell_id, nds);
        coords.resize(nds.size());
        for (std::size_t i = 0; i < nds.size(); ++i)
            coords[i] = nodes_.at(static_cast<std::size_t>(nds[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override {
        return (face_id == 0) ? 1 : -1;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }
    void forEachCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { cb(0); cb(1); }
    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> cb) const override {
        // Face 0: marker 1, belongs to cell 0 (tet A)
        if (marker < 0 || marker == 1) {
            cb(0, 0);
        }
    }
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
};

} // namespace

// ============================================================================
// Scalar Poisson with pure Neumann → auto gauge constraint
// ============================================================================

TEST(GaugeIntegration, ScalarPoisson_PureNeumann_AutoZeroMean)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // No Dirichlet BCs → pure Neumann-like
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Check that the gauge registry was populated and resolved
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);

    EXPECT_FALSE(reg->candidates().empty());
    EXPECT_TRUE(reg->isResolved());

    // Should have created an ExactNullspace mode with MeanZeroElimination
    bool found_exact = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::ExactNullspace &&
            mode.policy == EnforcementPolicy::MeanZeroElimination) {
            found_exact = true;
        }
    }
    EXPECT_TRUE(found_exact);

    // The AffineConstraints should have at least one constrained DOF
    // (from the auto-created GlobalConstraint::zeroMean)
    const auto& ac = sys.constraints();
    bool any_constrained = false;
    for (GlobalIndex d = 0; d < 4; ++d) {
        if (ac.isConstrained(d)) {
            any_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(any_constrained);
}

// ============================================================================
// Scalar Poisson with Dirichlet → anchored, no auto constraint
// ============================================================================

TEST(GaugeIntegration, ScalarPoisson_WithDirichlet_NoAutoConstraint)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Add a Dirichlet constraint on DOF 0
    auto dirichlet = std::make_unique<constraints::DirichletBC>(0, 0.0);
    sys.addConstraint(std::move(dirichlet));

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // The gauge registry should detect the Dirichlet anchoring
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_anchored = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) {
            found_anchored = true;
        }
    }
    EXPECT_TRUE(found_anchored);
}

// ============================================================================
// Reaction-diffusion: mass term anchors, no auto constraint
// ============================================================================

TEST(GaugeIntegration, ScalarReactionDiffusion_NoNullspace)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx() + (u * v).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // Mass term anchors the mode → no candidates should be produced
    if (sys.hasGaugeRegistry()) {
        EXPECT_TRUE(sys.gaugeRegistryIfPresent()->candidates().empty());
    }
}

// ============================================================================
// BC anchoring verdicts — StandardBCs
// ============================================================================

TEST(GaugeIntegration, EssentialBC_DescriptorAnchorsConstantAndTranslation)
{
    // EssentialBC descriptor has anchors_constant_mode=true and
    // anchors_rigid_body_translation=true. descriptorToVerdict returns
    // Anchored for ScalarConstant/ComponentwiseConstant, PartiallyAnchored
    // for KernelOfSymGrad (translation anchored, rotation not).
    bc::EssentialBC bc(1, FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::PartiallyAnchored);
}

TEST(GaugeIntegration, NaturalBC_Preserves_AllFamilies)
{
    bc::NaturalBC bc(1, FormExpr::constant(1.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::Preserved);
}

TEST(GaugeIntegration, RobinBC_Anchors_ConstantModes_PartiallyAnchors_KernelOfSymGrad)
{
    bc::RobinBC bc(1, FormExpr::constant(1.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant), AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad), AnchoringVerdict::PartiallyAnchored);
}

TEST(GaugeIntegration, ReservedBC_EmptyDescriptors)
{
    // ReservedBC returns empty descriptors — no mathematical constraint.
    bc::ReservedBC bc(1);
    auto descs = bc.analysisMetadata(0, nullptr);
    EXPECT_TRUE(descs.empty());
}

// ============================================================================
// Vector field — componentwise constant detection through FESystem
// ============================================================================

TEST(GaugeIntegration, VectorField_GradOnly_ComponentwiseConstantCandidate)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto space = std::make_shared<spaces::ProductSpace>(base, 3);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 3});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Gauge candidates are populated during setup() from NullspaceHints.
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto& candidates = sys.gaugeRegistryIfPresent()->candidates();
    ASSERT_FALSE(candidates.empty());
    bool found_cw = false;
    for (const auto& c : candidates) {
        if (c.family == NullspaceModeFamily::ComponentwiseConstant ||
            c.family == NullspaceModeFamily::ScalarConstant) {
            found_cw = true;
        }
    }
    EXPECT_TRUE(found_cw) << "Expected ComponentwiseConstant or per-component ScalarConstant candidate";
}

// ============================================================================
// Vector field — rigid-body mode detection through FESystem
// ============================================================================

TEST(GaugeIntegration, VectorField_SymGrad_KernelOfSymGradCandidate)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    auto space = std::make_shared<spaces::ProductSpace>(base, 3);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 3});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Gauge candidates are populated during setup() from NullspaceHints.
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto& candidates = sys.gaugeRegistryIfPresent()->candidates();
    ASSERT_FALSE(candidates.empty());
    bool found_rbm = false;
    for (const auto& c : candidates) {
        if (c.family == NullspaceModeFamily::KernelOfSymGrad) {
            found_rbm = true;
        }
    }
    EXPECT_TRUE(found_rbm) << "Expected KernelOfSymGrad candidate";
}

// ============================================================================
// Phase 3: Non-Forms kernel analysisContributions() hooks
// ============================================================================

namespace {

/// Minimal AssemblyKernel that declares a gauge candidate via analysisContributions()
class GaugeDeclaringKernel final : public assembly::AssemblyKernel {
public:
    GaugeDeclaringKernel(FieldId field, NullspaceModeFamily family)
        : field_(field), family_(family)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const override {
        return assembly::RequiredData::None;
    }

    void computeCell(const assembly::AssemblyContext& /*ctx*/,
                     assembly::KernelOutput& out) override
    {
        out.reserve(0, 0, true, false);
    }

    [[nodiscard]] std::vector<analysis::ContributionDescriptor> analysisContributions() const override
    {
        analysis::ContributionDescriptor cd;
        cd.operator_tag = "gauge_declaring_kernel";
        cd.origin = "GaugeDeclaringKernel";
        cd.domain = analysis::DomainKind::Cell;
        cd.role = analysis::ContributionRole::DiagonalBlock;
        cd.test_variables = {analysis::VariableKey::field(field_)};
        cd.trial_variables = {analysis::VariableKey::field(field_)};
        cd.confidence = analysis::AnalysisConfidence::High;

        analysis::NullspaceHint hint;
        hint.field = field_;
        hint.component = -1;
        hint.confidence = analysis::AnalysisConfidence::High;
        hint.reason = "Explicitly declared by GaugeDeclaringKernel";
        if (family_ == NullspaceModeFamily::ScalarConstant) {
            hint.family = analysis::NullspaceFamily::ScalarConstant;
        } else if (family_ == NullspaceModeFamily::KernelOfSymGrad) {
            hint.family = analysis::NullspaceFamily::KernelOfSymGrad;
        } else {
            hint.family = analysis::NullspaceFamily::ComponentwiseConstant;
        }
        cd.nullspace_hints.push_back(std::move(hint));
        return {cd};
    }

private:
    FieldId field_;
    NullspaceModeFamily family_;
};

/// Minimal GlobalKernel that declares a gauge candidate via analysisContributions()
class GaugeDeclaringGlobalKernel final : public systems::GlobalKernel {
public:
    GaugeDeclaringGlobalKernel(FieldId field, NullspaceModeFamily family)
        : field_(field), family_(family)
    {
    }

    [[nodiscard]] assembly::AssemblyResult assemble(
        const systems::FESystem& /*system*/,
        const systems::AssemblyRequest& /*request*/,
        const systems::SystemStateView& /*state*/,
        assembly::GlobalSystemView* /*matrix_out*/,
        assembly::GlobalSystemView* /*vector_out*/) override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::ContributionDescriptor> analysisContributions() const override
    {
        analysis::ContributionDescriptor cd;
        cd.operator_tag = "gauge_declaring_global_kernel";
        cd.origin = "GaugeDeclaringGlobalKernel";
        cd.domain = analysis::DomainKind::Cell;
        cd.role = analysis::ContributionRole::DiagonalBlock;
        cd.test_variables = {analysis::VariableKey::field(field_)};
        cd.trial_variables = {analysis::VariableKey::field(field_)};
        cd.confidence = analysis::AnalysisConfidence::High;

        analysis::NullspaceHint hint;
        hint.field = field_;
        hint.component = -1;
        hint.confidence = analysis::AnalysisConfidence::High;
        hint.reason = "Explicitly declared by GaugeDeclaringGlobalKernel";
        if (family_ == NullspaceModeFamily::ScalarConstant) {
            hint.family = analysis::NullspaceFamily::ScalarConstant;
        } else if (family_ == NullspaceModeFamily::KernelOfSymGrad) {
            hint.family = analysis::NullspaceFamily::KernelOfSymGrad;
        } else {
            hint.family = analysis::NullspaceFamily::ComponentwiseConstant;
        }
        cd.nullspace_hints.push_back(std::move(hint));
        return {cd};
    }

private:
    FieldId field_;
    NullspaceModeFamily family_;
};

} // namespace

TEST(GaugeIntegration, CellKernel_AnalysisContributions_CollectedDuringSetup)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Add a hand-written kernel that declares ScalarConstant nullspace
    auto kernel = std::make_shared<GaugeDeclaringKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addCellKernel("op", u_field, kernel);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // The gauge registry should have collected the candidate from the kernel's
    // analysisContributions() NullspaceHint
    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_NE(reg, nullptr);

    bool found_explicit = false;
    for (const auto& c : reg->candidates()) {
        if (c.field == u_field &&
            c.family == NullspaceModeFamily::ScalarConstant &&
            c.source == gauge::CandidateSource::ExplicitDeclaration) {
            found_explicit = true;
        }
    }
    EXPECT_TRUE(found_explicit);

    // Without Dirichlet BC, it should resolve to ExactNullspace
    bool found_exact = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == u_field &&
            mode.status == gauge::GaugeStatus::ExactNullspace) {
            found_exact = true;
        }
    }
    EXPECT_TRUE(found_exact);
}

TEST(GaugeIntegration, GlobalKernel_AnalysisContributions_CollectedDuringSetup)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Add a global kernel that declares ScalarConstant nullspace
    auto gk = std::make_shared<GaugeDeclaringGlobalKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addGlobalKernel("op", gk);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_explicit = false;
    for (const auto& c : reg->candidates()) {
        if (c.field == u_field &&
            c.source == gauge::CandidateSource::ExplicitDeclaration) {
            found_explicit = true;
        }
    }
    EXPECT_TRUE(found_explicit);
}

TEST(GaugeIntegration, CellKernel_AnalysisContributions_AnchoredByDirichlet)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto kernel = std::make_shared<GaugeDeclaringKernel>(u_field, NullspaceModeFamily::ScalarConstant);
    sys.addCellKernel("op", u_field, kernel);

    // Add Dirichlet to anchor the mode
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(0, 0.0));

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    bool found_anchored = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.candidate.field == u_field &&
            mode.status == gauge::GaugeStatus::Anchored) {
            found_anchored = true;
        }
    }
    EXPECT_TRUE(found_anchored);
}

TEST(GaugeIntegration, DefaultKernel_EmptyAnalysisContributions)
{
    // Verify that the base class default returns empty
    class NoMetadataKernel final : public assembly::AssemblyKernel {
    public:
        [[nodiscard]] assembly::RequiredData getRequiredData() const override {
            return assembly::RequiredData::None;
        }
        void computeCell(const assembly::AssemblyContext&, assembly::KernelOutput& out) override {
            out.reserve(0, 0, true, false);
        }
    };

    NoMetadataKernel nk;
    EXPECT_TRUE(nk.analysisContributions().empty());
}

TEST(GaugeIntegration, DefaultGlobalKernel_EmptyAnalysisContributions)
{
    class NoMetadataGlobalKernel final : public systems::GlobalKernel {
    public:
        [[nodiscard]] assembly::AssemblyResult assemble(
            const systems::FESystem&, const systems::AssemblyRequest&,
            const systems::SystemStateView&,
            assembly::GlobalSystemView*, assembly::GlobalSystemView*) override
        {
            return {};
        }
    };

    NoMetadataGlobalKernel nk;
    EXPECT_TRUE(nk.analysisContributions().empty());
}

// ============================================================================
// Phase 4: LinearSolver nullspace interface
// ============================================================================

TEST(GaugeIntegration, LinearSolver_DefaultSupportsNullspace_IsFalse)
{
    // Create a minimal solver to check the default
    class MinimalSolver final : public backends::LinearSolver {
    public:
        [[nodiscard]] backends::BackendKind backendKind() const noexcept override {
            return backends::BackendKind::Eigen;
        }
        void setOptions(const backends::SolverOptions& o) override { opts_ = o; }
        [[nodiscard]] const backends::SolverOptions& getOptions() const noexcept override { return opts_; }
        [[nodiscard]] backends::SolverReport solve(
            const backends::GenericMatrix&, backends::GenericVector&,
            const backends::GenericVector&) override { return {}; }
    private:
        backends::SolverOptions opts_{};
    };

    MinimalSolver solver;
    EXPECT_FALSE(solver.supportsNullspace());

    // setNullspaceBasis should be a no-op (shouldn't crash)
    std::vector<std::vector<double>> basis = {{1.0, 0.0}, {0.0, 1.0}};
    solver.setNullspaceBasis(basis);  // no-op
}

// ============================================================================
// Phase 4: GaugeRegistry — resolver always picks algebraic enforcement
// ============================================================================

TEST(GaugeIntegration, Resolve_AlwaysAlgebraic_ScalarConstant)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.component = -1;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };

    // Resolver always picks algebraic enforcement (MeanZero for scalar constant)
    reg.resolve(dof_provider);

    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    const auto& mode = reg.resolvedModes()[0];
    EXPECT_EQ(mode.status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(mode.policy, EnforcementPolicy::MeanZeroElimination);
}

// ============================================================================
// Phase 4: buildNullspaceBasis — used for solver-side supplemental projection
// ============================================================================

TEST(GaugeIntegration, BuildNullspaceBasis_Empty_WhenAlgebraicEnforcement)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    // buildNullspaceBasis only emits for SolverNullspace policy.
    // Algebraic enforcement (MeanZero) modifies the constraint system,
    // so the original nullspace vector is NOT the nullspace of the
    // constrained operator — projecting against it would be incorrect.
    auto basis = reg.buildNullspaceBasis(/*n_total_dofs=*/4, dof_provider);
    EXPECT_TRUE(basis.empty());
}

TEST(GaugeIntegration, BuildNullspaceBasis_Empty_WhenAnchored)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    reg.addAnchoring({0, -1, -1, {}, AnchoringVerdict::Anchored, "Dirichlet"});

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    auto basis = reg.buildNullspaceBasis(4, dof_provider);
    EXPECT_TRUE(basis.empty());
}

TEST(GaugeIntegration, ApplyEnforcement_AlgebraicMeanZero_CreatesConstraint)
{
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    auto dof_provider = [](FieldId, int) { return std::vector<GlobalIndex>{0, 1, 2, 3}; };
    reg.resolve(dof_provider);

    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);

    // Algebraic MeanZero enforcement creates 1 constraint
    EXPECT_EQ(n, 1);
    EXPECT_TRUE(ac.isConstrained(0));
}

// ============================================================================
// Resolver always uses algebraic — solver-side projection supplements in Newton
// ============================================================================

TEST(GaugeIntegration, Resolve_AlwaysAlgebraic_SetupWithNoPreference)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();

    // Resolver always picks algebraic enforcement (MeanZero)
    bool found_mean_zero = false;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.policy == EnforcementPolicy::MeanZeroElimination) {
            found_mean_zero = true;
        }
    }
    EXPECT_TRUE(found_mean_zero);

    // Algebraic constraint should have been created
    const auto& ac = sys.constraints();
    bool any_constrained = false;
    for (GlobalIndex d = 0; d < 4; ++d) {
        if (ac.isConstrained(d)) {
            any_constrained = true;
        }
    }
    EXPECT_TRUE(any_constrained);
}

// ============================================================================
// Connected-component scoping: two disconnected regions, Dirichlet on one
// ============================================================================

TEST(GaugeIntegration, TwoRegions_DirichletOnRegionA_RegionBGetsGauge)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    systems::installFormulation(sys, "op", {u_field}, residual);

    // Add Dirichlet on DOF 0 (vertex 0, in region A).
    // Region B (vertices 4-7) has no Dirichlet → should get auto-gauge.
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(0, 0.0));

    systems::SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_TRUE(reg->isResolved());

    // The two-tet mesh has 2 connected components.  Region expansion should
    // produce 2 candidate modes (one per region).  The Dirichlet on DOF 0
    // (region A) should anchor only region A; region B should be ExactNullspace.
    ASSERT_EQ(reg->resolvedModes().size(), 2u)
        << "Expected 2 resolved modes (one per disconnected region)";

    int n_anchored = 0;
    int n_exact_nullspace = 0;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) ++n_anchored;
        if (mode.status == GaugeStatus::ExactNullspace) ++n_exact_nullspace;
    }

    EXPECT_EQ(n_anchored, 1) << "Exactly one region should be anchored (has Dirichlet)";
    EXPECT_EQ(n_exact_nullspace, 1) << "Exactly one region should be ExactNullspace (no Dirichlet)";

    // The unanchored region should have gotten a gauge constraint
    const auto& ac = sys.constraints();
    bool region_b_constrained = false;
    // Region B DOFs are 4-7; check if any of them got constrained by gauge
    for (GlobalIndex d = 4; d < 8; ++d) {
        if (ac.isConstrained(d)) {
            region_b_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(region_b_constrained)
        << "Region B (no Dirichlet) should have an auto-gauge constraint";
}

// ============================================================================
// Connected-component scoping: Robin BC on one region, end-to-end
// ============================================================================

TEST(GaugeIntegration, TwoRegions_RobinOnRegionA_RegionBGetsGauge)
{
    // End-to-end test: Robin BC on boundary marker 1 (only touches tet A / region A).
    // Region A is anchored by the Robin BC.  Region B has pure Neumann → should
    // get auto-gauge enforcement.  This exercises:
    //  - BC manager setting boundary_marker on evidence
    //  - retagEvidenceRegions() converting marker 1 → region A
    //  - Resolver correctly scoping the anchor to region A only
    auto mesh = std::make_shared<TwoTetraOneBoundaryMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = FormExpr::stateField(u_field, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    // Robin BC on marker 1 (only on tet A): u + penalty*u = g
    // descriptorToVerdict() returns Anchored for ScalarConstant.
    systems::BoundaryConditionManager bc_mgr;
    bc_mgr.add(std::make_unique<bc::RobinBC>(1, FormExpr::constant(1.0), FormExpr::constant(0.0)));
    bc_mgr.apply(sys, residual, u, v, u_field);

    systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    sys.setup({}, inputs);

    ASSERT_TRUE(sys.hasGaugeRegistry());
    const auto* reg = sys.gaugeRegistryIfPresent();
    ASSERT_TRUE(reg->isResolved());

    // Expect 2 resolved modes (one per region).
    // Region A: anchored by Robin BC (via retagged evidence).
    // Region B: ExactNullspace (no anchor) → gets gauge enforcement.
    ASSERT_EQ(reg->resolvedModes().size(), 2u)
        << "Expected 2 resolved modes (one per disconnected region)";

    int n_anchored = 0;
    int n_exact_nullspace = 0;
    for (const auto& mode : reg->resolvedModes()) {
        if (mode.status == GaugeStatus::Anchored) ++n_anchored;
        if (mode.status == GaugeStatus::ExactNullspace) ++n_exact_nullspace;
    }

    EXPECT_EQ(n_anchored, 1)
        << "Exactly one region should be anchored (has Robin BC)";
    EXPECT_EQ(n_exact_nullspace, 1)
        << "Exactly one region should be ExactNullspace (no BC)";

    // Verify that region B DOFs actually got a gauge constraint.
    // Region B vertices are 4-7 → DOFs 4-7 for a scalar field.
    const auto& ac = sys.constraints();
    bool region_b_constrained = false;
    for (GlobalIndex d = 4; d < 8; ++d) {
        if (ac.isConstrained(d)) {
            region_b_constrained = true;
            break;
        }
    }
    EXPECT_TRUE(region_b_constrained)
        << "Region B (no Robin BC) should have an auto-gauge constraint";
}

// ============================================================================
// Global anchor without boundary_marker on connected mesh → anchors correctly
// ============================================================================

TEST(GaugeIntegration, GlobalAnchor_NoBoundaryMarker_ConnectedMesh_AnchorsCorrectly)
{
    // Regression: a global Anchored evidence without boundary_marker (e.g.,
    // from the NS module's unlabeled-face detection) should anchor the pressure
    // on a connected mesh.  The resolver allows ev.region=-1 to match
    // candidate.region=-1 (both global).  Without this, a naturally-anchored
    // pressure would get an unnecessary MeanZeroElimination constraint.
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    c.source = CandidateSource::ExplicitDeclaration;
    c.reason = "Pressure nullspace";
    reg.addCandidate(c);

    // Global anchor without boundary_marker (mimics NS unlabeled-face path)
    gauge::AnchoringEvidence ev;
    ev.field = 0;
    ev.family = NullspaceModeFamily::ScalarConstant;
    ev.verdict = AnchoringVerdict::Anchored;
    ev.source = "Velocity natural BC (unlabeled faces)";
    // ev.boundary_marker stays -1 (no marker)
    reg.addAnchoring(ev);

    // No region provider → connected mesh → no region expansion
    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    reg.resolve(dof_provider);

    // Single global candidate should be Anchored
    ASSERT_EQ(reg.resolvedModes().size(), 1u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::Anchored);
    EXPECT_EQ(reg.resolvedModes()[0].policy, EnforcementPolicy::None);

    // No constraints should be created
    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 0);
}

TEST(GaugeIntegration, GlobalAnchor_NoBoundaryMarker_DisconnectedMesh_KnownLimitation)
{
    // Known limitation: on a disconnected mesh, global Anchored evidence
    // without boundary_marker is blocked from per-region candidates.  Both
    // regions get gauge enforcement even if one is genuinely anchored.
    // This can over-constrain the anchored region (MeanZeroElimination may
    // change the pressure level).  Production meshes should label all
    // boundary faces to enable correct per-region scoping via
    // retagEvidenceRegions().
    GaugeRegistry reg;

    GaugeCandidate c;
    c.field = 0;
    c.family = NullspaceModeFamily::ScalarConstant;
    c.confidence = Confidence::High;
    reg.addCandidate(c);

    gauge::AnchoringEvidence ev;
    ev.field = 0;
    ev.family = NullspaceModeFamily::ScalarConstant;
    ev.verdict = AnchoringVerdict::Anchored;
    ev.source = "Velocity natural BC (unlabeled faces)";
    reg.addAnchoring(ev);

    auto dof_provider = [](FieldId, int) {
        return std::vector<GlobalIndex>{0, 1, 2, 3};
    };
    auto region_provider = [](GlobalIndex dof) -> int {
        return (dof < 2) ? 0 : 1;
    };

    reg.resolve(dof_provider, region_provider);

    // Two regions expanded.  Global anchor blocked → both ExactNullspace.
    // This is the known-limitation behavior, not the ideal outcome.
    ASSERT_EQ(reg.resolvedModes().size(), 2u);
    EXPECT_EQ(reg.resolvedModes()[0].status, GaugeStatus::ExactNullspace);
    EXPECT_EQ(reg.resolvedModes()[1].status, GaugeStatus::ExactNullspace);

    // Both regions get gauge enforcement (may over-constrain)
    constraints::AffineConstraints ac;
    int n = reg.applyEnforcement(ac, dof_provider);
    EXPECT_EQ(n, 2);
}
