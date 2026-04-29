/**
 * @file test_AnalyzerPasses.cpp
 * @brief Unit tests for all Phase 7 analyzer passes
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/ConstraintAnalysisSummary.h"

#include "Analysis/KernelAnalyzer.h"
#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ConstraintRankAnalyzer.h"
#include "Analysis/MixedOperatorAnalyzer.h"
#include "Analysis/OperatorClassAnalyzer.h"
#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/CompatibilityAnalyzer.h"
#include "Analysis/TopologyScopeAnalyzer.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

std::shared_ptr<spaces::FunctionSpace> scalarH1() {
    return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
}

std::shared_ptr<spaces::FunctionSpace> vectorH1(int dim = 3) {
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    return std::make_shared<spaces::ProductSpace>(base, dim);
}

// Helper: build a FormulationRecord for scalar Poisson (pure Neumann)
FormulationRecord makePoissonRecord() {
    auto space = scalarH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = false;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(0));
    rec.block_couplings = {{0, 0}};
    return rec;
}

// Helper: Stokes record
FormulationRecord makeStokesRecord() {
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    auto p = FormExpr::stateField(0, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(1, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");
    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0, 1};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = true;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(0));
    rec.active_variables.push_back(VariableKey::field(1));
    rec.block_couplings = {{0,0}, {0,1}, {1,0}, {1,1}};
    return rec;
}

// Helper: stabilized Poisson record
FormulationRecord makeStabilizedPoissonRecord() {
    auto space = scalarH1();
    auto p = FormExpr::stateField(0, *space, "p");
    auto q = FormExpr::testFunction(*space, "q");
    auto h = FormExpr::cellDiameter();
    auto residual = (h * inner(grad(p), grad(q))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = false;
    rec.has_stabilization_terms = true;
    rec.active_variables.push_back(VariableKey::field(0));
    return rec;
}

// Helper: linear elasticity record (sym_grad only)
FormulationRecord makeElasticityRecord() {
    auto space = vectorH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = false;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(0));
    rec.block_couplings = {{0, 0}};
    return rec;
}

} // namespace

// ============================================================================
// All passes are no-ops on empty context
// ============================================================================

TEST(AnalyzerPasses, AllPasses_EmptyContext_NoOutput) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    auto report = analyzer.analyze(ctx);

    EXPECT_TRUE(report.claims.empty());
    EXPECT_TRUE(report.issues.empty());
}

// ============================================================================
// KernelAnalyzer
// ============================================================================

TEST(KernelAnalyzer, ScalarPoisson_DetectsNullspace) {
    KernelAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->field, FieldId{0});
    EXPECT_EQ(nullspace[0]->status, PropertyStatus::Exact);
    EXPECT_EQ(nullspace[0]->confidence, AnalysisConfidence::High);
}

TEST(KernelAnalyzer, LinearElasticity_DetectsRigidBody) {
    KernelAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeElasticityRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->field, FieldId{0});
    // The description should mention rigid-body or sym_grad
    EXPECT_NE(nullspace[0]->description.find("rigid"), std::string::npos);
}

TEST(KernelAnalyzer, StabilizedPoisson_LikelyNullspace) {
    KernelAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeStabilizedPoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->status, PropertyStatus::Likely);
    EXPECT_EQ(nullspace[0]->confidence, AnalysisConfidence::Medium);
}

// ============================================================================
// MixedOperatorAnalyzer
// ============================================================================

TEST(MixedOperatorAnalyzer, Stokes_DetectsSaddlePoint) {
    MixedOperatorAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeStokesRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto saddle = report.claimsOfKind(PropertyKind::MixedSaddlePoint);
    ASSERT_GE(saddle.size(), 1u);
    EXPECT_EQ(saddle[0]->status, PropertyStatus::Exact);
}

TEST(MixedOperatorAnalyzer, SingleField_NoSaddlePoint) {
    MixedOperatorAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    EXPECT_EQ(report.countByKind(PropertyKind::MixedSaddlePoint), 0u);
}

TEST(MixedOperatorAnalyzer, StokesFallbackEmitsStructuredPressureNullspace) {
    MixedOperatorAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeStokesRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->field, FieldId{0});
    ASSERT_TRUE(nullspace[0]->nullspace_family.has_value());
    EXPECT_EQ(*nullspace[0]->nullspace_family, NullspaceFamily::ScalarConstant);
    EXPECT_EQ(nullspace[0]->claim_origin, "MixedOperatorAnalyzer");
}

// ============================================================================
// OperatorClassAnalyzer
// ============================================================================

TEST(OperatorClassAnalyzer, Poisson_SymmetricSemidefinite) {
    OperatorClassAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto symmetry = report.claimsOfKind(PropertyKind::OperatorSymmetry);
    EXPECT_GE(symmetry.size(), 1u);
}

// ============================================================================
// StabilizationAnalyzer
// ============================================================================

TEST(StabilizationAnalyzer, StabilizedPoisson_Detected) {
    StabilizationAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeStabilizedPoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto stab = report.claimsOfKind(PropertyKind::Stabilization);
    EXPECT_GE(stab.size(), 1u);
    if (!stab.empty()) {
        EXPECT_EQ(stab[0]->status, PropertyStatus::Exact);
    }
}

TEST(StabilizationAnalyzer, Poisson_NoStabilization) {
    StabilizationAnalyzer pass;
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    EXPECT_EQ(report.countByKind(PropertyKind::Stabilization), 0u);
}

// ============================================================================
// ConstraintRankAnalyzer
// ============================================================================

TEST(ConstraintRankAnalyzer, NullspaceNoBC_EmitsUnderConstraint) {
    ConstraintRankAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Pre-populate a Nullspace claim (as if KernelAnalyzer ran)
    ProblemAnalysisReport report;
    PropertyClaim nullspace_claim;
    nullspace_claim.kind = PropertyKind::Nullspace;
    nullspace_claim.status = PropertyStatus::Exact;
    nullspace_claim.field = 0;
    report.claims.push_back(nullspace_claim);

    // No BCs, no constraint summary → under-constrained
    pass.run(ctx, report);

    auto under = report.claimsOfKind(PropertyKind::UnderConstraint);
    EXPECT_GE(under.size(), 1u);
}

TEST(ConstraintRankAnalyzer, NullspaceWithDirichletBC_NoUnderConstraint) {
    ConstraintRankAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Add a Dirichlet BC that anchors the field
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::Value;
    desc.enforcement_kind = EnforcementKind::Strong;
    desc.anchors_constant_mode = true;
    ctx.addBCDescriptor(desc);

    ProblemAnalysisReport report;
    PropertyClaim nullspace_claim;
    nullspace_claim.kind = PropertyKind::Nullspace;
    nullspace_claim.status = PropertyStatus::Exact;
    nullspace_claim.field = 0;
    report.claims.push_back(nullspace_claim);

    pass.run(ctx, report);

    auto under = report.claimsOfKind(PropertyKind::UnderConstraint);
    EXPECT_EQ(under.size(), 0u);
}

TEST(ConstraintRankAnalyzer, NullspaceWithWeakInequalityBC_NoUnderConstraintButInfo) {
    ConstraintRankAnalyzer pass;
    ProblemAnalysisContext ctx;

    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::NormalComponent;
    desc.enforcement_kind = EnforcementKind::WeakInequality;
    desc.anchors_constant_mode = true;
    desc.state_dependent_activation = true;
    desc.inequality_sense = InequalitySense::LessEqual;
    ctx.addBCDescriptor(desc);

    ProblemAnalysisReport report;
    PropertyClaim nullspace_claim;
    nullspace_claim.kind = PropertyKind::Nullspace;
    nullspace_claim.status = PropertyStatus::Exact;
    nullspace_claim.field = 0;
    nullspace_claim.description = "scalar constant mode";
    report.claims.push_back(nullspace_claim);

    pass.run(ctx, report);

    auto under = report.claimsOfKind(PropertyKind::UnderConstraint);
    EXPECT_EQ(under.size(), 0u);
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_NE(report.issues[0].message.find("weakly anchored"), std::string::npos);
}

// ============================================================================
// CompatibilityAnalyzer
// ============================================================================

TEST(CompatibilityAnalyzer, PureNeumannNullspace_EmitsCompatibility) {
    CompatibilityAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Only Neumann BCs
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::Flux;
    desc.enforcement_kind = EnforcementKind::WeakConsistent;
    ctx.addBCDescriptor(desc);

    ProblemAnalysisReport report;
    PropertyClaim nc;
    nc.kind = PropertyKind::Nullspace;
    nc.status = PropertyStatus::Exact;
    nc.field = 0;
    report.claims.push_back(nc);

    pass.run(ctx, report);

    auto compat = report.claimsOfKind(PropertyKind::CompatibilityCondition);
    EXPECT_GE(compat.size(), 1u);
}

TEST(CompatibilityAnalyzer, NoNullspace_NoCompatibility) {
    CompatibilityAnalyzer pass;
    ProblemAnalysisContext ctx;
    ProblemAnalysisReport report;

    pass.run(ctx, report);

    EXPECT_EQ(report.countByKind(PropertyKind::CompatibilityCondition), 0u);
}

TEST(CompatibilityAnalyzer, WeakInequalityBC_SuppressesPureFluxCompatibilityClaim) {
    CompatibilityAnalyzer pass;
    ProblemAnalysisContext ctx;

    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::NormalComponent;
    desc.enforcement_kind = EnforcementKind::WeakInequality;
    desc.anchors_constant_mode = true;
    desc.state_dependent_activation = true;
    desc.inequality_sense = InequalitySense::LessEqual;
    ctx.addBCDescriptor(desc);

    ProblemAnalysisReport report;
    PropertyClaim nc;
    nc.kind = PropertyKind::Nullspace;
    nc.status = PropertyStatus::Exact;
    nc.field = 0;
    report.claims.push_back(nc);

    pass.run(ctx, report);

    auto compat = report.claimsOfKind(PropertyKind::CompatibilityCondition);
    EXPECT_EQ(compat.size(), 0u);
}

// ============================================================================
// CouplingGraphAnalyzer
// ============================================================================

TEST(CouplingGraphAnalyzer, CoupledRecord_EmitsCoupledStructure) {
    CouplingGraphAnalyzer pass;
    ProblemAnalysisContext ctx;

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    auto bf = VariableKey::named(VariableKind::BoundaryFunctional, "Q");
    rec.variable_couplings.emplace_back(VariableKey::field(0), bf);
    rec.boundary_functional_dependencies.push_back(bf);
    ctx.addFormulationRecord(rec);

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    auto coupled = report.claimsOfKind(PropertyKind::CoupledSystemStructure);
    EXPECT_GE(coupled.size(), 1u);
}

TEST(CouplingGraphAnalyzer, NoCouplings_NoClaims) {
    CouplingGraphAnalyzer pass;
    ProblemAnalysisContext ctx;

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    ctx.addFormulationRecord(rec);

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    EXPECT_EQ(report.countByKind(PropertyKind::CoupledSystemStructure), 0u);
}

// ============================================================================
// TopologyScopeAnalyzer
// ============================================================================

TEST(TopologyScopeAnalyzer, DisconnectedMesh_UnanchoredRegion) {
    TopologyScopeAnalyzer pass;
    ProblemAnalysisContext ctx;

    // Two disconnected regions
    TopologyAnalysisContext topo;
    ConnectedComponent c0;
    c0.region_id = 0;
    c0.num_cells = 1;
    c0.boundary_markers = {1};
    topo.components.push_back(c0);

    ConnectedComponent c1;
    c1.region_id = 1;
    c1.num_cells = 1;
    // No boundary markers on region 1
    topo.components.push_back(c1);

    topo.boundary_mapping.marker_to_regions[1] = {0};
    topo.boundary_mapping.region_to_markers[0] = {1};

    ctx.setTopologyContext(topo);

    // BC only on marker 1 (region 0)
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.boundary_marker = 1;
    desc.anchors_constant_mode = true;
    ctx.addBCDescriptor(desc);

    // Global nullspace claim
    ProblemAnalysisReport report;
    PropertyClaim nc;
    nc.kind = PropertyKind::Nullspace;
    nc.status = PropertyStatus::Exact;
    nc.field = 0;
    nc.region = -1;  // global
    report.claims.push_back(nc);

    pass.run(ctx, report);

    auto scoped = report.claimsOfKind(PropertyKind::TopologyScopedKernel);
    EXPECT_GE(scoped.size(), 1u);
    // The scoped claim should reference region 1 (unanchored)
    if (!scoped.empty()) {
        EXPECT_EQ(scoped[0]->region, 1);
    }
}

TEST(TopologyScopeAnalyzer, ConnectedMesh_NoClaims) {
    TopologyScopeAnalyzer pass;
    ProblemAnalysisContext ctx;

    TopologyAnalysisContext topo;
    ConnectedComponent c0;
    c0.region_id = 0;
    c0.num_cells = 2;
    topo.components.push_back(c0);
    ctx.setTopologyContext(topo);

    ProblemAnalysisReport report;
    pass.run(ctx, report);

    EXPECT_EQ(report.countByKind(PropertyKind::TopologyScopedKernel), 0u);
}

// ============================================================================
// Full pipeline integration
// ============================================================================

TEST(AnalyzerPasses, FullPipeline_PoissonPureNeumann) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Field descriptor
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "pressure";
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    ctx.addFieldDescriptor(fd);

    // Poisson formulation (pure Neumann → no BCs anchoring)
    ctx.addFormulationRecord(makePoissonRecord());

    // Neumann BC only
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.trace_kind = TraceKind::Flux;
    desc.enforcement_kind = EnforcementKind::WeakConsistent;
    desc.anchors_constant_mode = false;
    ctx.addBCDescriptor(desc);

    auto report = analyzer.analyze(ctx);

    // Should detect: Nullspace, UnderConstraint, CompatibilityCondition, OperatorSymmetry
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);
    EXPECT_GE(report.countByKind(PropertyKind::UnderConstraint), 1u);
    EXPECT_GE(report.countByKind(PropertyKind::CompatibilityCondition), 1u);
    EXPECT_GE(report.countByKind(PropertyKind::OperatorSymmetry), 1u);

    // No saddle-point (single field)
    EXPECT_EQ(report.countByKind(PropertyKind::MixedSaddlePoint), 0u);
}

TEST(AnalyzerPasses, FullPipeline_StokesWithDirichlet) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    ctx.addFormulationRecord(makeStokesRecord());

    // Dirichlet on velocity
    BoundaryConditionDescriptor vel_bc;
    vel_bc.primary_variable = VariableKey::field(1);
    vel_bc.trace_kind = TraceKind::Value;
    vel_bc.enforcement_kind = EnforcementKind::Strong;
    vel_bc.anchors_constant_mode = true;
    ctx.addBCDescriptor(vel_bc);

    auto report = analyzer.analyze(ctx);

    // Saddle-point
    EXPECT_GE(report.countByKind(PropertyKind::MixedSaddlePoint), 1u);

    // Velocity is anchored, pressure has nullspace but may or may not be flagged
    // depending on how the Stokes form is structured (p*div(v) is absolute for p)
}
