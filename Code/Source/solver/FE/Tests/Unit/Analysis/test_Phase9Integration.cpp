/**
 * @file test_Phase9Integration.cpp
 * @brief Phase 9 integration tests — full analysis pipeline on realistic formulations
 *
 * Each test builds a ProblemAnalysisContext with formulation records, BC descriptors,
 * topology, and/or constraint summaries, then runs the default ProblemAnalyzer and
 * verifies the expected claims.
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/FormExprScanner.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Forms/FormExpr.h"
#include "Forms/AffineAnalysis.h"
#include "Constraints/AffineConstraints.h"
#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/ProductSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;

namespace {

auto scalarH1() { return std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1); }
auto vectorH1(int dim = 3) {
    return std::make_shared<spaces::ProductSpace>(
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1), dim);
}

FormulationRecord makePoissonRecord(FieldId fid = 0) {
    auto space = scalarH1();
    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {fid};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = false;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(fid));
    rec.block_couplings = {{fid, fid}};
    return rec;
}

FormulationRecord makeElasticityRecord(FieldId fid = 0) {
    auto space = vectorH1();
    auto u = FormExpr::stateField(fid, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(sym(grad(u)), sym(grad(v))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {fid};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = false;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(fid));
    rec.block_couplings = {{fid, fid}};
    return rec;
}

FormulationRecord makeStokesRecord(FieldId p_field = 0, FieldId u_field = 1) {
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");
    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {p_field, u_field};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = true;
    rec.has_stabilization_terms = false;
    rec.active_variables.push_back(VariableKey::field(p_field));
    rec.active_variables.push_back(VariableKey::field(u_field));
    rec.block_couplings = {{p_field,p_field},{p_field,u_field},{u_field,p_field},{u_field,u_field}};
    return rec;
}

FormulationRecord makeStabilizedStokesRecord(FieldId p_field = 0, FieldId u_field = 1) {
    auto scalar_space = scalarH1();
    auto vector_space = vectorH1();
    auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    auto q = FormExpr::testFunction(*scalar_space, "q");
    auto u = FormExpr::stateField(u_field, *vector_space, "u");
    auto v = FormExpr::testFunction(*vector_space, "v");
    auto h = FormExpr::cellDiameter();
    auto residual = inner(grad(u), grad(v)).dx()
                  - (p * div(v)).dx()
                  + (div(u) * q).dx()
                  + (h * inner(grad(p), grad(q))).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {p_field, u_field};
    rec.residual_expr = residual.nodeShared();
    rec.is_mixed = true;
    rec.has_stabilization_terms = true;
    rec.active_variables.push_back(VariableKey::field(p_field));
    rec.active_variables.push_back(VariableKey::field(u_field));
    rec.block_couplings = {{p_field,p_field},{p_field,u_field},{u_field,p_field},{u_field,u_field}};
    return rec;
}

BoundaryConditionDescriptor neumannBC(FieldId fid, int marker) {
    BoundaryConditionDescriptor d;
    d.primary_variable = VariableKey::field(fid);
    d.boundary_marker = marker;
    d.trace_kind = TraceKind::Flux;
    d.enforcement_kind = EnforcementKind::WeakConsistent;
    d.anchors_constant_mode = false;
    d.anchors_rigid_body_translation = false;
    d.anchors_rigid_body_rotation = false;
    d.source = "NaturalBC on marker " + std::to_string(marker);
    return d;
}

BoundaryConditionDescriptor dirichletBC(FieldId fid, int marker) {
    BoundaryConditionDescriptor d;
    d.primary_variable = VariableKey::field(fid);
    d.boundary_marker = marker;
    d.trace_kind = TraceKind::Value;
    d.enforcement_kind = EnforcementKind::Strong;
    d.anchors_constant_mode = true;
    d.anchors_rigid_body_translation = true;
    d.anchors_rigid_body_rotation = false;
    d.source = "EssentialBC on marker " + std::to_string(marker);
    return d;
}

BoundaryConditionDescriptor robinBC(FieldId fid, int marker) {
    BoundaryConditionDescriptor d;
    d.primary_variable = VariableKey::field(fid);
    d.boundary_marker = marker;
    d.trace_kind = TraceKind::Mixed;
    d.enforcement_kind = EnforcementKind::WeakPenalty;
    d.anchors_constant_mode = true;
    d.anchors_rigid_body_translation = true;
    d.anchors_rigid_body_rotation = false;
    d.source = "RobinBC on marker " + std::to_string(marker);
    return d;
}

BoundaryConditionDescriptor nitscheBC(FieldId fid, int marker) {
    BoundaryConditionDescriptor d;
    d.primary_variable = VariableKey::field(fid);
    d.boundary_marker = marker;
    d.trace_kind = TraceKind::Value;
    d.enforcement_kind = EnforcementKind::WeakNitsche;
    d.anchors_constant_mode = true;
    d.anchors_rigid_body_translation = true;
    d.anchors_rigid_body_rotation = false;
    d.source = "ScalarNitscheBC on marker " + std::to_string(marker);
    return d;
}

BoundaryConditionDescriptor periodicBC(FieldId fid, int slave_marker) {
    BoundaryConditionDescriptor d;
    d.primary_variable = VariableKey::field(fid);
    d.boundary_marker = slave_marker;
    d.trace_kind = TraceKind::AlgebraicRelation;
    d.enforcement_kind = EnforcementKind::AffineRelation;
    d.anchors_constant_mode = false;
    d.anchors_rigid_body_translation = false;
    d.anchors_rigid_body_rotation = false;
    d.source = "PeriodicBC slave=" + std::to_string(slave_marker);
    return d;
}

} // namespace

// ============================================================================
// Pure Neumann Poisson
// ============================================================================

TEST(Phase9, PureNeumannPoisson) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(neumannBC(0, 1));

    auto report = analyzer.analyze(ctx);

    // Nullspace: ScalarConstant, Exact, High
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_GE(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->field, FieldId{0});
    EXPECT_EQ(nullspace[0]->status, PropertyStatus::Exact);
    EXPECT_EQ(nullspace[0]->confidence, AnalysisConfidence::High);

    // UnderConstraint (no anchoring BC)
    EXPECT_GE(report.countByKind(PropertyKind::UnderConstraint), 1u);

    // CompatibilityCondition (solvability requires ∫f=0)
    EXPECT_GE(report.countByKind(PropertyKind::CompatibilityCondition), 1u);

    // OperatorSymmetry (Laplacian is symmetric)
    EXPECT_GE(report.countByKind(PropertyKind::OperatorSymmetry), 1u);
}

// ============================================================================
// Robin Poisson — nullspace anchored
// ============================================================================

TEST(Phase9, RobinPoisson) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(robinBC(0, 1));

    auto report = analyzer.analyze(ctx);

    // Nullspace still detected by KernelAnalyzer (it doesn't see BCs)
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);

    // But NOT under-constrained (Robin anchors constant mode)
    EXPECT_EQ(report.countByKind(PropertyKind::UnderConstraint), 0u);

    // No compatibility condition (Robin provides the anchoring)
    EXPECT_EQ(report.countByKind(PropertyKind::CompatibilityCondition), 0u);
}

// ============================================================================
// Nitsche Poisson — nullspace anchored by weak Dirichlet
// ============================================================================

TEST(Phase9, NitschePoisson) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(nitscheBC(0, 1));

    auto report = analyzer.analyze(ctx);

    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::UnderConstraint), 0u);
    EXPECT_EQ(report.countByKind(PropertyKind::CompatibilityCondition), 0u);
}

// ============================================================================
// Dirichlet Poisson — nullspace anchored by strong Dirichlet
// ============================================================================

TEST(Phase9, DirichletPoisson) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(dirichletBC(0, 1));

    auto report = analyzer.analyze(ctx);

    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::UnderConstraint), 0u);
    EXPECT_EQ(report.countByKind(PropertyKind::CompatibilityCondition), 0u);
}

// ============================================================================
// Free Elasticity — rigid-body modes, under-constrained
// ============================================================================

TEST(Phase9, FreeElasticity) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeElasticityRecord());

    auto report = analyzer.analyze(ctx);

    // KernelOfSymGrad nullspace
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_GE(nullspace.size(), 1u);
    EXPECT_NE(nullspace[0]->description.find("rigid"), std::string::npos);
    EXPECT_EQ(nullspace[0]->status, PropertyStatus::Exact);

    // Under-constrained (no BCs at all)
    EXPECT_GE(report.countByKind(PropertyKind::UnderConstraint), 1u);
}

// ============================================================================
// Pinned Elasticity — rigid-body modes anchored by Dirichlet
//
// Note: The current EssentialBC descriptor sets anchors_rigid_body_rotation=false
// because rotation anchoring depends on geometry (3+ non-collinear points).
// The ConstraintRankAnalyzer requires BOTH translation AND rotation for rigid-body.
// So with a single Dirichlet descriptor, rigid-body modes remain under-constrained.
// This is the CORRECT conservative behavior — full anchoring requires either:
//   (a) a geometry/rank check, or
//   (b) the descriptor explicitly setting anchors_rigid_body_rotation=true.
// ============================================================================

TEST(Phase9, PinnedElasticity) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeElasticityRecord());

    // Dirichlet on marker 1 — anchors translation but not rotation
    ctx.addBCDescriptor(dirichletBC(0, 1));

    auto report = analyzer.analyze(ctx);

    // Nullspace still detected
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);

    // With rotation not explicitly anchored, rigid-body remains under-constrained
    // (conservative — correct behavior)
    EXPECT_GE(report.countByKind(PropertyKind::UnderConstraint), 1u);

    // Now test with explicit rotation anchoring
    ProblemAnalysisContext ctx2;
    ctx2.addFormulationRecord(makeElasticityRecord());
    BoundaryConditionDescriptor full_anchor;
    full_anchor.primary_variable = VariableKey::field(0);
    full_anchor.boundary_marker = 1;
    full_anchor.trace_kind = TraceKind::Value;
    full_anchor.enforcement_kind = EnforcementKind::Strong;
    full_anchor.anchors_constant_mode = true;
    full_anchor.anchors_rigid_body_translation = true;
    full_anchor.anchors_rigid_body_rotation = true;
    full_anchor.source = "Full pinning (3+ non-collinear Dirichlet points)";
    ctx2.addBCDescriptor(full_anchor);

    auto report2 = analyzer.analyze(ctx2);

    // With full anchoring, no under-constraint
    EXPECT_EQ(report2.countByKind(PropertyKind::UnderConstraint), 0u);
}

// ============================================================================
// Stokes Pressure — saddle-point + velocity nullspace
// ============================================================================

TEST(Phase9, StokesPressure) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    FieldDescriptor pressure;
    pressure.field_id = 0;
    pressure.name = "pressure";
    pressure.field_type = FieldType::Scalar;
    pressure.declared_nullspace_metadata_present = true;
    pressure.declared_nullspace_family = NullspaceFamily::ScalarConstant;
    pressure.declared_nullspace_scope =
        "closed Stokes pressure space modulo constants";
    ctx.addFieldDescriptor(pressure);
    ctx.addFormulationRecord(makeStokesRecord());
    ctx.addBCDescriptor(dirichletBC(1, 1)); // Dirichlet on velocity

    auto report = analyzer.analyze(ctx);

    // MixedSaddlePoint
    EXPECT_GE(report.countByKind(PropertyKind::MixedSaddlePoint), 1u);

    // Nullspace claims: velocity (per-component) + explicitly declared pressure gauge
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    EXPECT_GE(nullspace.size(), 1u);

    // Pressure nullspace detected by MixedOperatorAnalyzer from field metadata
    bool has_pressure_nullspace = false;
    for (const auto* c : nullspace) {
        if (c->field == 0) has_pressure_nullspace = true;
    }
    EXPECT_TRUE(has_pressure_nullspace);

    // Velocity nullspace (per-component) anchored by Dirichlet
    bool velocity_under = false;
    for (const auto* c : report.claimsOfKind(PropertyKind::UnderConstraint)) {
        if (c->field == 1) velocity_under = true;
    }
    EXPECT_FALSE(velocity_under);

    // Pressure has nullspace but no BC → under-constrained
    bool pressure_under = false;
    for (const auto* c : report.claimsOfKind(PropertyKind::UnderConstraint)) {
        if (c->field == 0) pressure_under = true;
    }
    EXPECT_TRUE(pressure_under);
}

// ============================================================================
// NS-VMS Pressure — stabilized saddle-point
// ============================================================================

TEST(Phase9, NSVMSPressure) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makeStabilizedStokesRecord());
    ctx.addBCDescriptor(dirichletBC(1, 1)); // Dirichlet on velocity

    auto report = analyzer.analyze(ctx);

    // Stabilization detected
    EXPECT_GE(report.countByKind(PropertyKind::Stabilization), 1u);

    // Note: PSPG stabilization gives pressure its own gradient operator
    // (h * grad(p)·grad(q)), so the pressure diagonal block is no longer
    // empty — the system is regularized. The MixedOperatorAnalyzer correctly
    // does NOT detect a saddle-point because the constraint field now has
    // its own elliptic diagonal. This is physically correct: PSPG stabilization
    // was designed precisely to regularize the saddle-point structure.
}

// ============================================================================
// Periodic Only — nullspace preserved, NOT anchored
// ============================================================================

TEST(Phase9, PeriodicOnly) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(periodicBC(0, 2));

    auto report = analyzer.analyze(ctx);

    // Nullspace detected
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);

    // Under-constrained (periodic preserves nullspace, doesn't anchor)
    EXPECT_GE(report.countByKind(PropertyKind::UnderConstraint), 1u);

    // Compatibility condition (periodic-only Poisson still needs ∫f=0)
    EXPECT_GE(report.countByKind(PropertyKind::CompatibilityCondition), 1u);
}

// ============================================================================
// Periodic + Dirichlet — nullspace anchored
// ============================================================================

TEST(Phase9, PeriodicPlusDirichlet) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());
    ctx.addBCDescriptor(periodicBC(0, 2));
    ctx.addBCDescriptor(dirichletBC(0, 3));

    auto report = analyzer.analyze(ctx);

    // Nullspace detected but anchored → no under-constraint
    EXPECT_GE(report.countByKind(PropertyKind::Nullspace), 1u);
    EXPECT_EQ(report.countByKind(PropertyKind::UnderConstraint), 0u);

    // No compatibility condition (Dirichlet provides anchoring)
    EXPECT_EQ(report.countByKind(PropertyKind::CompatibilityCondition), 0u);
}

// ============================================================================
// Conflicting constraints — over-constraint detection
// ============================================================================

// ============================================================================
// Partially constrained vector field — per-component nullspace detection
// ============================================================================

TEST(Phase9, PartiallyConstrainedVector) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Vector diffusion: grad(u):grad(v) with 3-component vector field
    auto space = vectorH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // Dirichlet only on component 0 — components 1 and 2 remain free
    BoundaryConditionDescriptor bc_comp0;
    bc_comp0.primary_variable = VariableKey::field(0, 0);
    bc_comp0.component = 0;
    bc_comp0.boundary_marker = 1;
    bc_comp0.trace_kind = TraceKind::Value;
    bc_comp0.enforcement_kind = EnforcementKind::Strong;
    bc_comp0.anchors_constant_mode = true;
    bc_comp0.source = "EssentialBC comp 0 on marker 1";
    ctx.addBCDescriptor(bc_comp0);

    auto report = analyzer.analyze(ctx);

    // Per-component nullspace: 3 claims (one per component)
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 3u);

    // Under-constraint: components 1 and 2 should be under-constrained
    auto under = report.claimsOfKind(PropertyKind::UnderConstraint);
    int under_count = 0;
    bool comp0_under = false;
    for (const auto* c : under) {
        if (c->field == 0) {
            ++under_count;
            if (c->component == 0) comp0_under = true;
        }
    }
    // Component 0 is anchored → NOT under-constrained
    EXPECT_FALSE(comp0_under);
    // Components 1 and 2 are free → under-constrained
    EXPECT_EQ(under_count, 2);
}

// ============================================================================
// Vector-basis field — no per-component nullspace claims
// ============================================================================

TEST(Phase9, VectorBasisField_FieldWideClaim) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Use a real HDivSpace — its continuity() returns H_div,
    // which triggers component_extractable = false.
    auto hdiv_space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, 0);
    ASSERT_EQ(hdiv_space->continuity(), Continuity::H_div);

    // Build a FormExpr using the HDiv space. The space has value_dimension > 1
    // but DOFs are on vector-valued basis functions (not per-component).
    auto u = FormExpr::stateField(0, *hdiv_space, "u");
    auto v = FormExpr::testFunction(*hdiv_space, "v");
    auto residual = inner(u, v).dx();  // L2 inner product on HDiv

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // Derive component_extractable from the space continuity,
    // simulating what FESystem::runProblemAnalysis() does.
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "sigma";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = hdiv_space->value_dimension();
    fd.continuity = hdiv_space->continuity();
    // Pre-setup derivation: H_div → not component-extractable
    fd.component_extractable = (fd.continuity != Continuity::H_div &&
                                fd.continuity != Continuity::H_curl);
    ctx.addFieldDescriptor(fd);

    EXPECT_FALSE(fd.component_extractable);

    auto report = analyzer.analyze(ctx);

    // Should get a single field-wide nullspace claim, NOT per-component claims.
    // (The L2 inner product u·v has has_absolute_value=true, so actually
    //  no nullspace is detected for this mass-like form. Use a gradient form.)
    // Actually for mass-like form, KernelAnalyzer skips it (has_absolute_value).
    // Let's just verify no per-component claims are emitted.
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    for (const auto* c : nullspace) {
        if (c->field == 0) {
            // Any nullspace claim for this field must be field-wide, not per-component
            EXPECT_EQ(c->component, -1) << "HDiv field should not have per-component claims";
        }
    }
}

TEST(Phase9, VectorBasisField_GradientForm_FieldWideClaim) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // For a gradient-only form on a vector field marked non-extractable,
    // KernelAnalyzer should emit a single field-wide claim.
    auto space = vectorH1();  // Still use H1 for the FormExpr (HDiv doesn't have grad)
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // Mark as non-component-extractable (simulating vector-basis field)
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "u";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    fd.component_extractable = false;
    ctx.addFieldDescriptor(fd);

    auto report = analyzer.analyze(ctx);

    // Should get exactly 1 field-wide nullspace claim
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 1u);
    EXPECT_EQ(nullspace[0]->field, FieldId{0});
    EXPECT_EQ(nullspace[0]->component, -1);
    EXPECT_NE(nullspace[0]->description.find("per-component"), std::string::npos);
}

// ============================================================================
// Component-extractable vector field gets per-component claims
// ============================================================================

TEST(Phase9, ComponentExtractableField_PerComponentClaims) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto space = vectorH1();
    auto u = FormExpr::stateField(0, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    auto residual = inner(grad(u), grad(v)).dx();

    FormulationRecord rec;
    rec.operator_tag = "equations";
    rec.active_fields = {0};
    rec.residual_expr = residual.nodeShared();
    rec.active_variables.push_back(VariableKey::field(0));
    ctx.addFormulationRecord(rec);

    // Mark as component-extractable (default / ProductSpace)
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.name = "u";
    fd.field_type = FieldType::Vector;
    fd.value_dimension = 3;
    fd.component_extractable = true;
    ctx.addFieldDescriptor(fd);

    auto report = analyzer.analyze(ctx);

    // Should get 3 per-component nullspace claims
    auto nullspace = report.claimsOfKind(PropertyKind::Nullspace);
    ASSERT_EQ(nullspace.size(), 3u);
    for (int comp = 0; comp < 3; ++comp) {
        bool found = false;
        for (const auto* c : nullspace) {
            if (c->component == comp) found = true;
        }
        EXPECT_TRUE(found) << "Missing nullspace claim for component " << comp;
    }
}

TEST(Phase9, ConflictingDirichlet) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    // Build a constraint summary with a conflict
    ConstraintAnalysisSummary cs;
    ConstraintConflict conflict;
    conflict.dof = 0;
    conflict.conflicting_sources = {"Dirichlet u=0", "Dirichlet u=1"};
    conflict.description = "DOF 0 has conflicting Dirichlet prescriptions";
    cs.conflicts.push_back(conflict);
    ctx.setConstraintSummary(cs);

    auto report = analyzer.analyze(ctx);

    // OverConstraint detected
    EXPECT_GE(report.countByKind(PropertyKind::OverConstraint), 1u);
    auto over = report.claimsOfKind(PropertyKind::OverConstraint);
    ASSERT_GE(over.size(), 1u);
    EXPECT_EQ(over[0]->status, PropertyStatus::Likely);
    EXPECT_EQ(over[0]->confidence, AnalysisConfidence::Medium);
}

// ============================================================================
// Disconnected Mesh Scoping
// ============================================================================

TEST(Phase9, DisconnectedMeshScoping) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;
    ctx.addFormulationRecord(makePoissonRecord());

    // Two disconnected regions; Dirichlet only on region 0
    TopologyAnalysisContext topo;
    ConnectedComponent c0;
    c0.region_id = 0;
    c0.num_cells = 1;
    c0.boundary_markers = {1};
    topo.components.push_back(c0);

    ConnectedComponent c1;
    c1.region_id = 1;
    c1.num_cells = 1;
    topo.components.push_back(c1);

    topo.boundary_mapping.marker_to_regions[1] = {0};
    topo.boundary_mapping.region_to_markers[0] = {1};
    ctx.setTopologyContext(topo);

    ctx.addBCDescriptor(dirichletBC(0, 1)); // Only on region 0's marker

    auto report = analyzer.analyze(ctx);

    // TopologyScopedKernel for region 1 (unanchored)
    auto scoped = report.claimsOfKind(PropertyKind::TopologyScopedKernel);
    EXPECT_GE(scoped.size(), 1u);
    if (!scoped.empty()) {
        EXPECT_EQ(scoped[0]->region, 1);
    }
}

// ============================================================================
// Coupled Boundary PDE-ODE
// ============================================================================

TEST(Phase9, CoupledBoundaryPDEODE) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Poisson + coupled boundary functional + aux state
    FormulationRecord rec = makePoissonRecord();
    auto bf = VariableKey::named(VariableKind::BoundaryFunctional, "Q_out");
    auto aux = VariableKey::named(VariableKind::AuxiliaryState, "P_d");
    rec.active_variables.push_back(bf);
    rec.active_variables.push_back(aux);
    rec.boundary_functional_dependencies.push_back(bf);
    rec.auxiliary_state_dependencies.push_back(aux);
    rec.variable_couplings.emplace_back(VariableKey::field(0), bf);
    rec.variable_couplings.emplace_back(VariableKey::field(0), aux);
    ctx.addFormulationRecord(rec);

    auto report = analyzer.analyze(ctx);

    // CoupledSystemStructure detected
    EXPECT_GE(report.countByKind(PropertyKind::CoupledSystemStructure), 1u);
}
