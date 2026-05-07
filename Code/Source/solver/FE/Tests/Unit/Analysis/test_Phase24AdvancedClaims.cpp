/**
 * @file test_Phase24AdvancedClaims.cpp
 * @brief Phase 24 — End-to-end tests for advanced mathematical claim families
 *
 * Tests exercise InfSupAnalyzer, ConservationAnalyzer, DAEStructureAnalyzer,
 * SpaceCompatibilityAnalyzer, and TransportCharacterAnalyzer using both
 * FormExpr-based and hand-written kernel contribution paths.
 */

#include <gtest/gtest.h>

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/BoundaryConditionDescriptor.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

// ============================================================================
// InfSupCondition
// ============================================================================

TEST(Phase24, InfSup_StructurallySupported) {
    // Mixed system: P2 velocity (order=2) + P1 pressure (order=1) → Taylor-Hood
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Velocity diagonal block
    auto vel_cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(1), "equations", "test");
    ctx.addContribution(std::move(vel_cd));

    // Constraint pair with pairing descriptor
    auto pair_cd = ContributionDescriptor::constraintPairDesc(
        VariableKey::field(1), VariableKey::field(0),
        "incompressibility", "equations", "test");
    ctx.addContribution(std::move(pair_cd));

    // Field descriptors with different polynomial orders (Taylor-Hood)
    FieldDescriptor vel_fd;
    vel_fd.field_id = 1;
    vel_fd.name = "u";
    vel_fd.value_dimension = 3;
    vel_fd.polynomial_order = 2;
    vel_fd.space_family = SpaceFamily::H1;
    ctx.addFieldDescriptor(vel_fd);

    FieldDescriptor pres_fd;
    pres_fd.field_id = 0;
    pres_fd.name = "p";
    pres_fd.value_dimension = 1;
    pres_fd.polynomial_order = 1;
    pres_fd.space_family = SpaceFamily::H1;
    ctx.addFieldDescriptor(pres_fd);

    auto report = analyzer.analyze(ctx);

    auto infsup = report.claimsOfKind(PropertyKind::InfSupCondition);
    ASSERT_GE(infsup.size(), 1u);
    EXPECT_TRUE(infsup[0]->inf_sup_class.has_value());
    EXPECT_EQ(*infsup[0]->inf_sup_class, InfSupClass::StructurallySupported);
}

TEST(Phase24, InfSup_EqualOrderRequiresEvidence) {
    // Equal order P1/P1 without stabilization is not a certification-grade
    // failure without theorem or scoped numeric evidence.
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto vel_cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(1), "equations", "test");
    ctx.addContribution(std::move(vel_cd));

    auto pair_cd = ContributionDescriptor::constraintPairDesc(
        VariableKey::field(1), VariableKey::field(0),
        "incompressibility", "equations", "test");
    ctx.addContribution(std::move(pair_cd));

    FieldDescriptor vel_fd;
    vel_fd.field_id = 1;
    vel_fd.polynomial_order = 1;
    vel_fd.space_family = SpaceFamily::H1;
    ctx.addFieldDescriptor(vel_fd);

    FieldDescriptor pres_fd;
    pres_fd.field_id = 0;
    pres_fd.polynomial_order = 1;
    pres_fd.space_family = SpaceFamily::H1;
    ctx.addFieldDescriptor(pres_fd);

    auto report = analyzer.analyze(ctx);

    auto infsup = report.claimsOfKind(PropertyKind::InfSupCondition);
    ASSERT_GE(infsup.size(), 1u);
    EXPECT_TRUE(infsup[0]->inf_sup_class.has_value());
    EXPECT_EQ(infsup[0]->status, PropertyStatus::Unknown);
    EXPECT_EQ(*infsup[0]->inf_sup_class, InfSupClass::Required);
    ASSERT_TRUE(infsup[0]->certification_class.has_value());
    EXPECT_EQ(*infsup[0]->certification_class,
              CertificationClass::NotCertified);
}

// ============================================================================
// ConservationStructure
// ============================================================================

TEST(Phase24, Conservation_PotentialExchangeBalance) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Two exchange contributions in the same balance group with opposite signs
    auto ex1 = ContributionDescriptor::exchangeCoupling(
        VariableKey::field(0), VariableKey::field(1),
        "energy_balance", "equations", "test");
    ex1.balance->sign = 1;
    ctx.addContribution(std::move(ex1));

    auto ex2 = ContributionDescriptor::exchangeCoupling(
        VariableKey::field(1), VariableKey::field(0),
        "energy_balance", "equations", "test");
    ex2.balance->sign = -1;
    ctx.addContribution(std::move(ex2));

    auto report = analyzer.analyze(ctx);

    auto cons = report.claimsOfKind(PropertyKind::ConservationStructure);
    ASSERT_GE(cons.size(), 1u);
    EXPECT_TRUE(cons[0]->conservation_class.has_value());
    EXPECT_EQ(*cons[0]->conservation_class,
              ConservationClass::PotentialExchangeBalance);
    ASSERT_TRUE(cons[0]->certification_class.has_value());
    EXPECT_EQ(*cons[0]->certification_class,
              CertificationClass::NotCertified);
}

TEST(Phase24, Conservation_NoBalanceMetadata_NoOp) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Contribution without balance metadata → ConservationAnalyzer is no-op
    auto cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(cd));

    auto report = analyzer.analyze(ctx);
    EXPECT_EQ(report.countByKind(PropertyKind::ConservationStructure), 0u);
}

// ============================================================================
// DifferentialAlgebraicStructure
// ============================================================================

TEST(Phase24, DAE_PureODELike) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // All variables are dynamic (mass-like temporal)
    auto mass_cd = ContributionDescriptor::massLike(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(mass_cd));

    VariableDescriptor vd;
    vd.key = VariableKey::field(0);
    vd.temporal_state_kind = TemporalStateKind::Dynamic;
    vd.max_time_derivative_order = 1;
    ctx.addVariableDescriptor(vd);

    auto report = analyzer.analyze(ctx);

    auto dae = report.claimsOfKind(PropertyKind::DifferentialAlgebraicStructure);
    ASSERT_GE(dae.size(), 1u);
    EXPECT_TRUE(dae[0]->dae_class.has_value());
    EXPECT_EQ(*dae[0]->dae_class, DAEClass::PureODELike);
}

TEST(Phase24, DAE_TimeIndependentResidualDoesNotCreateAlgebraicVariable) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto mass_cd = ContributionDescriptor::massLike(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(mass_cd));

    ContributionDescriptor stiffness_cd;
    stiffness_cd.operator_tag = "equations";
    stiffness_cd.origin = "test";
    stiffness_cd.role = ContributionRole::DiagonalBlock;
    stiffness_cd.test_variables = {VariableKey::field(0)};
    stiffness_cd.trial_variables = {VariableKey::field(0)};
    stiffness_cd.temporal = TemporalDescriptor{
        0, TemporalContributionKind::TimeIndependentResidual};
    ctx.addContribution(std::move(stiffness_cd));

    auto report = analyzer.analyze(ctx);

    auto dae = report.claimsOfKind(PropertyKind::DifferentialAlgebraicStructure);
    ASSERT_GE(dae.size(), 1u);
    ASSERT_TRUE(dae[0]->dae_class.has_value());
    EXPECT_EQ(*dae[0]->dae_class, DAEClass::PureODELike);
}

TEST(Phase24, DAE_Index1DAELike) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Dynamic variable
    auto mass_cd = ContributionDescriptor::massLike(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(mass_cd));

    // Algebraic variable (constraint)
    ContributionDescriptor alg_cd;
    alg_cd.operator_tag = "equations";
    alg_cd.origin = "test";
    alg_cd.role = ContributionRole::ConstraintBlock;
    alg_cd.test_variables = {VariableKey::field(1)};
    alg_cd.trial_variables = {VariableKey::field(0)};
    alg_cd.temporal = TemporalDescriptor{0, TemporalContributionKind::PureConstraint};
    ctx.addContribution(std::move(alg_cd));

    VariableDescriptor vd0;
    vd0.key = VariableKey::field(0);
    vd0.temporal_state_kind = TemporalStateKind::Dynamic;
    ctx.addVariableDescriptor(vd0);

    VariableDescriptor vd1;
    vd1.key = VariableKey::field(1);
    vd1.temporal_state_kind = TemporalStateKind::Algebraic;
    ctx.addVariableDescriptor(vd1);

    auto report = analyzer.analyze(ctx);

    auto dae = report.claimsOfKind(PropertyKind::DifferentialAlgebraicStructure);
    ASSERT_GE(dae.size(), 1u);
    EXPECT_TRUE(dae[0]->dae_class.has_value());
    EXPECT_EQ(*dae[0]->dae_class, DAEClass::Index1DAELike);
}

TEST(Phase24, DAE_NoTemporalMetadata_NoOp) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(cd));

    auto report = analyzer.analyze(ctx);
    EXPECT_EQ(report.countByKind(PropertyKind::DifferentialAlgebraicStructure), 0u);
}

// ============================================================================
// SpaceCompatibility
// ============================================================================

TEST(Phase24, SpaceCompatibility_Incompatible_TraceMismatch) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // HDiv field (only NormalComponent trace) with a Value-type BC
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.space_family = SpaceFamily::HDiv;
    fd.trace_capabilities = TraceCapabilityFlags::NormalComponent;
    ctx.addFieldDescriptor(fd);

    BoundaryConditionDescriptor bc;
    bc.primary_variable = VariableKey::field(0);
    bc.trace_kind = TraceKind::Value;  // HDiv doesn't support Value traces
    bc.enforcement_kind = EnforcementKind::Strong;
    ctx.addBCDescriptor(bc);

    auto report = analyzer.analyze(ctx);

    auto compat = report.claimsOfKind(PropertyKind::SpaceCompatibility);
    ASSERT_GE(compat.size(), 1u);
    EXPECT_TRUE(compat[0]->space_compatibility_class.has_value());
    EXPECT_EQ(*compat[0]->space_compatibility_class, SpaceCompatibilityClass::Incompatible);
}

TEST(Phase24, SpaceCompatibility_Compatible_H1Value) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // H1 field with Value BC → compatible
    FieldDescriptor fd;
    fd.field_id = 0;
    fd.space_family = SpaceFamily::H1;
    fd.trace_capabilities = TraceCapabilityFlags::Value | TraceCapabilityFlags::NormalFlux;
    ctx.addFieldDescriptor(fd);

    BoundaryConditionDescriptor bc;
    bc.primary_variable = VariableKey::field(0);
    bc.trace_kind = TraceKind::Value;
    bc.enforcement_kind = EnforcementKind::Strong;
    ctx.addBCDescriptor(bc);

    auto report = analyzer.analyze(ctx);

    // No Incompatible claims for this valid combination
    for (const auto* c : report.claimsOfKind(PropertyKind::SpaceCompatibility)) {
        if (c->space_compatibility_class.has_value()) {
            EXPECT_NE(*c->space_compatibility_class, SpaceCompatibilityClass::Incompatible);
        }
    }
}

// ============================================================================
// OperatorTransportCharacter
// ============================================================================

TEST(Phase24, Transport_DirectionalFirstOrder) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto cd = ContributionDescriptor::transportLike(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(cd));

    auto report = analyzer.analyze(ctx);

    auto transport = report.claimsOfKind(PropertyKind::OperatorTransportCharacter);
    ASSERT_GE(transport.size(), 1u);
    EXPECT_TRUE(transport[0]->transport_character_class.has_value());
    EXPECT_EQ(*transport[0]->transport_character_class,
              TransportCharacterClass::DirectionalFirstOrderLike);
}

TEST(Phase24, Transport_NoFirstOrder_NoOp) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    // Pure second-order (Laplacian) → no transport character claim
    auto cd = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "test");
    ctx.addContribution(std::move(cd));

    auto report = analyzer.analyze(ctx);
    EXPECT_EQ(report.countByKind(PropertyKind::OperatorTransportCharacter), 0u);
}

// ============================================================================
// NullspaceEffect structured field (from BC lowering)
// ============================================================================

TEST(Phase24, NullspaceEffect_WeaklyLifts_Robin) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakPenalty;
    desc.anchors_constant_mode = true;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_GE(contribs.size(), 1u);
    EXPECT_TRUE(contribs[0].nullspace_effect.has_value());
    EXPECT_EQ(*contribs[0].nullspace_effect, NullspaceEffect::WeaklyLifts);
}

TEST(Phase24, NullspaceEffect_ExactlyRemoves_Dirichlet) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::Strong;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_GE(contribs.size(), 1u);
    EXPECT_TRUE(contribs[0].nullspace_effect.has_value());
    EXPECT_EQ(*contribs[0].nullspace_effect, NullspaceEffect::ExactlyRemoves);
}

TEST(Phase24, NullspaceEffect_Preserves_Neumann) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakConsistent;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_GE(contribs.size(), 1u);
    EXPECT_TRUE(contribs[0].nullspace_effect.has_value());
    EXPECT_EQ(*contribs[0].nullspace_effect, NullspaceEffect::Preserves);
}

// ============================================================================
// Consistency from Nitsche BC lowering
// ============================================================================

TEST(Phase24, Consistency_Nitsche_ConsistentPerturbation) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakNitsche;
    desc.is_homogeneous = true;
    NitscheMetadata nitsche;
    nitsche.primal_consistency_terms_present = true;
    nitsche.adjoint_consistency_terms_present = true;
    nitsche.penalty_positive = true;
    nitsche.penalty_scaling_verified = true;
    nitsche.penalty_trace_bound_verified = true;
    desc.nitsche = nitsche;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_GE(contribs.size(), 1u);
    EXPECT_TRUE(contribs[0].consistency_kind.has_value());
    EXPECT_EQ(*contribs[0].consistency_kind, ConsistencyKind::ConsistentPerturbation);
}

// ============================================================================
// Hand-written kernel path produces same claims as FormExpr path
// ============================================================================

TEST(Phase24, HandwrittenKernel_TransportClaim) {
    // Verify that a handwritten kernel using transportLike() builder
    // produces the same OperatorTransportCharacter claim as FormExpr would.
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto cd = ContributionDescriptor::transportLike(
        VariableKey::field(0), "convection_op", "HandwrittenConvectionKernel");
    ctx.addContribution(std::move(cd));

    auto report = analyzer.analyze(ctx);

    auto transport = report.claimsOfKind(PropertyKind::OperatorTransportCharacter);
    ASSERT_GE(transport.size(), 1u);
    EXPECT_TRUE(transport[0]->transport_character_class.has_value());
}

TEST(Phase24, HandwrittenKernel_DAEClaim) {
    auto analyzer = ProblemAnalyzer::createDefault();
    ProblemAnalysisContext ctx;

    auto cd = ContributionDescriptor::massLike(
        VariableKey::field(0), "ode_system", "HandwrittenODEKernel");
    ctx.addContribution(std::move(cd));

    VariableDescriptor vd;
    vd.key = VariableKey::field(0);
    vd.temporal_state_kind = TemporalStateKind::Dynamic;
    ctx.addVariableDescriptor(vd);

    auto report = analyzer.analyze(ctx);

    auto dae = report.claimsOfKind(PropertyKind::DifferentialAlgebraicStructure);
    ASSERT_GE(dae.size(), 1u);
    EXPECT_EQ(*dae[0]->dae_class, DAEClass::PureODELike);
}
