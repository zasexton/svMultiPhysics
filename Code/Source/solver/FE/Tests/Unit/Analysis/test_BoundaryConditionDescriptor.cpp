/**
 * @file test_BoundaryConditionDescriptor.cpp
 * @brief Unit tests for BoundaryConditionDescriptor and compatibility adapter
 */

#include <gtest/gtest.h>

#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Constraints/GaugeRegistry.h"

#include "Forms/StandardBCs.h"
#include "Forms/ConstraintBCs.h"
#include "Forms/NitscheBC.h"
#include "Forms/FormExpr.h"

#include "Spaces/H1Space.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;
using namespace svmp::FE::forms;
using namespace svmp::FE::gauge;

// ============================================================================
// TraceKind / EnforcementKind toString
// ============================================================================

TEST(BoundaryConditionDescriptor, ToString_TraceKind) {
    EXPECT_STREQ(toString(TraceKind::Value), "Value");
    EXPECT_STREQ(toString(TraceKind::NormalComponent), "NormalComponent");
    EXPECT_STREQ(toString(TraceKind::TangentialComponent), "TangentialComponent");
    EXPECT_STREQ(toString(TraceKind::Flux), "Flux");
    EXPECT_STREQ(toString(TraceKind::NormalFlux), "NormalFlux");
    EXPECT_STREQ(toString(TraceKind::Mixed), "Mixed");
    EXPECT_STREQ(toString(TraceKind::AlgebraicRelation), "AlgebraicRelation");
}

TEST(BoundaryConditionDescriptor, ToString_EnforcementKind) {
    EXPECT_STREQ(toString(EnforcementKind::Strong), "Strong");
    EXPECT_STREQ(toString(EnforcementKind::WeakConsistent), "WeakConsistent");
    EXPECT_STREQ(toString(EnforcementKind::WeakPenalty), "WeakPenalty");
    EXPECT_STREQ(toString(EnforcementKind::WeakNitsche), "WeakNitsche");
    EXPECT_STREQ(toString(EnforcementKind::WeakInequality), "WeakInequality");
    EXPECT_STREQ(toString(EnforcementKind::AffineRelation), "AffineRelation");
}

TEST(BoundaryConditionDescriptor, ToString_InequalitySense) {
    EXPECT_STREQ(toString(InequalitySense::None), "None");
    EXPECT_STREQ(toString(InequalitySense::LessEqual), "LessEqual");
    EXPECT_STREQ(toString(InequalitySense::GreaterEqual), "GreaterEqual");
    EXPECT_STREQ(toString(InequalitySense::Complementarity), "Complementarity");
}

TEST(BoundaryConditionDescriptor, ToString_NitscheVariant) {
    EXPECT_STREQ(toString(NitscheVariant::Unknown), "Unknown");
    EXPECT_STREQ(toString(NitscheVariant::Symmetric), "Symmetric");
    EXPECT_STREQ(toString(NitscheVariant::Nonsymmetric), "Nonsymmetric");
    EXPECT_STREQ(toString(NitscheVariant::Skew), "Skew");
    EXPECT_STREQ(toString(NitscheVariant::Incomplete), "Incomplete");
}

// ============================================================================
// EssentialBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, EssentialBC_StrongValueAnchorsAll) {
    bc::EssentialBC bc(3, FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.primary_variable, VariableKey::field(0));
    EXPECT_EQ(d.boundary_marker, 3);
    EXPECT_EQ(d.trace_kind, TraceKind::Value);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::Strong);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    EXPECT_NE(d.source.find("EssentialBC"), std::string::npos);
    EXPECT_NE(d.source.find("3"), std::string::npos);
}

TEST(BoundaryConditionDescriptor, EssentialBC_MultiComponent_PerComponentDescriptors) {
    // 3-component vector Dirichlet BC
    std::vector<FormExpr> comps = {
        FormExpr::constant(0.0),
        FormExpr::constant(0.0),
        FormExpr::constant(0.0)
    };
    bc::EssentialBC bc(5, std::move(comps));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    // Should produce 3 per-component descriptors
    ASSERT_EQ(descs.size(), 3u);
    for (int comp = 0; comp < 3; ++comp) {
        EXPECT_EQ(descs[static_cast<std::size_t>(comp)].component, comp);
        EXPECT_EQ(descs[static_cast<std::size_t>(comp)].primary_variable,
                  VariableKey::field(0, comp));
        EXPECT_EQ(descs[static_cast<std::size_t>(comp)].boundary_marker, 5);
        EXPECT_TRUE(descs[static_cast<std::size_t>(comp)].anchors_constant_mode);
        EXPECT_NE(descs[static_cast<std::size_t>(comp)].source.find("comp " + std::to_string(comp)),
                  std::string::npos);
    }
}

TEST(BoundaryConditionDescriptor, NormalTraceEssentialBC_StrongNormalComponentAnchors) {
    bc::NormalTraceEssentialBC bc(9, FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::Strong);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
}

// ============================================================================
// NaturalBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, NaturalBC_FluxAnchorsNothing) {
    bc::NaturalBC bc(5, FormExpr::constant(1.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::Flux);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakConsistent);
    EXPECT_FALSE(d.anchors_constant_mode);
    EXPECT_FALSE(d.anchors_rigid_body_translation);
    EXPECT_FALSE(d.anchors_rigid_body_rotation);
}

TEST(BoundaryConditionDescriptor, TraceLoadBC_WeakConsistentNormalComponentAnchorsNothing) {
    bc::TraceLoadBC bc(6, FormExpr::constant(1.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakConsistent);
    EXPECT_FALSE(d.anchors_constant_mode);
    EXPECT_FALSE(d.anchors_rigid_body_translation);
    EXPECT_FALSE(d.anchors_rigid_body_rotation);
}

// ============================================================================
// RobinBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, RobinBC_MixedAnchorsConstantNotRotation) {
    bc::RobinBC bc(2, FormExpr::constant(10.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::Mixed);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakPenalty);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    EXPECT_FALSE(d.anchors_rigid_body_rotation);
}

TEST(BoundaryConditionDescriptor, TraceRobinBC_WeakPenaltyNormalComponentAnchors) {
    bc::TraceRobinBC bc(8, FormExpr::constant(3.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakPenalty);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    EXPECT_FALSE(d.anchors_rigid_body_rotation);
}

TEST(BoundaryConditionDescriptor, TraceInequalityBC_WeakInequalityNormalComponentIsStateDependent) {
    bc::TraceInequalityOptions opts;
    opts.sense = bc::TraceInequalitySense::GreaterEqual;
    bc::TraceInequalityBC bc(15,
                             FormExpr::constant(0.0),
                             FormExpr::constant(4.0),
                             opts);
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakInequality);
    ASSERT_TRUE(d.inequality_sense.has_value());
    EXPECT_EQ(*d.inequality_sense, InequalitySense::GreaterEqual);
    EXPECT_TRUE(d.state_dependent_activation);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
}

TEST(BoundaryConditionDescriptor, InterfaceTraceLoadBC_WeakConsistentNormalComponentOnInterface) {
    bc::InterfaceTraceLoadBC bc(11, FormExpr::constant(1.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.domain, DomainKind::InterfaceFace);
    EXPECT_EQ(d.interface_marker, 11);
    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakConsistent);
    EXPECT_FALSE(d.anchors_constant_mode);
}

TEST(BoundaryConditionDescriptor, InterfaceTraceRobinBC_WeakPenaltyNormalComponentOnInterface) {
    bc::InterfaceTraceRobinBC bc(12, FormExpr::constant(3.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.domain, DomainKind::InterfaceFace);
    EXPECT_EQ(d.interface_marker, 12);
    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakPenalty);
    EXPECT_TRUE(d.anchors_constant_mode);
}

TEST(BoundaryConditionDescriptor, InterfaceTraceJumpPenaltyBC_WeakPenaltyNormalComponentOnInterface) {
    bc::InterfaceTraceJumpPenaltyBC bc(13, FormExpr::constant(5.0), FormExpr::constant(0.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.domain, DomainKind::InterfaceFace);
    EXPECT_EQ(d.interface_marker, 13);
    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakPenalty);
    EXPECT_TRUE(d.anchors_constant_mode);
}

// ============================================================================
// NitscheBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, NitscheBC_WeakNitscheAnchorsAll) {
    bc::ScalarNitscheBC bc(7, FormExpr::constant(0.0), FormExpr::constant(1.0),
                           /*penalty=*/10.0, /*symmetric=*/true);
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::Value);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakNitsche);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    ASSERT_TRUE(d.nitsche.has_value());
    EXPECT_EQ(d.nitsche->variant, NitscheVariant::Symmetric);
    EXPECT_TRUE(d.nitsche->primal_consistency_terms_present);
    EXPECT_TRUE(d.nitsche->adjoint_consistency_terms_present);
    EXPECT_TRUE(d.nitsche->rhs_consistency_terms_present);
    EXPECT_TRUE(d.nitsche->penalty_positive);
}

TEST(BoundaryConditionDescriptor, TraceNitscheBC_WeakNitscheNormalComponentAnchorsAll) {
    bc::TraceNitscheBC bc(10,
                          FormExpr::constant(0.0),
                          FormExpr::constant(1.0),
                          FormExpr::constant(1.0),
                          FormExpr::constant(2.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakNitsche);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    ASSERT_TRUE(d.nitsche.has_value());
    EXPECT_EQ(d.nitsche->variant, NitscheVariant::Symmetric);
    EXPECT_TRUE(d.nitsche->primal_consistency_terms_present);
}

TEST(BoundaryConditionDescriptor, InterfaceTraceNitscheBC_WeakNitscheNormalComponentOnInterface) {
    bc::InterfaceTraceNitscheBC bc(14,
                                   FormExpr::constant(0.0),
                                   FormExpr::constant(1.0),
                                   FormExpr::constant(1.0),
                                   FormExpr::constant(3.0));
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.domain, DomainKind::InterfaceFace);
    EXPECT_EQ(d.interface_marker, 14);
    EXPECT_EQ(d.trace_kind, TraceKind::NormalComponent);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::WeakNitsche);
    EXPECT_TRUE(d.anchors_constant_mode);
    EXPECT_TRUE(d.anchors_rigid_body_translation);
    ASSERT_TRUE(d.nitsche.has_value());
    EXPECT_EQ(d.nitsche->variant, NitscheVariant::Symmetric);
    EXPECT_TRUE(d.nitsche->primal_consistency_terms_present);
}

// ============================================================================
// PeriodicBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, PeriodicBC_AlgebraicRelationAnchorsNothing) {
    bc::PeriodicBC bc(1, 2, {1.0, 0.0, 0.0});
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);

    ASSERT_EQ(descs.size(), 1u);
    const auto& d = descs[0];

    EXPECT_EQ(d.trace_kind, TraceKind::AlgebraicRelation);
    EXPECT_EQ(d.enforcement_kind, EnforcementKind::AffineRelation);
    EXPECT_FALSE(d.anchors_constant_mode);
    EXPECT_FALSE(d.anchors_rigid_body_translation);
    EXPECT_FALSE(d.anchors_rigid_body_rotation);
}

// ============================================================================
// ReservedBC descriptor
// ============================================================================

TEST(BoundaryConditionDescriptor, ReservedBC_EmptyDescriptors) {
    bc::ReservedBC bc(1);
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);
    EXPECT_TRUE(descs.empty());
}

// ============================================================================
// Default BoundaryCondition base class
// ============================================================================

TEST(BoundaryConditionDescriptor, DefaultBase_ReturnsEmpty) {
    // The base class default returns empty — conservative (no claims)
    // We can't instantiate BoundaryCondition directly (abstract), but
    // ReservedBC delegates to base default behavior via explicit empty return
    bc::ReservedBC bc(0);
    auto descs = bc.analysisMetadata(/*field_id=*/0, nullptr);
    EXPECT_TRUE(descs.empty());
}

// ============================================================================
// Compatibility adapter: descriptor → verdict roundtrip
// ============================================================================

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_ScalarConstant_Anchored) {
    BoundaryConditionDescriptor d;
    d.anchors_constant_mode = true;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Anchored);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_ComponentwiseConstant_Anchored) {
    BoundaryConditionDescriptor d;
    d.anchors_constant_mode = true;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ComponentwiseConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Anchored);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_KernelOfSymGrad_FullyAnchored) {
    BoundaryConditionDescriptor d;
    d.anchors_rigid_body_translation = true;
    d.anchors_rigid_body_rotation = true;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::KernelOfSymGrad);
    EXPECT_EQ(verdict, AnchoringVerdict::Anchored);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_KernelOfSymGrad_PartiallyAnchored) {
    BoundaryConditionDescriptor d;
    d.anchors_rigid_body_translation = true;
    d.anchors_rigid_body_rotation = false;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::KernelOfSymGrad);
    EXPECT_EQ(verdict, AnchoringVerdict::PartiallyAnchored);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_Neumann_Preserved) {
    BoundaryConditionDescriptor d;
    d.trace_kind = TraceKind::Flux;
    d.enforcement_kind = EnforcementKind::WeakConsistent;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Preserved);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_WeakConsistentNormalComponentPreserved) {
    BoundaryConditionDescriptor d;
    d.trace_kind = TraceKind::NormalComponent;
    d.enforcement_kind = EnforcementKind::WeakConsistent;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Preserved);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_Periodic_Preserved) {
    BoundaryConditionDescriptor d;
    d.enforcement_kind = EnforcementKind::AffineRelation;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Preserved);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_NoAnchoring_Unknown) {
    BoundaryConditionDescriptor d;
    // Default descriptor with no special flags

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::Unknown);
}

// ============================================================================
// BC.analysisMetadata() → descriptorToVerdict produces expected verdicts
// ============================================================================

TEST(BoundaryConditionDescriptor, RobinBC_DescriptorToVerdict) {
    bc::RobinBC bc(1, FormExpr::constant(1.0), FormExpr::constant(0.0));

    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);

    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant),
              AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant),
              AnchoringVerdict::Anchored);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad),
              AnchoringVerdict::PartiallyAnchored);
}

TEST(BoundaryConditionDescriptor, NaturalBC_DescriptorToVerdict) {
    bc::NaturalBC bc(1, FormExpr::constant(0.0));

    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);

    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant),
              AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant),
              AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad),
              AnchoringVerdict::Preserved);
}

TEST(BoundaryConditionDescriptor, DescriptorToVerdict_WeakInequality_PartiallyAnchored) {
    BoundaryConditionDescriptor d;
    d.enforcement_kind = EnforcementKind::WeakInequality;
    d.anchors_constant_mode = true;

    auto verdict = descriptorToVerdict(d, NullspaceModeFamily::ScalarConstant);
    EXPECT_EQ(verdict, AnchoringVerdict::PartiallyAnchored);
}

TEST(BoundaryConditionDescriptor, PeriodicBC_DescriptorToVerdict) {
    bc::PeriodicBC bc(1, 2, {1.0, 0.0, 0.0});

    auto descs = bc.analysisMetadata(0, nullptr);
    ASSERT_EQ(descs.size(), 1u);

    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ScalarConstant),
              AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::ComponentwiseConstant),
              AnchoringVerdict::Preserved);
    EXPECT_EQ(descriptorToVerdict(descs[0], NullspaceModeFamily::KernelOfSymGrad),
              AnchoringVerdict::Preserved);
}

// ============================================================================
// lowerBCDescriptor — BC → ContributionDescriptor lowering
// ============================================================================

TEST(BoundaryConditionDescriptor, LowerDirichlet) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.boundary_marker = 3;
    desc.enforcement_kind = EnforcementKind::Strong;
    desc.anchors_constant_mode = true;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
    EXPECT_EQ(contribs[0].boundary_marker, 3);
    // Phase 22 extended metadata
    EXPECT_TRUE(contribs[0].nullspace_effect.has_value());
    EXPECT_EQ(*contribs[0].nullspace_effect, NullspaceEffect::ExactlyRemoves);
    EXPECT_TRUE(contribs[0].consistency_kind.has_value());
    EXPECT_EQ(*contribs[0].consistency_kind, ConsistencyKind::ExactContinuum);
}

TEST(BoundaryConditionDescriptor, LowerPeriodic) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::AffineRelation;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_EQ(contribs[0].role, ContributionRole::ConstraintBlock);
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspacePreserving));
    EXPECT_FALSE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
}

TEST(BoundaryConditionDescriptor, LowerRobin) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.boundary_marker = 2;
    desc.enforcement_kind = EnforcementKind::WeakPenalty;
    desc.anchors_constant_mode = true;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::HasMass));
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
}

TEST(BoundaryConditionDescriptor, LowerNitsche) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakNitsche;

    auto contribs = lowerBCDescriptor(desc);
    // Nitsche produces TWO contributions: BoundaryConstraint + StabilizationBlock
    ASSERT_EQ(contribs.size(), 2u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::HasSecondOrder));
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
    EXPECT_EQ(contribs[1].role, ContributionRole::StabilizationBlock);
    EXPECT_TRUE(hasFlag(contribs[1].traits, OperatorTraitFlags::HasMass));
    EXPECT_EQ(contribs[0].operator_tag, "bc:nitsche:consistency");
    EXPECT_EQ(contribs[1].operator_tag, "bc:nitsche:penalty");
    EXPECT_FALSE(contribs[0].contribution_id.empty());
    EXPECT_FALSE(contribs[1].contribution_id.empty());
    EXPECT_NE(contribs[0].contribution_id, contribs[1].contribution_id);
    ASSERT_TRUE(contribs[0].consistency_kind.has_value());
    EXPECT_EQ(*contribs[0].consistency_kind, ConsistencyKind::Unknown);
    ASSERT_TRUE(contribs[0].adjoint_consistency.has_value());
    EXPECT_EQ(*contribs[0].adjoint_consistency,
              AdjointConsistencyKind::Unknown);
}

TEST(BoundaryConditionDescriptor, LowerNitsche_UsesExplicitConsistencyMetadata) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakNitsche;
    desc.is_homogeneous = false;
    NitscheMetadata nitsche;
    nitsche.variant = NitscheVariant::Symmetric;
    nitsche.primal_consistency_terms_present = true;
    nitsche.adjoint_consistency_terms_present = true;
    nitsche.rhs_consistency_terms_present = true;
    nitsche.penalty_positive = true;
    nitsche.penalty_scaling_verified = true;
    nitsche.penalty_trace_bound_verified = true;
    desc.nitsche = nitsche;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 2u);
    ASSERT_TRUE(contribs[0].consistency_kind.has_value());
    EXPECT_EQ(*contribs[0].consistency_kind,
              ConsistencyKind::ConsistentPerturbation);
    ASSERT_TRUE(contribs[0].adjoint_consistency.has_value());
    EXPECT_EQ(*contribs[0].adjoint_consistency,
              AdjointConsistencyKind::Yes);
    ASSERT_TRUE(contribs[1].consistency_kind.has_value());
    EXPECT_EQ(*contribs[1].consistency_kind,
              ConsistencyKind::ConsistentPerturbation);
}

TEST(BoundaryConditionDescriptor, LowerWeakInequality) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.boundary_marker = 4;
    desc.enforcement_kind = EnforcementKind::WeakInequality;
    desc.anchors_constant_mode = true;
    desc.state_dependent_activation = true;
    desc.inequality_sense = InequalitySense::LessEqual;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::HasMass));
    EXPECT_TRUE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
    ASSERT_TRUE(contribs[0].nullspace_effect.has_value());
    EXPECT_EQ(*contribs[0].nullspace_effect, NullspaceEffect::WeaklyLifts);
    ASSERT_TRUE(contribs[0].consistency_kind.has_value());
    EXPECT_EQ(*contribs[0].consistency_kind, ConsistencyKind::ConsistentPerturbation);
}

TEST(BoundaryConditionDescriptor, LowerNeumann) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.enforcement_kind = EnforcementKind::WeakConsistent;
    desc.trace_kind = TraceKind::Flux;

    auto contribs = lowerBCDescriptor(desc);
    ASSERT_EQ(contribs.size(), 1u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_FALSE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspaceLifting));
    EXPECT_FALSE(hasFlag(contribs[0].traits, OperatorTraitFlags::NullspacePreserving));
}

TEST(BoundaryConditionDescriptor, LowerCoupledBC_EmitsGlobalCoupling) {
    BoundaryConditionDescriptor desc;
    desc.primary_variable = VariableKey::field(0);
    desc.domain = DomainKind::CoupledBoundary;
    desc.enforcement_kind = EnforcementKind::WeakConsistent;
    desc.introduces_global_coupling = true;
    desc.related_variables = {
        VariableKey::named(VariableKind::AuxiliaryState, "P_d")
    };

    auto contribs = lowerBCDescriptor(desc);
    // Should produce: BoundaryConstraint + GlobalCoupling
    ASSERT_EQ(contribs.size(), 2u);
    EXPECT_EQ(contribs[0].role, ContributionRole::BoundaryConstraint);
    EXPECT_EQ(contribs[1].role, ContributionRole::GlobalCoupling);
    EXPECT_EQ(contribs[1].domain, DomainKind::CoupledBoundary);
    ASSERT_EQ(contribs[1].trial_variables.size(), 1u);
    EXPECT_EQ(contribs[1].trial_variables[0].kind, VariableKind::AuxiliaryState);
}
