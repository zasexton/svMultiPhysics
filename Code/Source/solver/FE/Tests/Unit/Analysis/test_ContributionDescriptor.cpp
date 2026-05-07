/**
 * @file test_ContributionDescriptor.cpp
 * @brief Unit tests for ContributionDescriptor — normalized operator IR
 */

#include <gtest/gtest.h>

#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ProblemAnalysisContext.h"

using namespace svmp::FE;
using namespace svmp::FE::analysis;

// ============================================================================
// toString
// ============================================================================

TEST(ContributionDescriptor, ToString_ContributionRole) {
    EXPECT_STREQ(toString(ContributionRole::DiagonalBlock), "DiagonalBlock");
    EXPECT_STREQ(toString(ContributionRole::OffDiagonalBlock), "OffDiagonalBlock");
    EXPECT_STREQ(toString(ContributionRole::ConstraintBlock), "ConstraintBlock");
    EXPECT_STREQ(toString(ContributionRole::StabilizationBlock), "StabilizationBlock");
    EXPECT_STREQ(toString(ContributionRole::BoundaryConstraint), "BoundaryConstraint");
    EXPECT_STREQ(toString(ContributionRole::GlobalCoupling), "GlobalCoupling");
    EXPECT_STREQ(toString(ContributionRole::FieldToAuxiliary), "FieldToAuxiliary");
    EXPECT_STREQ(toString(ContributionRole::AuxiliaryToField), "AuxiliaryToField");
    EXPECT_STREQ(toString(ContributionRole::AuxiliaryToAuxiliary), "AuxiliaryToAuxiliary");
    EXPECT_STREQ(toString(ContributionRole::AuxiliarySelf), "AuxiliarySelf");
}

TEST(ContributionDescriptor, ToString_NullspaceFamily) {
    EXPECT_STREQ(toString(NullspaceFamily::ScalarConstant), "ScalarConstant");
    EXPECT_STREQ(toString(NullspaceFamily::ComponentwiseConstant), "ComponentwiseConstant");
    EXPECT_STREQ(toString(NullspaceFamily::KernelOfSymGrad), "KernelOfSymGrad");
    EXPECT_STREQ(toString(NullspaceFamily::UserDefined), "UserDefined");
}

TEST(ContributionDescriptor, ToString_InterfaceScope) {
    EXPECT_STREQ(toString(InterfaceScope::SpecificMarker), "SpecificMarker");
    EXPECT_STREQ(toString(InterfaceScope::AllRegisteredInterfaces), "AllRegisteredInterfaces");
}

// ============================================================================
// OperatorTraitFlags bitmask
// ============================================================================

TEST(ContributionDescriptor, TraitFlags_Bitmask) {
    auto flags = OperatorTraitFlags::SymmetricLike | OperatorTraitFlags::HasSecondOrder;
    EXPECT_TRUE(hasFlag(flags, OperatorTraitFlags::SymmetricLike));
    EXPECT_TRUE(hasFlag(flags, OperatorTraitFlags::HasSecondOrder));
    EXPECT_FALSE(hasFlag(flags, OperatorTraitFlags::SkewLike));
    EXPECT_FALSE(hasFlag(flags, OperatorTraitFlags::HasMass));

    auto combined = flags | OperatorTraitFlags::PositiveSemiDefiniteLike;
    EXPECT_TRUE(hasFlag(combined, OperatorTraitFlags::PositiveSemiDefiniteLike));

    EXPECT_FALSE(hasFlag(OperatorTraitFlags::None, OperatorTraitFlags::SymmetricLike));
}

// ============================================================================
// Builder helpers
// ============================================================================

TEST(ContributionDescriptor, DiagonalSymmetric) {
    auto d = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "equations", "FormsInstaller");

    EXPECT_EQ(d.operator_tag, "equations");
    EXPECT_EQ(d.origin, "FormsInstaller");
    EXPECT_EQ(d.role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::SymmetricLike));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::PositiveSemiDefiniteLike));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasSecondOrder));
    ASSERT_EQ(d.test_variables.size(), 1u);
    ASSERT_EQ(d.trial_variables.size(), 1u);
    EXPECT_EQ(d.test_variables[0], VariableKey::field(0));
    EXPECT_EQ(d.trial_variables[0], VariableKey::field(0));
}

TEST(ContributionDescriptor, ConstraintBlock) {
    auto d = ContributionDescriptor::constraintBlock(
        VariableKey::field(1), VariableKey::field(0), "equations", "FormsInstaller");

    EXPECT_EQ(d.role, ContributionRole::ConstraintBlock);
    EXPECT_EQ(d.traits, OperatorTraitFlags::None);
    EXPECT_EQ(d.test_variables[0], VariableKey::field(1));
    EXPECT_EQ(d.trial_variables[0], VariableKey::field(0));
}

TEST(ContributionDescriptor, Stabilization) {
    auto d = ContributionDescriptor::stabilization(
        VariableKey::field(0), "equations", "FormsInstaller");

    EXPECT_EQ(d.role, ContributionRole::StabilizationBlock);
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasSecondOrder));
}

TEST(ContributionDescriptor, GlobalCoupling) {
    auto d = ContributionDescriptor::globalCoupling(
        {VariableKey::field(0)},
        {VariableKey::named(VariableKind::BoundaryFunctional, "Q")},
        "coupled_bc", "CoupledBoundaryManager");

    EXPECT_EQ(d.role, ContributionRole::GlobalCoupling);
    EXPECT_EQ(d.domain, DomainKind::Global);
    ASSERT_EQ(d.test_variables.size(), 1u);
    ASSERT_EQ(d.trial_variables.size(), 1u);
    EXPECT_EQ(d.trial_variables[0].kind, VariableKind::BoundaryFunctional);
}

// ============================================================================
// Stokes example: build all blocks as descriptors
// ============================================================================

TEST(ContributionDescriptor, Stokes_AllBlocks) {
    // VV block: diagonal symmetric (Laplacian on velocity)
    auto vv = ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(1), "equations", "FormsInstaller");

    // VP block: off-diagonal (grad(p) in momentum)
    ContributionDescriptor vp;
    vp.operator_tag = "equations";
    vp.origin = "FormsInstaller";
    vp.role = ContributionRole::OffDiagonalBlock;
    vp.test_variables = {VariableKey::field(1)};
    vp.trial_variables = {VariableKey::field(0)};

    // PV block: constraint (div(u) in continuity)
    auto pv = ContributionDescriptor::constraintBlock(
        VariableKey::field(0), VariableKey::field(1), "equations", "FormsInstaller");

    // PP block: empty (no diagonal for pressure in unstabilized Stokes)
    // → constraint field, no contribution

    EXPECT_EQ(vv.role, ContributionRole::DiagonalBlock);
    EXPECT_EQ(vp.role, ContributionRole::OffDiagonalBlock);
    EXPECT_EQ(pv.role, ContributionRole::ConstraintBlock);
}

// ============================================================================
// ProblemAnalysisContext integration
// ============================================================================

TEST(ContributionDescriptor, StoredInContext) {
    ProblemAnalysisContext ctx;
    EXPECT_TRUE(ctx.contributions().empty());

    ctx.addContribution(ContributionDescriptor::diagonalSymmetric(
        VariableKey::field(0), "eq", "test"));

    EXPECT_FALSE(ctx.contributions().empty());
    ASSERT_EQ(ctx.contributions().size(), 1u);
    EXPECT_EQ(ctx.contributions()[0].role, ContributionRole::DiagonalBlock);
    EXPECT_FALSE(ctx.empty());
    EXPECT_EQ(ctx.inputsVersion(), 1u);
}

TEST(ContributionDescriptor, RobinBC_AsDescriptor) {
    // Robin BC lowers to a BoundaryConstraint with HasMass trait
    ContributionDescriptor d;
    d.operator_tag = "robin_bc";
    d.origin = "BoundaryConditionManager";
    d.domain = DomainKind::Boundary;
    d.boundary_marker = 3;
    d.role = ContributionRole::BoundaryConstraint;
    d.traits = OperatorTraitFlags::HasMass | OperatorTraitFlags::NullspaceLifting;
    d.test_variables = {VariableKey::field(0)};
    d.trial_variables = {VariableKey::field(0)};

    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasMass));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::NullspaceLifting));
    EXPECT_EQ(d.boundary_marker, 3);
}

// ============================================================================
// Phase 22 — Extended metadata
// ============================================================================

TEST(ContributionDescriptor, Phase22_ToString_NewEnums) {
    EXPECT_STREQ(toString(NullspaceEffect::Preserves), "Preserves");
    EXPECT_STREQ(toString(NullspaceEffect::WeaklyLifts), "WeaklyLifts");
    EXPECT_STREQ(toString(NullspaceEffect::ExactlyRemoves), "ExactlyRemoves");

    EXPECT_STREQ(toString(ConsistencyKind::ExactContinuum), "ExactContinuum");
    EXPECT_STREQ(toString(ConsistencyKind::ConsistentPerturbation), "ConsistentPerturbation");

    EXPECT_STREQ(toString(AdjointConsistencyKind::Yes), "Yes");
    EXPECT_STREQ(toString(AdjointConsistencyKind::No), "No");

    EXPECT_STREQ(toString(TemporalContributionKind::MassLike), "MassLike");
    EXPECT_STREQ(toString(TemporalContributionKind::TimeIndependentResidual), "TimeIndependentResidual");
    EXPECT_STREQ(toString(TemporalContributionKind::PureAlgebraicConstraint), "PureAlgebraicConstraint");
    EXPECT_STREQ(toString(TemporalContributionKind::PureConstraint), "PureConstraint");

    EXPECT_STREQ(toString(BalanceRole::Accumulation), "Accumulation");
    EXPECT_STREQ(toString(BalanceRole::FluxLike), "FluxLike");
    EXPECT_STREQ(toString(BalanceRole::ExchangeLike), "ExchangeLike");

    EXPECT_STREQ(toString(PairingKind::FormalAdjointPair), "FormalAdjointPair");
    EXPECT_STREQ(toString(PairingKind::ConstraintPair), "ConstraintPair");

    EXPECT_STREQ(toString(TransportCharacter::DirectionalFirstOrder), "DirectionalFirstOrder");
    EXPECT_STREQ(toString(TransportCharacter::DiffusionLike), "DiffusionLike");
}

TEST(ContributionDescriptor, Phase22_MassLikeBuilder) {
    auto d = ContributionDescriptor::massLike(
        VariableKey::field(0), "equations", "test");

    EXPECT_EQ(d.role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasMass));
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::SymmetricLike));
    EXPECT_TRUE(d.temporal.has_value());
    EXPECT_EQ(d.temporal->derivative_order, 1);
    EXPECT_EQ(d.temporal->kind, TemporalContributionKind::MassLike);
    EXPECT_TRUE(d.balance.has_value());
    EXPECT_EQ(d.balance->role, BalanceRole::Accumulation);
}

TEST(ContributionDescriptor, Phase22_ExchangeCouplingBuilder) {
    auto d = ContributionDescriptor::exchangeCoupling(
        VariableKey::field(0), VariableKey::field(1),
        "energy_balance", "equations", "test");

    EXPECT_EQ(d.role, ContributionRole::GlobalCoupling);
    EXPECT_TRUE(d.balance.has_value());
    EXPECT_EQ(d.balance->balance_group, "energy_balance");
    EXPECT_EQ(d.balance->role, BalanceRole::ExchangeLike);
}

TEST(ContributionDescriptor, Phase22_ConstraintPairBuilder) {
    auto d = ContributionDescriptor::constraintPairDesc(
        VariableKey::field(1), VariableKey::field(0),
        "incompressibility", "equations", "test");

    EXPECT_EQ(d.role, ContributionRole::ConstraintBlock);
    ASSERT_EQ(d.pairings.size(), 1u);
    EXPECT_EQ(d.pairings[0].kind, PairingKind::ConstraintPair);
    EXPECT_EQ(d.pairings[0].pairing_group, "incompressibility");
    EXPECT_EQ(d.pairings[0].row_var, VariableKey::field(1));
    EXPECT_EQ(d.pairings[0].col_var, VariableKey::field(0));
}

TEST(ContributionDescriptor, Phase22_TransportLikeBuilder) {
    auto d = ContributionDescriptor::transportLike(
        VariableKey::field(0), "equations", "test");

    EXPECT_EQ(d.role, ContributionRole::DiagonalBlock);
    EXPECT_TRUE(hasFlag(d.traits, OperatorTraitFlags::HasFirstOrder));
    EXPECT_TRUE(d.transport_character.has_value());
    EXPECT_EQ(*d.transport_character, TransportCharacter::DirectionalFirstOrder);
}

TEST(ContributionDescriptor, Phase22_ExtendedFields) {
    ContributionDescriptor d;
    d.nullspace_effect = NullspaceEffect::WeaklyLifts;
    d.consistency_kind = ConsistencyKind::ConsistentPerturbation;
    d.adjoint_consistency = AdjointConsistencyKind::Yes;
    d.scaling = ScalingDescriptor{-1, 0, false, true};
    d.temporal = TemporalDescriptor{1, TemporalContributionKind::MassLike};
    d.balance = BalanceDescriptor{"group1", BalanceRole::FluxLike, -1, true};
    d.transport_character = TransportCharacter::DiffusionLike;

    PairingDescriptor pd;
    pd.row_var = VariableKey::field(0);
    pd.col_var = VariableKey::field(1);
    pd.kind = PairingKind::FormalAdjointPair;
    pd.pairing_group = "velocity_pressure";
    d.pairings.push_back(pd);

    EXPECT_EQ(*d.nullspace_effect, NullspaceEffect::WeaklyLifts);
    EXPECT_EQ(*d.consistency_kind, ConsistencyKind::ConsistentPerturbation);
    EXPECT_EQ(*d.adjoint_consistency, AdjointConsistencyKind::Yes);
    EXPECT_EQ(d.scaling->h_power, -1);
    EXPECT_TRUE(d.scaling->coefficient_scaled);
    EXPECT_EQ(d.temporal->derivative_order, 1);
    EXPECT_EQ(d.balance->balance_group, "group1");
    EXPECT_EQ(d.balance->sign, -1);
    EXPECT_TRUE(d.balance->local_closure_expected);
    EXPECT_EQ(*d.transport_character, TransportCharacter::DiffusionLike);
    ASSERT_EQ(d.pairings.size(), 1u);
    EXPECT_EQ(d.pairings[0].kind, PairingKind::FormalAdjointPair);
}
