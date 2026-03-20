/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ContributionDescriptor.h"

#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

// ============================================================================
// String conversion
// ============================================================================

const char* toString(ContributionRole r) noexcept {
    switch (r) {
        case ContributionRole::DiagonalBlock:      return "DiagonalBlock";
        case ContributionRole::OffDiagonalBlock:   return "OffDiagonalBlock";
        case ContributionRole::ConstraintBlock:    return "ConstraintBlock";
        case ContributionRole::StabilizationBlock: return "StabilizationBlock";
        case ContributionRole::BoundaryConstraint: return "BoundaryConstraint";
        case ContributionRole::GlobalCoupling:     return "GlobalCoupling";
    }
    return "Unknown";
}

const char* toString(NullspaceFamily f) noexcept {
    switch (f) {
        case NullspaceFamily::ScalarConstant:        return "ScalarConstant";
        case NullspaceFamily::ComponentwiseConstant:  return "ComponentwiseConstant";
        case NullspaceFamily::KernelOfSymGrad:        return "KernelOfSymGrad";
        case NullspaceFamily::UserDefined:            return "UserDefined";
    }
    return "Unknown";
}

const char* toString(InterfaceScope s) noexcept {
    switch (s) {
        case InterfaceScope::SpecificMarker:          return "SpecificMarker";
        case InterfaceScope::AllRegisteredInterfaces:  return "AllRegisteredInterfaces";
    }
    return "Unknown";
}

// ============================================================================
// Builder helpers
// ============================================================================

ContributionDescriptor ContributionDescriptor::diagonalSymmetric(
    VariableKey field, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::DiagonalBlock;
    d.traits = OperatorTraitFlags::SymmetricLike
             | OperatorTraitFlags::PositiveSemiDefiniteLike
             | OperatorTraitFlags::HasSecondOrder;
    d.test_variables = {field};
    d.trial_variables = {field};
    return d;
}

ContributionDescriptor ContributionDescriptor::constraintBlock(
    VariableKey test, VariableKey trial, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::ConstraintBlock;
    d.traits = OperatorTraitFlags::None;
    d.test_variables = {test};
    d.trial_variables = {trial};
    return d;
}

ContributionDescriptor ContributionDescriptor::stabilization(
    VariableKey field, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::StabilizationBlock;
    d.traits = OperatorTraitFlags::HasSecondOrder;
    d.test_variables = {field};
    d.trial_variables = {field};
    return d;
}

ContributionDescriptor ContributionDescriptor::globalCoupling(
    std::vector<VariableKey> test, std::vector<VariableKey> trial,
    std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.domain = DomainKind::Global;
    d.role = ContributionRole::GlobalCoupling;
    d.traits = OperatorTraitFlags::None;
    d.test_variables = std::move(test);
    d.trial_variables = std::move(trial);
    return d;
}

// ============================================================================
// Phase 22 toString helpers
// ============================================================================

const char* toString(NullspaceEffect e) noexcept {
    switch (e) {
        case NullspaceEffect::Preserves:       return "Preserves";
        case NullspaceEffect::WeaklyLifts:     return "WeaklyLifts";
        case NullspaceEffect::ExactlyRemoves:  return "ExactlyRemoves";
        case NullspaceEffect::Unknown:         return "Unknown";
    }
    return "Unknown";
}

const char* toString(ConsistencyKind k) noexcept {
    switch (k) {
        case ConsistencyKind::ExactContinuum:           return "ExactContinuum";
        case ConsistencyKind::ConsistentPerturbation:   return "ConsistentPerturbation";
        case ConsistencyKind::InconsistentPerturbation: return "InconsistentPerturbation";
        case ConsistencyKind::Unknown:                  return "Unknown";
    }
    return "Unknown";
}

const char* toString(AdjointConsistencyKind k) noexcept {
    switch (k) {
        case AdjointConsistencyKind::Yes:     return "Yes";
        case AdjointConsistencyKind::No:      return "No";
        case AdjointConsistencyKind::Unknown: return "Unknown";
    }
    return "Unknown";
}

const char* toString(TemporalContributionKind k) noexcept {
    switch (k) {
        case TemporalContributionKind::None:           return "None";
        case TemporalContributionKind::MassLike:       return "MassLike";
        case TemporalContributionKind::DampedMassLike: return "DampedMassLike";
        case TemporalContributionKind::PureConstraint: return "PureConstraint";
        case TemporalContributionKind::Unknown:        return "Unknown";
    }
    return "Unknown";
}

const char* toString(BalanceRole r) noexcept {
    switch (r) {
        case BalanceRole::Accumulation: return "Accumulation";
        case BalanceRole::FluxLike:     return "FluxLike";
        case BalanceRole::SourceLike:   return "SourceLike";
        case BalanceRole::SinkLike:     return "SinkLike";
        case BalanceRole::ExchangeLike: return "ExchangeLike";
        case BalanceRole::Unknown:      return "Unknown";
    }
    return "Unknown";
}

const char* toString(PairingKind k) noexcept {
    switch (k) {
        case PairingKind::FormalAdjointPair:       return "FormalAdjointPair";
        case PairingKind::ConstraintPair:          return "ConstraintPair";
        case PairingKind::StabilizedConstraintPair: return "StabilizedConstraintPair";
        case PairingKind::Unknown:                 return "Unknown";
    }
    return "Unknown";
}

const char* toString(TransportCharacter c) noexcept {
    switch (c) {
        case TransportCharacter::None:                  return "None";
        case TransportCharacter::DirectionalFirstOrder: return "DirectionalFirstOrder";
        case TransportCharacter::DiffusionLike:         return "DiffusionLike";
        case TransportCharacter::NonNormalLike:          return "NonNormalLike";
        case TransportCharacter::TransportDominatedRisk: return "TransportDominatedRisk";
    }
    return "Unknown";
}

// ============================================================================
// Phase 22 builder helpers
// ============================================================================

ContributionDescriptor ContributionDescriptor::massLike(
    VariableKey field, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::DiagonalBlock;
    d.traits = OperatorTraitFlags::HasMass | OperatorTraitFlags::SymmetricLike
             | OperatorTraitFlags::PositiveSemiDefiniteLike;
    d.test_variables = {field};
    d.trial_variables = {field};
    d.temporal = TemporalDescriptor{1, TemporalContributionKind::MassLike};
    d.balance = BalanceDescriptor{"", BalanceRole::Accumulation, 1, false};
    return d;
}

ContributionDescriptor ContributionDescriptor::exchangeCoupling(
    VariableKey test, VariableKey trial,
    std::string balance_group, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::GlobalCoupling;
    d.traits = OperatorTraitFlags::None;
    d.test_variables = {test};
    d.trial_variables = {trial};
    d.balance = BalanceDescriptor{std::move(balance_group), BalanceRole::ExchangeLike, 1, false};
    return d;
}

ContributionDescriptor ContributionDescriptor::constraintPairDesc(
    VariableKey primal, VariableKey dual,
    std::string pairing_group, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::ConstraintBlock;
    d.traits = OperatorTraitFlags::None;
    d.test_variables = {primal};
    d.trial_variables = {dual};
    PairingDescriptor pd;
    pd.row_var = primal;
    pd.col_var = dual;
    pd.kind = PairingKind::ConstraintPair;
    pd.pairing_group = std::move(pairing_group);
    d.pairings.push_back(std::move(pd));
    return d;
}

ContributionDescriptor ContributionDescriptor::transportLike(
    VariableKey field, std::string op_tag, std::string orig) {
    ContributionDescriptor d;
    d.operator_tag = std::move(op_tag);
    d.origin = std::move(orig);
    d.role = ContributionRole::DiagonalBlock;
    d.traits = OperatorTraitFlags::HasFirstOrder;
    d.test_variables = {field};
    d.trial_variables = {field};
    d.transport_character = TransportCharacter::DirectionalFirstOrder;
    return d;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
