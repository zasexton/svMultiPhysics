/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ContributionDescriptor.h"

#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

std::string makeContributionId(const std::string& op_tag,
                               const std::string& origin)
{
    if (origin.empty()) {
        return op_tag;
    }
    if (op_tag.empty()) {
        return origin;
    }
    return origin + ":" + op_tag;
}

void appendVariableId(std::ostringstream& os, const VariableKey& variable)
{
    os << static_cast<int>(variable.kind) << ':'
       << variable.field_id << ':'
       << variable.component << ':'
       << variable.name;
}

void appendVariables(std::ostringstream& os,
                     const std::vector<VariableKey>& variables)
{
    for (const auto& variable : variables) {
        os << '[';
        appendVariableId(os, variable);
        os << ']';
    }
}

} // namespace

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
        case ContributionRole::InterfaceCoupling:  return "InterfaceCoupling";
        case ContributionRole::SourceVector:       return "SourceVector";
        case ContributionRole::ExternalForcing:    return "ExternalForcing";
        case ContributionRole::InitialCondition:   return "InitialCondition";
        case ContributionRole::DiagnosticOnly:     return "DiagnosticOnly";
        case ContributionRole::GlobalCoupling:     return "GlobalCoupling";
        case ContributionRole::FieldToAuxiliary:   return "FieldToAuxiliary";
        case ContributionRole::AuxiliaryToField:   return "AuxiliaryToField";
        case ContributionRole::AuxiliaryToAuxiliary:
            return "AuxiliaryToAuxiliary";
        case ContributionRole::AuxiliarySelf:      return "AuxiliarySelf";
    }
    return "Unknown";
}

const char* toString(NullspaceFamily f) noexcept {
    switch (f) {
        case NullspaceFamily::ScalarConstant:        return "ScalarConstant";
        case NullspaceFamily::ComponentwiseConstant:  return "ComponentwiseConstant";
        case NullspaceFamily::VectorConstant:         return "VectorConstant";
        case NullspaceFamily::RigidTranslation:       return "RigidTranslation";
        case NullspaceFamily::RigidRotation:          return "RigidRotation";
        case NullspaceFamily::RigidBody:              return "RigidBody";
        case NullspaceFamily::GaugeConstant:          return "GaugeConstant";
        case NullspaceFamily::HarmonicField:          return "HarmonicField";
        case NullspaceFamily::GradientKernel:         return "GradientKernel";
        case NullspaceFamily::CurlKernel:             return "CurlKernel";
        case NullspaceFamily::DivergenceFreeKernel:   return "DivergenceFreeKernel";
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
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
        case TemporalContributionKind::Unknown:
            return "Unknown";
        case TemporalContributionKind::TimeIndependentResidual:
            return "TimeIndependentResidual";
        case TemporalContributionKind::MassLike:
            return "MassLike";
        case TemporalContributionKind::DampedMassLike:
            return "DampedMassLike";
        case TemporalContributionKind::PureAlgebraicConstraint:
            return "PureAlgebraicConstraint";
        case TemporalContributionKind::LagrangeMultiplierConstraint:
            return "LagrangeMultiplierConstraint";
        case TemporalContributionKind::PreviousTimeState:
            return "PreviousTimeState";
        case TemporalContributionKind::TimeIntegratorResidual:
            return "TimeIntegratorResidual";
        case TemporalContributionKind::None:
            return "None";
        case TemporalContributionKind::PureConstraint:
            return "PureConstraint";
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
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
    d.ensureStableContributionId();
    return d;
}

std::string ContributionDescriptor::stableContributionId(
    const ContributionDescriptor& desc)
{
    const auto base = makeContributionId(desc.operator_tag, desc.origin);
    std::ostringstream os;
    os << (base.empty() ? std::string("contribution") : base)
       << "|domain=" << static_cast<int>(desc.domain)
       << "|role=" << static_cast<int>(desc.role)
       << "|boundary=" << desc.boundary_marker
       << "|interface=" << desc.interface_marker
       << "|scope=" << static_cast<int>(desc.interface_scope)
       << "|test=";
    appendVariables(os, desc.test_variables);
    os << "|trial=";
    appendVariables(os, desc.trial_variables);
    os << "|related=";
    appendVariables(os, desc.related_variables);
    if (desc.source_block_key) {
        os << "|block=" << desc.source_block_key->first
           << ',' << desc.source_block_key->second;
    }
    return os.str();
}

void ContributionDescriptor::ensureStableContributionId()
{
    if (contribution_id.empty()) {
        contribution_id = stableContributionId(*this);
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
