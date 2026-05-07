/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/BoundaryConditionDescriptor.h"
#include "Constraints/GaugeRegistry.h"

#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

const char* toString(TraceKind k) noexcept {
    switch (k) {
        case TraceKind::Value:               return "Value";
        case TraceKind::NormalComponent:      return "NormalComponent";
        case TraceKind::TangentialComponent:  return "TangentialComponent";
        case TraceKind::Flux:                return "Flux";
        case TraceKind::NormalFlux:           return "NormalFlux";
        case TraceKind::Mixed:               return "Mixed";
        case TraceKind::AlgebraicRelation:   return "AlgebraicRelation";
    }
    return "Unknown";
}

const char* toString(EnforcementKind k) noexcept {
    switch (k) {
        case EnforcementKind::Strong:          return "Strong";
        case EnforcementKind::WeakConsistent:  return "WeakConsistent";
        case EnforcementKind::WeakPenalty:     return "WeakPenalty";
        case EnforcementKind::WeakNitsche:     return "WeakNitsche";
        case EnforcementKind::WeakInequality:  return "WeakInequality";
        case EnforcementKind::AffineRelation:  return "AffineRelation";
    }
    return "Unknown";
}

const char* toString(InequalitySense k) noexcept {
    switch (k) {
        case InequalitySense::None:            return "None";
        case InequalitySense::LessEqual:       return "LessEqual";
        case InequalitySense::GreaterEqual:    return "GreaterEqual";
        case InequalitySense::Complementarity: return "Complementarity";
    }
    return "Unknown";
}

const char* toString(NitscheVariant k) noexcept {
    switch (k) {
        case NitscheVariant::Unknown:      return "Unknown";
        case NitscheVariant::Symmetric:    return "Symmetric";
        case NitscheVariant::Nonsymmetric: return "Nonsymmetric";
        case NitscheVariant::Skew:         return "Skew";
        case NitscheVariant::Incomplete:   return "Incomplete";
    }
    return "Unknown";
}

gauge::AnchoringVerdict
descriptorToVerdict(const BoundaryConditionDescriptor& desc,
                    gauge::NullspaceModeFamily family)
{
    using Family = gauge::NullspaceModeFamily;
    using Verdict = gauge::AnchoringVerdict;

    switch (family) {
        case Family::ScalarConstant:
        case Family::ComponentwiseConstant:
            if (desc.enforcement_kind == EnforcementKind::WeakInequality &&
                desc.anchors_constant_mode) {
                return Verdict::PartiallyAnchored;
            }
            if (desc.anchors_constant_mode) return Verdict::Anchored;
            break;

        case Family::KernelOfSymGrad:
            if (desc.anchors_rigid_body_translation && desc.anchors_rigid_body_rotation) {
                return Verdict::Anchored;
            }
            if (desc.enforcement_kind == EnforcementKind::WeakInequality &&
                (desc.anchors_rigid_body_translation || desc.anchors_rigid_body_rotation)) {
                return Verdict::PartiallyAnchored;
            }
            if (desc.anchors_rigid_body_translation) {
                return Verdict::PartiallyAnchored;
            }
            break;
    }

    // Pure weak-consistent boundary loads preserve the mode unless the
    // producer explicitly claimed anchoring through the flags above.
    if (desc.enforcement_kind == EnforcementKind::WeakConsistent) {
        return Verdict::Preserved;
    }

    // Algebraic relations (periodic, MPC) preserve modes
    if (desc.enforcement_kind == EnforcementKind::AffineRelation) {
        return Verdict::Preserved;
    }

    return Verdict::Unknown;
}

// ============================================================================
// lowerBCDescriptor
// ============================================================================

std::vector<ContributionDescriptor>
lowerBCDescriptor(const BoundaryConditionDescriptor& desc) {
    std::vector<ContributionDescriptor> result;
    auto pushContribution = [&](ContributionDescriptor contrib) {
        contrib.ensureStableContributionId();
        result.push_back(std::move(contrib));
    };

    ContributionDescriptor d;
    d.operator_tag = "bc";
    d.origin = desc.source;
    d.domain = desc.domain;
    d.boundary_marker = desc.boundary_marker;
    d.interface_marker = desc.interface_marker;
    d.confidence = AnalysisConfidence::High;

    // Primary variable
    if (desc.primary_variable.kind == VariableKind::FieldComponent) {
        d.test_variables = {desc.primary_variable};
        d.trial_variables = {desc.primary_variable};
    }

    // Related variables
    d.related_variables = desc.related_variables;

    switch (desc.enforcement_kind) {
        case EnforcementKind::Strong:
            // Strong Dirichlet → BoundaryConstraint + NullspaceLifting
            d.role = ContributionRole::BoundaryConstraint;
            d.traits = OperatorTraitFlags::NullspaceLifting;
            d.nullspace_effect = NullspaceEffect::ExactlyRemoves;
            d.consistency_kind = ConsistencyKind::ExactContinuum;
            pushContribution(d);
            break;

        case EnforcementKind::WeakPenalty:
            // Robin → BoundaryConstraint + HasMass + NullspaceLifting
            d.role = ContributionRole::BoundaryConstraint;
            d.traits = OperatorTraitFlags::HasMass;
            if (desc.anchors_constant_mode || desc.anchors_rigid_body_translation) {
                d.traits = d.traits | OperatorTraitFlags::NullspaceLifting;
                d.nullspace_effect = NullspaceEffect::WeaklyLifts;
            } else {
                d.nullspace_effect = NullspaceEffect::Preserves;
            }
            d.consistency_kind = ConsistencyKind::ExactContinuum;
            pushContribution(d);
            break;

        case EnforcementKind::WeakNitsche:
            // Nitsche → BoundaryConstraint + HasSecondOrder + NullspaceLifting
            d.operator_tag = "bc:nitsche:consistency";
            d.role = ContributionRole::BoundaryConstraint;
            d.traits = OperatorTraitFlags::HasSecondOrder | OperatorTraitFlags::NullspaceLifting;
            d.nullspace_effect = NullspaceEffect::WeaklyLifts;
            if (desc.nitsche) {
                const bool rhs_consistency_ok =
                    desc.is_homogeneous ||
                    desc.nitsche->rhs_consistency_terms_present;
                d.consistency_kind =
                    (desc.nitsche->primal_consistency_terms_present &&
                     rhs_consistency_ok)
                    ? ConsistencyKind::ConsistentPerturbation
                    : ConsistencyKind::Unknown;
                d.adjoint_consistency =
                    desc.nitsche->adjoint_consistency_terms_present
                        ? AdjointConsistencyKind::Yes
                        : AdjointConsistencyKind::No;
            } else {
                d.consistency_kind = ConsistencyKind::Unknown;
                d.adjoint_consistency = AdjointConsistencyKind::Unknown;
            }
            pushContribution(d);
            // Also a stabilization penalty component
            {
                ContributionDescriptor stab = d;
                stab.contribution_id.clear();
                stab.operator_tag = "bc:nitsche:penalty";
                stab.role = ContributionRole::StabilizationBlock;
                stab.traits = OperatorTraitFlags::HasMass;
                stab.consistency_kind = desc.nitsche &&
                    desc.nitsche->penalty_positive &&
                    desc.nitsche->penalty_scaling_verified &&
                    desc.nitsche->penalty_trace_bound_verified
                        ? ConsistencyKind::ConsistentPerturbation
                        : ConsistencyKind::Unknown;
                pushContribution(std::move(stab));
            }
            break;

        case EnforcementKind::WeakInequality:
            // One-sided trace law → BoundaryConstraint + HasMass + weak lifting
            d.role = ContributionRole::BoundaryConstraint;
            d.traits = OperatorTraitFlags::HasMass | OperatorTraitFlags::NullspaceLifting;
            d.nullspace_effect = NullspaceEffect::WeaklyLifts;
            d.consistency_kind = ConsistencyKind::ConsistentPerturbation;
            pushContribution(d);
            break;

        case EnforcementKind::WeakConsistent:
            // Natural (Neumann) → BoundaryConstraint, no nullspace effect
            d.role = ContributionRole::BoundaryConstraint;
            d.traits = OperatorTraitFlags::None;
            d.nullspace_effect = NullspaceEffect::Preserves;
            d.consistency_kind = ConsistencyKind::ExactContinuum;
            // Only emit a BalanceDescriptor when the producer has set an explicit
            // pairing_group, indicating it models a real conservation/exchange
            // structure. Without producer-side orientation and exchange metadata,
            // fabricating balance groups from field IDs or source strings just
            // creates false ClosureBroken claims in ConservationAnalyzer.
            if (!desc.pairing_group.empty()) {
                d.balance = BalanceDescriptor{
                    desc.pairing_group, BalanceRole::FluxLike, 1, false};
            }
            pushContribution(d);
            break;

        case EnforcementKind::AffineRelation:
            // Periodic/MPC → ConstraintBlock + NullspacePreserving
            d.role = ContributionRole::ConstraintBlock;
            d.traits = OperatorTraitFlags::NullspacePreserving;
            d.nullspace_effect = NullspaceEffect::Preserves;
            pushContribution(d);
            break;
    }

    // For coupled boundaries, also emit a GlobalCoupling if there are related variables
    if (desc.introduces_global_coupling && !desc.related_variables.empty()) {
        ContributionDescriptor gc;
        gc.operator_tag = "coupled_bc";
        gc.origin = desc.source;
        gc.domain = desc.domain;
        gc.boundary_marker = desc.boundary_marker;
        gc.interface_marker = desc.interface_marker;
        gc.role = ContributionRole::GlobalCoupling;
        gc.traits = OperatorTraitFlags::None;
        gc.confidence = AnalysisConfidence::High;
        if (desc.primary_variable.kind == VariableKind::FieldComponent) {
            gc.test_variables = {desc.primary_variable};
        }
        gc.trial_variables = desc.related_variables;
        gc.related_variables = desc.related_variables;
        pushContribution(std::move(gc));
    }

    return result;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
