/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_KERNEL_CONTRIBUTION_RECORD_H
#define SVMP_FE_ANALYSIS_KERNEL_CONTRIBUTION_RECORD_H

/**
 * @file KernelContributionRecord.h
 * @brief [DEPRECATED] Structured metadata for non-Forms operators
 *
 * This type is superseded by ContributionDescriptor (Phase 10). It is retained
 * as a temporary producer-side shim for kernels that still use analysisMetadata()
 * instead of analysisContributions(). Use toContributionDescriptor() to convert.
 *
 * New kernel implementations should use analysisContributions() directly and
 * produce ContributionDescriptor objects.
 *
 * @see ContributionDescriptor for the replacement type
 * @see AssemblyKernel::analysisContributions() for the preferred producer interface
 * @see GlobalKernel::analysisMetadata() for the producer interface (Phase 8)
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Metadata record for a non-Forms operator contribution
 *
 * Describes the coupling structure of an operator that cannot be inferred
 * from FormExpr analysis alone: hand-written kernels, global operators,
 * interface couplings, and coupled-boundary PDE-ODE models.
 */
struct KernelContributionRecord {
    std::string operator_tag;                          ///< e.g. "ContactPenalty", "RCRBoundary"
    DomainKind domain{DomainKind::Cell};               ///< Where this operator acts
    std::string source_name;                           ///< Human-readable origin

    std::vector<VariableKey> test_variables;            ///< Test-side variables
    std::vector<VariableKey> trial_variables;           ///< Trial-side variables
    std::vector<VariableKey> related_variables;         ///< Other variables involved (data dependencies)

    int boundary_marker{-1};                           ///< -1 = not boundary-specific
    int interface_marker{-1};                          ///< -1 = not interface-specific

    bool is_linear{false};                             ///< Operator is linear in trial variables
    bool is_symmetric_like{false};                     ///< Bilinear form is approximately symmetric
    bool is_constraint_like{false};                    ///< Acts as a constraint / Lagrange multiplier
    bool has_global_support{false};                    ///< Non-local (couples all DOFs)
    bool has_stabilization{false};                     ///< Contains stabilization terms

    /// Optional nullspace hints for hand-written kernels that know their own structure.
    /// Empty = no hints (analyzers must infer). Populated by kernels that override
    /// analysisMetadata() with domain-specific knowledge.
    std::vector<PropertyClaim> nullspace_hints;

    /**
     * @brief Convert to normalized ContributionDescriptor
     *
     * This is a compatibility shim for the transition from KernelContributionRecord
     * to ContributionDescriptor.  The conversion maps boolean flags to structured
     * enums and trait flags.
     */
    [[nodiscard]] ContributionDescriptor toContributionDescriptor() const {
        ContributionDescriptor d;
        d.operator_tag = operator_tag;
        d.origin = source_name;
        d.domain = domain;
        d.boundary_marker = boundary_marker;
        if (interface_marker < 0 && domain == DomainKind::InterfaceFace) {
            d.interface_scope = InterfaceScope::AllRegisteredInterfaces;
        } else {
            d.interface_marker = interface_marker;
        }
        d.test_variables = test_variables;
        d.trial_variables = trial_variables;
        d.related_variables = related_variables;

        // Map booleans to role
        if (is_constraint_like) {
            d.role = ContributionRole::ConstraintBlock;
        } else if (has_stabilization) {
            d.role = ContributionRole::StabilizationBlock;
        } else if (has_global_support) {
            d.role = ContributionRole::GlobalCoupling;
        } else if (!test_variables.empty() && !trial_variables.empty() &&
                   test_variables[0] == trial_variables[0]) {
            d.role = ContributionRole::DiagonalBlock;
        } else {
            d.role = ContributionRole::OffDiagonalBlock;
        }

        // Map booleans to traits
        auto flags = OperatorTraitFlags::None;
        if (is_symmetric_like) flags = flags | OperatorTraitFlags::SymmetricLike;
        if (has_stabilization) flags = flags | OperatorTraitFlags::HasSecondOrder;
        d.traits = flags;

        d.confidence = AnalysisConfidence::Medium;  // kernel metadata is heuristic

        // Convert nullspace hints
        for (const auto& hint : nullspace_hints) {
            if (hint.kind == PropertyKind::Nullspace) {
                NullspaceHint nh;
                nh.field = hint.field;
                nh.component = hint.component;
                nh.confidence = hint.confidence;
                nh.reason = hint.description;
                // Infer family from description (same heuristic as GaugeAdapter)
                if (hint.description.find("rigid") != std::string::npos) {
                    nh.family = NullspaceFamily::KernelOfSymGrad;
                } else if (hint.description.find("component") != std::string::npos) {
                    nh.family = NullspaceFamily::ComponentwiseConstant;
                } else {
                    nh.family = NullspaceFamily::ScalarConstant;
                }
                d.nullspace_hints.push_back(std::move(nh));
            }
        }

        return d;
    }
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_KERNEL_CONTRIBUTION_RECORD_H
