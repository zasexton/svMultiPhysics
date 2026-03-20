/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/GaugeAdapter.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

// ============================================================================
// claimsToCandidates
// ============================================================================

std::vector<gauge::GaugeCandidate>
claimsToCandidates(const ProblemAnalysisReport& report) {
    std::vector<gauge::GaugeCandidate> candidates;

    for (const auto& claim : report.claims) {
        if (claim.kind != PropertyKind::Nullspace) continue;
        if (claim.field == INVALID_FIELD_ID) continue;

        gauge::GaugeCandidate c;
        c.field = claim.field;
        c.component = claim.component;
        c.region = claim.region;
        c.source = gauge::CandidateSource::FormsInference;

        // Map confidence
        switch (claim.confidence) {
            case AnalysisConfidence::High:   c.confidence = gauge::Confidence::High; break;
            case AnalysisConfidence::Medium: c.confidence = gauge::Confidence::Medium; break;
            case AnalysisConfidence::Low:    c.confidence = gauge::Confidence::Low; break;
        }

        // Determine NullspaceModeFamily: prefer structured field, fall back to text.
        if (claim.nullspace_family.has_value()) {
            // Structured path (Phase 17+): use the typed family directly.
            switch (*claim.nullspace_family) {
                case NullspaceFamily::ScalarConstant:
                    c.family = gauge::NullspaceModeFamily::ScalarConstant;
                    break;
                case NullspaceFamily::ComponentwiseConstant:
                    c.family = gauge::NullspaceModeFamily::ComponentwiseConstant;
                    break;
                case NullspaceFamily::KernelOfSymGrad:
                    c.family = gauge::NullspaceModeFamily::KernelOfSymGrad;
                    break;
                case NullspaceFamily::UserDefined:
                    c.family = gauge::NullspaceModeFamily::ScalarConstant;  // conservative
                    break;
            }
        } else {
            // Fallback: parse description text (legacy claims without structured fields)
            c.family = gauge::NullspaceModeFamily::ScalarConstant;
            if (claim.description.find("rigid") != std::string::npos
                || claim.description.find("sym(grad") != std::string::npos) {
                c.family = gauge::NullspaceModeFamily::KernelOfSymGrad;
            } else if (claim.description.find("componentwise") != std::string::npos
                       || claim.description.find("per-component") != std::string::npos
                       || claim.description.find("vector") != std::string::npos) {
                c.family = gauge::NullspaceModeFamily::ComponentwiseConstant;
            }
        }

        c.reason = claim.description;
        candidates.push_back(std::move(c));
    }

    return candidates;
}

// ============================================================================
// descriptorsToEvidence
// ============================================================================

std::vector<gauge::AnchoringEvidence>
descriptorsToEvidence(const std::vector<BoundaryConditionDescriptor>& descriptors) {
    std::vector<gauge::AnchoringEvidence> evidence;

    for (const auto& desc : descriptors) {
        if (desc.primary_variable.kind != VariableKind::FieldComponent) continue;

        // For each nullspace mode family, check if the descriptor anchors it
        for (auto family : {gauge::NullspaceModeFamily::ScalarConstant,
                            gauge::NullspaceModeFamily::ComponentwiseConstant,
                            gauge::NullspaceModeFamily::KernelOfSymGrad}) {

            auto verdict = descriptorToVerdict(desc, family);

            // Only emit evidence for non-Unknown verdicts
            if (verdict == gauge::AnchoringVerdict::Unknown) continue;

            gauge::AnchoringEvidence ev;
            ev.field = desc.primary_variable.field_id;
            ev.component = desc.component;
            ev.region = -1;  // BC descriptors are global unless topology-scoped
            ev.family = family;
            ev.verdict = verdict;
            ev.source = desc.source;
            ev.boundary_marker = desc.boundary_marker;
            evidence.push_back(std::move(ev));
        }
    }

    return evidence;
}

// ============================================================================
// populateRegistryFromReport
// ============================================================================

void populateRegistryFromReport(gauge::GaugeRegistry& registry,
                                 const ProblemAnalysisReport& report,
                                 const std::vector<BoundaryConditionDescriptor>& bc_descriptors) {
    auto candidates = claimsToCandidates(report);
    for (auto& c : candidates) {
        registry.addCandidate(std::move(c));
    }

    auto evidence = descriptorsToEvidence(bc_descriptors);
    for (auto& e : evidence) {
        registry.addAnchoring(std::move(e));
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
