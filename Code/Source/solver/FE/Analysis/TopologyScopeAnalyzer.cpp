/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TopologyScopeAnalyzer.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"

#include <set>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct RegionNullspaceAnchoring {
    bool scalar_constant_anchored{false};
    bool translation_anchored{false};
    bool rotation_anchored{false};
    bool all_components_anchored{false};
    std::set<int> anchored_components;
    std::vector<std::string> evidence_ids;
};

bool componentMatches(int claim_component, int evidence_component) noexcept
{
    return claim_component < 0 ||
           evidence_component < 0 ||
           claim_component == evidence_component;
}

void recordAnchoringEvidence(RegionNullspaceAnchoring& anchoring,
                             const BoundaryConditionDescriptor& bc)
{
    if (bc.anchors_constant_mode) {
        anchoring.scalar_constant_anchored = true;
    }
    if (bc.anchors_rigid_body_translation) {
        anchoring.translation_anchored = true;
    }
    if (bc.anchors_rigid_body_rotation) {
        anchoring.rotation_anchored = true;
    }
    if (bc.component < 0) {
        anchoring.all_components_anchored = true;
    } else {
        anchoring.anchored_components.insert(bc.component);
    }
    if (!bc.source.empty()) {
        anchoring.evidence_ids.push_back(bc.source);
    }
}

bool componentAnchored(const RegionNullspaceAnchoring& anchoring,
                       int component) noexcept
{
    if (anchoring.all_components_anchored) {
        return true;
    }
    if (component < 0) {
        return false;
    }
    return anchoring.anchored_components.find(component) !=
           anchoring.anchored_components.end();
}

bool familyAnchored(NullspaceFamily family,
                    const RegionNullspaceAnchoring& anchoring,
                    int component) noexcept
{
    switch (family) {
        case NullspaceFamily::ScalarConstant:
        case NullspaceFamily::GaugeConstant:
            return anchoring.scalar_constant_anchored;
        case NullspaceFamily::ComponentwiseConstant:
        case NullspaceFamily::VectorConstant:
            return componentAnchored(anchoring, component);
        case NullspaceFamily::RigidTranslation:
            return anchoring.translation_anchored;
        case NullspaceFamily::RigidRotation:
            return anchoring.rotation_anchored;
        case NullspaceFamily::RigidBody:
        case NullspaceFamily::KernelOfSymGrad:
            return anchoring.translation_anchored &&
                   anchoring.rotation_anchored;
        case NullspaceFamily::HarmonicField:
        case NullspaceFamily::GradientKernel:
        case NullspaceFamily::CurlKernel:
        case NullspaceFamily::DivergenceFreeKernel:
        case NullspaceFamily::UserDefined:
            return false;
    }
    return false;
}

} // namespace

std::string TopologyScopeAnalyzer::name() const {
    return "TopologyScopeAnalyzer";
}

void TopologyScopeAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    const auto* topo = context.topologyContext();
    if (!topo) return;
    if (!topo->isDisconnected()) return;

    auto nullspace_claims = report.claimsOfKind(PropertyKind::Nullspace);
    if (nullspace_claims.empty()) return;

    const auto& bcs = context.bcDescriptors();
    const auto* cs = context.constraintSummary();
    int num_regions = topo->numRegions();

    // ContributionDescriptor included for future use (e.g., per-region
    // contribution filtering when interface topology is available).
    // Currently the topology-scope analysis relies on BC descriptors and
    // constraint summary, which are the authoritative sources for
    // per-marker anchoring information.

    for (const auto* ns_claim : nullspace_claims) {
        if (ns_claim->region >= 0) continue;  // already region-scoped

        FieldId fid = ns_claim->field;
        if (fid == INVALID_FIELD_ID) continue;
        int claim_comp = ns_claim->component;

        for (int region = 0; region < num_regions; ++region) {
            const auto& comp = topo->components[static_cast<std::size_t>(region)];
            const auto& region_markers = comp.boundary_markers;

            RegionNullspaceAnchoring anchoring;

            for (const auto& bc : bcs) {
                if (bc.primary_variable.kind != VariableKind::FieldComponent) continue;
                if (bc.primary_variable.field_id != fid) continue;

                // Skip algebraic relations — they preserve, not anchor
                if (bc.enforcement_kind == EnforcementKind::AffineRelation) continue;

                // Component filtering: a BC with component == -1 applies to all.
                // A BC with a specific component only anchors that component's claim.
                if (!componentMatches(claim_comp, bc.component)) continue;

                // Check if this BC's marker is on this region
                if (bc.boundary_marker >= 0) {
                    if (region_markers.find(bc.boundary_marker) == region_markers.end()) {
                        continue;
                    }
                }

                // Use explicit anchoring flags, consistent with ConstraintRankAnalyzer
                recordAnchoringEvidence(anchoring, bc);
            }

            // Also check constraint summary for per-region Dirichlet DOFs
            if (cs) {
                for (const auto& cset : cs->constrained_sets) {
                    if (cset.field != fid) continue;
                    if (cset.region != region) continue;
                    // Match component
                    if (claim_comp >= 0 && cset.component != claim_comp) continue;
                    if (cset.num_constrained_dofs > 0 &&
                        cset.constraint_source != "AffineRelation") {
                        anchoring.scalar_constant_anchored = true;
                        if (cset.component < 0) {
                            anchoring.all_components_anchored = true;
                        } else {
                            anchoring.anchored_components.insert(cset.component);
                        }
                        anchoring.evidence_ids.push_back(cset.constraint_source);
                    }
                }
            }

            const bool family_known = ns_claim->nullspace_family.has_value();
            const bool region_anchored =
                family_known &&
                familyAnchored(*ns_claim->nullspace_family,
                               anchoring,
                               claim_comp);

            if (!region_anchored) {
                PropertyClaim claim;
                claim.kind = PropertyKind::TopologyScopedKernel;
                claim.status = family_known
                    ? ns_claim->status
                    : PropertyStatus::Unknown;
                claim.confidence = family_known
                    ? ns_claim->confidence
                    : AnalysisConfidence::Low;
                claim.certification_class = CertificationClass::NotCertified;
                claim.field = fid;
                claim.component = ns_claim->component;
                claim.region = region;
                claim.domain = DomainKind::Cell;
                claim.variables = ns_claim->variables;
                claim.nullspace_family = ns_claim->nullspace_family;
                claim.description =
                    (family_known
                        ? "Structured nullspace family is not anchored on mesh region "
                        : "Nullspace family metadata is missing for mesh region ") +
                    std::to_string(region) + " (of " +
                    std::to_string(num_regions) +
                    " disconnected regions): " + ns_claim->description;
                claim.addEvidence("TopologyScopeAnalyzer",
                    "Region " + std::to_string(region) + " has " +
                    std::to_string(comp.boundary_markers.size()) +
                    " boundary markers; anchoring evidence is scalar=" +
                    std::string(anchoring.scalar_constant_anchored ? "true" : "false") +
                    ", translation=" +
                    std::string(anchoring.translation_anchored ? "true" : "false") +
                    ", rotation=" +
                    std::string(anchoring.rotation_anchored ? "true" : "false"),
                    claim.confidence);
                report.claims.push_back(std::move(claim));
            }
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
