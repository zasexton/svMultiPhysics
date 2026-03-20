/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TopologyScopeAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <set>

namespace svmp {
namespace FE {
namespace analysis {

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

        bool is_rigid_body = ns_claim->description.find("rigid") !=
                             std::string::npos;

        for (int region = 0; region < num_regions; ++region) {
            const auto& comp = topo->components[static_cast<std::size_t>(region)];
            const auto& region_markers = comp.boundary_markers;

            bool region_anchored = false;

            for (const auto& bc : bcs) {
                if (bc.primary_variable.kind != VariableKind::FieldComponent) continue;
                if (bc.primary_variable.field_id != fid) continue;

                // Skip algebraic relations — they preserve, not anchor
                if (bc.enforcement_kind == EnforcementKind::AffineRelation) continue;

                // Component filtering: a BC with component == -1 applies to all.
                // A BC with a specific component only anchors that component's claim.
                if (claim_comp >= 0 && bc.component >= 0 && bc.component != claim_comp) continue;

                // Check if this BC's marker is on this region
                if (bc.boundary_marker >= 0) {
                    if (region_markers.find(bc.boundary_marker) == region_markers.end()) {
                        continue;
                    }
                }

                // Use explicit anchoring flags, consistent with ConstraintRankAnalyzer
                if (is_rigid_body) {
                    if (bc.anchors_rigid_body_translation && bc.anchors_rigid_body_rotation) {
                        region_anchored = true;
                        break;
                    }
                } else {
                    if (bc.anchors_constant_mode) {
                        region_anchored = true;
                        break;
                    }
                }
            }

            // Also check constraint summary for per-region Dirichlet DOFs
            if (!region_anchored && !is_rigid_body && cs) {
                for (const auto& cset : cs->constrained_sets) {
                    if (cset.field != fid) continue;
                    if (cset.region != region) continue;
                    // Match component
                    if (claim_comp >= 0 && cset.component != claim_comp) continue;
                    if (claim_comp < 0 && cset.component != -1) continue;
                    if (cset.num_constrained_dofs > 0 &&
                        cset.constraint_source != "AffineRelation") {
                        region_anchored = true;
                        break;
                    }
                }
            }

            if (!region_anchored) {
                PropertyClaim claim;
                claim.kind = PropertyKind::TopologyScopedKernel;
                claim.status = ns_claim->status;
                claim.confidence = ns_claim->confidence;
                claim.field = fid;
                claim.component = ns_claim->component;
                claim.region = region;
                claim.domain = DomainKind::Cell;
                claim.variables = ns_claim->variables;
                claim.description =
                    "Nullspace mode unanchored on mesh region " +
                    std::to_string(region) + " (of " +
                    std::to_string(num_regions) +
                    " disconnected regions): " + ns_claim->description;
                claim.addEvidence("TopologyScopeAnalyzer",
                    "Region " + std::to_string(region) + " has " +
                    std::to_string(comp.boundary_markers.size()) +
                    " boundary markers, none with anchoring BC for this mode",
                    ns_claim->confidence);
                report.claims.push_back(std::move(claim));
            }
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
