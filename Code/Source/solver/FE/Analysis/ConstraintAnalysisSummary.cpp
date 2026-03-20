/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Constraints/AffineConstraints.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace svmp {
namespace FE {
namespace analysis {

// ============================================================================
// Queries
// ============================================================================

int ConstraintAnalysisSummary::totalConstrainedDofs() const noexcept {
    int total = 0;
    for (const auto& cs : constrained_sets) {
        if (cs.component == -1 && cs.region == -1) {
            total += cs.num_constrained_dofs;
        }
    }
    // If no aggregate sets exist, sum all sets
    if (total == 0) {
        for (const auto& cs : constrained_sets) {
            total += cs.num_constrained_dofs;
        }
    }
    return total;
}

int ConstraintAnalysisSummary::totalDofs() const noexcept {
    int total = 0;
    for (const auto& cs : constrained_sets) {
        if (cs.component == -1 && cs.region == -1) {
            total += cs.num_total_dofs;
        }
    }
    if (total == 0) {
        for (const auto& cs : constrained_sets) {
            total += cs.num_total_dofs;
        }
    }
    return total;
}

double ConstraintAnalysisSummary::constrainedFraction(FieldId field,
                                                       int component,
                                                       int region) const noexcept {
    for (const auto& cs : constrained_sets) {
        if (cs.field == field && cs.component == component && cs.region == region) {
            return cs.constrained_fraction;
        }
    }
    return -1.0;
}

std::vector<FieldId> ConstraintAnalysisSummary::unconstrainedFields() const {
    std::vector<FieldId> result;
    for (const auto& cs : constrained_sets) {
        if (cs.component == -1 && cs.region == -1 && cs.num_constrained_dofs == 0) {
            result.push_back(cs.field);
        }
    }
    return result;
}

std::vector<FieldId> ConstraintAnalysisSummary::fullyConstrainedFields() const {
    std::vector<FieldId> result;
    for (const auto& cs : constrained_sets) {
        if (cs.component == -1 && cs.region == -1
            && cs.num_total_dofs > 0
            && cs.num_constrained_dofs == cs.num_total_dofs) {
            result.push_back(cs.field);
        }
    }
    return result;
}

// ============================================================================
// Factory
// ============================================================================

namespace {

// Helper: classify a constraint as Dirichlet or affine
bool isDirichletConstraint(const constraints::AffineConstraints& ac, GlobalIndex d) {
    auto cv = ac.getConstraint(d);
    return cv && cv->isDirichlet();
}

std::string classifySource(bool has_dirichlet, bool has_affine) {
    if (has_dirichlet && has_affine) return "Mixed";
    if (has_dirichlet) return "StrongDirichlet";
    if (has_affine) return "AffineRelation";
    return "None";
}

double fraction(int constrained, int total) {
    return (total > 0) ? static_cast<double>(constrained) / static_cast<double>(total) : 0.0;
}

// Helper: scan a set of DOFs and produce a ConstrainedDofSet with per-slice classification.
ConstrainedDofSet scanDofSlice(const constraints::AffineConstraints& ac,
                                std::span<const GlobalIndex> dofs,
                                FieldId field, int component, int region) {
    ConstrainedDofSet cs;
    cs.field = field;
    cs.component = component;
    cs.region = region;
    cs.num_total_dofs = static_cast<int>(dofs.size());

    bool has_dirichlet = false;
    bool has_affine = false;
    int n_constrained = 0;

    for (auto d : dofs) {
        if (ac.isConstrained(d)) {
            ++n_constrained;
            if (isDirichletConstraint(ac, d)) {
                has_dirichlet = true;
            } else {
                has_affine = true;
            }
        }
    }

    cs.num_constrained_dofs = n_constrained;
    cs.constrained_fraction = fraction(n_constrained, cs.num_total_dofs);
    cs.constraint_source = classifySource(has_dirichlet, has_affine);
    return cs;
}

} // namespace

ConstraintAnalysisSummary
ConstraintAnalysisSummary::build(const constraints::AffineConstraints& ac,
                                  std::span<const FieldDofRange> fields,
                                  const TopologyAnalysisContext* /*topology*/,
                                  const DofRegionProvider& dof_region,
                                  const ComponentDofProvider& comp_dofs) {
    ConstraintAnalysisSummary summary;

    for (std::size_t fi = 0; fi < fields.size(); ++fi) {
        const auto& fr = fields[fi];

        // Build the full field DOF list
        std::vector<GlobalIndex> all_dofs;
        all_dofs.reserve(static_cast<std::size_t>(fr.num_dofs));
        for (GlobalIndex d = fr.dof_offset; d < fr.dof_offset + fr.num_dofs; ++d) {
            all_dofs.push_back(d);
        }

        // Aggregate set for the whole field (per-slice classification)
        summary.constrained_sets.push_back(
            scanDofSlice(ac, all_dofs, fr.field_id, -1, -1));

        // Per-component sets using the component DOF provider (layout-correct)
        if (fr.num_components > 1 && comp_dofs) {
            for (int comp = 0; comp < fr.num_components; ++comp) {
                auto comp_dof_vec = comp_dofs(fr.field_id, comp);
                if (!comp_dof_vec.empty()) {
                    summary.constrained_sets.push_back(
                        scanDofSlice(ac, comp_dof_vec, fr.field_id, comp, -1));
                }
            }
        }

        // Per-region sets (when dof_region provider is available)
        if (dof_region) {
            // Group DOFs by region
            std::map<int, std::vector<GlobalIndex>> region_dofs;
            for (auto d : all_dofs) {
                int region = dof_region(d);
                if (region >= 0) {
                    region_dofs[region].push_back(d);
                }
            }
            for (const auto& [region, rdofs] : region_dofs) {
                // Field-wide per-region
                summary.constrained_sets.push_back(
                    scanDofSlice(ac, rdofs, fr.field_id, -1, region));

                // Per-component per-region (if component DOFs are available)
                if (fr.num_components > 1 && comp_dofs) {
                    for (int comp = 0; comp < fr.num_components; ++comp) {
                        auto comp_dof_vec = comp_dofs(fr.field_id, comp);
                        // Intersect with region DOFs
                        std::vector<GlobalIndex> comp_region_dofs;
                        std::set<GlobalIndex> region_set(rdofs.begin(), rdofs.end());
                        for (auto cd : comp_dof_vec) {
                            if (region_set.count(cd)) {
                                comp_region_dofs.push_back(cd);
                            }
                        }
                        if (!comp_region_dofs.empty()) {
                            summary.constrained_sets.push_back(
                                scanDofSlice(ac, comp_region_dofs, fr.field_id, comp, region));
                        }
                    }
                }
            }
        }
    }

    // Detect structural conflicts in the constraint set.
    //
    // True value-level conflicts (same DOF prescribed with different values by
    // independent sources) cannot be detected because AffineConstraints silently
    // overwrites. However, we CAN detect:
    //   1. A DOF that appears as a slave in an affine relation (has master entries)
    //      but ALSO has a nonzero inhomogeneity that looks like a Dirichlet value —
    //      this typically happens when a periodic DOF is also Dirichlet-prescribed
    //      and the constraints were merged with overwrite.
    //
    // These are flagged as structural anomalies, not definitive conflicts, because
    // some legitimate constraint patterns (e.g., periodic + offset) have this shape.
    if (ac.isClosed()) {
        auto constrained_dofs = ac.getConstrainedDofs();
        for (auto d : constrained_dofs) {
            auto cv = ac.getConstraint(d);
            if (!cv) continue;

            // Flag DOFs that have BOTH master entries AND large inhomogeneity.
            // Pure Dirichlet: isDirichlet()=true (no masters, has inhomogeneity)
            // Pure periodic:  has masters, inhomogeneity ~0
            // Conflict:       has masters AND large inhomogeneity (Dirichlet
            //                 overwritten by periodic, or vice versa)
            if (!cv->isDirichlet() && !cv->entries.empty() &&
                std::abs(cv->inhomogeneity) > 1e-10) {
                ConstraintConflict conflict;
                conflict.dof = d;
                conflict.conflicting_sources.push_back("AffineRelation (master-slave)");
                conflict.conflicting_sources.push_back("Dirichlet-like (nonzero inhomogeneity)");
                conflict.description =
                    "DOF " + std::to_string(d) +
                    " has both master-slave entries and nonzero inhomogeneity (" +
                    std::to_string(cv->inhomogeneity) +
                    ") — possible conflicting Dirichlet/periodic/MPC constraints";
                summary.conflicts.push_back(std::move(conflict));
            }
        }
    }

    return summary;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
