/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GaugeRegistry.h"
#include "GlobalConstraint.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace gauge {

// ============================================================================
// String conversions
// ============================================================================

const char* toString(NullspaceModeFamily f) noexcept {
    switch (f) {
        case NullspaceModeFamily::ScalarConstant:       return "ScalarConstant";
        case NullspaceModeFamily::ComponentwiseConstant: return "ComponentwiseConstant";
        case NullspaceModeFamily::KernelOfSymGrad:       return "KernelOfSymGrad";
    }
    return "Unknown";
}

const char* toString(Confidence c) noexcept {
    switch (c) {
        case Confidence::High:   return "High";
        case Confidence::Medium: return "Medium";
        case Confidence::Low:    return "Low";
    }
    return "Unknown";
}

const char* toString(AnchoringVerdict v) noexcept {
    switch (v) {
        case AnchoringVerdict::Anchored:          return "Anchored";
        case AnchoringVerdict::PartiallyAnchored: return "PartiallyAnchored";
        case AnchoringVerdict::Preserved:         return "Preserved";
        case AnchoringVerdict::Unknown:           return "Unknown";
    }
    return "Unknown";
}

const char* toString(GaugeStatus s) noexcept {
    switch (s) {
        case GaugeStatus::Anchored:       return "Anchored";
        case GaugeStatus::ExactNullspace:  return "ExactNullspace";
        case GaugeStatus::NearNullspace:   return "NearNullspace";
        case GaugeStatus::Unknown:        return "Unknown";
    }
    return "Unknown";
}

const char* toString(EnforcementPolicy p) noexcept {
    switch (p) {
        case EnforcementPolicy::None:                return "None";
        case EnforcementPolicy::PinDof:              return "PinDof";
        case EnforcementPolicy::MeanZeroElimination: return "MeanZeroElimination";
        case EnforcementPolicy::LagrangeMultiplier:  return "LagrangeMultiplier";
        case EnforcementPolicy::SolverNullspace:     return "SolverNullspace";
    }
    return "Unknown";
}

// ============================================================================
// GaugeRegistry — registration
// ============================================================================

void GaugeRegistry::addCandidate(GaugeCandidate candidate) {
    // Deduplicate: if a candidate with the same field/component/family/region
    // exists, an explicit declaration overrides a weaker symbolic inference,
    // and a higher confidence overrides a lower one within the same source class.
    for (auto& existing : candidates_) {
        if (existing.field == candidate.field &&
            existing.component == candidate.component &&
            existing.region == candidate.region &&
            existing.family == candidate.family) {
            // Explicit declaration always wins over inference
            if (candidate.source == CandidateSource::ExplicitDeclaration &&
                existing.source == CandidateSource::FormsInference) {
                existing = std::move(candidate);
                resolved_flag_ = false;
            }
            // Within same source class, higher confidence wins
            else if (candidate.source == existing.source &&
                     candidate.confidence < existing.confidence) {
                // Confidence enum: High=0, Medium=1, Low=2 — lower value is higher confidence
                existing = std::move(candidate);
                resolved_flag_ = false;
            }
            return;
        }
    }
    candidates_.push_back(std::move(candidate));
    resolved_flag_ = false;
}

void GaugeRegistry::addAnchoring(AnchoringEvidence evidence) {
    anchoring_.push_back(std::move(evidence));
    resolved_flag_ = false;
}

void GaugeRegistry::retagEvidenceRegions(const MarkerRegionResolver& get_marker_regions) {
    if (!get_marker_regions) return;

    std::vector<AnchoringEvidence> updated;
    updated.reserve(anchoring_.size());

    for (auto& ev : anchoring_) {
        // Only post-process global anchoring evidence that has a boundary marker.
        if (ev.region == -1 && ev.boundary_marker >= 0 &&
            (ev.verdict == AnchoringVerdict::Anchored ||
             ev.verdict == AnchoringVerdict::PartiallyAnchored)) {
            auto regions = get_marker_regions(ev.boundary_marker);
            if (!regions.empty()) {
                for (int r : regions) {
                    AnchoringEvidence per_region = ev;
                    per_region.region = r;
                    per_region.source += " [region " + std::to_string(r) + "]";
                    updated.push_back(std::move(per_region));
                }
                continue;  // replaced — don't keep the original
            }
        }
        updated.push_back(std::move(ev));
    }

    anchoring_ = std::move(updated);
    resolved_flag_ = false;
}

// ============================================================================
// GaugeRegistry — resolution
// ============================================================================

std::vector<GlobalIndex>
GaugeRegistry::getRegionFilteredDofs(const DofProvider& get_field_dofs,
                                      FieldId field, int component, int region) const {
    auto dofs = get_field_dofs(field, component);
    if (region < 0 || !region_provider_) {
        return dofs;
    }
    std::vector<GlobalIndex> filtered;
    filtered.reserve(dofs.size());
    for (const auto d : dofs) {
        if (region_provider_(d) == region) {
            filtered.push_back(d);
        }
    }
    return filtered;
}

void GaugeRegistry::resolve(const DofProvider& get_field_dofs,
                             const RegionProvider& get_region,
                             const CoordinateProvider& get_coords) {
    resolved_.clear();
    resolved_.reserve(candidates_.size());
    region_provider_ = get_region;  // store for applyEnforcement/buildNullspaceBasis

    // ================================================================
    // Phase 1: Expand multi-component candidates
    // ================================================================
    // When a ComponentwiseConstant or KernelOfSymGrad candidate has
    // component=-1, expand it into N per-component modes.
    std::vector<GaugeCandidate> comp_expanded;
    comp_expanded.reserve(candidates_.size() * 3);

    for (const auto& candidate : candidates_) {
        if (candidate.component == -1 &&
            (candidate.family == NullspaceModeFamily::ComponentwiseConstant ||
             candidate.family == NullspaceModeFamily::KernelOfSymGrad)) {
            // Infer component count from DofProvider
            auto all_dofs = get_field_dofs(candidate.field, -1);
            if (all_dofs.empty()) {
                comp_expanded.push_back(candidate);
                continue;
            }

            // Probe: get DOFs for component 0
            auto comp0_dofs = get_field_dofs(candidate.field, 0);
            if (comp0_dofs.empty() || comp0_dofs.size() == all_dofs.size()) {
                comp_expanded.push_back(candidate);
                continue;
            }

            const auto dofs_per_comp = comp0_dofs.size();
            const int n_comp = static_cast<int>(all_dofs.size() / dofs_per_comp);
            if (n_comp <= 0 || static_cast<std::size_t>(n_comp) * dofs_per_comp != all_dofs.size()) {
                comp_expanded.push_back(candidate);
                continue;
            }

            // Expand translation modes (one per component)
            for (int c = 0; c < n_comp; ++c) {
                GaugeCandidate per_comp = candidate;
                per_comp.component = c;
                if (candidate.family == NullspaceModeFamily::ComponentwiseConstant) {
                    per_comp.reason = candidate.reason + " [component " + std::to_string(c) + "]";
                } else {
                    per_comp.family = NullspaceModeFamily::ComponentwiseConstant;
                    per_comp.reason = "Translation mode from sym(grad) kernel [component " +
                                     std::to_string(c) + "]";
                }
                comp_expanded.push_back(std::move(per_comp));
            }

            // Expand rotation modes when CoordinateProvider is available
            if (candidate.family == NullspaceModeFamily::KernelOfSymGrad && get_coords) {
                const int n_rot = n_comp * (n_comp - 1) / 2;  // 3 in 3D, 1 in 2D
                for (int r = 0; r < n_rot; ++r) {
                    GaugeCandidate rot;
                    rot.field = candidate.field;
                    rot.component = -(2 + r);  // Negative sentinel: -2=rot0, -3=rot1, -4=rot2
                    rot.region = candidate.region;
                    rot.family = NullspaceModeFamily::KernelOfSymGrad;
                    rot.confidence = candidate.confidence;
                    rot.source = candidate.source;
                    rot.reason = "Rotation mode " + std::to_string(r) +
                                 " from sym(grad) kernel [dim=" + std::to_string(n_comp) + "]";
                    comp_expanded.push_back(std::move(rot));
                }

                std::fprintf(stderr,
                    "[GaugeRegistry] KernelOfSymGrad for field %d expanded to %d "
                    "translation + %d rotation constraints.\n",
                    static_cast<int>(candidate.field), n_comp, n_rot);
            } else if (candidate.family == NullspaceModeFamily::KernelOfSymGrad) {
                std::fprintf(stderr,
                    "[GaugeRegistry] KernelOfSymGrad for field %d expanded to %d "
                    "translation-only constraints. Rotational nullspace modes "
                    "require CoordinateProvider.\n",
                    static_cast<int>(candidate.field), n_comp);
            }
        } else {
            comp_expanded.push_back(candidate);
        }
    }

    // ================================================================
    // Phase 2: Expand region=-1 candidates when RegionProvider is set
    // ================================================================
    std::vector<GaugeCandidate> expanded;
    expanded.reserve(comp_expanded.size());

    if (get_region) {
        // Discover all regions by scanning DOFs of each field
        // Key: (field, component) → set of region IDs
        std::unordered_map<std::size_t, std::vector<int>> field_regions;

        for (const auto& candidate : comp_expanded) {
            if (candidate.region != -1) {
                expanded.push_back(candidate);
                continue;
            }

            // Build a hash key from field + component
            const auto key = static_cast<std::size_t>(candidate.field) * 10000 +
                             static_cast<std::size_t>(candidate.component + 1000);

            auto it = field_regions.find(key);
            if (it == field_regions.end()) {
                // Discover regions for this field/component
                auto dofs = get_field_dofs(candidate.field, candidate.component);
                std::vector<int> regions;
                for (const auto d : dofs) {
                    const int r = get_region(d);
                    bool found = false;
                    for (int existing_r : regions) {
                        if (existing_r == r) { found = true; break; }
                    }
                    if (!found) regions.push_back(r);
                }
                std::sort(regions.begin(), regions.end());
                it = field_regions.emplace(key, std::move(regions)).first;
            }

            const auto& regions = it->second;
            if (regions.size() <= 1) {
                // Single region — keep as global
                expanded.push_back(candidate);
            } else {
                // Multiple regions — expand into per-region candidates
                for (int r : regions) {
                    GaugeCandidate per_region = candidate;
                    per_region.region = r;
                    per_region.reason = candidate.reason + " [region " + std::to_string(r) + "]";
                    expanded.push_back(std::move(per_region));
                }
            }
        }
    } else {
        expanded = std::move(comp_expanded);
    }

    // ================================================================
    // Phase 3: Match evidence and classify each mode
    // ================================================================
    resolved_.reserve(expanded.size());

    for (const auto& candidate : expanded) {
        ResolvedMode mode;
        mode.candidate = candidate;

        // Gather all anchoring evidence for this field/component/family/region.
        // Matching rules:
        //  - Field must match exactly.
        //  - Component: evidence comp -1 matches any candidate comp; exact match otherwise.
        //  - Family: evidence family nullopt matches any candidate family.
        //  - Region: after retagEvidenceRegions(), most BC evidence has been
        //    converted from global to per-region.  Remaining global evidence
        //    with anchoring verdicts is blocked from per-region candidates to
        //    prevent over-anchoring.  Non-anchoring global evidence (Preserved,
        //    Unknown) passes through since it doesn't affect status.
        for (const auto& ev : anchoring_) {
            if (ev.field != candidate.field) continue;

            // Component matching
            const bool comp_match =
                (ev.component == -1) ||
                (ev.component == candidate.component);
            if (!comp_match) continue;

            // Family matching
            if (ev.family.has_value() && ev.family.value() != candidate.family) continue;

            // Region matching.
            // After retagEvidenceRegions(), BC evidence with a boundary_marker
            // has been converted from global (region=-1) to per-region.
            // Remaining global Anchored/PartiallyAnchored evidence (no
            // boundary_marker — e.g., explicit physics-module anchors for
            // unlabeled boundaries) is blocked from per-region candidates to
            // prevent over-anchoring.  On connected meshes (no region
            // expansion) this blocking does not apply since both evidence and
            // candidate are global.
            //
            // Known limitation: on disconnected meshes, any anchor source
            // that cannot provide a boundary_marker (kernel metadata,
            // formulation-level anchors for unlabeled faces) will be unable
            // to scope to the correct region.  The affected regions get
            // gauge enforcement even if one is genuinely anchored.  Labeled
            // boundary markers are required for correct per-region scoping.
            if (ev.region != candidate.region) {
                if (ev.region == -1 && candidate.region >= 0) {
                    // Global evidence → region-specific candidate.
                    if (ev.verdict == AnchoringVerdict::Anchored ||
                        ev.verdict == AnchoringVerdict::PartiallyAnchored) {
                        continue;  // reject: would over-anchor
                    }
                    // Preserved/Unknown: allow
                } else if (candidate.region == -1 && ev.region >= 0) {
                    // Region-specific evidence → global candidate: allow
                } else {
                    // Both region-specific but different regions: reject
                    continue;
                }
            }

            mode.anchoring.push_back(ev);
        }

        // Classify status based on anchoring evidence
        bool has_anchored = false;
        bool has_partially_anchored = false;
        for (const auto& ev : mode.anchoring) {
            if (ev.verdict == AnchoringVerdict::Anchored) {
                has_anchored = true;
            } else if (ev.verdict == AnchoringVerdict::PartiallyAnchored) {
                has_partially_anchored = true;
            }
        }

        if (has_anchored) {
            mode.status = GaugeStatus::Anchored;
            mode.policy = EnforcementPolicy::None;
        } else if (has_partially_anchored) {
            mode.status = GaugeStatus::NearNullspace;
            mode.policy = EnforcementPolicy::None;
        } else {
            if (candidate.confidence == Confidence::High) {
                mode.status = GaugeStatus::ExactNullspace;
            } else if (candidate.confidence == Confidence::Medium) {
                mode.status = GaugeStatus::NearNullspace;
            } else {
                mode.status = GaugeStatus::Unknown;
            }

            if (mode.status == GaugeStatus::ExactNullspace) {
                if (candidate.family == NullspaceModeFamily::ScalarConstant ||
                    candidate.family == NullspaceModeFamily::ComponentwiseConstant) {
                    mode.policy = EnforcementPolicy::MeanZeroElimination;
                } else {
                    // KernelOfSymGrad rotation modes → PinDof
                    mode.policy = EnforcementPolicy::PinDof;
                }
            } else if (mode.status == GaugeStatus::NearNullspace) {
                if (candidate.confidence == Confidence::Medium) {
                    mode.policy = EnforcementPolicy::PinDof;
                } else {
                    mode.policy = EnforcementPolicy::None;
                }
            } else {
                mode.policy = EnforcementPolicy::None;
            }
        }

        resolved_.push_back(std::move(mode));
    }

    resolved_flag_ = true;
}

// ============================================================================
// GaugeRegistry — enforcement
// ============================================================================

int GaugeRegistry::applyEnforcement(constraints::AffineConstraints& constraints,
                                     const DofProvider& get_field_dofs,
                                     const MassWeightProvider& get_mass_weights) {
    if (!resolved_flag_) {
        return 0;
    }

    int count = 0;

    for (const auto& mode : resolved_) {
        if (mode.policy == EnforcementPolicy::None) {
            continue;
        }

        // For rotation modes (component < -1), the sentinel component is not a
        // valid DofProvider key.  Map each rotation to a distinct component so
        // that independent rotation modes pin distinct DOFs:
        //   rot_idx 0 (ωz) → component 0
        //   rot_idx 1 (ωx) → component 1
        //   rot_idx 2 (ωy) → component 2
        int lookup_comp = mode.candidate.component;
        if (lookup_comp < -1) {
            const int rot_idx = -(lookup_comp + 2);
            lookup_comp = rot_idx;  // distinct component per rotation
        }
        auto dofs = getRegionFilteredDofs(get_field_dofs,
                                           mode.candidate.field, lookup_comp,
                                           mode.candidate.region);
        if (dofs.empty()) {
            continue;
        }

        switch (mode.policy) {
            case EnforcementPolicy::MeanZeroElimination: {
                // Use FE-correct lumped mass weights when available,
                // otherwise fall back to uniform weights (zeroMean).
                bool used_weights = false;
                if (get_mass_weights) {
                    // Fetch unfiltered DOFs and weights so indices align.
                    auto all_dofs = get_field_dofs(mode.candidate.field, lookup_comp);
                    auto all_weights = get_mass_weights(mode.candidate.field,
                                                         mode.candidate.component);
                    // Provider returns empty when weights are invalid (e.g.,
                    // unsupported cell types like Quad4 that produce zero volume).
                    if (!all_weights.empty() && all_weights.size() == all_dofs.size()) {
                        std::vector<double> final_weights;
                        std::vector<GlobalIndex> final_dofs;

                        // If region-filtered, extract matching weight subset
                        if (dofs.size() < all_dofs.size() && mode.candidate.region >= 0) {
                            final_dofs = dofs;
                            final_weights.reserve(dofs.size());
                            for (std::size_t i = 0; i < all_dofs.size(); ++i) {
                                if (region_provider_ &&
                                    region_provider_(all_dofs[i]) == mode.candidate.region) {
                                    final_weights.push_back(all_weights[i]);
                                }
                            }
                        } else {
                            final_dofs = std::move(dofs);
                            final_weights = std::move(all_weights);
                        }

                        if (final_weights.size() == final_dofs.size()) {
                            constraints::GlobalConstraintOptions wopts;
                            wopts.strategy = constraints::GlobalConstraintStrategy::WeightedMean;
                            auto gc = constraints::GlobalConstraint(
                                std::move(final_dofs), std::move(final_weights),
                                /*target=*/0.0, wopts);
                            gc.apply(constraints);
                            ++count;
                            used_weights = true;
                        }
                    }

                    if (used_weights) {
                        std::fprintf(stderr,
                            "[GaugeRegistry] Auto-enforcing weighted mean-zero constraint "
                            "for field %d (family=%s, confidence=%s, status=%s)\n",
                            static_cast<int>(mode.candidate.field),
                            toString(mode.candidate.family),
                            toString(mode.candidate.confidence),
                            toString(mode.status));
                        break;
                    }
                    // weights not usable — fall through to uniform
                }

                auto gc = constraints::GlobalConstraint::zeroMean(std::move(dofs));
                gc.apply(constraints);
                ++count;

                std::fprintf(stderr,
                    "[GaugeRegistry] Auto-enforcing mean-zero constraint for field %d "
                    "(family=%s, confidence=%s, status=%s)\n",
                    static_cast<int>(mode.candidate.field),
                    toString(mode.candidate.family),
                    toString(mode.candidate.confidence),
                    toString(mode.status));
                break;
            }

            case EnforcementPolicy::PinDof: {
                // Find an unconstrained DOF that provides a geometrically
                // independent constraint.  For rotation modes, successive
                // rotations must pin at different mesh vertices.  We skip
                // already-constrained DOFs and additionally skip one more
                // unconstrained DOF per rotation index to spread pins across
                // distinct vertices (important for rank-3 rigid-body gauge in 3D).
                //
                // Note: this is a pragmatic heuristic that depends on DOF ordering
                // (component-blocked, vertices numbered consecutively).  It works
                // for current H1/P1 component-blocked layouts but is not a general
                // geometry-aware ker(sym(grad)) enforcement.  A robust solution
                // would construct scalar constraints from the actual rotation basis
                // vectors — e.g., nullspacePinning(dofs, rotation_vector).
                int skip_count = 0;
                if (mode.candidate.component < -1) {
                    // rot_idx 0 → skip 0 extra, rot_idx 1 → skip 1, rot_idx 2 → skip 2
                    skip_count = -(mode.candidate.component + 2);
                }
                GlobalIndex pin_dof = dofs[0];
                int unconstrained_seen = 0;
                for (const auto d : dofs) {
                    if (!constraints.isConstrained(d)) {
                        if (unconstrained_seen >= skip_count) {
                            pin_dof = d;
                            break;
                        }
                        ++unconstrained_seen;
                    }
                }
                auto gc = constraints::GlobalConstraint::pinDof(pin_dof, 0.0);
                gc.apply(constraints);
                ++count;

                if (mode.status == GaugeStatus::NearNullspace) {
                    std::fprintf(stderr,
                        "[GaugeRegistry] WARNING: Uncertain nullspace for field %d "
                        "(family=%s, confidence=%s). Applying conservative pin fallback. "
                        "Reason: %s\n",
                        static_cast<int>(mode.candidate.field),
                        toString(mode.candidate.family),
                        toString(mode.candidate.confidence),
                        mode.candidate.reason.c_str());
                } else {
                    std::fprintf(stderr,
                        "[GaugeRegistry] Auto-pinning DOF %lld for field %d "
                        "(family=%s, status=%s)\n",
                        static_cast<long long>(pin_dof),
                        static_cast<int>(mode.candidate.field),
                        toString(mode.candidate.family),
                        toString(mode.status));
                }
                break;
            }

            case EnforcementPolicy::SolverNullspace:
                std::fprintf(stderr,
                    "[GaugeRegistry] Solver-side nullspace handling for field %d "
                    "(family=%s) — no algebraic constraint created\n",
                    static_cast<int>(mode.candidate.field),
                    toString(mode.candidate.family));
                break;

            case EnforcementPolicy::LagrangeMultiplier:
                std::fprintf(stderr,
                    "[GaugeRegistry] Lagrange multiplier enforcement requested for field %d "
                    "but not yet implemented — falling back to PinDof\n",
                    static_cast<int>(mode.candidate.field));
                {
                    auto gc = constraints::GlobalConstraint::pinDof(dofs[0], 0.0);
                    gc.apply(constraints);
                    ++count;
                }
                break;

            case EnforcementPolicy::None:
                break;
        }
    }

    return count;
}

// ============================================================================
// GaugeRegistry — nullspace basis construction
// ============================================================================

// Helper: compute rotation mode basis vector for a given rotation index.
// rot_idx: 0=ωz (2D/3D), 1=ωx (3D), 2=ωy (3D)
// For 2D (dim=2): only rot_idx=0 → v[dof_x] = -(y-ȳ), v[dof_y] = (x-x̄)
// For 3D (dim=3):
//   rot_idx 0 (ωz): v[dof_x] = -(y-ȳ), v[dof_y] = (x-x̄)
//   rot_idx 1 (ωx): v[dof_y] = -(z-z̄), v[dof_z] = (y-ȳ)
//   rot_idx 2 (ωy): v[dof_x] = (z-z̄), v[dof_z] = -(x-x̄)
static void buildRotationVector(
    std::vector<double>& vec,
    int rot_idx, int dim,
    FieldId field,
    const GaugeRegistry::DofProvider& get_field_dofs,
    const GaugeRegistry::CoordinateProvider& get_coords)
{
    // Get DOFs for each component
    std::vector<std::vector<GlobalIndex>> comp_dofs(static_cast<std::size_t>(dim));
    for (int c = 0; c < dim; ++c) {
        comp_dofs[static_cast<std::size_t>(c)] = get_field_dofs(field, c);
    }
    const auto n_verts = comp_dofs[0].size();
    if (n_verts == 0) return;

    // Compute centroid
    double cx = 0, cy = 0, cz = 0;
    for (std::size_t i = 0; i < n_verts; ++i) {
        auto coords = get_coords(field, comp_dofs[0][i]);
        cx += coords[0]; cy += coords[1]; cz += coords[2];
    }
    const double inv_n = 1.0 / static_cast<double>(n_verts);
    cx *= inv_n; cy *= inv_n; cz *= inv_n;

    // Fill rotation vector entries
    for (std::size_t i = 0; i < n_verts; ++i) {
        auto coords = get_coords(field, comp_dofs[0][i]);
        const double dx = coords[0] - cx;
        const double dy = coords[1] - cy;
        const double dz = coords[2] - cz;

        const auto n = vec.size();
        auto set = [&](int comp, std::size_t vert_idx, double val) {
            auto d = comp_dofs[static_cast<std::size_t>(comp)][vert_idx];
            if (d >= 0 && static_cast<std::size_t>(d) < n) {
                vec[static_cast<std::size_t>(d)] = val;
            }
        };

        if (dim == 2) {
            // ωz rotation: v_x = -dy, v_y = dx
            set(0, i, -dy);
            set(1, i, dx);
        } else {
            switch (rot_idx) {
                case 0: // ωz: v_x = -(y-ȳ), v_y = (x-x̄)
                    set(0, i, -dy);
                    set(1, i, dx);
                    break;
                case 1: // ωx: v_y = -(z-z̄), v_z = (y-ȳ)
                    set(1, i, -dz);
                    set(2, i, dy);
                    break;
                case 2: // ωy: v_x = (z-z̄), v_z = -(x-x̄)
                    set(0, i, dz);
                    set(2, i, -dx);
                    break;
                default: break;
            }
        }
    }
}

// Helper: Gram-Schmidt orthogonalize vec against existing basis vectors.
// Returns the norm after orthogonalization.
static double gramSchmidtProject(std::vector<double>& vec,
                                  const std::vector<std::vector<double>>& basis) {
    for (const auto& b : basis) {
        double dot = 0;
        for (std::size_t i = 0; i < vec.size(); ++i) {
            dot += vec[i] * b[i];
        }
        for (std::size_t i = 0; i < vec.size(); ++i) {
            vec[i] -= dot * b[i];
        }
    }
    double norm = 0;
    for (double v : vec) norm += v * v;
    return std::sqrt(norm);
}

std::vector<std::vector<double>>
GaugeRegistry::buildNullspaceBasis(GlobalIndex n_total_dofs,
                                    const DofProvider& get_field_dofs,
                                    const CoordinateProvider& get_coords) const
{
    std::vector<std::vector<double>> basis;

    if (!resolved_flag_ || n_total_dofs <= 0) {
        return basis;
    }

    for (const auto& mode : resolved_) {
        // Only build basis vectors for modes with SolverNullspace policy.
        if (mode.policy != EnforcementPolicy::SolverNullspace) {
            continue;
        }

        const auto& candidate = mode.candidate;
        int lookup_comp = candidate.component;
        if (lookup_comp < -1) lookup_comp = 0;
        auto dofs = getRegionFilteredDofs(get_field_dofs, candidate.field,
                                           lookup_comp, candidate.region);
        if (dofs.empty() && candidate.component < -1) {
            // For rotation modes, DOFs are fetched per-component inside the helper
        } else if (dofs.empty()) {
            continue;
        }

        const auto n = static_cast<std::size_t>(n_total_dofs);

        switch (candidate.family) {
            case NullspaceModeFamily::ScalarConstant: {
                std::vector<double> vec(n, 0.0);
                const double val = 1.0 / std::sqrt(static_cast<double>(dofs.size()));
                for (GlobalIndex d : dofs) {
                    if (d >= 0 && static_cast<std::size_t>(d) < n) {
                        vec[static_cast<std::size_t>(d)] = val;
                    }
                }
                basis.push_back(std::move(vec));
                break;
            }

            case NullspaceModeFamily::ComponentwiseConstant: {
                int n_comp = 3;
                if (dofs.size() % 3u != 0 && dofs.size() % 2u == 0) {
                    n_comp = 2;
                } else if (dofs.size() % 3u != 0) {
                    n_comp = 1;
                }
                const auto dofs_per_comp = dofs.size() / static_cast<std::size_t>(n_comp);
                if (dofs_per_comp == 0) break;

                for (int c = 0; c < n_comp; ++c) {
                    std::vector<double> vec(n, 0.0);
                    const double val = 1.0 / std::sqrt(static_cast<double>(dofs_per_comp));
                    const std::size_t start = static_cast<std::size_t>(c) * dofs_per_comp;
                    const std::size_t end = start + dofs_per_comp;
                    for (std::size_t i = start; i < end && i < dofs.size(); ++i) {
                        GlobalIndex d = dofs[i];
                        if (d >= 0 && static_cast<std::size_t>(d) < n) {
                            vec[static_cast<std::size_t>(d)] = val;
                        }
                    }
                    basis.push_back(std::move(vec));
                }
                break;
            }

            case NullspaceModeFamily::KernelOfSymGrad: {
                // Rotation mode: candidate.component is -(2+rot_idx)
                if (candidate.component < -1 && get_coords) {
                    const int rot_idx = -(candidate.component + 2);
                    // Infer dimension from the reason string or from DOFs
                    auto comp0_dofs = get_field_dofs(candidate.field, 0);
                    auto all_dofs = get_field_dofs(candidate.field, -1);
                    if (comp0_dofs.empty() || all_dofs.empty()) break;
                    const int dim = static_cast<int>(all_dofs.size() / comp0_dofs.size());

                    std::vector<double> vec(n, 0.0);
                    buildRotationVector(vec, rot_idx, dim, candidate.field,
                                        get_field_dofs, get_coords);

                    // Orthogonalize against all existing basis vectors
                    double norm = gramSchmidtProject(vec, basis);
                    if (norm > 1e-12) {
                        const double inv_norm = 1.0 / norm;
                        for (auto& v : vec) v *= inv_norm;
                        basis.push_back(std::move(vec));
                    }
                } else {
                    // Translation modes — same as ComponentwiseConstant
                    int n_comp = 3;
                    if (dofs.size() % 3u != 0 && dofs.size() % 2u == 0) {
                        n_comp = 2;
                    } else if (dofs.size() % 3u != 0) {
                        n_comp = 1;
                    }
                    const auto dofs_per_comp = dofs.size() / static_cast<std::size_t>(n_comp);
                    if (dofs_per_comp == 0) break;

                    for (int c = 0; c < n_comp; ++c) {
                        std::vector<double> vec(n, 0.0);
                        const double val = 1.0 / std::sqrt(static_cast<double>(dofs_per_comp));
                        const std::size_t start = static_cast<std::size_t>(c) * dofs_per_comp;
                        const std::size_t end = start + dofs_per_comp;
                        for (std::size_t i = start; i < end && i < dofs.size(); ++i) {
                            GlobalIndex d = dofs[i];
                            if (d >= 0 && static_cast<std::size_t>(d) < n) {
                                vec[static_cast<std::size_t>(d)] = val;
                            }
                        }
                        basis.push_back(std::move(vec));
                    }

                    if (!get_coords) {
                        std::fprintf(stderr,
                            "[GaugeRegistry] WARNING: KernelOfSymGrad nullspace basis for "
                            "field %d includes only translation modes (%d vectors). "
                            "Rotation modes require CoordinateProvider.\n",
                            static_cast<int>(candidate.field), n_comp);
                    }
                }
                break;
            }
        }
    }

    return basis;
}

// ============================================================================
// GaugeRegistry — diagnostics
// ============================================================================

std::string GaugeRegistry::diagnosticReport() const {
    std::ostringstream os;

    os << "=== GaugeRegistry Diagnostic Report ===\n";
    os << "Candidates: " << candidates_.size() << "\n";
    os << "Anchoring evidence: " << anchoring_.size() << "\n";
    os << "Resolved: " << (resolved_flag_ ? "yes" : "no") << "\n\n";

    if (!candidates_.empty()) {
        os << "--- Candidates ---\n";
        for (std::size_t i = 0; i < candidates_.size(); ++i) {
            const auto& c = candidates_[i];
            os << "  [" << i << "] field=" << c.field
               << " component=" << c.component
               << " region=" << c.region
               << " family=" << toString(c.family)
               << " confidence=" << toString(c.confidence)
               << " source=" << (c.source == CandidateSource::FormsInference ? "Forms" : "Explicit")
               << "\n      reason: " << c.reason << "\n";
        }
        os << "\n";
    }

    if (!anchoring_.empty()) {
        os << "--- Anchoring Evidence ---\n";
        for (std::size_t i = 0; i < anchoring_.size(); ++i) {
            const auto& e = anchoring_[i];
            os << "  [" << i << "] field=" << e.field
               << " component=" << e.component
               << " region=" << e.region
               << " family=" << (e.family.has_value() ? toString(e.family.value()) : "*")
               << " verdict=" << toString(e.verdict)
               << " source: " << e.source << "\n";
        }
        os << "\n";
    }

    if (resolved_flag_ && !resolved_.empty()) {
        os << "--- Resolved Modes ---\n";
        for (std::size_t i = 0; i < resolved_.size(); ++i) {
            const auto& m = resolved_[i];
            os << "  [" << i << "] field=" << m.candidate.field
               << " family=" << toString(m.candidate.family)
               << " status=" << toString(m.status)
               << " policy=" << toString(m.policy)
               << " (anchoring evidence: " << m.anchoring.size() << ")\n";
        }
    }

    return os.str();
}

// ============================================================================
// GaugeRegistry — reset
// ============================================================================

void GaugeRegistry::clear() noexcept {
    candidates_.clear();
    anchoring_.clear();
    resolved_.clear();
    resolved_flag_ = false;
}

} // namespace gauge
} // namespace FE
} // namespace svmp
