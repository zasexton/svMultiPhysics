/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/DAEStructureAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace analysis {

std::string DAEStructureAnalyzer::name() const {
    return "DAEStructureAnalyzer";
}

void DAEStructureAnalyzer::run(const ProblemAnalysisContext& context,
                               ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();
    const auto& var_descs = context.variableDescriptors();

    // =====================================================================
    // Build per-variable temporal classification from contributions
    // =====================================================================

    struct VarTemporalInfo {
        VariableKey key;
        bool has_mass_like{false};       // has temporal (mass-like) contribution
        bool has_algebraic{false};       // has steady/constraint contribution
        bool participates_in_constraint{false};
        int max_derivative_order{0};
    };

    std::unordered_map<VariableKey, VarTemporalInfo, VariableKeyHash> var_temporal;

    // Populate from contributions' TemporalDescriptor
    for (const auto& contrib : contributions) {
        for (const auto& tv : contrib.test_variables) {
            auto& info = var_temporal[tv];
            info.key = tv;

            if (contrib.temporal.has_value()) {
                const auto& td = *contrib.temporal;
                if (td.kind == TemporalContributionKind::MassLike ||
                    td.kind == TemporalContributionKind::DampedMassLike) {
                    info.has_mass_like = true;
                    if (td.derivative_order > info.max_derivative_order) {
                        info.max_derivative_order = td.derivative_order;
                    }
                } else if (td.kind == TemporalContributionKind::PureConstraint) {
                    info.has_algebraic = true;
                } else if (td.kind == TemporalContributionKind::None) {
                    info.has_algebraic = true;
                }
            }

            if (contrib.role == ContributionRole::ConstraintBlock) {
                info.participates_in_constraint = true;
            }
        }
    }

    // Enrich from variable descriptors
    for (const auto& vd : var_descs) {
        auto it = var_temporal.find(vd.key);
        if (it != var_temporal.end()) {
            if (vd.temporal_state_kind == TemporalStateKind::Dynamic) {
                it->second.has_mass_like = true;
            } else if (vd.temporal_state_kind == TemporalStateKind::Algebraic) {
                it->second.has_algebraic = true;
            }
            if (vd.participates_in_constraint_blocks) {
                it->second.participates_in_constraint = true;
            }
            if (vd.max_time_derivative_order > it->second.max_derivative_order) {
                it->second.max_derivative_order = vd.max_time_derivative_order;
            }
        } else {
            // Variable descriptor exists but no contributions for it yet --
            // still track its temporal info
            VarTemporalInfo info;
            info.key = vd.key;
            if (vd.temporal_state_kind == TemporalStateKind::Dynamic) {
                info.has_mass_like = true;
            } else if (vd.temporal_state_kind == TemporalStateKind::Algebraic) {
                info.has_algebraic = true;
            }
            info.participates_in_constraint = vd.participates_in_constraint_blocks;
            info.max_derivative_order = vd.max_time_derivative_order;
            var_temporal[vd.key] = info;
        }
    }

    // No temporal information at all => no-op
    if (var_temporal.empty()) return;

    // Check if we have ANY temporal information (at least one variable with
    // mass_like or algebraic classification)
    bool any_temporal_info = false;
    for (const auto& [key, info] : var_temporal) {
        if (info.has_mass_like || info.has_algebraic) {
            any_temporal_info = true;
            break;
        }
    }
    if (!any_temporal_info) return;

    // =====================================================================
    // Classify the DAE structure
    // =====================================================================

    int dynamic_count = 0;
    int algebraic_count = 0;
    bool has_higher_order_dynamic = false;  // derivative_order > 1
    bool algebraic_in_constraint = false;

    std::vector<VariableKey> all_vars;

    for (const auto& [key, info] : var_temporal) {
        all_vars.push_back(key);

        if (info.has_mass_like && !info.has_algebraic) {
            ++dynamic_count;
            if (info.max_derivative_order > 1) {
                has_higher_order_dynamic = true;
            }
        } else if (info.has_algebraic && !info.has_mass_like) {
            ++algebraic_count;
            if (info.participates_in_constraint) {
                algebraic_in_constraint = true;
            }
        } else if (info.has_mass_like && info.has_algebraic) {
            // Mixed: treat as dynamic for counting
            ++dynamic_count;
            if (info.max_derivative_order > 1) {
                has_higher_order_dynamic = true;
            }
        }
        // else: neither mass_like nor algebraic => skip classification
    }

    DAEClass dae_class = DAEClass::Unknown;
    PropertyStatus status = PropertyStatus::Unknown;
    AnalysisConfidence confidence = AnalysisConfidence::Medium;
    std::string description;

    if (dynamic_count > 0 && algebraic_count == 0) {
        dae_class = DAEClass::PureODELike;
        status = PropertyStatus::Exact;
        confidence = AnalysisConfidence::High;
        description =
            "All " + std::to_string(dynamic_count) +
            " variable(s) have mass-like temporal contributions (pure ODE-like)";
    } else if (algebraic_count > 0 && dynamic_count == 0) {
        dae_class = DAEClass::AlgebraicSystem;
        status = PropertyStatus::Exact;
        confidence = AnalysisConfidence::High;
        description =
            "All " + std::to_string(algebraic_count) +
            " variable(s) have no temporal contributions (algebraic system)";
    } else if (dynamic_count > 0 && algebraic_count > 0) {
        // Mix of dynamic and algebraic
        if (algebraic_in_constraint && has_higher_order_dynamic) {
            dae_class = DAEClass::HigherIndexRisk;
            status = PropertyStatus::Likely;
            confidence = AnalysisConfidence::Medium;
            description =
                "Higher-index DAE risk: " + std::to_string(algebraic_count) +
                " algebraic variable(s) in constraint blocks with " +
                std::to_string(dynamic_count) +
                " higher-order dynamic variable(s)";
        } else {
            dae_class = DAEClass::Index1DAELike;
            status = PropertyStatus::Likely;
            confidence = AnalysisConfidence::Medium;
            description =
                "Index-1 DAE-like: " + std::to_string(dynamic_count) +
                " dynamic + " + std::to_string(algebraic_count) +
                " algebraic variable(s)";
        }
    }

    if (dae_class == DAEClass::Unknown) return;

    PropertyClaim claim;
    claim.kind = PropertyKind::DifferentialAlgebraicStructure;
    claim.status = status;
    claim.confidence = confidence;
    claim.dae_class = dae_class;
    claim.variables = all_vars;
    claim.description = std::move(description);
    claim.claim_origin = "DAEStructureAnalyzer";
    claim.addEvidence("DAEStructureAnalyzer",
        "Classification from TemporalDescriptor and temporal_state_kind metadata",
        confidence);
    report.claims.push_back(std::move(claim));
}

} // namespace analysis
} // namespace FE
} // namespace svmp
