/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/DAEStructureAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

bool containsVariable(const std::vector<VariableKey>& values,
                      const VariableKey& variable)
{
    return std::find(values.begin(), values.end(), variable) != values.end();
}

bool coversVariables(const std::vector<VariableKey>& evidence_variables,
                     const std::vector<VariableKey>& claim_variables)
{
    if (evidence_variables.empty()) return false;
    for (const auto& variable : claim_variables) {
        if (!containsVariable(evidence_variables, variable)) {
            return false;
        }
    }
    return true;
}

const DAEStructureEvidenceSummary* findDAEEvidence(
    const AnalysisSummarySet* summaries,
    const std::vector<VariableKey>& variables)
{
    if (!summaries) return nullptr;
    for (const auto& summary : summaries->dae_structure_evidence) {
        if (coversVariables(summary.variables, variables)) {
            return &summary;
        }
    }
    return nullptr;
}

Real effectiveTolerance(const DAEStructureEvidenceSummary& summary) noexcept
{
    return numeric::finiteDeclaredTolerance(summary.residual_tolerance)
        ? summary.residual_tolerance
        : Real{1.0e-10};
}

} // namespace

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
            } else if (vd.temporal_state_kind == TemporalStateKind::Mixed) {
                it->second.has_mass_like = true;
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
            } else if (vd.temporal_state_kind == TemporalStateKind::Mixed) {
                info.has_mass_like = true;
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
            ++dynamic_count;
            ++algebraic_count;
            if (info.participates_in_constraint) {
                algebraic_in_constraint = true;
            }
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

    const auto* dae_evidence =
        findDAEEvidence(context.analysisSummaries(), all_vars);
    std::optional<CertificationClass> certification;
    if (dynamic_count > 0 && algebraic_count > 0 && dae_evidence) {
        const Real tol = effectiveTolerance(*dae_evidence);
        const bool tolerance_declared =
            numeric::finiteDeclaredTolerance(
                dae_evidence->residual_tolerance);
        const bool residual_finite =
            numeric::finite(dae_evidence->initial_constraint_residual);
        const bool numeric_evidence_valid =
            tolerance_declared && residual_finite;
        const bool consistent_initial_state =
            dae_evidence->consistent_initial_condition_evidence_present &&
            numeric_evidence_valid &&
            std::abs(dae_evidence->initial_constraint_residual) <= tol;
        const bool hidden_constraint_violation =
            dae_evidence->hidden_constraint_metadata_present &&
            dae_evidence->hidden_constraint_count > 0u;
        const bool rank_violation =
            dae_evidence->algebraic_jacobian_rank_metadata_present &&
            !dae_evidence->algebraic_jacobian_full_rank;
        const bool semi_explicit_scope_complete =
            !dae_evidence->dae_index_theorem_id.empty() &&
            dae_evidence->local_validity_scope_present &&
            !dae_evidence->dae_index_scope.empty() &&
            dae_evidence->smoothness_or_regular_operator_evidence_present;
        const bool semi_explicit_index1_evidence =
            dae_evidence->dae_form_class == DAEFormClass::SemiExplicit &&
            dae_evidence->mass_matrix_rank_metadata_present &&
            dae_evidence->algebraic_jacobian_rank_metadata_present &&
            dae_evidence->algebraic_jacobian_full_rank &&
            dae_evidence->hidden_constraint_metadata_present &&
            dae_evidence->hidden_constraint_count == 0u &&
            consistent_initial_state;
        const bool index1_certified =
            semi_explicit_index1_evidence &&
            semi_explicit_scope_complete;
        const bool descriptor_index1_certified =
            (dae_evidence->dae_form_class == DAEFormClass::DescriptorPencil ||
             dae_evidence->dae_form_class == DAEFormClass::FullyImplicit) &&
            dae_evidence->descriptor_pencil_metadata_present &&
            dae_evidence->regular_descriptor_pencil_evidence_present &&
            dae_evidence->strangeness_index_metadata_present &&
            dae_evidence->strangeness_index >= 0 &&
            dae_evidence->strangeness_index <= 1 &&
            dae_evidence->projector_index_metadata_present &&
            dae_evidence->projector_consistency_evidence_present &&
            dae_evidence->hidden_constraint_metadata_present &&
            dae_evidence->hidden_constraint_count == 0u &&
            consistent_initial_state &&
            !dae_evidence->dae_index_theorem_id.empty();
        const bool semi_explicit_metadata_missing =
            dae_evidence->dae_form_class != DAEFormClass::SemiExplicit &&
            !descriptor_index1_certified;

        if (hidden_constraint_violation || rank_violation) {
            dae_class = DAEClass::HigherIndexRisk;
            status = PropertyStatus::Violated;
            confidence = AnalysisConfidence::High;
            certification = CertificationClass::Violated;
            description =
                "DAE structure evidence reports hidden constraints or rank-deficient algebraic Jacobian";
        } else if (index1_certified) {
            dae_class = DAEClass::Index1DAELike;
            status = PropertyStatus::Preserved;
            confidence = AnalysisConfidence::High;
            certification = CertificationClass::Certified;
            description =
                "Index-1 DAE structure certified by mass-rank, full-rank algebraic Jacobian, hidden-constraint, consistent-initialization, theorem, local-scope, and regularity evidence";
        } else if (descriptor_index1_certified) {
            dae_class = DAEClass::Index1DAELike;
            status = PropertyStatus::Preserved;
            confidence = AnalysisConfidence::High;
            certification = CertificationClass::Certified;
            description =
                "Index-1 DAE structure certified by regular descriptor pencil, strangeness/projector, hidden-constraint, theorem, and consistent-initialization evidence";
        } else if (semi_explicit_index1_evidence &&
                   !semi_explicit_scope_complete) {
            certification = CertificationClass::NotCertified;
            status = PropertyStatus::Likely;
            confidence = AnalysisConfidence::Medium;
            description =
                "DAE evidence is compatible with semi-explicit index-1 form, but certification requires theorem, local validity scope, and smooth/regular operator metadata";
        } else if (semi_explicit_metadata_missing) {
            certification = CertificationClass::NotCertified;
            status = PropertyStatus::Likely;
            confidence = AnalysisConfidence::Medium;
            description =
                "DAE evidence is compatible with an index-1 interpretation, but certification requires semi-explicit form metadata or a richer descriptor/pencil index certificate";
        } else if (dae_evidence->consistent_initial_condition_evidence_present &&
                   !numeric_evidence_valid) {
            certification = CertificationClass::NotCertified;
            status = PropertyStatus::Likely;
            confidence = AnalysisConfidence::Medium;
            description =
                "DAE evidence is compatible with an index-1 interpretation, but consistent-initialization certification requires finite residual and tolerance evidence";
        } else {
            certification = CertificationClass::NotCertified;
        }
    }

    PropertyClaim claim;
    claim.kind = PropertyKind::DifferentialAlgebraicStructure;
    claim.status = status;
    claim.confidence = confidence;
    claim.dae_class = dae_class;
    claim.certification_class = certification;
    claim.variables = all_vars;
    claim.description = std::move(description);
    claim.claim_origin = "DAEStructureAnalyzer";
    claim.addEvidence("DAEStructureAnalyzer",
        "Classification from TemporalDescriptor and temporal_state_kind metadata",
        confidence);
    if (dae_evidence) {
        claim.addEvidence("DAEStructureAnalyzer",
            "DAEStructureEvidenceSummary system='" +
            dae_evidence->system_id +
            "', mass_rank_metadata=" +
            std::string(dae_evidence->mass_matrix_rank_metadata_present ? "true" : "false") +
            ", algebraic_rank_metadata=" +
            std::string(dae_evidence->algebraic_jacobian_rank_metadata_present ? "true" : "false") +
            ", algebraic_full_rank=" +
            std::string(dae_evidence->algebraic_jacobian_full_rank ? "true" : "false") +
            ", form_class=" +
            std::to_string(static_cast<int>(dae_evidence->dae_form_class)) +
            ", descriptor_pencil=" +
            std::string(dae_evidence->descriptor_pencil_metadata_present ? "true" : "false") +
            ", regular_descriptor_pencil=" +
            std::string(dae_evidence->regular_descriptor_pencil_evidence_present ? "true" : "false") +
            ", strangeness_index=" +
            std::to_string(dae_evidence->strangeness_index) +
            ", projector_index=" +
            std::string(dae_evidence->projector_index_metadata_present ? "true" : "false") +
            ", projector_consistency=" +
            std::string(dae_evidence->projector_consistency_evidence_present ? "true" : "false") +
            ", theorem='" + dae_evidence->dae_index_theorem_id + "'" +
            ", index_scope='" + dae_evidence->dae_index_scope + "'" +
            ", local_validity_scope=" +
            std::string(dae_evidence->local_validity_scope_present ? "true" : "false") +
            ", smooth_or_regular_operator=" +
            std::string(dae_evidence->smoothness_or_regular_operator_evidence_present ? "true" : "false") +
            ", hidden_constraints=" +
            std::to_string(dae_evidence->hidden_constraint_count) +
            ", initial_residual=" +
            std::to_string(dae_evidence->initial_constraint_residual) +
            ", residual_tolerance=" +
            std::to_string(dae_evidence->residual_tolerance) +
            ", finite_initial_residual=" +
            std::string(numeric::finite(
                dae_evidence->initial_constraint_residual) ? "true" : "false") +
            ", finite_declared_tolerance=" +
            std::string(numeric::finiteDeclaredTolerance(
                dae_evidence->residual_tolerance) ? "true" : "false"),
            confidence);
        if (dae_evidence->consistent_initial_condition_evidence_present &&
            (!numeric::finite(dae_evidence->initial_constraint_residual) ||
             !numeric::finiteDeclaredTolerance(
                 dae_evidence->residual_tolerance))) {
            AnalysisIssue issue;
            issue.severity = IssueSeverity::Warning;
            issue.message =
                "DAE structure evidence for system '" +
                dae_evidence->system_id +
                "' has invalid consistent-initialization residual or tolerance";
            report.issues.push_back(std::move(issue));
        }
    }
    report.claims.push_back(std::move(claim));
}

} // namespace analysis
} // namespace FE
} // namespace svmp
