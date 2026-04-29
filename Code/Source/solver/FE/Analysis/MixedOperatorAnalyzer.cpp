/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/MixedOperatorAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<VariableKey> blockVariables(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& v : block.test_variables) {
        appendUnique(variables, v);
    }
    for (const auto& v : block.trial_variables) {
        appendUnique(variables, v);
    }
    return variables;
}

void emitSchurSummaryClaim(ProblemAnalysisReport& report,
                           const ReducedMatrixSummary& summary)
{
    const auto& matrix = summary.free_free_matrix;
    const bool tagged_schur =
        matrix.block.role == ContributionRole::ConstraintBlock ||
        matrix.block.operator_tag.find("schur") != std::string::npos ||
        matrix.block.operator_tag.find("Schur") != std::string::npos;
    if (!tagged_schur) {
        return;
    }

    const bool exact_reduction =
        summary.reduction_exact_for_analysis &&
        matrix.rows > 0 &&
        matrix.cols > 0;

    PropertyClaim claim;
    claim.kind = PropertyKind::IndefiniteOperatorResolution;
    claim.status = exact_reduction ? PropertyStatus::Likely
                                   : PropertyStatus::Unknown;
    claim.confidence = exact_reduction ? AnalysisConfidence::Medium
                                       : AnalysisConfidence::Medium;
    claim.domain = matrix.block.domain;
    claim.variables = blockVariables(matrix.block);
    claim.reduced_definiteness_class =
        exact_reduction ? CertificationClass::NotCertified
                        : CertificationClass::Unknown;
    claim.tested_block_id = matrix.block.operator_tag;
    claim.description = exact_reduction
        ? "Reduced Schur/constraint summary is available, but stability certification requires inf-sup and Schur conditioning/equivalence evidence"
        : "Reduced Schur/constraint summary is incomplete for indefinite block resolution";
    claim.claim_origin = "MixedOperatorAnalyzer";
    claim.addEvidence("MixedOperatorAnalyzer",
        "ReducedMatrixSummary for Schur-like block has rows=" +
        std::to_string(matrix.rows) +
        ", cols=" + std::to_string(matrix.cols) +
        ", exact_reduction=" +
        std::string(summary.reduction_exact_for_analysis ? "true" : "false"),
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

bool nullspaceHandlingAcceptable(NullspaceHandlingClass handling) noexcept
{
    return handling == NullspaceHandlingClass::NotApplicable ||
           handling == NullspaceHandlingClass::AnchoredByConstraints ||
           handling == NullspaceHandlingClass::ProjectedOut;
}

bool positiveSchurEvidence(PositivityClass positivity) noexcept
{
    return positivity == PositivityClass::Positive ||
           positivity == PositivityClass::Nonnegative;
}

bool schurCertificationComplete(const SchurComplementSummary& summary) noexcept
{
    return summary.schur_available &&
           summary.reduction_exact_for_analysis &&
           summary.primal_block_invertible_evidence_present &&
           summary.inf_sup_evidence_present &&
           summary.nullspace_handling_evidence_present &&
           nullspaceHandlingAcceptable(summary.nullspace_handling) &&
           summary.schur_definiteness_evidence_present &&
           positiveSchurEvidence(summary.schur_positivity) &&
           summary.spectral_equivalence_bounds_present &&
           summary.preconditioner_equivalence_bounds_present;
}

void emitSchurComplementClaim(ProblemAnalysisReport& report,
                              const SchurComplementSummary& summary)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::IndefiniteOperatorResolution;
    claim.domain = summary.block.domain;
    claim.variables = !summary.variables.empty()
        ? summary.variables
        : blockVariables(summary.block);
    claim.tested_block_id = summary.block.operator_tag.empty()
        ? summary.schur_id
        : summary.block.operator_tag;
    claim.estimate_scope = summary.schur_id;
    claim.claim_origin = "MixedOperatorAnalyzer";
    claim.nullspace_handling_class = summary.nullspace_handling;

    const bool certified = schurCertificationComplete(summary);
    if (!summary.schur_available) {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.reduced_definiteness_class = CertificationClass::Unknown;
        claim.description =
            "Schur complement resolution is unknown because no Schur operator is available";
    } else if (certified) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.reduced_definiteness_class = CertificationClass::Certified;
        claim.description =
            "Schur complement resolution is certified by exact reduction, inf-sup, nullspace handling, and spectral/preconditioner equivalence evidence";
    } else {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.reduced_definiteness_class = CertificationClass::NotCertified;
        claim.description =
            "Schur complement is available but lacks complete inf-sup, nullspace, definiteness, or spectral/preconditioner equivalence evidence";
    }

    claim.addEvidence("MixedOperatorAnalyzer",
        "SchurComplementSummary id='" + summary.schur_id +
        "', available=" +
        std::string(summary.schur_available ? "true" : "false") +
        ", exact_reduction=" +
        std::string(summary.reduction_exact_for_analysis ? "true" : "false") +
        ", primal_invertible=" +
        std::string(summary.primal_block_invertible_evidence_present ? "true" : "false") +
        ", inf_sup=" +
        std::string(summary.inf_sup_evidence_present ? "true" : "false") +
        ", spectral_equivalence=" +
        std::string(summary.spectral_equivalence_bounds_present ? "true" : "false") +
        ", preconditioner_equivalence=" +
        std::string(summary.preconditioner_equivalence_bounds_present ? "true" : "false"),
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

void emitSchurSummaryHooks(const ProblemAnalysisContext& context,
                           ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->reduced_matrices) {
        emitSchurSummaryClaim(report, summary);
    }
    for (const auto& summary : summaries->schur_complements) {
        emitSchurComplementClaim(report, summary);
    }
}

} // namespace

std::string MixedOperatorAnalyzer::name() const {
    return "MixedOperatorAnalyzer";
}

void MixedOperatorAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // Dedup for MixedSaddlePoint claims: track emitted (momentum, constraint)
    // field pairs so neither path re-emits a claim the other already produced.
    // Keyed by pair (not just constraint field) so the same constraint field
    // can appear in multiple mixed structures with different momentum partners.
    struct FieldPair {
        FieldId momentum, constraint;
        bool operator==(const FieldPair& o) const {
            return momentum == o.momentum && constraint == o.constraint;
        }
    };
    struct FieldPairHash {
        std::size_t operator()(const FieldPair& p) const {
            return std::hash<FieldId>{}(p.momentum) ^
                   (std::hash<FieldId>{}(p.constraint) * 2654435761u);
        }
    };
    std::unordered_set<FieldPair, FieldPairHash> emitted_mixed_pairs;

    // Nullspace claims are field-intrinsic (constant nullspace) — one per
    // constraint field regardless of how many momentum partners it has.
    std::unordered_set<FieldId> emitted_nullspace_fields;

    // =====================================================================
    // PRIMARY PATH: Connected-component analysis on the variable graph
    //
    // Build an adjacency graph where variables are nodes and each
    // contribution creates edges between its test and trial variables.
    // Connected components naturally separate:
    //   - Distinct formulations that share an operator_tag but use
    //     different fields (Issue 1)
    //   - Independent saddle-point subsystems within one formulation
    //     that don't share variables
    //
    // Within each component, constraint variables are paired only with
    // the momentum variables they actually couple with via
    // ConstraintBlock/OffDiagonalBlock contributions (Issue 2).
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        // --- Phase 1: Build variable adjacency graph ---
        std::unordered_map<VariableKey,
            std::unordered_set<VariableKey, VariableKeyHash>,
            VariableKeyHash> adj;

        for (const auto& contrib : contributions) {
            // Collect all variables from this contribution
            std::vector<VariableKey> vars;
            for (const auto& v : contrib.test_variables) vars.push_back(v);
            for (const auto& v : contrib.trial_variables) vars.push_back(v);

            // Ensure all nodes exist (including isolated ones)
            for (const auto& v : vars) adj[v];

            // Connect all pairs within this contribution
            for (std::size_t i = 0; i < vars.size(); ++i) {
                for (std::size_t j = i + 1; j < vars.size(); ++j) {
                    adj[vars[i]].insert(vars[j]);
                    adj[vars[j]].insert(vars[i]);
                }
            }
        }

        // --- Phase 2: BFS to find connected components ---
        std::unordered_map<VariableKey, int, VariableKeyHash> var_to_comp;
        int num_components = 0;

        for (const auto& [v, _] : adj) {
            if (var_to_comp.count(v)) continue;

            int comp_id = num_components++;
            std::vector<VariableKey> queue;
            queue.push_back(v);
            var_to_comp[v] = comp_id;

            for (std::size_t qi = 0; qi < queue.size(); ++qi) {
                for (const auto& neighbor : adj[queue[qi]]) {
                    if (!var_to_comp.count(neighbor)) {
                        var_to_comp[neighbor] = comp_id;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // --- Phase 3: Assign contributions to components ---
        std::unordered_map<int, std::vector<const ContributionDescriptor*>>
            comp_contribs;
        for (const auto& contrib : contributions) {
            int comp_id = -1;
            for (const auto& v : contrib.test_variables) {
                auto it = var_to_comp.find(v);
                if (it != var_to_comp.end()) { comp_id = it->second; break; }
            }
            if (comp_id < 0) {
                for (const auto& v : contrib.trial_variables) {
                    auto it = var_to_comp.find(v);
                    if (it != var_to_comp.end()) { comp_id = it->second; break; }
                }
            }
            if (comp_id >= 0) {
                comp_contribs[comp_id].push_back(&contrib);
            }
        }

        // --- Phase 4: Analyze each component for saddle-point structure ---
        for (const auto& [comp_id, comp_group] : comp_contribs) {
            struct VarInfo {
                bool has_coercive_diagonal = false;
                bool appears_in_constraint = false;
                bool appears_in_offdiagonal = false;
            };
            std::unordered_map<VariableKey, VarInfo, VariableKeyHash> var_info;

            for (const auto* contrib : comp_group) {
                for (const auto& tv : contrib->test_variables) var_info[tv];
                for (const auto& tv : contrib->trial_variables) var_info[tv];

                if (contrib->role == ContributionRole::DiagonalBlock) {
                    if (hasFlag(contrib->traits,
                                OperatorTraitFlags::HasSecondOrder)) {
                        for (const auto& tv : contrib->test_variables) {
                            var_info[tv].has_coercive_diagonal = true;
                        }
                    }
                } else if (contrib->role == ContributionRole::ConstraintBlock) {
                    for (const auto& tv : contrib->test_variables)
                        var_info[tv].appears_in_constraint = true;
                    for (const auto& tv : contrib->trial_variables)
                        var_info[tv].appears_in_constraint = true;
                } else if (contrib->role == ContributionRole::OffDiagonalBlock) {
                    for (const auto& tv : contrib->test_variables)
                        var_info[tv].appears_in_offdiagonal = true;
                    for (const auto& tv : contrib->trial_variables)
                        var_info[tv].appears_in_offdiagonal = true;
                }
            }

            // Identify momentum and constraint variables
            std::unordered_set<VariableKey, VariableKeyHash> momentum_set;
            std::unordered_set<VariableKey, VariableKeyHash> constraint_set;

            for (const auto& [vk, info] : var_info) {
                if (info.has_coercive_diagonal) {
                    momentum_set.insert(vk);
                }
                if ((info.appears_in_constraint || info.appears_in_offdiagonal)
                    && !info.has_coercive_diagonal) {
                    constraint_set.insert(vk);
                }
            }

            if (momentum_set.empty() || constraint_set.empty()) continue;

            // Build per-constraint coupling map: which momentum vars does
            // each constraint var actually couple with via
            // ConstraintBlock/OffDiagonalBlock contributions?
            std::unordered_map<VariableKey,
                std::unordered_set<VariableKey, VariableKeyHash>,
                VariableKeyHash> constraint_to_momentum;

            for (const auto* contrib : comp_group) {
                if (contrib->role != ContributionRole::ConstraintBlock &&
                    contrib->role != ContributionRole::OffDiagonalBlock) {
                    continue;
                }
                // Check all (test, trial) pairs for momentum↔constraint links
                for (const auto& tv : contrib->test_variables) {
                    for (const auto& trv : contrib->trial_variables) {
                        if (constraint_set.count(tv) && momentum_set.count(trv))
                            constraint_to_momentum[tv].insert(trv);
                        if (constraint_set.count(trv) && momentum_set.count(tv))
                            constraint_to_momentum[trv].insert(tv);
                    }
                }
            }

            // Emit one MixedSaddlePoint claim per (momentum, constraint) pair.
            // Each claim carries exactly two variables so InfSupAnalyzer's
            // pair-based coverage check is precisely scoped.
            for (const auto& cv : constraint_set) {
                FieldId constraint_fid = cv.field_id;

                // Determine this constraint's momentum partners
                auto coup_it = constraint_to_momentum.find(cv);
                const auto& partners =
                    (coup_it != constraint_to_momentum.end() &&
                     !coup_it->second.empty())
                        ? coup_it->second
                        : momentum_set;  // defensive fallback

                for (const auto& mv : partners) {
                    // Pair-level dedup
                    if (mv.kind == VariableKind::FieldComponent &&
                        cv.kind == VariableKind::FieldComponent &&
                        mv.field_id != INVALID_FIELD_ID &&
                        constraint_fid != INVALID_FIELD_ID) {
                        FieldPair fp{mv.field_id, constraint_fid};
                        if (emitted_mixed_pairs.count(fp)) continue;
                        emitted_mixed_pairs.insert(fp);
                    }

                    PropertyClaim claim;
                    claim.kind = PropertyKind::MixedSaddlePoint;
                    claim.status = PropertyStatus::Exact;
                    claim.confidence = AnalysisConfidence::High;
                    claim.domain = DomainKind::Cell;
                    claim.variables.push_back(mv);
                    claim.variables.push_back(cv);

                    claim.description =
                        "Saddle-point structure: variable " +
                        (cv.kind == VariableKind::FieldComponent
                            ? ("field " + std::to_string(constraint_fid))
                            : cv.name) +
                        " (constraint/multiplier, no diagonal elliptic"
                        " block)";
                    claim.addEvidence("MixedOperatorAnalyzer",
                        "Constraint variable has no contribution with "
                        "DiagonalBlock + HasSecondOrder traits");
                    report.claims.push_back(std::move(claim));
                }

                // Nullspace claim: one per constraint field (field-intrinsic)
                if (cv.kind == VariableKind::FieldComponent &&
                    constraint_fid != INVALID_FIELD_ID &&
                    !emitted_nullspace_fields.count(constraint_fid)) {
                    PropertyClaim ns_claim;
                    ns_claim.kind = PropertyKind::Nullspace;
                    ns_claim.status = PropertyStatus::Exact;
                    ns_claim.confidence = AnalysisConfidence::High;
                    ns_claim.field = constraint_fid;
                    ns_claim.component = -1;
                    ns_claim.domain = DomainKind::Cell;
                    ns_claim.variables.push_back(
                        VariableKey::field(constraint_fid));
                    ns_claim.description =
                        "Constraint field " +
                        std::to_string(constraint_fid) +
                        " in saddle-point system has constant nullspace"
                        " (pressure gauge)";
                    ns_claim.nullspace_family =
                        NullspaceFamily::ScalarConstant;
                    ns_claim.claim_origin = "MixedOperatorAnalyzer";
                    ns_claim.addEvidence("MixedOperatorAnalyzer",
                        "Field has no diagonal elliptic block — constant"
                        " shift is in the operator nullspace");
                    report.claims.push_back(std::move(ns_claim));
                    emitted_nullspace_fields.insert(constraint_fid);
                }
            }
        }
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords
    // Emits one MixedSaddlePoint claim per constraint field with a SINGLE
    // variable (the constraint). Without block expressions we can verify
    // that a field is a constraint (no gradient operator) but NOT which
    // momentum field it couples with. Rather than fabricating pair
    // structure that downstream passes (SpaceCompatibilityAnalyzer,
    // InfSupAnalyzer) would consume as verified, we emit only what's
    // known. Downstream passes handle single-variable claims:
    //   - InfSupAnalyzer: emits generic InfSupCondition::Required
    //   - SpaceCompatibilityAnalyzer: skips (no pair to check)
    // =====================================================================
    {
        const auto& records = context.formulationRecords();

        FormStructureAnalyzer fsa;

        for (const auto& rec : records) {
            if (!rec.residual_expr) continue;
            if (rec.active_fields.size() < 2) continue;

            // Collect ALL momentum and constraint fields
            std::vector<FieldId> momentum_fids;
            std::vector<FieldId> constraint_fids;

            for (FieldId fid : rec.active_fields) {
                auto fs = fsa.analyzeField(*rec.residual_expr, fid);
                if (fs.occurrence_count == 0) continue;

                if (fs.value_dimension <= 1 &&
                    fs.has_absolute_value &&
                    !fs.has_gradient && !fs.has_sym_grad &&
                    !fs.has_stabilization) {
                    constraint_fids.push_back(fid);
                }

                if (fs.value_dimension > 1 &&
                    (fs.has_gradient || fs.has_sym_grad)) {
                    momentum_fids.push_back(fid);
                }
            }

            if (momentum_fids.empty() || constraint_fids.empty()) continue;

            // Emit one claim per constraint field (single variable, no
            // fabricated pair). The claim asserts: "this field is a
            // constraint/multiplier in a saddle-point system."
            for (FieldId cfid : constraint_fids) {
                // Skip if primary path already covered this constraint
                if (emitted_nullspace_fields.count(cfid)) continue;

                PropertyClaim claim;
                claim.kind = PropertyKind::MixedSaddlePoint;
                claim.status = PropertyStatus::Exact;
                claim.confidence = AnalysisConfidence::High;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(cfid));
                claim.claim_origin = "MixedOperatorAnalyzer";

                claim.description =
                    "Formulation '" + rec.operator_tag +
                    "' has saddle-point structure: scalar field " +
                    std::to_string(cfid) +
                    " (constraint/multiplier, no diagonal elliptic"
                    " block)";
                claim.addEvidence("MixedOperatorAnalyzer",
                    "Constraint field " + std::to_string(cfid) +
                    " has no own gradient operator (no block structure"
                    " available to verify specific momentum partner)");
                report.claims.push_back(std::move(claim));

                // Nullspace: one per constraint field (field-intrinsic)
                if (!emitted_nullspace_fields.count(cfid)) {
                    PropertyClaim ns_claim;
                    ns_claim.kind = PropertyKind::Nullspace;
                    ns_claim.status = PropertyStatus::Exact;
                    ns_claim.confidence = AnalysisConfidence::High;
                    ns_claim.field = cfid;
                    ns_claim.component = -1;
                    ns_claim.domain = DomainKind::Cell;
                    ns_claim.variables.push_back(
                        VariableKey::field(cfid));
                    ns_claim.description =
                        "Constraint field " + std::to_string(cfid) +
                        " in saddle-point system has constant nullspace"
                        " (pressure gauge)";
                    ns_claim.nullspace_family =
                        NullspaceFamily::ScalarConstant;
                    ns_claim.claim_origin = "MixedOperatorAnalyzer";
                    ns_claim.addEvidence("MixedOperatorAnalyzer",
                        "Field has no diagonal elliptic block — constant"
                        " shift is in the operator nullspace");
                    report.claims.push_back(std::move(ns_claim));
                    emitted_nullspace_fields.insert(cfid);
                }
            }
        }
    }

    emitSchurSummaryHooks(context, report);
}

} // namespace analysis
} // namespace FE
} // namespace svmp
