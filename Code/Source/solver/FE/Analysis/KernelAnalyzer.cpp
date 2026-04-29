/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/KernelAnalyzer.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

std::string KernelAnalyzer::name() const {
    return "KernelAnalyzer";
}

void KernelAnalyzer::run(const ProblemAnalysisContext& context,
                         ProblemAnalysisReport& report) const
{
    // =====================================================================
    // PRIMARY PATH: Consume ContributionDescriptors
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        for (const auto& contrib : contributions) {
            // Process nullspace hints from each contribution
            for (const auto& hint : contrib.nullspace_hints) {
                FieldId fid = hint.field;
                if (fid == INVALID_FIELD_ID) continue;

                auto make_description = [&]() -> std::string {
                    if (!hint.reason.empty()) return hint.reason;

                    switch (hint.family) {
                        case NullspaceFamily::KernelOfSymGrad:
                            return "Vector field appears only through sym(grad(u)) — "
                                   "rigid-body modes (translations + rotations) are in "
                                   "the operator nullspace";
                        case NullspaceFamily::ComponentwiseConstant:
                            return "Vector field appears only through gradient-like operators — "
                                   "per-component constant shifts are in the operator nullspace";
                        case NullspaceFamily::ScalarConstant:
                            return "Field appears only through gradient-like operators — "
                                   "constant shift is in the operator nullspace";
                        case NullspaceFamily::UserDefined:
                            return "User-defined nullspace mode";
                    }
                    return "Nullspace mode detected";
                };

                // For ComponentwiseConstant hints with a specific component,
                // emit per-component claims
                if (hint.family == NullspaceFamily::ComponentwiseConstant &&
                    hint.component >= 0) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    claim.status = (hint.confidence == AnalysisConfidence::High)
                        ? PropertyStatus::Exact : PropertyStatus::Likely;
                    claim.confidence = hint.confidence;
                    claim.field = fid;
                    claim.component = hint.component;
                    claim.domain = contrib.domain;
                    claim.variables.push_back(VariableKey::field(fid, hint.component));
                    claim.description =
                        "Vector field component " + std::to_string(hint.component) +
                        " appears only through gradient-like operators — "
                        "constant shift is in the operator nullspace";
                    if (!hint.reason.empty() &&
                        hint.reason.find("stabilization") != std::string::npos) {
                        claim.description += " (stabilization terms weakly break the nullspace)";
                    }
                    claim.nullspace_family = hint.family;
                    claim.claim_origin = "KernelAnalyzer";
                    claim.addEvidence("KernelAnalyzer",
                        "ContributionDescriptor nullspace hint, component=" +
                        std::to_string(hint.component),
                        hint.confidence);
                    report.claims.push_back(std::move(claim));
                    continue;
                }

                // For KernelOfSymGrad, emit a single field-wide claim
                if (hint.family == NullspaceFamily::KernelOfSymGrad) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    claim.status = (hint.confidence == AnalysisConfidence::High)
                        ? PropertyStatus::Exact : PropertyStatus::Likely;
                    claim.confidence = hint.confidence;
                    claim.field = fid;
                    claim.component = -1;
                    claim.domain = contrib.domain;
                    claim.variables.push_back(VariableKey::field(fid));
                    claim.description = make_description();
                    claim.nullspace_family = NullspaceFamily::KernelOfSymGrad;
                    claim.claim_origin = "KernelAnalyzer";
                    claim.addEvidence("KernelAnalyzer",
                        "ContributionDescriptor nullspace hint, family=KernelOfSymGrad",
                        hint.confidence);
                    report.claims.push_back(std::move(claim));
                    continue;
                }

                // ScalarConstant or field-wide ComponentwiseConstant or UserDefined
                PropertyClaim claim;
                claim.kind = PropertyKind::Nullspace;
                claim.status = (hint.confidence == AnalysisConfidence::High)
                    ? PropertyStatus::Exact : PropertyStatus::Likely;
                claim.confidence = hint.confidence;
                claim.field = fid;
                claim.component = hint.component;
                claim.domain = contrib.domain;
                claim.variables.push_back(
                    hint.component >= 0
                        ? VariableKey::field(fid, hint.component)
                        : VariableKey::field(fid));
                claim.description = make_description();
                claim.nullspace_family = hint.family;
                claim.claim_origin = "KernelAnalyzer";
                claim.addEvidence("KernelAnalyzer",
                    "ContributionDescriptor nullspace hint, family=" +
                    std::string(toString(hint.family)),
                    hint.confidence);
                report.claims.push_back(std::move(claim));
            }
        }

        return;
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords (when no contributions available)
    // =====================================================================
    const auto& records = context.formulationRecords();

    FormStructureAnalyzer fsa;

    for (const auto& rec : records) {
        if (!rec.residual_expr) continue;
        if (rec.active_fields.empty()) continue;

        for (FieldId fid : rec.active_fields) {
            auto fs = fsa.analyzeField(*rec.residual_expr, fid);
            if (fs.occurrence_count == 0) continue;

            const bool has_stab = fs.has_stabilization;
            const bool has_dt = fs.has_time_derivative;

            // Skip if field has absolute-value terms — mode is anchored
            // by the operator itself (mass term, Robin term, etc.)
            if (fs.has_absolute_value) continue;

            if (!fs.only_through_annihilating_ops) continue;

            // Time derivative mass matrix anchors the constant mode
            if (has_dt) continue;

            auto make_confidence = [&]() -> AnalysisConfidence {
                return has_stab ? AnalysisConfidence::Medium : AnalysisConfidence::High;
            };
            auto status = [&]() -> PropertyStatus {
                return has_stab ? PropertyStatus::Likely : PropertyStatus::Exact;
            };
            auto stab_suffix = [&]() -> std::string {
                return has_stab ? " (stabilization terms weakly break the nullspace)" : "";
            };

            // Rigid-body modes: sym(grad(u)) only, vector field
            if (fs.only_through_sym_grad && !fs.has_plain_grad &&
                fs.value_dimension > 1) {
                PropertyClaim claim;
                claim.kind = PropertyKind::Nullspace;
                claim.status = status();
                claim.confidence = make_confidence();
                claim.field = fid;
                claim.component = -1;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(fid));
                claim.description =
                    "Vector field appears only through sym(grad(u)) — "
                    "rigid-body modes (translations + rotations) are in "
                    "the operator nullspace" + stab_suffix();
                claim.nullspace_family = NullspaceFamily::KernelOfSymGrad;
                claim.addEvidence("KernelAnalyzer",
                    "only_through_sym_grad=true, value_dimension=" +
                    std::to_string(fs.value_dimension),
                    make_confidence());
                report.claims.push_back(std::move(claim));
                continue;
            }

            // Componentwise vector constant — emit PER-COMPONENT claims
            // only for component-extractable fields (ProductSpace / H1).
            // Vector-basis fields (HDiv/HCurl) get a single field-wide claim.
            bool per_component = (fs.value_dimension > 1);
            if (per_component) {
                const auto* fd = context.fieldDescriptor(fid);
                if (fd && !fd->component_extractable) {
                    per_component = false;
                }
            }
            if (per_component) {
                for (int comp = 0; comp < fs.value_dimension; ++comp) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    claim.status = status();
                    claim.confidence = make_confidence();
                    claim.field = fid;
                    claim.component = comp;
                    claim.domain = DomainKind::Cell;
                    claim.variables.push_back(VariableKey::field(fid, comp));
                    claim.description =
                        "Vector field component " + std::to_string(comp) +
                        " appears only through gradient-like operators — "
                        "constant shift is in the operator nullspace" + stab_suffix();
                    claim.addEvidence("KernelAnalyzer",
                        "only_through_annihilating_ops=true, component=" +
                        std::to_string(comp),
                        make_confidence());
                    report.claims.push_back(std::move(claim));
                }
            } else {
                // Scalar constant OR non-extractable vector field (field-wide claim)
                PropertyClaim claim;
                claim.kind = PropertyKind::Nullspace;
                claim.status = status();
                claim.confidence = make_confidence();
                claim.field = fid;
                claim.component = -1;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(fid));
                if (fs.value_dimension > 1) {
                    claim.description =
                        "Vector field appears only through gradient-like operators — "
                        "per-component constant shifts are in the operator nullspace" + stab_suffix();
                } else {
                    claim.description =
                        "Field appears only through gradient-like operators — "
                        "constant shift is in the operator nullspace" + stab_suffix();
                }
                claim.addEvidence("KernelAnalyzer",
                    "only_through_annihilating_ops=true, value_dimension=" +
                    std::to_string(fs.value_dimension),
                    make_confidence());
                report.claims.push_back(std::move(claim));
            }
        }
    }

}

} // namespace analysis
} // namespace FE
} // namespace svmp
