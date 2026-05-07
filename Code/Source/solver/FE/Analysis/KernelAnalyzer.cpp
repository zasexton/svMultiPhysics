/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/KernelAnalyzer.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_set>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct FieldComponentKey {
    FieldId field{INVALID_FIELD_ID};
    int component{-1};

    bool operator==(const FieldComponentKey& other) const noexcept {
        return field == other.field && component == other.component;
    }
};

struct FieldComponentKeyHash {
    std::size_t operator()(const FieldComponentKey& key) const noexcept {
        std::size_t seed = std::hash<FieldId>{}(key.field);
        seed ^= std::hash<int>{}(key.component) + 0x9e3779b97f4a7c15ULL +
                (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

using KernelCoverage =
    std::unordered_set<FieldComponentKey, FieldComponentKeyHash>;

PropertyStatus statusForNullspaceEvidence(NullspaceEvidenceKind kind,
                                          AnalysisConfidence confidence) noexcept
{
    switch (kind) {
        case NullspaceEvidenceKind::MatrixRankEvidence:
        case NullspaceEvidenceKind::TheoremBackedKernel:
        case NullspaceEvidenceKind::ProducerCertifiedKernel:
            return confidence == AnalysisConfidence::High
                ? PropertyStatus::Exact
                : PropertyStatus::Likely;
        case NullspaceEvidenceKind::SymbolicOperatorIdentity:
        case NullspaceEvidenceKind::DescriptorHint:
            return PropertyStatus::Likely;
    }
    return PropertyStatus::Likely;
}

EvidenceLevel evidenceLevelForNullspaceEvidence(
    NullspaceEvidenceKind kind) noexcept
{
    switch (kind) {
        case NullspaceEvidenceKind::DescriptorHint:
            return EvidenceLevel::DescriptorHint;
        case NullspaceEvidenceKind::SymbolicOperatorIdentity:
            return EvidenceLevel::StructuralMetadata;
        case NullspaceEvidenceKind::MatrixRankEvidence:
            return EvidenceLevel::ScopedNumericSummary;
        case NullspaceEvidenceKind::TheoremBackedKernel:
            return EvidenceLevel::TheoremRegistryMatch;
        case NullspaceEvidenceKind::ProducerCertifiedKernel:
            return EvidenceLevel::CertifiedNumericTheorem;
    }
    return EvidenceLevel::DescriptorHint;
}

CertificationClass certificationForNullspaceEvidence(
    NullspaceEvidenceKind kind,
    PropertyStatus status) noexcept
{
    if (status != PropertyStatus::Exact) {
        return CertificationClass::NotCertified;
    }
    switch (kind) {
        case NullspaceEvidenceKind::MatrixRankEvidence:
        case NullspaceEvidenceKind::TheoremBackedKernel:
        case NullspaceEvidenceKind::ProducerCertifiedKernel:
            return CertificationClass::Certified;
        case NullspaceEvidenceKind::SymbolicOperatorIdentity:
        case NullspaceEvidenceKind::DescriptorHint:
            return CertificationClass::NotCertified;
    }
    return CertificationClass::NotCertified;
}

void applyNullspaceEvidence(PropertyClaim& claim,
                            NullspaceEvidenceKind kind,
                            AnalysisConfidence confidence)
{
    claim.status = statusForNullspaceEvidence(kind, confidence);
    claim.confidence = confidence;
    claim.evidence_level = evidenceLevelForNullspaceEvidence(kind);
    claim.nullspace_evidence_kind = kind;
    claim.certification_class =
        certificationForNullspaceEvidence(kind, claim.status);
}

bool coverageContains(const KernelCoverage& coverage,
                      FieldId field,
                      int component) noexcept
{
    return coverage.count({field, component}) > 0u ||
           coverage.count({field, -1}) > 0u;
}

void markCovered(KernelCoverage& coverage, FieldId field, int component)
{
    if (field != INVALID_FIELD_ID) {
        coverage.insert({field, component});
    }
}

} // namespace

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
    KernelCoverage covered;
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
                        case NullspaceFamily::RigidBody:
                            return "Vector field appears only through sym(grad(u)) — "
                                   "rigid-body modes (translations + rotations) are in "
                                   "the operator nullspace";
                        case NullspaceFamily::RigidTranslation:
                            return "Translational nullspace family metadata is present";
                        case NullspaceFamily::RigidRotation:
                            return "Rotational nullspace family metadata is present";
                        case NullspaceFamily::ComponentwiseConstant:
                        case NullspaceFamily::VectorConstant:
                            return "Vector field appears only through gradient-like operators — "
                                   "per-component constant shifts are in the operator nullspace";
                        case NullspaceFamily::ScalarConstant:
                        case NullspaceFamily::GaugeConstant:
                            return "Field appears only through gradient-like operators — "
                                   "constant shift is in the operator nullspace";
                        case NullspaceFamily::HarmonicField:
                        case NullspaceFamily::GradientKernel:
                        case NullspaceFamily::CurlKernel:
                        case NullspaceFamily::DivergenceFreeKernel:
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
                    applyNullspaceEvidence(claim,
                                           hint.evidence_kind,
                                           hint.confidence);
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
                    markCovered(covered, fid, hint.component);
                    continue;
                }

                // For KernelOfSymGrad, emit a single field-wide claim
                if (hint.family == NullspaceFamily::KernelOfSymGrad) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    applyNullspaceEvidence(claim,
                                           hint.evidence_kind,
                                           hint.confidence);
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
                    markCovered(covered, fid, -1);
                    continue;
                }

                // ScalarConstant or field-wide ComponentwiseConstant or UserDefined
                PropertyClaim claim;
                claim.kind = PropertyKind::Nullspace;
                applyNullspaceEvidence(claim,
                                       hint.evidence_kind,
                                       hint.confidence);
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
                markCovered(covered, fid, hint.component);
            }
        }
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
            if (coverageContains(covered, fid, -1)) continue;

            const bool has_stab = fs.has_stabilization;
            const bool has_dt = fs.has_time_derivative;

            if (fs.has_absolute_value || fs.has_time_derivative) {
                if (fs.only_through_annihilating_ops) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    claim.status = PropertyStatus::Unknown;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.evidence_level = EvidenceLevel::SyntaxPattern;
                    claim.nullspace_evidence_kind =
                        NullspaceEvidenceKind::DescriptorHint;
                    claim.certification_class =
                        CertificationClass::NotCertified;
                    claim.field = fid;
                    claim.domain = DomainKind::Cell;
                    claim.variables.push_back(VariableKey::field(fid));
                    claim.description =
                        "Field has constant-annihilating spatial syntax plus value or time-derivative terms; kernel anchoring requires scoped mass/reaction/coercivity evidence";
                    claim.addEvidence("KernelAnalyzer",
                        "has_absolute_value=" +
                        std::string(fs.has_absolute_value ? "true" : "false") +
                        ", has_time_derivative=" +
                        std::string(fs.has_time_derivative ? "true" : "false"),
                        AnalysisConfidence::Medium);
                    report.claims.push_back(std::move(claim));
                }
                continue;
            }

            if (!fs.only_through_annihilating_ops) continue;

            auto make_confidence = [&]() -> AnalysisConfidence {
                return has_stab ? AnalysisConfidence::Medium : AnalysisConfidence::High;
            };
            auto status = [&]() -> PropertyStatus {
                (void)has_dt;
                return PropertyStatus::Likely;
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
                claim.evidence_level = EvidenceLevel::SyntaxPattern;
                claim.nullspace_evidence_kind =
                    NullspaceEvidenceKind::DescriptorHint;
                claim.certification_class = CertificationClass::NotCertified;
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
                markCovered(covered, fid, -1);
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
                    if (coverageContains(covered, fid, comp)) {
                        continue;
                    }
                    PropertyClaim claim;
                    claim.kind = PropertyKind::Nullspace;
                    claim.status = status();
                    claim.confidence = make_confidence();
                    claim.evidence_level = EvidenceLevel::SyntaxPattern;
                    claim.nullspace_evidence_kind =
                        NullspaceEvidenceKind::DescriptorHint;
                    claim.certification_class =
                        CertificationClass::NotCertified;
                    claim.field = fid;
                    claim.component = comp;
                    claim.domain = DomainKind::Cell;
                    claim.variables.push_back(VariableKey::field(fid, comp));
                    claim.description =
                        "Vector field component " + std::to_string(comp) +
                        " appears only through gradient-like operators — "
                        "constant shift is in the operator nullspace" + stab_suffix();
                    claim.nullspace_family =
                        NullspaceFamily::ComponentwiseConstant;
                    claim.addEvidence("KernelAnalyzer",
                        "only_through_annihilating_ops=true, component=" +
                        std::to_string(comp),
                        make_confidence());
                    report.claims.push_back(std::move(claim));
                    markCovered(covered, fid, comp);
                }
            } else {
                // Scalar constant OR non-extractable vector field (field-wide claim)
                if (coverageContains(covered, fid, -1)) {
                    continue;
                }
                PropertyClaim claim;
                claim.kind = PropertyKind::Nullspace;
                claim.status = status();
                claim.confidence = make_confidence();
                claim.evidence_level = EvidenceLevel::SyntaxPattern;
                claim.nullspace_evidence_kind =
                    NullspaceEvidenceKind::DescriptorHint;
                claim.certification_class = CertificationClass::NotCertified;
                claim.field = fid;
                claim.component = -1;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(fid));
                if (fs.value_dimension > 1) {
                    claim.description =
                        "Vector field appears only through gradient-like operators — "
                        "per-component constant shifts are in the operator nullspace" + stab_suffix();
                    claim.nullspace_family = NullspaceFamily::VectorConstant;
                } else {
                    claim.description =
                        "Field appears only through gradient-like operators — "
                        "constant shift is in the operator nullspace" + stab_suffix();
                    claim.nullspace_family = NullspaceFamily::ScalarConstant;
                }
                claim.addEvidence("KernelAnalyzer",
                    "only_through_annihilating_ops=true, value_dimension=" +
                    std::to_string(fs.value_dimension),
                    make_confidence());
                report.claims.push_back(std::move(claim));
                markCovered(covered, fid, -1);
            }
        }
    }

}

} // namespace analysis
} // namespace FE
} // namespace svmp
