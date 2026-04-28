/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SolverCompatibilityAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Backends/Utils/BackendOptions.h"

#include <sstream>
#include <string>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

bool isSPDOnlyMethod(backends::SolverMethod method) noexcept
{
    return method == backends::SolverMethod::CG;
}

bool isGeneralIndefiniteMethod(backends::SolverMethod method) noexcept
{
    return method == backends::SolverMethod::Direct ||
           method == backends::SolverMethod::BiCGSTAB ||
           method == backends::SolverMethod::GMRES ||
           method == backends::SolverMethod::PGMRES ||
           method == backends::SolverMethod::FGMRES ||
           method == backends::SolverMethod::BlockSchur;
}

void addSolverIssue(ProblemAnalysisReport& report,
                    IssueSeverity severity,
                    const std::string& message)
{
    AnalysisIssue issue;
    issue.severity = severity;
    issue.message = message;
    report.issues.push_back(std::move(issue));
}

void addSolverClaim(ProblemAnalysisReport& report,
                    const backends::SolverOptions& options,
                    PropertyStatus status,
                    CertificationClass certification,
                    std::string description,
                    std::string evidence,
                    AnalysisConfidence confidence = AnalysisConfidence::High)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::SolverCompatibility;
    claim.status = status;
    claim.confidence = confidence;
    claim.domain = DomainKind::Global;
    claim.certification_class = certification;
    claim.description = std::move(description);
    claim.claim_origin = "SolverCompatibilityAnalyzer";
    claim.addEvidence("SolverCompatibilityAnalyzer",
        "Configured solver method=" +
            std::string(backends::solverMethodToString(options.method)) +
            ", preconditioner=" +
            std::string(backends::preconditionerToString(options.preconditioner)) +
            ": " + std::move(evidence),
        confidence);
    report.claims.push_back(std::move(claim));
}

struct OperatorFacts {
    bool has_mixed_or_indefinite_structure{false};
    bool has_nonsymmetric_or_transport_structure{false};
    bool has_spd_like_support{false};
    std::string reason;
};

void appendReason(OperatorFacts& facts, const std::string& reason)
{
    if (!facts.reason.empty()) {
        facts.reason += "; ";
    }
    facts.reason += reason;
}

OperatorFacts collectFacts(const ProblemAnalysisContext& context,
                           const ProblemAnalysisReport& report)
{
    OperatorFacts facts;

    for (const auto& claim : report.claims) {
        switch (claim.kind) {
            case PropertyKind::MixedSaddlePoint:
            case PropertyKind::InfSupCondition:
            case PropertyKind::IndefiniteOperatorResolution:
                facts.has_mixed_or_indefinite_structure = true;
                appendReason(facts, std::string(toString(claim.kind)) + " claim present");
                break;
            case PropertyKind::OperatorTransportCharacter:
                if (claim.transport_character_class &&
                    *claim.transport_character_class != TransportCharacterClass::DiffusionLike &&
                    *claim.transport_character_class != TransportCharacterClass::None) {
                    facts.has_nonsymmetric_or_transport_structure = true;
                    appendReason(facts, "first-order/transport character claim present");
                }
                break;
            case PropertyKind::OperatorSymmetry:
                if (claim.operator_symmetry_class &&
                    *claim.operator_symmetry_class == OperatorSymmetryClass::Nonsymmetric) {
                    facts.has_nonsymmetric_or_transport_structure = true;
                    appendReason(facts, "nonsymmetric operator claim present");
                }
                break;
            case PropertyKind::OperatorDefiniteness:
                if (claim.coercivity_class &&
                    (*claim.coercivity_class == CoercivityClass::Indefinite ||
                     *claim.coercivity_class == CoercivityClass::NotCoercive)) {
                    facts.has_mixed_or_indefinite_structure = true;
                    appendReason(facts, "indefinite/not-coercive definiteness claim present");
                } else if (claim.coercivity_class &&
                    (*claim.coercivity_class == CoercivityClass::Coercive ||
                     *claim.coercivity_class == CoercivityClass::Semicoercive)) {
                    facts.has_spd_like_support = true;
                    appendReason(facts, "coercive or semicoercive definiteness claim present");
                }
                if (claim.reduced_definiteness_class &&
                    *claim.reduced_definiteness_class == CertificationClass::Certified) {
                    facts.has_spd_like_support = true;
                    appendReason(facts, "reduced definiteness certified");
                }
                break;
            default:
                break;
        }
    }

    if (const auto* summaries = context.analysisSummaries()) {
        for (const auto& matrix : summaries->discrete_matrices) {
            if (!matrix.square) {
                facts.has_mixed_or_indefinite_structure = true;
                appendReason(facts, "nonsquare discrete matrix summary present");
            }
            if (matrix.symmetry_evidence_complete &&
                (!matrix.structurally_symmetric || !matrix.numerically_symmetric)) {
                facts.has_nonsymmetric_or_transport_structure = true;
                appendReason(facts, "matrix symmetry summary reports nonsymmetry");
            }
            if (matrix.square &&
                matrix.symmetry_evidence_complete &&
                matrix.structurally_symmetric &&
                matrix.numerically_symmetric &&
                matrix.nonpositive_diagonal_count == 0u &&
                matrix.negative_diagonal_count == 0u) {
                facts.has_spd_like_support = true;
                appendReason(facts, "matrix summary has symmetric positive-diagonal evidence");
            }
        }
    }

    return facts;
}

} // namespace

std::string SolverCompatibilityAnalyzer::name() const {
    return "SolverCompatibilityAnalyzer";
}

void SolverCompatibilityAnalyzer::run(const ProblemAnalysisContext& context,
                                      ProblemAnalysisReport& report) const
{
    const auto* options = context.solverOptions();
    if (!options) {
        return;
    }

    const auto facts = collectFacts(context, report);
    if (!facts.has_mixed_or_indefinite_structure &&
        !facts.has_nonsymmetric_or_transport_structure &&
        !facts.has_spd_like_support) {
        return;
    }

    const auto method_name = std::string(backends::solverMethodToString(options->method));

    if (isSPDOnlyMethod(options->method) &&
        facts.has_mixed_or_indefinite_structure) {
        addSolverClaim(report, *options,
            PropertyStatus::Violated,
            CertificationClass::Violated,
            "Solver method '" + method_name +
                "' is incompatible with mixed, indefinite, or inf-sup operator structure",
            facts.reason);
        addSolverIssue(report, IssueSeverity::Error,
            "CG requires a symmetric positive definite operator, but analysis found " +
            facts.reason);
        return;
    }

    if (isSPDOnlyMethod(options->method) &&
        facts.has_nonsymmetric_or_transport_structure) {
        addSolverClaim(report, *options,
            PropertyStatus::Violated,
            CertificationClass::Violated,
            "Solver method '" + method_name +
                "' is incompatible with nonsymmetric or transport-dominated operator structure",
            facts.reason);
        addSolverIssue(report, IssueSeverity::Error,
            "CG requires symmetry, but analysis found " + facts.reason);
        return;
    }

    if (isSPDOnlyMethod(options->method) && facts.has_spd_like_support) {
        addSolverClaim(report, *options,
            PropertyStatus::Preserved,
            CertificationClass::Certified,
            "Solver method '" + method_name +
                "' is compatible with current SPD-like structural evidence",
            facts.reason);
        return;
    }

    if (isGeneralIndefiniteMethod(options->method) &&
        (facts.has_mixed_or_indefinite_structure ||
         facts.has_nonsymmetric_or_transport_structure)) {
        addSolverClaim(report, *options,
            PropertyStatus::Preserved,
            CertificationClass::Certified,
            "Solver method '" + method_name +
                "' is compatible with nonsymmetric, transport, or indefinite structural evidence",
            facts.reason);
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
