/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/SolverCompatibilityAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Backends/Utils/BackendOptions.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

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
    bool has_spd_certification{false};
    bool has_incomplete_spd_evidence{false};
    bool has_positive_diagonal_certification{false};
    bool has_uncertain_solver_route_evidence{false};
    bool has_indefinite_resolution_certification{false};
    std::string reason;
};

struct SolverRouteEvidence {
    bool operator_spd_certified{false};
    bool preconditioner_symmetric_certified{false};
    bool preconditioner_positive_certified{false};
    bool flexible_krylov_required{false};
    bool indefinite_route_certified{false};
    bool left_right_preconditioning_metadata_present{false};
    bool preconditioner_known_incompatible{false};
    std::string reason;
};

void appendReason(OperatorFacts& facts, const std::string& reason)
{
    if (!facts.reason.empty()) {
        facts.reason += "; ";
    }
    facts.reason += reason;
}

bool nullspaceHandlingAcceptable(
    const std::optional<NullspaceHandlingClass>& handling) noexcept
{
    if (!handling) {
        return false;
    }
    return *handling == NullspaceHandlingClass::NotApplicable ||
           *handling == NullspaceHandlingClass::AnchoredByConstraints ||
           *handling == NullspaceHandlingClass::ProjectedOut;
}

bool evidenceAtLeast(EvidenceLevel actual, EvidenceLevel required) noexcept
{
    return static_cast<std::uint8_t>(actual) >=
           static_cast<std::uint8_t>(required);
}

bool claimHasHardCertification(const PropertyClaim& claim) noexcept
{
    return claim.certification_class &&
           (*claim.certification_class == CertificationClass::Certified ||
            *claim.certification_class == CertificationClass::Violated);
}

bool claimIsHardEvidence(const PropertyClaim& claim) noexcept
{
    const bool high_confidence_scoped =
        claim.confidence == AnalysisConfidence::High &&
        (claim.status == PropertyStatus::Preserved ||
         claim.status == PropertyStatus::Exact ||
         claim.status == PropertyStatus::Violated) &&
        evidenceAtLeast(claim.evidence_level,
                        EvidenceLevel::ScopedNumericSummary);

    return claimHasHardCertification(claim) || high_confidence_scoped;
}

void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<VariableKey> blockVariables(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& variable : block.test_variables) {
        appendUnique(variables, variable);
    }
    for (const auto& variable : block.trial_variables) {
        appendUnique(variables, variable);
    }
    return variables;
}

std::vector<VariableKey> activeSystemVariables(
    const ProblemAnalysisContext& context)
{
    std::vector<VariableKey> variables;
    for (const auto& contribution : context.contributions()) {
        for (const auto& variable : contribution.test_variables) {
            appendUnique(variables, variable);
        }
        for (const auto& variable : contribution.trial_variables) {
            appendUnique(variables, variable);
        }
        for (const auto& variable : contribution.related_variables) {
            appendUnique(variables, variable);
        }
    }
    for (const auto& field : context.fieldDescriptors()) {
        appendUnique(variables, VariableKey::field(field.field_id));
    }
    return variables;
}

bool containsVariable(const std::vector<VariableKey>& values,
                      const VariableKey& variable)
{
    return std::find(values.begin(), values.end(), variable) != values.end();
}

bool variablesCoverActiveSystem(const std::vector<VariableKey>& variables,
                                const std::vector<VariableKey>& active)
{
    if (active.empty()) {
        return true;
    }
    if (variables.empty()) {
        return false;
    }
    return std::all_of(active.begin(), active.end(),
                       [&](const VariableKey& variable) {
                           return containsVariable(variables, variable);
                       });
}

bool claimCoversActiveSystem(const PropertyClaim& claim,
                             const std::vector<VariableKey>& active)
{
    return variablesCoverActiveSystem(claim.variables, active);
}

bool matrixCoversActiveSystem(const DiscreteMatrixSummary& matrix,
                              const std::vector<VariableKey>& active)
{
    if ((matrix.scope == NumericSummaryScope::FullMatrix ||
         matrix.scope == NumericSummaryScope::ConstrainedFullMatrix) &&
        blockVariables(matrix.block).empty()) {
        return true;
    }
    return variablesCoverActiveSystem(blockVariables(matrix.block), active);
}

Real definitenessTolerance(const DiscreteMatrixSummary& matrix) noexcept
{
    return std::max({matrix.sign_tolerance,
                     matrix.symmetry_tolerance,
                     Real{1.0e-12}});
}

bool hasPositiveDefinitenessEvidence(const DiscreteMatrixSummary& matrix)
{
    const Real tol = definitenessTolerance(matrix);
    if (matrix.coercivity_lower_bound &&
        numeric::finite(*matrix.coercivity_lower_bound) &&
        *matrix.coercivity_lower_bound > tol) {
        return true;
    }
    if (matrix.min_eigenvalue_estimate &&
        numeric::finite(*matrix.min_eigenvalue_estimate) &&
        *matrix.min_eigenvalue_estimate > tol) {
        return true;
    }
    return matrix.cholesky_factorization_succeeded;
}

bool hasSymmetricPositiveDiagonalEvidence(const DiscreteMatrixSummary& matrix)
{
    return matrix.square &&
           matrix.symmetry_evidence_complete &&
           matrix.structurally_symmetric &&
           matrix.numerically_symmetric &&
           matrix.global_row_coverage_exact &&
           matrix.diagonal_count == matrix.expected_row_count &&
           matrix.missing_diagonal_count == 0u &&
           matrix.nonpositive_diagonal_count == 0u &&
           matrix.negative_diagonal_count == 0u;
}

SolverRouteEvidence collectPreconditionerRouteEvidence(
    const backends::SolverOptions& options,
    const OperatorFacts& facts)
{
    SolverRouteEvidence route;
    route.operator_spd_certified = facts.has_spd_certification;
    route.indefinite_route_certified =
        facts.has_indefinite_resolution_certification;

    switch (options.preconditioner) {
        case backends::PreconditionerType::None:
            route.preconditioner_symmetric_certified = true;
            route.preconditioner_positive_certified = true;
            route.left_right_preconditioning_metadata_present = true;
            route.reason = "identity preconditioner is SPD for a certified SPD operator";
            break;
        case backends::PreconditionerType::Diagonal:
            route.left_right_preconditioning_metadata_present = true;
            if (facts.has_positive_diagonal_certification) {
                route.preconditioner_symmetric_certified = true;
                route.preconditioner_positive_certified = true;
                route.reason =
                    "diagonal preconditioner has scoped positive diagonal evidence";
            } else {
                route.reason =
                    "diagonal preconditioner requires complete positive-diagonal evidence";
            }
            break;
        case backends::PreconditionerType::ILU:
            route.preconditioner_known_incompatible = true;
            route.reason =
                "ILU route is not declared symmetric positive definite for CG";
            break;
        case backends::PreconditionerType::AMG:
        case backends::PreconditionerType::RowColumnScaling:
        case backends::PreconditionerType::FieldSplit:
            route.reason =
                "preconditioner route lacks explicit symmetry, positivity, and left/right metadata for CG";
            break;
    }

    return route;
}

OperatorFacts collectFacts(const ProblemAnalysisContext& context,
                           const ProblemAnalysisReport& report)
{
    OperatorFacts facts;
    const auto active_variables = activeSystemVariables(context);

    for (const auto& claim : report.claims) {
        switch (claim.kind) {
            case PropertyKind::MixedSaddlePoint:
                if (claimIsHardEvidence(claim) &&
                    claimCoversActiveSystem(claim, active_variables)) {
                    facts.has_mixed_or_indefinite_structure = true;
                    appendReason(facts, std::string(toString(claim.kind)) +
                        " hard scoped claim present");
                } else {
                    facts.has_uncertain_solver_route_evidence = true;
                    appendReason(facts,
                        "mixed/saddle-point claim is structural or not scoped to the active system");
                }
                break;
            case PropertyKind::InfSupCondition:
                if (claim.applicability_class &&
                    *claim.applicability_class ==
                        ApplicabilityClass::NotApplicable) {
                    appendReason(facts,
                        "inf-sup claim marked degenerate/out-of-scope");
                    break;
                }
                if (claimIsHardEvidence(claim) &&
                    claimCoversActiveSystem(claim, active_variables)) {
                    facts.has_mixed_or_indefinite_structure = true;
                    appendReason(facts,
                        "InfSupCondition hard scoped claim present");
                } else {
                    facts.has_uncertain_solver_route_evidence = true;
                    appendReason(facts,
                        "inf-sup claim is not certification-grade solver-route evidence");
                }
                break;
            case PropertyKind::IndefiniteOperatorResolution:
                if (claimIsHardEvidence(claim) &&
                    claimCoversActiveSystem(claim, active_variables)) {
                    facts.has_mixed_or_indefinite_structure = true;
                    appendReason(facts,
                        "indefinite-resolution hard scoped claim present");
                } else {
                    facts.has_uncertain_solver_route_evidence = true;
                    appendReason(facts,
                        "indefinite-resolution claim is not scoped hard evidence");
                }
                if (claim.status == PropertyStatus::Preserved &&
                    claim.reduced_definiteness_class &&
                    *claim.reduced_definiteness_class ==
                        CertificationClass::Certified) {
                    facts.has_indefinite_resolution_certification = true;
                    appendReason(facts,
                        "Schur/preconditioner resolution certified");
                }
                break;
            case PropertyKind::OperatorTransportCharacter:
                if (claim.transport_character_class &&
                    *claim.transport_character_class != TransportCharacterClass::DiffusionLike &&
                    *claim.transport_character_class != TransportCharacterClass::None) {
                    if (claimIsHardEvidence(claim) &&
                        claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_nonsymmetric_or_transport_structure = true;
                        appendReason(facts,
                            "first-order/transport hard scoped claim present");
                    } else {
                        facts.has_uncertain_solver_route_evidence = true;
                        appendReason(facts,
                            "transport claim is diagnostic or lacks scoped numeric/theorem evidence");
                    }
                }
                break;
            case PropertyKind::OperatorSymmetry:
                if (claim.operator_symmetry_class &&
                    *claim.operator_symmetry_class == OperatorSymmetryClass::Nonsymmetric) {
                    if (claimIsHardEvidence(claim) &&
                        claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_nonsymmetric_or_transport_structure = true;
                        appendReason(facts,
                            "nonsymmetric operator hard scoped claim present");
                    } else {
                        facts.has_uncertain_solver_route_evidence = true;
                        appendReason(facts,
                            "nonsymmetry claim is not certification-grade solver-route evidence");
                    }
                }
                break;
            case PropertyKind::OperatorDefiniteness:
                if (claim.coercivity_class &&
                    (*claim.coercivity_class == CoercivityClass::Indefinite ||
                     *claim.coercivity_class == CoercivityClass::NotCoercive)) {
                    if (claimIsHardEvidence(claim) &&
                        claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_mixed_or_indefinite_structure = true;
                        appendReason(facts,
                            "indefinite/not-coercive hard scoped claim present");
                    } else {
                        facts.has_uncertain_solver_route_evidence = true;
                        appendReason(facts,
                            "indefinite/not-coercive claim lacks hard scoped evidence");
                    }
                } else if (claim.coercivity_class &&
                    *claim.coercivity_class == CoercivityClass::Coercive &&
                    claim.reduced_definiteness_class &&
                    *claim.reduced_definiteness_class == CertificationClass::Certified) {
                    if (claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_spd_certification = true;
                        appendReason(facts, "coercive reduced definiteness certified");
                    } else {
                        facts.has_incomplete_spd_evidence = true;
                        appendReason(facts, "local coercive reduced definiteness does not cover the active system");
                    }
                } else if (claim.coercivity_class &&
                    *claim.coercivity_class == CoercivityClass::Semicoercive &&
                    claim.reduced_definiteness_class &&
                    *claim.reduced_definiteness_class == CertificationClass::Certified &&
                    nullspaceHandlingAcceptable(claim.nullspace_handling_class)) {
                    if (claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_spd_certification = true;
                        appendReason(facts, "semicoercive reduced definiteness has nullspace handling");
                    } else {
                        facts.has_incomplete_spd_evidence = true;
                        appendReason(facts, "local semicoercive definiteness evidence does not cover the active system");
                    }
                } else if (claim.coercivity_class &&
                    (*claim.coercivity_class == CoercivityClass::Coercive ||
                     *claim.coercivity_class == CoercivityClass::Semicoercive)) {
                    facts.has_incomplete_spd_evidence = true;
                    appendReason(facts, "definiteness claim lacks certified SPD/nullspace evidence");
                }
                if (claim.reduced_definiteness_class &&
                    *claim.reduced_definiteness_class == CertificationClass::Certified &&
                    claim.coercivity_class &&
                    *claim.coercivity_class == CoercivityClass::Coercive) {
                    if (claimCoversActiveSystem(claim, active_variables)) {
                        facts.has_spd_certification = true;
                        appendReason(facts, "reduced SPD definiteness certified");
                    } else {
                        facts.has_incomplete_spd_evidence = true;
                        appendReason(facts, "local reduced SPD evidence does not cover the active system");
                    }
                }
                break;
            default:
                break;
        }
    }

    if (const auto* summaries = context.analysisSummaries()) {
        for (const auto& matrix : summaries->discrete_matrices) {
            if (!matrix.square) {
                if (matrixCoversActiveSystem(matrix, active_variables)) {
                    facts.has_mixed_or_indefinite_structure = true;
                    appendReason(facts,
                        "nonsquare discrete matrix summary covers active system");
                } else {
                    facts.has_uncertain_solver_route_evidence = true;
                    appendReason(facts,
                        "local nonsquare matrix summary does not cover active system");
                }
            }
            if (matrix.symmetry_evidence_complete &&
                (!matrix.structurally_symmetric || !matrix.numerically_symmetric)) {
                if (matrixCoversActiveSystem(matrix, active_variables)) {
                    facts.has_nonsymmetric_or_transport_structure = true;
                    appendReason(facts,
                        "matrix symmetry summary reports scoped nonsymmetry");
                } else {
                    facts.has_uncertain_solver_route_evidence = true;
                    appendReason(facts,
                        "local nonsymmetry matrix summary does not cover active system");
                }
            }
            if (hasSymmetricPositiveDiagonalEvidence(matrix)) {
                if (matrixCoversActiveSystem(matrix, active_variables)) {
                    facts.has_positive_diagonal_certification = true;
                }
                if (hasPositiveDefinitenessEvidence(matrix)) {
                    if (matrixCoversActiveSystem(matrix, active_variables)) {
                        facts.has_spd_certification = true;
                        appendReason(facts, "matrix summary has symmetric positive-definite evidence");
                    } else {
                        facts.has_incomplete_spd_evidence = true;
                        appendReason(facts, "local SPD matrix summary does not cover the active system");
                    }
                } else {
                    facts.has_incomplete_spd_evidence = true;
                    appendReason(facts, "matrix summary has only symmetric positive-diagonal evidence");
                }
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
        !facts.has_spd_certification &&
        !facts.has_incomplete_spd_evidence &&
        !facts.has_uncertain_solver_route_evidence) {
        addSolverClaim(report, *options,
            PropertyStatus::Unknown,
            CertificationClass::NotCertified,
            "Solver compatibility cannot be certified because active operator evidence is incomplete",
            "Solver options are present, but no scoped SPD, symmetry, mixed/indefinite, or nonsymmetric operator evidence covers the active system",
            AnalysisConfidence::Medium);
        return;
    }

    const auto method_name = std::string(backends::solverMethodToString(options->method));
    const auto route = collectPreconditionerRouteEvidence(*options, facts);

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

    if (isSPDOnlyMethod(options->method) &&
        facts.has_spd_certification &&
        route.preconditioner_known_incompatible) {
        addSolverClaim(report, *options,
            PropertyStatus::Violated,
            CertificationClass::Violated,
            "Solver method '" + method_name +
                "' has certified SPD operator evidence but the configured preconditioner route is not CG-compatible",
            route.reason);
        addSolverIssue(report, IssueSeverity::Error,
            "CG preconditioner route is incompatible with SPD-only Krylov requirements: " +
            route.reason);
        return;
    }

    if (isSPDOnlyMethod(options->method) &&
        facts.has_spd_certification &&
        (!route.preconditioner_symmetric_certified ||
         !route.preconditioner_positive_certified ||
         !route.left_right_preconditioning_metadata_present)) {
        addSolverClaim(report, *options,
            PropertyStatus::Unknown,
            CertificationClass::NotCertified,
            "Solver method '" + method_name +
                "' has certified SPD operator evidence but the CG preconditioner route is not certified",
            facts.reason + "; " + route.reason,
            AnalysisConfidence::Medium);
        addSolverIssue(report, IssueSeverity::Warning,
            "CG compatibility is not certified because preconditioner symmetry/positivity metadata is incomplete: " +
            route.reason);
        return;
    }

    if (isSPDOnlyMethod(options->method) && facts.has_spd_certification) {
        addSolverClaim(report, *options,
            PropertyStatus::Preserved,
            CertificationClass::Certified,
            "Solver method '" + method_name +
                "' is compatible with certified SPD operator and preconditioner-route evidence",
            facts.reason + "; " + route.reason);
        return;
    }

    if (isSPDOnlyMethod(options->method) &&
        (facts.has_incomplete_spd_evidence ||
         facts.has_uncertain_solver_route_evidence)) {
        addSolverClaim(report, *options,
            PropertyStatus::Unknown,
            CertificationClass::NotCertified,
            "Solver method '" + method_name +
                "' requires hard scoped SPD, symmetry, definiteness, and preconditioner-route evidence that is incomplete in the current analysis",
            facts.reason,
            AnalysisConfidence::Medium);
        addSolverIssue(report, IssueSeverity::Warning,
            "CG compatibility is not certified because analysis found only incomplete or structural solver-route evidence: " +
            facts.reason);
        return;
    }

    if (isGeneralIndefiniteMethod(options->method) &&
        (facts.has_mixed_or_indefinite_structure ||
         facts.has_nonsymmetric_or_transport_structure)) {
        if (facts.has_indefinite_resolution_certification) {
            addSolverClaim(report, *options,
                PropertyStatus::Preserved,
                CertificationClass::Certified,
                "Solver method '" + method_name +
                    "' is backed by certified indefinite/Schur resolution evidence",
                facts.reason);
        } else {
            addSolverClaim(report, *options,
                PropertyStatus::Likely,
                CertificationClass::NotCertified,
                "Solver method '" + method_name +
                    "' is admissible for nonsymmetric, transport, or indefinite structure but lacks convergence/preconditioner certification evidence",
                facts.reason,
                AnalysisConfidence::Medium);
        }
        return;
    }

    if (isGeneralIndefiniteMethod(options->method) &&
        facts.has_spd_certification) {
        addSolverClaim(report, *options,
            PropertyStatus::Likely,
            CertificationClass::NotCertified,
            "Solver method '" + method_name +
                "' is admissible for a certified SPD system but is not the SPD-specialized route",
            facts.reason + "; convergence and preconditioner quality are not certified for this general route",
            AnalysisConfidence::Medium);
        return;
    }

    if (isGeneralIndefiniteMethod(options->method) &&
        (facts.has_incomplete_spd_evidence ||
         facts.has_uncertain_solver_route_evidence)) {
        addSolverClaim(report, *options,
            PropertyStatus::Unknown,
            CertificationClass::NotCertified,
            "Solver method '" + method_name +
                "' is a general route but operator/preconditioner compatibility is not certified from the available evidence",
            facts.reason,
            AnalysisConfidence::Medium);
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
