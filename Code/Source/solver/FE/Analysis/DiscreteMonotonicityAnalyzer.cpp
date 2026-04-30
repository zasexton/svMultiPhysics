/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/DiscreteMonotonicityAnalyzer.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void appendMissing(std::vector<std::string>& missing,
                   bool present,
                   std::string label)
{
    if (!present) {
        missing.push_back(std::move(label));
    }
}

std::string joinMissing(const std::vector<std::string>& missing)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < missing.size(); ++i) {
        if (i != 0u) os << ", ";
        os << missing[i];
    }
    return os.str();
}

bool allVariablesNodalScalar(const ProblemAnalysisContext& context,
                             const std::vector<VariableKey>& variables)
{
    if (variables.empty()) {
        return false;
    }
    for (const auto& var : variables) {
        if (var.kind != VariableKind::FieldComponent) {
            return false;
        }
        const auto* fd = context.fieldDescriptor(var.field_id);
        if (!fd ||
            fd->field_type != FieldType::Scalar ||
            fd->value_dimension != 1 ||
            fd->space_family != SpaceFamily::H1 ||
            fd->continuity != Continuity::C0 ||
            !fd->component_extractable) {
            return false;
        }
    }
    return true;
}

bool contributionMatchesBlock(const ContributionDescriptor& contribution,
                              const OperatorBlockId& block)
{
    OperatorBlockId cblock;
    cblock.contribution_id = contribution.contribution_id;
    cblock.test_variables = contribution.test_variables;
    cblock.trial_variables = contribution.trial_variables;
    cblock.domain = contribution.domain;
    cblock.role = contribution.role;
    cblock.operator_tag = contribution.operator_tag;
    cblock.marker = contribution.boundary_marker >= 0
        ? contribution.boundary_marker
        : contribution.interface_marker;
    return blockEvidenceMatches(cblock, block);
}

bool containsString(const std::vector<std::string>& values,
                    const std::string& value)
{
    return !value.empty() &&
           std::find(values.begin(), values.end(), value) != values.end();
}

std::vector<VariableKey> contributionVariables(
    const ContributionDescriptor& contribution)
{
    std::vector<VariableKey> variables;
    for (const auto& v : contribution.test_variables) {
        appendUniqueVariable(variables, v);
    }
    for (const auto& v : contribution.trial_variables) {
        appendUniqueVariable(variables, v);
    }
    return variables;
}

bool contributionConservativelyTouchesBlock(
    const ContributionDescriptor& contribution,
    const OperatorBlockId& block)
{
    if (contribution.domain != block.domain) {
        return false;
    }
    const auto matrix_variables = variablesForBlock(block);
    if (matrix_variables.empty()) {
        return false;
    }
    return variableSetsIntersect(contributionVariables(contribution),
                                 matrix_variables);
}

bool contributionMatchesMatrix(const ContributionDescriptor& contribution,
                               const DiscreteMatrixSummary& matrix,
                               bool conservative_when_unprovenanced)
{
    const bool id_match =
        containsString(matrix.contribution_ids,
                       contribution.contribution_id) ||
        (!matrix.block.contribution_id.empty() &&
         matrix.block.contribution_id == contribution.contribution_id);
    const bool tag_match =
        containsString(matrix.contribution_tags,
                       contribution.operator_tag);

    if (matrix.contribution_provenance_complete) {
        if (!matrix.contribution_ids.empty() ||
            !matrix.block.contribution_id.empty()) {
            return id_match;
        }
        if (!matrix.contribution_tags.empty()) {
            return tag_match;
        }
    } else if (id_match || tag_match) {
        return true;
    }
    if (contributionMatchesBlock(contribution, matrix.block)) {
        return true;
    }
    return conservative_when_unprovenanced &&
           contributionConservativelyTouchesBlock(contribution, matrix.block);
}

bool hasSecondOrderDiffusionEvidence(const ProblemAnalysisContext& context,
                                     const DiscreteMatrixSummary& matrix)
{
    for (const auto& contribution : context.contributions()) {
        if (contribution.role != ContributionRole::DiagonalBlock ||
            contribution.domain != DomainKind::Cell ||
            !contributionMatchesMatrix(contribution, matrix,
                                       /*conservative_when_unprovenanced=*/true)) {
            continue;
        }
        if (hasFlag(contribution.traits, OperatorTraitFlags::HasSecondOrder) &&
            !hasFlag(contribution.traits, OperatorTraitFlags::HasFirstOrder)) {
            return true;
        }
    }
    return false;
}

bool hasTransportContamination(const ProblemAnalysisContext& context,
                               const DiscreteMatrixSummary& matrix)
{
    for (const auto& contribution : context.contributions()) {
        if (contribution.domain != DomainKind::Cell ||
            !contributionMatchesMatrix(contribution, matrix,
                                       /*conservative_when_unprovenanced=*/true)) {
            continue;
        }
        if (hasFlag(contribution.traits, OperatorTraitFlags::HasFirstOrder)) {
            return true;
        }
        if (contribution.transport_character &&
            *contribution.transport_character != TransportCharacter::None &&
            *contribution.transport_character != TransportCharacter::DiffusionLike) {
            return true;
        }
    }
    return false;
}

bool hasScopedPositiveCoefficientEvidence(const AnalysisSummarySet* summaries,
                                          const DiscreteMatrixSummary& matrix)
{
    if (!summaries) return false;
    for (const auto& coefficient : summaries->coefficient_properties) {
        if (!coefficientSummaryCovers(coefficient, matrix)) {
            continue;
        }
        const bool positive =
            coefficient.positivity == PositivityClass::Positive ||
            coefficient.positivity == PositivityClass::Nonnegative;
        const bool state_scope_ok =
            (!coefficient.state_dependent && !coefficient.time_dependent) ||
            coefficient.state_sample_coverage_complete;
        const bool coverage_complete =
            coefficient.coefficient_region_coverage_complete &&
            coefficient.quadrature_point_coverage_complete &&
            coefficient.lower_bound_valid_for_all_samples &&
            coefficient.tolerance_metadata_present &&
            coefficientLowerBoundMatchesPositivity(coefficient) &&
            state_scope_ok;
        if (positive && coverage_complete) {
            return true;
        }
    }
    return false;
}

struct DmpGateEvidence {
    bool nodal_scalar_space{false};
    bool second_order_diffusion{false};
    bool positive_coefficient{false};
    bool mesh_operator_applicability{false};
    bool rhs_sign_compatible{false};
    bool no_transport_contamination{false};
};

DmpGateEvidence collectDmpGates(const ProblemAnalysisContext& context,
                                const DiscreteMatrixSummary& matrix)
{
    DmpGateEvidence gates;
    const auto variables = variablesForBlock(matrix.block);
    gates.nodal_scalar_space = allVariablesNodalScalar(context, variables);
    gates.second_order_diffusion =
        hasSecondOrderDiffusionEvidence(context, matrix);
    gates.positive_coefficient =
        hasScopedPositiveCoefficientEvidence(context.analysisSummaries(),
                                            matrix);
    gates.mesh_operator_applicability = matrix.dmp_applicability_evidence;
    gates.rhs_sign_compatible = matrix.dmp_rhs_sign_evidence;
    gates.no_transport_contamination =
        !hasTransportContamination(context, matrix);
    return gates;
}

bool dmpGatesSatisfied(const DmpGateEvidence& gates) noexcept
{
    return gates.nodal_scalar_space &&
           gates.second_order_diffusion &&
           gates.positive_coefficient &&
           gates.mesh_operator_applicability &&
           gates.rhs_sign_compatible &&
           gates.no_transport_contamination;
}

std::string missingDmpGateDescription(const DmpGateEvidence& gates)
{
    std::vector<std::string> missing;
    appendMissing(missing, gates.nodal_scalar_space,
                  "nodal scalar H1/C0 space evidence");
    appendMissing(missing, gates.second_order_diffusion,
                  "second-order diffusion operator evidence");
    appendMissing(missing, gates.positive_coefficient,
                  "scoped positive coefficient evidence");
    appendMissing(missing, gates.mesh_operator_applicability,
                  "mesh/operator DMP applicability evidence");
    appendMissing(missing, gates.rhs_sign_compatible,
                  "source/reaction/RHS sign compatibility evidence");
    appendMissing(missing, gates.no_transport_contamination,
                  "absence of transport contamination");
    return joinMissing(missing);
}

Real effectiveTolerance(const DiscreteMatrixSummary& matrix) noexcept
{
    if (!numeric::finiteNonnegative(matrix.sign_tolerance) ||
        !numeric::finiteNonnegative(matrix.row_sum_tolerance)) {
        return Real{};
    }
    return std::max(matrix.sign_tolerance, matrix.row_sum_tolerance);
}

struct MatrixSignVerdict {
    bool analyzable{false};
    bool sign_evidence_complete{false};
    bool row_sum_evidence_complete{false};
    bool z_matrix{false};
    bool m_matrix{false};
    bool positive_offdiag_violation{false};
    bool diagonal_violation{false};
    bool row_sum_violation{false};
    bool numeric_evidence_valid{false};
};

struct MMatrixTheoremEvidence {
    bool certified{false};
    std::string label;
    std::string missing;
};

bool symmetricEvidenceComplete(const DiscreteMatrixSummary& matrix) noexcept
{
    return matrix.symmetry_evidence_complete &&
           (matrix.structurally_symmetric || matrix.numerically_symmetric);
}

bool positiveDefiniteEvidenceComplete(
    const DiscreteMatrixSummary& matrix,
    Real tolerance) noexcept
{
    const bool eigenvalue_positive =
        matrix.min_eigenvalue_estimate &&
        numeric::finite(*matrix.min_eigenvalue_estimate) &&
        *matrix.min_eigenvalue_estimate > tolerance;
    const bool coercivity_positive =
        matrix.coercivity_lower_bound &&
        numeric::finite(*matrix.coercivity_lower_bound) &&
        *matrix.coercivity_lower_bound > tolerance;
    return matrix.cholesky_factorization_succeeded ||
           eigenvalue_positive ||
           coercivity_positive;
}

MMatrixTheoremEvidence mMatrixTheoremEvidence(
    const DiscreteMatrixSummary& matrix,
    const MatrixSignVerdict& verdict)
{
    MMatrixTheoremEvidence evidence;
    std::vector<std::string> missing;
    const Real tol = effectiveTolerance(matrix);
    const bool theorem_id_present = !matrix.m_matrix_theorem_id.empty();
    const bool stieltjes_route =
        theorem_id_present &&
        matrix.stieltjes_matrix_evidence &&
        symmetricEvidenceComplete(matrix) &&
        positiveDefiniteEvidenceComplete(matrix, tol);
    const bool inverse_positive_route =
        theorem_id_present &&
        matrix.inverse_positivity_evidence &&
        matrix.inverse_positivity_metadata_present;
    const bool irreducible_dd_route =
        theorem_id_present &&
        matrix.irreducible_diagonal_dominance_evidence &&
        matrix.diagonal_dominance_evidence_complete &&
        matrix.irreducibility_evidence_present;

    if (stieltjes_route) {
        evidence.label = theorem_id_present
            ? matrix.m_matrix_theorem_id
            : "Stieltjes/SPD Z-matrix evidence";
        if (matrix.m_matrix_certification_evidence) {
            evidence.certified = verdict.m_matrix;
            return evidence;
        }
    }
    if (inverse_positive_route) {
        evidence.label = theorem_id_present
            ? matrix.m_matrix_theorem_id
            : "inverse-positivity evidence";
        if (matrix.m_matrix_certification_evidence) {
            evidence.certified = verdict.m_matrix;
            return evidence;
        }
    }
    if (irreducible_dd_route) {
        evidence.label = theorem_id_present
            ? matrix.m_matrix_theorem_id
            : "irreducible diagonal-dominance evidence";
        if (matrix.m_matrix_certification_evidence) {
            evidence.certified = verdict.m_matrix;
            return evidence;
        }
    }
    appendMissing(missing, matrix.m_matrix_certification_evidence,
                  "M-matrix certification evidence flag");
    appendMissing(missing, theorem_id_present,
                  "M-matrix theorem identifier");
    if (matrix.stieltjes_matrix_evidence) {
        appendMissing(missing, symmetricEvidenceComplete(matrix),
                      "symmetric matrix evidence");
        appendMissing(missing, positiveDefiniteEvidenceComplete(matrix, tol),
                      "SPD evidence from Cholesky, eigenvalue, or coercivity lower bound");
    }
    if (matrix.inverse_positivity_evidence) {
        appendMissing(missing, matrix.inverse_positivity_metadata_present,
                      "inverse-positivity metadata");
    }
    if (matrix.irreducible_diagonal_dominance_evidence) {
        appendMissing(missing, matrix.diagonal_dominance_evidence_complete,
                      "diagonal-dominance metadata");
        appendMissing(missing, matrix.irreducibility_evidence_present,
                      "irreducibility/connectivity metadata");
    }
    if (!matrix.stieltjes_matrix_evidence &&
        !matrix.inverse_positivity_evidence &&
        !matrix.irreducible_diagonal_dominance_evidence) {
        missing.push_back(
            "one theorem route: Stieltjes/SPD, inverse positivity, or irreducible diagonal dominance");
    }
    if (evidence.label.empty()) {
        evidence.label = "missing theorem-specific M-matrix evidence";
    }
    evidence.missing = joinMissing(missing);
    return evidence;
}

MatrixSignVerdict classifyMatrix(const DiscreteMatrixSummary& matrix)
{
    MatrixSignVerdict verdict;
    verdict.analyzable = matrix.square && matrix.rows == matrix.cols && matrix.rows > 0;
    const bool tolerance_evidence_valid =
        numeric::finiteNonnegative(matrix.sign_tolerance) &&
        numeric::finiteNonnegative(matrix.row_sum_tolerance);
    const bool sign_numeric_valid =
        tolerance_evidence_valid &&
        matrix.nonfinite_entry_count == 0u &&
        numeric::finiteNonnegative(matrix.max_positive_offdiag);
    const bool row_sum_numeric_valid =
        sign_numeric_valid &&
        matrix.nonfinite_row_sum_count == 0u &&
        numeric::finite(matrix.min_row_sum) &&
        numeric::finite(matrix.max_row_sum) &&
        numeric::finiteNonnegative(matrix.max_abs_row_sum);
    verdict.numeric_evidence_valid = sign_numeric_valid;
    verdict.sign_evidence_complete =
        matrix.sign_evidence_complete && sign_numeric_valid;
    verdict.row_sum_evidence_complete =
        matrix.row_sum_evidence_complete && row_sum_numeric_valid;
    if (!verdict.analyzable) {
        return verdict;
    }
    if (!verdict.sign_evidence_complete) {
        return verdict;
    }

    const Real tol = effectiveTolerance(matrix);
    verdict.positive_offdiag_violation =
        matrix.positive_offdiag_count > 0u ||
        matrix.max_positive_offdiag > tol;
    verdict.diagonal_violation =
        matrix.nonpositive_diagonal_count > 0u ||
        matrix.negative_diagonal_count > 0u ||
        matrix.near_zero_diagonal_count > 0u;
    verdict.row_sum_violation =
        verdict.row_sum_evidence_complete &&
        (matrix.row_sum_violation_count > 0u ||
         matrix.min_row_sum < -tol);

    verdict.z_matrix = !verdict.positive_offdiag_violation;
    verdict.m_matrix = verdict.z_matrix &&
                       verdict.row_sum_evidence_complete &&
                       !verdict.diagonal_violation &&
                       !verdict.row_sum_violation &&
                       matrix.diagonal_count > 0u;
    return verdict;
}

std::string matrixLabel(const DiscreteMatrixSummary& matrix)
{
    if (!matrix.block.operator_tag.empty()) {
        return "'" + matrix.block.operator_tag + "'";
    }
    return "matrix block";
}

void addMatrixClaim(ProblemAnalysisReport& report,
                    const DiscreteMatrixSummary& matrix,
                    PropertyKind kind,
                    PropertyStatus status,
                    CertificationClass certification,
                    MatrixSignStructureClass sign_class,
                    std::string description,
                    std::string evidence,
                    ApplicabilityClass applicability = ApplicabilityClass::Applicable)
{
    PropertyClaim claim;
    claim.kind = kind;
    claim.status = status;
    claim.confidence = AnalysisConfidence::High;
    claim.domain = matrix.block.domain;
    claim.variables = variablesForBlock(matrix.block);
    claim.applicability_class = applicability;
    claim.certification_class = certification;
    claim.matrix_sign_structure_class = sign_class;
    claim.tested_block_id = matrix.block.operator_tag;
    claim.description = std::move(description);
    claim.claim_origin = "DiscreteMonotonicityAnalyzer";
    claim.addEvidence("DiscreteMonotonicityAnalyzer", std::move(evidence));
    report.claims.push_back(std::move(claim));
}

void addRiskIssue(ProblemAnalysisReport& report,
                  const DiscreteMatrixSummary& matrix,
                  std::string message)
{
    AnalysisIssue issue;
    issue.severity = IssueSeverity::Warning;
    issue.message = "Discrete monotonicity risk in " + matrixLabel(matrix) +
                    ": " + std::move(message);
    report.issues.push_back(std::move(issue));
}

void analyzeMatrix(const ProblemAnalysisContext& context,
                   ProblemAnalysisReport& report,
                   const DiscreteMatrixSummary& matrix,
                   bool reduced,
                   bool reduction_exact)
{
    const auto verdict = classifyMatrix(matrix);
    const auto gates = collectDmpGates(context, matrix);
    const bool dmp_applicable = dmpGatesSatisfied(gates);
    const bool nodal_scalar_applicable = gates.nodal_scalar_space;
    const auto theorem_evidence = mMatrixTheoremEvidence(matrix, verdict);
    const bool m_matrix_certified = theorem_evidence.certified;
    const std::string m_matrix_theorem = theorem_evidence.label;
    const std::string reduction_prefix = reduced ? "Reduced free-free " : "";
    const std::string label = matrixLabel(matrix);

    if (!verdict.analyzable) {
        addMatrixClaim(report, matrix, PropertyKind::ZMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "Z-matrix status unknown for " + label,
            "Matrix summary is missing square nonempty dimensions");
        return;
    }

    if (!verdict.sign_evidence_complete) {
        const std::string coverage =
            "Matrix sign evidence is incomplete: scanned_rows=" +
            std::to_string(matrix.scanned_row_count) +
            ", expected_rows=" + std::to_string(matrix.expected_row_count) +
            ", nonfinite_entries=" +
            std::to_string(matrix.nonfinite_entry_count);
        addMatrixClaim(report, matrix, PropertyKind::ZMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "Z-matrix status unknown for " + label,
            coverage);
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "M-matrix status unknown for " + label,
            coverage);
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "DMP applicability is unknown for " + label,
            coverage,
            nodal_scalar_applicable ? ApplicabilityClass::Applicable
                                    : ApplicabilityClass::Unknown);
        return;
    }

    if (verdict.z_matrix) {
        addMatrixClaim(report, matrix, PropertyKind::ZMatrixStructure,
            PropertyStatus::Exact, CertificationClass::Certified,
            MatrixSignStructureClass::ZMatrix,
            reduction_prefix + "off-diagonal sign structure is a Z-matrix for " + label,
            "No positive off-diagonal entries exceeded the sign tolerance");
    } else {
        addMatrixClaim(report, matrix, PropertyKind::ZMatrixStructure,
            PropertyStatus::Violated, CertificationClass::Violated,
            MatrixSignStructureClass::NotZMatrix,
            reduction_prefix + "off-diagonal sign structure violates Z-matrix requirements for " + label,
            std::to_string(matrix.positive_offdiag_count) +
                " positive off-diagonal entries exceeded the sign tolerance");
        addRiskIssue(report, matrix,
            "positive off-diagonal entries break Z-matrix monotonicity evidence");
    }

    if (verdict.z_matrix && !verdict.row_sum_evidence_complete) {
        const std::string coverage =
            "Matrix row-sum evidence is incomplete: scanned_rows=" +
            std::to_string(matrix.scanned_row_count) +
            ", expected_rows=" + std::to_string(matrix.expected_row_count) +
            ", nonfinite_row_sums=" +
            std::to_string(matrix.nonfinite_row_sum_count);
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "M-matrix row-sum status unknown for " + label,
            coverage);
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "DMP applicability is unknown for " + label,
            coverage,
            nodal_scalar_applicable ? ApplicabilityClass::Applicable
                                    : ApplicabilityClass::Unknown);
        return;
    }

    if (reduced && !reduction_exact) {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            "Reduced M-matrix status unknown for " + label +
                " because affine or retained constraints were not accounted for exactly",
            "ReducedMatrixSummary reports reduction_exact_for_analysis=false");
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            "DMP certification unknown for " + label +
                " because the reduced operator does not fully account for constraints",
            "ReducedMatrixSummary lacks exact reduction evidence",
            nodal_scalar_applicable ? ApplicabilityClass::Applicable
                                    : ApplicabilityClass::Unknown);
        return;
    }

    if (m_matrix_certified) {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Exact, CertificationClass::Certified,
            MatrixSignStructureClass::MMatrixCertified,
            reduction_prefix + "matrix is M-matrix eligible for " + label,
            "Z-matrix sign pattern, positive diagonals, nonnegative row sums, and " +
                m_matrix_theorem + " were certified");
    } else if (verdict.m_matrix) {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::MMatrixNotCertified,
            reduction_prefix + "matrix has M-matrix sign prerequisites but no complete M-matrix certificate for " + label,
            "Z-matrix sign pattern, positive diagonals, and nonnegative row sums were present, but theorem-specific nonsingularity, inverse-positivity, Stieltjes/SPD, or irreducible diagonal-dominance evidence is incomplete: " +
                theorem_evidence.missing);
    } else {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Violated, CertificationClass::NotCertified,
            MatrixSignStructureClass::MMatrixNotCertified,
            reduction_prefix + "matrix is not M-matrix certified for " + label,
            "Matrix summary reports positive off-diagonal, diagonal, or row-sum violations");
        addRiskIssue(report, matrix,
            "M-matrix certification failed from sign, diagonal, or row-sum evidence");
    }

    if (dmp_applicable && m_matrix_certified) {
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Preserved, CertificationClass::Certified,
            MatrixSignStructureClass::MMatrixCertified,
            reduction_prefix + "scalar operator has DMP-compatible M-matrix evidence for " + label,
            "Nodal scalar space, diffusion/coefficient/source gates, transport exclusion, and certified M-matrix evidence support DMP applicability");
    } else if (nodal_scalar_applicable && verdict.positive_offdiag_violation) {
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Violated, CertificationClass::Violated,
            MatrixSignStructureClass::NotZMatrix,
            reduction_prefix + "scalar operator violates DMP sign prerequisites for " + label,
            "Positive off-diagonal entries violate the Z-matrix prerequisite");
    } else {
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Unknown, CertificationClass::Unknown,
            MatrixSignStructureClass::Unknown,
            reduction_prefix + "DMP applicability is unknown for " + label,
            "DMP checks require certified M-matrix evidence plus missing gates: " +
                missingDmpGateDescription(gates),
            nodal_scalar_applicable ? ApplicabilityClass::Applicable
                                    : ApplicabilityClass::Unknown);
    }
}

void analyzeLocalStencil(ProblemAnalysisReport& report,
                         const LocalStencilSummary& stencil)
{
    if (stencil.positive_offdiag_count == 0u) {
        return;
    }

    PropertyClaim claim;
    claim.kind = PropertyKind::MatrixMonotonicityRisk;
    claim.status = PropertyStatus::Likely;
    claim.confidence = AnalysisConfidence::Medium;
    claim.domain = stencil.block.domain;
    claim.variables = variablesForBlock(stencil.block);
    claim.applicability_class = ApplicabilityClass::Applicable;
    claim.certification_class = CertificationClass::NotCertified;
    claim.matrix_sign_structure_class = MatrixSignStructureClass::NotZMatrix;
    claim.tested_block_id = stencil.block.operator_tag;
    claim.description =
        "Local stencil has positive off-diagonal entries on element " +
        std::to_string(stencil.element) +
        ", so DMP/Z-matrix monotonicity needs matrix-level confirmation";
    claim.claim_origin = "DiscreteMonotonicityAnalyzer";
    claim.addEvidence("DiscreteMonotonicityAnalyzer",
        std::to_string(stencil.positive_offdiag_count) +
        " local positive off-diagonal entries exceeded tolerance",
        AnalysisConfidence::Medium);
    report.claims.push_back(std::move(claim));
}

} // namespace

std::string DiscreteMonotonicityAnalyzer::name() const {
    return "DiscreteMonotonicityAnalyzer";
}

void DiscreteMonotonicityAnalyzer::run(const ProblemAnalysisContext& context,
                                       ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& matrix : summaries->discrete_matrices) {
        analyzeMatrix(context, report, matrix, /*reduced=*/false,
                      /*reduction_exact=*/true);
    }

    for (const auto& reduced : summaries->reduced_matrices) {
        const bool exact = reduced.reduction_exact_for_analysis &&
                           (reduced.reduction_kind != ConstraintReductionKind::AffineTransform ||
                            reduced.affine_terms_accounted_for);
        analyzeMatrix(context, report, reduced.free_free_matrix,
                      /*reduced=*/true, exact);
    }

    for (const auto& stencil : summaries->local_stencils) {
        analyzeLocalStencil(report, stencil);
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
