/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/DiscreteMonotonicityAnalyzer.h"
#include "Analysis/AnalysisSummaryTypes.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
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

bool allVariablesScalar(const ProblemAnalysisContext& context,
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
        if (fd && fd->value_dimension > 1) {
            return false;
        }
    }
    return true;
}

Real effectiveTolerance(const DiscreteMatrixSummary& matrix) noexcept
{
    return std::max(matrix.sign_tolerance, matrix.row_sum_tolerance);
}

struct MatrixSignVerdict {
    bool analyzable{false};
    bool z_matrix{false};
    bool m_matrix{false};
    bool positive_offdiag_violation{false};
    bool diagonal_violation{false};
    bool row_sum_violation{false};
};

MatrixSignVerdict classifyMatrix(const DiscreteMatrixSummary& matrix)
{
    MatrixSignVerdict verdict;
    verdict.analyzable = matrix.square && matrix.rows == matrix.cols && matrix.rows > 0;
    if (!verdict.analyzable) {
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
        matrix.row_sum_violation_count > 0u ||
        matrix.min_row_sum < -tol;

    verdict.z_matrix = !verdict.positive_offdiag_violation;
    verdict.m_matrix = verdict.z_matrix &&
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
    claim.variables = blockVariables(matrix.block);
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
    const auto variables = blockVariables(matrix.block);
    const bool scalar_applicable = allVariablesScalar(context, variables);
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
            scalar_applicable ? ApplicabilityClass::Applicable
                              : ApplicabilityClass::Unknown);
        return;
    }

    if (verdict.m_matrix) {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Exact, CertificationClass::Certified,
            MatrixSignStructureClass::MMatrixCertified,
            reduction_prefix + "matrix is M-matrix eligible for " + label,
            "Z-matrix sign pattern, positive diagonals, and nonnegative row sums were certified");
    } else {
        addMatrixClaim(report, matrix, PropertyKind::MMatrixStructure,
            PropertyStatus::Violated, CertificationClass::NotCertified,
            MatrixSignStructureClass::MMatrixNotCertified,
            reduction_prefix + "matrix is not M-matrix certified for " + label,
            "Matrix summary reports positive off-diagonal, diagonal, or row-sum violations");
        addRiskIssue(report, matrix,
            "M-matrix certification failed from sign, diagonal, or row-sum evidence");
    }

    if (scalar_applicable && verdict.m_matrix) {
        addMatrixClaim(report, matrix, PropertyKind::DiscreteMaximumPrinciple,
            PropertyStatus::Preserved, CertificationClass::Certified,
            MatrixSignStructureClass::MMatrixCertified,
            reduction_prefix + "scalar operator has DMP-compatible M-matrix evidence for " + label,
            "Scalar variables and certified M-matrix sign evidence support DMP applicability");
    } else if (scalar_applicable && verdict.positive_offdiag_violation) {
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
            "DMP checks require scalar-variable M-matrix evidence",
            scalar_applicable ? ApplicabilityClass::Applicable
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
    claim.variables = blockVariables(stencil.block);
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
