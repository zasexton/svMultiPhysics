/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/MeshGeometryAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"

#include <algorithm>
#include <string>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void addGeometryIssue(ProblemAnalysisReport& report,
                      const MeshGeometryQualitySummary& summary,
                      const std::string& message)
{
    AnalysisIssue issue;
    issue.severity = IssueSeverity::Warning;
    issue.message = "Mesh geometry quality risk on " +
                    std::string(toString(summary.domain)) +
                    " domain: " + message;
    report.issues.push_back(std::move(issue));
}

void addGeometryClaim(ProblemAnalysisReport& report,
                      const MeshGeometryQualitySummary& summary,
                      PropertyStatus status,
                      CertificationClass certification,
                      std::string description,
                      std::string evidence,
                      int region = -1)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::MeshGeometryValidity;
    claim.status = status;
    claim.confidence = AnalysisConfidence::High;
    claim.domain = summary.domain;
    claim.region = region;
    claim.certification_class = certification;
    claim.description = std::move(description);
    claim.claim_origin = "MeshGeometryAnalyzer";
    claim.addEvidence("MeshGeometryAnalyzer", std::move(evidence));
    report.claims.push_back(std::move(claim));
}

bool hasPositiveJacobianEvidence(const MeshGeometryQualitySummary& summary) noexcept
{
    if (!summary.jacobian_bounds_present) {
        return false;
    }
    return numeric::finitePositiveOrdered(summary.min_jacobian,
                                          summary.max_jacobian);
}

bool hasCertifiedJacobianCoverage(
    const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.jacobian_bounds_present &&
           summary.jacobian_bounds_cover_all_active_cells &&
           summary.jacobian_bounds_are_certified_bounds &&
           summary.jacobian_bounds_cover_high_order_interior;
}

bool hasSampledPositiveJacobianEvidence(
    const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.jacobian_bounds_present &&
           numeric::finitePositiveOrdered(summary.min_jacobian,
                                          summary.max_jacobian) &&
           !hasCertifiedJacobianCoverage(summary);
}

bool hasAspectRatioRisk(const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.aspect_ratio_warning_threshold > 0.0 &&
           summary.max_aspect_ratio > summary.aspect_ratio_warning_threshold;
}

bool hasJacobianViolation(const MeshGeometryQualitySummary& summary) noexcept
{
    if (summary.inverted_element_count > 0u) {
        return true;
    }
    if (!summary.jacobian_bounds_present) {
        return false;
    }
    if (!numeric::finiteOrdered(summary.min_jacobian,
                                summary.max_jacobian)) {
        return hasCertifiedJacobianCoverage(summary);
    }
    return hasCertifiedJacobianCoverage(summary) &&
           summary.min_jacobian <= Real{};
}

std::uint64_t countRegionWorstElements(const ConnectedComponent& component,
                                       const MeshGeometryQualitySummary& summary)
{
    std::uint64_t count = 0;
    for (const auto element : summary.worst_elements) {
        if (std::find(component.cell_indices.begin(),
                      component.cell_indices.end(),
                      element) != component.cell_indices.end()) {
            ++count;
        }
    }
    return count;
}

void addTopologyScopedGeometryClaims(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report,
                                     const MeshGeometryQualitySummary& summary)
{
    const auto* topo = context.topologyContext();
    if (!topo || topo->components.empty() || summary.worst_elements.empty()) {
        return;
    }

    for (const auto& component : topo->components) {
        const auto worst_count = countRegionWorstElements(component, summary);
        if (worst_count == 0u) {
            continue;
        }

        addGeometryClaim(report, summary,
            PropertyStatus::Likely,
            CertificationClass::NotCertified,
            "Mesh geometry quality risk is localized to topology region " +
                std::to_string(component.region_id),
            std::to_string(worst_count) +
                " worst-quality element samples fall in this connected component",
            component.region_id);
    }
}

bool cutCellMitigationPresent(const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.cut_cell_mitigation_metadata_present ||
           summary.ghost_penalty_present ||
           summary.agglomeration_present ||
           summary.cell_merging_present ||
           summary.robust_cut_cell_quadrature_present ||
           summary.cut_cell_conditioning_evidence_present;
}

} // namespace

std::string MeshGeometryAnalyzer::name() const {
    return "MeshGeometryAnalyzer";
}

void MeshGeometryAnalyzer::run(const ProblemAnalysisContext& context,
                               ProblemAnalysisReport& report) const
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->mesh_geometry_quality) {
        if (hasJacobianViolation(summary)) {
            addGeometryClaim(report, summary,
                PropertyStatus::Violated,
                CertificationClass::Violated,
                "MeshMappingOrientation violation: certified Jacobian bounds or explicit inverted-element counts report invalid mapping orientation/invertibility",
                std::to_string(summary.inverted_element_count) +
                    " inverted elements; min_jacobian=" +
                    std::to_string(summary.min_jacobian) +
                    ", bound_method='" + summary.jacobian_bound_method + "'");
            report.claims.back().evidence_level =
                EvidenceLevel::CertifiedNumericTheorem;
            addGeometryIssue(report, summary,
                "inverted or nonpositive-Jacobian elements invalidate coercivity and monotonicity evidence");
        } else if (!summary.jacobian_bounds_present) {
            addGeometryClaim(report, summary,
                PropertyStatus::Unknown,
                CertificationClass::Unknown,
                "MeshMappingOrientation unknown: geometry summary lacks explicit Jacobian bound presence metadata",
                "jacobian_bounds_present=false; request mesh geometry summary with coverage, method, mesh revision, and geometry mapping revision");
            report.claims.back().evidence_level =
                EvidenceLevel::ScopedNumericSummary;
        } else if (summary.poor_quality_element_count > 0u ||
                   hasAspectRatioRisk(summary)) {
            addGeometryClaim(report, summary,
                PropertyStatus::Likely,
                CertificationClass::NotCertified,
                "MeshShapeRegularity or interpolation-quality risk reported for active discretization",
                std::to_string(summary.poor_quality_element_count) +
                    " poor-quality elements, max_aspect_ratio=" +
                    std::to_string(summary.max_aspect_ratio));
            report.claims.back().evidence_level =
                EvidenceLevel::ScopedNumericSummary;
            addTopologyScopedGeometryClaims(context, report, summary);
        } else if (summary.cut_cell_count > 0u) {
            const bool mitigated = cutCellMitigationPresent(summary);
            addGeometryClaim(report, summary,
                PropertyStatus::Likely,
                CertificationClass::NotCertified,
                mitigated
                    ? "CutCellConditioningRisk is reported with mitigation metadata"
                    : "CutCellConditioningRisk is reported without mitigation metadata",
                std::to_string(summary.cut_cell_count) +
                    " cut cells; ghost_penalty=" +
                    std::string(summary.ghost_penalty_present ? "true" : "false") +
                    ", agglomeration=" +
                    std::string(summary.agglomeration_present ? "true" : "false") +
                    ", cell_merging=" +
                    std::string(summary.cell_merging_present ? "true" : "false") +
                    ", robust_quadrature=" +
                    std::string(summary.robust_cut_cell_quadrature_present ? "true" : "false") +
                    ", conditioning_evidence=" +
                    std::string(summary.cut_cell_conditioning_evidence_present ? "true" : "false"));
            report.claims.back().evidence_level =
                EvidenceLevel::ScopedNumericSummary;
            if (!mitigated) {
                addGeometryIssue(report, summary,
                    "cut cells are present without ghost penalty, agglomeration, cell-merging, robust-quadrature, or conditioning mitigation metadata");
            }
            addTopologyScopedGeometryClaims(context, report, summary);
        } else if (hasSampledPositiveJacobianEvidence(summary)) {
            addGeometryClaim(report, summary,
                PropertyStatus::Likely,
                CertificationClass::NotCertified,
                "MeshMappingOrientation likely valid from positive Jacobian evidence, but not certified over the full active mesh/high-order interior",
                "min_jacobian=" + std::to_string(summary.min_jacobian) +
                    ", max_jacobian=" +
                    std::to_string(summary.max_jacobian) +
                    ", cover_all_active_cells=" +
                    std::string(summary.jacobian_bounds_cover_all_active_cells ? "true" : "false") +
                    ", cover_high_order_interior=" +
                    std::string(summary.jacobian_bounds_cover_high_order_interior ? "true" : "false") +
                    ", certified_bounds=" +
                    std::string(summary.jacobian_bounds_are_certified_bounds ? "true" : "false") +
                    ", method='" + summary.jacobian_bound_method + "'");
            report.claims.back().evidence_level =
                EvidenceLevel::ScopedNumericSummary;
        } else if (hasPositiveJacobianEvidence(summary)) {
            const bool shape_evidence_valid =
                summary.shape_regular_evidence_present &&
                numeric::finitePositive(summary.shape_regular_constant) &&
                summary.mesh_family_scope_present &&
                !summary.mesh_family_scope.empty();
            const std::string shape_text = shape_evidence_valid
                ? ", shape_regular_constant=" +
                      std::to_string(summary.shape_regular_constant) +
                      ", mesh_family_scope='" + summary.mesh_family_scope + "'"
                : ", shape regularity/stability assumptions not certified by this summary";
            addGeometryClaim(report, summary,
                PropertyStatus::Preserved,
                CertificationClass::Certified,
                shape_evidence_valid
                    ? "MeshMappingOrientation and MeshShapeRegularity evidence certified"
                    : "MeshMappingOrientation and MeshMappingInvertibility certified; MeshShapeRegularity stability evidence is out of scope",
                "min_jacobian=" + std::to_string(summary.min_jacobian) +
                    ", max_jacobian=" +
                    std::to_string(summary.max_jacobian) +
                    ", bound_method='" + summary.jacobian_bound_method + "'" +
                    shape_text);
            report.claims.back().evidence_level =
                EvidenceLevel::CertifiedNumericTheorem;
        } else {
            addGeometryClaim(report, summary,
                PropertyStatus::Unknown,
                CertificationClass::Unknown,
                "Mesh geometry validity summary is present but lacks decisive Jacobian evidence",
                "No inverted elements were reported, but positive-Jacobian bounds are unavailable");
            report.claims.back().evidence_level =
                EvidenceLevel::ScopedNumericSummary;
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
