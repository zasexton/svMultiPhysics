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
    return numeric::finitePositiveOrdered(summary.min_jacobian,
                                          summary.max_jacobian);
}

bool hasAspectRatioRisk(const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.aspect_ratio_warning_threshold > 0.0 &&
           summary.max_aspect_ratio > summary.aspect_ratio_warning_threshold;
}

bool hasJacobianViolation(const MeshGeometryQualitySummary& summary) noexcept
{
    return summary.inverted_element_count > 0u ||
           !numeric::finiteOrdered(summary.min_jacobian,
                                   summary.max_jacobian) ||
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
                "Mesh mapping validity violated: inverted or nonpositive-Jacobian elements were reported",
                std::to_string(summary.inverted_element_count) +
                    " inverted elements; min_jacobian=" +
                    std::to_string(summary.min_jacobian));
            addGeometryIssue(report, summary,
                "inverted or nonpositive-Jacobian elements invalidate coercivity and monotonicity evidence");
        } else if (summary.poor_quality_element_count > 0u ||
                   hasAspectRatioRisk(summary) ||
                   summary.cut_cell_count > 0u) {
            addGeometryClaim(report, summary,
                PropertyStatus::Likely,
                CertificationClass::NotCertified,
                "Mesh geometry quality risk reported for active discretization",
                std::to_string(summary.poor_quality_element_count) +
                    " poor-quality elements, " +
                    std::to_string(summary.cut_cell_count) +
                    " cut cells, max_aspect_ratio=" +
                    std::to_string(summary.max_aspect_ratio));
            addTopologyScopedGeometryClaims(context, report, summary);
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
                    ? "Mesh mapping validity and reported shape-regularity evidence certified"
                    : "Mesh mapping invertibility certified by positive Jacobian summary; FEM shape-regularity stability evidence is out of scope",
                "min_jacobian=" + std::to_string(summary.min_jacobian) +
                    ", max_jacobian=" +
                    std::to_string(summary.max_jacobian) +
                    shape_text);
        } else {
            addGeometryClaim(report, summary,
                PropertyStatus::Unknown,
                CertificationClass::Unknown,
                "Mesh geometry validity summary is present but lacks decisive Jacobian evidence",
                "No inverted elements were reported, but positive-Jacobian bounds are unavailable");
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
