/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_GAUGE_ADAPTER_H
#define SVMP_FE_ANALYSIS_GAUGE_ADAPTER_H

/**
 * @file GaugeAdapter.h
 * @brief Adapter between ProblemAnalysisReport and GaugeRegistry
 *
 * Maps PropertyClaims (from the analysis subsystem) into GaugeCandidates
 * and AnchoringEvidence (for the existing GaugeRegistry enforcement pipeline).
 *
 * NOTE: The production gauge candidate path uses a simpler mechanism:
 *   FormContributionLowerer → NullspaceHint in ContributionDescriptor
 *   → SystemSetup NullspaceHint→GaugeCandidate conversion
 * This adapter (populateRegistryFromReport) is used in tests for roundtrip
 * validation between the analysis report and the direct NullspaceAnalyzer path.
 *
 * @see GaugeRegistry for the enforcement backend
 * @see KernelAnalyzer for the nullspace detection pass
 */

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Constraints/GaugeRegistry.h"

#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Convert Nullspace claims from a report to GaugeCandidates
 *
 * Filters for PropertyKind::Nullspace claims and maps each to a
 * GaugeCandidate with appropriate NullspaceModeFamily, Confidence,
 * field, component, and region.
 */
[[nodiscard]] std::vector<gauge::GaugeCandidate>
claimsToCandidates(const ProblemAnalysisReport& report);

/**
 * @brief Convert BC descriptors and constraint claims to AnchoringEvidence
 *
 * Scans the report's claims and the context's BC descriptors to produce
 * AnchoringEvidence suitable for GaugeRegistry.
 */
[[nodiscard]] std::vector<gauge::AnchoringEvidence>
descriptorsToEvidence(const std::vector<BoundaryConditionDescriptor>& descriptors);

/**
 * @brief Populate a GaugeRegistry from a ProblemAnalysisReport and BC descriptors
 *
 * Combines claimsToCandidates + descriptorsToEvidence and adds all entries
 * to the given GaugeRegistry.
 */
void populateRegistryFromReport(gauge::GaugeRegistry& registry,
                                 const ProblemAnalysisReport& report,
                                 const std::vector<BoundaryConditionDescriptor>& bc_descriptors);

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_GAUGE_ADAPTER_H
