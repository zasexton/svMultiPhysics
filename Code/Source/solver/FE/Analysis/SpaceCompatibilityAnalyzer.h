/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_SPACE_COMPATIBILITY_ANALYZER_H
#define SVMP_FE_ANALYSIS_SPACE_COMPATIBILITY_ANALYZER_H

/**
 * @file SpaceCompatibilityAnalyzer.h
 * @brief FE space pair compatibility analysis pass
 *
 * Reads field descriptors for space_family and trace_capabilities,
 * BC descriptors for trace_kind, and prior MixedSaddlePoint claims
 * to check compatibility of FE space pairs and BC enforcement.
 *
 * Detects mismatches such as HDiv fields with Value trace BCs,
 * L2 fields with Value trace BCs, and incompatible space pairings
 * in mixed saddle-point systems.
 *
 * Emits PropertyKind::SpaceCompatibility claims.
 *
 * @see MixedOperatorAnalyzer for MixedSaddlePoint claims consumed here
 * @see BoundaryConditionDescriptor for trace_kind
 * @see FieldDescriptor for space_family and trace_capabilities
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

class SpaceCompatibilityAnalyzer : public AnalyzerPass {
public:
    SpaceCompatibilityAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_SPACE_COMPATIBILITY_ANALYZER_H
