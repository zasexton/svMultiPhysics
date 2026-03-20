/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_INTERFACE_VALIDATION_ANALYZER_H
#define SVMP_FE_ANALYSIS_INTERFACE_VALIDATION_ANALYZER_H

/**
 * @file InterfaceValidationAnalyzer.h
 * @brief Validates interface contributions against InterfaceTopologyContext
 *
 * Checks that:
 *   - SpecificMarker contributions have a matching InterfaceMesh
 *   - AllRegisteredInterfaces contributions have at least one InterfaceMesh
 *   - Registered InterfaceMesh objects are targeted by at least one contribution
 *
 * Post-setup: strict validation (Error for missing meshes).
 * Pre-setup: provisional warnings only.
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class InterfaceValidationAnalyzer : public AnalyzerPass {
public:
    InterfaceValidationAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_INTERFACE_VALIDATION_ANALYZER_H
