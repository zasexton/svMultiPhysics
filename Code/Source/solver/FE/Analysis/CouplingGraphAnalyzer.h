/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_COUPLING_GRAPH_ANALYZER_H
#define SVMP_FE_ANALYSIS_COUPLING_GRAPH_ANALYZER_H

/**
 * @file CouplingGraphAnalyzer.h
 * @brief Variable coupling graph analysis pass
 *
 * Iterates formulation records, kernel contribution records, and BC
 * descriptors to discover how variables (FE fields, auxiliary states,
 * boundary functionals, global scalars) couple to each other.
 *
 * Emits CoupledSystemStructure claims for each discovered coupling edge.
 * Detects InterfaceCondition when variables couple only through
 * interface or global-domain operators.
 *
 * @see KernelContributionRecord for non-Forms operator metadata
 * @see BoundaryConditionDescriptor for BC-induced couplings
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class CouplingGraphAnalyzer : public AnalyzerPass {
public:
    CouplingGraphAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_COUPLING_GRAPH_ANALYZER_H
