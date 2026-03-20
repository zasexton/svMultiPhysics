/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_TOPOLOGY_SCOPE_ANALYZER_H
#define SVMP_FE_ANALYSIS_TOPOLOGY_SCOPE_ANALYZER_H

/**
 * @file TopologyScopeAnalyzer.h
 * @brief Per-region nullspace replication pass
 *
 * Reads TopologyAnalysisContext from the context and prior Nullspace
 * claims from the report.  For disconnected meshes, checks whether
 * each connected component has BC anchoring for each nullspace mode.
 *
 * Emits TopologyScopedKernel claims for regions where a nullspace
 * mode is present but unanchored by boundary conditions.
 *
 * Must run after KernelAnalyzer and (ideally) ConstraintRankAnalyzer.
 *
 * @see TopologyAnalysisContext for mesh connectivity data
 * @see BoundaryConditionDescriptor for per-marker anchoring info
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class TopologyScopeAnalyzer : public AnalyzerPass {
public:
    TopologyScopeAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_TOPOLOGY_SCOPE_ANALYZER_H
