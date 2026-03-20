/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_KERNEL_ANALYZER_H
#define SVMP_FE_ANALYSIS_KERNEL_ANALYZER_H

/**
 * @file KernelAnalyzer.h
 * @brief Nullspace detection pass for the FE analysis subsystem
 *
 * Iterates FormulationRecords and uses FormStructureAnalyzer to classify
 * how each field appears in the residual expression.  Maps the resulting
 * FieldOperatorSummary to Nullspace PropertyClaims using the same logic
 * as NullspaceAnalyzer::analyzeFromSummary().
 *
 * Does NOT interact with GaugeRegistry — emits pure PropertyClaims.
 *
 * @see FormStructureAnalyzer for the DAG walker
 * @see NullspaceAnalyzer for the original gauge-oriented classification
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class KernelAnalyzer : public AnalyzerPass {
public:
    KernelAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_KERNEL_ANALYZER_H
