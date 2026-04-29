/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_STABILIZATION_ANALYZER_H
#define SVMP_FE_ANALYSIS_STABILIZATION_ANALYZER_H

/**
 * @file StabilizationAnalyzer.h
 * @brief Stabilization detection pass
 *
 * Iterates formulation records to detect stabilization mechanisms
 * (SUPG, PSPG, GLS, penalty, etc.) via the has_stabilization_terms
 * flag and FormStructureAnalyzer's per-field stabilization detection.
 *
 * Emits exact detection claims for stabilization presence. Preserved/certified
 * adequacy requires StabilizationAdequacySummary metadata.
 *
 * @see FormulationRecord for has_stabilization_terms flag
 * @see FormStructureAnalyzer for per-field has_stabilization
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class StabilizationAnalyzer : public AnalyzerPass {
public:
    StabilizationAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_STABILIZATION_ANALYZER_H
