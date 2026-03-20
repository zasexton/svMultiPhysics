/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_MIXED_OPERATOR_ANALYZER_H
#define SVMP_FE_ANALYSIS_MIXED_OPERATOR_ANALYZER_H

/**
 * @file MixedOperatorAnalyzer.h
 * @brief Saddle-point structure detection pass
 *
 * Iterates formulation records and uses FormStructureAnalyzer to check
 * for saddle-point (indefinite block) structure in the system.
 *
 * Emits MixedSaddlePoint claims for formulations exhibiting mixed structure.
 *
 * @see FormStructureAnalyzer for has_saddle_point_structure detection
 * @see FormulationRecord for is_mixed flag
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class MixedOperatorAnalyzer : public AnalyzerPass {
public:
    MixedOperatorAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_MIXED_OPERATOR_ANALYZER_H
