/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_DISCRETE_MONOTONICITY_ANALYZER_H
#define SVMP_FE_ANALYSIS_DISCRETE_MONOTONICITY_ANALYZER_H

/**
 * @file DiscreteMonotonicityAnalyzer.h
 * @brief Consumes compact matrix-sign summaries for DMP/Z/M-matrix claims.
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class DiscreteMonotonicityAnalyzer : public AnalyzerPass {
public:
    DiscreteMonotonicityAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_DISCRETE_MONOTONICITY_ANALYZER_H
