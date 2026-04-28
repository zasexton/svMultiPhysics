/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_NUMERIC_SUMMARY_PLANNER_H
#define SVMP_FE_ANALYSIS_NUMERIC_SUMMARY_PLANNER_H

/**
 * @file NumericSummaryPlanner.h
 * @brief Request planner for numeric/discrete summaries required by symbolic claims.
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Converts symbolic analyzer output and context metadata into summary requests.
 *
 * This pass runs after the symbolic analyzers. It does not compute numeric
 * evidence; it records which compact summaries are needed and why, so mesh,
 * assembly, and backend owners can avoid unconditional diagnostic storage.
 */
class NumericSummaryPlanner : public AnalyzerPass {
public:
    NumericSummaryPlanner() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_NUMERIC_SUMMARY_PLANNER_H
