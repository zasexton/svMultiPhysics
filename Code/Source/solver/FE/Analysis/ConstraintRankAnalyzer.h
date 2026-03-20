/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSTRAINT_RANK_ANALYZER_H
#define SVMP_FE_ANALYSIS_CONSTRAINT_RANK_ANALYZER_H

/**
 * @file ConstraintRankAnalyzer.h
 * @brief Under/over-constraint detection pass
 *
 * Reads KernelAnalyzer's Nullspace claims from the report, the
 * ConstraintAnalysisSummary from the context, and BC descriptors
 * to determine whether nullspace modes are properly anchored.
 *
 * Emits UnderConstraint claims for unanchored nullspace modes and
 * OverConstraint claims when constraint conflicts are detected.
 *
 * Must run after KernelAnalyzer.
 *
 * @see KernelAnalyzer for Nullspace claims
 * @see ConstraintAnalysisSummary for DOF constraint data
 * @see BoundaryConditionDescriptor for anchoring information
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class ConstraintRankAnalyzer : public AnalyzerPass {
public:
    ConstraintRankAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSTRAINT_RANK_ANALYZER_H
