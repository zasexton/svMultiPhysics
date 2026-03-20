/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_OPERATOR_CLASS_ANALYZER_H
#define SVMP_FE_ANALYSIS_OPERATOR_CLASS_ANALYZER_H

/**
 * @file OperatorClassAnalyzer.h
 * @brief Symmetry and definiteness classification pass
 *
 * Iterates formulation records and examines single-field formulations
 * to classify the bilinear form as symmetric/non-symmetric and
 * positive-(semi)definite/indefinite.
 *
 * Emits OperatorSymmetry and OperatorDefiniteness claims.
 *
 * @see FormStructureAnalyzer for per-field operator summaries
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class OperatorClassAnalyzer : public AnalyzerPass {
public:
    OperatorClassAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_OPERATOR_CLASS_ANALYZER_H
