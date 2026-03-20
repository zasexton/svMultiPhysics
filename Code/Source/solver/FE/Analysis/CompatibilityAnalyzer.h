/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_COMPATIBILITY_ANALYZER_H
#define SVMP_FE_ANALYSIS_COMPATIBILITY_ANALYZER_H

/**
 * @file CompatibilityAnalyzer.h
 * @brief Solvability condition detection pass
 *
 * Reads KernelAnalyzer's Nullspace claims from the report and BC
 * descriptors from the context.  For fields with exact nullspace
 * modes and all-Neumann (flux-only) boundary conditions, emits
 * CompatibilityCondition claims indicating a solvability condition
 * on the right-hand side (e.g., integral of f must be zero for
 * pure Neumann Poisson).
 *
 * Must run after KernelAnalyzer.
 *
 * @see KernelAnalyzer for Nullspace claims
 * @see BoundaryConditionDescriptor for BC trace kinds
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class CompatibilityAnalyzer : public AnalyzerPass {
public:
    CompatibilityAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_COMPATIBILITY_ANALYZER_H
