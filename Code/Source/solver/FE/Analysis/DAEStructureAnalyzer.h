/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_DAE_STRUCTURE_ANALYZER_H
#define SVMP_FE_ANALYSIS_DAE_STRUCTURE_ANALYZER_H

/**
 * @file DAEStructureAnalyzer.h
 * @brief Differential-algebraic equation structure analysis pass
 *
 * Reads TemporalDescriptor from contributions and temporal_state_kind
 * from variable descriptors to classify the system as PureODELike,
 * AlgebraicSystem, Index1DAELike, or HigherIndexRisk.
 *
 * Graceful no-op when no temporal metadata is populated.
 *
 * Emits PropertyKind::DifferentialAlgebraicStructure claims.
 *
 * @see ContributionDescriptor for TemporalDescriptor metadata
 * @see VariableDescriptor for temporal_state_kind
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

class DAEStructureAnalyzer : public AnalyzerPass {
public:
    DAEStructureAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_DAE_STRUCTURE_ANALYZER_H
