/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSERVATION_ANALYZER_H
#define SVMP_FE_ANALYSIS_CONSERVATION_ANALYZER_H

/**
 * @file ConservationAnalyzer.h
 * @brief Conservation / balance closure analysis pass
 *
 * Reads contributions with populated BalanceDescriptor metadata,
 * groups them by balance_group, and classifies the conservation
 * structure of each group (balanced exchange, local closure, broken).
 *
 * Graceful no-op when no BalanceDescriptors are populated.
 *
 * Emits PropertyKind::ConservationStructure claims.
 *
 * @see ContributionDescriptor for BalanceDescriptor metadata
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

class ConservationAnalyzer : public AnalyzerPass {
public:
    ConservationAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSERVATION_ANALYZER_H
