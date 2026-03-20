/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_INF_SUP_ANALYZER_H
#define SVMP_FE_ANALYSIS_INF_SUP_ANALYZER_H

/**
 * @file InfSupAnalyzer.h
 * @brief Inf-sup (LBB) stability condition analysis pass
 *
 * Reads PairingDescriptor entries from contributions looking for
 * ConstraintPair or FormalAdjointPair pairings, and checks prior
 * MixedSaddlePoint claims to classify the inf-sup condition status.
 *
 * If a StabilizedConstraintPair is found, emits StabilizedSurrogate.
 * If field descriptors show compatible space families (different polynomial
 * orders), emits StructurallySupported.  Otherwise emits Required with
 * appropriate confidence.
 *
 * Emits PropertyKind::InfSupCondition claims.
 *
 * @see MixedOperatorAnalyzer for MixedSaddlePoint claims consumed here
 * @see ContributionDescriptor for PairingDescriptor metadata
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

class InfSupAnalyzer : public AnalyzerPass {
public:
    InfSupAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_INF_SUP_ANALYZER_H
