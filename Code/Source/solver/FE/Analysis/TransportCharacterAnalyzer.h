/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_TRANSPORT_CHARACTER_ANALYZER_H
#define SVMP_FE_ANALYSIS_TRANSPORT_CHARACTER_ANALYZER_H

/**
 * @file TransportCharacterAnalyzer.h
 * @brief Transport / convection character analysis pass
 *
 * Reads contributions with transport_character populated or HasFirstOrder
 * trait to classify operator transport character per field.
 *
 * Detects directional first-order (convection-like) contributions,
 * transport-dominated regimes (more first-order than second-order
 * contributions), and unknown transport character.
 *
 * Emits PropertyKind::OperatorTransportCharacter claims.
 *
 * @see ContributionDescriptor for transport_character and OperatorTraitFlags
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

namespace svmp {
namespace FE {
namespace analysis {

class TransportCharacterAnalyzer : public AnalyzerPass {
public:
    TransportCharacterAnalyzer() = default;
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_TRANSPORT_CHARACTER_ANALYZER_H
