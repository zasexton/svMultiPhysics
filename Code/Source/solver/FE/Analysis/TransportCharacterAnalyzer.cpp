/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TransportCharacterAnalyzer.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void appendUniqueString(std::vector<std::string>& values,
                        const std::string& value)
{
    if (value.empty()) return;
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

void enrichTransportClaimFromSummaries(const AnalysisSummarySet* summaries,
                                       const OperatorBlockId& claim_block,
                                       PropertyClaim& claim)
{
    if (!summaries) return;

    bool added_numeric_evidence = false;

    for (const auto& scale : summaries->parameter_scales) {
        if (!parameterScaleMatches(scale, claim_block,
                                   ParameterScaleRole::PecletLike)) {
            continue;
        }
        if (!claim.peclet_number ||
            scale.max_scale_value > *claim.peclet_number) {
            claim.peclet_number = scale.max_scale_value;
        }
    }
    if (claim.peclet_number) {
        added_numeric_evidence = true;
        if (*claim.peclet_number > Real{1}) {
            claim.transport_character_class =
                TransportCharacterClass::TransportDominatedRisk;
            if (claim.status == PropertyStatus::Exact) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
            }
        }
    }

    for (const auto& temporal : summaries->temporal_stability) {
        if (!temporalSummaryMatches(temporal, claim_block)) {
            continue;
        }
        if (!temporal.cfl_estimate_present) {
            continue;
        }
        if (!claim.cfl_number || temporal.cfl_estimate > *claim.cfl_number) {
            claim.cfl_number = temporal.cfl_estimate;
        }
    }
    if (claim.cfl_number) {
        added_numeric_evidence = true;
        if (*claim.cfl_number > Real{1} &&
            claim.status != PropertyStatus::Violated) {
            claim.status = PropertyStatus::Likely;
            claim.confidence = AnalysisConfidence::Medium;
        }
    }

    for (const auto& matrix : summaries->discrete_matrices) {
        if (!blockEvidenceMatches(matrix.block, claim_block)) {
            continue;
        }
        if (!matrix.nonnormality_indicator) {
            continue;
        }
        const Real indicator = *matrix.nonnormality_indicator;
        if (!claim.nonnormality_indicator ||
            indicator > *claim.nonnormality_indicator) {
            claim.nonnormality_indicator = indicator;
        }
    }
    if (claim.nonnormality_indicator) {
        added_numeric_evidence = true;
        if (*claim.nonnormality_indicator > Real{1.0e-8}) {
            claim.transport_character_class =
                TransportCharacterClass::NonNormalRisk;
            if (claim.status == PropertyStatus::Exact) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
            }
        }
    }

    if (added_numeric_evidence) {
        claim.addEvidence("TransportCharacterAnalyzer",
            "Scoped numeric summary enrichment: Peclet-like dimensionless scale, "
            "CFL, and explicit nonnormality indicators matched this transport block",
            AnalysisConfidence::Medium);
    }
}

} // namespace

std::string TransportCharacterAnalyzer::name() const {
    return "TransportCharacterAnalyzer";
}

void TransportCharacterAnalyzer::run(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();
    if (contributions.empty()) return;
    const auto* summaries = context.analysisSummaries();

    // =====================================================================
    // Per-field transport character accumulation
    // =====================================================================

    struct FieldTransportInfo {
        VariableKey key;
        int first_order_count{0};
        int second_order_count{0};
        bool has_explicit_transport_character{false};
        TransportCharacter explicit_character{TransportCharacter::None};
        bool has_first_order_trait{false};
        bool has_second_order_trait{false};
        std::vector<std::string> operator_tags;
    };

    std::unordered_map<VariableKey, FieldTransportInfo, VariableKeyHash> field_info;

    for (const auto& contrib : contributions) {
        // Only analyze cell domain contributions (not BCs)
        if (contrib.domain != DomainKind::Cell) continue;

        for (const auto& tv : contrib.test_variables) {
            auto& info = field_info[tv];
            info.key = tv;
            appendUniqueString(info.operator_tags, contrib.operator_tag);

            // Check explicit transport_character
            if (contrib.transport_character.has_value()) {
                info.has_explicit_transport_character = true;
                const auto tc = *contrib.transport_character;
                // Keep the most specific character
                if (tc == TransportCharacter::DirectionalFirstOrder ||
                    tc == TransportCharacter::TransportDominatedRisk) {
                    info.explicit_character = tc;
                } else if (info.explicit_character == TransportCharacter::None) {
                    info.explicit_character = tc;
                }
            }

            // Check trait flags
            if (hasFlag(contrib.traits, OperatorTraitFlags::HasFirstOrder)) {
                info.has_first_order_trait = true;
                ++info.first_order_count;
            }
            if (hasFlag(contrib.traits, OperatorTraitFlags::HasSecondOrder)) {
                info.has_second_order_trait = true;
                ++info.second_order_count;
            }
        }
    }

    // =====================================================================
    // Emit claims per field
    // =====================================================================

    for (const auto& [var_key, info] : field_info) {
        // Skip fields with no first-order character at all
        if (!info.has_explicit_transport_character && !info.has_first_order_trait) {
            continue;
        }

        PropertyClaim claim;
        claim.kind = PropertyKind::OperatorTransportCharacter;
        claim.variables.push_back(var_key);
        claim.claim_origin = "TransportCharacterAnalyzer";

        if (var_key.kind == VariableKind::FieldComponent) {
            claim.field = var_key.field_id;
            claim.component = var_key.component;
        }

        if (info.has_explicit_transport_character) {
            // Use explicit transport character
            if (info.explicit_character == TransportCharacter::DirectionalFirstOrder) {
                claim.status = PropertyStatus::Exact;
                claim.confidence = AnalysisConfidence::High;
                claim.transport_character_class = TransportCharacterClass::DirectionalFirstOrderLike;
                claim.description =
                    "Directional first-order (convection-like) transport character";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "Explicit transport_character=DirectionalFirstOrder on contribution");
            } else if (info.explicit_character == TransportCharacter::TransportDominatedRisk) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.transport_character_class = TransportCharacterClass::TransportDominatedRisk;
                claim.description =
                    "Transport-dominated regime risk (explicit annotation)";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "Explicit transport_character=TransportDominatedRisk on contribution",
                    AnalysisConfidence::Medium);
            } else if (info.explicit_character == TransportCharacter::DiffusionLike) {
                claim.status = PropertyStatus::Exact;
                claim.confidence = AnalysisConfidence::High;
                claim.transport_character_class = TransportCharacterClass::DiffusionLike;
                claim.description =
                    "Diffusion-like transport character";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "Explicit transport_character=DiffusionLike on contribution");
            } else if (info.explicit_character == TransportCharacter::NonNormalLike) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Medium;
                claim.transport_character_class = TransportCharacterClass::NonNormalRisk;
                claim.description =
                    "Non-normal operator risk from transport character";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "Explicit transport_character=NonNormalLike on contribution",
                    AnalysisConfidence::Medium);
            } else {
                // TransportCharacter::None with explicit flag -- skip
                continue;
            }
        } else if (info.has_first_order_trait) {
            // No explicit transport_character but HasFirstOrder trait

            // Check ratio heuristic: more first-order than second-order
            if (info.has_second_order_trait &&
                info.first_order_count > info.second_order_count) {
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Low;
                claim.transport_character_class = TransportCharacterClass::TransportDominatedRisk;
                claim.description =
                    "Transport-dominated risk: " +
                    std::to_string(info.first_order_count) +
                    " first-order vs " +
                    std::to_string(info.second_order_count) +
                    " second-order contributions (ratio heuristic)";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "More HasFirstOrder than HasSecondOrder contributions",
                    AnalysisConfidence::Low);
            } else if (info.has_second_order_trait) {
                // Both present, second-order dominant or equal
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Low;
                claim.transport_character_class = TransportCharacterClass::DirectionalFirstOrderLike;
                claim.description =
                    "Has first-order (convection-like) contributions alongside"
                    " second-order terms (" +
                    std::to_string(info.first_order_count) + " first-order, " +
                    std::to_string(info.second_order_count) + " second-order)";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "HasFirstOrder trait detected without explicit transport_character",
                    AnalysisConfidence::Low);
            } else {
                // Only first-order, no second-order
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Low;
                claim.transport_character_class = TransportCharacterClass::Unknown;
                claim.description =
                    "First-order operator trait detected but no explicit transport"
                    " character annotation";
                claim.addEvidence("TransportCharacterAnalyzer",
                    "HasFirstOrder trait without transport_character metadata",
                    AnalysisConfidence::Low);
            }
        } else {
            continue;
        }

        OperatorBlockId claim_block;
        claim_block.test_variables = claim.variables;
        claim_block.trial_variables = claim.variables;
        claim_block.domain = DomainKind::Cell;
        if (info.operator_tags.size() == 1u) {
            claim_block.operator_tag = info.operator_tags.front();
            claim.tested_block_id = info.operator_tags.front();
        }
        enrichTransportClaimFromSummaries(summaries, claim_block, claim);
        report.claims.push_back(std::move(claim));
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
