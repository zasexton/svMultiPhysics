/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/TransportCharacterAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <unordered_map>

namespace svmp {
namespace FE {
namespace analysis {

std::string TransportCharacterAnalyzer::name() const {
    return "TransportCharacterAnalyzer";
}

void TransportCharacterAnalyzer::run(const ProblemAnalysisContext& context,
                                     ProblemAnalysisReport& report) const
{
    const auto& contributions = context.contributions();
    if (contributions.empty()) return;

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
    };

    std::unordered_map<VariableKey, FieldTransportInfo, VariableKeyHash> field_info;

    for (const auto& contrib : contributions) {
        // Only analyze cell domain contributions (not BCs)
        if (contrib.domain != DomainKind::Cell) continue;

        for (const auto& tv : contrib.test_variables) {
            auto& info = field_info[tv];
            info.key = tv;

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

        report.claims.push_back(std::move(claim));
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
