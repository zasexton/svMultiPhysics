/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGraph.h"

#include <string>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

bool isRequired(CouplingRequirement requirement) noexcept
{
    return requirement == CouplingRequirement::Required;
}

void validateContextReferences(const CouplingContext& context,
                               const CouplingContractDeclaration& declaration,
                               CouplingValidationResult& result)
{
    for (const auto& participant : declaration.participants) {
        if (isRequired(participant.requirement) &&
            !context.hasParticipant(participant.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = participant.participant_name,
                .message = "required coupling participant is missing from the context",
            });
        }
    }

    for (const auto& field : declaration.fields) {
        if (isRequired(field.requirement) &&
            !context.hasField(field.participant_name, field.field_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "required coupling field is missing from the context",
            });
        }
    }

    for (const auto& region : declaration.regions) {
        if (!context.hasRegion(region.participant_name, region.region_name)) {
            if (isRequired(region.requirement)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = region.participant_name,
                    .region_name = region.region_name,
                    .message = "required coupling region is missing from the context",
                });
            }
            continue;
        }
        if (region.required_region_kind.has_value() &&
            context.region(region.participant_name, region.region_name).kind !=
                *region.required_region_kind) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region kind does not satisfy the declaration",
            });
        }
    }

    for (const auto& shared_region : declaration.shared_regions) {
        if (!context.hasSharedRegion(shared_region.shared_region_name)) {
            if (isRequired(shared_region.requirement)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .region_name = shared_region.shared_region_name,
                    .message = "required shared region is missing from the context",
                });
            }
            continue;
        }
        if (shared_region.required_region_kind.has_value()) {
            const auto group = context.sharedRegionGroup(shared_region.shared_region_name);
            for (const auto& participant_region : group.participant_regions) {
                if (participant_region.kind != *shared_region.required_region_kind) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .participant_name = participant_region.participant_name,
                        .region_name = shared_region.shared_region_name,
                        .message = "shared-region participant mapping does not satisfy the declaration",
                    });
                }
            }
        }
    }
}

} // namespace

CouplingValidationResult CouplingGraph::buildDeclarationGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations)
{
    declarations_.assign(declarations.begin(), declarations.end());

    CouplingValidationResult result;
    for (std::size_t i = 0; i < declarations_.size(); ++i) {
        const auto& declaration = declarations_[i];
        result.append(validateContractDeclarationShape(declaration));
        validateContextReferences(context, declaration, result);
        for (std::size_t j = i + 1u; j < declarations_.size(); ++j) {
            if (!declaration.contract_name.empty() &&
                declaration.contract_name == declarations_[j].contract_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message = "duplicate coupling contract instance name",
                });
            }
        }
    }
    return result;
}

const std::vector<CouplingContractDeclaration>& CouplingGraph::declarations() const noexcept
{
    return declarations_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
