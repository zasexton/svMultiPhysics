/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/MonolithicCouplingBuilder.h"

#include "Coupling/CouplingGraph.h"
#include "Core/FEException.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

struct AdditionalFieldTarget {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
};

std::string additionalFieldSystemName(const CouplingAdditionalFieldDeclaration& field)
{
    return field.namespace_name + "." + field.field_name;
}

AdditionalFieldTarget explicitAdditionalFieldTarget(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    if (!field.system_participant_name.empty()) {
        const auto participant = context.participant(field.system_participant_name);
        return AdditionalFieldTarget{
            .participant_name = participant.participant_name,
            .system_name = participant.system_name,
            .system = participant.system,
        };
    }

    if (field.field_namespace == CouplingAdditionalFieldNamespace::Participant) {
        const auto participant = context.participant(field.namespace_name);
        return AdditionalFieldTarget{
            .participant_name = participant.participant_name,
            .system_name = participant.system_name,
            .system = participant.system,
        };
    }

    FE_THROW(InvalidArgumentException,
             "contract-owned additional field requires a target participant or shared-region system");
}

AdditionalFieldTarget sharedRegionAdditionalFieldTarget(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    if (!field.shared_region_name.has_value() ||
        field.field_namespace != CouplingAdditionalFieldNamespace::Contract ||
        !field.system_participant_name.empty()) {
        return explicitAdditionalFieldTarget(context, field);
    }

    const auto group = context.sharedRegionGroup(*field.shared_region_name);
    FE_THROW_IF(group.participant_regions.empty(), InvalidArgumentException,
                "contract-owned interface field shared region has no participants");

    const auto& first = group.participant_regions.front();
    FE_CHECK_NOT_NULL(first.system,
                      "contract-owned interface field shared-region system");
    for (const auto& region : group.participant_regions) {
        FE_THROW_IF(region.system != first.system, InvalidArgumentException,
                    "contract-owned interface field requires shared-region participants in one system");
    }
    return AdditionalFieldTarget{
        .participant_name = first.participant_name,
        .system_name = first.system_name,
        .system = first.system,
    };
}

int interfaceMarkerForAdditionalField(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field,
    const AdditionalFieldTarget& target)
{
    if (field.region_name.has_value()) {
        const auto region = context.region(target.participant_name, *field.region_name);
        FE_THROW_IF(region.kind != CouplingRegionKind::InterfaceFace,
                    InvalidArgumentException,
                    "interface additional field participant region must be an interface face");
        FE_THROW_IF(region.marker < 0, InvalidArgumentException,
                    "interface additional field participant region requires a marker");
        return region.marker;
    }

    FE_THROW_IF(!field.shared_region_name.has_value(), InvalidArgumentException,
                "interface additional field requires a shared region or participant region");
    const auto group = context.sharedRegionGroup(*field.shared_region_name);
    std::optional<int> marker;
    for (const auto& region : group.participant_regions) {
        if (region.system != target.system) {
            continue;
        }
        FE_THROW_IF(region.kind != CouplingRegionKind::InterfaceFace,
                    InvalidArgumentException,
                    "interface additional field shared region must map to interface faces");
        FE_THROW_IF(region.marker < 0, InvalidArgumentException,
                    "interface additional field shared region requires markers");
        if (!marker.has_value()) {
            marker = region.marker;
        } else {
            FE_THROW_IF(*marker != region.marker, InvalidArgumentException,
                        "interface additional field shared-region markers must agree in one system");
        }
    }
    FE_THROW_IF(!marker.has_value(), InvalidArgumentException,
                "interface additional field target participant is not in the shared region");
    return *marker;
}

systems::FieldSpec fieldSpecForAdditionalField(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field,
    const AdditionalFieldTarget& target)
{
    FE_CHECK_NOT_NULL(field.space.get(), "additional field function space");
    systems::FieldSpec spec;
    spec.name = additionalFieldSystemName(field);
    spec.space = field.space;
    spec.components = field.components == 0
        ? field.space->value_dimension()
        : field.components;
    if (field.scope == CouplingAdditionalFieldScope::InterfaceFace) {
        spec.scope = systems::FieldScope::InterfaceFace;
        spec.interface_marker = interfaceMarkerForAdditionalField(context, field, target);
    }
    return spec;
}

systems::FESystem& mutableSystemByName(const CouplingContext& context,
                                       std::string_view system_name)
{
    const auto it = std::find_if(
        context.participants().begin(),
        context.participants().end(),
        [&](const CouplingParticipantRef& participant) {
            return participant.system_name == system_name;
        });
    FE_THROW_IF(it == context.participants().end(), InvalidArgumentException,
                "additional field target system is missing from context");
    FE_CHECK_NOT_NULL(it->system, "additional field target system");
    return const_cast<systems::FESystem&>(*it->system);
}

} // namespace

CouplingValidationResult MonolithicCouplingBuilder::validateDeclarations(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations) const
{
    CouplingGraph graph;
    return graph.buildDeclarationGraph(context, declarations);
}

std::vector<ResolvedCouplingAdditionalFieldDeclaration>
MonolithicCouplingBuilder::resolveAdditionalFields(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations) const
{
    std::vector<ResolvedCouplingAdditionalFieldDeclaration> resolved;
    for (const auto& declaration : declarations) {
        for (const auto& field : declaration.additional_fields) {
            const auto target = sharedRegionAdditionalFieldTarget(context, field);
            resolved.push_back(ResolvedCouplingAdditionalFieldDeclaration{
                .declaration = field,
                .system_name = target.system_name,
                .field_spec = fieldSpecForAdditionalField(context, field, target),
            });
        }
    }
    return resolved;
}

std::vector<ResolvedCouplingAdditionalFieldDeclaration>
MonolithicCouplingBuilder::registerAdditionalFields(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations) const
{
    auto resolved = resolveAdditionalFields(context, declarations);
    for (auto& field : resolved) {
        auto& system = mutableSystemByName(context, field.system_name);
        field.field_id = system.addField(field.field_spec);
    }
    return resolved;
}

ResolvedCouplingFormContribution MonolithicCouplingBuilder::resolveFormContribution(
    const CouplingContext& context,
    const CouplingFormContribution& contribution) const
{
    FE_THROW_IF(contribution.contribution_name.empty(), InvalidArgumentException,
                "coupling form contribution requires a contribution name");
    FE_THROW_IF(contribution.origin.empty(), InvalidArgumentException,
                "coupling form contribution requires a diagnostic origin");
    FE_THROW_IF(contribution.field_uses.empty(), InvalidArgumentException,
                "coupling form contribution requires at least one primary field use");
    FE_THROW_IF(!contribution.residual.isValid(), InvalidArgumentException,
                "coupling form contribution requires a residual expression");

    ResolvedCouplingFormContribution resolved;
    resolved.contribution_name = contribution.contribution_name;
    resolved.origin = contribution.origin;
    resolved.operator_name = contribution.operator_name;
    resolved.install_options = contribution.install_options;
    resolved.terminal_provenance = contribution.terminal_provenance;

    const systems::FESystem* owning_system = nullptr;
    auto resolve_field = [&](const CouplingFieldUse& use) {
        const auto ref = context.field(use.participant_name, use.field_name);
        if (owning_system == nullptr) {
            owning_system = ref.system;
            resolved.system_name = ref.system_name;
        }
        FE_THROW_IF(ref.system != owning_system, InvalidArgumentException,
                    "monolithic coupling form fields must resolve to one owning system");
        return ref.field_id;
    };

    for (const auto& use : contribution.field_uses) {
        const auto field = resolve_field(use);
        FE_THROW_IF(std::find(resolved.fields.begin(), resolved.fields.end(), field) !=
                        resolved.fields.end(),
                    InvalidArgumentException,
                    "duplicate primary field use in coupling form contribution");
        resolved.fields.push_back(field);
    }

    resolved.install_options.extra_trial_fields.clear();
    for (const auto& use : contribution.extra_trial_field_uses) {
        const auto field = resolve_field(use);
        FE_THROW_IF(std::find(resolved.fields.begin(), resolved.fields.end(), field) !=
                        resolved.fields.end(),
                    InvalidArgumentException,
                    "extra trial field overlaps a primary coupling form field");
        FE_THROW_IF(std::find(resolved.extra_trial_fields.begin(),
                              resolved.extra_trial_fields.end(),
                              field) != resolved.extra_trial_fields.end(),
                    InvalidArgumentException,
                    "duplicate extra trial field use in coupling form contribution");
        resolved.extra_trial_fields.push_back(field);
        resolved.install_options.extra_trial_fields.push_back(field);
    }

    resolved.residual = contribution.residual;
    return resolved;
}

CouplingFormAnalysisMetadata MonolithicCouplingBuilder::installResolvedFormContribution(
    systems::FESystem& system,
    const ResolvedCouplingFormContribution& contribution) const
{
    analysis::FormAnalysisBridgeOptions bridge_options;
    bridge_options.contribution_name = contribution.contribution_name;
    bridge_options.origin = contribution.origin;
    bridge_options.system_name = contribution.system_name;
    bridge_options.geometry_sensitivity =
        contribution.install_options.compiler_options.geometry_sensitivity;

    const auto installed = systems::installFormulationWithMetadata(
        system,
        contribution.operator_name,
        contribution.fields,
        contribution.residual,
        contribution.install_options,
        bridge_options);

    auto adapted = adaptFormAnalysisMetadata(installed.analysis);
    adapted.declaration_terminal_provenance = contribution.terminal_provenance;
    return adapted;
}

std::vector<CouplingFormAnalysisMetadata> MonolithicCouplingBuilder::installFormContributions(
    systems::FESystem& system,
    const CouplingContext& context,
    std::span<const CouplingFormContribution> contributions) const
{
    std::vector<CouplingFormAnalysisMetadata> installed;
    installed.reserve(contributions.size());
    for (const auto& contribution : contributions) {
        const auto resolved = resolveFormContribution(context, contribution);
        installed.push_back(installResolvedFormContribution(system, resolved));
    }
    return installed;
}

CouplingFormAnalysisMetadata MonolithicCouplingBuilder::adaptFormAnalysisMetadata(
    const analysis::FormContributionAnalysisMetadata& metadata)
{
    CouplingFormAnalysisMetadata adapted;
    adapted.contribution_name = metadata.contribution_name;
    adapted.origin = metadata.origin;
    adapted.system_name = metadata.system_name;
    adapted.operator_name = metadata.operator_tag;
    adapted.installed_fields = metadata.installed_fields;
    adapted.geometry_sensitivity = metadata.geometry_sensitivity;
    adapted.feature_gates = metadata.feature_gates;

    adapted.installed_dependencies.reserve(metadata.installed_dependencies.size());
    for (const auto& dependency : metadata.installed_dependencies) {
        adapted.installed_dependencies.push_back(CouplingInstalledDependency{
            .residual_row = dependency.residual_row,
            .dependency = dependency.dependency,
            .mode = CouplingDependencyMode::ImplicitMonolithic,
            .domain = dependency.domain,
            .contributes_matrix_block = dependency.contributes_matrix_block,
            .contributes_vector = dependency.contributes_vector,
            .provider = dependency.provider,
        });
    }

    adapted.installed_blocks.reserve(metadata.installed_blocks.size());
    for (const auto& block : metadata.installed_blocks) {
        adapted.installed_blocks.push_back(CouplingInstalledBlockProvenance{
            .residual_row = block.residual_row,
            .dependency = block.dependency,
            .domains = block.domains,
            .has_matrix = block.has_matrix,
            .has_vector = block.has_vector,
        });
    }

    return adapted;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
