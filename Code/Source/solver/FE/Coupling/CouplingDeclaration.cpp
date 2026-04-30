/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDeclaration.h"

#include "Spaces/FunctionSpace.h"

#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

std::optional<analysis::VariableKind> analysisVariableKindForNonFieldRequirement(
    CouplingNonFieldDependencyRequirementKind kind) noexcept
{
    switch (kind) {
    case CouplingNonFieldDependencyRequirementKind::BoundaryFunctional:
    case CouplingNonFieldDependencyRequirementKind::BoundaryIntegral:
        return analysis::VariableKind::BoundaryFunctional;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryState:
        return analysis::VariableKind::AuxiliaryState;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryInput:
        return analysis::VariableKind::AuxiliaryInput;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput:
        return analysis::VariableKind::AuxiliaryOutput;
    case CouplingNonFieldDependencyRequirementKind::Parameter:
    case CouplingNonFieldDependencyRequirementKind::Coefficient:
    case CouplingNonFieldDependencyRequirementKind::MaterialStateOld:
    case CouplingNonFieldDependencyRequirementKind::MaterialStateWork:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<analysis::VariableKind> analysisVariableKindForFormNonFieldDependency(
    CouplingFormNonFieldDependencyKind kind) noexcept
{
    switch (kind) {
    case CouplingFormNonFieldDependencyKind::BoundaryFunctional:
    case CouplingFormNonFieldDependencyKind::BoundaryIntegral:
        return analysis::VariableKind::BoundaryFunctional;
    case CouplingFormNonFieldDependencyKind::AuxiliaryState:
        return analysis::VariableKind::AuxiliaryState;
    case CouplingFormNonFieldDependencyKind::AuxiliaryInput:
        return analysis::VariableKind::AuxiliaryInput;
    case CouplingFormNonFieldDependencyKind::AuxiliaryOutput:
        return analysis::VariableKind::AuxiliaryOutput;
    case CouplingFormNonFieldDependencyKind::Parameter:
    case CouplingFormNonFieldDependencyKind::Coefficient:
    case CouplingFormNonFieldDependencyKind::MaterialStateOld:
    case CouplingFormNonFieldDependencyKind::MaterialStateWork:
        return std::nullopt;
    }
    return std::nullopt;
}

namespace {

void addDuplicateIfRepeated(CouplingValidationResult& result,
                            std::string_view first,
                            std::string_view second,
                            std::string message)
{
    if (!first.empty() && first == second) {
        result.addError(std::move(message));
    }
}

std::string fieldKey(const CouplingFieldUse& field)
{
    return field.participant_name + "/" + field.field_name;
}

std::string fieldRequirementKey(const CouplingFieldRequirement& requirement)
{
    return fieldKey(requirement.field);
}

std::string regionKey(const CouplingRegionUse& region)
{
    return region.participant_name + "/" + region.region_name;
}

std::string regionEndpointKey(const CouplingRegionEndpointDeclaration& endpoint)
{
    return endpoint.participant_name + "/" + endpoint.region_name + "/" +
           endpoint.shared_region_name.value_or("");
}

std::string variableKey(const CouplingVariableUse& variable)
{
    return std::to_string(static_cast<int>(variable.kind)) + "/" +
           variable.participant_name + "/" + variable.name + "/" +
           std::to_string(variable.component);
}

std::string additionalFieldKey(const CouplingAdditionalFieldDeclaration& field)
{
    return std::to_string(static_cast<int>(field.field_namespace)) + "/" +
           field.namespace_name + "/" + field.system_participant_name + "/" +
           field.field_name + "/" + std::to_string(static_cast<int>(field.scope)) +
           "/" + field.region_name.value_or("") + "/" +
           field.shared_region_name.value_or("");
}

bool additionalFieldSelected(const CouplingAdditionalFieldDeclaration& field)
{
    return field.requirement == CouplingRequirement::Required || field.enabled;
}

bool variableReferencesAdditionalField(const CouplingAdditionalFieldDeclaration& field,
                                       const CouplingVariableUse& variable)
{
    return variable.kind == CouplingVariableKind::Field &&
           variable.participant_name == field.namespace_name &&
           variable.name == field.field_name;
}

bool fieldUseReferencesAdditionalField(
    const CouplingAdditionalFieldDeclaration& field,
    const CouplingFieldUse& use)
{
    return use.participant_name == field.namespace_name &&
           use.field_name == field.field_name;
}

bool declarationReferencesAdditionalField(
    const CouplingContractDeclaration& declaration,
    const CouplingAdditionalFieldDeclaration& field)
{
    for (const auto& dependency : declaration.dependencies) {
        if (variableReferencesAdditionalField(field, dependency.residual_row) ||
            variableReferencesAdditionalField(field, dependency.dependency)) {
            return true;
        }
    }
    for (const auto& block : declaration.expected_blocks) {
        if (variableReferencesAdditionalField(field, block.residual_row) ||
            variableReferencesAdditionalField(field, block.dependency)) {
            return true;
        }
    }
    return false;
}

bool contributionReferencesAdditionalField(
    const CouplingFormContribution& contribution,
    const CouplingAdditionalFieldDeclaration& field)
{
    for (const auto& use : contribution.field_uses) {
        if (fieldUseReferencesAdditionalField(field, use)) {
            return true;
        }
    }
    for (const auto& use : contribution.extra_trial_field_uses) {
        if (fieldUseReferencesAdditionalField(field, use)) {
            return true;
        }
    }
    for (const auto& terminal : contribution.terminal_provenance) {
        if (terminal.field.has_value() &&
            fieldUseReferencesAdditionalField(field, *terminal.field)) {
            return true;
        }
    }
    if (contribution.install_options_declaration.geometry_sensitivity.has_value()) {
        const auto& geometry =
            *contribution.install_options_declaration.geometry_sensitivity;
        return geometry.mesh_motion_field.has_value() &&
               fieldUseReferencesAdditionalField(field,
                                                 *geometry.mesh_motion_field);
    }
    return false;
}

std::string nonFieldDependencyKey(const CouplingNonFieldDependencyRequirement& dependency)
{
    std::string key =
        std::to_string(static_cast<int>(dependency.kind)) + "/" +
        dependency.participant_name + "/" + dependency.name + "/" +
        std::to_string(static_cast<int>(dependency.requirement)) + "/" +
        std::to_string(dependency.require_analysis_variable_key);
    if (dependency.region.has_value()) {
        key += "/" + dependency.region->participant_name + "/" +
               dependency.region->region_name + "/" +
               dependency.region->shared_region_name.value_or("");
    }
    key += "/" + std::to_string(
        dependency.required_region_kind.has_value()
            ? static_cast<int>(*dependency.required_region_kind)
            : -1);
    key += "/" + std::to_string(
        dependency.expected_parameter_value_type.has_value()
            ? static_cast<int>(*dependency.expected_parameter_value_type)
            : -1);
    key += "/" + dependency.expected_value_type;
    key += dependency.material_state_byte_offset.has_value()
               ? "/1:" + std::to_string(*dependency.material_state_byte_offset)
               : "/0:";
    return key;
}

bool isMaterialStateRequirement(CouplingNonFieldDependencyRequirementKind kind)
{
    return kind == CouplingNonFieldDependencyRequirementKind::MaterialStateOld ||
           kind == CouplingNonFieldDependencyRequirementKind::MaterialStateWork;
}

bool supportsAnalysisVariableKey(CouplingNonFieldDependencyRequirementKind kind)
{
    return analysisVariableKindForNonFieldRequirement(kind).has_value();
}

} // namespace

CouplingValidationResult validateContractDeclarationShape(
    const CouplingContractDeclaration& declaration)
{
    CouplingValidationResult result;
    if (declaration.contract_type.empty()) {
        result.addError("coupling contract declaration requires a contract type");
    }
    if (declaration.contract_name.empty()) {
        result.addError("coupling contract declaration requires a configured contract name");
    }

    for (std::size_t i = 0; i < declaration.participants.size(); ++i) {
        const auto& participant = declaration.participants[i];
        if (participant.participant_name.empty()) {
            result.addError("participant requirement requires a participant name");
        }
        for (std::size_t j = i + 1u; j < declaration.participants.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   participant.participant_name,
                                   declaration.participants[j].participant_name,
                                   "duplicate participant requirement");
        }
    }

    for (std::size_t i = 0; i < declaration.fields.size(); ++i) {
        const auto& field = declaration.fields[i];
        if (field.participant_name.empty() || field.field_name.empty()) {
            result.addError("field requirement requires participant and field names");
        }
        for (std::size_t j = i + 1u; j < declaration.fields.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   fieldKey(field),
                                   fieldKey(declaration.fields[j]),
                                   "duplicate field requirement");
        }
    }

    for (std::size_t i = 0; i < declaration.field_requirements.size(); ++i) {
        const auto& requirement = declaration.field_requirements[i];
        const auto& field = requirement.field;
        if (field.participant_name.empty() || field.field_name.empty()) {
            result.addError("field-shape requirement requires participant and field names");
        }
        result.append(validateCouplingValueDescriptor(requirement.value));
        for (std::size_t j = i + 1u;
             j < declaration.field_requirements.size();
             ++j) {
            addDuplicateIfRepeated(
                result,
                fieldRequirementKey(requirement),
                fieldRequirementKey(declaration.field_requirements[j]),
                "duplicate field-shape requirement");
        }
    }

    for (std::size_t i = 0; i < declaration.regions.size(); ++i) {
        const auto& region = declaration.regions[i];
        if (region.participant_name.empty() || region.region_name.empty()) {
            result.addError("participant-local region requirement requires participant and region names");
        }
        for (std::size_t j = i + 1u; j < declaration.regions.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   regionKey(region),
                                   regionKey(declaration.regions[j]),
                                   "duplicate participant-local region requirement");
        }
    }

    for (std::size_t i = 0; i < declaration.shared_regions.size(); ++i) {
        const auto& shared_region = declaration.shared_regions[i];
        if (shared_region.shared_region_name.empty()) {
            result.addError("shared-region requirement requires a shared-region name");
        }
        for (std::size_t j = i + 1u; j < declaration.shared_regions.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   shared_region.shared_region_name,
                                   declaration.shared_regions[j].shared_region_name,
                                   "duplicate shared-region requirement");
        }
    }

    for (std::size_t i = 0;
         i < declaration.shared_interface_requirements.size();
         ++i) {
        const auto& requirement = declaration.shared_interface_requirements[i];
        if (requirement.shared_region_name.empty()) {
            result.addError("shared-interface requirement requires a shared-region name");
        }
        if (requirement.participant_names.empty()) {
            result.addError("shared-interface requirement requires participant names");
        }
        if (requirement.required_region_kind != CouplingRegionKind::InterfaceFace) {
            result.addError(
                "shared-interface requirement requires interface-face region kind");
        }
        for (std::size_t j = i + 1u;
             j < declaration.shared_interface_requirements.size();
             ++j) {
            addDuplicateIfRepeated(
                result,
                requirement.shared_region_name,
                declaration.shared_interface_requirements[j].shared_region_name,
                "duplicate shared-interface requirement");
        }
        for (std::size_t j = 0; j < requirement.participant_names.size(); ++j) {
            if (requirement.participant_names[j].empty()) {
                result.addError(
                    "shared-interface requirement requires nonempty participant names");
            }
            for (std::size_t k = j + 1u;
                 k < requirement.participant_names.size();
                 ++k) {
                addDuplicateIfRepeated(
                    result,
                    requirement.participant_names[j],
                    requirement.participant_names[k],
                    "duplicate participant in shared-interface requirement");
            }
        }
    }

    for (std::size_t i = 0;
         i < declaration.region_relation_requirements.size();
         ++i) {
        const auto& requirement = declaration.region_relation_requirements[i];
        if (requirement.relation_name.empty()) {
            result.addError("region-relation requirement requires a name");
        }
        if (requirement.require_all_endpoints && requirement.endpoints.empty()) {
            result.addError("region-relation requirement requires endpoints");
        }
        if (requirement.relation_kind ==
                CouplingRegionRelationKind::SidePairedInterface &&
            requirement.endpoints.size() != 2u) {
            result.addError(
                "side-paired interface relation requires exactly two endpoints");
        }
        if (requirement.relation_kind == CouplingRegionRelationKind::NWayInterface &&
            requirement.endpoints.size() < 2u) {
            result.addError("N-way interface relation requires at least two endpoints");
        }
        if (requirement.require_opposite_sides_for_side_pair &&
            requirement.required_region_kind.has_value() &&
            *requirement.required_region_kind != CouplingRegionKind::InterfaceFace) {
            result.addError(
                "opposite-side relation requires interface-face region kind");
        }
        for (std::size_t j = i + 1u;
             j < declaration.region_relation_requirements.size();
             ++j) {
            addDuplicateIfRepeated(
                result,
                requirement.relation_name,
                declaration.region_relation_requirements[j].relation_name,
                "duplicate region-relation requirement");
        }
        for (std::size_t j = 0; j < requirement.endpoints.size(); ++j) {
            const auto& endpoint = requirement.endpoints[j];
            if (endpoint.participant_name.empty() || endpoint.region_name.empty()) {
                result.addError(
                    "region-relation endpoint requires participant and region names");
            }
            if (endpoint.shared_region_name.has_value() &&
                endpoint.shared_region_name->empty()) {
                result.addError("region-relation endpoint shared-region name cannot be empty");
            }
            for (std::size_t k = j + 1u;
                 k < requirement.endpoints.size();
                 ++k) {
                const auto& other = requirement.endpoints[k];
                addDuplicateIfRepeated(
                    result,
                    regionEndpointKey(endpoint),
                    regionEndpointKey(other),
                    "duplicate endpoint in region-relation requirement");
                if (requirement.require_distinct_participants) {
                    addDuplicateIfRepeated(
                        result,
                        endpoint.participant_name,
                        other.participant_name,
                        "region-relation requirement requires distinct participants");
                }
            }
        }
    }

    for (std::size_t i = 0; i < declaration.additional_fields.size(); ++i) {
        const auto& field = declaration.additional_fields[i];
        if (field.namespace_name.empty()) {
            result.addError("additional field declaration requires a namespace name");
        }
        if (field.field_name.empty()) {
            result.addError("additional field declaration requires a field name");
        }
        if (!field.enabled && field.requirement == CouplingRequirement::Required) {
            result.addError("required additional fields cannot be disabled");
        }
        if (!additionalFieldSelected(field)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Warning,
                .contract_name = declaration.contract_name,
                .field_name = field.field_name,
                .message = "disabled optional additional field is skipped",
            });
            if (declarationReferencesAdditionalField(declaration, field)) {
                result.addError("disabled optional additional field is referenced by a dependency or expected block");
            }
            for (std::size_t j = i + 1u; j < declaration.additional_fields.size(); ++j) {
                addDuplicateIfRepeated(result,
                                       additionalFieldKey(field),
                                       additionalFieldKey(declaration.additional_fields[j]),
                                       "duplicate additional field declaration");
            }
            continue;
        }
        if (field.space == nullptr) {
            result.addError("additional field declaration requires a function space");
        }
        if (field.components < 0) {
            result.addError("additional field component count must be nonnegative");
        }
        if (field.space != nullptr && field.components > 0 &&
            field.components != field.space->value_dimension()) {
            result.addError(
                "additional field component count must match the function space");
        }
        const bool has_region = field.region_name.has_value();
        const bool has_shared_region = field.shared_region_name.has_value();
        if (field.scope == CouplingAdditionalFieldScope::VolumeCell &&
            (has_region || has_shared_region)) {
            result.addError("volume additional fields must not declare region attachments");
        }
        if (field.scope == CouplingAdditionalFieldScope::InterfaceFace &&
            has_region == has_shared_region) {
            result.addError("interface additional fields require exactly one region attachment");
        }
        for (std::size_t j = i + 1u; j < declaration.additional_fields.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   additionalFieldKey(field),
                                   additionalFieldKey(declaration.additional_fields[j]),
                                   "duplicate additional field declaration");
        }
    }

    for (std::size_t i = 0; i < declaration.non_field_dependencies.size(); ++i) {
        const auto& dependency = declaration.non_field_dependencies[i];
        if (dependency.participant_name.empty()) {
            result.addError("non-field dependency requirement requires a participant name");
        }
        if (dependency.name.empty()) {
            result.addError("non-field dependency requirement requires a name");
        }
        if (dependency.region.has_value()) {
            const auto& region = *dependency.region;
            if (region.region_name.empty()) {
                result.addError("non-field dependency region scope requires a region name");
            }
            if (region.shared_region_name.has_value() &&
                region.shared_region_name->empty()) {
                result.addError("non-field dependency shared-region scope requires a name");
            }
            if (!region.participant_name.empty() &&
                !dependency.participant_name.empty() &&
                region.participant_name != dependency.participant_name) {
                result.addError("non-field dependency region scope must match the participant");
            }
        }
        if (dependency.expected_parameter_value_type.has_value() &&
            dependency.kind != CouplingNonFieldDependencyRequirementKind::Parameter) {
            result.addError("expected parameter value type is only valid for parameter dependencies");
        }
        if (dependency.material_state_byte_offset.has_value() &&
            !isMaterialStateRequirement(dependency.kind)) {
            result.addError("material-state byte offset is only valid for material-state dependencies");
        }
        if (dependency.require_analysis_variable_key &&
            !supportsAnalysisVariableKey(dependency.kind)) {
            result.addError("non-field dependency kind cannot require analysis variable identity");
        }
        for (std::size_t j = i + 1u; j < declaration.non_field_dependencies.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   nonFieldDependencyKey(dependency),
                                   nonFieldDependencyKey(declaration.non_field_dependencies[j]),
                                   "duplicate non-field dependency requirement");
        }
    }

    for (std::size_t i = 0; i < declaration.dependencies.size(); ++i) {
        const auto& dependency = declaration.dependencies[i];
        if (dependency.residual_row.name.empty() || dependency.dependency.name.empty()) {
            result.addError("residual dependency requires residual and dependency variable names");
        }
        for (std::size_t j = i + 1u; j < declaration.dependencies.size(); ++j) {
            const auto& other = declaration.dependencies[j];
            const bool same_edge =
                variableKey(dependency.residual_row) == variableKey(other.residual_row) &&
                variableKey(dependency.dependency) == variableKey(other.dependency) &&
                dependency.mode == other.mode;
            if (same_edge) {
                result.addError("duplicate residual dependency declaration");
            }
        }
    }

    for (const auto& temporal : declaration.temporal_requirements) {
        result.append(validateTemporalRequirement(temporal));
    }

    for (const auto& exchange : declaration.partitioned_exchange_declarations) {
        result.append(validateCouplingPortId(exchange.producer_port));
        result.append(validateCouplingPortId(exchange.consumer_port));
        result.append(validateCouplingValueDescriptor(exchange.value));
        if (exchange.producer.has_value()) {
            result.append(validateCouplingEndpointRef(*exchange.producer));
        }
        if (exchange.consumer.has_value()) {
            result.append(validateCouplingEndpointRef(*exchange.consumer));
        }
    }

    for (std::size_t i = 0; i < declaration.group_hints.size(); ++i) {
        const auto& hint = declaration.group_hints[i];
        if (hint.name.empty()) {
            result.addError("coupling group hint requires a name");
        }
        for (std::size_t j = i + 1u; j < declaration.group_hints.size(); ++j) {
            addDuplicateIfRepeated(result,
                                   hint.name,
                                   declaration.group_hints[j].name,
                                   "duplicate coupling group hint");
        }
        for (std::size_t j = 0; j < hint.participant_names.size(); ++j) {
            if (hint.participant_names[j].empty()) {
                result.addError("coupling group hint requires nonempty participant names");
            }
            for (std::size_t k = j + 1u; k < hint.participant_names.size(); ++k) {
                addDuplicateIfRepeated(result,
                                       hint.participant_names[j],
                                       hint.participant_names[k],
                                       "duplicate participant in coupling group hint");
            }
        }
    }

    return result;
}

CouplingValidationResult validateFormContributionDeclarations(
    std::span<const CouplingFormContribution> contributions)
{
    CouplingValidationResult result;
    for (std::size_t i = 0; i < contributions.size(); ++i) {
        const auto& contribution = contributions[i];
        if (contribution.contribution_name.empty()) {
            result.addError("coupling form contribution requires a contribution name");
        }
        for (std::size_t j = i + 1u; j < contributions.size(); ++j) {
            if (!contribution.contribution_name.empty() &&
                contribution.contribution_name == contributions[j].contribution_name) {
                result.addError("duplicate coupling form contribution name");
            }
        }
    }
    return result;
}

CouplingValidationResult validateFormContributionDeclarations(
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormContribution> contributions)
{
    CouplingValidationResult result =
        validateFormContributionDeclarations(contributions);
    for (const auto& declaration : declarations) {
        for (const auto& field : declaration.additional_fields) {
            if (additionalFieldSelected(field)) {
                continue;
            }
            for (const auto& contribution : contributions) {
                if (!contributionReferencesAdditionalField(contribution, field)) {
                    continue;
                }
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .field_name = field.field_name,
                    .endpoint_name = contribution.contribution_name,
                    .message = "disabled optional additional field is referenced by a form contribution",
                });
            }
        }
    }
    return result;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
