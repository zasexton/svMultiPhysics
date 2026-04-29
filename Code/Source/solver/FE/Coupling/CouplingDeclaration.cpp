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

std::string regionKey(const CouplingRegionUse& region)
{
    return region.participant_name + "/" + region.region_name;
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

    for (std::size_t i = 0; i < declaration.additional_fields.size(); ++i) {
        const auto& field = declaration.additional_fields[i];
        if (field.namespace_name.empty()) {
            result.addError("additional field declaration requires a namespace name");
        }
        if (field.field_name.empty()) {
            result.addError("additional field declaration requires a field name");
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

} // namespace coupling
} // namespace FE
} // namespace svmp
