/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGraph.h"

#include "Auxiliary/AuxiliaryStateManager.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

analysis::VariableKind toAnalysisVariableKind(CouplingVariableKind kind) noexcept
{
    switch (kind) {
    case CouplingVariableKind::Field:
        return analysis::VariableKind::FieldComponent;
    case CouplingVariableKind::AuxiliaryState:
        return analysis::VariableKind::AuxiliaryState;
    case CouplingVariableKind::AuxiliaryInput:
        return analysis::VariableKind::AuxiliaryInput;
    case CouplingVariableKind::AuxiliaryOutput:
        return analysis::VariableKind::AuxiliaryOutput;
    case CouplingVariableKind::BoundaryFunctional:
        return analysis::VariableKind::BoundaryFunctional;
    case CouplingVariableKind::GlobalScalar:
        return analysis::VariableKind::GlobalScalar;
    }
    return analysis::VariableKind::FieldComponent;
}

std::string couplingVariableUseAnalysisName(const CouplingContext& context,
                                            const CouplingVariableUse& variable)
{
    if (variable.participant_name.empty()) {
        return variable.name;
    }
    if (context.hasParticipant(variable.participant_name)) {
        const auto participant = context.participant(variable.participant_name);
        if (!participant.system_name.empty()) {
            return participant.system_name + "/" + variable.name;
        }
    }
    return variable.participant_name + "/" + variable.name;
}

std::optional<analysis::VariableKey> resolveCouplingVariableUse(
    const CouplingContext& context,
    const CouplingVariableUse& variable)
{
    if (variable.kind == CouplingVariableKind::Field) {
        if (!context.hasField(variable.participant_name, variable.name)) {
            return std::nullopt;
        }
        return analysis::VariableKey::field(
            context.field(variable.participant_name, variable.name).field_id,
            variable.component);
    }

    if (variable.name.empty()) {
        return std::nullopt;
    }
    return analysis::VariableKey::named(
        toAnalysisVariableKind(variable.kind),
        couplingVariableUseAnalysisName(context, variable));
}

namespace {

bool isRequired(CouplingRequirement requirement) noexcept
{
    return requirement == CouplingRequirement::Required;
}

const char* toString(CouplingPartitionedSolveStrategy strategy) noexcept
{
    switch (strategy) {
    case CouplingPartitionedSolveStrategy::ExplicitLagged:
        return "explicit_lagged";
    case CouplingPartitionedSolveStrategy::StaggeredFixedPoint:
        return "staggered_fixed_point";
    }
    return "unknown";
}

const CouplingRelationLoweringCapability* findLoweringCapability(
    const CouplingRegionRelationRequirement& requirement,
    CouplingRelationLoweringKind kind)
{
    const auto it = std::find_if(
        requirement.lowering_capabilities.begin(),
        requirement.lowering_capabilities.end(),
        [kind](const CouplingRelationLoweringCapability& capability) {
            return capability.lowering_kind == kind;
        });
    if (it == requirement.lowering_capabilities.end()) {
        return nullptr;
    }
    return &(*it);
}

bool supportsEnforcementStrategy(
    const CouplingRelationLoweringCapability& capability,
    std::string_view enforcement_strategy)
{
    return enforcement_strategy.empty() ||
           capability.enforcement_strategies.empty() ||
           std::find(capability.enforcement_strategies.begin(),
                     capability.enforcement_strategies.end(),
                     enforcement_strategy) != capability.enforcement_strategies.end();
}

bool supportsPartitionedSolveStrategy(
    const CouplingRelationLoweringCapability& capability,
    const std::optional<CouplingPartitionedSolveStrategy>& strategy)
{
    return !strategy.has_value() ||
           capability.partitioned_solve_strategies.empty() ||
           std::find(capability.partitioned_solve_strategies.begin(),
                     capability.partitioned_solve_strategies.end(),
                     *strategy) != capability.partitioned_solve_strategies.end();
}

std::string selectedLoweringDescription(
    const CouplingRelationLoweringRequest& request)
{
    std::ostringstream stream;
    stream << "mode=" << toString(request.mode)
           << ", lowering=" << toString(request.lowering_kind);
    if (!request.enforcement_strategy.empty()) {
        stream << ", enforcement=" << request.enforcement_strategy;
    }
    if (request.partitioned_solve_strategy.has_value()) {
        stream << ", partitioned_strategy="
               << toString(*request.partitioned_solve_strategy);
    }
    if (request.expert_fallback_enabled) {
        stream << ", expert_fallback=enabled";
    }
    return stream.str();
}

std::string availableLoweringDescription(
    const CouplingRegionRelationRequirement& requirement)
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < requirement.lowering_capabilities.size(); ++i) {
        const auto& capability = requirement.lowering_capabilities[i];
        if (i != 0u) {
            stream << ", ";
        }
        stream << toString(capability.lowering_kind);
        if (capability.supported) {
            stream << "(supported)";
        } else {
            stream << "(unsupported";
            if (!capability.unsupported_reason.empty()) {
                stream << ": " << capability.unsupported_reason;
            }
            stream << ")";
        }
        if (!capability.enforcement_strategies.empty()) {
            stream << "{enforcement=";
            for (std::size_t j = 0;
                 j < capability.enforcement_strategies.size();
                 ++j) {
                if (j != 0u) {
                    stream << "|";
                }
                stream << capability.enforcement_strategies[j];
            }
            stream << "}";
        }
        if (!capability.partitioned_solve_strategies.empty()) {
            stream << "{partitioned_strategy=";
            for (std::size_t j = 0;
                 j < capability.partitioned_solve_strategies.size();
                 ++j) {
                if (j != 0u) {
                    stream << "|";
                }
                stream << toString(capability.partitioned_solve_strategies[j]);
            }
            stream << "}";
        }
    }
    return stream.str();
}

bool isRequiredFieldRequirement(const CouplingFieldRequirement& requirement) noexcept
{
    return isRequired(requirement.requirement) &&
           isRequired(requirement.field.requirement);
}

bool additionalFieldSelected(
    const CouplingAdditionalFieldDeclaration& field) noexcept
{
    return field.requirement == CouplingRequirement::Required || field.enabled;
}

bool variableReferencesAdditionalField(
    const CouplingAdditionalFieldDeclaration& field,
    const CouplingVariableUse& variable)
{
    return variable.kind == CouplingVariableKind::Field &&
           variable.participant_name == field.namespace_name &&
           variable.name == field.field_name;
}

struct AdditionalFieldTargetResolution {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
};

std::optional<CouplingVariableKind> graphVariableKindForNonFieldRequirement(
    CouplingNonFieldDependencyRequirementKind kind) noexcept
{
    switch (kind) {
    case CouplingNonFieldDependencyRequirementKind::BoundaryFunctional:
        return CouplingVariableKind::BoundaryFunctional;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryState:
        return CouplingVariableKind::AuxiliaryState;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryInput:
        return CouplingVariableKind::AuxiliaryInput;
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput:
        return CouplingVariableKind::AuxiliaryOutput;
    case CouplingNonFieldDependencyRequirementKind::Parameter:
    case CouplingNonFieldDependencyRequirementKind::Coefficient:
    case CouplingNonFieldDependencyRequirementKind::MaterialStateOld:
    case CouplingNonFieldDependencyRequirementKind::MaterialStateWork:
    case CouplingNonFieldDependencyRequirementKind::BoundaryIntegral:
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<analysis::VariableKey> resolveNonFieldRequirementVariable(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement)
{
    const auto kind = graphVariableKindForNonFieldRequirement(requirement.kind);
    if (!kind.has_value()) {
        return std::nullopt;
    }
    return resolveCouplingVariableUse(
        context,
        CouplingVariableUse{
            .kind = *kind,
            .participant_name = requirement.participant_name,
            .name = requirement.name,
            .requirement = requirement.requirement,
        });
}

bool isNonFieldVariableRequirement(
    const CouplingNonFieldDependencyRequirement& requirement) noexcept
{
    return graphVariableKindForNonFieldRequirement(requirement.kind).has_value();
}

std::optional<CouplingFormNonFieldDependencyKind>
providerMetadataKindForRequirement(
    CouplingNonFieldDependencyRequirementKind kind) noexcept
{
    switch (kind) {
    case CouplingNonFieldDependencyRequirementKind::Parameter:
        return CouplingFormNonFieldDependencyKind::Parameter;
    case CouplingNonFieldDependencyRequirementKind::Coefficient:
        return CouplingFormNonFieldDependencyKind::Coefficient;
    case CouplingNonFieldDependencyRequirementKind::MaterialStateOld:
        return CouplingFormNonFieldDependencyKind::MaterialStateOld;
    case CouplingNonFieldDependencyRequirementKind::MaterialStateWork:
        return CouplingFormNonFieldDependencyKind::MaterialStateWork;
    case CouplingNonFieldDependencyRequirementKind::BoundaryIntegral:
        return CouplingFormNonFieldDependencyKind::BoundaryIntegral;
    case CouplingNonFieldDependencyRequirementKind::BoundaryFunctional:
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryState:
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryInput:
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput:
        return std::nullopt;
    }
    return std::nullopt;
}

bool isGraphNonFieldVariableKind(CouplingVariableKind kind) noexcept
{
    switch (kind) {
    case CouplingVariableKind::AuxiliaryState:
    case CouplingVariableKind::AuxiliaryInput:
    case CouplingVariableKind::AuxiliaryOutput:
    case CouplingVariableKind::BoundaryFunctional:
    case CouplingVariableKind::GlobalScalar:
        return true;
    case CouplingVariableKind::Field:
        return false;
    }
    return false;
}

bool hasContractTypeNode(const CouplingGraphSnapshot& snapshot,
                         const std::string& contract_type)
{
    return std::any_of(
        snapshot.contract_types.begin(),
        snapshot.contract_types.end(),
        [&](const CouplingGraphContractTypeNode& node) {
            return node.contract_type == contract_type;
        });
}

void populateDeclarationSnapshot(
    CouplingGraphSnapshot& snapshot,
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations)
{
    snapshot = CouplingGraphSnapshot{};

    for (const auto& participant : context.participants()) {
        snapshot.participants.push_back({.participant = participant});
    }
    for (const auto& field : context.fields()) {
        snapshot.fields.push_back({.field = field});
    }
    for (const auto& region : context.regions()) {
        snapshot.regions.push_back({.region = region});
    }
    for (const auto& shared_region : context.sharedRegions()) {
        snapshot.shared_regions.push_back({.shared_region = shared_region});
    }

    for (const auto& declaration : declarations) {
        if (!declaration.contract_type.empty() &&
            !hasContractTypeNode(snapshot, declaration.contract_type)) {
            snapshot.contract_types.push_back(
                {.contract_type = declaration.contract_type});
        }
        snapshot.contract_instances.push_back({
            .contract_type = declaration.contract_type,
            .contract_name = declaration.contract_name,
        });

        for (const auto& additional_field : declaration.additional_fields) {
            snapshot.additional_fields.push_back({
                .contract_name = declaration.contract_name,
                .declaration = additional_field,
            });
        }
        for (const auto& requirement : declaration.non_field_dependencies) {
            if (isNonFieldVariableRequirement(requirement)) {
                snapshot.non_field_variables.push_back({
                    .contract_name = declaration.contract_name,
                    .requirement = requirement,
                    .variable = resolveNonFieldRequirementVariable(
                        context,
                        requirement),
                });
            } else {
                snapshot.provider_metadata_requirements.push_back({
                    .contract_name = declaration.contract_name,
                    .requirement = requirement,
                });
            }
        }
        for (const auto& temporal : declaration.temporal_requirements) {
            snapshot.temporal_requirements.push_back({
                .contract_name = declaration.contract_name,
                .requirement = temporal,
            });
        }
        for (const auto& geometry : declaration.geometry_requirements) {
            snapshot.geometry_requirements.push_back({
                .contract_name = declaration.contract_name,
                .requirement = geometry,
            });
        }
        for (const auto& exchange :
             declaration.partitioned_exchange_declarations) {
            snapshot.partitioned_exchange_declarations.push_back({
                .contract_name = declaration.contract_name,
                .declaration = exchange,
            });
        }
        for (const auto& dependency : declaration.dependencies) {
            snapshot.dependency_expectations.push_back({
                .contract_name = declaration.contract_name,
                .declaration = dependency,
                .residual_row = resolveCouplingVariableUse(
                    context,
                    dependency.residual_row),
                .dependency = resolveCouplingVariableUse(
                    context,
                    dependency.dependency),
            });
        }
        for (const auto& expected_block : declaration.expected_blocks) {
            snapshot.expected_blocks.push_back({
                .contract_name = declaration.contract_name,
                .declaration = expected_block,
                .residual_row = resolveCouplingVariableUse(
                    context,
                    expected_block.residual_row),
                .dependency = resolveCouplingVariableUse(
                    context,
                    expected_block.dependency),
            });
        }
    }
}

std::optional<AdditionalFieldTargetResolution> participantAdditionalFieldTarget(
    const CouplingContext& context,
    const std::string& participant_name)
{
    if (participant_name.empty() || !context.hasParticipant(participant_name)) {
        return std::nullopt;
    }
    const auto participant = context.participant(participant_name);
    if (participant.system == nullptr) {
        return std::nullopt;
    }
    return AdditionalFieldTargetResolution{
        .participant_name = participant.participant_name,
        .system_name = participant.system_name,
        .system = participant.system,
    };
}

std::optional<AdditionalFieldTargetResolution> sharedRegionAdditionalFieldTarget(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    if (!field.shared_region_name.has_value() ||
        !context.hasSharedRegion(*field.shared_region_name)) {
        return std::nullopt;
    }
    const auto group = context.sharedRegionGroup(*field.shared_region_name);
    if (group.participant_regions.empty()) {
        return std::nullopt;
    }

    const auto& first = group.participant_regions.front();
    if (first.system == nullptr) {
        return std::nullopt;
    }
    const auto same_system = std::all_of(
        group.participant_regions.begin(),
        group.participant_regions.end(),
        [&](const CouplingRegionRef& region) {
            return region.system == first.system;
        });
    if (!same_system) {
        return std::nullopt;
    }
    return AdditionalFieldTargetResolution{
        .participant_name = first.participant_name,
        .system_name = first.system_name,
        .system = first.system,
    };
}

std::optional<AdditionalFieldTargetResolution> additionalFieldTarget(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    if (!field.system_participant_name.empty()) {
        return participantAdditionalFieldTarget(context,
                                                field.system_participant_name);
    }
    if (field.field_namespace == CouplingAdditionalFieldNamespace::Participant) {
        return participantAdditionalFieldTarget(context, field.namespace_name);
    }
    if (field.field_namespace == CouplingAdditionalFieldNamespace::Contract &&
        field.shared_region_name.has_value()) {
        return sharedRegionAdditionalFieldTarget(context, field);
    }
    return std::nullopt;
}

std::string additionalFieldRegistrationTargetKey(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    const auto target = additionalFieldTarget(context, field);
    if (target.has_value()) {
        return target->system_name;
    }

    std::string participant_name = field.system_participant_name;
    if (participant_name.empty() &&
        field.field_namespace == CouplingAdditionalFieldNamespace::Participant) {
        participant_name = field.namespace_name;
    }
    if (!participant_name.empty() && context.hasParticipant(participant_name)) {
        const auto participant = context.participant(participant_name);
        if (!participant.system_name.empty()) {
            return participant.system_name;
        }
    }
    return participant_name;
}

void validateContractOwnedAdditionalField(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingAdditionalFieldDeclaration& field,
    CouplingValidationResult& result)
{
    if (field.field_namespace != CouplingAdditionalFieldNamespace::Contract ||
        !additionalFieldSelected(field)) {
        return;
    }
    if (field.namespace_name != declaration.contract_name) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .field_name = field.field_name,
            .message = "contract-owned additional field namespace must match the contract instance name",
        });
    }
    if (context.hasParticipant(field.namespace_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = field.namespace_name,
            .field_name = field.field_name,
            .message = "contract-owned additional field namespace must not be a participant",
        });
    }
    if (!additionalFieldTarget(context, field).has_value()) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .field_name = field.field_name,
            .message = "contract-owned additional field does not resolve to a target system",
        });
    }
}

void validateAdditionalFieldLowering(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingAdditionalFieldDeclaration& field,
    CouplingValidationResult& result)
{
    if (!additionalFieldSelected(field)) {
        return;
    }
    if (!additionalFieldTarget(context, field).has_value()) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = field.system_participant_name.empty()
                ? field.namespace_name
                : field.system_participant_name,
            .field_name = field.field_name,
            .message = "additional field declaration cannot be lowered to an FE field registration target",
        });
    }
}

void validateInterfaceAdditionalFieldMarker(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingAdditionalFieldDeclaration& field,
    CouplingValidationResult& result)
{
    if (!additionalFieldSelected(field) ||
        field.scope != CouplingAdditionalFieldScope::InterfaceFace) {
        return;
    }

    const auto target = additionalFieldTarget(context, field);
    if (!target.has_value()) {
        return;
    }

    if (field.region_name.has_value()) {
        if (!context.hasRegion(target->participant_name, *field.region_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = target->participant_name,
                .field_name = field.field_name,
                .region_name = *field.region_name,
                .message = "interface additional field participant region is missing",
            });
            return;
        }
        const auto region =
            context.region(target->participant_name, *field.region_name);
        if (region.kind != CouplingRegionKind::InterfaceFace) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = target->participant_name,
                .field_name = field.field_name,
                .region_name = *field.region_name,
                .message = "interface additional field participant region must be an interface face",
            });
        }
        if (region.marker < 0) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = target->participant_name,
                .field_name = field.field_name,
                .region_name = *field.region_name,
                .message = "interface additional field participant region requires a marker",
            });
        }
        return;
    }

    if (!field.shared_region_name.has_value()) {
        return;
    }
    if (!context.hasSharedRegion(*field.shared_region_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .field_name = field.field_name,
            .region_name = *field.shared_region_name,
            .message = "interface additional field shared region is missing",
        });
        return;
    }

    const auto group = context.sharedRegionGroup(*field.shared_region_name);
    std::optional<int> marker;
    bool found_target_region = false;
    for (const auto& region : group.participant_regions) {
        if (region.system != target->system) {
            continue;
        }
        found_target_region = true;
        if (region.kind != CouplingRegionKind::InterfaceFace) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = region.participant_name,
                .field_name = field.field_name,
                .region_name = *field.shared_region_name,
                .message = "interface additional field shared region must map to interface faces",
            });
        }
        if (region.marker < 0) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = region.participant_name,
                .field_name = field.field_name,
                .region_name = *field.shared_region_name,
                .message = "interface additional field shared region requires markers",
            });
            continue;
        }
        if (!marker.has_value()) {
            marker = region.marker;
        } else if (*marker != region.marker) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = region.participant_name,
                .field_name = field.field_name,
                .region_name = *field.shared_region_name,
                .message = "interface additional field shared-region markers must agree in one system",
            });
        }
    }
    if (!found_target_region) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .field_name = field.field_name,
            .region_name = *field.shared_region_name,
            .message = "interface additional field target participant is not in the shared region",
        });
    }
}

std::string additionalFieldGraphKey(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    std::ostringstream os;
    os << static_cast<int>(field.field_namespace) << "/"
       << field.namespace_name << "/"
       << field.field_name << "/"
       << additionalFieldRegistrationTargetKey(context, field) << "/"
       << static_cast<int>(field.scope) << "/"
       << field.region_name.value_or(std::string{}) << "/"
       << field.shared_region_name.value_or(std::string{});
    return os.str();
}

bool additionalFieldCollidesWithBaseField(
    const CouplingContext& context,
    const CouplingAdditionalFieldDeclaration& field)
{
    const auto target_system = additionalFieldRegistrationTargetKey(context, field);
    return std::any_of(
        context.fields().begin(),
        context.fields().end(),
        [&](const CouplingFieldRef& base_field) {
            if (base_field.coupling_owned) {
                return false;
            }
            if (base_field.field_name != field.field_name) {
                return false;
            }
            if (field.field_namespace ==
                    CouplingAdditionalFieldNamespace::Participant &&
                base_field.participant_name == field.namespace_name) {
                return true;
            }
            return !target_system.empty() &&
                   !base_field.system_name.empty() &&
                   base_field.system_name == target_system;
        });
}

void validateAdditionalFieldGraphDeclarations(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    struct SeenAdditionalField {
        std::string key;
    };
    struct SkippedAdditionalField {
        std::size_t declaration_index{0};
        CouplingAdditionalFieldDeclaration field;
    };
    std::vector<SeenAdditionalField> seen;
    std::vector<SkippedAdditionalField> skipped;

    for (std::size_t i = 0; i < declarations.size(); ++i) {
        const auto& declaration = declarations[i];
        for (const auto& field : declaration.additional_fields) {
            if (!additionalFieldSelected(field)) {
                skipped.push_back(SkippedAdditionalField{
                    .declaration_index = i,
                    .field = field,
                });
                continue;
            }
            validateContractOwnedAdditionalField(context,
                                                 declaration,
                                                 field,
                                                 result);
            validateAdditionalFieldLowering(context,
                                            declaration,
                                            field,
                                            result);
            validateInterfaceAdditionalFieldMarker(context,
                                                   declaration,
                                                   field,
                                                   result);
            if (additionalFieldCollidesWithBaseField(context, field)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .field_name = field.field_name,
                    .message = "additional field declaration collides with a base field",
                });
            }

            const auto key = additionalFieldGraphKey(context, field);
            const auto duplicate = std::find_if(
                seen.begin(),
                seen.end(),
                [&](const SeenAdditionalField& prior) {
                    return prior.key == key;
                });
            if (duplicate != seen.end()) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .field_name = field.field_name,
                    .message = "duplicate additional field declaration in coupling graph",
                });
                continue;
            }
            seen.push_back(SeenAdditionalField{
                .key = key,
            });
        }
    }

    for (std::size_t i = 0; i < declarations.size(); ++i) {
        const auto& declaration = declarations[i];
        for (const auto& skipped_field : skipped) {
            if (skipped_field.declaration_index == i) {
                continue;
            }
            const auto references_skipped_dependency =
                std::any_of(declaration.dependencies.begin(),
                            declaration.dependencies.end(),
                            [&](const CouplingResidualDependency& dependency) {
                                return variableReferencesAdditionalField(
                                           skipped_field.field,
                                           dependency.residual_row) ||
                                       variableReferencesAdditionalField(
                                           skipped_field.field,
                                           dependency.dependency);
                            });
            const auto references_skipped_block =
                std::any_of(declaration.expected_blocks.begin(),
                            declaration.expected_blocks.end(),
                            [&](const CouplingBlockExpectation& block) {
                                return variableReferencesAdditionalField(
                                           skipped_field.field,
                                           block.residual_row) ||
                                       variableReferencesAdditionalField(
                                           skipped_field.field,
                                           block.dependency);
                            });
            if (!references_skipped_dependency && !references_skipped_block) {
                continue;
            }
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .field_name = skipped_field.field.field_name,
                .message = "disabled optional additional field is referenced by another contract",
            });
        }
    }
}

void appendResolvedPartitionedExchangeNodes(
    CouplingGraphSnapshot& snapshot,
    const PartitionedCouplingPlan& partitioned_plan)
{
    snapshot.resolved_partitioned_exchanges.clear();
    for (const auto& exchange : partitioned_plan.exchanges) {
        snapshot.resolved_partitioned_exchanges.push_back({.exchange = exchange});
    }
}

std::string variableLabel(const analysis::VariableKey& variable)
{
    std::ostringstream os;
    os << analysis::toString(variable.kind) << "(";
    if (variable.kind == analysis::VariableKind::FieldComponent) {
        os << variable.field_id;
        if (variable.component >= 0) {
            os << ":" << variable.component;
        }
    } else {
        os << variable.name;
    }
    os << ")";
    return os.str();
}

const char* providerMetadataKindName(
    CouplingNonFieldDependencyRequirementKind kind) noexcept
{
    switch (kind) {
    case CouplingNonFieldDependencyRequirementKind::Parameter:
        return "Parameter";
    case CouplingNonFieldDependencyRequirementKind::Coefficient:
        return "Coefficient";
    case CouplingNonFieldDependencyRequirementKind::MaterialStateOld:
        return "MaterialStateOld";
    case CouplingNonFieldDependencyRequirementKind::MaterialStateWork:
        return "MaterialStateWork";
    case CouplingNonFieldDependencyRequirementKind::BoundaryIntegral:
        return "BoundaryIntegral";
    case CouplingNonFieldDependencyRequirementKind::BoundaryFunctional:
        return "BoundaryFunctional";
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryState:
        return "AuxiliaryState";
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryInput:
        return "AuxiliaryInput";
    case CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput:
        return "AuxiliaryOutput";
    }
    return "NonField";
}

std::string providerMetadataRequirementLabel(
    const CouplingNonFieldDependencyRequirement& requirement)
{
    std::ostringstream os;
    os << providerMetadataKindName(requirement.kind) << "(";
    if (!requirement.participant_name.empty()) {
        os << requirement.participant_name << "/";
    }
    os << requirement.name << ")";
    return os.str();
}

const char* graphVariableRegistryName(CouplingVariableKind kind) noexcept
{
    switch (kind) {
    case CouplingVariableKind::AuxiliaryState:
        return "AuxiliaryStateManager";
    case CouplingVariableKind::AuxiliaryInput:
        return "AuxiliaryInputRegistry";
    case CouplingVariableKind::AuxiliaryOutput:
        return "AuxiliaryOutputRegistry";
    case CouplingVariableKind::BoundaryFunctional:
        return "BoundaryReductionService";
    case CouplingVariableKind::GlobalScalar:
        return "ParameterRegistry";
    case CouplingVariableKind::Field:
        return "FieldRegistry";
    }
    return "registry";
}

std::string graphVariableDiagnosticLabel(const CouplingContext& context,
                                         const CouplingVariableUse& variable)
{
    const auto resolved = resolveCouplingVariableUse(context, variable);
    if (resolved.has_value()) {
        return variableLabel(*resolved);
    }
    return variable.name;
}

bool isMissingIndex(std::size_t value) noexcept
{
    return value == static_cast<std::size_t>(-1);
}

std::size_t auxiliaryOutputSlotOf(const systems::FESystem& system,
                                  std::string_view name)
{
    const auto slash = name.find('/');
    if (slash != std::string_view::npos) {
        return system.auxiliaryOutputSlotOf(name.substr(0, slash),
                                            name.substr(slash + 1));
    }
    return system.auxiliaryOutputSlotOf(name);
}

bool hasBoundaryFunctional(const CouplingContext& context,
                           const std::string& participant_name,
                           std::string_view functional_name)
{
    if (!context.hasParticipant(participant_name)) {
        return false;
    }
    const auto participant = context.participant(participant_name);
    if (participant.system == nullptr) {
        return false;
    }
    return std::any_of(
        context.fields().begin(),
        context.fields().end(),
        [&](const CouplingFieldRef& field) {
            if (field.participant_name != participant_name) {
                return false;
            }
            const auto* service =
                participant.system->boundaryReductionServiceIfPresent(field.field_id);
            return service != nullptr &&
                   service->hasFunctional(functional_name);
        });
}

void reportMissingNonFieldGraphVariable(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingVariableUse& variable,
    CouplingValidationResult& result)
{
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = declaration.contract_name,
        .participant_name = variable.participant_name,
        .message = "required non-field graph variable is missing from " +
                   std::string(graphVariableRegistryName(variable.kind)) + ": " +
                   graphVariableDiagnosticLabel(context, variable),
    });
}

void reportNonFieldGraphVariableRegistryFailure(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingVariableUse& variable,
    std::string message,
    CouplingValidationResult& result)
{
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = declaration.contract_name,
        .participant_name = variable.participant_name,
        .message = std::move(message) + ": " +
                   graphVariableDiagnosticLabel(context, variable),
    });
}

void validateRequiredNonFieldGraphVariable(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingVariableUse& variable,
    CouplingValidationResult& result)
{
    if (!isGraphNonFieldVariableKind(variable.kind) ||
        !isRequired(variable.requirement)) {
        return;
    }
    if (variable.name.empty()) {
        return;
    }
    if (variable.kind == CouplingVariableKind::GlobalScalar &&
        variable.participant_name.empty()) {
        return;
    }
    if (variable.participant_name.empty()) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .message = "required non-field graph variable needs a participant scope: " +
                       graphVariableDiagnosticLabel(context, variable),
        });
        return;
    }
    if (!context.hasParticipant(variable.participant_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = variable.participant_name,
            .message = "non-field graph variable participant is missing from the context",
        });
        return;
    }

    const auto participant = context.participant(variable.participant_name);
    if (participant.system == nullptr) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = variable.participant_name,
            .message = "non-field graph variable participant has no owning system",
        });
        return;
    }

    switch (variable.kind) {
    case CouplingVariableKind::AuxiliaryState: {
        const auto* manager = participant.system->auxiliaryStateManagerIfPresent();
        if (manager == nullptr || !manager->hasBlock(variable.name)) {
            reportMissingNonFieldGraphVariable(
                context,
                declaration,
                variable,
                result);
        }
        return;
    }
    case CouplingVariableKind::AuxiliaryInput: {
        const auto* registry = participant.system->auxiliaryInputRegistryIfPresent();
        if (registry == nullptr || !registry->hasInput(variable.name)) {
            reportMissingNonFieldGraphVariable(
                context,
                declaration,
                variable,
                result);
            return;
        }
        (void)registry->slotOf(variable.name);
        return;
    }
    case CouplingVariableKind::AuxiliaryOutput:
        try {
            const auto slot = auxiliaryOutputSlotOf(*participant.system,
                                                    variable.name);
            if (isMissingIndex(slot)) {
                reportMissingNonFieldGraphVariable(
                    context,
                    declaration,
                    variable,
                    result);
                return;
            }
            const auto id = participant.system->auxiliaryOutputIdOf(variable.name);
            if (participant.system->auxiliaryOutputDescriptor(id) == nullptr) {
                reportMissingNonFieldGraphVariable(
                    context,
                    declaration,
                    variable,
                    result);
            }
        } catch (const std::exception& ex) {
            reportNonFieldGraphVariableRegistryFailure(
                context,
                declaration,
                variable,
                std::string("AuxiliaryOutputRegistry lookup failed: ") + ex.what(),
                result);
        }
        return;
    case CouplingVariableKind::BoundaryFunctional:
        if (!hasBoundaryFunctional(context,
                                   variable.participant_name,
                                   variable.name)) {
            reportMissingNonFieldGraphVariable(
                context,
                declaration,
                variable,
                result);
        }
        return;
    case CouplingVariableKind::GlobalScalar: {
        const auto* spec = participant.system->parameterRegistry().find(variable.name);
        const auto slot = participant.system->parameterRegistry().slotOf(variable.name);
        if (spec == nullptr || !slot.has_value()) {
            reportMissingNonFieldGraphVariable(
                context,
                declaration,
                variable,
                result);
        }
        return;
    }
    case CouplingVariableKind::Field:
        return;
    }
}

void validateRequiredNonFieldGraphVariables(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    for (const auto& declaration : declarations) {
        for (const auto& requirement : declaration.non_field_dependencies) {
            const auto kind = graphVariableKindForNonFieldRequirement(requirement.kind);
            if (!kind.has_value()) {
                continue;
            }
            validateRequiredNonFieldGraphVariable(
                context,
                declaration,
                CouplingVariableUse{
                    .kind = *kind,
                    .participant_name = requirement.participant_name,
                    .name = requirement.name,
                    .requirement = requirement.requirement,
                },
                result);
        }
        for (const auto& dependency : declaration.dependencies) {
            validateRequiredNonFieldGraphVariable(
                context,
                declaration,
                dependency.residual_row,
                result);
            validateRequiredNonFieldGraphVariable(
                context,
                declaration,
                dependency.dependency,
                result);
        }
        for (const auto& block : declaration.expected_blocks) {
            validateRequiredNonFieldGraphVariable(
                context,
                declaration,
                block.residual_row,
                result);
            validateRequiredNonFieldGraphVariable(
                context,
                declaration,
                block.dependency,
                result);
        }
    }
}

bool providerParticipantMatches(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement,
    const CouplingFormNonFieldDependencyProvenance& provenance)
{
    if (requirement.participant_name.empty()) {
        return true;
    }
    if (!provenance.participant_name.empty() &&
        provenance.participant_name != requirement.participant_name) {
        return false;
    }
    if (!context.hasParticipant(requirement.participant_name)) {
        return true;
    }
    const auto participant = context.participant(requirement.participant_name);
    return !participant.system_name.empty() &&
           provenance.system_name == participant.system_name;
}

std::optional<CouplingRegionRef> providerRequirementRegion(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement)
{
    if (!requirement.region.has_value()) {
        return std::nullopt;
    }
    const auto& region = *requirement.region;
    const auto participant_name = region.participant_name.empty()
                                      ? requirement.participant_name
                                      : region.participant_name;
    if (region.shared_region_name.has_value() &&
        context.hasSharedRegion(*region.shared_region_name)) {
        if (!participant_name.empty()) {
            return context.sharedRegion(*region.shared_region_name,
                                        participant_name);
        }
        const auto group = context.sharedRegionGroup(*region.shared_region_name);
        if (group.participant_regions.size() == 1u) {
            return group.participant_regions.front();
        }
    }
    if (!participant_name.empty() &&
        context.hasRegion(participant_name, region.region_name)) {
        return context.region(participant_name, region.region_name);
    }
    return std::nullopt;
}

bool providerRegionMatches(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement,
    const CouplingFormNonFieldDependencyProvenance& provenance)
{
    if (requirement.required_region_kind.has_value()) {
        const auto expected_domain =
            toAnalysisDomainKind(*requirement.required_region_kind);
        if (expected_domain.has_value() && provenance.domain != *expected_domain) {
            return false;
        }
    }
    if (!requirement.region.has_value()) {
        return true;
    }

    const auto& region = *requirement.region;
    if (!region.region_name.empty()) {
        if (!provenance.region_name.has_value() ||
            *provenance.region_name != region.region_name) {
            return false;
        }
    }
    if (region.shared_region_name.has_value()) {
        if (!provenance.shared_region_name.has_value() ||
            *provenance.shared_region_name != *region.shared_region_name) {
            return false;
        }
    }

    const auto resolved_region = providerRequirementRegion(context, requirement);
    if (!resolved_region.has_value()) {
        return true;
    }
    if (resolved_region->marker >= 0 &&
        provenance.marker != resolved_region->marker) {
        return false;
    }
    if (resolved_region->side != CouplingInterfaceSide::None &&
        provenance.side != resolved_region->side) {
        return false;
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (resolved_region->logical_region.has_value() &&
        (!provenance.logical_region.has_value() ||
         !resolved_region->logical_region->compatible_with(
             *provenance.logical_region))) {
        return false;
    }
#endif
    return true;
}

bool providerValueMetadataMatches(
    const CouplingNonFieldDependencyRequirement& requirement,
    const CouplingFormNonFieldDependencyProvenance& provenance)
{
    if (!requirement.expected_value_type.empty() &&
        provenance.value_type != requirement.expected_value_type) {
        return false;
    }
    if (requirement.expected_parameter_value_type.has_value() &&
        (!provenance.parameter_value_type.has_value() ||
         *provenance.parameter_value_type !=
             *requirement.expected_parameter_value_type)) {
        return false;
    }
    if (requirement.material_state_byte_offset.has_value() &&
        (!provenance.byte_offset.has_value() ||
         *provenance.byte_offset !=
             *requirement.material_state_byte_offset)) {
        return false;
    }
    return true;
}

bool providerMetadataMatchesRequirement(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement,
    const CouplingFormNonFieldDependencyProvenance& provenance)
{
    const auto expected_kind = providerMetadataKindForRequirement(requirement.kind);
    if (!expected_kind.has_value() || provenance.kind != *expected_kind) {
        return false;
    }
    if (provenance.provider.empty()) {
        return false;
    }
    return provenance.name == requirement.name &&
           providerParticipantMatches(context, requirement, provenance) &&
           providerRegionMatches(context, requirement, provenance) &&
           providerValueMetadataMatches(requirement, provenance);
}

bool hasProviderMetadataEvidence(
    const CouplingContext& context,
    const CouplingNonFieldDependencyRequirement& requirement,
    std::span<const CouplingFormAnalysisMetadata> installed_forms)
{
    for (const auto& form : installed_forms) {
        for (const auto& provenance : form.non_field_dependencies) {
            if (providerMetadataMatchesRequirement(context,
                                                   requirement,
                                                   provenance)) {
                return true;
            }
        }
    }
    return false;
}

void validateProviderMetadataRequirements(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    CouplingValidationResult& result)
{
    for (const auto& declaration : declarations) {
        for (const auto& requirement : declaration.non_field_dependencies) {
            if (!isRequired(requirement.requirement) ||
                !providerMetadataKindForRequirement(requirement.kind).has_value()) {
                continue;
            }
            if (hasProviderMetadataEvidence(context,
                                            requirement,
                                            installed_forms)) {
                continue;
            }
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = requirement.participant_name,
                .message = "required provider metadata is missing for non-field dependency: " +
                           providerMetadataRequirementLabel(requirement),
            });
        }
    }
}

std::optional<analysis::VariableKey> resolveVariable(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingVariableUse& variable,
    CouplingValidationResult& result)
{
    const auto resolved = resolveCouplingVariableUse(context, variable);
    if (!resolved.has_value() && variable.kind == CouplingVariableKind::Field) {
        if (!context.hasField(variable.participant_name, variable.name)) {
            if (isRequired(variable.requirement)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = variable.participant_name,
                    .field_name = variable.name,
                    .message = "coupling graph variable field is missing from the context",
                });
            }
            return std::nullopt;
        }
    }

    return resolved;
}

std::optional<CouplingFieldRef> resolvedFieldRef(
    const CouplingContext& context,
    const CouplingVariableUse& variable)
{
    if (variable.kind != CouplingVariableKind::Field ||
        !context.hasField(variable.participant_name, variable.name)) {
        return std::nullopt;
    }
    return context.field(variable.participant_name, variable.name);
}

bool validateMonolithicFieldSystemCompatibility(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingVariableUse& residual_row,
    const CouplingVariableUse& dependency,
    std::string message,
    CouplingValidationResult& result)
{
    const auto row_field = resolvedFieldRef(context, residual_row);
    const auto dependency_field = resolvedFieldRef(context, dependency);
    if (!row_field.has_value() || !dependency_field.has_value()) {
        return true;
    }
    if (row_field->system == dependency_field->system) {
        return true;
    }

    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = declaration.contract_name,
        .participant_name = dependency.participant_name,
        .field_name = dependency.name,
        .message = std::move(message) +
                   " fields must resolve to one owning system: " +
                   residual_row.participant_name + "/" + residual_row.name +
                   " depends on " + dependency.participant_name + "/" +
                   dependency.name,
    });
    return false;
}

void validateInterfaceRegionTopology(const CouplingContractDeclaration& declaration,
                                     const CouplingRegionRef& region,
                                     CouplingValidationResult& result)
{
    if (region.kind != CouplingRegionKind::InterfaceFace) {
        return;
    }
    if (region.marker < 0) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = region.participant_name,
            .region_name = region.region_name,
            .message = "interface-face coupling region requires an interface marker",
        });
        return;
    }
    if (region.system != nullptr && !region.system->hasInterfaceMesh(region.marker)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = region.participant_name,
            .region_name = region.region_name,
            .message = "interface-face coupling region is missing registered interface topology",
        });
    }
}

struct ResolvedDeclaredDependency {
    std::string contract_name;
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    CouplingDependencyMode mode{CouplingDependencyMode::ImplicitMonolithic};
};

struct ResolvedExpectedBlock {
    std::string contract_name;
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    bool expected_nonzero{true};
    bool expect_matrix_block{true};
};

struct ResolvedTemporalRequirement {
    std::string contract_name;
    CouplingTemporalQuantity quantity{CouplingTemporalQuantity::Time};
    std::optional<FieldId> field;
    std::optional<CouplingGeometryTerminalScope> mesh_motion_scope;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    int derivative_order{0};
    int history_index{0};
};

struct ResolvedGeometryRequirement {
    std::string contract_name;
    CouplingGeometryTerminalQuantity quantity{
        CouplingGeometryTerminalQuantity::MeshDisplacement};
    std::optional<analysis::DomainKind> domain;
    CouplingGeometryTerminalLocationProvenance location;
    std::optional<CouplingGeometryTerminalOwnerProvenance> owner;
};

bool isMeshTemporalQuantity(CouplingTemporalQuantity quantity) noexcept
{
    switch (quantity) {
    case CouplingTemporalQuantity::MeshVelocity:
    case CouplingTemporalQuantity::MeshAcceleration:
    case CouplingTemporalQuantity::PreviousMeshVelocity:
    case CouplingTemporalQuantity::PredictedMeshVelocity:
        return true;
    case CouplingTemporalQuantity::Time:
    case CouplingTemporalQuantity::TimeStep:
    case CouplingTemporalQuantity::EffectiveTimeStep:
    case CouplingTemporalQuantity::FieldDerivative:
    case CouplingTemporalQuantity::FieldHistoryValue:
        return false;
    }
    return false;
}

std::optional<std::string> meshScopeParticipant(
    const CouplingGeometryTerminalScope& scope)
{
    if (scope.participant_name.has_value()) {
        return scope.participant_name;
    }
    if (scope.region.has_value() && !scope.region->participant_name.empty()) {
        return scope.region->participant_name;
    }
    return std::nullopt;
}

std::optional<std::string> meshScopeSharedRegion(
    const CouplingGeometryTerminalScope& scope)
{
    if (scope.region.has_value() &&
        scope.region->shared_region_name.has_value()) {
        return scope.region->shared_region_name;
    }
    if (scope.location.has_value() &&
        scope.location->shared_region_name.has_value()) {
        return scope.location->shared_region_name;
    }
    return std::nullopt;
}

bool sharedRegionHasParticipant(const SharedRegionRef& group,
                                const std::string& participant_name)
{
    return std::any_of(
        group.participant_regions.begin(),
        group.participant_regions.end(),
        [&](const CouplingRegionRef& region) {
            return region.participant_name == participant_name;
        });
}

const CouplingRegionRef* sharedRegionParticipant(
    const SharedRegionRef& group,
    const std::string& participant_name)
{
    const auto it = std::find_if(
        group.participant_regions.begin(),
        group.participant_regions.end(),
        [&](const CouplingRegionRef& region) {
            return region.participant_name == participant_name;
        });
    return it == group.participant_regions.end() ? nullptr : &*it;
}

void validateMeshTemporalScope(const CouplingContext& context,
                               const CouplingContractDeclaration& declaration,
                               const CouplingTemporalRequirement& temporal,
                               CouplingValidationResult& result)
{
    if (!isMeshTemporalQuantity(temporal.quantity) ||
        !temporal.mesh_motion_scope.has_value() ||
        !isRequired(temporal.requirement)) {
        return;
    }

    const auto& scope = *temporal.mesh_motion_scope;
    const auto participant_name = meshScopeParticipant(scope);
    if (participant_name.has_value() &&
        !context.hasParticipant(*participant_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = *participant_name,
            .message = "mesh temporal requirement participant is missing from the context",
        });
    }

    const auto shared_region_name = meshScopeSharedRegion(scope);
    if (!shared_region_name.has_value()) {
        return;
    }
    if (!context.hasSharedRegion(*shared_region_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = participant_name.value_or(std::string{}),
            .region_name = *shared_region_name,
            .message = "mesh temporal shared region is missing from the context",
        });
        return;
    }

    const auto group = context.sharedRegionGroup(*shared_region_name);
    if (!participant_name.has_value()) {
        if (group.participant_regions.size() > 1u) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .region_name = *shared_region_name,
                .message = "shared-region mesh temporal requirement is ambiguous without a participant owner",
            });
        }
        return;
    }

    if (!sharedRegionHasParticipant(group, *participant_name)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = *participant_name,
            .region_name = *shared_region_name,
            .message = "mesh temporal shared-region participant mapping is missing from the context",
        });
    }
}

std::vector<ResolvedDeclaredDependency> resolveDeclaredDependencies(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    std::vector<ResolvedDeclaredDependency> resolved;
    for (const auto& declaration : declarations) {
        for (const auto& dependency : declaration.dependencies) {
            const auto row =
                resolveVariable(context, declaration, dependency.residual_row, result);
            const auto col =
                resolveVariable(context, declaration, dependency.dependency, result);
            if (!row.has_value() || !col.has_value()) {
                continue;
            }
            if (dependency.mode == CouplingDependencyMode::ImplicitMonolithic &&
                !validateMonolithicFieldSystemCompatibility(
                    context,
                    declaration,
                    dependency.residual_row,
                    dependency.dependency,
                    "implicit monolithic coupling dependency",
                    result)) {
                continue;
            }
            resolved.push_back(ResolvedDeclaredDependency{
                .contract_name = declaration.contract_name,
                .residual_row = *row,
                .dependency = *col,
                .mode = dependency.mode,
            });
        }
    }
    return resolved;
}

std::vector<ResolvedTemporalRequirement> resolveDeclaredTemporalRequirements(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    std::vector<ResolvedTemporalRequirement> resolved;
    for (const auto& declaration : declarations) {
        for (const auto& temporal : declaration.temporal_requirements) {
            validateMeshTemporalScope(context, declaration, temporal, result);

            ResolvedTemporalRequirement requirement{
                .contract_name = declaration.contract_name,
                .quantity = temporal.quantity,
                .mesh_motion_scope = temporal.mesh_motion_scope,
                .mesh_motion_role = temporal.mesh_motion_role,
                .derivative_order = temporal.derivative_order,
                .history_index = temporal.history_index,
            };

            if (temporal.field.has_value()) {
                if (context.hasField(temporal.field->participant_name,
                                     temporal.field->field_name)) {
                    requirement.field =
                        context.field(temporal.field->participant_name,
                                      temporal.field->field_name).field_id;
                } else if (isRequired(temporal.field->requirement) &&
                           isRequired(temporal.requirement)) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .participant_name = temporal.field->participant_name,
                        .field_name = temporal.field->field_name,
                        .message = "temporal requirement field is missing from the context",
                    });
                    continue;
                }
            }

            resolved.push_back(std::move(requirement));
        }
    }
    return resolved;
}

forms::GeometryConfiguration graphGeometryConfiguration(
    CouplingCoordinateConfiguration configuration) noexcept
{
    switch (configuration) {
    case CouplingCoordinateConfiguration::Reference:
        return forms::GeometryConfiguration::Reference;
    case CouplingCoordinateConfiguration::Current:
        return forms::GeometryConfiguration::Current;
    }
    return forms::GeometryConfiguration::Reference;
}

void reportMissingGeometryRegion(const CouplingContractDeclaration& declaration,
                                 const CouplingGeometryTerminalRequirement& requirement,
                                 const CouplingRegionEndpointDeclaration& region,
                                 CouplingValidationResult& result)
{
    if (!isRequired(requirement.requirement)) {
        return;
    }
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = declaration.contract_name,
        .participant_name = region.participant_name,
        .region_name = region.region_name,
        .message = "geometry terminal region is missing from the context",
    });
}

void applyResolvedGeometryRegion(ResolvedGeometryRequirement& requirement,
                                 const CouplingRegionRef& region,
                                 std::optional<std::string> shared_region_name)
{
    requirement.location.region_kind = region.kind;
    requirement.location.marker = region.marker;
    if (shared_region_name.has_value()) {
        requirement.location.shared_region_name = std::move(shared_region_name);
    }
    if (region.side != CouplingInterfaceSide::None) {
        requirement.location.side = region.side;
    }
    requirement.location.geometry_revision = region.geometry_revision;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    requirement.location.logical_region = region.logical_region;
    if (requirement.location.geometry_revision == 0u &&
        region.revision_snapshot.has_value()) {
        requirement.location.geometry_revision =
            region.revision_snapshot->geometry_revision;
    }
#endif

    requirement.owner = CouplingGeometryTerminalOwnerProvenance{
        .participant_name = region.participant_name,
        .system_name = region.system_name,
        .region_name = region.region_name.empty()
                           ? std::optional<std::string>{}
                           : std::optional<std::string>{region.region_name},
        .shared_region_name = requirement.location.shared_region_name,
    };
}

ResolvedGeometryRequirement resolveGeometryRequirement(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const CouplingGeometryTerminalRequirement& geometry,
    CouplingValidationResult& result)
{
    ResolvedGeometryRequirement requirement{
        .contract_name = declaration.contract_name,
        .quantity = geometry.quantity,
    };

    if (geometry.scope.location.has_value()) {
        const auto& location = *geometry.scope.location;
        requirement.location.region_kind = location.region_kind;
        requirement.location.shared_region_name = location.shared_region_name;
        requirement.location.side = location.side;
        requirement.location.coordinate_configuration =
            location.coordinate_configuration;
        requirement.location.transform_from_configuration =
            location.transform_from_configuration;
        requirement.location.transform_to_configuration =
            location.transform_to_configuration;
        requirement.location.quadrature_policy_key =
            location.quadrature_policy_key;
    }

    if (geometry.scope.participant_name.has_value() &&
        geometry.scope.region.has_value() &&
        !geometry.scope.region->participant_name.empty() &&
        *geometry.scope.participant_name !=
            geometry.scope.region->participant_name) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .participant_name = *geometry.scope.participant_name,
            .region_name = geometry.scope.region->region_name,
            .message = "geometry terminal owner participant conflicts with region participant",
        });
    }

    if (geometry.scope.participant_name.has_value() &&
        context.hasParticipant(*geometry.scope.participant_name)) {
        const auto participant =
            context.participant(*geometry.scope.participant_name);
        requirement.owner = CouplingGeometryTerminalOwnerProvenance{
            .participant_name = participant.participant_name,
            .system_name = participant.system_name,
        };
    }

    if (geometry.scope.region.has_value()) {
        const auto& region = *geometry.scope.region;
        if (region.shared_region_name.has_value()) {
            if (region.participant_name.empty()) {
                if (isRequired(geometry.requirement)) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .region_name = *region.shared_region_name,
                        .message = "shared-region geometry terminal requirement needs a participant owner",
                    });
                }
            } else if (!context.hasSharedRegion(*region.shared_region_name)) {
                if (isRequired(geometry.requirement)) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .participant_name = region.participant_name,
                        .region_name = *region.shared_region_name,
                        .message = "geometry terminal shared region is missing from the context",
                    });
                }
            } else {
                applyResolvedGeometryRegion(
                    requirement,
                    context.sharedRegion(*region.shared_region_name,
                                         region.participant_name),
                    region.shared_region_name);
            }
        } else if (!context.hasRegion(region.participant_name,
                                      region.region_name)) {
            reportMissingGeometryRegion(declaration,
                                        geometry,
                                        region,
                                        result);
        } else {
            applyResolvedGeometryRegion(
                requirement,
                context.region(region.participant_name, region.region_name),
                std::nullopt);
        }
    }

    if (!geometry.scope.location.has_value()) {
        if (requirement.owner.has_value() &&
            requirement.owner->region_name.has_value() &&
            context.hasRegion(requirement.owner->participant_name,
                              *requirement.owner->region_name)) {
            requirement.location.coordinate_configuration =
                graphGeometryConfiguration(
                    context.region(requirement.owner->participant_name,
                                   *requirement.owner->region_name)
                        .coordinate_configuration);
        } else {
            requirement.location.coordinate_configuration =
                forms::GeometryConfiguration::Reference;
        }
    }

    requirement.domain = toAnalysisDomainKind(requirement.location.region_kind);
    if (!requirement.domain.has_value() && isRequired(geometry.requirement)) {
        result.add(CouplingDiagnostic{
            .severity = CouplingDiagnosticSeverity::Error,
            .contract_name = declaration.contract_name,
            .message = "geometry terminal requirement uses an unsupported region kind",
        });
    }
    return requirement;
}

std::vector<ResolvedGeometryRequirement> resolveDeclaredGeometryRequirements(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    std::vector<ResolvedGeometryRequirement> resolved;
    for (const auto& declaration : declarations) {
        for (const auto& geometry : declaration.geometry_requirements) {
            resolved.push_back(resolveGeometryRequirement(context,
                                                         declaration,
                                                         geometry,
                                                         result));
        }
    }
    return resolved;
}

std::vector<ResolvedExpectedBlock> resolveExpectedBlocks(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    CouplingValidationResult& result)
{
    std::vector<ResolvedExpectedBlock> resolved;
    for (const auto& declaration : declarations) {
        for (const auto& block : declaration.expected_blocks) {
            const auto row =
                resolveVariable(context, declaration, block.residual_row, result);
            const auto col =
                resolveVariable(context, declaration, block.dependency, result);
            if (!row.has_value() || !col.has_value()) {
                continue;
            }
            if (block.expected_nonzero && block.expect_matrix_block &&
                !validateMonolithicFieldSystemCompatibility(
                    context,
                    declaration,
                    block.residual_row,
                    block.dependency,
                    "expected monolithic coupling block",
                    result)) {
                continue;
            }
            resolved.push_back(ResolvedExpectedBlock{
                .contract_name = declaration.contract_name,
                .residual_row = *row,
                .dependency = *col,
                .expected_nonzero = block.expected_nonzero,
                .expect_matrix_block = block.expect_matrix_block,
            });
        }
    }
    return resolved;
}

bool sameVariables(const analysis::VariableKey& a_row,
                   const analysis::VariableKey& a_dependency,
                   const analysis::VariableKey& b_row,
                   const analysis::VariableKey& b_dependency)
{
    return a_row == b_row && a_dependency == b_dependency;
}

const ResolvedDeclaredDependency* findDeclaredDependency(
    const std::vector<ResolvedDeclaredDependency>& dependencies,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency)
{
    const auto it = std::find_if(
        dependencies.begin(),
        dependencies.end(),
        [&](const ResolvedDeclaredDependency& declared) {
            return sameVariables(declared.residual_row,
                                 declared.dependency,
                                 row,
                                 dependency);
        });
    return it == dependencies.end() ? nullptr : &*it;
}

const ResolvedExpectedBlock* findExpectedBlock(
    const std::vector<ResolvedExpectedBlock>& blocks,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency)
{
    const auto it = std::find_if(
        blocks.begin(),
        blocks.end(),
        [&](const ResolvedExpectedBlock& block) {
            return sameVariables(block.residual_row,
                                 block.dependency,
                                 row,
                                 dependency);
        });
    return it == blocks.end() ? nullptr : &*it;
}

bool resolvedVariableKey(const analysis::VariableKey& key) noexcept
{
    if (key.kind == analysis::VariableKind::FieldComponent) {
        return key.field_id != INVALID_FIELD_ID;
    }
    return !key.name.empty();
}

bool dependencyModeInfersInstalledForms(
    std::span<const CouplingContractDeclaration> declarations)
{
    return !declarations.empty() &&
           std::all_of(
               declarations.begin(),
               declarations.end(),
               [](const CouplingContractDeclaration& declaration) {
                   return declaration.dependency_declaration_mode ==
                          CouplingDependencyDeclarationMode::InferFromInstalledForms;
               });
}

std::string inferredDependencyContractName(
    std::span<const CouplingContractDeclaration> declarations,
    const CouplingFormAnalysisMetadata& form)
{
    if (declarations.size() == 1u) {
        return declarations.front().contract_name;
    }
    return form.contribution_name;
}

void appendInferredDependency(
    std::vector<ResolvedDeclaredDependency>& dependencies,
    std::string contract_name,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency,
    CouplingDependencyMode mode)
{
    if (!resolvedVariableKey(row) || !resolvedVariableKey(dependency) ||
        findDeclaredDependency(dependencies, row, dependency) != nullptr) {
        return;
    }
    dependencies.push_back(ResolvedDeclaredDependency{
        .contract_name = std::move(contract_name),
        .residual_row = row,
        .dependency = dependency,
        .mode = mode,
    });
}

void appendInferredExpectedBlock(
    std::vector<ResolvedExpectedBlock>& expected_blocks,
    std::string contract_name,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency,
    bool has_matrix,
    bool has_vector)
{
    if ((!has_matrix && !has_vector) ||
        !resolvedVariableKey(row) ||
        !resolvedVariableKey(dependency) ||
        findExpectedBlock(expected_blocks, row, dependency) != nullptr) {
        return;
    }
    expected_blocks.push_back(ResolvedExpectedBlock{
        .contract_name = std::move(contract_name),
        .residual_row = row,
        .dependency = dependency,
        .expected_nonzero = true,
        .expect_matrix_block = has_matrix,
    });
}

void appendInstalledFormDependencyInferences(
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    std::vector<ResolvedDeclaredDependency>& dependencies,
    std::vector<ResolvedExpectedBlock>& expected_blocks)
{
    if (!dependencyModeInfersInstalledForms(declarations)) {
        return;
    }

    for (const auto& form : installed_forms) {
        const auto contract_name =
            inferredDependencyContractName(declarations, form);
        for (const auto& installed : form.installed_dependencies) {
            appendInferredDependency(dependencies,
                                     contract_name,
                                     installed.residual_row,
                                     installed.dependency,
                                     installed.mode);
        }
        for (const auto& block : form.installed_blocks) {
            appendInferredDependency(dependencies,
                                     contract_name,
                                     block.residual_row,
                                     block.dependency,
                                     CouplingDependencyMode::ImplicitMonolithic);
            appendInferredExpectedBlock(expected_blocks,
                                        contract_name,
                                        block.residual_row,
                                        block.dependency,
                                        block.has_matrix,
                                        block.has_vector);
        }
    }
}

bool variableReferencesField(const analysis::VariableKey& variable,
                             FieldId field) noexcept
{
    return variable.kind == analysis::VariableKind::FieldComponent &&
           variable.field_id == field;
}

bool hasVariableDependencyEvidence(const CouplingFormAnalysisMetadata& form,
                                   const analysis::VariableKey& row,
                                   const analysis::VariableKey& dependency)
{
    return std::any_of(
        form.variable_dependencies.begin(),
        form.variable_dependencies.end(),
        [&](const CouplingFormVariableDependencyProvenance& variable) {
            return sameVariables(variable.residual_row,
                                 variable.dependency,
                                 row,
                                 dependency);
        });
}

bool hasFieldUseDependencyEvidence(const CouplingFormAnalysisMetadata& form,
                                   const analysis::VariableKey& row,
                                   const analysis::VariableKey& dependency)
{
    return std::any_of(
        form.field_uses.begin(),
        form.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return variableReferencesField(row, field.residual_row) &&
                   variableReferencesField(dependency, field.field) &&
                   (field.appears_as_state_field ||
                    field.appears_as_discrete_field ||
                    field.appears_as_geometry_sensitivity);
        });
}

bool hasInstalledDependencyEvidence(
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency)
{
    for (const auto& form : installed_forms) {
        for (const auto& installed : form.installed_dependencies) {
            if (sameVariables(installed.residual_row,
                              installed.dependency,
                              row,
                              dependency)) {
                return true;
            }
        }
        for (const auto& block : form.installed_blocks) {
            if (sameVariables(block.residual_row, block.dependency, row, dependency)) {
                return true;
            }
        }
        if (hasVariableDependencyEvidence(form, row, dependency) ||
            hasFieldUseDependencyEvidence(form, row, dependency)) {
            return true;
        }
    }
    return false;
}

bool hasInstalledMatrixEvidence(
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency)
{
    for (const auto& form : installed_forms) {
        for (const auto& installed : form.installed_dependencies) {
            if (sameVariables(installed.residual_row,
                              installed.dependency,
                              row,
                              dependency) &&
                installed.contributes_matrix_block) {
                return true;
            }
        }
        for (const auto& block : form.installed_blocks) {
            if (sameVariables(block.residual_row, block.dependency, row, dependency) &&
                block.has_matrix) {
                return true;
            }
        }
        for (const auto& variable : form.variable_dependencies) {
            if (sameVariables(variable.residual_row,
                              variable.dependency,
                              row,
                              dependency) &&
                variable.contributes_matrix_block) {
                return true;
            }
        }
    }
    return false;
}

bool hasInstalledNonzeroEvidence(
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    const analysis::VariableKey& row,
    const analysis::VariableKey& dependency)
{
    for (const auto& form : installed_forms) {
        for (const auto& installed : form.installed_dependencies) {
            if (sameVariables(installed.residual_row,
                              installed.dependency,
                              row,
                              dependency) &&
                (installed.contributes_matrix_block || installed.contributes_vector)) {
                return true;
            }
        }
        for (const auto& block : form.installed_blocks) {
            if (sameVariables(block.residual_row, block.dependency, row, dependency) &&
                (block.has_matrix || block.has_vector)) {
                return true;
            }
        }
        for (const auto& variable : form.variable_dependencies) {
            if (sameVariables(variable.residual_row,
                              variable.dependency,
                              row,
                              dependency) &&
                (variable.contributes_matrix_block || variable.contributes_vector)) {
                return true;
            }
        }
        if (hasFieldUseDependencyEvidence(form, row, dependency)) {
            return true;
        }
    }
    return false;
}

std::string temporalSymbolLabel(const CouplingFormTemporalProvenance& temporal)
{
    std::ostringstream os;
    os << toString(temporal.quantity);
    if (temporal.field.has_value()) {
        os << " field=" << *temporal.field;
    }
    if (temporal.quantity == CouplingTemporalQuantity::FieldDerivative) {
        os << " order=" << temporal.derivative_order;
    }
    if (temporal.quantity == CouplingTemporalQuantity::FieldHistoryValue) {
        os << " history=" << temporal.history_index;
    }
    return os.str();
}

bool temporalFieldMatches(std::optional<FieldId> declared,
                          std::optional<FieldId> used) noexcept
{
    if (!declared.has_value() || !used.has_value()) {
        return true;
    }
    return *declared == *used;
}

bool meshScopeRegionMatches(const CouplingRegionEndpointDeclaration& declared,
                            const CouplingRegionEndpointDeclaration& used)
{
    return (declared.participant_name.empty() ||
            declared.participant_name == used.participant_name) &&
           (declared.region_name.empty() ||
            declared.region_name == used.region_name) &&
           (!declared.shared_region_name.has_value() ||
            declared.shared_region_name == used.shared_region_name);
}

bool meshScopeLocationMatches(
    const CouplingGeometryTerminalLocationDeclaration& declared,
    const CouplingGeometryTerminalLocationDeclaration& used)
{
    return declared.region_kind == used.region_kind &&
           declared.shared_region_name == used.shared_region_name &&
           (declared.side == CouplingInterfaceSide::None ||
            declared.side == used.side) &&
           declared.coordinate_configuration == used.coordinate_configuration &&
           declared.transform_from_configuration ==
               used.transform_from_configuration &&
           declared.transform_to_configuration == used.transform_to_configuration &&
           (declared.quadrature_policy_key == 0u ||
            declared.quadrature_policy_key == used.quadrature_policy_key);
}

bool meshTemporalScopeMatchesRequirement(
    const CouplingFormTemporalProvenance& symbol,
    const ResolvedTemporalRequirement& requirement)
{
    if (!requirement.mesh_motion_scope.has_value()) {
        return true;
    }
    if (!symbol.mesh_motion_scope.has_value()) {
        return false;
    }

    const auto& declared = *requirement.mesh_motion_scope;
    const auto& used = *symbol.mesh_motion_scope;
    if (const auto participant_name = meshScopeParticipant(declared);
        participant_name.has_value()) {
        if (const auto used_participant_name = meshScopeParticipant(used);
            !used_participant_name.has_value() ||
            *used_participant_name != *participant_name) {
            return false;
        }
    }
    if (declared.region.has_value()) {
        if (!used.region.has_value() ||
            !meshScopeRegionMatches(*declared.region, *used.region)) {
            return false;
        }
    }
    if (declared.location.has_value()) {
        if (!used.location.has_value() ||
            !meshScopeLocationMatches(*declared.location, *used.location)) {
            return false;
        }
    }
    return true;
}

bool meshTemporalRoleMatchesRequirement(
    const CouplingFormTemporalProvenance& symbol,
    const ResolvedTemporalRequirement& requirement)
{
    return !requirement.mesh_motion_role.has_value() ||
           (symbol.mesh_motion_role.has_value() &&
            *symbol.mesh_motion_role == *requirement.mesh_motion_role);
}

bool temporalSymbolMatchesRequirement(
    const CouplingFormTemporalProvenance& symbol,
    const ResolvedTemporalRequirement& requirement)
{
    if (symbol.quantity != requirement.quantity ||
        !temporalFieldMatches(requirement.field, symbol.field)) {
        return false;
    }

    switch (symbol.quantity) {
    case CouplingTemporalQuantity::FieldDerivative:
        return symbol.derivative_order == requirement.derivative_order;
    case CouplingTemporalQuantity::FieldHistoryValue:
        return symbol.history_index == requirement.history_index;
    case CouplingTemporalQuantity::Time:
    case CouplingTemporalQuantity::TimeStep:
    case CouplingTemporalQuantity::EffectiveTimeStep:
    case CouplingTemporalQuantity::MeshVelocity:
    case CouplingTemporalQuantity::MeshAcceleration:
    case CouplingTemporalQuantity::PreviousMeshVelocity:
    case CouplingTemporalQuantity::PredictedMeshVelocity:
        return meshTemporalRoleMatchesRequirement(symbol, requirement) &&
               meshTemporalScopeMatchesRequirement(symbol, requirement);
    }
    return false;
}

bool hasDeclaredTemporalSymbol(
    const std::vector<ResolvedTemporalRequirement>& temporal_requirements,
    const CouplingFormTemporalProvenance& symbol)
{
    return std::any_of(
        temporal_requirements.begin(),
        temporal_requirements.end(),
        [&](const ResolvedTemporalRequirement& requirement) {
            return temporalSymbolMatchesRequirement(symbol, requirement);
        });
}

void validateTemporalSymbolEvidence(
    const std::vector<ResolvedTemporalRequirement>& temporal_requirements,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    CouplingValidationResult& result)
{
    for (const auto& form : installed_forms) {
        for (const auto& symbol : form.temporal_symbols) {
            if (hasDeclaredTemporalSymbol(temporal_requirements, symbol)) {
                continue;
            }
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .message = "installed form temporal symbol has no declared temporal requirement: " +
                           form.contribution_name + " " +
                           temporalSymbolLabel(symbol),
            });
        }
    }
}

std::string geometryTerminalLabel(
    const CouplingFormGeometryTerminalProvenance& geometry)
{
    std::ostringstream os;
    os << toString(geometry.quantity)
       << " domain=" << analysis::toString(geometry.analysis_domain);
    os << " region=" << toString(geometry.location.region_kind);
    if (geometry.location.marker >= 0) {
        os << " marker=" << geometry.location.marker;
    }
    if (geometry.location.shared_region_name.has_value()) {
        os << " shared=" << *geometry.location.shared_region_name;
    }
    return os.str();
}

bool geometryOwnerMatchesRequirement(
    const CouplingFormGeometryTerminalProvenance& terminal,
    const ResolvedGeometryRequirement& requirement)
{
    if (!requirement.owner.has_value()) {
        return true;
    }
    if (!terminal.owner.has_value()) {
        return false;
    }
    const auto& declared = *requirement.owner;
    const auto& used = *terminal.owner;
    return (declared.participant_name.empty() ||
            declared.participant_name == used.participant_name) &&
           (declared.system_name.empty() ||
            declared.system_name == used.system_name) &&
           (!declared.region_name.has_value() ||
            declared.region_name == used.region_name) &&
           (!declared.shared_region_name.has_value() ||
            declared.shared_region_name == used.shared_region_name);
}

bool geometryTerminalMatchesRequirement(
    const CouplingFormGeometryTerminalProvenance& terminal,
    const ResolvedGeometryRequirement& requirement)
{
    if (terminal.quantity != requirement.quantity) {
        return false;
    }
    if (requirement.domain.has_value() &&
        terminal.analysis_domain != *requirement.domain) {
        return false;
    }
    if (terminal.location.region_kind != requirement.location.region_kind ||
        terminal.location.coordinate_configuration !=
            requirement.location.coordinate_configuration ||
        terminal.location.transform_from_configuration !=
            requirement.location.transform_from_configuration ||
        terminal.location.transform_to_configuration !=
            requirement.location.transform_to_configuration) {
        return false;
    }
    if (requirement.location.marker >= 0 &&
        terminal.location.marker != requirement.location.marker) {
        return false;
    }
    if (requirement.location.shared_region_name.has_value() &&
        terminal.location.shared_region_name !=
            requirement.location.shared_region_name) {
        return false;
    }
    if (requirement.location.side != CouplingInterfaceSide::None &&
        terminal.location.side != requirement.location.side) {
        return false;
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (requirement.location.logical_region.has_value() &&
        (!terminal.location.logical_region.has_value() ||
         !requirement.location.logical_region->compatible_with(
             *terminal.location.logical_region))) {
        return false;
    }
#endif
    if (requirement.location.geometry_revision != 0u &&
        terminal.location.geometry_revision !=
            requirement.location.geometry_revision) {
        return false;
    }
    if (requirement.location.quadrature_policy_key != 0u &&
        terminal.location.quadrature_policy_key !=
            requirement.location.quadrature_policy_key) {
        return false;
    }
    if (!geometryOwnerMatchesRequirement(terminal, requirement)) {
        return false;
    }
    return true;
}

bool hasDeclaredGeometryTerminal(
    const std::vector<ResolvedGeometryRequirement>& geometry_requirements,
    const CouplingFormGeometryTerminalProvenance& terminal)
{
    return std::any_of(
        geometry_requirements.begin(),
        geometry_requirements.end(),
        [&](const ResolvedGeometryRequirement& requirement) {
            return geometryTerminalMatchesRequirement(terminal, requirement);
        });
}

bool geometryTerminalHasAvailableData(
    const CouplingFormGeometryTerminalProvenance& terminal) noexcept
{
    return terminal.value_available ||
           terminal.gradient_or_jacobian_available ||
           terminal.normal_available ||
           terminal.measure_available;
}

void validateGeometryTerminalEvidence(
    const std::vector<ResolvedGeometryRequirement>& geometry_requirements,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    CouplingValidationResult& result)
{
    for (const auto& form : installed_forms) {
        for (const auto& terminal : form.geometry_terminals) {
            if (!hasDeclaredGeometryTerminal(geometry_requirements, terminal)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .message = "installed form geometry terminal has no declared geometry requirement: " +
                               form.contribution_name + " " +
                               geometryTerminalLabel(terminal),
                });
            }
            if (!geometryTerminalHasAvailableData(terminal)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .message = "installed form geometry terminal has no available geometry data: " +
                               form.contribution_name + " " +
                               geometryTerminalLabel(terminal),
                });
            }
        }
    }
}

bool installedFormFeatureAvailable(const CouplingFormAnalysisMetadata& form,
                                   analysis::FormBridgeFeature feature) noexcept
{
    const auto it = std::find_if(
        form.feature_gates.begin(),
        form.feature_gates.end(),
        [feature](const auto& gate) {
            return gate.feature == feature;
        });
    return it != form.feature_gates.end() &&
           it->status == analysis::FormBridgeFeatureStatus::Available;
}

bool anyInstalledBlocks(std::span<const CouplingFormAnalysisMetadata> installed_forms)
{
    return std::any_of(
        installed_forms.begin(),
        installed_forms.end(),
        [](const auto& form) {
            return !form.installed_blocks.empty();
        });
}

void addMissingBridgeGateDiagnostic(const CouplingFormAnalysisMetadata& form,
                                    analysis::FormBridgeFeature feature,
                                    CouplingValidationResult& result)
{
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .message = std::string{"installed-form validators require public bridge feature gate: "} +
                   form.contribution_name + " " + analysis::toString(feature),
    });
}

void validateInstalledFormBridgeReadiness(
    const std::vector<ResolvedDeclaredDependency>& declared_dependencies,
    const std::vector<ResolvedExpectedBlock>& expected_blocks,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    CouplingValidationResult& result)
{
    if (installed_forms.empty()) {
        return;
    }

    const bool dependency_validators_enabled =
        !declared_dependencies.empty() || !expected_blocks.empty();
    const bool block_validators_enabled =
        anyInstalledBlocks(installed_forms) ||
        std::any_of(expected_blocks.begin(),
                    expected_blocks.end(),
                    [](const auto& block) {
                        return block.expected_nonzero && block.expect_matrix_block;
                    });

    for (const auto& form : installed_forms) {
        if (dependency_validators_enabled &&
            !installedFormFeatureAvailable(
                form,
                analysis::FormBridgeFeature::InstalledDependencies)) {
            addMissingBridgeGateDiagnostic(
                form,
                analysis::FormBridgeFeature::InstalledDependencies,
                result);
        }
        if (block_validators_enabled &&
            !installedFormFeatureAvailable(
                form,
                analysis::FormBridgeFeature::InstalledBlocks)) {
            addMissingBridgeGateDiagnostic(
                form,
                analysis::FormBridgeFeature::InstalledBlocks,
                result);
        }
    }
}

bool sameTemporalSlot(const CouplingTemporalSlotDescriptor& lhs,
                      const CouplingTemporalSlotDescriptor& rhs) noexcept
{
    return lhs.slot == rhs.slot &&
           lhs.history_index == rhs.history_index &&
           lhs.stage_index == rhs.stage_index;
}

bool sameEndpointRef(const CouplingEndpointRef& lhs,
                     const CouplingEndpointRef& rhs)
{
    return lhs.kind == rhs.kind &&
           lhs.participant_name == rhs.participant_name &&
           lhs.endpoint_name == rhs.endpoint_name &&
           sameTemporalSlot(lhs.temporal, rhs.temporal);
}

bool sameGroupHint(const CouplingGroupHint& lhs,
                   const CouplingGroupHint& rhs)
{
    return lhs.name == rhs.name &&
           lhs.participant_names == rhs.participant_names;
}

bool partitionedExchangeMatches(const CouplingExchangeDeclaration& declared,
                                const CouplingExchange& generated)
{
    if (declared.producer_port != generated.producer_port ||
        declared.consumer_port != generated.consumer_port ||
        !couplingValueDescriptorsCompatible(declared.value, generated.value) ||
        declared.shared_region_name != generated.shared_region_name ||
        declared.transfer.kind != generated.transfer.kind ||
        declared.transfer.driver_owned_name != generated.transfer.driver_owned_name) {
        return false;
    }
    if (declared.producer.has_value() &&
        !sameEndpointRef(*declared.producer, generated.producer.declaration_provenance)) {
        return false;
    }
    if (declared.consumer.has_value() &&
        !sameEndpointRef(*declared.consumer, generated.consumer.declaration_provenance)) {
        return false;
    }
    return true;
}

void validatePartitionedPlanCoverage(
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingExchangeDeclaration> exchange_templates,
    const PartitionedCouplingPlan& partitioned_plan,
    CouplingValidationResult& result)
{
    std::vector<CouplingExchangeDeclaration> declared_exchanges;
    std::vector<CouplingGroupHint> declared_group_hints;
    for (const auto& declaration : declarations) {
        declared_exchanges.insert(
            declared_exchanges.end(),
            declaration.partitioned_exchange_declarations.begin(),
            declaration.partitioned_exchange_declarations.end());
        declared_group_hints.insert(declared_group_hints.end(),
                                    declaration.group_hints.begin(),
                                    declaration.group_hints.end());
    }
    declared_exchanges.insert(declared_exchanges.end(),
                              exchange_templates.begin(),
                              exchange_templates.end());

    for (const auto& declared : declared_exchanges) {
        const auto found = std::any_of(
            partitioned_plan.exchanges.begin(),
            partitioned_plan.exchanges.end(),
            [&declared](const CouplingExchange& generated) {
                return partitionedExchangeMatches(declared, generated);
            });
        if (!found) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declared.producer_port.contract_instance_name,
                .message = "declared partitioned exchange is missing from generated plan",
            });
        }
    }

    for (const auto& generated : partitioned_plan.exchanges) {
        const auto found = std::any_of(
            declared_exchanges.begin(),
            declared_exchanges.end(),
            [&generated](const CouplingExchangeDeclaration& declared) {
                return partitionedExchangeMatches(declared, generated);
            });
        if (!found) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = generated.producer_port.contract_instance_name,
                .message = "generated partitioned exchange has no declaration",
            });
        }
    }

    for (const auto& declared : declared_group_hints) {
        const auto found = std::any_of(
            partitioned_plan.group_hints.begin(),
            partitioned_plan.group_hints.end(),
            [&declared](const CouplingGroupHint& generated) {
                return sameGroupHint(declared, generated);
            });
        if (!found) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .message = "declared partitioned group hint is missing from generated plan",
            });
        }
    }
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
void validatePartitionedInterfaceRuntimeHandles(
    const CouplingContext& context,
    const PartitionedCouplingPlan& partitioned_plan,
    CouplingValidationResult& result)
{
    for (const auto& exchange : partitioned_plan.exchanges) {
        if (!exchange.transfer.interface_map.has_value()) {
            continue;
        }
        const auto& provenance = *exchange.transfer.interface_map;
        if (!exchange.producer.system_name.empty() &&
            exchange.producer.system_name != provenance.source_system_name) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = exchange.producer_port.contract_instance_name,
                .endpoint_name = exchange.producer_port.port_name,
                .message = "resolved partitioned interface transfer source system does not match the producer endpoint",
            });
        }
        if (!exchange.consumer.system_name.empty() &&
            exchange.consumer.system_name != provenance.target_system_name) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = exchange.consumer_port.contract_instance_name,
                .endpoint_name = exchange.consumer_port.port_name,
                .message = "resolved partitioned interface transfer target system does not match the consumer endpoint",
            });
        }

        try {
            static_cast<void>(context.interfaceMapHandles(provenance));
        } catch (const std::exception& e) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = exchange.producer_port.contract_instance_name,
                .endpoint_name = exchange.producer_port.port_name,
                .message = "resolved partitioned interface transfer has invalid runtime map handles: " +
                           std::string(e.what()),
            });
        }
    }
}
#endif

void validateFinalizedDependencyEvidence(
    const std::vector<ResolvedDeclaredDependency>& declared_dependencies,
    const std::vector<ResolvedExpectedBlock>& expected_blocks,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    CouplingValidationResult& result)
{
    validateInstalledFormBridgeReadiness(
        declared_dependencies,
        expected_blocks,
        installed_forms,
        result);

    for (const auto& dependency : declared_dependencies) {
        if (dependency.mode != CouplingDependencyMode::ImplicitMonolithic) {
            continue;
        }
        if (!hasInstalledDependencyEvidence(installed_forms,
                                            dependency.residual_row,
                                            dependency.dependency)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = dependency.contract_name,
                .message = "declared implicit coupling dependency is not reported by installed metadata: " +
                           variableLabel(dependency.residual_row) + " depends on " +
                           variableLabel(dependency.dependency),
            });
        }
    }

    for (const auto& block : expected_blocks) {
        const auto* declared = findDeclaredDependency(
            declared_dependencies,
            block.residual_row,
            block.dependency);
        if (declared == nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = block.contract_name,
                .message = "expected coupling block has no declared dependency: " +
                           variableLabel(block.residual_row) + " depends on " +
                           variableLabel(block.dependency),
            });
            continue;
        }
        if (declared->mode == CouplingDependencyMode::ExternalLagged &&
            block.expected_nonzero && block.expect_matrix_block) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = block.contract_name,
                .message = "external or lagged coupling dependency cannot require a monolithic matrix block: " +
                           variableLabel(block.residual_row) + " depends on " +
                           variableLabel(block.dependency),
            });
            continue;
        }
        if (!block.expected_nonzero) {
            if (hasInstalledNonzeroEvidence(installed_forms,
                                            block.residual_row,
                                            block.dependency)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = block.contract_name,
                    .message = "expected zero coupling block has installed evidence: " +
                               variableLabel(block.residual_row) + " depends on " +
                               variableLabel(block.dependency),
                });
            }
            continue;
        }
        if (block.expect_matrix_block) {
            if (!hasInstalledMatrixEvidence(installed_forms,
                                            block.residual_row,
                                            block.dependency)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = block.contract_name,
                    .message = "expected monolithic block is missing installed matrix evidence: " +
                               variableLabel(block.residual_row) + " depends on " +
                               variableLabel(block.dependency),
                });
            }
        } else if (!hasInstalledDependencyEvidence(installed_forms,
                                                   block.residual_row,
                                                   block.dependency)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = block.contract_name,
                .message = "expected coupling dependency is missing installed evidence: " +
                           variableLabel(block.residual_row) + " depends on " +
                           variableLabel(block.dependency),
            });
        }
    }

    for (const auto& form : installed_forms) {
        for (const auto& block : form.installed_blocks) {
            if (!hasInstalledNonzeroEvidence(installed_forms,
                                             block.residual_row,
                                             block.dependency)) {
                continue;
            }
            const auto* declared = findDeclaredDependency(
                declared_dependencies,
                block.residual_row,
                block.dependency);
            if (declared == nullptr ||
                declared->mode != CouplingDependencyMode::ImplicitMonolithic) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .message = "installed coupling block has no declared implicit dependency: " +
                               form.contribution_name + " " +
                               variableLabel(block.residual_row) + " depends on " +
                               variableLabel(block.dependency),
                });
            }
        }
    }
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

    for (const auto& requirement : declaration.field_requirements) {
        const auto& field = requirement.field;
        if (!context.hasField(field.participant_name, field.field_name)) {
            if (isRequiredFieldRequirement(requirement)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = field.participant_name,
                    .field_name = field.field_name,
                    .message = "required coupling field-shape field is missing from the context",
                });
            }
            continue;
        }

        const auto resolved_field =
            context.field(field.participant_name, field.field_name);
        if (requirement.value.components > 0 &&
            resolved_field.components != requirement.value.components) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field component count does not satisfy the value-shape declaration",
            });
        }
        if (requirement.required_scope.has_value() &&
            resolved_field.scope != *requirement.required_scope) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field scope does not satisfy the value-shape declaration",
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
        const auto resolved_region = context.region(region.participant_name, region.region_name);
        if (region.required_region_kind.has_value() &&
            resolved_region.kind != *region.required_region_kind) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region kind does not satisfy the declaration",
            });
        }
        validateInterfaceRegionTopology(declaration, resolved_region, result);
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
        const auto group = context.sharedRegionGroup(shared_region.shared_region_name);
        for (const auto& participant_region : group.participant_regions) {
            if (shared_region.required_region_kind.has_value()) {
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
            validateInterfaceRegionTopology(declaration, participant_region, result);
        }
    }

    for (const auto& requirement : declaration.shared_interface_requirements) {
        if (!context.hasSharedRegion(requirement.shared_region_name)) {
            if (requirement.require_all_participants) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .region_name = requirement.shared_region_name,
                    .message = "required shared-interface region is missing from the context",
                });
            }
            continue;
        }

        const auto group = context.sharedRegionGroup(requirement.shared_region_name);
        std::vector<const CouplingRegionRef*> resolved_participants;
        for (const auto& participant_name : requirement.participant_names) {
            const auto* participant_region =
                sharedRegionParticipant(group, participant_name);
            if (participant_region == nullptr) {
                if (requirement.require_all_participants) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .participant_name = participant_name,
                        .region_name = requirement.shared_region_name,
                        .message = "shared-interface participant mapping is missing from the context",
                    });
                }
                continue;
            }

            resolved_participants.push_back(participant_region);
            if (participant_region->kind != requirement.required_region_kind) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = participant_name,
                    .region_name = requirement.shared_region_name,
                    .message = "shared-interface participant mapping does not satisfy the declaration",
                });
            }
        }

        if (requirement.require_opposite_sides_for_two_participants &&
            resolved_participants.size() == 2u &&
            (resolved_participants[0]->side == CouplingInterfaceSide::None ||
             resolved_participants[1]->side == CouplingInterfaceSide::None ||
             resolved_participants[0]->side == resolved_participants[1]->side)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .region_name = requirement.shared_region_name,
                .message = "two-participant shared-interface requirement needs opposite nonempty sides",
            });
        }
    }

    for (const auto& requirement : declaration.region_relation_requirements) {
        if (requirement.selected_lowering.has_value()) {
            const auto& request = *requirement.selected_lowering;
            const auto* capability =
                findLoweringCapability(requirement, request.lowering_kind);
            if (capability == nullptr) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message =
                        "selected relation lowering is not declared: relation=" +
                        requirement.relation_name + ", kind=" +
                        toString(requirement.relation_kind) + ", selected=(" +
                        selectedLoweringDescription(request) +
                        "), available=[" +
                        availableLoweringDescription(requirement) + "]",
                });
            } else if (!capability->supported) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message =
                        "selected relation lowering is declared unsupported: relation=" +
                        requirement.relation_name + ", kind=" +
                        toString(requirement.relation_kind) + ", selected=(" +
                        selectedLoweringDescription(request) + "), reason=" +
                        capability->unsupported_reason + ", available=[" +
                        availableLoweringDescription(requirement) + "]",
                });
            } else if (!supportsEnforcementStrategy(
                           *capability,
                           request.enforcement_strategy) ||
                       !supportsPartitionedSolveStrategy(
                           *capability,
                           request.partitioned_solve_strategy)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message =
                        "selected relation lowering strategy is unsupported: relation=" +
                        requirement.relation_name + ", kind=" +
                        toString(requirement.relation_kind) + ", selected=(" +
                        selectedLoweringDescription(request) +
                        "), available=[" +
                        availableLoweringDescription(requirement) + "]",
                });
            }
        }

        std::vector<CouplingRegionRef> resolved_endpoints;
        for (const auto& endpoint : requirement.endpoints) {
            if (!context.hasRegion(endpoint.participant_name,
                                   endpoint.region_name)) {
                if (requirement.require_all_endpoints) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .participant_name = endpoint.participant_name,
                        .region_name = endpoint.region_name,
                        .message = "region-relation endpoint is missing from the context",
                    });
                }
                continue;
            }

            const auto resolved_region =
                context.region(endpoint.participant_name, endpoint.region_name);
            resolved_endpoints.push_back(resolved_region);
            if (requirement.required_region_kind.has_value() &&
                resolved_region.kind != *requirement.required_region_kind) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = endpoint.participant_name,
                    .region_name = endpoint.region_name,
                    .message = "region-relation endpoint kind does not satisfy the declaration",
                });
            }
            if (requirement.require_registered_topology) {
                validateInterfaceRegionTopology(
                    declaration,
                    resolved_region,
                    result);
            }

            if (!endpoint.shared_region_name.has_value()) {
                continue;
            }
            if (!context.hasSharedRegion(*endpoint.shared_region_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = endpoint.participant_name,
                    .region_name = *endpoint.shared_region_name,
                    .message = "region-relation endpoint shared region is missing from the context",
                });
                continue;
            }

            const auto group = context.sharedRegionGroup(*endpoint.shared_region_name);
            const auto* participant_region =
                sharedRegionParticipant(group, endpoint.participant_name);
            if (participant_region == nullptr ||
                participant_region->region_name != endpoint.region_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .participant_name = endpoint.participant_name,
                    .region_name = *endpoint.shared_region_name,
                    .message = "region-relation endpoint shared-region mapping is missing from the context",
                });
            }
        }

        if (requirement.relation_kind ==
                CouplingRegionRelationKind::SidePairedInterface &&
            requirement.require_opposite_sides_for_side_pair &&
            resolved_endpoints.size() == 2u &&
            (resolved_endpoints[0].side == CouplingInterfaceSide::None ||
             resolved_endpoints[1].side == CouplingInterfaceSide::None ||
             resolved_endpoints[0].side == resolved_endpoints[1].side)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .contract_name = declaration.contract_name,
                .message = "side-paired region relation needs opposite nonempty sides",
            });
        }
        if (requirement.require_common_monolithic_system &&
            !resolved_endpoints.empty()) {
            const auto* owner = resolved_endpoints.front().system;
            const auto same_owner = std::all_of(
                resolved_endpoints.begin(),
                resolved_endpoints.end(),
                [owner](const CouplingRegionRef& endpoint) {
                    return endpoint.system == owner;
                });
            if (!same_owner) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message = "region-relation endpoints must resolve to one owning system for monolithic lowering",
                });
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
    installed_forms_.clear();
    populateDeclarationSnapshot(
        snapshot_,
        context,
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()));

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
    validateAdditionalFieldGraphDeclarations(
        context,
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        result);
    PartitionedCouplingPlanGenerator partitioned_validator;
    result.append(partitioned_validator.validate(
        context,
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size())));
    return result;
}

CouplingValidationResult CouplingGraph::buildFinalizedGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms)
{
    CouplingValidationResult result = buildDeclarationGraph(context, declarations);
    installed_forms_.assign(installed_forms.begin(), installed_forms.end());

    auto declared_dependencies =
        resolveDeclaredDependencies(context, declarations_, result);
    const auto declared_temporal_requirements =
        resolveDeclaredTemporalRequirements(context, declarations_, result);
    const auto declared_geometry_requirements =
        resolveDeclaredGeometryRequirements(context, declarations_, result);
    auto expected_blocks = resolveExpectedBlocks(context, declarations_, result);
    appendInstalledFormDependencyInferences(
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        installed_forms,
        declared_dependencies,
        expected_blocks);
    validateRequiredNonFieldGraphVariables(
        context,
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        result);
    validateProviderMetadataRequirements(
        context,
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        installed_forms,
        result);
    validateTemporalSymbolEvidence(
        declared_temporal_requirements,
        installed_forms,
        result);
    validateGeometryTerminalEvidence(
        declared_geometry_requirements,
        installed_forms,
        result);
    validateFinalizedDependencyEvidence(
        declared_dependencies,
        expected_blocks,
        installed_forms,
        result);
    return result;
}

CouplingValidationResult CouplingGraph::buildFinalizedGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    const PartitionedCouplingPlan& partitioned_plan)
{
    CouplingValidationResult result =
        buildFinalizedGraph(context, declarations, installed_forms);
    validatePartitionedPlanCoverage(
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        std::span<const CouplingExchangeDeclaration>{},
        partitioned_plan,
        result);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    validatePartitionedInterfaceRuntimeHandles(context, partitioned_plan, result);
#endif
    appendResolvedPartitionedExchangeNodes(snapshot_, partitioned_plan);
    return result;
}

CouplingValidationResult CouplingGraph::buildFinalizedGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms,
    const PartitionedCouplingPlan& partitioned_plan,
    std::span<const CouplingExchangeDeclaration> exchange_templates)
{
    CouplingValidationResult result =
        buildFinalizedGraph(context, declarations, installed_forms);
    validatePartitionedPlanCoverage(
        std::span<const CouplingContractDeclaration>(declarations_.data(),
                                                     declarations_.size()),
        exchange_templates,
        partitioned_plan,
        result);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    validatePartitionedInterfaceRuntimeHandles(context, partitioned_plan, result);
#endif
    appendResolvedPartitionedExchangeNodes(snapshot_, partitioned_plan);
    return result;
}

CouplingValidationResult CouplingGraph::validateTemporalRequirements(
    const CouplingTemporalAvailability& availability) const
{
    std::vector<CouplingTemporalRequirement> requirements;
    for (const auto& declaration : declarations_) {
        requirements.insert(requirements.end(),
                            declaration.temporal_requirements.begin(),
                            declaration.temporal_requirements.end());
    }
    return coupling::validateTemporalRequirements(
        std::span<const CouplingTemporalRequirement>(requirements),
        availability);
}

CouplingValidationResult CouplingGraph::validateGeometryTerminalRequirements(
    const CouplingContext& context,
    const CouplingGeometryTerminalAvailability& availability) const
{
    std::vector<CouplingGeometryTerminalRequirement> requirements;
    for (const auto& declaration : declarations_) {
        requirements.insert(requirements.end(),
                            declaration.geometry_requirements.begin(),
                            declaration.geometry_requirements.end());
    }
    return coupling::validateGeometryTerminalRequirements(
        context,
        std::span<const CouplingGeometryTerminalRequirement>(requirements),
        availability);
}

const std::vector<CouplingContractDeclaration>& CouplingGraph::declarations() const noexcept
{
    return declarations_;
}

const std::vector<CouplingFormAnalysisMetadata>&
CouplingGraph::installedFormMetadata() const noexcept
{
    return installed_forms_;
}

const CouplingGraphSnapshot& CouplingGraph::snapshot() const noexcept
{
    return snapshot_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
