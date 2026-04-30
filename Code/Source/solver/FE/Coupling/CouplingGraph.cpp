/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGraph.h"

#include "Systems/FESystem.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
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
        return true;
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
}

} // namespace

CouplingValidationResult CouplingGraph::buildDeclarationGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations)
{
    declarations_.assign(declarations.begin(), declarations.end());
    installed_forms_.clear();

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

CouplingValidationResult CouplingGraph::buildFinalizedGraph(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations,
    std::span<const CouplingFormAnalysisMetadata> installed_forms)
{
    CouplingValidationResult result = buildDeclarationGraph(context, declarations);
    installed_forms_.assign(installed_forms.begin(), installed_forms.end());

    const auto declared_dependencies =
        resolveDeclaredDependencies(context, declarations_, result);
    const auto declared_temporal_requirements =
        resolveDeclaredTemporalRequirements(context, declarations_, result);
    const auto declared_geometry_requirements =
        resolveDeclaredGeometryRequirements(context, declarations_, result);
    const auto expected_blocks = resolveExpectedBlocks(context, declarations_, result);
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

} // namespace coupling
} // namespace FE
} // namespace svmp
