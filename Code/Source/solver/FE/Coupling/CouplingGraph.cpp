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

namespace {

bool isRequired(CouplingRequirement requirement) noexcept
{
    return requirement == CouplingRequirement::Required;
}

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

std::string scopedNonFieldName(const CouplingVariableUse& variable)
{
    if (variable.participant_name.empty()) {
        return variable.name;
    }
    return variable.participant_name + "/" + variable.name;
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
    if (variable.kind == CouplingVariableKind::Field) {
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
        return analysis::VariableKey::field(
            context.field(variable.participant_name, variable.name).field_id,
            variable.component);
    }

    if (variable.name.empty()) {
        return std::nullopt;
    }
    return analysis::VariableKey::named(toAnalysisVariableKind(variable.kind),
                                        scopedNonFieldName(variable));
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
    }
    return false;
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
    const auto expected_blocks = resolveExpectedBlocks(context, declarations_, result);
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
