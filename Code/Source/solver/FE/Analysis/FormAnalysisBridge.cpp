/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FormAnalysisBridge.h"
#include "Analysis/FormExprScanner.h"

#include <algorithm>
#include <functional>
#include <span>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct TerminalWalkContext {
    DomainKind domain{DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
};

[[nodiscard]] bool isFieldTerminal(FormTerminalKind kind) noexcept
{
    return kind == FormTerminalKind::TestField ||
           kind == FormTerminalKind::TrialField ||
           kind == FormTerminalKind::StateField ||
           kind == FormTerminalKind::DiscreteField;
}

[[nodiscard]] std::optional<VariableKey> graphVariableForTerminal(
    FormTerminalKind kind,
    FieldId field_id,
    const std::string& symbol_name,
    const std::optional<std::uint32_t>& slot)
{
    if (isFieldTerminal(kind) && field_id != INVALID_FIELD_ID) {
        return VariableKey::field(field_id);
    }

    const auto named_or_slot = [&]() -> std::string {
        if (!symbol_name.empty()) {
            return symbol_name;
        }
        if (slot.has_value()) {
            return "slot:" + std::to_string(*slot);
        }
        return {};
    };

    switch (kind) {
        case FormTerminalKind::BoundaryFunctionalSymbol:
        case FormTerminalKind::BoundaryIntegralSymbol:
        case FormTerminalKind::BoundaryIntegralRef: {
            const auto name = named_or_slot();
            if (!name.empty()) {
                return VariableKey::named(VariableKind::BoundaryFunctional, name);
            }
            break;
        }
        case FormTerminalKind::AuxiliaryStateSymbol:
        case FormTerminalKind::AuxiliaryStateRef: {
            const auto name = named_or_slot();
            if (!name.empty()) {
                return VariableKey::named(VariableKind::AuxiliaryState, name);
            }
            break;
        }
        case FormTerminalKind::AuxiliaryInputSymbol:
        case FormTerminalKind::AuxiliaryInputRef: {
            const auto name = named_or_slot();
            if (!name.empty()) {
                return VariableKey::named(VariableKind::AuxiliaryInput, name);
            }
            break;
        }
        case FormTerminalKind::AuxiliaryOutputSymbol:
        case FormTerminalKind::AuxiliaryOutputRef: {
            const auto name = named_or_slot();
            if (!name.empty()) {
                return VariableKey::named(VariableKind::AuxiliaryOutput, name);
            }
            break;
        }
        default:
            break;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<FormTerminalKind> terminalKindForNode(forms::FormExprType type)
{
    using FT = forms::FormExprType;

    switch (type) {
        case FT::TestFunction:
            return FormTerminalKind::TestField;
        case FT::TrialFunction:
            return FormTerminalKind::TrialField;
        case FT::StateField:
            return FormTerminalKind::StateField;
        case FT::DiscreteField:
            return FormTerminalKind::DiscreteField;
        case FT::ParameterSymbol:
            return FormTerminalKind::ParameterSymbol;
        case FT::ParameterRef:
            return FormTerminalKind::ParameterRef;
        case FT::Coefficient:
            return FormTerminalKind::Coefficient;
        case FT::BoundaryFunctionalSymbol:
            return FormTerminalKind::BoundaryFunctionalSymbol;
        case FT::BoundaryIntegralSymbol:
            return FormTerminalKind::BoundaryIntegralSymbol;
        case FT::BoundaryIntegralRef:
            return FormTerminalKind::BoundaryIntegralRef;
        case FT::AuxiliaryStateSymbol:
            return FormTerminalKind::AuxiliaryStateSymbol;
        case FT::AuxiliaryStateRef:
            return FormTerminalKind::AuxiliaryStateRef;
        case FT::AuxiliaryInputSymbol:
            return FormTerminalKind::AuxiliaryInputSymbol;
        case FT::AuxiliaryInputRef:
            return FormTerminalKind::AuxiliaryInputRef;
        case FT::AuxiliaryOutputSymbol:
            return FormTerminalKind::AuxiliaryOutputSymbol;
        case FT::AuxiliaryOutputRef:
            return FormTerminalKind::AuxiliaryOutputRef;
        case FT::MaterialStateOldRef:
            return FormTerminalKind::MaterialStateOldRef;
        case FT::MaterialStateWorkRef:
            return FormTerminalKind::MaterialStateWorkRef;
        case FT::Time:
            return FormTerminalKind::Time;
        case FT::TimeStep:
            return FormTerminalKind::TimeStep;
        case FT::EffectiveTimeStep:
            return FormTerminalKind::EffectiveTimeStep;
        case FT::TimeDerivative:
            return FormTerminalKind::TimeDerivative;
        case FT::PreviousSolutionRef:
            return FormTerminalKind::PreviousSolutionRef;
        case FT::Coordinate:
            return FormTerminalKind::Coordinate;
        case FT::ReferenceCoordinate:
            return FormTerminalKind::ReferenceCoordinate;
        case FT::CurrentCoordinate:
            return FormTerminalKind::CurrentCoordinate;
        case FT::PreviousCoordinate:
            return FormTerminalKind::PreviousCoordinate;
        case FT::ReferencePhysicalCoordinate:
            return FormTerminalKind::ReferencePhysicalCoordinate;
        case FT::MeshDisplacement:
            return FormTerminalKind::MeshDisplacement;
        case FT::MeshVelocity:
            return FormTerminalKind::MeshVelocity;
        case FT::MeshAcceleration:
            return FormTerminalKind::MeshAcceleration;
        case FT::PreviousMeshVelocity:
            return FormTerminalKind::PreviousMeshVelocity;
        case FT::PredictedMeshVelocity:
            return FormTerminalKind::PredictedMeshVelocity;
        case FT::CurrentJacobian:
            return FormTerminalKind::CurrentJacobian;
        case FT::ReferenceJacobian:
            return FormTerminalKind::ReferenceJacobian;
        case FT::CurrentJacobianDeterminant:
            return FormTerminalKind::CurrentJacobianDeterminant;
        case FT::ReferenceJacobianDeterminant:
            return FormTerminalKind::ReferenceJacobianDeterminant;
        case FT::CurrentNormal:
            return FormTerminalKind::CurrentNormal;
        case FT::ReferenceNormal:
            return FormTerminalKind::ReferenceNormal;
        case FT::CurrentMeasure:
            return FormTerminalKind::CurrentMeasure;
        case FT::ReferenceMeasure:
            return FormTerminalKind::ReferenceMeasure;
        case FT::SurfaceJacobian:
            return FormTerminalKind::SurfaceJacobian;
        case FT::GeometryTrialVectorVariation:
            return FormTerminalKind::GeometryTrialVectorVariation;
        case FT::GeometryTrialJacobianVariation:
            return FormTerminalKind::GeometryTrialJacobianVariation;
        case FT::MeshVelocityVariation:
            return FormTerminalKind::MeshVelocityVariation;
        case FT::CurrentMeasureVariation:
            return FormTerminalKind::CurrentMeasureVariation;
        case FT::CurrentNormalVariation:
            return FormTerminalKind::CurrentNormalVariation;
        case FT::SurfaceJacobianVariation:
            return FormTerminalKind::SurfaceJacobianVariation;
        case FT::CellDiameter:
            return FormTerminalKind::CellDiameter;
        case FT::CellVolume:
            return FormTerminalKind::CellVolume;
        case FT::FacetArea:
            return FormTerminalKind::FacetArea;
        case FT::CellDomainId:
            return FormTerminalKind::CellDomainId;
        case FT::Pullback:
        case FT::Pushforward:
            return FormTerminalKind::FrameTransform;
        default:
            return std::nullopt;
    }
}

void populateTimeDerivativeFieldMetadata(const forms::FormExprNode& node,
                                         FormTerminalMetadata& terminal)
{
    if (terminal.kind != FormTerminalKind::TimeDerivative) {
        return;
    }

    const std::function<bool(const forms::FormExprNode&)> visit =
        [&](const forms::FormExprNode& current) {
            if (auto field_id = current.fieldId()) {
                terminal.field_id = *field_id;
                if (auto symbol_name = current.symbolName()) {
                    terminal.symbol_name = std::string(*symbol_name);
                }
                return true;
            }
            for (const auto& child : current.childrenShared()) {
                if (child && visit(*child)) {
                    return true;
                }
            }
            return false;
        };

    for (const auto& child : node.childrenShared()) {
        if (child && visit(*child)) {
            return;
        }
    }
}

void appendTerminal(const forms::FormExprNode& node,
                    const FormAnalysisBridgeOptions& options,
                    const std::string& operator_tag,
                    const TerminalWalkContext& context,
                    std::vector<FormTerminalMetadata>& terminals)
{
    const auto kind = terminalKindForNode(node.type());
    if (!kind.has_value()) {
        return;
    }

    FormTerminalMetadata terminal;
    terminal.kind = *kind;
    terminal.terminal_sequence = terminals.size();
    if (const auto* signature = node.spaceSignature()) {
        terminal.value_type = signature->field_type;
        terminal.value_dimension = signature->value_dimension;
    }
    terminal.owner_system_name = options.system_name;
    terminal.owner_participant_name = options.owner_participant_name;
    terminal.contribution_name = options.contribution_name;
    terminal.origin = options.origin.empty() ? std::string{"Forms"} : options.origin;
    terminal.operator_tag = operator_tag;
    terminal.domain = context.domain;
    terminal.boundary_marker = context.boundary_marker;
    terminal.interface_marker = context.interface_marker;

    if (auto field_id = node.fieldId()) {
        terminal.field_id = *field_id;
    }
    if (auto symbol_name = node.symbolName()) {
        terminal.symbol_name = std::string(*symbol_name);
    } else if (terminal.kind == FormTerminalKind::Coefficient) {
        terminal.symbol_name = node.toString();
    }
    if (auto slot = node.slotIndex()) {
        terminal.slot = *slot;
    }
    if (auto state_offset = node.stateOffsetBytes()) {
        terminal.state_offset_bytes = *state_offset;
    }
    if (auto history_index = node.historyIndex()) {
        terminal.history_index = *history_index;
    }
    if (auto derivative_order = node.timeDerivativeOrder()) {
        terminal.derivative_order = *derivative_order;
    }
    populateTimeDerivativeFieldMetadata(node, terminal);
    if (auto boundary_marker = node.boundaryMarker()) {
        terminal.boundary_marker = *boundary_marker;
        if (terminal.kind == FormTerminalKind::BoundaryFunctionalSymbol) {
            terminal.domain = DomainKind::Boundary;
        }
    }
    if (auto interface_marker = node.interfaceMarker()) {
        terminal.interface_marker = *interface_marker;
    }
    terminal.frame_from = node.fromConfiguration();
    terminal.frame_to = node.toConfiguration();
    terminal.graph_variable = graphVariableForTerminal(
        terminal.kind, terminal.field_id, terminal.symbol_name, terminal.slot);

    terminals.push_back(std::move(terminal));
}

void collectTerminals(const forms::FormExprNode& node,
                      const FormAnalysisBridgeOptions& options,
                      const std::string& operator_tag,
                      TerminalWalkContext context,
                      std::vector<FormTerminalMetadata>& terminals)
{
    using FT = forms::FormExprType;

    switch (node.type()) {
        case FT::CellIntegral:
            context.domain = DomainKind::Cell;
            context.boundary_marker = -1;
            context.interface_marker = -1;
            break;
        case FT::BoundaryIntegral:
            context.domain = DomainKind::Boundary;
            context.boundary_marker = node.boundaryMarker().value_or(-1);
            context.interface_marker = -1;
            break;
        case FT::InteriorFaceIntegral:
            context.domain = DomainKind::InteriorFace;
            context.boundary_marker = -1;
            context.interface_marker = -1;
            break;
        case FT::InterfaceIntegral:
            context.domain = DomainKind::InterfaceFace;
            context.boundary_marker = -1;
            context.interface_marker = node.interfaceMarker().value_or(-1);
            break;
        default:
            break;
    }

    appendTerminal(node, options, operator_tag, context, terminals);

    if (node.type() == FT::BoundaryFunctionalSymbol) {
        context.domain = DomainKind::Boundary;
        context.boundary_marker = node.boundaryMarker().value_or(context.boundary_marker);
        context.interface_marker = -1;
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectTerminals(*child, options, operator_tag, context, terminals);
        }
    }
}

[[nodiscard]] std::string defaultContributionName(const FormulationRecord& formulation)
{
    if (!formulation.operator_tag.empty()) {
        return "formulation:" + formulation.operator_tag;
    }
    return "formulation";
}

void appendFeatureGates(FormContributionAnalysisMetadata& metadata)
{
    const auto add = [&](FormBridgeFeature feature,
                         FormBridgeFeatureStatus status,
                         std::string reason) {
        metadata.feature_gates.push_back(
            FormBridgeFeatureGate{feature, status, std::move(reason)});
    };

    add(FormBridgeFeature::ContributionIdentity,
        metadata.contribution_name_explicit
            ? FormBridgeFeatureStatus::Available
            : FormBridgeFeatureStatus::Partial,
        metadata.contribution_name_explicit
            ? "Caller-supplied contribution name is present."
            : "No caller-supplied contribution name was present; a fallback name was generated.");
    add(FormBridgeFeature::OwningSystem,
        metadata.system_name.empty()
            ? FormBridgeFeatureStatus::Partial
            : FormBridgeFeatureStatus::Available,
        metadata.system_name.empty()
            ? "Owning system name was not supplied by the caller."
            : "Owning system name is present.");
    add(FormBridgeFeature::FieldTerminals,
        FormBridgeFeatureStatus::Available,
        "Field terminals are collected from public Forms node accessors.");
    add(FormBridgeFeature::NonFieldTerminals,
        FormBridgeFeatureStatus::Partial,
        "Non-field terminal kind, slot, name, and graph-variable fallback are reported; provider-specific type data is not resolved here.");
    add(FormBridgeFeature::TemporalTerminals,
        FormBridgeFeatureStatus::Partial,
        "Time, time-step, derivative, and previous-solution terminals are reported; active-trial ownership is resolved by later install context.");
    add(FormBridgeFeature::GeometryTerminals,
        FormBridgeFeatureStatus::Partial,
        "Geometry terminal kind and integration domain are reported; geometry revision and provider metadata require Systems/Assembly extensions.");
    add(FormBridgeFeature::GeometrySensitivity,
        metadata.geometry_sensitivity.mode == forms::GeometrySensitivityMode::MeshMotionUnknowns
            ? FormBridgeFeatureStatus::Partial
            : FormBridgeFeatureStatus::Available,
        metadata.geometry_sensitivity.mode == forms::GeometrySensitivityMode::MeshMotionUnknowns
            ? "Mesh-motion field option is reported; structured sensitivity provenance requires Systems/Assembly extensions."
            : "No geometry-sensitivity field was requested.");
    add(FormBridgeFeature::InstalledDependencies,
        FormBridgeFeatureStatus::Available,
        "Installed dependency evidence is adapted from ContributionDescriptor and FormulationRecord variables.");
    add(FormBridgeFeature::InstalledBlocks,
        FormBridgeFeatureStatus::Available,
        "Installed block evidence is adapted from ContributionDescriptor test/trial variables.");
}

[[nodiscard]] bool dependencyExists(const std::vector<FormInstalledDependencyMetadata>& deps,
                                    const VariableKey& row,
                                    const VariableKey& dependency,
                                    DomainKind domain,
                                    bool matrix_block,
                                    DependencyKind dependency_kind,
                                    const std::string& dependency_name)
{
    return std::any_of(deps.begin(), deps.end(), [&](const auto& dep) {
        return dep.residual_row == row &&
               dep.dependency == dependency &&
               dep.domain == domain &&
               dep.contributes_matrix_block == matrix_block &&
               dep.dependency_kind == dependency_kind &&
               dep.dependency_name == dependency_name;
    });
}

[[nodiscard]] bool blockExists(const std::vector<FormInstalledBlockMetadata>& blocks,
                               const VariableKey& row,
                               const VariableKey& dependency,
                               DomainKind domain)
{
    return std::any_of(blocks.begin(), blocks.end(), [&](const auto& block) {
        return block.residual_row == row &&
               block.dependency == dependency &&
               std::find(block.domains.begin(), block.domains.end(), domain) != block.domains.end();
    });
}

[[nodiscard]] DomainKind defaultDependencyDomain(const VariableKey& dependency) noexcept
{
    switch (dependency.kind) {
        case VariableKind::BoundaryFunctional:
            return DomainKind::CoupledBoundary;
        case VariableKind::AuxiliaryState:
        case VariableKind::AuxiliaryInput:
        case VariableKind::AuxiliaryOutput:
            return DomainKind::AuxiliaryCoupling;
        case VariableKind::GlobalScalar:
            return DomainKind::Global;
        case VariableKind::FieldComponent:
            return DomainKind::Cell;
    }
    return DomainKind::Cell;
}

[[nodiscard]] DependencyKind dependencyKindForVariable(
    const VariableKey& dependency) noexcept
{
    switch (dependency.kind) {
        case VariableKind::FieldComponent:
            return DependencyKind::FieldUnknown;
        case VariableKind::BoundaryFunctional:
            return DependencyKind::BoundaryFunctional;
        case VariableKind::AuxiliaryState:
            return DependencyKind::AuxiliaryState;
        case VariableKind::AuxiliaryInput:
            return DependencyKind::AuxiliaryInput;
        case VariableKind::AuxiliaryOutput:
            return DependencyKind::AuxiliaryOutput;
        case VariableKind::GlobalScalar:
            return DependencyKind::FieldUnknown;
    }
    return DependencyKind::FieldUnknown;
}

[[nodiscard]] std::string dependencyNameForVariable(
    const VariableKey& dependency)
{
    if (dependency.kind == VariableKind::FieldComponent) {
        return "field:" + std::to_string(dependency.field_id);
    }
    return dependency.name;
}

[[nodiscard]] DependencyKind dependencyKindForTerminal(
    FormTerminalKind kind) noexcept
{
    switch (kind) {
        case FormTerminalKind::TrialField:
        case FormTerminalKind::StateField:
        case FormTerminalKind::DiscreteField:
            return DependencyKind::TrialFunction;
        case FormTerminalKind::TestField:
            return DependencyKind::TestFunction;
        case FormTerminalKind::ParameterSymbol:
        case FormTerminalKind::ParameterRef:
            return DependencyKind::Parameter;
        case FormTerminalKind::Coefficient:
            return DependencyKind::Coefficient;
        case FormTerminalKind::BoundaryFunctionalSymbol:
        case FormTerminalKind::BoundaryIntegralSymbol:
        case FormTerminalKind::BoundaryIntegralRef:
            return DependencyKind::BoundaryFunctional;
        case FormTerminalKind::AuxiliaryStateSymbol:
        case FormTerminalKind::AuxiliaryStateRef:
            return DependencyKind::AuxiliaryState;
        case FormTerminalKind::AuxiliaryInputSymbol:
        case FormTerminalKind::AuxiliaryInputRef:
            return DependencyKind::AuxiliaryInput;
        case FormTerminalKind::AuxiliaryOutputSymbol:
        case FormTerminalKind::AuxiliaryOutputRef:
            return DependencyKind::AuxiliaryOutput;
        case FormTerminalKind::TimeStep:
        case FormTerminalKind::EffectiveTimeStep:
            return DependencyKind::TimeStep;
        case FormTerminalKind::CellDiameter:
        case FormTerminalKind::CellVolume:
        case FormTerminalKind::FacetArea:
        case FormTerminalKind::CellDomainId:
            return DependencyKind::MeshMetric;
        case FormTerminalKind::Coordinate:
        case FormTerminalKind::ReferenceCoordinate:
        case FormTerminalKind::CurrentCoordinate:
        case FormTerminalKind::PreviousCoordinate:
        case FormTerminalKind::ReferencePhysicalCoordinate:
        case FormTerminalKind::MeshDisplacement:
        case FormTerminalKind::MeshVelocity:
        case FormTerminalKind::MeshAcceleration:
        case FormTerminalKind::PreviousMeshVelocity:
        case FormTerminalKind::PredictedMeshVelocity:
        case FormTerminalKind::CurrentJacobian:
        case FormTerminalKind::ReferenceJacobian:
        case FormTerminalKind::CurrentJacobianDeterminant:
        case FormTerminalKind::ReferenceJacobianDeterminant:
        case FormTerminalKind::CurrentNormal:
        case FormTerminalKind::ReferenceNormal:
        case FormTerminalKind::CurrentMeasure:
        case FormTerminalKind::ReferenceMeasure:
        case FormTerminalKind::SurfaceJacobian:
        case FormTerminalKind::GeometryTrialVectorVariation:
        case FormTerminalKind::GeometryTrialJacobianVariation:
        case FormTerminalKind::MeshVelocityVariation:
        case FormTerminalKind::CurrentMeasureVariation:
        case FormTerminalKind::CurrentNormalVariation:
        case FormTerminalKind::SurfaceJacobianVariation:
        case FormTerminalKind::FrameTransform:
            return DependencyKind::GeometryMap;
        default:
            return DependencyKind::FieldUnknown;
    }
}

void appendInstalledDependency(FormContributionAnalysisMetadata& metadata,
                               const VariableKey& row,
                               const VariableKey& dependency,
                               DomainKind domain,
                               bool contributes_matrix_block,
                               bool contributes_vector,
                               std::string provider,
                               DependencyKind dependency_kind =
                                   DependencyKind::FieldUnknown,
                               std::string dependency_name = {},
                               int marker = -1,
                               std::string contribution_id = {},
                               bool affects_coefficient = false,
                               bool affects_geometry = false)
{
    if (dependency_kind == DependencyKind::FieldUnknown) {
        dependency_kind = dependencyKindForVariable(dependency);
    }
    if (dependency_name.empty()) {
        dependency_name = dependencyNameForVariable(dependency);
    }
    if (dependencyExists(metadata.installed_dependencies,
                         row,
                         dependency,
                         domain,
                         contributes_matrix_block,
                         dependency_kind,
                         dependency_name)) {
        return;
    }
    FormInstalledDependencyMetadata dependency_metadata;
    dependency_metadata.residual_row = row;
    dependency_metadata.dependency = dependency;
    dependency_metadata.domain = domain;
    dependency_metadata.contributes_matrix_block = contributes_matrix_block;
    dependency_metadata.contributes_vector = contributes_vector;
    dependency_metadata.provider = std::move(provider);
    dependency_metadata.dependency_kind = dependency_kind;
    dependency_metadata.dependency_name = std::move(dependency_name);
    dependency_metadata.row_variable = row;
    dependency_metadata.marker = marker;
    dependency_metadata.contribution_id = std::move(contribution_id);
    dependency_metadata.affects_coefficient = affects_coefficient;
    dependency_metadata.affects_geometry = affects_geometry;
    metadata.installed_dependencies.push_back(std::move(dependency_metadata));
}

[[nodiscard]] std::vector<VariableKey> dependencyRows(
    const FormulationRecord& formulation,
    const std::vector<ContributionDescriptor>& contributions)
{
    std::vector<VariableKey> rows;
    const auto add_row = [&](const VariableKey& row) {
        if (std::find(rows.begin(), rows.end(), row) == rows.end()) {
            rows.push_back(row);
        }
    };

    for (const auto& contribution : contributions) {
        for (const auto& row : contribution.test_variables) {
            add_row(row);
        }
    }

    if (rows.empty()) {
        for (const auto field : formulation.active_fields) {
            add_row(VariableKey::field(field));
        }
    }

    return rows;
}

void appendDependencySet(FormContributionAnalysisMetadata& metadata,
                         std::span<const VariableKey> rows,
                         std::span<const VariableKey> dependencies,
                         DomainKind domain,
                         const std::string& provider)
{
    for (const auto& row : rows) {
        for (const auto& dependency : dependencies) {
            appendInstalledDependency(metadata,
                                      row,
                                      dependency,
                                      domain,
                                      false,
                                      true,
                                      provider);
        }
    }
}

} // namespace

const char* toString(FormBridgeFeature feature) noexcept
{
    switch (feature) {
        case FormBridgeFeature::ContributionIdentity:
            return "ContributionIdentity";
        case FormBridgeFeature::OwningSystem:
            return "OwningSystem";
        case FormBridgeFeature::FieldTerminals:
            return "FieldTerminals";
        case FormBridgeFeature::NonFieldTerminals:
            return "NonFieldTerminals";
        case FormBridgeFeature::TemporalTerminals:
            return "TemporalTerminals";
        case FormBridgeFeature::GeometryTerminals:
            return "GeometryTerminals";
        case FormBridgeFeature::GeometrySensitivity:
            return "GeometrySensitivity";
        case FormBridgeFeature::InstalledDependencies:
            return "InstalledDependencies";
        case FormBridgeFeature::InstalledBlocks:
            return "InstalledBlocks";
    }
    return "Unknown";
}

const char* toString(FormBridgeFeatureStatus status) noexcept
{
    switch (status) {
        case FormBridgeFeatureStatus::Available:
            return "Available";
        case FormBridgeFeatureStatus::Partial:
            return "Partial";
        case FormBridgeFeatureStatus::Unavailable:
            return "Unavailable";
    }
    return "Unknown";
}

const char* toString(FormTerminalKind kind) noexcept
{
    switch (kind) {
        case FormTerminalKind::TestField:
            return "TestField";
        case FormTerminalKind::TrialField:
            return "TrialField";
        case FormTerminalKind::StateField:
            return "StateField";
        case FormTerminalKind::DiscreteField:
            return "DiscreteField";
        case FormTerminalKind::ParameterSymbol:
            return "ParameterSymbol";
        case FormTerminalKind::ParameterRef:
            return "ParameterRef";
        case FormTerminalKind::Coefficient:
            return "Coefficient";
        case FormTerminalKind::BoundaryFunctionalSymbol:
            return "BoundaryFunctionalSymbol";
        case FormTerminalKind::BoundaryIntegralSymbol:
            return "BoundaryIntegralSymbol";
        case FormTerminalKind::BoundaryIntegralRef:
            return "BoundaryIntegralRef";
        case FormTerminalKind::AuxiliaryStateSymbol:
            return "AuxiliaryStateSymbol";
        case FormTerminalKind::AuxiliaryStateRef:
            return "AuxiliaryStateRef";
        case FormTerminalKind::AuxiliaryInputSymbol:
            return "AuxiliaryInputSymbol";
        case FormTerminalKind::AuxiliaryInputRef:
            return "AuxiliaryInputRef";
        case FormTerminalKind::AuxiliaryOutputSymbol:
            return "AuxiliaryOutputSymbol";
        case FormTerminalKind::AuxiliaryOutputRef:
            return "AuxiliaryOutputRef";
        case FormTerminalKind::MaterialStateOldRef:
            return "MaterialStateOldRef";
        case FormTerminalKind::MaterialStateWorkRef:
            return "MaterialStateWorkRef";
        case FormTerminalKind::Time:
            return "Time";
        case FormTerminalKind::TimeStep:
            return "TimeStep";
        case FormTerminalKind::EffectiveTimeStep:
            return "EffectiveTimeStep";
        case FormTerminalKind::TimeDerivative:
            return "TimeDerivative";
        case FormTerminalKind::PreviousSolutionRef:
            return "PreviousSolutionRef";
        case FormTerminalKind::Coordinate:
            return "Coordinate";
        case FormTerminalKind::ReferenceCoordinate:
            return "ReferenceCoordinate";
        case FormTerminalKind::CurrentCoordinate:
            return "CurrentCoordinate";
        case FormTerminalKind::PreviousCoordinate:
            return "PreviousCoordinate";
        case FormTerminalKind::ReferencePhysicalCoordinate:
            return "ReferencePhysicalCoordinate";
        case FormTerminalKind::MeshDisplacement:
            return "MeshDisplacement";
        case FormTerminalKind::MeshVelocity:
            return "MeshVelocity";
        case FormTerminalKind::MeshAcceleration:
            return "MeshAcceleration";
        case FormTerminalKind::PreviousMeshVelocity:
            return "PreviousMeshVelocity";
        case FormTerminalKind::PredictedMeshVelocity:
            return "PredictedMeshVelocity";
        case FormTerminalKind::CurrentJacobian:
            return "CurrentJacobian";
        case FormTerminalKind::ReferenceJacobian:
            return "ReferenceJacobian";
        case FormTerminalKind::CurrentJacobianDeterminant:
            return "CurrentJacobianDeterminant";
        case FormTerminalKind::ReferenceJacobianDeterminant:
            return "ReferenceJacobianDeterminant";
        case FormTerminalKind::CurrentNormal:
            return "CurrentNormal";
        case FormTerminalKind::ReferenceNormal:
            return "ReferenceNormal";
        case FormTerminalKind::CurrentMeasure:
            return "CurrentMeasure";
        case FormTerminalKind::ReferenceMeasure:
            return "ReferenceMeasure";
        case FormTerminalKind::SurfaceJacobian:
            return "SurfaceJacobian";
        case FormTerminalKind::GeometryTrialVectorVariation:
            return "GeometryTrialVectorVariation";
        case FormTerminalKind::GeometryTrialJacobianVariation:
            return "GeometryTrialJacobianVariation";
        case FormTerminalKind::MeshVelocityVariation:
            return "MeshVelocityVariation";
        case FormTerminalKind::CurrentMeasureVariation:
            return "CurrentMeasureVariation";
        case FormTerminalKind::CurrentNormalVariation:
            return "CurrentNormalVariation";
        case FormTerminalKind::SurfaceJacobianVariation:
            return "SurfaceJacobianVariation";
        case FormTerminalKind::CellDiameter:
            return "CellDiameter";
        case FormTerminalKind::CellVolume:
            return "CellVolume";
        case FormTerminalKind::FacetArea:
            return "FacetArea";
        case FormTerminalKind::CellDomainId:
            return "CellDomainId";
        case FormTerminalKind::FrameTransform:
            return "FrameTransform";
    }
    return "Unknown";
}

std::vector<FormTerminalMetadata> collectFormTerminalMetadata(
    const forms::FormExprNode& root,
    const FormAnalysisBridgeOptions& options,
    std::string operator_tag)
{
    std::vector<FormTerminalMetadata> terminals;
    collectTerminals(root, options, operator_tag, TerminalWalkContext{}, terminals);
    return terminals;
}

FormContributionAnalysisMetadata buildFormAnalysisMetadata(
    const FormulationRecord& formulation,
    const std::vector<ContributionDescriptor>& contributions,
    const FormAnalysisBridgeOptions& options)
{
    FormContributionAnalysisMetadata metadata;
    metadata.contribution_name = options.contribution_name.empty()
        ? defaultContributionName(formulation)
        : options.contribution_name;
    metadata.origin = options.origin.empty() ? std::string{"FormsInstaller"} : options.origin;
    metadata.system_name = options.system_name;
    metadata.operator_tag = formulation.operator_tag;
    metadata.contribution_name_explicit = !options.contribution_name.empty();
    metadata.installed_fields = formulation.active_fields;
    metadata.geometry_sensitivity = options.geometry_sensitivity;
    metadata.geometry_sensitivity_provenance =
        options.geometry_sensitivity_provenance;

    FormAnalysisBridgeOptions terminal_options = options;
    terminal_options.contribution_name = metadata.contribution_name;
    terminal_options.origin = metadata.origin;

    if (formulation.residual_expr) {
        metadata.terminals = collectFormTerminalMetadata(
            *formulation.residual_expr, terminal_options, formulation.operator_tag);
    }

    FormExprScanResult residual_scan;
    const bool residual_scan_available = formulation.residual_expr != nullptr;
    if (residual_scan_available) {
        residual_scan = scanFormExpr(*formulation.residual_expr);
    }

    const auto dependency_rows = dependencyRows(formulation, contributions);

    for (const auto& contribution : contributions) {
        for (const auto& row : contribution.test_variables) {
            for (const auto& dependency : contribution.trial_variables) {
                const int marker = contribution.boundary_marker >= 0
                    ? contribution.boundary_marker
                    : contribution.interface_marker;
                appendInstalledDependency(metadata,
                                          row,
                                          dependency,
                                          contribution.domain,
                                          true,
                                          true,
                                          contribution.origin,
                                          DependencyKind::TrialFunction,
                                          {},
                                          marker,
                                          contribution.contribution_id);
                if (!blockExists(metadata.installed_blocks,
                                 row,
                                 dependency,
                                 contribution.domain)) {
                    metadata.installed_blocks.push_back(
                        FormInstalledBlockMetadata{
                            row,
                            dependency,
                            {contribution.domain},
                            true,
                            true,
                            contribution.origin});
                }
            }
        }
    }

    for (const auto& [row, dependency] : formulation.variable_couplings) {
        appendInstalledDependency(metadata,
                                  row,
                                  dependency,
                                  defaultDependencyDomain(dependency),
                                  false,
                                  true,
                                  "FormulationRecord::variable_couplings");
    }

    appendDependencySet(metadata,
                        dependency_rows,
                        formulation.boundary_functional_dependencies,
                        DomainKind::CoupledBoundary,
                        "FormulationRecord::boundary_functional_dependencies");
    appendDependencySet(metadata,
                        dependency_rows,
                        formulation.auxiliary_state_dependencies,
                        DomainKind::AuxiliaryCoupling,
                        "FormulationRecord::auxiliary_state_dependencies");
    appendDependencySet(metadata,
                        dependency_rows,
                        formulation.auxiliary_input_dependencies,
                        DomainKind::AuxiliaryCoupling,
                        "FormulationRecord::auxiliary_input_dependencies");
    appendDependencySet(metadata,
                        dependency_rows,
                        formulation.auxiliary_output_dependencies,
                        DomainKind::AuxiliaryCoupling,
                        "FormulationRecord::auxiliary_output_dependencies");

    const auto& parameter_usages = formulation.parameter_usages.empty() &&
            residual_scan_available
        ? residual_scan.parameter_usages
        : formulation.parameter_usages;
    for (const auto& parameter : parameter_usages) {
        const std::string name = !parameter.name.empty()
            ? parameter.name
            : (parameter.slot ? "slot:" + std::to_string(*parameter.slot)
                              : std::string{"unnamed-parameter"});
        const int marker = parameter.boundary_marker >= 0
            ? parameter.boundary_marker
            : parameter.interface_marker;
        for (const auto& row : dependency_rows) {
            appendInstalledDependency(metadata,
                                      row,
                                      VariableKey::named(
                                          VariableKind::GlobalScalar,
                                          "parameter:" + name),
                                      DomainKind::ParameterDependency,
                                      false,
                                      true,
                                      "FormulationRecord::parameter_usages",
                                      DependencyKind::Parameter,
                                      name,
                                      marker,
                                      {},
                                      true,
                                      false);
        }
    }

    const auto& coefficient_usages = formulation.coefficient_usages.empty() &&
            residual_scan_available
        ? residual_scan.coefficient_usages
        : formulation.coefficient_usages;
    for (const auto& coefficient : coefficient_usages) {
        const std::string name = !coefficient.name.empty()
            ? coefficient.name
            : std::string{"unnamed-coefficient"};
        const int marker = coefficient.boundary_marker >= 0
            ? coefficient.boundary_marker
            : coefficient.interface_marker;
        for (const auto& row : dependency_rows) {
            appendInstalledDependency(metadata,
                                      row,
                                      VariableKey::named(
                                          VariableKind::GlobalScalar,
                                          "coefficient:" + name),
                                      DomainKind::CoefficientDependency,
                                      false,
                                      true,
                                      "FormulationRecord::coefficient_usages",
                                      DependencyKind::Coefficient,
                                      name,
                                      marker,
                                      {},
                                      true,
                                      false);
        }
    }

    for (const auto& terminal : metadata.terminals) {
        if (isFieldTerminal(terminal.kind) || !terminal.graph_variable.has_value()) {
            continue;
        }
        for (const auto& row : dependency_rows) {
            appendInstalledDependency(metadata,
                                      row,
                                      *terminal.graph_variable,
                                      terminal.domain,
                                      false,
                                      true,
                                      "Forms terminal metadata",
                                      dependencyKindForTerminal(terminal.kind),
                                      terminal.symbol_name,
                                      terminal.boundary_marker >= 0
                                          ? terminal.boundary_marker
                                          : terminal.interface_marker,
                                      {},
                                      terminal.kind == FormTerminalKind::Coefficient,
                                      dependencyKindForTerminal(terminal.kind) ==
                                          DependencyKind::GeometryMap);
        }
    }

    appendFeatureGates(metadata);
    return metadata;
}

bool bridgeFeatureAvailable(const FormContributionAnalysisMetadata& metadata,
                            FormBridgeFeature feature) noexcept
{
    const auto it = std::find_if(
        metadata.feature_gates.begin(),
        metadata.feature_gates.end(),
        [feature](const auto& gate) {
            return gate.feature == feature;
        });
    return it != metadata.feature_gates.end() &&
           it->status == FormBridgeFeatureStatus::Available;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
