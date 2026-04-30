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
#include <iterator>
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

struct ResolvedGeometryTerminalScope {
    std::optional<CouplingRegionRef> region;
    std::optional<std::string> shared_region_name;
    std::optional<CouplingGeometryTerminalOwnerProvenance> owner;
    const systems::FESystem* owner_system{nullptr};
};

std::string additionalFieldSystemName(const CouplingAdditionalFieldDeclaration& field)
{
    return field.namespace_name + "." + field.field_name;
}

bool additionalFieldSelected(const CouplingAdditionalFieldDeclaration& field)
{
    return field.requirement == CouplingRequirement::Required || field.enabled;
}

forms::GeometryConfiguration toFormsGeometryConfiguration(
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

ResolvedGeometryTerminalScope resolveGeometryTerminalScope(
    const CouplingContext& context,
    const CouplingGeometryTerminalScope& scope)
{
    ResolvedGeometryTerminalScope resolved;

    const auto resolve_participant_name =
        [&](const CouplingRegionEndpointDeclaration& region) -> std::string {
        if (!region.participant_name.empty()) {
            return region.participant_name;
        }
        return scope.participant_name.value_or(std::string{});
    };

    if (scope.region.has_value()) {
        const auto& region = *scope.region;
        const auto participant_name = resolve_participant_name(region);
        if (region.shared_region_name.has_value()) {
            FE_THROW_IF(participant_name.empty(), InvalidArgumentException,
                        "shared-region geometry terminal provenance requires a participant owner");
            resolved.region =
                context.sharedRegion(*region.shared_region_name, participant_name);
            resolved.shared_region_name = region.shared_region_name;
        } else if (!region.region_name.empty()) {
            FE_THROW_IF(participant_name.empty(), InvalidArgumentException,
                        "region geometry terminal provenance requires a participant owner");
            resolved.region = context.region(participant_name, region.region_name);
        }
    } else if (scope.location.has_value() &&
               scope.location->shared_region_name.has_value()) {
        resolved.shared_region_name = scope.location->shared_region_name;
        if (scope.participant_name.has_value()) {
            resolved.region = context.sharedRegion(*resolved.shared_region_name,
                                                   *scope.participant_name);
        } else {
            const auto shared =
                context.sharedRegionGroup(*resolved.shared_region_name);
            if (shared.participant_regions.size() == 1u) {
                resolved.region = shared.participant_regions.front();
            }
        }
    }

    if (resolved.region.has_value()) {
        const auto& region = *resolved.region;
        resolved.owner = CouplingGeometryTerminalOwnerProvenance{
            .participant_name = region.participant_name,
            .system_name = region.system_name,
            .region_name = region.region_name,
            .shared_region_name = resolved.shared_region_name,
        };
        resolved.owner_system = region.system;
    } else if (scope.participant_name.has_value()) {
        const auto participant = context.participant(*scope.participant_name);
        resolved.owner = CouplingGeometryTerminalOwnerProvenance{
            .participant_name = participant.participant_name,
            .system_name = participant.system_name,
        };
        resolved.owner_system = participant.system;
    }

    return resolved;
}

void markGeometryTerminalAvailability(
    CouplingFormGeometryTerminalProvenance& provenance)
{
    switch (provenance.quantity) {
    case CouplingGeometryTerminalQuantity::Jacobian:
    case CouplingGeometryTerminalQuantity::JacobianInverse:
    case CouplingGeometryTerminalQuantity::JacobianDeterminant:
    case CouplingGeometryTerminalQuantity::CurrentJacobian:
    case CouplingGeometryTerminalQuantity::ReferenceJacobian:
    case CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant:
    case CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant:
        provenance.gradient_or_jacobian_available = true;
        break;
    case CouplingGeometryTerminalQuantity::Normal:
    case CouplingGeometryTerminalQuantity::CurrentNormal:
    case CouplingGeometryTerminalQuantity::ReferenceNormal:
        provenance.normal_available = true;
        break;
    case CouplingGeometryTerminalQuantity::CurrentMeasure:
    case CouplingGeometryTerminalQuantity::ReferenceMeasure:
    case CouplingGeometryTerminalQuantity::SurfaceJacobian:
        provenance.measure_available = true;
        break;
    case CouplingGeometryTerminalQuantity::MeshDisplacement:
    case CouplingGeometryTerminalQuantity::Coordinate:
    case CouplingGeometryTerminalQuantity::ReferenceCoordinate:
    case CouplingGeometryTerminalQuantity::CurrentCoordinate:
    case CouplingGeometryTerminalQuantity::PreviousCoordinate:
    case CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate:
    case CouplingGeometryTerminalQuantity::CellDiameter:
    case CouplingGeometryTerminalQuantity::CellVolume:
    case CouplingGeometryTerminalQuantity::FacetArea:
    case CouplingGeometryTerminalQuantity::CellDomainId:
        provenance.value_available = true;
        break;
    }
}

CouplingFormGeometryTerminalProvenance resolveGeometryTerminalProvenance(
    const CouplingContext& context,
    const CouplingFormTerminalProvenanceDeclaration& terminal)
{
    FE_THROW_IF(terminal.kind != CouplingFormTerminalProvenanceKind::GeometryTerminal,
                InvalidArgumentException,
                "geometry terminal provenance requires a geometry terminal declaration");

    const CouplingGeometryTerminalScope empty_scope;
    const auto& scope = terminal.scope.value_or(empty_scope);
    const auto resolved_scope = resolveGeometryTerminalScope(context, scope);

    CouplingGeometryTerminalLocationProvenance location;
    if (scope.location.has_value()) {
        const auto& declared = *scope.location;
        location.region_kind = declared.region_kind;
        location.shared_region_name = declared.shared_region_name;
        location.side = declared.side;
        location.coordinate_configuration = declared.coordinate_configuration;
        location.transform_from_configuration =
            declared.transform_from_configuration;
        location.transform_to_configuration = declared.transform_to_configuration;
    }

    if (resolved_scope.region.has_value()) {
        const auto& region = *resolved_scope.region;
        location.region_kind = region.kind;
        location.marker = region.marker;
        if (!location.shared_region_name.has_value()) {
            location.shared_region_name = resolved_scope.shared_region_name;
        }
        if (region.side != CouplingInterfaceSide::None) {
            location.side = region.side;
        }
        if (!scope.location.has_value()) {
            location.coordinate_configuration =
                toFormsGeometryConfiguration(region.coordinate_configuration);
        }
        location.geometry_revision = region.geometry_revision;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        location.logical_region = region.logical_region;
        if (location.geometry_revision == 0u &&
            region.revision_snapshot.has_value()) {
            location.geometry_revision =
                region.revision_snapshot->geometry_revision;
        }
#endif
    }

    const auto domain = toAnalysisDomainKind(location.region_kind);
    FE_THROW_IF(!domain.has_value(), InvalidArgumentException,
                "geometry terminal provenance requires a supported analysis domain");

    CouplingFormGeometryTerminalProvenance provenance;
    provenance.quantity = terminal.geometry_quantity;
    provenance.location = std::move(location);
    provenance.analysis_domain = *domain;
    provenance.owner = resolved_scope.owner;
    provenance.provider = "forms";
    if (terminal.mesh_motion_role.has_value() &&
        resolved_scope.owner_system != nullptr) {
        if (const auto field =
                resolved_scope.owner_system->meshMotionField(*terminal.mesh_motion_role)) {
            provenance.mesh_motion_field = *field;
        }
    }
    markGeometryTerminalAvailability(provenance);
    return provenance;
}

void applySymbolicOptionsDeclaration(
    const CouplingSymbolicOptionsDeclaration& declaration,
    forms::SymbolicOptions& options)
{
    if (declaration.jit.has_value()) {
        options.jit = *declaration.jit;
    }
    if (declaration.simplify_expressions.has_value()) {
        options.simplify_expressions = *declaration.simplify_expressions;
    }
    if (declaration.exploit_sparsity.has_value()) {
        options.exploit_sparsity = *declaration.exploit_sparsity;
    }
    if (declaration.cache_expressions.has_value()) {
        options.cache_expressions = *declaration.cache_expressions;
    }
    if (declaration.verbose.has_value()) {
        options.verbose = *declaration.verbose;
    }
}

void applyGeometrySensitivityDeclaration(
    const CouplingContext& context,
    const CouplingGeometrySensitivityDeclaration& declaration,
    forms::SymbolicOptions& options)
{
    options.geometry_sensitivity.mode = declaration.mode;
    if (declaration.mode == forms::GeometrySensitivityMode::MeshMotionUnknowns) {
        FE_THROW_IF(!declaration.mesh_motion_field.has_value(),
                    InvalidArgumentException,
                    "mesh-motion geometry sensitivity requires a coupling field");
        const auto field = context.field(
            declaration.mesh_motion_field->participant_name,
            declaration.mesh_motion_field->field_name);
        options.geometry_sensitivity.mesh_motion_field = field.field_id;
    } else {
        options.geometry_sensitivity.mesh_motion_field = INVALID_FIELD_ID;
    }
    options.geometry_tangent_path = declaration.tangent_path;
    options.use_symbolic_tangent = declaration.use_symbolic_tangent;
}

void rejectRawFormInstallOptionOverrides(
    const systems::FormInstallOptions& options)
{
    const systems::FormInstallOptions defaults;
    FE_THROW_IF(options.ad_mode != defaults.ad_mode, InvalidArgumentException,
                "coupling form AD mode must be declared through coupling install options");
    FE_THROW_IF(!options.extra_trial_fields.empty(), InvalidArgumentException,
                "coupling form extra trial fields must be declared by name");

    const auto& compiler = options.compiler_options;
    const auto& default_compiler = defaults.compiler_options;
    FE_THROW_IF(compiler.ad_mode != default_compiler.ad_mode,
                InvalidArgumentException,
                "coupling form compiler AD mode cannot be set through raw symbolic options");
    FE_THROW_IF(compiler.use_symbolic_tangent !=
                    default_compiler.use_symbolic_tangent,
                InvalidArgumentException,
                "coupling form symbolic tangent policy must be declared through coupling install options");
    FE_THROW_IF(compiler.geometry_tangent_path !=
                    default_compiler.geometry_tangent_path,
                InvalidArgumentException,
                "coupling form geometry tangent path must be declared through coupling install options");
    FE_THROW_IF(compiler.geometry_sensitivity.mode !=
                        default_compiler.geometry_sensitivity.mode ||
                    compiler.geometry_sensitivity.mesh_motion_field !=
                        default_compiler.geometry_sensitivity.mesh_motion_field,
                InvalidArgumentException,
                "coupling form geometry sensitivity must be declared through coupling install options");
}

systems::FormInstallOptions resolveFormInstallOptions(
    const CouplingContext& context,
    const CouplingFormContribution& contribution)
{
    rejectRawFormInstallOptionOverrides(contribution.install_options);

    systems::FormInstallOptions options;
    const auto& declaration = contribution.install_options_declaration;
    if (declaration.ad_mode.has_value()) {
        options.ad_mode = *declaration.ad_mode;
    }
    applySymbolicOptionsDeclaration(declaration.compiler_options,
                                    options.compiler_options);
    if (declaration.geometry_sensitivity.has_value()) {
        applyGeometrySensitivityDeclaration(
            context,
            *declaration.geometry_sensitivity,
            options.compiler_options);
    }
    return options;
}

const char* fieldTypeName(FieldType type) noexcept
{
    switch (type) {
    case FieldType::Scalar:
        return "scalar";
    case FieldType::Vector:
        return "vector";
    case FieldType::Tensor:
        return "tensor";
    case FieldType::SymmetricTensor:
        return "symmetric_tensor";
    case FieldType::Mixed:
        return "mixed";
    }
    return "unknown";
}

CouplingRegionKind regionKindForAnalysisDomain(
    analysis::DomainKind domain) noexcept
{
    switch (domain) {
    case analysis::DomainKind::Cell:
        return CouplingRegionKind::Domain;
    case analysis::DomainKind::Boundary:
        return CouplingRegionKind::Boundary;
    case analysis::DomainKind::InteriorFace:
        return CouplingRegionKind::InteriorFace;
    case analysis::DomainKind::InterfaceFace:
        return CouplingRegionKind::InterfaceFace;
    case analysis::DomainKind::Global:
    case analysis::DomainKind::CoupledBoundary:
    case analysis::DomainKind::AuxiliaryCoupling:
        return CouplingRegionKind::UserDefined;
    }
    return CouplingRegionKind::UserDefined;
}

std::optional<CouplingFormNonFieldDependencyKind> nonFieldKindForTerminal(
    analysis::FormTerminalKind kind) noexcept
{
    using Kind = analysis::FormTerminalKind;
    switch (kind) {
    case Kind::ParameterSymbol:
    case Kind::ParameterRef:
        return CouplingFormNonFieldDependencyKind::Parameter;
    case Kind::Coefficient:
        return CouplingFormNonFieldDependencyKind::Coefficient;
    case Kind::MaterialStateOldRef:
        return CouplingFormNonFieldDependencyKind::MaterialStateOld;
    case Kind::MaterialStateWorkRef:
        return CouplingFormNonFieldDependencyKind::MaterialStateWork;
    case Kind::BoundaryFunctionalSymbol:
        return CouplingFormNonFieldDependencyKind::BoundaryFunctional;
    case Kind::BoundaryIntegralSymbol:
    case Kind::BoundaryIntegralRef:
        return CouplingFormNonFieldDependencyKind::BoundaryIntegral;
    case Kind::AuxiliaryStateSymbol:
    case Kind::AuxiliaryStateRef:
        return CouplingFormNonFieldDependencyKind::AuxiliaryState;
    case Kind::AuxiliaryInputSymbol:
    case Kind::AuxiliaryInputRef:
        return CouplingFormNonFieldDependencyKind::AuxiliaryInput;
    case Kind::AuxiliaryOutputSymbol:
    case Kind::AuxiliaryOutputRef:
        return CouplingFormNonFieldDependencyKind::AuxiliaryOutput;
    default:
        return std::nullopt;
    }
}

std::optional<CouplingTemporalQuantity> temporalQuantityForTerminal(
    analysis::FormTerminalKind kind) noexcept
{
    using Kind = analysis::FormTerminalKind;
    switch (kind) {
    case Kind::Time:
        return CouplingTemporalQuantity::Time;
    case Kind::TimeStep:
        return CouplingTemporalQuantity::TimeStep;
    case Kind::EffectiveTimeStep:
        return CouplingTemporalQuantity::EffectiveTimeStep;
    case Kind::TimeDerivative:
        return CouplingTemporalQuantity::FieldDerivative;
    case Kind::PreviousSolutionRef:
        return CouplingTemporalQuantity::FieldHistoryValue;
    case Kind::MeshVelocity:
        return CouplingTemporalQuantity::MeshVelocity;
    case Kind::MeshAcceleration:
        return CouplingTemporalQuantity::MeshAcceleration;
    case Kind::PreviousMeshVelocity:
        return CouplingTemporalQuantity::PreviousMeshVelocity;
    case Kind::PredictedMeshVelocity:
        return CouplingTemporalQuantity::PredictedMeshVelocity;
    default:
        return std::nullopt;
    }
}

std::optional<CouplingGeometryTerminalQuantity> geometryQuantityForTerminal(
    analysis::FormTerminalKind kind) noexcept
{
    using Kind = analysis::FormTerminalKind;
    switch (kind) {
    case Kind::MeshDisplacement:
        return CouplingGeometryTerminalQuantity::MeshDisplacement;
    case Kind::Coordinate:
        return CouplingGeometryTerminalQuantity::Coordinate;
    case Kind::ReferenceCoordinate:
        return CouplingGeometryTerminalQuantity::ReferenceCoordinate;
    case Kind::CurrentCoordinate:
        return CouplingGeometryTerminalQuantity::CurrentCoordinate;
    case Kind::PreviousCoordinate:
        return CouplingGeometryTerminalQuantity::PreviousCoordinate;
    case Kind::ReferencePhysicalCoordinate:
        return CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate;
    case Kind::CurrentJacobian:
        return CouplingGeometryTerminalQuantity::CurrentJacobian;
    case Kind::ReferenceJacobian:
        return CouplingGeometryTerminalQuantity::ReferenceJacobian;
    case Kind::CurrentJacobianDeterminant:
        return CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant;
    case Kind::ReferenceJacobianDeterminant:
        return CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant;
    case Kind::CurrentNormal:
        return CouplingGeometryTerminalQuantity::CurrentNormal;
    case Kind::ReferenceNormal:
        return CouplingGeometryTerminalQuantity::ReferenceNormal;
    case Kind::CurrentMeasure:
        return CouplingGeometryTerminalQuantity::CurrentMeasure;
    case Kind::ReferenceMeasure:
        return CouplingGeometryTerminalQuantity::ReferenceMeasure;
    case Kind::SurfaceJacobian:
        return CouplingGeometryTerminalQuantity::SurfaceJacobian;
    case Kind::CellDiameter:
        return CouplingGeometryTerminalQuantity::CellDiameter;
    case Kind::CellVolume:
        return CouplingGeometryTerminalQuantity::CellVolume;
    case Kind::FacetArea:
        return CouplingGeometryTerminalQuantity::FacetArea;
    case Kind::CellDomainId:
        return CouplingGeometryTerminalQuantity::CellDomainId;
    default:
        return std::nullopt;
    }
}

void appendFieldTerminalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormTerminalMetadata& terminal)
{
    if (terminal.field_id == INVALID_FIELD_ID) {
        return;
    }
    auto it = std::find_if(
        adapted.field_uses.begin(),
        adapted.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return field.field == terminal.field_id;
        });
    if (it == adapted.field_uses.end()) {
        adapted.field_uses.push_back(
            CouplingFormFieldProvenance{.field = terminal.field_id});
        it = std::prev(adapted.field_uses.end());
    }

    switch (terminal.kind) {
    case analysis::FormTerminalKind::TestField:
        it->appears_as_test_field = true;
        break;
    case analysis::FormTerminalKind::StateField:
        it->appears_as_state_field = true;
        break;
    case analysis::FormTerminalKind::DiscreteField:
        it->appears_as_discrete_field = true;
        break;
    default:
        break;
    }
    if (adapted.geometry_sensitivity.mesh_motion_field == terminal.field_id) {
        it->appears_as_geometry_sensitivity = true;
    }
}

void appendNonFieldTerminalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormTerminalMetadata& terminal)
{
    const auto kind = nonFieldKindForTerminal(terminal.kind);
    if (!kind.has_value()) {
        return;
    }

    CouplingFormNonFieldDependencyProvenance provenance;
    provenance.kind = *kind;
    provenance.participant_name = terminal.owner_participant_name;
    provenance.system_name = terminal.owner_system_name;
    provenance.name = terminal.symbol_name;
    provenance.domain = terminal.domain;
    provenance.marker = terminal.boundary_marker >= 0
                            ? terminal.boundary_marker
                            : terminal.interface_marker;
    provenance.provider = terminal.provider;
    provenance.value_type = fieldTypeName(terminal.value_type);
    provenance.slot = terminal.slot;
    provenance.byte_offset = terminal.state_offset_bytes;
    adapted.non_field_dependencies.push_back(std::move(provenance));
}

void appendTemporalTerminalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormTerminalMetadata& terminal)
{
    const auto quantity = temporalQuantityForTerminal(terminal.kind);
    if (!quantity.has_value()) {
        return;
    }
    adapted.temporal_symbols.push_back(CouplingFormTemporalProvenance{
        .field = terminal.field_id == INVALID_FIELD_ID
                     ? std::optional<FieldId>{}
                     : std::optional<FieldId>{terminal.field_id},
        .quantity = *quantity,
        .derivative_order = terminal.derivative_order,
        .history_index = terminal.history_index.value_or(0),
    });
}

void appendGeometryTerminalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormTerminalMetadata& terminal)
{
    const auto quantity = geometryQuantityForTerminal(terminal.kind);
    if (!quantity.has_value()) {
        return;
    }

    CouplingFormGeometryTerminalProvenance provenance;
    provenance.quantity = *quantity;
    provenance.location.region_kind =
        regionKindForAnalysisDomain(terminal.domain);
    provenance.location.marker = terminal.boundary_marker >= 0
                                     ? terminal.boundary_marker
                                     : terminal.interface_marker;
    provenance.analysis_domain = terminal.domain;
    if (!terminal.owner_participant_name.empty() ||
        !terminal.owner_system_name.empty()) {
        provenance.owner = CouplingGeometryTerminalOwnerProvenance{
            .participant_name = terminal.owner_participant_name,
            .system_name = terminal.owner_system_name,
        };
    }
    provenance.provider = terminal.provider;
    markGeometryTerminalAvailability(provenance);
    adapted.geometry_terminals.push_back(std::move(provenance));
}

void appendBridgeTerminalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormTerminalMetadata& terminal)
{
    appendFieldTerminalMetadata(adapted, terminal);
    appendNonFieldTerminalMetadata(adapted, terminal);
    appendTemporalTerminalMetadata(adapted, terminal);
    appendGeometryTerminalMetadata(adapted, terminal);
}

void appendGeometrySensitivityMetadata(
    CouplingFormAnalysisMetadata& adapted)
{
    if (adapted.geometry_sensitivity.mode !=
            forms::GeometrySensitivityMode::MeshMotionUnknowns ||
        adapted.geometry_sensitivity.mesh_motion_field == INVALID_FIELD_ID) {
        return;
    }

    const auto mesh_motion_field =
        adapted.geometry_sensitivity.mesh_motion_field;
    auto field_it = std::find_if(
        adapted.field_uses.begin(),
        adapted.field_uses.end(),
        [&](const CouplingFormFieldProvenance& field) {
            return field.field == mesh_motion_field;
        });
    if (field_it == adapted.field_uses.end()) {
        adapted.field_uses.push_back(
            CouplingFormFieldProvenance{.field = mesh_motion_field});
        field_it = std::prev(adapted.field_uses.end());
    }
    field_it->appears_as_geometry_sensitivity = true;

    adapted.geometry_sensitivity_provenance.push_back(
        CouplingGeometrySensitivityProvenance{
            .kind =
                CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns,
            .mesh_motion_field = mesh_motion_field,
            .provenance_id = "forms.mesh_motion_unknowns",
            .construction_policy = "form_install_options",
            .target_kind = "mesh_motion",
            .ad_compatible = true,
            .location_sensitivity_available = true,
            .jacobian_sensitivity_available = true,
            .measure_sensitivity_available = true,
            .normal_sensitivity_available = true,
            .geometry_fields = {mesh_motion_field},
            .visible_to_assembly_paths = {"forms"},
        });
}

bool resolvedContributionDeclaresField(
    const ResolvedCouplingFormContribution& contribution,
    FieldId field)
{
    return std::find(contribution.fields.begin(), contribution.fields.end(), field) !=
               contribution.fields.end() ||
           std::find(contribution.extra_trial_fields.begin(),
                     contribution.extra_trial_fields.end(),
                     field) != contribution.extra_trial_fields.end();
}

bool resolvedContributionDeclaresTemporal(
    const ResolvedCouplingFormContribution& contribution,
    const CouplingFormTemporalProvenance& temporal)
{
    if (temporal.quantity == CouplingTemporalQuantity::FieldHistoryValue) {
        return std::find_if(
                   contribution.terminal_provenance.begin(),
                   contribution.terminal_provenance.end(),
                   [&](const CouplingFormTerminalProvenanceDeclaration& declared) {
                       return declared.kind ==
                                  CouplingFormTerminalProvenanceKind::
                                      PreviousSolution &&
                              declared.history_index == temporal.history_index;
                   }) != contribution.terminal_provenance.end();
    }
    if (temporal.quantity == CouplingTemporalQuantity::MeshVelocity ||
        temporal.quantity == CouplingTemporalQuantity::MeshAcceleration ||
        temporal.quantity == CouplingTemporalQuantity::PreviousMeshVelocity ||
        temporal.quantity == CouplingTemporalQuantity::PredictedMeshVelocity) {
        return std::find_if(
                   contribution.terminal_provenance.begin(),
                   contribution.terminal_provenance.end(),
                   [&](const CouplingFormTerminalProvenanceDeclaration& declared) {
                       return declared.kind ==
                                  CouplingFormTerminalProvenanceKind::
                                      MeshTemporal &&
                              declared.temporal_quantity == temporal.quantity;
                   }) != contribution.terminal_provenance.end();
    }
    return true;
}

bool resolvedContributionDeclaresGeometry(
    const ResolvedCouplingFormContribution& contribution,
    const CouplingFormGeometryTerminalProvenance& geometry)
{
    return std::find_if(
               contribution.geometry_terminals.begin(),
               contribution.geometry_terminals.end(),
               [&](const CouplingFormGeometryTerminalProvenance& declared) {
                   return declared.quantity == geometry.quantity;
               }) != contribution.geometry_terminals.end();
}

void validateBridgeMetadataAgainstContribution(
    const ResolvedCouplingFormContribution& contribution,
    const CouplingFormAnalysisMetadata& metadata)
{
    for (const auto& field : metadata.field_uses) {
        FE_THROW_IF(field.field != INVALID_FIELD_ID &&
                        !resolvedContributionDeclaresField(contribution,
                                                           field.field),
                    InvalidArgumentException,
                    "Forms metadata references an undeclared coupling form field");
    }
    for (const auto& temporal : metadata.temporal_symbols) {
        FE_THROW_IF(!resolvedContributionDeclaresTemporal(contribution,
                                                          temporal),
                    InvalidArgumentException,
                    "Forms metadata references an undeclared coupling temporal terminal");
    }
    for (const auto& geometry : metadata.geometry_terminals) {
        FE_THROW_IF(!resolvedContributionDeclaresGeometry(contribution,
                                                          geometry),
                    InvalidArgumentException,
                    "Forms metadata references an undeclared coupling geometry terminal");
    }
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
            if (!additionalFieldSelected(field)) {
                continue;
            }
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
    resolved.install_options = resolveFormInstallOptions(context, contribution);
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

    auto is_active_trial_field = [&](const CouplingFieldUse& use) {
        const auto field = context.field(use.participant_name, use.field_name).field_id;
        return std::find(resolved.fields.begin(), resolved.fields.end(), field) !=
                   resolved.fields.end() ||
               std::find(resolved.extra_trial_fields.begin(),
                         resolved.extra_trial_fields.end(),
                         field) != resolved.extra_trial_fields.end();
    };
    for (const auto& terminal : contribution.terminal_provenance) {
        if (terminal.kind == CouplingFormTerminalProvenanceKind::GeometryTerminal) {
            resolved.geometry_terminals.push_back(
                resolveGeometryTerminalProvenance(context, terminal));
            continue;
        }
        if (terminal.kind != CouplingFormTerminalProvenanceKind::PreviousSolution) {
            continue;
        }

        FE_THROW_IF(!terminal.field.has_value(), InvalidArgumentException,
                    "previous solution terminal provenance requires a field");
        FE_THROW_IF(!is_active_trial_field(*terminal.field), InvalidArgumentException,
                    "previous solution terminal provenance requires an active trial field");
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
    validateBridgeMetadataAgainstContribution(contribution, adapted);
    adapted.declaration_terminal_provenance = contribution.terminal_provenance;
    adapted.geometry_terminals.insert(adapted.geometry_terminals.end(),
                                      contribution.geometry_terminals.begin(),
                                      contribution.geometry_terminals.end());
    return adapted;
}

std::vector<CouplingFormAnalysisMetadata> MonolithicCouplingBuilder::installFormContributions(
    systems::FESystem& system,
    const CouplingContext& context,
    std::span<const CouplingFormContribution> contributions) const
{
    throwIfInvalid(validateFormContributionDeclarations(contributions));

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

    for (const auto& terminal : metadata.terminals) {
        appendBridgeTerminalMetadata(adapted, terminal);
    }
    appendGeometrySensitivityMetadata(adapted);

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
