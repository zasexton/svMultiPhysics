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
#include <utility>
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

bool isResolvedVariableKey(const analysis::VariableKey& key) noexcept
{
    if (key.kind == analysis::VariableKind::FieldComponent) {
        return key.field_id != INVALID_FIELD_ID;
    }
    return !key.name.empty();
}

void addExpertMetadataDiagnostic(CouplingValidationResult& result,
                                 const CouplingInstallMetadata& metadata,
                                 std::string message)
{
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = metadata.contribution_name,
        .message = std::move(message),
    });
}

CouplingValidationResult validateInstallMetadata(
    const CouplingInstallMetadata& metadata)
{
    CouplingValidationResult result;

    if (metadata.contribution_name.empty()) {
        addExpertMetadataDiagnostic(
            result, metadata,
            "expert install metadata is missing contribution name");
    }
    if (metadata.origin.empty()) {
        addExpertMetadataDiagnostic(
            result, metadata,
            "expert install metadata is missing diagnostic origin");
    }
    if (metadata.system_name.empty()) {
        addExpertMetadataDiagnostic(
            result, metadata,
            "expert install metadata is missing owning system name");
    }
    if (metadata.operator_name.empty()) {
        addExpertMetadataDiagnostic(
            result, metadata,
            "expert install metadata is missing operator name");
    }
    if (metadata.installed_dependencies.empty() &&
        metadata.installed_blocks.empty()) {
        addExpertMetadataDiagnostic(
            result, metadata,
            "expert install metadata has no installed dependency or block evidence");
    }

    for (const auto& dependency : metadata.installed_dependencies) {
        if (!isResolvedVariableKey(dependency.residual_row)) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install dependency has unresolved residual row");
        }
        if (!isResolvedVariableKey(dependency.dependency)) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install dependency has unresolved dependency key");
        }
        if (dependency.provider.empty()) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install dependency is missing provider provenance");
        }
        if (!dependency.contributes_matrix_block &&
            !dependency.contributes_vector) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install dependency is missing matrix/vector contribution flags");
        }
    }

    for (const auto& block : metadata.installed_blocks) {
        if (!isResolvedVariableKey(block.residual_row)) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install block has unresolved residual row");
        }
        if (!isResolvedVariableKey(block.dependency)) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install block has unresolved dependency key");
        }
        if (block.domains.empty()) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install block is missing domain provenance");
        }
        if (!block.has_matrix && !block.has_vector) {
            addExpertMetadataDiagnostic(
                result, metadata,
                "expert install block is missing matrix/vector contribution flags");
        }
    }

    return result;
}

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
        location.quadrature_policy_key = declared.quadrature_policy_key;
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
    if (declaration.mode == forms::GeometrySensitivityMode::GeometryConstant) {
        FE_THROW_IF(declaration.mesh_motion_field.has_value(),
                    InvalidArgumentException,
                    "geometry-constant sensitivity rejects mesh-motion fields");
        FE_THROW_IF(declaration.tangent_path ==
                            forms::GeometryTangentPath::SymbolicRequired ||
                        declaration.tangent_path ==
                            forms::GeometryTangentPath::SymbolicWithADCheck,
                    InvalidArgumentException,
                    "geometry-constant sensitivity rejects symbolic geometry tangent paths");
    }
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
    if (declaration.mode == forms::GeometrySensitivityMode::MeshMotionUnknowns &&
        (declaration.tangent_path == forms::GeometryTangentPath::SymbolicRequired ||
         declaration.tangent_path == forms::GeometryTangentPath::SymbolicWithADCheck)) {
        options.use_symbolic_tangent = true;
    }
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

bool resolvedContributionDeclaresField(
    const ResolvedCouplingFormContribution& contribution,
    FieldId field);

void validateMeshMotionGeometrySensitivity(
    const systems::FESystem& system,
    const ResolvedCouplingFormContribution& contribution)
{
    const auto& sensitivity =
        contribution.install_options.compiler_options.geometry_sensitivity;
    if (sensitivity.mode != forms::GeometrySensitivityMode::MeshMotionUnknowns) {
        return;
    }

    const auto mesh_motion_field = sensitivity.mesh_motion_field;
    FE_THROW_IF(mesh_motion_field == INVALID_FIELD_ID, InvalidArgumentException,
                "mesh-motion geometry sensitivity requires a mesh-motion field");
    FE_THROW_IF(!resolvedContributionDeclaresField(contribution,
                                                   mesh_motion_field),
                InvalidArgumentException,
                "mesh-motion geometry sensitivity field must be declared as a coupling form field");

    const auto bound =
        system.meshMotionField(systems::MeshMotionFieldRole::Displacement);
    FE_THROW_IF(!bound.has_value() || *bound != mesh_motion_field,
                InvalidArgumentException,
                "mesh-motion geometry sensitivity requires a matching displacement field binding");
}

void validateInstalledGeometrySensitivityMetadata(
    const ResolvedCouplingFormContribution& contribution,
    const CouplingFormAnalysisMetadata& metadata)
{
    const auto& sensitivity =
        contribution.install_options.compiler_options.geometry_sensitivity;
    if (sensitivity.mode != forms::GeometrySensitivityMode::MeshMotionUnknowns) {
        return;
    }

    const auto mesh_motion_field = sensitivity.mesh_motion_field;
    const auto field_use = std::find_if(
        metadata.field_uses.begin(),
        metadata.field_uses.end(),
        [mesh_motion_field](const CouplingFormFieldProvenance& field) {
            return field.field == mesh_motion_field &&
                   field.appears_as_geometry_sensitivity;
        });
    FE_THROW_IF(field_use == metadata.field_uses.end(),
                InvalidArgumentException,
                "installed metadata is missing mesh-motion geometry-sensitivity field provenance");

    const auto provenance = std::find_if(
        metadata.geometry_sensitivity_provenance.begin(),
        metadata.geometry_sensitivity_provenance.end(),
        [mesh_motion_field](const CouplingGeometrySensitivityProvenance& item) {
            return item.kind ==
                       CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns &&
                   item.mesh_motion_field == mesh_motion_field;
        });
    FE_THROW_IF(provenance == metadata.geometry_sensitivity_provenance.end(),
                InvalidArgumentException,
                "installed metadata is missing structured mesh-motion geometry-sensitivity provenance");
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

bool isUnscopedBridgeMeshTemporalSymbol(
    const CouplingFormTemporalProvenance& temporal,
    const CouplingFormTerminalProvenanceDeclaration& declaration)
{
    return temporal.quantity == declaration.temporal_quantity &&
           !temporal.field.has_value() &&
           !temporal.mesh_motion_scope.has_value() &&
           !temporal.mesh_motion_role.has_value();
}

void appendDeclaredMeshTemporalMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const std::vector<CouplingFormTerminalProvenanceDeclaration>& declarations)
{
    for (const auto& declaration : declarations) {
        if (declaration.kind != CouplingFormTerminalProvenanceKind::MeshTemporal) {
            continue;
        }

        adapted.temporal_symbols.erase(
            std::remove_if(adapted.temporal_symbols.begin(),
                           adapted.temporal_symbols.end(),
                           [&](const CouplingFormTemporalProvenance& temporal) {
                               return isUnscopedBridgeMeshTemporalSymbol(
                                   temporal,
                                   declaration);
                           }),
            adapted.temporal_symbols.end());

        adapted.temporal_symbols.push_back(CouplingFormTemporalProvenance{
            .mesh_motion_scope = declaration.scope,
            .mesh_motion_role = declaration.mesh_motion_role,
            .quantity = declaration.temporal_quantity,
            .derivative_order = declaration.derivative_order,
            .history_index = declaration.history_index,
        });
    }
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

void markGeometrySensitivityFieldUse(CouplingFormAnalysisMetadata& adapted,
                                     FieldId field_id)
{
    if (field_id == INVALID_FIELD_ID) {
        return;
    }

    auto field_it = std::find_if(
        adapted.field_uses.begin(),
        adapted.field_uses.end(),
        [field_id](const CouplingFormFieldProvenance& field) {
            return field.field == field_id;
        });
    if (field_it == adapted.field_uses.end()) {
        adapted.field_uses.push_back(
            CouplingFormFieldProvenance{.field = field_id});
        field_it = std::prev(adapted.field_uses.end());
    }
    field_it->appears_as_geometry_sensitivity = true;
}

CouplingGeometrySensitivityProvenanceKind couplingGeometrySensitivityKind(
    analysis::FormGeometrySensitivityProvenanceKind kind) noexcept
{
    switch (kind) {
    case analysis::FormGeometrySensitivityProvenanceKind::CutGeometry:
        return CouplingGeometrySensitivityProvenanceKind::CutGeometry;
    case analysis::FormGeometrySensitivityProvenanceKind::DriverProvided:
        return CouplingGeometrySensitivityProvenanceKind::DriverProvided;
    }
    return CouplingGeometrySensitivityProvenanceKind::DriverProvided;
}

void appendBridgeGeometrySensitivityMetadata(
    CouplingFormAnalysisMetadata& adapted,
    const analysis::FormGeometrySensitivityProvenanceMetadata& metadata)
{
    for (const auto field : metadata.geometry_fields) {
        markGeometrySensitivityFieldUse(adapted, field);
    }

    adapted.geometry_sensitivity_provenance.push_back(
        CouplingGeometrySensitivityProvenance{
            .kind = couplingGeometrySensitivityKind(metadata.kind),
            .provenance_id = metadata.provenance_id,
            .construction_policy = metadata.construction_policy,
            .target_kind = metadata.target_kind,
            .source_stable_id = metadata.source_stable_id,
            .cut_topology_revision = metadata.cut_topology_revision,
            .quadrature_policy_key = metadata.quadrature_policy_key,
            .parent_entity = metadata.parent_entity,
            .ad_compatible = metadata.ad_compatible,
            .location_sensitivity_available =
                metadata.location_sensitivity_available,
            .jacobian_sensitivity_available =
                metadata.jacobian_sensitivity_available,
            .measure_sensitivity_available =
                metadata.measure_sensitivity_available,
            .normal_sensitivity_available =
                metadata.normal_sensitivity_available,
            .quadrature_weight_sensitivity_available =
                metadata.quadrature_weight_sensitivity_available,
            .geometry_fields = metadata.geometry_fields,
            .parent_geometry_dofs = metadata.parent_geometry_dofs,
            .visible_to_assembly_paths = metadata.visible_to_assembly_paths,
            .sensitivity_sample_count = metadata.sensitivity_sample_count,
        });
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
    markGeometrySensitivityFieldUse(adapted, mesh_motion_field);

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

CouplingParticipantRef participantBySystemName(const CouplingContext& context,
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
    return *it;
}

bool containsName(const std::vector<std::string>& names,
                  std::string_view name) noexcept
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

void copyContextRegistrations(const CouplingContext& context,
                              CouplingContextBuilder& builder,
                              std::vector<std::string>& participant_names)
{
    for (const auto& participant : context.participants()) {
        builder.addParticipant(participant);
        participant_names.push_back(participant.participant_name);
    }
    for (const auto& field : context.fields()) {
        builder.addField(field);
    }
    for (const auto& region : context.regions()) {
        builder.addRegion(region);
    }
    for (const auto& shared_region : context.sharedRegions()) {
        builder.addSharedRegion(shared_region);
    }
    for (const auto& buffer : context.externalBuffers()) {
        builder.addExternalBuffer(buffer);
    }
    for (const auto& transfer : context.driverOwnedTransfers()) {
        builder.addDriverOwnedTransfer(transfer);
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    for (const auto& registry : context.interfaceSearchRegistries()) {
        builder.addInterfaceSearchRegistry(registry);
    }
    for (const auto& map : context.slidingInterfaceMaps()) {
        builder.addSlidingInterfaceMap(map);
    }
#endif
}

CouplingFieldRef fieldRefForRegisteredAdditionalField(
    const CouplingContext& context,
    const ResolvedCouplingAdditionalFieldDeclaration& field)
{
    FE_THROW_IF(field.field_id == INVALID_FIELD_ID, InvalidArgumentException,
                "registered additional field is missing a resolved field id");
    const auto target = participantBySystemName(context, field.system_name);
    return CouplingFieldRef{
        .participant_name = field.declaration.namespace_name,
        .system_name = target.system_name,
        .system = target.system,
        .field_name = field.declaration.field_name,
        .field_id = field.field_id,
        .space = field.field_spec.space,
        .components = field.field_spec.components,
        .scope = field.field_spec.scope,
        .interface_marker = field.field_spec.interface_marker,
    };
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
    for (const auto& field : resolved) {
        const auto& system = mutableSystemByName(context, field.system_name);
        FE_THROW_IF(system.isSetup(), systems::InvalidStateException,
                    "coupling additional fields must be registered before FESystem::setup()");
    }
    for (auto& field : resolved) {
        auto& system = mutableSystemByName(context, field.system_name);
        field.field_id = system.addField(field.field_spec);
    }
    return resolved;
}

CouplingContext MonolithicCouplingBuilder::refreshContextWithAdditionalFields(
    const CouplingContext& context,
    std::span<const ResolvedCouplingAdditionalFieldDeclaration> additional_fields) const
{
    CouplingContextBuilder builder;
    std::vector<std::string> participant_names;
    copyContextRegistrations(context, builder, participant_names);

    for (const auto& field : additional_fields) {
        const auto target = participantBySystemName(context, field.system_name);
        const auto& lookup_namespace = field.declaration.namespace_name;
        if (!containsName(participant_names, lookup_namespace)) {
            builder.addParticipant(CouplingParticipantRef{
                .participant_name = lookup_namespace,
                .system_name = target.system_name,
                .system = target.system,
            });
            participant_names.push_back(lookup_namespace);
        }
        builder.addField(fieldRefForRegisteredAdditionalField(context, field));
    }

    return builder.build();
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

    FE_CHECK_NOT_NULL(owning_system,
                      "monolithic coupling form owning system");
    validateMeshMotionGeometrySensitivity(*owning_system, resolved);

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
    validateInstalledGeometrySensitivityMetadata(contribution, adapted);
    adapted.declaration_terminal_provenance = contribution.terminal_provenance;
    appendDeclaredMeshTemporalMetadata(adapted, contribution.terminal_provenance);
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
    for (const auto& sensitivity : metadata.geometry_sensitivity_provenance) {
        appendBridgeGeometrySensitivityMetadata(adapted, sensitivity);
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

CouplingFormAnalysisMetadata MonolithicCouplingBuilder::adaptInstallMetadata(
    const CouplingInstallMetadata& metadata)
{
    throwIfInvalid(validateInstallMetadata(metadata));

    CouplingFormAnalysisMetadata adapted;
    adapted.contribution_name = metadata.contribution_name;
    adapted.origin = metadata.origin;
    adapted.system_name = metadata.system_name;
    adapted.operator_name = metadata.operator_name;
    adapted.installed_dependencies = metadata.installed_dependencies;
    adapted.installed_blocks = metadata.installed_blocks;
    adapted.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Expert install metadata reports resolved installed dependencies."});
    adapted.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledBlocks,
        analysis::FormBridgeFeatureStatus::Available,
        "Expert install metadata reports resolved installed blocks."});
    return adapted;
}

std::vector<CouplingFormAnalysisMetadata>
MonolithicCouplingBuilder::adaptInstallMetadataRecords(
    std::span<const CouplingInstallMetadata> metadata)
{
    std::vector<CouplingFormAnalysisMetadata> adapted;
    adapted.reserve(metadata.size());
    for (const auto& record : metadata) {
        adapted.push_back(adaptInstallMetadata(record));
    }
    return adapted;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
