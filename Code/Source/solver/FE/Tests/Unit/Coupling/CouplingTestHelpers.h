/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_COUPLING_COUPLINGTESTHELPERS_H
#define SVMP_FE_TESTS_UNIT_COUPLING_COUPLINGTESTHELPERS_H

#include "Analysis/ProblemAnalysisTypes.h"
#include "Coupling/CouplingDeclaration.h"
#include "Coupling/CouplingGeometryRequirements.h"
#include "Coupling/PartitionedCouplingPlan.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {
namespace test {

struct ParticipantBinding {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
};

[[nodiscard]] inline const systems::FESystem* systemToken(std::uint64_t id)
{
    return reinterpret_cast<const systems::FESystem*>(
        static_cast<std::uintptr_t>(id));
}

[[nodiscard]] inline ParticipantBinding participantBinding(
    std::string participant_name,
    std::uint64_t token_id)
{
    ParticipantBinding binding;
    binding.participant_name = std::move(participant_name);
    binding.system_name = binding.participant_name + "_system";
    binding.system = systemToken(token_id);
    return binding;
}

[[nodiscard]] inline CouplingParticipantRef participantRef(
    const ParticipantBinding& binding)
{
    return CouplingParticipantRef{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .system = binding.system,
    };
}

[[nodiscard]] inline CouplingFieldRef fieldRef(
    const ParticipantBinding& binding,
    std::string field_name,
    FieldId field_id,
    std::shared_ptr<const spaces::FunctionSpace> space,
    int components)
{
    return CouplingFieldRef{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .system = binding.system,
        .field_name = std::move(field_name),
        .field_id = field_id,
        .space = std::move(space),
        .components = components,
    };
}

[[nodiscard]] inline CouplingRegionRef boundaryRegionRef(
    const ParticipantBinding& binding,
    std::string region_name,
    int marker)
{
    return CouplingRegionRef{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .system = binding.system,
        .region_name = std::move(region_name),
        .kind = CouplingRegionKind::Boundary,
        .marker = marker,
    };
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] inline svmp::search::LogicalInterfaceRegionId logicalInterfaceRegion(
    std::string persistent_id,
    int marker)
{
    return svmp::search::LogicalInterfaceRegionId{
        .persistent_id = persistent_id,
        .name = persistent_id,
        .physical_label = marker,
    };
}

[[nodiscard]] inline svmp::search::InterfaceRevisionSnapshot
interfaceRevisionSnapshot(std::uint64_t base)
{
    return svmp::search::InterfaceRevisionSnapshot{
        .configuration = svmp::Configuration::Reference,
        .geometry_revision = base + 1u,
        .reference_geometry_revision = base + 2u,
        .current_geometry_revision = base + 3u,
        .topology_revision = base + 4u,
        .ownership_revision = base + 5u,
        .numbering_revision = base + 6u,
        .field_layout_revision = base + 7u,
        .label_revision = base + 8u,
        .active_configuration_epoch = base + 9u,
    };
}
#endif

[[nodiscard]] inline CouplingRegionRef interfaceRegionRef(
    const ParticipantBinding& binding,
    std::string region_name,
    int marker,
    CouplingInterfaceSide side,
    std::uint64_t revision_base = 100u)
{
    CouplingRegionRef region{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .system = binding.system,
        .region_name = std::move(region_name),
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = side,
        .coordinate_configuration = CouplingCoordinateConfiguration::Reference,
        .geometry_revision = revision_base + 1u,
        .topology_revision = revision_base + 4u,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    region.logical_region =
        logicalInterfaceRegion(binding.participant_name + "_interface", marker);
    region.revision_snapshot = interfaceRevisionSnapshot(revision_base);
#endif
    return region;
}

[[nodiscard]] inline std::string contractTypeKey(std::string_view stem)
{
    return std::string("fixture.") + std::string(stem);
}

[[nodiscard]] inline std::string contractInstanceName(std::string_view stem,
                                                       int index)
{
    return std::string(stem) + "_" + std::to_string(index);
}

[[nodiscard]] inline CouplingPortId portId(std::string instance_name,
                                           std::string port_name)
{
    return CouplingPortId{std::move(instance_name), std::move(port_name)};
}

[[nodiscard]] inline CouplingValueDescriptor vectorValueDescriptor(
    int components)
{
    CouplingValueDescriptor descriptor;
    descriptor.rank = CouplingValueRank::Vector;
    descriptor.components = components;
    for (int component = 0; component < components; ++component) {
        descriptor.component_layout.push_back(
            "component_" + std::to_string(component));
    }
    return descriptor;
}

[[nodiscard]] inline CouplingValueDescriptor generalTensorValueDescriptor(
    std::vector<int> extents,
    std::string packing = "row_major")
{
    int components = 1;
    for (const int extent : extents) {
        components *= extent;
    }
    CouplingValueDescriptor descriptor;
    descriptor.rank = CouplingValueRank::GeneralTensor;
    descriptor.components = components;
    descriptor.tensor_extents = std::move(extents);
    descriptor.tensor_packing = std::move(packing);
    return descriptor;
}

[[nodiscard]] inline CouplingTemporalSlotDescriptor currentSlot()
{
    return CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current};
}

[[nodiscard]] inline CouplingTemporalSlotDescriptor acceptedSlot()
{
    return CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted};
}

[[nodiscard]] inline CouplingTemporalSlotDescriptor predictedSlot()
{
    return CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted};
}

[[nodiscard]] inline CouplingTemporalSlotDescriptor historySlot(
    int logical_history_index)
{
    return CouplingTemporalSlotDescriptor{
        .slot = CouplingTemporalSlot::History,
        .history_index = logical_history_index,
    };
}

[[nodiscard]] inline CouplingTemporalSlotDescriptor stageSlot(int stage_index)
{
    return CouplingTemporalSlotDescriptor{
        .slot = CouplingTemporalSlot::Stage,
        .stage_index = stage_index,
    };
}

[[nodiscard]] inline CouplingEndpointRef endpointRef(
    CouplingEndpointKind kind,
    std::optional<std::string> participant_name,
    std::string endpoint_name,
    CouplingTemporalSlotDescriptor temporal = currentSlot())
{
    return CouplingEndpointRef{
        .kind = kind,
        .participant_name = std::move(participant_name),
        .endpoint_name = std::move(endpoint_name),
        .temporal = temporal,
    };
}

[[nodiscard]] inline CouplingExternalBufferDescriptor externalBufferDescriptor(
    std::string name,
    CouplingValueDescriptor value)
{
    return CouplingExternalBufferDescriptor{
        .buffer_name = std::move(name),
        .value = std::move(value),
        .access = CouplingExternalBufferAccess::ReadWrite,
        .distribution = CouplingExternalBufferDistribution::DistributedOwned,
        .lifetime = CouplingExternalBufferLifetime::TimeStepPersistent,
        .extents = {3},
        .strides = {1},
        .packing = "contiguous",
        .supported_temporal_slots = {currentSlot(), acceptedSlot(), predictedSlot()},
        .layout_revision_key = 11u,
        .data_revision_key = 12u,
    };
}

[[nodiscard]] inline CouplingDriverOwnedTransferDescriptor
driverOwnedTransferDescriptor(std::string name)
{
    return CouplingDriverOwnedTransferDescriptor{
        .transfer_name = std::move(name),
        .supported_ranks = {CouplingValueRank::Scalar,
                            CouplingValueRank::Vector,
                            CouplingValueRank::GeneralTensor},
        .preserves_component_layout = true,
        .supported_source_temporal_slots = {currentSlot(), acceptedSlot()},
        .supported_target_temporal_slots = {currentSlot(), predictedSlot()},
        .registry_revision_key = 21u,
    };
}

[[nodiscard]] inline CouplingTransferDeclaration unspecifiedTransfer()
{
    return CouplingTransferDeclaration{.kind = CouplingTransferKind::Unspecified};
}

[[nodiscard]] inline CouplingTransferDeclaration identityTransfer()
{
    return CouplingTransferDeclaration{.kind = CouplingTransferKind::Identity};
}

[[nodiscard]] inline CouplingTransferDeclaration driverOwnedTransfer(
    std::string transfer_name)
{
    return CouplingTransferDeclaration{
        .kind = CouplingTransferKind::DriverOwned,
        .driver_owned_name = std::move(transfer_name),
    };
}

[[nodiscard]] inline ResolvedCouplingTransfer resolvedDriverOwnedTransfer(
    CouplingDriverOwnedTransferDescriptor descriptor)
{
    return ResolvedCouplingTransfer{
        .kind = CouplingTransferKind::DriverOwned,
        .driver_owned_name = descriptor.transfer_name,
        .driver_owned_descriptor = std::move(descriptor),
    };
}

[[nodiscard]] inline ResolvedCouplingTemporalSlot resolvedTemporalSlot(
    CouplingTemporalSlotDescriptor request,
    CouplingResolvedTemporalBackingKind backing)
{
    return ResolvedCouplingTemporalSlot{
        .request = request,
        .provided = request,
        .backing = backing,
        .provider_name = "system_state",
        .storage_index = request.history_index,
        .state_revision_key = 31u,
        .time = 2.5,
    };
}

[[nodiscard]] inline ResolvedCouplingEndpoint resolvedFieldEndpoint(
    const ParticipantBinding& binding,
    std::string field_name,
    FieldId field_id,
    CouplingValueDescriptor value,
    CouplingTemporalSlotDescriptor temporal = currentSlot())
{
    return ResolvedCouplingEndpoint{
        .declaration_provenance = endpointRef(
            CouplingEndpointKind::Field,
            binding.participant_name,
            field_name,
            temporal),
        .resolved_kind = CouplingEndpointKind::Field,
        .value = std::move(value),
        .resolved_participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .registry_provider = "field_registry",
        .resolved_endpoint_key = field_name,
        .temporal = resolvedTemporalSlot(
            temporal,
            CouplingResolvedTemporalBackingKind::SystemStateCurrent),
        .system = binding.system,
        .field_id = field_id,
        .layout_revision_key = 41u,
        .registry_revision_key = 42u,
    };
}

[[nodiscard]] inline CouplingAdditionalFieldDeclaration participantAdditionalField(
    const ParticipantBinding& binding,
    std::shared_ptr<const spaces::FunctionSpace> space,
    std::string field_name = "participant_auxiliary")
{
    return CouplingAdditionalFieldDeclaration{
        .field_namespace = CouplingAdditionalFieldNamespace::Participant,
        .namespace_name = binding.participant_name,
        .system_participant_name = binding.participant_name,
        .field_name = std::move(field_name),
        .space = std::move(space),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::VolumeCell,
    };
}

[[nodiscard]] inline CouplingAdditionalFieldDeclaration contractAdditionalField(
    const ParticipantBinding& binding,
    std::shared_ptr<const spaces::FunctionSpace> space,
    std::string contract_instance,
    std::string field_name = "contract_auxiliary")
{
    return CouplingAdditionalFieldDeclaration{
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = std::move(contract_instance),
        .system_participant_name = binding.participant_name,
        .field_name = std::move(field_name),
        .space = std::move(space),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .region_name = "interface",
        .shared_region_name = "interface",
    };
}

[[nodiscard]] inline CouplingGeometryTerminalLocationDeclaration
geometryTerminalLocation(CouplingRegionKind region_kind,
                         std::optional<std::string> shared_region_name =
                             std::nullopt)
{
    return CouplingGeometryTerminalLocationDeclaration{
        .region_kind = region_kind,
        .shared_region_name = std::move(shared_region_name),
        .side = region_kind == CouplingRegionKind::InterfaceFace
                    ? CouplingInterfaceSide::Minus
                    : CouplingInterfaceSide::None,
        .coordinate_configuration = forms::GeometryConfiguration::Current,
        .transform_from_configuration = forms::GeometryConfiguration::Reference,
        .transform_to_configuration = forms::GeometryConfiguration::Current,
        .quadrature_policy_key = 51u,
    };
}

[[nodiscard]] inline CouplingGeometryTerminalRequirement
geometryTerminalRequirement(CouplingGeometryTerminalQuantity quantity,
                            const ParticipantBinding& binding,
                            std::string region_name,
                            CouplingRegionKind region_kind)
{
    return CouplingGeometryTerminalRequirement{
        .quantity = quantity,
        .scope = CouplingGeometryTerminalScope{
            .participant_name = binding.participant_name,
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = binding.participant_name,
                .region_name = std::move(region_name),
                .shared_region_name = region_kind == CouplingRegionKind::InterfaceFace
                                          ? std::optional<std::string>("interface")
                                          : std::nullopt,
            },
            .location = geometryTerminalLocation(
                region_kind,
                region_kind == CouplingRegionKind::InterfaceFace
                    ? std::optional<std::string>("interface")
                    : std::nullopt),
        },
    };
}

[[nodiscard]] inline CouplingFormGeometryTerminalProvenance
geometryTerminalProvenance(CouplingGeometryTerminalQuantity quantity,
                           const ParticipantBinding& binding,
                           std::string region_name,
                           int marker,
                           CouplingRegionKind region_kind)
{
    CouplingFormGeometryTerminalProvenance provenance;
    provenance.quantity = quantity;
    provenance.location.region_kind = region_kind;
    provenance.location.shared_region_name =
        region_kind == CouplingRegionKind::InterfaceFace
            ? std::optional<std::string>("interface")
            : std::nullopt;
    provenance.location.marker = marker;
    provenance.location.side = region_kind == CouplingRegionKind::InterfaceFace
                                   ? CouplingInterfaceSide::Minus
                                   : CouplingInterfaceSide::None;
    provenance.location.coordinate_configuration =
        forms::GeometryConfiguration::Current;
    provenance.location.transform_from_configuration =
        forms::GeometryConfiguration::Reference;
    provenance.location.transform_to_configuration =
        forms::GeometryConfiguration::Current;
    provenance.location.geometry_revision = 101u;
    provenance.location.quadrature_policy_key = 51u;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (region_kind == CouplingRegionKind::InterfaceFace) {
        provenance.location.logical_region =
            logicalInterfaceRegion(binding.participant_name + "_interface", marker);
    }
#endif
    provenance.analysis_domain = region_kind == CouplingRegionKind::InterfaceFace
                                     ? analysis::DomainKind::InterfaceFace
                                     : analysis::DomainKind::Boundary;
    provenance.owner = CouplingGeometryTerminalOwnerProvenance{
        .participant_name = binding.participant_name,
        .system_name = binding.system_name,
        .region_name = std::move(region_name),
        .shared_region_name = provenance.location.shared_region_name,
    };
    provenance.provider = "forms";
    provenance.value_available = true;
    provenance.gradient_or_jacobian_available = true;
    provenance.normal_available = true;
    provenance.measure_available = true;
    return provenance;
}

[[nodiscard]] inline CouplingGeometrySensitivityDeclaration
geometrySensitivityDeclaration(const ParticipantBinding& binding,
                               std::string mesh_motion_field = "mesh_motion")
{
    return CouplingGeometrySensitivityDeclaration{
        .mode = forms::GeometrySensitivityMode::MeshMotionUnknowns,
        .mesh_motion_field = CouplingFieldUse{
            .participant_name = binding.participant_name,
            .field_name = std::move(mesh_motion_field),
        },
        .tangent_path = forms::GeometryTangentPath::SymbolicWithADCheck,
        .use_symbolic_tangent = true,
    };
}

[[nodiscard]] inline CouplingGeometrySensitivityProvenance
geometrySensitivityProvenance(FieldId mesh_motion_field)
{
    CouplingGeometrySensitivityProvenance provenance;
    provenance.kind = CouplingGeometrySensitivityProvenanceKind::MeshMotionUnknowns;
    provenance.mesh_motion_field = mesh_motion_field;
    provenance.provenance_id = "mesh_motion_sensitivity";
    provenance.construction_policy = "symbolic_with_ad_check";
    provenance.target_kind = "current_geometry";
    provenance.source_stable_id = 61u;
    provenance.cut_topology_revision = 62u;
    provenance.quadrature_policy_key = 63u;
    provenance.parent_entity = 64;
    provenance.ad_compatible = true;
    provenance.location_sensitivity_available = true;
    provenance.jacobian_sensitivity_available = true;
    provenance.measure_sensitivity_available = true;
    provenance.normal_sensitivity_available = true;
    provenance.quadrature_weight_sensitivity_available = true;
    provenance.geometry_fields = {mesh_motion_field};
    provenance.parent_geometry_dofs = {65, 66, 67};
    provenance.visible_to_assembly_paths = {"current_coordinate"};
    provenance.sensitivity_sample_count = 3u;
    return provenance;
}

[[nodiscard]] inline CouplingFormInstallOptionsDeclaration
formInstallOptionsDeclaration(const ParticipantBinding& binding)
{
    return CouplingFormInstallOptionsDeclaration{
        .ad_mode = forms::ADMode::Reverse,
        .compiler_options = CouplingSymbolicOptionsDeclaration{
            .simplify_expressions = true,
            .exploit_sparsity = true,
            .cache_expressions = true,
            .verbose = false,
        },
        .geometry_sensitivity = geometrySensitivityDeclaration(binding),
    };
}

[[nodiscard]] inline systems::FormInstallOptions resolvedFormInstallOptions()
{
    systems::FormInstallOptions options;
    options.ad_mode = forms::ADMode::Reverse;
    options.compiler_options.ad_mode = forms::ADMode::Reverse;
    options.compiler_options.geometry_sensitivity.mode =
        forms::GeometrySensitivityMode::MeshMotionUnknowns;
    options.compiler_options.geometry_tangent_path =
        forms::GeometryTangentPath::SymbolicWithADCheck;
    options.compiler_options.use_symbolic_tangent = true;
    options.extra_trial_fields = {3};
    return options;
}

[[nodiscard]] inline CouplingFormContribution formContribution(
    const ParticipantBinding& binding,
    std::string contribution_name,
    std::string origin)
{
    CouplingFormContribution contribution;
    contribution.contribution_name = std::move(contribution_name);
    contribution.origin = std::move(origin);
    contribution.operator_name = "equations";
    contribution.field_uses = {CouplingFieldUse{
        .participant_name = binding.participant_name,
        .field_name = "primary",
    }};
    contribution.extra_trial_field_uses = {CouplingFieldUse{
        .participant_name = binding.participant_name,
        .field_name = "mesh_motion",
    }};
    contribution.install_options_declaration =
        formInstallOptionsDeclaration(binding);
    contribution.install_options = resolvedFormInstallOptions();
    return contribution;
}

[[nodiscard]] inline CouplingInstallMetadata expertInstallMetadata(
    const ParticipantBinding& binding,
    std::string contribution_name = "expert_balance")
{
    CouplingInstallMetadata metadata;
    metadata.contribution_name = std::move(contribution_name);
    metadata.origin = "test_fixture";
    metadata.system_name = binding.system_name;
    metadata.operator_name = "equations";
    metadata.installed_dependencies.push_back(CouplingInstalledDependency{
        .residual_row = analysis::VariableKey::field(1),
        .dependency = analysis::VariableKey::field(2),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::InterfaceFace,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "expert_fixture",
    });
    metadata.installed_blocks.push_back(CouplingInstalledBlockProvenance{
        .residual_row = analysis::VariableKey::field(1),
        .dependency = analysis::VariableKey::field(2),
        .domains = {analysis::DomainKind::InterfaceFace},
        .has_matrix = true,
        .has_vector = true,
    });
    return metadata;
}

[[nodiscard]] inline CouplingContext twoParticipantContext(int components = 1)
{
    const auto left = participantBinding("left", 1u);
    const auto right = participantBinding("right", 2u);
    const auto space =
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const auto left_interface = interfaceRegionRef(
        left,
        "interface",
        7,
        CouplingInterfaceSide::Minus,
        100u);
    const auto right_interface = interfaceRegionRef(
        right,
        "interface",
        8,
        CouplingInterfaceSide::Plus,
        200u);

    CouplingContextBuilder builder;
    builder.addParticipant(participantRef(left));
    builder.addParticipant(participantRef(right));
    builder.addField(fieldRef(left, "primary", 1, space, components));
    builder.addField(fieldRef(right, "primary", 2, space, components));
    builder.addRegion(boundaryRegionRef(left, "surface", 4));
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .required_participant_names = {"left", "right"},
        .participant_regions = {left_interface, right_interface},
    });
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_buffer",
            vectorValueDescriptor(components)),
    });
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("driver_transfer"));
    return builder.build();
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] inline CouplingInterfaceMapProvenance interfaceMapProvenance()
{
    CouplingInterfaceMapProvenance provenance;
    provenance.interface_map_name = "left_to_right_map";
    provenance.interface_entry_name = "left_to_right_entry";
    provenance.interface_search_registry_name = "interface_search";
    provenance.source_system_name = "left_system";
    provenance.target_system_name = "right_system";
    provenance.source_interface_marker = 7;
    provenance.target_interface_marker = 8;
    provenance.sliding_map_kind = systems::SlidingInterfaceMapKind::Sliding;
    provenance.source_configuration = svmp::Configuration::Reference;
    provenance.target_configuration = svmp::Configuration::Current;
    provenance.source_logical_region = logicalInterfaceRegion("left_interface", 7);
    provenance.target_logical_region = logicalInterfaceRegion("right_interface", 8);
    provenance.source_revision_snapshot = interfaceRevisionSnapshot(100u);
    provenance.target_revision_snapshot = interfaceRevisionSnapshot(200u);
    provenance.source_search_revision_key = 71u;
    provenance.target_search_revision_key = 72u;
    provenance.map_revision_key = 73u;
    provenance.map_state = svmp::search::InterfaceMapState::Committed;
    provenance.operator_state = systems::InterfaceOperatorState::AcceptedTimeStep;
    provenance.accepted_revision_key = 74u;
    provenance.trial_revision_key = 75u;
    provenance.time = 1.25;
    provenance.time_level_epoch = 76u;
    return provenance;
}

[[nodiscard]] inline CouplingTransferDeclaration interfaceTransfer()
{
    CouplingInterfaceTransferDeclaration interface_declaration;
    interface_declaration.frame_policy =
        CouplingInterfaceFramePolicy::SourceToTargetVector;
    interface_declaration.source_embedding_policy =
        CouplingFrameSourceEmbeddingPolicy::Embed2DInXY;
    interface_declaration.target_restriction_policy =
        CouplingFrameTargetRestrictionPolicy::RestrictToXY;

    CouplingTransferDeclaration transfer;
    transfer.kind = CouplingTransferKind::InterfacePointwiseInterpolation;
    transfer.interface_declaration = interface_declaration;
    transfer.interface_map = interfaceMapProvenance();
    return transfer;
}
#endif

} // namespace test
} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_COUPLING_COUPLINGTESTHELPERS_H
