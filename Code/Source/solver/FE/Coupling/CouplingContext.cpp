/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingContext.h"

#include "Coupling/SharedRegionRegistry.h"
#include "Core/FEException.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

bool sameScope(const std::optional<std::string>& lhs,
               std::optional<std::string_view> rhs) noexcept
{
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    return *lhs == *rhs;
}

bool sameScope(const std::optional<std::string>& lhs,
               const std::optional<std::string>& rhs) noexcept
{
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    return *lhs == *rhs;
}

void appendTemporalSlotValidation(
    CouplingValidationResult& result,
    const std::vector<CouplingTemporalSlotDescriptor>& slots)
{
    for (const auto& slot : slots) {
        result.append(validateCouplingTemporalSlot(slot));
    }
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
bool hasRevisionEvidence(
    const svmp::search::InterfaceRevisionSnapshot& revision) noexcept
{
    return revision.configuration != svmp::Configuration::Reference ||
           revision.geometry_revision != 0 ||
           revision.reference_geometry_revision != 0 ||
           revision.current_geometry_revision != 0 ||
           revision.topology_revision != 0 ||
           revision.ownership_revision != 0 ||
           revision.numbering_revision != 0 ||
           revision.field_layout_revision != 0 ||
           revision.label_revision != 0 ||
           revision.active_configuration_epoch != 0;
}

void validateInterfaceMapRuntime(
    const CouplingInterfaceMapProvenance& provenance,
    const svmp::search::InterfaceMap& interface_map)
{
    FE_THROW_IF(interface_map.name != provenance.interface_map_name,
                InvalidArgumentException,
                "interface map provenance resolved an unexpected interface map");
    FE_THROW_IF(interface_map.state != provenance.map_state,
                InvalidArgumentException,
                "interface map runtime state does not match provenance");
    if (provenance.map_revision_key != 0) {
        FE_THROW_IF(interface_map.revision_key() != provenance.map_revision_key,
                    InvalidArgumentException,
                    "interface map runtime revision does not match provenance");
    }
    if (interface_map.source.boundary_label != INVALID_LABEL) {
        FE_THROW_IF(interface_map.source.boundary_label !=
                        provenance.source_interface_marker,
                    InvalidArgumentException,
                    "interface map source marker does not match provenance");
    }
    if (interface_map.target.boundary_label != INVALID_LABEL) {
        FE_THROW_IF(interface_map.target.boundary_label !=
                        provenance.target_interface_marker,
                    InvalidArgumentException,
                    "interface map target marker does not match provenance");
    }
    FE_THROW_IF(interface_map.source.configuration != provenance.source_configuration,
                InvalidArgumentException,
                "interface map source configuration does not match provenance");
    FE_THROW_IF(interface_map.target.configuration != provenance.target_configuration,
                InvalidArgumentException,
                "interface map target configuration does not match provenance");
    if (!interface_map.source.logical_region.empty() &&
        !provenance.source_logical_region.empty()) {
        FE_THROW_IF(
            !interface_map.source.logical_region.compatible_with(
                provenance.source_logical_region),
            InvalidArgumentException,
            "interface map source logical region does not match provenance");
    }
    if (!interface_map.target.logical_region.empty() &&
        !provenance.target_logical_region.empty()) {
        FE_THROW_IF(
            !interface_map.target.logical_region.compatible_with(
                provenance.target_logical_region),
            InvalidArgumentException,
            "interface map target logical region does not match provenance");
    }
    if (hasRevisionEvidence(provenance.source_revision_snapshot)) {
        FE_THROW_IF(interface_map.source_revision.revision_key() !=
                        provenance.source_revision_snapshot.revision_key(),
                    InvalidArgumentException,
                    "interface map source revision snapshot does not match provenance");
    }
    if (hasRevisionEvidence(provenance.target_revision_snapshot)) {
        FE_THROW_IF(interface_map.target_revision.revision_key() !=
                        provenance.target_revision_snapshot.revision_key(),
                    InvalidArgumentException,
                    "interface map target revision snapshot does not match provenance");
    }
    if (provenance.source_search_revision_key != 0) {
        FE_THROW_IF(interface_map.source_revision.revision_key() !=
                        provenance.source_search_revision_key,
                    InvalidArgumentException,
                    "interface map source search revision does not match provenance");
    }
    if (provenance.target_search_revision_key != 0) {
        FE_THROW_IF(interface_map.target_revision.revision_key() !=
                        provenance.target_search_revision_key,
                    InvalidArgumentException,
                    "interface map target search revision does not match provenance");
    }
    if (interface_map.source.valid() && interface_map.target.valid()) {
        FE_THROW_IF(!interface_map.valid_for_current_revisions(),
                    InvalidArgumentException,
                    "interface map runtime revisions are stale");
    }
}

void validateSlidingInterfaceMapRuntime(
    const CouplingInterfaceMapProvenance& provenance,
    const systems::SlidingInterfaceMap& sliding_map)
{
    FE_THROW_IF(!sliding_map.name.empty() &&
                    sliding_map.name != provenance.interface_map_name,
                InvalidArgumentException,
                "sliding interface map name does not match provenance");
    FE_THROW_IF(sliding_map.map_kind != provenance.sliding_map_kind,
                InvalidArgumentException,
                "sliding interface map kind does not match provenance");
    FE_THROW_IF(sliding_map.state != provenance.operator_state,
                InvalidArgumentException,
                "sliding interface operator state does not match provenance");
    if (provenance.accepted_revision_key != 0) {
        FE_THROW_IF(sliding_map.accepted_revision_key !=
                        provenance.accepted_revision_key,
                    InvalidArgumentException,
                    "sliding interface accepted revision does not match provenance");
    }
    if (provenance.trial_revision_key != 0) {
        FE_THROW_IF(sliding_map.trial_revision_key != provenance.trial_revision_key,
                    InvalidArgumentException,
                    "sliding interface trial revision does not match provenance");
    }
    if (provenance.time_level_epoch != 0) {
        FE_THROW_IF(sliding_map.time_level_epoch != provenance.time_level_epoch,
                    InvalidArgumentException,
                    "sliding interface epoch does not match provenance");
    }
    if (provenance.time != Real{0.0}) {
        FE_THROW_IF(sliding_map.time != provenance.time,
                    InvalidArgumentException,
                    "sliding interface time does not match provenance");
    }
}
#endif

} // namespace

bool CouplingParticipantRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr;
}

bool CouplingFieldRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !field_name.empty() && field_id != INVALID_FIELD_ID && space != nullptr &&
           components > 0;
}

bool CouplingRegionRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !region_name.empty();
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
bool CouplingInterfaceSearchRegistryRegistration::valid() const noexcept
{
    return !registry_name.empty() && registry != nullptr;
}

bool CouplingSlidingInterfaceMapRegistration::valid() const noexcept
{
    return !interface_map_name.empty() && sliding_map != nullptr;
}
#endif

CouplingParticipantRef CouplingContext::participant(std::string_view participant_name) const
{
    const auto it = std::find_if(participants_.begin(), participants_.end(),
                                 [participant_name](const CouplingParticipantRef& participant) {
                                     return participant.participant_name == participant_name;
                                 });
    FE_THROW_IF(it == participants_.end(), InvalidArgumentException,
                "unknown coupling participant");
    return *it;
}

CouplingFieldRef CouplingContext::field(std::string_view participant_name,
                                        std::string_view field_name) const
{
    const auto it = std::find_if(fields_.begin(), fields_.end(),
                                 [participant_name, field_name](const CouplingFieldRef& field) {
                                     return field.participant_name == participant_name &&
                                            field.field_name == field_name;
                                 });
    FE_THROW_IF(it == fields_.end(), InvalidArgumentException,
                "unknown coupling field");
    return *it;
}

CouplingRegionRef CouplingContext::region(std::string_view participant_name,
                                          std::string_view region_name) const
{
    const auto it = std::find_if(regions_.begin(), regions_.end(),
                                 [participant_name, region_name](const CouplingRegionRef& region) {
                                     return region.participant_name == participant_name &&
                                            region.region_name == region_name;
                                 });
    FE_THROW_IF(it == regions_.end(), InvalidArgumentException,
                "unknown coupling region");
    return *it;
}

CouplingRegionRef CouplingContext::sharedRegion(std::string_view name,
                                                std::string_view participant_name) const
{
    const auto group = sharedRegionGroup(name);
    const auto it = std::find_if(group.participant_regions.begin(),
                                 group.participant_regions.end(),
                                 [participant_name](const CouplingRegionRef& region) {
                                     return region.participant_name == participant_name;
                                 });
    FE_THROW_IF(it == group.participant_regions.end(), InvalidArgumentException,
                "unknown coupling shared-region participant mapping");
    return *it;
}

SharedRegionRef CouplingContext::sharedRegionGroup(std::string_view name) const
{
    const auto it = std::find_if(shared_regions_.begin(), shared_regions_.end(),
                                 [name](const SharedRegionRef& region) {
                                     return region.name == name;
                                 });
    FE_THROW_IF(it == shared_regions_.end(), InvalidArgumentException,
                "unknown coupling shared region");
    return *it;
}

bool CouplingContext::hasParticipant(std::string_view participant_name) const noexcept
{
    return std::any_of(participants_.begin(), participants_.end(),
                       [participant_name](const CouplingParticipantRef& participant) {
                           return participant.participant_name == participant_name;
                       });
}

bool CouplingContext::hasField(std::string_view participant_name,
                               std::string_view field_name) const noexcept
{
    return std::any_of(fields_.begin(), fields_.end(),
                       [participant_name, field_name](const CouplingFieldRef& field) {
                           return field.participant_name == participant_name &&
                                  field.field_name == field_name;
                       });
}

bool CouplingContext::hasRegion(std::string_view participant_name,
                                std::string_view region_name) const noexcept
{
    return std::any_of(regions_.begin(), regions_.end(),
                       [participant_name, region_name](const CouplingRegionRef& region) {
                           return region.participant_name == participant_name &&
                                  region.region_name == region_name;
                       });
}

bool CouplingContext::hasSharedRegion(std::string_view name) const noexcept
{
    return std::any_of(shared_regions_.begin(), shared_regions_.end(),
                       [name](const SharedRegionRef& region) {
                           return region.name == name;
                       });
}

const CouplingExternalBufferDescriptor* CouplingContext::externalBufferDescriptor(
    std::optional<std::string_view> participant,
    std::string_view external_buffer_key) const noexcept
{
    const auto it = std::find_if(
        external_buffers_.begin(),
        external_buffers_.end(),
        [participant, external_buffer_key](
            const CouplingExternalBufferRegistration& registration) {
            return sameScope(registration.participant_name, participant) &&
                   registration.descriptor.buffer_name == external_buffer_key;
        });
    return it == external_buffers_.end() ? nullptr : &it->descriptor;
}

const CouplingDriverOwnedTransferDescriptor* CouplingContext::driverOwnedTransfer(
    std::string_view transfer_name) const noexcept
{
    const auto it = std::find_if(
        driver_owned_transfers_.begin(),
        driver_owned_transfers_.end(),
        [transfer_name](const CouplingDriverOwnedTransferDescriptor& descriptor) {
            return descriptor.transfer_name == transfer_name;
        });
    return it == driver_owned_transfers_.end() ? nullptr : &*it;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
const svmp::search::InterfaceSearchRegistry* CouplingContext::interfaceSearchRegistry(
    std::string_view registry_name) const noexcept
{
    const auto it = std::find_if(
        interface_search_registries_.begin(),
        interface_search_registries_.end(),
        [registry_name](const CouplingInterfaceSearchRegistryRegistration& registration) {
            return registration.registry_name == registry_name;
        });
    return it == interface_search_registries_.end() ? nullptr : it->registry;
}

const systems::SlidingInterfaceMap* CouplingContext::slidingInterfaceMap(
    std::string_view interface_map_name) const noexcept
{
    const auto it = std::find_if(
        sliding_interface_maps_.begin(),
        sliding_interface_maps_.end(),
        [interface_map_name](const CouplingSlidingInterfaceMapRegistration& registration) {
            return registration.interface_map_name == interface_map_name;
        });
    return it == sliding_interface_maps_.end() ? nullptr : it->sliding_map;
}

CouplingInterfaceMapRuntimeHandles CouplingContext::interfaceMapHandles(
    const CouplingInterfaceMapProvenance& provenance) const
{
    const auto source_it = std::find_if(
        participants_.begin(),
        participants_.end(),
        [&provenance](const CouplingParticipantRef& participant) {
            return participant.system_name == provenance.source_system_name;
        });
    FE_THROW_IF(source_it == participants_.end(), InvalidArgumentException,
                "interface map provenance references an unknown source system");

    const auto target_it = std::find_if(
        participants_.begin(),
        participants_.end(),
        [&provenance](const CouplingParticipantRef& participant) {
            return participant.system_name == provenance.target_system_name;
        });
    FE_THROW_IF(target_it == participants_.end(), InvalidArgumentException,
                "interface map provenance references an unknown target system");

    const auto* registry =
        interfaceSearchRegistry(provenance.interface_search_registry_name);
    FE_THROW_IF(registry == nullptr, InvalidArgumentException,
                "interface map provenance references an unknown search registry");

    const auto* sliding_map = slidingInterfaceMap(provenance.interface_map_name);
    const auto* interface_map = sliding_map != nullptr
                                    ? &sliding_map->interface_map
                                    : registry->committed_map(provenance.interface_map_name);
    FE_THROW_IF(interface_map == nullptr, InvalidArgumentException,
                "interface map provenance references an unknown interface map");
    validateInterfaceMapRuntime(provenance, *interface_map);
    if (sliding_map != nullptr) {
        validateSlidingInterfaceMapRuntime(provenance, *sliding_map);
    }

    return CouplingInterfaceMapRuntimeHandles{
        .source_system = source_it->system,
        .target_system = target_it->system,
        .search_registry = registry,
        .sliding_map = sliding_map,
        .interface_map = interface_map,
    };
}
#endif

const std::vector<CouplingParticipantRef>& CouplingContext::participants() const noexcept
{
    return participants_;
}

const std::vector<CouplingFieldRef>& CouplingContext::fields() const noexcept
{
    return fields_;
}

const std::vector<CouplingRegionRef>& CouplingContext::regions() const noexcept
{
    return regions_;
}

const std::vector<SharedRegionRef>& CouplingContext::sharedRegions() const noexcept
{
    return shared_regions_;
}

const std::vector<CouplingExternalBufferRegistration>&
CouplingContext::externalBuffers() const noexcept
{
    return external_buffers_;
}

const std::vector<CouplingDriverOwnedTransferDescriptor>&
CouplingContext::driverOwnedTransfers() const noexcept
{
    return driver_owned_transfers_;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
const std::vector<CouplingInterfaceSearchRegistryRegistration>&
CouplingContext::interfaceSearchRegistries() const noexcept
{
    return interface_search_registries_;
}

const std::vector<CouplingSlidingInterfaceMapRegistration>&
CouplingContext::slidingInterfaceMaps() const noexcept
{
    return sliding_interface_maps_;
}
#endif

CouplingContextBuilder& CouplingContextBuilder::addParticipant(CouplingParticipantRef participant)
{
    context_.participants_.push_back(std::move(participant));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addField(CouplingFieldRef field)
{
    context_.fields_.push_back(std::move(field));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addRegion(CouplingRegionRef region)
{
    context_.regions_.push_back(std::move(region));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addSharedRegion(SharedRegionRef region)
{
    context_.shared_regions_.push_back(std::move(region));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addExternalBuffer(
    CouplingExternalBufferRegistration registration)
{
    context_.external_buffers_.push_back(std::move(registration));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addDriverOwnedTransfer(
    CouplingDriverOwnedTransferDescriptor descriptor)
{
    context_.driver_owned_transfers_.push_back(std::move(descriptor));
    return *this;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
CouplingContextBuilder& CouplingContextBuilder::addInterfaceSearchRegistry(
    CouplingInterfaceSearchRegistryRegistration registration)
{
    context_.interface_search_registries_.push_back(std::move(registration));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addSlidingInterfaceMap(
    CouplingSlidingInterfaceMapRegistration registration)
{
    context_.sliding_interface_maps_.push_back(std::move(registration));
    return *this;
}
#endif

CouplingValidationResult CouplingContextBuilder::validate() const
{
    CouplingValidationResult result;

    for (std::size_t i = 0; i < context_.participants_.size(); ++i) {
        const auto& participant = context_.participants_[i];
        if (!participant.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = participant.participant_name,
                .message = "coupling participant requires participant name, system name, and owning system",
            });
        }
        for (std::size_t j = i + 1u; j < context_.participants_.size(); ++j) {
            if (!participant.participant_name.empty() &&
                participant.participant_name == context_.participants_[j].participant_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = participant.participant_name,
                    .message = "duplicate coupling participant",
                });
            }
        }
    }

    for (std::size_t i = 0; i < context_.fields_.size(); ++i) {
        const auto& field = context_.fields_[i];
        if (!field.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field requires participant, system ownership, field id, space, and positive component count",
            });
        }
        if (!field.participant_name.empty() &&
            !context_.hasParticipant(field.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field references an unknown participant",
            });
        }
        for (std::size_t j = i + 1u; j < context_.fields_.size(); ++j) {
            if (field.participant_name == context_.fields_[j].participant_name &&
                field.field_name == context_.fields_[j].field_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = field.participant_name,
                    .field_name = field.field_name,
                    .message = "duplicate coupling field mapping",
                });
            }
        }
    }

    for (std::size_t i = 0; i < context_.regions_.size(); ++i) {
        const auto& region = context_.regions_[i];
        if (!region.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region requires participant, system ownership, and region name",
            });
        }
        if (!region.participant_name.empty() &&
            !context_.hasParticipant(region.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region references an unknown participant",
            });
        }
        for (std::size_t j = i + 1u; j < context_.regions_.size(); ++j) {
            if (region.participant_name == context_.regions_[j].participant_name &&
                region.region_name == context_.regions_[j].region_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = region.participant_name,
                    .region_name = region.region_name,
                    .message = "duplicate coupling region mapping",
                });
            }
        }
    }

    SharedRegionRegistry registry;
    for (const auto& region : context_.shared_regions_) {
        registry.add(region);
    }
    result.append(registry.validate());

    for (const auto& shared_region : context_.shared_regions_) {
        for (const auto& region : shared_region.participant_regions) {
            if (!context_.hasRegion(region.participant_name, region.region_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = region.participant_name,
                    .region_name = region.region_name,
                    .message = "shared region references a participant region missing from the context",
                });
            }
        }
    }

    for (std::size_t i = 0; i < context_.external_buffers_.size(); ++i) {
        const auto& registration = context_.external_buffers_[i];
        const auto& descriptor = registration.descriptor;
        if (registration.participant_name.has_value()) {
            if (registration.participant_name->empty()) {
                result.addError("external buffer participant scope must be nonempty");
            } else if (!context_.hasParticipant(*registration.participant_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = *registration.participant_name,
                    .endpoint_name = descriptor.buffer_name,
                    .message = "external buffer descriptor references an unknown participant",
                });
            }
        }
        if (descriptor.buffer_name.empty()) {
            result.addError("external buffer descriptor requires a buffer name");
        }
        if (descriptor.scalar_type.empty()) {
            result.addError("external buffer descriptor requires a scalar type");
        } else if (descriptor.scalar_type != "Real") {
            result.addError("external buffer descriptor scalar type must be Real");
        }
        result.append(validateCouplingValueDescriptor(descriptor.value));
        if (descriptor.extents.empty()) {
            result.addError("external buffer descriptor requires extents");
        }
        if (descriptor.strides.size() != descriptor.extents.size()) {
            result.addError("external buffer descriptor strides must match extents");
        }
        for (const auto extent : descriptor.extents) {
            if (extent <= 0) {
                result.addError("external buffer descriptor extents must be positive");
            }
        }
        for (const auto stride : descriptor.strides) {
            if (stride == 0) {
                result.addError("external buffer descriptor strides must be nonzero");
            }
        }
        if (descriptor.packing.empty()) {
            result.addError("external buffer descriptor requires packing metadata");
        }
        if (descriptor.supported_temporal_slots.empty()) {
            result.addError("external buffer descriptor requires supported temporal slots");
        }
        appendTemporalSlotValidation(result, descriptor.supported_temporal_slots);
        if (descriptor.value.rank == CouplingValueRank::GeneralTensor) {
            if (descriptor.packing != descriptor.value.tensor_packing) {
                result.addError(
                    "external buffer packing must match the general tensor value descriptor");
            }
            if (descriptor.extents.size() != descriptor.value.tensor_extents.size()) {
                result.addError(
                    "external buffer extents must match the general tensor value descriptor");
            } else {
                for (std::size_t j = 0; j < descriptor.extents.size(); ++j) {
                    if (descriptor.extents[j] != descriptor.value.tensor_extents[j]) {
                        result.addError(
                            "external buffer extents must match the general tensor value descriptor");
                        break;
                    }
                }
            }
        }
        for (std::size_t j = i + 1u; j < context_.external_buffers_.size(); ++j) {
            if (sameScope(registration.participant_name,
                          context_.external_buffers_[j].participant_name) &&
                descriptor.buffer_name ==
                    context_.external_buffers_[j].descriptor.buffer_name) {
                result.addError("duplicate external buffer descriptor in one scope");
            }
        }
    }

    for (std::size_t i = 0; i < context_.driver_owned_transfers_.size(); ++i) {
        const auto& descriptor = context_.driver_owned_transfers_[i];
        if (descriptor.transfer_name.empty()) {
            result.addError("driver-owned transfer descriptor requires a name");
        }
        if (descriptor.supported_ranks.empty()) {
            result.addError("driver-owned transfer descriptor requires supported ranks");
        }
        for (std::size_t j = 0; j < descriptor.supported_ranks.size(); ++j) {
            for (std::size_t k = j + 1u; k < descriptor.supported_ranks.size(); ++k) {
                if (descriptor.supported_ranks[j] == descriptor.supported_ranks[k]) {
                    result.addError("driver-owned transfer descriptor has duplicate ranks");
                }
            }
        }
        if (descriptor.supported_source_temporal_slots.empty()) {
            result.addError(
                "driver-owned transfer descriptor requires source temporal slots");
        }
        if (descriptor.supported_target_temporal_slots.empty()) {
            result.addError(
                "driver-owned transfer descriptor requires target temporal slots");
        }
        appendTemporalSlotValidation(result, descriptor.supported_source_temporal_slots);
        appendTemporalSlotValidation(result, descriptor.supported_target_temporal_slots);
        for (std::size_t j = i + 1u; j < context_.driver_owned_transfers_.size(); ++j) {
            if (descriptor.transfer_name ==
                context_.driver_owned_transfers_[j].transfer_name) {
                result.addError("duplicate driver-owned transfer descriptor");
            }
        }
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    for (std::size_t i = 0; i < context_.interface_search_registries_.size(); ++i) {
        const auto& registration = context_.interface_search_registries_[i];
        if (!registration.valid()) {
            result.addError(
                "interface search registry registration requires a name and registry");
        }
        for (std::size_t j = i + 1u;
             j < context_.interface_search_registries_.size();
             ++j) {
            if (registration.registry_name ==
                context_.interface_search_registries_[j].registry_name) {
                result.addError("duplicate interface search registry registration");
            }
        }
    }

    for (std::size_t i = 0; i < context_.sliding_interface_maps_.size(); ++i) {
        const auto& registration = context_.sliding_interface_maps_[i];
        if (!registration.valid()) {
            result.addError(
                "sliding interface map registration requires a name and map");
        }
        for (std::size_t j = i + 1u; j < context_.sliding_interface_maps_.size();
             ++j) {
            if (registration.interface_map_name ==
                context_.sliding_interface_maps_[j].interface_map_name) {
                result.addError("duplicate sliding interface map registration");
            }
        }
    }
#endif

    return result;
}

CouplingContext CouplingContextBuilder::build() const
{
    const auto validation = validate();
    throwIfInvalid(validation);
    return context_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
