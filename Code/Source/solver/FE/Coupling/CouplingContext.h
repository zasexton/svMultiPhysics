/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGCONTEXT_H
#define SVMP_FE_COUPLING_COUPLINGCONTEXT_H

/**
 * @file CouplingContext.h
 * @brief Name-to-FE ownership records for coupling setup.
 */

#include "Core/Types.h"
#include "Coupling/CouplingTypes.h"
#include "Coupling/TransferPlan.h"
#include "Systems/FieldRegistry.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces {
class FunctionSpace;
}

namespace systems {
class FESystem;
}

namespace coupling {

struct CouplingParticipantRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingFieldRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
    std::string field_name;
    FieldId field_id{INVALID_FIELD_ID};
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{1};
    systems::FieldScope scope{systems::FieldScope::VolumeCell};
    int interface_marker{-1};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingRegionRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
    std::string region_name;
    CouplingRegionKind kind{CouplingRegionKind::UserDefined};
    int marker{-1};
    CouplingInterfaceSide side{CouplingInterfaceSide::None};
    CouplingCoordinateConfiguration coordinate_configuration{
        CouplingCoordinateConfiguration::Reference};
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
    std::optional<svmp::search::InterfaceRevisionSnapshot> revision_snapshot;
#endif
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};

    [[nodiscard]] bool valid() const noexcept;
};

struct SharedRegionRef {
    std::string name;
    std::optional<CouplingRegionKind> required_region_kind;
    std::vector<CouplingRegionRef> participant_regions;
};

struct CouplingExternalBufferRegistration {
    std::optional<std::string> participant_name;
    CouplingExternalBufferDescriptor descriptor;
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
struct CouplingInterfaceSearchRegistryRegistration {
    std::string registry_name;
    const svmp::search::InterfaceSearchRegistry* registry{nullptr};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingSlidingInterfaceMapRegistration {
    std::string interface_map_name;
    const systems::SlidingInterfaceMap* sliding_map{nullptr};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingInterfaceMapRuntimeHandles {
    const systems::FESystem* source_system{nullptr};
    const systems::FESystem* target_system{nullptr};
    const svmp::search::InterfaceSearchRegistry* search_registry{nullptr};
    const systems::SlidingInterfaceMap* sliding_map{nullptr};
    const svmp::search::InterfaceMap* interface_map{nullptr};
};
#endif

class CouplingContext {
public:
    CouplingContext() = default;

    [[nodiscard]] CouplingParticipantRef participant(std::string_view participant) const;
    [[nodiscard]] CouplingFieldRef field(std::string_view participant,
                                         std::string_view field) const;
    [[nodiscard]] CouplingRegionRef region(std::string_view participant,
                                           std::string_view region) const;
    [[nodiscard]] CouplingRegionRef sharedRegion(std::string_view name,
                                                 std::string_view participant) const;
    [[nodiscard]] SharedRegionRef sharedRegionGroup(std::string_view name) const;

    [[nodiscard]] bool hasParticipant(std::string_view participant) const noexcept;
    [[nodiscard]] bool hasField(std::string_view participant,
                                std::string_view field) const noexcept;
    [[nodiscard]] bool hasRegion(std::string_view participant,
                                 std::string_view region) const noexcept;
    [[nodiscard]] bool hasSharedRegion(std::string_view name) const noexcept;
    [[nodiscard]] const CouplingExternalBufferDescriptor* externalBufferDescriptor(
        std::optional<std::string_view> participant,
        std::string_view external_buffer_key) const noexcept;
    [[nodiscard]] const CouplingDriverOwnedTransferDescriptor* driverOwnedTransfer(
        std::string_view transfer_name) const noexcept;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    [[nodiscard]] const svmp::search::InterfaceSearchRegistry* interfaceSearchRegistry(
        std::string_view registry_name) const noexcept;
    [[nodiscard]] const systems::SlidingInterfaceMap* slidingInterfaceMap(
        std::string_view interface_map_name) const noexcept;
    [[nodiscard]] CouplingInterfaceMapRuntimeHandles interfaceMapHandles(
        const CouplingInterfaceMapProvenance& provenance) const;
#endif

    [[nodiscard]] const std::vector<CouplingParticipantRef>& participants() const noexcept;
    [[nodiscard]] const std::vector<CouplingFieldRef>& fields() const noexcept;
    [[nodiscard]] const std::vector<CouplingRegionRef>& regions() const noexcept;
    [[nodiscard]] const std::vector<SharedRegionRef>& sharedRegions() const noexcept;
    [[nodiscard]] const std::vector<CouplingExternalBufferRegistration>&
    externalBuffers() const noexcept;
    [[nodiscard]] const std::vector<CouplingDriverOwnedTransferDescriptor>&
    driverOwnedTransfers() const noexcept;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    [[nodiscard]] const std::vector<CouplingInterfaceSearchRegistryRegistration>&
    interfaceSearchRegistries() const noexcept;
    [[nodiscard]] const std::vector<CouplingSlidingInterfaceMapRegistration>&
    slidingInterfaceMaps() const noexcept;
#endif

private:
    friend class CouplingContextBuilder;

    std::vector<CouplingParticipantRef> participants_{};
    std::vector<CouplingFieldRef> fields_{};
    std::vector<CouplingRegionRef> regions_{};
    std::vector<SharedRegionRef> shared_regions_{};
    std::vector<CouplingExternalBufferRegistration> external_buffers_{};
    std::vector<CouplingDriverOwnedTransferDescriptor> driver_owned_transfers_{};
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::vector<CouplingInterfaceSearchRegistryRegistration> interface_search_registries_{};
    std::vector<CouplingSlidingInterfaceMapRegistration> sliding_interface_maps_{};
#endif
};

class CouplingContextBuilder {
public:
    CouplingContextBuilder& addParticipant(CouplingParticipantRef participant);
    CouplingContextBuilder& addField(CouplingFieldRef field);
    CouplingContextBuilder& addRegion(CouplingRegionRef region);
    CouplingContextBuilder& addSharedRegion(SharedRegionRef region);
    CouplingContextBuilder& addExternalBuffer(
        CouplingExternalBufferRegistration registration);
    CouplingContextBuilder& addDriverOwnedTransfer(
        CouplingDriverOwnedTransferDescriptor descriptor);
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    CouplingContextBuilder& addInterfaceSearchRegistry(
        CouplingInterfaceSearchRegistryRegistration registration);
    CouplingContextBuilder& addSlidingInterfaceMap(
        CouplingSlidingInterfaceMapRegistration registration);
#endif

    [[nodiscard]] CouplingValidationResult validate() const;
    [[nodiscard]] CouplingContext build() const;

private:
    CouplingContext context_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGCONTEXT_H
