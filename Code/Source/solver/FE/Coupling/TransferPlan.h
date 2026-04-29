/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_TRANSFERPLAN_H
#define SVMP_FE_COUPLING_TRANSFERPLAN_H

/**
 * @file TransferPlan.h
 * @brief Transfer declarations and resolved transfer metadata for coupling plans.
 */

#include "Coupling/CouplingTypes.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Systems/InterfaceOperators.h"
#endif

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

enum class CouplingTransferKind : std::uint8_t {
    Unspecified,
    Identity,
    InterfacePointwiseInterpolation,
    InterfaceConservativeProjection,
    InterfaceMortar,
    DriverOwned,
};

enum class CouplingInterfaceFramePolicy : std::uint8_t {
    None,
    SourceToTargetVector,
    SourceToTargetRank2Tensor,
};

enum class CouplingExternalBufferAccess : std::uint8_t {
    ReadOnly,
    WriteOnly,
    ReadWrite,
};

enum class CouplingExternalBufferDistribution : std::uint8_t {
    RankLocal,
    DistributedOwned,
    DistributedOwnedGhosted,
    DriverDefined,
};

enum class CouplingExternalBufferLifetime : std::uint8_t {
    StepLocal,
    TimeStepPersistent,
    RestartPersistent,
    DriverDefined,
};

struct CouplingInterfaceTransferDeclaration {
    CouplingInterfaceFramePolicy frame_policy{CouplingInterfaceFramePolicy::None};
    CouplingFrameSourceEmbeddingPolicy source_embedding_policy{
        CouplingFrameSourceEmbeddingPolicy::None};
    CouplingFrameTargetRestrictionPolicy target_restriction_policy{
        CouplingFrameTargetRestrictionPolicy::None};
    std::array<std::array<Real, 3>, 3> source_to_target_rotation{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}}};
    Real conservation_tolerance{1.0e-10};
};

struct CouplingTransferDeclaration {
    CouplingTransferKind kind{CouplingTransferKind::Unspecified};
    std::optional<CouplingInterfaceTransferDeclaration> interface_declaration;
    std::string driver_owned_name;
};

struct CouplingExternalBufferDescriptor {
    std::string buffer_name;
    std::string scalar_type{"Real"};
    CouplingValueDescriptor value{};
    CouplingExternalBufferAccess access{CouplingExternalBufferAccess::ReadWrite};
    CouplingExternalBufferDistribution distribution{
        CouplingExternalBufferDistribution::DriverDefined};
    CouplingExternalBufferLifetime lifetime{CouplingExternalBufferLifetime::DriverDefined};
    std::vector<std::int64_t> extents;
    std::vector<std::int64_t> strides;
    std::string packing;
    std::vector<CouplingTemporalSlotDescriptor> supported_temporal_slots;
    std::uint64_t layout_revision_key{0};
    std::uint64_t data_revision_key{0};
};

struct CouplingDriverOwnedTransferDescriptor {
    std::string transfer_name;
    std::vector<CouplingValueRank> supported_ranks;
    bool preserves_component_layout{true};
    std::vector<CouplingTemporalSlotDescriptor> supported_source_temporal_slots;
    std::vector<CouplingTemporalSlotDescriptor> supported_target_temporal_slots;
    std::uint64_t registry_revision_key{0};
};

struct ResolvedCouplingTransfer {
    CouplingTransferKind kind{CouplingTransferKind::Unspecified};
    CouplingFrameSourceEmbeddingPolicy source_embedding_policy{
        CouplingFrameSourceEmbeddingPolicy::None};
    CouplingFrameTargetRestrictionPolicy target_restriction_policy{
        CouplingFrameTargetRestrictionPolicy::None};
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<systems::InterfaceTransferOptions> interface_options;
#endif
    std::string driver_owned_name;
    std::optional<CouplingDriverOwnedTransferDescriptor> driver_owned_descriptor;
};

[[nodiscard]] const char* toString(CouplingTransferKind kind) noexcept;
[[nodiscard]] const char* toString(CouplingInterfaceFramePolicy policy) noexcept;
[[nodiscard]] bool isInterfaceTransferKind(CouplingTransferKind kind) noexcept;

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_TRANSFERPLAN_H
