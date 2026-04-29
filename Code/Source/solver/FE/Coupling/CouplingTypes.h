/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGTYPES_H
#define SVMP_FE_COUPLING_COUPLINGTYPES_H

/**
 * @file CouplingTypes.h
 * @brief Physics-agnostic coupling vocabulary shared by setup, graph, and plan code.
 */

#include "Core/Types.h"
#include "Coupling/CouplingDiagnostics.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

enum class CouplingMode : std::uint8_t {
    Monolithic,
    Partitioned,
};

enum class CouplingRequirement : std::uint8_t {
    Required,
    Optional,
};

enum class CouplingDependencyMode : std::uint8_t {
    ImplicitMonolithic,
    ExternalLagged,
};

enum class CouplingRegionKind : std::uint8_t {
    Domain,
    Boundary,
    InteriorFace,
    InterfaceFace,
    UserDefined,
};

enum class CouplingInterfaceSide : std::uint8_t {
    None,
    Minus,
    Plus,
};

enum class CouplingCoordinateConfiguration : std::uint8_t {
    Reference,
    Current,
};

enum class CouplingValueRank : std::uint8_t {
    Scalar,
    Vector,
    Rank2Tensor,
    SymmetricTensor,
    MixedBlock,
    GeneralTensor,
};

enum class CouplingFrameSourceEmbeddingPolicy : std::uint8_t {
    None,
    Embed2DInXY,
    Embed2DInXZ,
    Embed2DInYZ,
    DriverProvided,
};

enum class CouplingFrameTargetRestrictionPolicy : std::uint8_t {
    None,
    RestrictToXY,
    RestrictToXZ,
    RestrictToYZ,
    DriverProvided,
};

enum class CouplingTemporalSlot : std::uint8_t {
    Current,
    Accepted,
    Predicted,
    History,
    Stage,
    External,
};

enum class CouplingEndpointKind : std::uint8_t {
    Field,
    RegionData,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput,
    Parameter,
    ExternalBuffer,
};

struct CouplingPortId {
    std::string contract_instance_name;
    std::string port_name;

    [[nodiscard]] bool valid() const noexcept;
};

[[nodiscard]] bool operator==(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept;
[[nodiscard]] bool operator!=(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept;
[[nodiscard]] bool operator<(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept;

struct CouplingValueDescriptor {
    CouplingValueRank rank{CouplingValueRank::Scalar};
    int components{1};
    std::vector<std::string> component_layout;
    std::vector<int> tensor_extents;
    std::string tensor_packing;
};

struct CouplingTemporalSlotDescriptor {
    CouplingTemporalSlot slot{CouplingTemporalSlot::Current};
    std::optional<int> history_index;
    std::optional<int> stage_index;
};

struct CouplingEndpointRef {
    CouplingEndpointKind kind{CouplingEndpointKind::ExternalBuffer};
    std::optional<std::string> participant_name;
    std::string endpoint_name;
    CouplingTemporalSlotDescriptor temporal{};
};

struct CouplingParticipantUse {
    std::string participant_name;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingFieldUse {
    std::string participant_name;
    std::string field_name;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingRegionUse {
    std::string participant_name;
    std::string region_name;
    std::optional<CouplingRegionKind> required_region_kind;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingSharedRegionUse {
    std::string shared_region_name;
    std::optional<CouplingRegionKind> required_region_kind;
    CouplingRequirement requirement{CouplingRequirement::Required};
};

struct CouplingRegionEndpointDeclaration {
    std::string participant_name;
    std::string region_name;
    std::optional<std::string> shared_region_name;
};

[[nodiscard]] const char* toString(CouplingMode mode) noexcept;
[[nodiscard]] const char* toString(CouplingRequirement requirement) noexcept;
[[nodiscard]] const char* toString(CouplingDependencyMode mode) noexcept;
[[nodiscard]] const char* toString(CouplingRegionKind kind) noexcept;
[[nodiscard]] const char* toString(CouplingInterfaceSide side) noexcept;
[[nodiscard]] const char* toString(CouplingCoordinateConfiguration configuration) noexcept;
[[nodiscard]] const char* toString(CouplingValueRank rank) noexcept;
[[nodiscard]] const char* toString(CouplingTemporalSlot slot) noexcept;
[[nodiscard]] const char* toString(CouplingEndpointKind kind) noexcept;

[[nodiscard]] CouplingValidationResult validateCouplingPortId(const CouplingPortId& port);
[[nodiscard]] CouplingValidationResult validateCouplingValueDescriptor(
    const CouplingValueDescriptor& value);
[[nodiscard]] CouplingValidationResult validateCouplingTemporalSlot(
    const CouplingTemporalSlotDescriptor& temporal);
[[nodiscard]] CouplingValidationResult validateCouplingEndpointRef(
    const CouplingEndpointRef& endpoint);

[[nodiscard]] bool couplingValueDescriptorsCompatible(
    const CouplingValueDescriptor& producer,
    const CouplingValueDescriptor& consumer) noexcept;

[[nodiscard]] std::uint64_t couplingTensorExtentProduct(
    const std::vector<int>& extents) noexcept;

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGTYPES_H
