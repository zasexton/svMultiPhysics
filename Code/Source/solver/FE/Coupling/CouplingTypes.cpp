/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingTypes.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <tuple>

namespace svmp {
namespace FE {
namespace coupling {

bool CouplingPortId::valid() const noexcept
{
    return !contract_instance_name.empty() && !port_name.empty();
}

bool operator==(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept
{
    return lhs.contract_instance_name == rhs.contract_instance_name &&
           lhs.port_name == rhs.port_name;
}

bool operator!=(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept
{
    return !(lhs == rhs);
}

bool operator<(const CouplingPortId& lhs, const CouplingPortId& rhs) noexcept
{
    return std::tie(lhs.contract_instance_name, lhs.port_name) <
           std::tie(rhs.contract_instance_name, rhs.port_name);
}

const char* toString(CouplingMode mode) noexcept
{
    switch (mode) {
    case CouplingMode::Monolithic:
        return "monolithic";
    case CouplingMode::Partitioned:
        return "partitioned";
    }
    return "unknown";
}

const char* toString(CouplingRequirement requirement) noexcept
{
    switch (requirement) {
    case CouplingRequirement::Required:
        return "required";
    case CouplingRequirement::Optional:
        return "optional";
    }
    return "unknown";
}

const char* toString(CouplingDependencyMode mode) noexcept
{
    switch (mode) {
    case CouplingDependencyMode::ImplicitMonolithic:
        return "implicit_monolithic";
    case CouplingDependencyMode::ExternalLagged:
        return "external_lagged";
    }
    return "unknown";
}

const char* toString(CouplingRegionKind kind) noexcept
{
    switch (kind) {
    case CouplingRegionKind::Domain:
        return "domain";
    case CouplingRegionKind::Boundary:
        return "boundary";
    case CouplingRegionKind::InteriorFace:
        return "interior_face";
    case CouplingRegionKind::InterfaceFace:
        return "interface_face";
    case CouplingRegionKind::UserDefined:
        return "user_defined";
    case CouplingRegionKind::Curve:
        return "curve";
    case CouplingRegionKind::Point:
        return "point";
    case CouplingRegionKind::CutInterface:
        return "cut_interface";
    }
    return "unknown";
}

const char* toString(CouplingInterfaceSide side) noexcept
{
    switch (side) {
    case CouplingInterfaceSide::None:
        return "none";
    case CouplingInterfaceSide::Minus:
        return "minus";
    case CouplingInterfaceSide::Plus:
        return "plus";
    }
    return "unknown";
}

const char* toString(CouplingCoordinateConfiguration configuration) noexcept
{
    switch (configuration) {
    case CouplingCoordinateConfiguration::Reference:
        return "reference";
    case CouplingCoordinateConfiguration::Current:
        return "current";
    }
    return "unknown";
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
std::optional<svmp::Configuration> toMeshConfiguration(
    CouplingCoordinateConfiguration configuration) noexcept
{
    switch (configuration) {
    case CouplingCoordinateConfiguration::Reference:
        return svmp::Configuration::Reference;
    case CouplingCoordinateConfiguration::Current:
        return svmp::Configuration::Current;
    }
    return std::nullopt;
}
#endif

const char* toString(CouplingValueRank rank) noexcept
{
    switch (rank) {
    case CouplingValueRank::Scalar:
        return "scalar";
    case CouplingValueRank::Vector:
        return "vector";
    case CouplingValueRank::Rank2Tensor:
        return "rank2_tensor";
    case CouplingValueRank::SymmetricTensor:
        return "symmetric_tensor";
    case CouplingValueRank::MixedBlock:
        return "mixed_block";
    case CouplingValueRank::GeneralTensor:
        return "general_tensor";
    }
    return "unknown";
}

const char* toString(CouplingTemporalSlot slot) noexcept
{
    switch (slot) {
    case CouplingTemporalSlot::Current:
        return "current";
    case CouplingTemporalSlot::Accepted:
        return "accepted";
    case CouplingTemporalSlot::Predicted:
        return "predicted";
    case CouplingTemporalSlot::History:
        return "history";
    case CouplingTemporalSlot::Stage:
        return "stage";
    case CouplingTemporalSlot::External:
        return "external";
    }
    return "unknown";
}

const char* toString(CouplingEndpointKind kind) noexcept
{
    switch (kind) {
    case CouplingEndpointKind::Field:
        return "field";
    case CouplingEndpointKind::RegionData:
        return "region_data";
    case CouplingEndpointKind::AuxiliaryState:
        return "auxiliary_state";
    case CouplingEndpointKind::AuxiliaryInput:
        return "auxiliary_input";
    case CouplingEndpointKind::AuxiliaryOutput:
        return "auxiliary_output";
    case CouplingEndpointKind::Parameter:
        return "parameter";
    case CouplingEndpointKind::ExternalBuffer:
        return "external_buffer";
    }
    return "unknown";
}

namespace {

bool generatedNamePartContainsSeparator(std::string_view part) noexcept
{
    return part.find('.') != std::string_view::npos;
}

void validateGeneratedNamePart(CouplingValidationResult& result,
                               std::string_view part,
                               std::string_view label)
{
    if (part.empty()) {
        result.addError(std::string(label) + " name part must be nonempty");
        return;
    }
    if (generatedNamePartContainsSeparator(part)) {
        result.addError(std::string(label) +
                        " name part must not contain the generated-name separator");
    }
}

} // namespace

CouplingValidationResult validateCouplingPortId(const CouplingPortId& port)
{
    CouplingValidationResult result;
    if (port.contract_instance_name.empty()) {
        result.addError("coupling port requires a contract instance name");
    }
    if (port.port_name.empty()) {
        result.addError("coupling port requires a port name");
    }
    return result;
}

CouplingValidationResult validateCouplingGeneratedNameRequest(
    const CouplingGeneratedNameRequest& request)
{
    CouplingValidationResult result;
    if (request.explicit_name.has_value()) {
        if (request.explicit_name->empty()) {
            result.addError("explicit generated-name override must be nonempty");
        }
        return result;
    }

    validateGeneratedNamePart(result, request.contract_name, "contract");
    validateGeneratedNamePart(result, request.relation_name, "relation");
    validateGeneratedNamePart(result, request.local_name, "local");
    return result;
}

std::string makeCouplingGeneratedName(const CouplingGeneratedNameRequest& request)
{
    if (request.explicit_name.has_value()) {
        return *request.explicit_name;
    }
    return request.contract_name + "." + request.relation_name + "." +
           request.local_name;
}

std::uint64_t couplingTensorExtentProduct(const std::vector<int>& extents) noexcept
{
    std::uint64_t product = 1u;
    for (const int extent : extents) {
        if (extent <= 0) {
            return 0u;
        }
        const auto next = product * static_cast<std::uint64_t>(extent);
        if (next < product) {
            return 0u;
        }
        product = next;
    }
    return product;
}

CouplingValidationResult validateCouplingValueDescriptor(const CouplingValueDescriptor& value)
{
    CouplingValidationResult result;
    if (value.components <= 0) {
        result.addError("coupling value descriptor requires a positive component count");
    }

    if (!value.component_layout.empty() &&
        value.component_layout.size() != static_cast<std::size_t>(std::max(value.components, 0))) {
        result.addError("component layout size must match the component count");
    }
    for (std::size_t i = 0; i < value.component_layout.size(); ++i) {
        if (value.component_layout[i].empty()) {
            result.addError("component layout entries must be nonempty");
        }
        for (std::size_t j = i + 1u; j < value.component_layout.size(); ++j) {
            if (value.component_layout[i] == value.component_layout[j]) {
                result.addError("component layout entries must be unique");
            }
        }
    }

    const bool has_tensor_shape =
        !value.tensor_extents.empty() || !value.tensor_packing.empty();

    if (value.rank == CouplingValueRank::GeneralTensor) {
        if (value.tensor_extents.empty()) {
            result.addError("general tensor values require explicit extents");
        }
        if (value.tensor_packing.empty()) {
            result.addError("general tensor values require explicit packing metadata");
        }
        const std::uint64_t product = couplingTensorExtentProduct(value.tensor_extents);
        if (product == 0u) {
            result.addError("general tensor extents must be positive");
        } else if (value.components > 0 &&
                   product != static_cast<std::uint64_t>(value.components)) {
            result.addError("general tensor extent product must match the component count");
        }
    } else if (has_tensor_shape) {
        result.addError("tensor extents and tensor packing are only valid for general tensor values");
    }

    if (value.rank == CouplingValueRank::Scalar && value.components != 1) {
        result.addError("scalar coupling values require exactly one component");
    }
    if (value.rank == CouplingValueRank::Vector && value.components < 2) {
        result.addError("vector coupling values require at least two components");
    }
    if (value.rank == CouplingValueRank::Rank2Tensor && value.components < 4) {
        result.addError("rank-2 tensor coupling values require at least four components");
    }
    if (value.rank == CouplingValueRank::SymmetricTensor && value.components < 3) {
        result.addError("symmetric tensor coupling values require at least three components");
    }

    if (value.rank == CouplingValueRank::MixedBlock && value.component_layout.empty()) {
        result.addError("mixed block values require component layout metadata");
    }
    if (!value.unit.empty() && value.physical_dimension.empty()) {
        result.addError("unit metadata requires a physical dimension");
    }

    return result;
}

CouplingValidationResult validateCouplingTemporalSlot(
    const CouplingTemporalSlotDescriptor& temporal)
{
    CouplingValidationResult result;
    if (temporal.slot == CouplingTemporalSlot::History) {
        if (!temporal.history_index.has_value()) {
            result.addError("history temporal slots require a logical history index");
        } else if (*temporal.history_index <= 0) {
            result.addError("history temporal slots use positive logical history indices");
        }
    } else if (temporal.history_index.has_value()) {
        result.addError("history index is valid only for history temporal slots");
    }

    if (temporal.slot == CouplingTemporalSlot::Stage) {
        if (!temporal.stage_index.has_value()) {
            result.addError("stage temporal slots require a stage index");
        } else if (*temporal.stage_index < 0) {
            result.addError("stage temporal slots require a nonnegative stage index");
        }
    } else if (temporal.stage_index.has_value()) {
        result.addError("stage index is valid only for stage temporal slots");
    }

    return result;
}

CouplingValidationResult validateCouplingEndpointRef(const CouplingEndpointRef& endpoint)
{
    CouplingValidationResult result = validateCouplingTemporalSlot(endpoint.temporal);
    if (endpoint.endpoint_name.empty()) {
        result.addError("coupling endpoint requires a registry key");
    }

    const bool requires_participant =
        endpoint.kind != CouplingEndpointKind::ExternalBuffer;
    if (requires_participant &&
        (!endpoint.participant_name.has_value() || endpoint.participant_name->empty())) {
        result.addError("FE-backed coupling endpoints require participant scope");
    }
    if (endpoint.kind == CouplingEndpointKind::ExternalBuffer &&
        endpoint.participant_name.has_value() && endpoint.participant_name->empty()) {
        result.addError("external buffer participant scope must be absent or nonempty");
    }
    return result;
}

bool couplingValueDescriptorsCompatible(
    const CouplingValueDescriptor& producer,
    const CouplingValueDescriptor& consumer) noexcept
{
    const bool dimensions_compatible =
        producer.physical_dimension.empty() ||
        consumer.physical_dimension.empty() ||
        producer.physical_dimension == consumer.physical_dimension;
    const bool units_compatible =
        producer.unit.empty() ||
        consumer.unit.empty() ||
        producer.unit == consumer.unit;
    return producer.rank == consumer.rank &&
           producer.components == consumer.components &&
           producer.component_layout == consumer.component_layout &&
           producer.tensor_extents == consumer.tensor_extents &&
           producer.tensor_packing == consumer.tensor_packing &&
           dimensions_compatible &&
           units_compatible;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
