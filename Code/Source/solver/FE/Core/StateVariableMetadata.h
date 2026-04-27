#ifndef SVMP_FE_CORE_STATE_VARIABLE_METADATA_H
#define SVMP_FE_CORE_STATE_VARIABLE_METADATA_H

/**
 * @file StateVariableMetadata.h
 * @brief Physics-neutral metadata for history/state variables under moving geometry.
 *
 * The FE library owns storage, lifecycle, transaction, and invalidation
 * contracts. Physics and constitutive implementations own the actual update
 * equations and any frame transformations they require.
 */

#include "Core/FEException.h"
#include "Core/Types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace state {

enum class StateVariableFrame : std::uint8_t {
    FrameIndependent,
    Reference,
    CurrentSpatial,
    Material,
    InterfaceLocal,
    UserDefined
};

enum class StateVariableLifecycle : std::uint8_t {
    CommittedOld,
    TrialWork,
    Accepted,
    RolledBack
};

enum class StateFrameTransformPolicy : std::uint8_t {
    PreserveValue,
    Rotate,
    PushForward,
    PullBack,
    Project,
    Reinitialize,
    Custom
};

enum class StateFrameTransformEvent : std::uint8_t {
    OrdinaryGeometryMotion,
    ReferenceRebase,
    RemeshTransfer,
    AdaptivityTransfer,
    RepartitionTransfer,
    TransactionRollback
};

enum class StateStorageDomain : std::uint8_t {
    MaterialCellQuadrature,
    MaterialBoundaryFaceQuadrature,
    MaterialInteriorFaceQuadrature,
    AuxiliaryBlock
};

struct StateVariableMetadata {
    std::string name{};
    std::size_t offset_bytes{0};
    std::size_t size_bytes{0};
    StateVariableFrame frame{StateVariableFrame::FrameIndependent};
    StateFrameTransformPolicy transform_policy{StateFrameTransformPolicy::PreserveValue};
    bool ordinary_geometry_motion_preserves_value{true};
    bool participates_in_transactions{true};
    bool participates_in_transfer{true};
    std::string user_frame{};

    [[nodiscard]] bool operator==(const StateVariableMetadata& other) const noexcept
    {
        return name == other.name &&
               offset_bytes == other.offset_bytes &&
               size_bytes == other.size_bytes &&
               frame == other.frame &&
               transform_policy == other.transform_policy &&
               ordinary_geometry_motion_preserves_value == other.ordinary_geometry_motion_preserves_value &&
               participates_in_transactions == other.participates_in_transactions &&
               participates_in_transfer == other.participates_in_transfer &&
               user_frame == other.user_frame;
    }

    [[nodiscard]] bool operator!=(const StateVariableMetadata& other) const noexcept
    {
        return !(*this == other);
    }
};

struct StateFrameTransformRequest {
    StateFrameTransformEvent event{StateFrameTransformEvent::OrdinaryGeometryMotion};
    StateVariableLifecycle source_lifecycle{StateVariableLifecycle::TrialWork};
    StateVariableLifecycle target_lifecycle{StateVariableLifecycle::TrialWork};
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t field_layout_revision{0};
    std::uint64_t reference_rebase_epoch{0};
};

struct StateVariableTransformContext {
    StateFrameTransformRequest request{};
    StateStorageDomain storage_domain{StateStorageDomain::MaterialCellQuadrature};
    std::string_view storage_name{};
    GlobalIndex entity_id{INVALID_GLOBAL_INDEX};
    int quadrature_point{-1};
    StateVariableMetadata variable{};
    std::span<const std::byte> old_value{};
    std::span<std::byte> work_value{};
};

struct StateFrameTransformResult {
    std::size_t variables_seen{0};
    std::size_t variables_requiring_action{0};
    std::size_t variable_instances_transformed{0};
    std::size_t variable_instances_preserved{0};

    [[nodiscard]] bool empty() const noexcept { return variables_seen == 0u; }
};

using StateFrameTransformHook = std::function<void(const StateVariableTransformContext&)>;

[[nodiscard]] inline bool stateVariableRequiresFrameAction(
    const StateVariableMetadata& variable,
    StateFrameTransformEvent event) noexcept
{
    if (event == StateFrameTransformEvent::TransactionRollback &&
        variable.participates_in_transactions) {
        return false;
    }
    if ((event == StateFrameTransformEvent::RemeshTransfer ||
         event == StateFrameTransformEvent::AdaptivityTransfer ||
         event == StateFrameTransformEvent::RepartitionTransfer) &&
        !variable.participates_in_transfer) {
        return false;
    }
    return variable.transform_policy != StateFrameTransformPolicy::PreserveValue ||
           !variable.ordinary_geometry_motion_preserves_value;
}

inline void validateStateVariableMetadata(
    std::span<const StateVariableMetadata> variables,
    std::size_t bytes_per_entity,
    std::string_view caller)
{
    for (const auto& variable : variables) {
        FE_THROW_IF(variable.name.empty(), InvalidArgumentException,
                    std::string(caller) + ": state variable name must not be empty");
        FE_THROW_IF(variable.size_bytes == 0u, InvalidArgumentException,
                    std::string(caller) + ": state variable '" + variable.name +
                        "' has zero size");
        FE_THROW_IF(variable.offset_bytes > bytes_per_entity ||
                        variable.size_bytes > bytes_per_entity - variable.offset_bytes,
                    InvalidArgumentException,
                    std::string(caller) + ": state variable '" + variable.name +
                        "' exceeds the registered state storage size");
        FE_THROW_IF(variable.frame == StateVariableFrame::UserDefined &&
                        variable.user_frame.empty(),
                    InvalidArgumentException,
                    std::string(caller) + ": state variable '" + variable.name +
                        "' uses UserDefined frame without user_frame metadata");
    }
}

} // namespace state
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_STATE_VARIABLE_METADATA_H
