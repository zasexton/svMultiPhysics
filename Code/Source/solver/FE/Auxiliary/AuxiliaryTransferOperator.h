#ifndef SVMP_FE_AUXILIARY_TRANSFER_OPERATOR_H
#define SVMP_FE_AUXILIARY_TRANSFER_OPERATOR_H

/**
 * @file AuxiliaryTransferOperator.h
 * @brief Explicit transfer operators for auxiliary state restart,
 *        repartitioning, and remeshing.
 *
 * Promotes the hook-based transfer mechanism in `AuxiliaryStateManager`
 * into a structured operator framework with:
 * - Conservative and interpolatory transfer policies
 * - Layout/version validation for restart payloads
 * - Formulation-defined remap callbacks per scope
 * - Diagnostics for failed or lossy transfer paths
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Transfer policy
// ---------------------------------------------------------------------------

/**
 * @brief Transfer strategy for auxiliary state during mesh changes.
 */
enum class AuxiliaryTransferStrategy : std::uint8_t {
    /// Direct copy (1-to-1 entity mapping, same mesh).
    DirectCopy,

    /// Interpolate from old mesh to new mesh (e.g., L2 projection).
    Interpolatory,

    /// Conservative transfer (preserves integrals, e.g., for density-like state).
    Conservative,

    /// Nearest-entity copy (for repartitioning with unchanged mesh).
    NearestEntity,

    /// Zero-fill on new mesh (discard old state).
    ZeroFill,

    /// Formulation-defined custom transfer.
    Custom
};

// ---------------------------------------------------------------------------
//  Restart payload descriptor
// ---------------------------------------------------------------------------

/**
 * @brief Metadata for validating a restart payload against the current layout.
 */
struct AuxiliaryRestartSchema {
    std::string block_name{};
    int component_count{0};
    std::string scope_name{};           ///< "Global", "Node", etc.
    std::string ordering_name{};        ///< "ByEntityThenComponent", etc.
    std::string deployment_region_kind{};
    std::string region_identity{};      ///< Deployment region identity
    std::string region_version{};
    std::size_t entity_count{0};
    std::size_t owned_entity_count{0};
    std::size_t storage_size{0};
    std::size_t history_depth{0};
    std::vector<std::size_t> entity_ids{};
    std::vector<std::size_t> qp_offsets{};
    std::vector<std::size_t> qp_cell_ids{};
    std::vector<AuxiliaryRegionMembershipMetadata> region_membership{};

    /// Schema version hash (for detecting format changes).
    std::uint64_t schema_hash{0};
};

/**
 * @brief Result of validating a restart payload.
 */
struct RestartValidationResult {
    bool valid{true};
    std::vector<std::string> errors{};
    std::vector<std::string> warnings{};
};

// ---------------------------------------------------------------------------
//  Transfer operator
// ---------------------------------------------------------------------------

/**
 * @brief Transfer operator for one auxiliary block.
 */
class AuxiliaryTransferOperator {
public:
    /**
     * @brief Callback for custom transfer.
     *
     * Arguments: (old_data, old_entity_count, new_entity_count, output)
     */
    using CustomTransferFn = std::function<void(
        std::span<const Real> old_data,
        std::size_t old_entity_count,
        std::size_t new_entity_count,
        std::span<Real> output)>;

    AuxiliaryTransferOperator() = default;

    explicit AuxiliaryTransferOperator(std::string block_name,
                                       AuxiliaryTransferStrategy strategy =
                                           AuxiliaryTransferStrategy::DirectCopy);

    /// Set a custom transfer function (for Custom strategy).
    void setCustomTransfer(CustomTransferFn fn) { custom_fn_ = std::move(fn); }

    /// Block name this operator applies to.
    [[nodiscard]] const std::string& blockName() const noexcept
    {
        return block_name_;
    }

    /// Transfer strategy.
    [[nodiscard]] AuxiliaryTransferStrategy strategy() const noexcept
    {
        return strategy_;
    }

    /**
     * @brief Execute the transfer.
     *
     * @param old_data          Source data from old mesh/partition.
     * @param old_entity_count  Entity count on old mesh.
     * @param new_entity_count  Entity count on new mesh.
     * @param stride            Components per entity.
     * @param output            Destination buffer (new_entity_count * stride).
     */
    void execute(std::span<const Real> old_data,
                  std::size_t old_entity_count,
                  std::size_t new_entity_count,
                  int stride,
                  std::span<Real> output) const;

    // -----------------------------------------------------------------
    //  Restart validation
    // -----------------------------------------------------------------

    /**
     * @brief Build a schema descriptor for the current block layout.
     */
    [[nodiscard]] static AuxiliaryRestartSchema buildSchema(
        const std::string& block_name,
        int component_count,
        AuxiliaryStateScope scope,
        AuxiliaryEntityOrdering ordering,
        const std::string& region_identity,
        std::size_t entity_count,
        std::size_t history_depth,
        const AuxiliaryEntityRemapMetadata* entity_metadata = nullptr);

    /**
     * @brief Validate a restart payload against expected schema.
     */
    [[nodiscard]] static RestartValidationResult validateRestart(
        const AuxiliaryRestartSchema& expected,
        const AuxiliaryRestartSchema& payload);

    // -----------------------------------------------------------------
    //  Diagnostics
    // -----------------------------------------------------------------

    [[nodiscard]] const std::vector<std::string>& diagnostics() const noexcept
    {
        return diagnostics_;
    }

private:
    std::string block_name_{};
    AuxiliaryTransferStrategy strategy_{AuxiliaryTransferStrategy::DirectCopy};
    CustomTransferFn custom_fn_{};
    mutable std::vector<std::string> diagnostics_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_TRANSFER_OPERATOR_H
