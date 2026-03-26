#include "Systems/AuxiliaryTransferOperator.h"

#include <algorithm>
#include <cstring>

namespace svmp {
namespace FE {
namespace systems {

AuxiliaryTransferOperator::AuxiliaryTransferOperator(
    std::string block_name, AuxiliaryTransferStrategy strategy)
    : block_name_(std::move(block_name))
    , strategy_(strategy)
{
}

void AuxiliaryTransferOperator::execute(
    std::span<const Real> old_data,
    std::size_t old_entity_count,
    std::size_t new_entity_count,
    int stride,
    std::span<Real> output) const
{
    diagnostics_.clear();
    const auto new_size = new_entity_count * static_cast<std::size_t>(stride);

    FE_THROW_IF(output.size() < new_size, InvalidArgumentException,
                "AuxiliaryTransferOperator: output too small");

    switch (strategy_) {
        case AuxiliaryTransferStrategy::DirectCopy: {
            const auto copy_entities = std::min(old_entity_count, new_entity_count);
            const auto copy_size = copy_entities * static_cast<std::size_t>(stride);
            std::copy(old_data.begin(),
                      old_data.begin() + static_cast<std::ptrdiff_t>(copy_size),
                      output.begin());
            // Zero-fill any new entities.
            if (new_entity_count > old_entity_count) {
                std::fill(output.begin() + static_cast<std::ptrdiff_t>(copy_size),
                          output.begin() + static_cast<std::ptrdiff_t>(new_size),
                          Real{0.0});
            }
            if (new_entity_count < old_entity_count) {
                diagnostics_.push_back(
                    "DirectCopy: truncated from " + std::to_string(old_entity_count) +
                    " to " + std::to_string(new_entity_count) + " entities (lossy)");
            }
            break;
        }

        case AuxiliaryTransferStrategy::NearestEntity:
            // For repartitioning with same mesh topology: same as DirectCopy.
            {
                const auto copy_entities = std::min(old_entity_count, new_entity_count);
                const auto copy_size = copy_entities * static_cast<std::size_t>(stride);
                std::copy(old_data.begin(),
                          old_data.begin() + static_cast<std::ptrdiff_t>(copy_size),
                          output.begin());
                if (new_entity_count > old_entity_count) {
                    std::fill(output.begin() + static_cast<std::ptrdiff_t>(copy_size),
                              output.begin() + static_cast<std::ptrdiff_t>(new_size),
                              Real{0.0});
                }
            }
            break;

        case AuxiliaryTransferStrategy::ZeroFill:
            std::fill(output.begin(),
                      output.begin() + static_cast<std::ptrdiff_t>(new_size),
                      Real{0.0});
            diagnostics_.push_back("ZeroFill: all state discarded");
            break;

        case AuxiliaryTransferStrategy::Interpolatory:
        case AuxiliaryTransferStrategy::Conservative:
            // These require mesh-to-mesh mapping infrastructure not available
            // at the auxiliary-state level.  Fall back to DirectCopy with warning.
            diagnostics_.push_back(
                "Transfer strategy " +
                std::to_string(static_cast<int>(strategy_)) +
                " not yet implemented; falling back to DirectCopy");
            {
                const auto copy_entities = std::min(old_entity_count, new_entity_count);
                const auto copy_size = copy_entities * static_cast<std::size_t>(stride);
                std::copy(old_data.begin(),
                          old_data.begin() + static_cast<std::ptrdiff_t>(copy_size),
                          output.begin());
                if (new_entity_count > old_entity_count) {
                    std::fill(output.begin() + static_cast<std::ptrdiff_t>(copy_size),
                              output.begin() + static_cast<std::ptrdiff_t>(new_size),
                              Real{0.0});
                }
            }
            break;

        case AuxiliaryTransferStrategy::Custom:
            FE_THROW_IF(!custom_fn_, InvalidStateException,
                        "AuxiliaryTransferOperator: Custom strategy but no function set");
            custom_fn_(old_data, old_entity_count, new_entity_count, output);
            break;
    }
}

// ---------------------------------------------------------------------------
//  Restart validation
// ---------------------------------------------------------------------------

AuxiliaryRestartSchema AuxiliaryTransferOperator::buildSchema(
    const std::string& block_name,
    int component_count,
    AuxiliaryStateScope scope,
    AuxiliaryEntityOrdering ordering,
    const std::string& region_identity,
    std::size_t entity_count,
    std::size_t history_depth)
{
    AuxiliaryRestartSchema schema;
    schema.block_name = block_name;
    schema.component_count = component_count;
    schema.entity_count = entity_count;
    schema.storage_size = entity_count * static_cast<std::size_t>(component_count);
    schema.history_depth = history_depth;
    schema.region_identity = region_identity;

    switch (scope) {
        case AuxiliaryStateScope::Global: schema.scope_name = "Global"; break;
        case AuxiliaryStateScope::Node: schema.scope_name = "Node"; break;
        case AuxiliaryStateScope::Cell: schema.scope_name = "Cell"; break;
        case AuxiliaryStateScope::QuadraturePoint: schema.scope_name = "QuadraturePoint"; break;
        case AuxiliaryStateScope::BoundaryEntity: schema.scope_name = "BoundaryEntity"; break;
    }

    switch (ordering) {
        case AuxiliaryEntityOrdering::ByEntityThenComponent:
            schema.ordering_name = "ByEntityThenComponent"; break;
        case AuxiliaryEntityOrdering::ByComponentThenEntity:
            schema.ordering_name = "ByComponentThenEntity"; break;
    }

    // Simple hash of the schema fields.
    std::size_t h = std::hash<std::string>{}(schema.block_name);
    h ^= std::hash<int>{}(schema.component_count) * 2654435761u;
    h ^= std::hash<std::string>{}(schema.scope_name) * 40503u;
    h ^= std::hash<std::string>{}(schema.ordering_name) * 12345u;
    h ^= std::hash<std::string>{}(schema.region_identity) * 67890u;
    schema.schema_hash = static_cast<std::uint64_t>(h);

    return schema;
}

RestartValidationResult AuxiliaryTransferOperator::validateRestart(
    const AuxiliaryRestartSchema& expected,
    const AuxiliaryRestartSchema& payload)
{
    RestartValidationResult result;

    if (payload.block_name != expected.block_name) {
        result.valid = false;
        result.errors.push_back(
            "Block name mismatch: expected '" + expected.block_name +
            "', got '" + payload.block_name + "'");
    }

    if (payload.component_count != expected.component_count) {
        result.valid = false;
        result.errors.push_back(
            "Component count mismatch: expected " +
            std::to_string(expected.component_count) +
            ", got " + std::to_string(payload.component_count));
    }

    if (payload.scope_name != expected.scope_name) {
        result.valid = false;
        result.errors.push_back(
            "Scope mismatch: expected '" + expected.scope_name +
            "', got '" + payload.scope_name + "'");
    }

    if (payload.ordering_name != expected.ordering_name) {
        result.valid = false;
        result.errors.push_back(
            "Ordering mismatch: expected '" + expected.ordering_name +
            "', got '" + payload.ordering_name + "'");
    }

    if (payload.region_identity != expected.region_identity) {
        result.valid = false;
        result.errors.push_back(
            "Region mismatch: expected '" + expected.region_identity +
            "', got '" + payload.region_identity + "'");
    }

    if (payload.entity_count != expected.entity_count) {
        result.warnings.push_back(
            "Entity count changed: " + std::to_string(payload.entity_count) +
            " → " + std::to_string(expected.entity_count) +
            " (transfer may be needed)");
    }

    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
