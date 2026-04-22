#include "Auxiliary/AuxiliaryTransferOperator.h"

#include <algorithm>
#include <cstring>
#include <functional>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] const char* regionKindName(AuxiliaryRegionKind kind) noexcept
{
    switch (kind) {
        case AuxiliaryRegionKind::WholeDomain: return "WholeDomain";
        case AuxiliaryRegionKind::CellSet: return "CellSet";
        case AuxiliaryRegionKind::BoundarySet: return "BoundarySet";
        case AuxiliaryRegionKind::MaterialIdSet: return "MaterialIdSet";
        case AuxiliaryRegionKind::TopologyRegion: return "TopologyRegion";
        case AuxiliaryRegionKind::InterfaceSet: return "InterfaceSet";
        case AuxiliaryRegionKind::FormulationDefined: return "FormulationDefined";
    }
    return "Unknown";
}

template <typename T>
void hashCombine(std::size_t& seed, const T& value)
{
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

template <typename T>
void hashRange(std::size_t& seed, const std::vector<T>& values)
{
    hashCombine(seed, values.size());
    for (const auto& value : values) {
        hashCombine(seed, value);
    }
}

void hashRegionMembership(std::size_t& seed,
                          const std::vector<AuxiliaryRegionMembershipMetadata>& memberships)
{
    hashCombine(seed, memberships.size());
    for (const auto& membership : memberships) {
        hashCombine(seed, membership.region_id);
        hashRange(seed, membership.cell_ids);
        hashRange(seed, membership.node_ids);
        hashRange(seed, membership.boundary_markers);
        hashRange(seed, membership.interface_face_ids);
    }
}

} // namespace

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
    std::size_t history_depth,
    const AuxiliaryEntityRemapMetadata* entity_metadata)
{
    AuxiliaryRestartSchema schema;
    schema.block_name = block_name;
    schema.component_count = component_count;
    schema.entity_count = entity_count;
    schema.owned_entity_count = entity_metadata != nullptr
        ? entity_metadata->owned_entity_count
        : entity_count;
    schema.history_depth = history_depth;
    schema.region_identity = region_identity;

    switch (scope) {
        case AuxiliaryStateScope::Global: schema.scope_name = "Global"; break;
        case AuxiliaryStateScope::Boundary: schema.scope_name = "Boundary"; break;
        case AuxiliaryStateScope::Node: schema.scope_name = "Node"; break;
        case AuxiliaryStateScope::Cell: schema.scope_name = "Cell"; break;
        case AuxiliaryStateScope::QuadraturePoint: schema.scope_name = "QuadraturePoint"; break;
        case AuxiliaryStateScope::Region: schema.scope_name = "Region"; break;
        case AuxiliaryStateScope::Facet: schema.scope_name = "Facet"; break;
    }

    switch (ordering) {
        case AuxiliaryEntityOrdering::ByEntityThenComponent:
            schema.ordering_name = "ByEntityThenComponent"; break;
        case AuxiliaryEntityOrdering::ByComponentThenEntity:
            schema.ordering_name = "ByComponentThenEntity"; break;
    }

    if (entity_metadata != nullptr) {
        schema.deployment_region_kind =
            regionKindName(entity_metadata->deployment_region.kind);
        schema.region_identity = entity_metadata->deployment_region.identity;
        schema.region_version = entity_metadata->deployment_region.version;
        schema.entity_ids = entity_metadata->entity_ids;
        schema.component_offsets = entity_metadata->component_offsets;
        schema.qp_offsets = entity_metadata->qp_offsets;
        schema.qp_cell_ids = entity_metadata->qp_cell_ids;
        schema.region_membership = entity_metadata->region_membership;
    }
    schema.storage_size = !schema.component_offsets.empty()
        ? schema.component_offsets.back()
        : entity_count * static_cast<std::size_t>(component_count);

    // Simple hash of the schema fields.
    std::size_t h = std::hash<std::string>{}(schema.block_name);
    hashCombine(h, schema.component_count);
    hashCombine(h, schema.scope_name);
    hashCombine(h, schema.ordering_name);
    hashCombine(h, schema.deployment_region_kind);
    hashCombine(h, schema.region_identity);
    hashCombine(h, schema.region_version);
    hashCombine(h, schema.entity_count);
    hashCombine(h, schema.owned_entity_count);
    hashCombine(h, schema.storage_size);
    hashCombine(h, schema.history_depth);
    hashRange(h, schema.entity_ids);
    hashRange(h, schema.component_offsets);
    hashRange(h, schema.qp_offsets);
    hashRange(h, schema.qp_cell_ids);
    hashRegionMembership(h, schema.region_membership);
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

    if (payload.deployment_region_kind != expected.deployment_region_kind) {
        result.valid = false;
        result.errors.push_back(
            "Deployment region kind mismatch: expected '" +
            expected.deployment_region_kind + "', got '" +
            payload.deployment_region_kind + "'");
    }

    if (payload.region_version != expected.region_version) {
        result.valid = false;
        result.errors.push_back(
            "Deployment region version mismatch: expected '" +
            expected.region_version + "', got '" + payload.region_version + "'");
    }

    if (payload.owned_entity_count != expected.owned_entity_count) {
        result.valid = false;
        result.errors.push_back(
            "Owned entity count mismatch: expected " +
            std::to_string(expected.owned_entity_count) +
            ", got " + std::to_string(payload.owned_entity_count));
    }

    if (payload.entity_ids != expected.entity_ids) {
        result.valid = false;
        result.errors.push_back("Entity map mismatch: restart/remap requires matching stable entity ids");
    }

    if (payload.component_offsets != expected.component_offsets) {
        result.valid = false;
        result.errors.push_back(
            "Ragged component offsets mismatch: restart/remap requires matching component layout");
    }

    if (payload.qp_offsets != expected.qp_offsets) {
        result.valid = false;
        result.errors.push_back("QP offsets mismatch: restart/remap requires matching quadrature layout");
    }

    if (payload.qp_cell_ids != expected.qp_cell_ids) {
        result.valid = false;
        result.errors.push_back("QP covered-cell map mismatch: restart/remap requires matching cell ids");
    }

    if (payload.region_membership != expected.region_membership) {
        result.valid = false;
        result.errors.push_back(
            "Region membership mismatch: restart/remap requires matching topology-region maps");
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
