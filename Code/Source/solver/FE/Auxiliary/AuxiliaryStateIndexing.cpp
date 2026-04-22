#include "Auxiliary/AuxiliaryStateIndexing.h"

#include <algorithm>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

namespace {

void validateOffsets(std::span<const std::size_t> offsets,
                     const char* context,
                     bool allow_empty_entity_set = true)
{
    const auto min_size = allow_empty_entity_set ? 1u : 2u;
    FE_THROW_IF(offsets.size() < min_size, InvalidArgumentException,
                std::string(context) + ": offsets must have at least " +
                    std::to_string(min_size) + " entries");
    FE_THROW_IF(offsets.empty() || offsets[0] != 0u, InvalidArgumentException,
                std::string(context) + ": offsets[0] must be 0");
    FE_THROW_IF(!std::is_sorted(offsets.begin(), offsets.end()),
                InvalidArgumentException,
                std::string(context) + ": offsets must be nondecreasing");
}

} // namespace

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRaggedScoped_(
    AuxiliaryStateScope scope,
    std::size_t owned_entity_count,
    std::span<const std::size_t> component_offsets)
{
    validateOffsets(component_offsets,
                    "AuxiliaryBlockIndexing::createRaggedScoped",
                    /*allow_empty_entity_set=*/true);

    const auto total_entities = component_offsets.size() - 1u;
    FE_THROW_IF(owned_entity_count > total_entities, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createRaggedScoped: "
                "owned entity count exceeds total entity count");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = scope;
    idx.layout_mode_ = AuxiliaryLayoutMode::Ragged;
    idx.total_entity_count_ = total_entities;
    idx.owned_entity_count_ = owned_entity_count;
    idx.component_stride_ = 0;
    idx.total_storage_size_ = component_offsets.back();
    idx.component_offsets_.assign(component_offsets.begin(), component_offsets.end());
    return idx;
}

// ---------------------------------------------------------------------------
//  Global
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createGlobal(int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createGlobal: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Global;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = 1;
    idx.owned_entity_count_ = 1;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  Node
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createNode(
    std::size_t n_owned_nodes,
    std::size_t n_ghost_nodes,
    int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createNode: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Node;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = n_owned_nodes + n_ghost_nodes;
    idx.owned_entity_count_ = n_owned_nodes;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = idx.total_entity_count_ *
                              static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  Cell
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createCell(
    std::size_t n_owned_cells,
    int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createCell: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Cell;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = n_owned_cells;
    idx.owned_entity_count_ = n_owned_cells;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = n_owned_cells *
                              static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  QuadraturePoint
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createQuadraturePoint(
    std::span<const std::size_t> qp_offsets,
    int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createQuadraturePoint: stride must be > 0");
    validateOffsets(qp_offsets,
                    "AuxiliaryBlockIndexing::createQuadraturePoint",
                    /*allow_empty_entity_set=*/true);

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::QuadraturePoint;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.qp_offsets_.assign(qp_offsets.begin(), qp_offsets.end());

    const auto total_qps = qp_offsets.back();

    idx.total_entity_count_ = total_qps;
    idx.owned_entity_count_ = total_qps; // QP ownership follows cell ownership
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = total_qps *
                              static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  Ragged Node
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRaggedNode(
    std::size_t n_owned_nodes,
    std::size_t n_ghost_nodes,
    std::span<const std::size_t> component_offsets)
{
    const auto expected_entities = n_owned_nodes + n_ghost_nodes;
    FE_THROW_IF(component_offsets.size() != expected_entities + 1u,
                InvalidArgumentException,
                "AuxiliaryBlockIndexing::createRaggedNode: component_offsets "
                "size must be n_owned_nodes + n_ghost_nodes + 1");
    return createRaggedScoped_(
        AuxiliaryStateScope::Node, n_owned_nodes, component_offsets);
}

// ---------------------------------------------------------------------------
//  Ragged Cell
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRaggedCell(
    std::span<const std::size_t> component_offsets)
{
    const auto n_entities = component_offsets.empty() ? 0u : component_offsets.size() - 1u;
    return createRaggedScoped_(
        AuxiliaryStateScope::Cell, n_entities, component_offsets);
}

// ---------------------------------------------------------------------------
//  Ragged QuadraturePoint
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRaggedQuadraturePoint(
    std::span<const std::size_t> qp_offsets,
    std::span<const std::size_t> component_offsets)
{
    validateOffsets(qp_offsets,
                    "AuxiliaryBlockIndexing::createRaggedQuadraturePoint(qp_offsets)",
                    /*allow_empty_entity_set=*/true);
    validateOffsets(component_offsets,
                    "AuxiliaryBlockIndexing::createRaggedQuadraturePoint(component_offsets)",
                    /*allow_empty_entity_set=*/true);

    const auto total_qps = qp_offsets.back();
    FE_THROW_IF(component_offsets.size() != total_qps + 1u,
                InvalidArgumentException,
                "AuxiliaryBlockIndexing::createRaggedQuadraturePoint: "
                "component_offsets size must be qp_offsets.back() + 1");

    auto idx = createRaggedScoped_(
        AuxiliaryStateScope::QuadraturePoint, total_qps, component_offsets);
    idx.qp_offsets_.assign(qp_offsets.begin(), qp_offsets.end());
    return idx;
}

// ---------------------------------------------------------------------------
//  Region
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRegion(
    std::size_t n_regions,
    int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createRegion: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Region;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = n_regions;
    idx.owned_entity_count_ = n_regions;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = n_regions *
                              static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  Ragged Region
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createRaggedRegion(
    std::span<const std::size_t> component_offsets)
{
    const auto n_entities = component_offsets.empty() ? 0u : component_offsets.size() - 1u;
    return createRaggedScoped_(
        AuxiliaryStateScope::Region, n_entities, component_offsets);
}

// ---------------------------------------------------------------------------
//  Boundary (one instance per named boundary collection)
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createBoundary(int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createBoundary: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Boundary;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = 1;
    idx.owned_entity_count_ = 1;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = static_cast<std::size_t>(component_stride);
    return idx;
}

// ---------------------------------------------------------------------------
//  Facet (one instance per boundary facet)
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createFacet(
    std::size_t n_boundary_entities,
    int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createFacet: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Facet;
    idx.layout_mode_ = AuxiliaryLayoutMode::FixedStride;
    idx.total_entity_count_ = n_boundary_entities;
    idx.owned_entity_count_ = n_boundary_entities;
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = n_boundary_entities *
                              static_cast<std::size_t>(component_stride);
    return idx;
}

} // namespace systems
} // namespace FE
} // namespace svmp
