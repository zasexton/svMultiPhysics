#include "Auxiliary/AuxiliaryStateIndexing.h"

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Global
// ---------------------------------------------------------------------------

AuxiliaryBlockIndexing AuxiliaryBlockIndexing::createGlobal(int component_stride)
{
    FE_THROW_IF(component_stride <= 0, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createGlobal: stride must be > 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::Global;
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
    FE_THROW_IF(qp_offsets.size() < 2u, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createQuadraturePoint: "
                "qp_offsets must have >= 2 entries");
    FE_THROW_IF(qp_offsets[0] != 0u, InvalidArgumentException,
                "AuxiliaryBlockIndexing::createQuadraturePoint: "
                "qp_offsets[0] must be 0");

    AuxiliaryBlockIndexing idx;
    idx.scope_ = AuxiliaryStateScope::QuadraturePoint;
    idx.qp_offsets_.assign(qp_offsets.begin(), qp_offsets.end());

    const auto n_cells = qp_offsets.size() - 1;
    const auto total_qps = qp_offsets.back();

    idx.total_entity_count_ = total_qps;
    idx.owned_entity_count_ = total_qps; // QP ownership follows cell ownership
    idx.component_stride_ = component_stride;
    idx.total_storage_size_ = total_qps *
                              static_cast<std::size_t>(component_stride);
    return idx;
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
