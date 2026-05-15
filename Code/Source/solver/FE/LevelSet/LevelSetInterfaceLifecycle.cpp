#include "LevelSet/LevelSetInterfaceLifecycle.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] std::size_t cornerCount(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    default:
        return 0u;
    }
}

[[nodiscard]] Real coefficientAtVertex(const dofs::EntityDofMap& entity_map,
                                       GlobalIndex vertex,
                                       std::span<const Real> coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "generated level-set interface requires one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::invalid_argument(
            "generated level-set interface found a vertex DOF outside the coefficient span");
    }
    return coefficients[static_cast<std::size_t>(dof)];
}

[[nodiscard]] FieldId resolveLevelSetField(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options)
{
    const auto field = system.findFieldByName(options.level_set_field_name);
    if (field == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "generated level-set interface references unknown field '" +
            options.level_set_field_name + "'");
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "generated level-set interface field '" +
            options.level_set_field_name + "' must be scalar");
    }
    return field;
}

void appendGeneratedInterfaceCell(
    interfaces::LevelSetInterfaceDomain& domain,
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    std::span<const Real> coefficients,
    GlobalIndex cell_id)
{
    const auto type = mesh.getCellType(cell_id);
    const std::size_t count = cornerCount(type);
    if (count == 0u) {
        throw std::invalid_argument(
            "generated level-set interface encountered an unsupported element type");
    }

    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    mesh.getCellNodes(cell_id, cell_nodes);
    mesh.getCellCoordinates(cell_id, cell_coordinates);
    if (cell_nodes.size() < count || cell_coordinates.size() < count) {
        throw std::invalid_argument(
            "generated level-set interface found incomplete cell geometry");
    }

    interfaces::LevelSetCellCutInput input{};
    input.parent_cell = cell_id;
    input.element_type = type;
    input.node_coordinates.assign(cell_coordinates.begin(),
                                  cell_coordinates.begin() +
                                      static_cast<std::ptrdiff_t>(count));
    input.level_set_values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        input.level_set_values.push_back(
            coefficientAtVertex(entity_map, cell_nodes[i], coefficients));
    }

    interfaces::LevelSetCellCutResult cut_result;
    if (mesh.dimension() == 2) {
        cut_result = interfaces::cutLinearLevelSetCell2D(domain.request(), input);
    } else if (mesh.dimension() == 3) {
        cut_result = interfaces::cutLinearLevelSetCell3D(domain.request(), input);
    } else {
        throw std::invalid_argument(
            "generated level-set interface requires a 2D or 3D mesh");
    }
    if (!cut_result.supported) {
        throw std::invalid_argument(cut_result.diagnostic);
    }
    for (auto& fragment : cut_result.fragments) {
        domain.addFragment(std::move(fragment));
    }
    for (auto& region : cut_result.volume_regions) {
        domain.addVolumeRegion(std::move(region));
    }
}

} // namespace

LevelSetGeneratedInterfaceLifecycle::LevelSetGeneratedInterfaceLifecycle(
    int marker_base,
    int marker_range)
    : marker_registry_(marker_base, marker_range)
{
}

void LevelSetGeneratedInterfaceLifecycle::restoreValueRevision(
    std::uint64_t value_revision) noexcept
{
    value_revision_ = value_revision;
}

LevelSetGeneratedInterfaceResult LevelSetGeneratedInterfaceLifecycle::build(
    const systems::FESystem& system,
    const LevelSetGeneratedInterfaceOptions& options,
    std::span<const Real> solution)
{
    if (options.domain_id.empty()) {
        throw std::invalid_argument("generated level-set interface requires a non-empty domain_id");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("generated level-set interface requires a positive tolerance");
    }
    if (options.quadrature_order < 0) {
        throw std::invalid_argument("generated level-set interface requires nonnegative quadrature_order");
    }
    const int interface_quadrature_order =
        options.interface_quadrature_order >= 0 ? options.interface_quadrature_order
                                                : options.quadrature_order;
    const int volume_quadrature_order =
        options.volume_quadrature_order >= 0 ? options.volume_quadrature_order
                                             : options.quadrature_order;
    if (interface_quadrature_order < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative interface_quadrature_order");
    }
    if (volume_quadrature_order < 0) {
        throw std::invalid_argument(
            "generated level-set interface requires nonnegative volume_quadrature_order");
    }

    const auto field = resolveLevelSetField(system, options);
    const auto& field_dofs = system.fieldDofHandler(field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "generated level-set interface received an incompatible system solution span");
    }
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "generated level-set interface requires a scalar nodal field");
    }
    const auto& mesh = system.meshAccess();
    if (entity_map->numVertices() != mesh.numVertices()) {
        throw std::invalid_argument(
            "generated level-set interface requires field and mesh vertex counts to match");
    }

    const auto marker_source = interfaces::LevelSetInterfaceSource::fromField(field);
    interfaces::GeneratedInterfaceMarkerKey marker_key{};
    marker_key.source = marker_source;
    marker_key.domain_id = options.domain_id;
    marker_key.isovalue = options.isovalue;
    marker_key.requested_marker = options.requested_interface_marker;
    const int marker = marker_registry_.assign(marker_key);

    const auto revision = ++value_revision_;
    interfaces::CutInterfaceDomainRequest request{};
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        field,
        mesh.fieldLayoutRevision(),
        revision);
    request.interface_marker = marker;
    request.isovalue = options.isovalue;
    request.tolerance = options.tolerance;
    request.quadrature_order = options.quadrature_order;
    request.interface_quadrature_order = interface_quadrature_order;
    request.volume_quadrature_order = volume_quadrature_order;
    request.mesh_geometry_revision = mesh.geometryRevision();
    request.mesh_topology_revision = mesh.topologyRevision();
    request.ownership_revision = mesh.ownershipRevision();
    request.keep_degenerate_fragments = options.keep_degenerate_fragments;

    interfaces::LevelSetInterfaceDomain domain(request);
    const auto coefficients = solution.subspan(offset, n_field_dofs);
    mesh.forEachCell([&](GlobalIndex cell_id) {
        appendGeneratedInterfaceCell(domain, mesh, *entity_map, coefficients, cell_id);
    });

    LevelSetGeneratedInterfaceResult result;
    result.interface_marker = marker;
    result.value_revision = revision;
    result.domain = std::move(domain);
    result.summary = result.domain.summary();
    result.success =
        result.summary.active_fragment_count > 0u ||
        result.summary.active_volume_region_count > 0u;
    if (!result.success) {
        result.diagnostic = "generated level-set interface has no active fragments or volume regions";
    }
    return result;
}

} // namespace svmp::FE::level_set
