/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

constexpr FE::Real kPressureGaugeLevelSetMargin = 1.0e-8;

struct LevelSetVertexFieldView {
    const FE::Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

struct VertexScalarFieldView {
    const FE::Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

[[nodiscard]] LevelSetVertexFieldView activePressureLevelSetField(
    const FE::systems::FESystem& system,
    const ActivePressureDomain& active_domain,
    FE::GlobalIndex n_vertices,
    std::string_view context)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a native mesh vertex level-set field");
    }

    const auto& local_mesh = native_mesh->local_mesh();
    const auto& bc = *active_domain.boundary;
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex,
                               bc.level_set_field_name)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not find vertex level-set field '" +
            bc.level_set_field_name + "'");
    }

    const auto handle = MeshFields::get_field_handle(
        local_mesh, EntityKind::Vertex, bc.level_set_field_name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a Float64 vertex level-set field");
    }

    const auto components = MeshFields::field_components(local_mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires at least one level-set component");
    }

    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " level-set field has fewer entries than pressure vertices");
    }

    const auto* values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " found an empty level-set field");
    }

    return LevelSetVertexFieldView{values, components, entity_count};
#else
    (void)system;
    (void)active_domain;
    (void)n_vertices;
    (void)context;
    throw std::runtime_error(
        "IncompressibleNavierStokesVMSModule: active-domain pressure operations require native mesh support");
#endif
}

[[nodiscard]] VertexScalarFieldView pressureInitializationField(
    const FE::systems::FESystem& system,
    std::string_view field_name,
    FE::GlobalIndex n_vertices)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires a native mesh");
    }

    const auto& local_mesh = native_mesh->local_mesh();
    const std::string field_name_string(field_name);
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, field_name_string)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization could not find vertex field '" +
            field_name_string + "'");
    }

    const auto handle = MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, field_name_string);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires a Float64 vertex field");
    }

    const auto components = MeshFields::field_components(local_mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires at least one component");
    }

    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization has fewer entries than pressure vertices");
    }

    const auto* values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization found an empty vertex field");
    }

    return VertexScalarFieldView{values, components, entity_count};
#else
    (void)system;
    (void)field_name;
    (void)n_vertices;
    throw std::runtime_error(
        "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization requires native mesh support");
#endif
}

std::size_t initializeStateFieldFromMeshVertexField(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0,
    FE::FieldId field_id,
    std::string_view context)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (field_id == FE::INVALID_FIELD_ID) {
        return 0u;
    }

    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        return 0u;
    }

    const auto& rec = system.fieldRecord(field_id);
    const auto& local_mesh = native_mesh->local_mesh();
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, rec.name)) {
        return 0u;
    }

    const auto handle =
        MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, rec.name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires a Float64 vertex mesh field '" + rec.name + "'");
    }

    const auto components = static_cast<std::size_t>(std::max(1, rec.components));
    const auto mesh_components = MeshFields::field_components(local_mesh, handle);
    if (mesh_components < components) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " mesh field '" + rec.name + "' has fewer components than the FE field");
    }

    const auto n_vertices = static_cast<FE::GlobalIndex>(native_mesh->n_vertices());
    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " mesh field '" + rec.name + "' has fewer entries than mesh vertices");
    }

    const auto* mesh_values = MeshFields::field_data_as<FE::Real>(local_mesh, handle);
    if (mesh_values == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " found an empty mesh field '" + rec.name + "'");
    }

    const auto& field_dofs = system.fieldDofHandler(field_id);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr || entity_map->numVertices() < n_vertices) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " requires FE vertex DOF metadata for field '" + rec.name + "'");
    }

    const auto field_offset = system.fieldDofOffset(field_id);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    std::vector<FE::Real> coefficients(n_field_dofs, FE::Real{0.0});
    std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
    const auto projection =
        system.projectMeshVertexValuesToFieldCoefficients(
            field_id,
            std::span<const FE::Real>(
                reinterpret_cast<const FE::Real*>(mesh_values),
                entity_count * mesh_components),
            mesh_components,
            std::span<FE::Real>(coefficients.data(), coefficients.size()),
            std::span<std::uint8_t>(assigned.data(), assigned.size()),
            std::string("IncompressibleNavierStokesVMSModule: ") +
                std::string(context));
    if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not safely project " +
            std::to_string(projection.unassigned_dofs) +
            " field coefficient(s) from mesh vertices for field '" +
            rec.name + "'");
    }

    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    dofs.reserve(n_field_dofs);
    values.reserve(n_field_dofs);

    for (std::size_t dof = 0; dof < coefficients.size(); ++dof) {
        if (assigned[dof] == 0u) {
            continue;
        }
        dofs.push_back(field_offset + static_cast<FE::GlobalIndex>(dof));
        values.push_back(coefficients[dof]);
    }

    if (dofs.empty()) {
        return 0u;
    }

    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " could not create a vector view");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();
    return dofs.size();
#else
    (void)system;
    (void)u0;
    (void)field_id;
    (void)context;
    return 0u;
#endif
}

[[nodiscard]] bool pressureVertexOnActiveSide(
    FE::Real phi,
    FE::Real isovalue,
    FreeSurfaceActiveDomain active_domain) noexcept
{
    switch (active_domain) {
    case FreeSurfaceActiveDomain::None:
        return true;
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return phi <= isovalue;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return phi >= isovalue;
    }
    return true;
}

[[nodiscard]] FE::geometry::CutIntegrationSide activeDomainIntegrationSide(
    FreeSurfaceActiveDomain active_domain) noexcept
{
    switch (active_domain) {
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::geometry::CutIntegrationSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::geometry::CutIntegrationSide::Positive;
    case FreeSurfaceActiveDomain::None:
        break;
    }
    return FE::geometry::CutIntegrationSide::Negative;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] bool pressureCellHasActiveMeasureBySign(
    const LevelSetVertexFieldView& level_set_values,
    FreeSurfaceActiveDomain active_domain,
    FE::Real isovalue,
    const index_t* vertices,
    std::size_t vertex_count)
{
    if (vertices == nullptr || vertex_count == 0u) {
        return false;
    }
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const auto vertex = static_cast<FE::GlobalIndex>(vertices[i]);
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= level_set_values.entity_count) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain "
                << "pressure initialization references vertex " << vertex
                << " outside the level-set field";
            throw std::runtime_error(oss.str());
        }
        const auto phi = level_set_values.values[
            static_cast<std::size_t>(vertex) * level_set_values.components];
        const bool has_positive_active_measure =
            active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                ? phi < isovalue
                : phi > isovalue;
        if (has_positive_active_measure) {
            return true;
        }
    }
    return false;
}
#endif

[[nodiscard]] std::vector<unsigned char> activePressureSupportVertices(
    const FE::systems::FESystem& system,
    const ActivePressureDomain& active_pressure_domain,
    const LevelSetVertexFieldView& level_set_values,
    FE::GlobalIndex n_vertices)
{
    std::vector<unsigned char> active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto phi = level_set_values.values[
                static_cast<std::size_t>(vertex) * level_set_values.components];
            active_support[static_cast<std::size_t>(vertex)] =
                pressureVertexOnActiveSide(
                    phi,
                    active_pressure_domain.boundary->level_set_isovalue,
                    active_pressure_domain.active_domain)
                    ? static_cast<unsigned char>(1)
                    : static_cast<unsigned char>(0);
        }
        return active_support;
    }

    const auto& mesh = native_mesh->local_mesh();
    const auto mark_cell_active = [&](FE::GlobalIndex cell) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain "
                << "pressure initialization references cell " << cell
                << " outside the mesh";
            throw std::runtime_error(oss.str());
        }
        const auto [vertices, count] =
            mesh.cell_vertices_span(static_cast<index_t>(cell));
        if (vertices == nullptr || count == 0u) {
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto vertex = static_cast<FE::GlobalIndex>(vertices[i]);
            if (vertex >= 0 && vertex < n_vertices) {
                active_support[static_cast<std::size_t>(vertex)] =
                    static_cast<unsigned char>(1);
            }
        }
    };

    const auto* cut_context = system.cutIntegrationContext();
    if (cut_context != nullptr) {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
    }
    const auto& bc = *active_pressure_domain.boundary;
    if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume &&
        bc.interface_marker >= 0 &&
        cut_context != nullptr &&
        cut_context->hasGeneratedVolumeMarker(bc.interface_marker)) {
        const auto rule_indices =
            cut_context->generatedVolumeRuleIndexSpanForMarkerAndSide(
                bc.interface_marker,
                activeDomainIntegrationSide(active_pressure_domain.active_domain));
        const auto& metadata = cut_context->metadata();
        for (const auto index : rule_indices) {
            if (index >= metadata.size()) {
                continue;
            }
            const auto& rule_metadata = metadata[index];
            const auto cell = rule_metadata.parent_entity >= 0
                                  ? rule_metadata.parent_entity
                                  : rule_metadata.cell;
            mark_cell_active(static_cast<FE::GlobalIndex>(cell));
        }
    } else {
        for (FE::GlobalIndex cell = 0;
             cell < static_cast<FE::GlobalIndex>(mesh.n_cells());
             ++cell) {
            const auto [vertices, count] =
                mesh.cell_vertices_span(static_cast<index_t>(cell));
            if (pressureCellHasActiveMeasureBySign(
                    level_set_values,
                    active_pressure_domain.active_domain,
                    bc.level_set_isovalue,
                    vertices,
                    count)) {
                mark_cell_active(cell);
            }
        }
    }
#else
    for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto phi = level_set_values.values[
            static_cast<std::size_t>(vertex) * level_set_values.components];
        active_support[static_cast<std::size_t>(vertex)] =
            pressureVertexOnActiveSide(
                phi,
                active_pressure_domain.boundary->level_set_isovalue,
                active_pressure_domain.active_domain)
                ? static_cast<unsigned char>(1)
                : static_cast<unsigned char>(0);
    }
#endif

    return active_support;
}

[[nodiscard]] std::optional<FE::GlobalIndex> pressureConstraintLocalVertex(
    const FE::systems::FESystem& system,
    IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType id_type,
    FE::GlobalIndex node_id)
{
    if (node_id < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id");
    }

    switch (id_type) {
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
        return node_id;
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
        break;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires a native mesh for global vertex ids");
    }
    const auto local_vertex =
        native_mesh->local_mesh().global_to_local_vertex(static_cast<gid_t>(node_id));
    if (local_vertex == INVALID_INDEX) {
        return std::nullopt;
    }
    return static_cast<FE::GlobalIndex>(local_vertex);
#else
    (void)system;
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires native mesh support for global vertex ids");
#endif
}

} // namespace

void validateActiveDomainPressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    if (options.node_pressure_constraints.values.empty()) {
        return;
    }

    std::unordered_map<FE::GlobalIndex, FE::Real> pressure_by_node;
    const auto n_vertices = system.meshAccess().numVertices();
    for (const auto& constraint : options.node_pressure_constraints.values) {
        if (constraint.node_id < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id");
        }
        if (!std::isfinite(constraint.pressure)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint value must be finite");
        }
        const auto [existing, inserted] = pressure_by_node.emplace(
            constraint.node_id, constraint.pressure);
        if (!inserted && existing->second != constraint.pressure) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: duplicate pressure constraint node has conflicting values");
        }
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (local_vertex.has_value() &&
            (*local_vertex < 0 || *local_vertex >= n_vertices)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references vertex " +
                std::to_string(constraint.node_id) +
                " outside the local pressure mesh");
        }
    }

    const auto active_pressure_domain = activePressureDomainFor(free_surfaces);
    if (!active_pressure_domain.has_value()) {
        return;
    }

    const auto level_set_values = activePressureLevelSetField(
        system,
        *active_pressure_domain,
        n_vertices,
        "active-domain pressure constraint validation");
    const auto& bc = *active_pressure_domain->boundary;
    std::size_t checked_local_constraints = 0u;
    std::size_t skipped_nonlocal_constraints = 0u;
    FE::Real constraint_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real constraint_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    FE::Real min_signed_gap = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_signed_gap = -std::numeric_limits<FE::Real>::infinity();
    for (const auto& constraint : options.node_pressure_constraints.values) {
        constraint_pressure_min = std::min(constraint_pressure_min, constraint.pressure);
        constraint_pressure_max = std::max(constraint_pressure_max, constraint.pressure);
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (!local_vertex.has_value()) {
            ++skipped_nonlocal_constraints;
            continue;
        }
        if (*local_vertex < 0 || *local_vertex >= n_vertices) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure constraint references vertex " +
                std::to_string(constraint.node_id) +
                " outside the local pressure mesh");
        }

        const auto phi = level_set_values.values[
            static_cast<std::size_t>(*local_vertex) * level_set_values.components];
        const auto signed_gap = phi - bc.level_set_isovalue;
        ++checked_local_constraints;
        min_signed_gap = std::min(min_signed_gap, signed_gap);
        max_signed_gap = std::max(max_signed_gap, signed_gap);
        if (!pressureVertexOnActiveSide(phi,
                                        bc.level_set_isovalue,
                                        active_pressure_domain->active_domain)) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain pressure "
                << "constraint node_id=" << constraint.node_id
                << " local_vertex=" << *local_vertex
                << " is on the dry side for Active_domain="
                << pressureActiveDomainName(active_pressure_domain->active_domain)
                << " phi=" << phi
                << " isovalue=" << bc.level_set_isovalue;
            throw std::invalid_argument(oss.str());
        }
        if (std::abs(signed_gap) < kPressureGaugeLevelSetMargin) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: active-domain pressure "
                << "constraint node_id=" << constraint.node_id
                << " local_vertex=" << *local_vertex
                << " is too close to the level-set interface: |phi-isovalue|="
                << std::abs(signed_gap)
                << " margin=" << kPressureGaugeLevelSetMargin;
            throw std::invalid_argument(oss.str());
        }
    }
    if (!std::isfinite(min_signed_gap)) {
        min_signed_gap = FE::Real{0.0};
    }
    if (!std::isfinite(max_signed_gap)) {
        max_signed_gap = FE::Real{0.0};
    }
    if (!std::isfinite(constraint_pressure_min)) {
        constraint_pressure_min = FE::Real{0.0};
    }
    if (!std::isfinite(constraint_pressure_max)) {
        constraint_pressure_max = FE::Real{0.0};
    }
    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: pressure gauge diagnostic"
        << " diagnostic=pressure_gauge_check"
        << " constraints=" << options.node_pressure_constraints.values.size()
        << " checked_local_constraints=" << checked_local_constraints
        << " skipped_nonlocal_constraints=" << skipped_nonlocal_constraints
        << " constraint_pressure_min=" << constraint_pressure_min
        << " constraint_pressure_max=" << constraint_pressure_max
        << " Active_domain="
        << pressureActiveDomainName(active_pressure_domain->active_domain)
        << " isovalue=" << bc.level_set_isovalue
        << " min_signed_gap=" << min_signed_gap
        << " max_signed_gap=" << max_signed_gap
        << " margin=" << kPressureGaugeLevelSetMargin;
    FE_LOG_INFO(oss.str());
}

void initializeNavierStokesState(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>&
        hydrostatic_pressure)
{
    const auto& init = options.hydrostatic_pressure_initialization;

    std::size_t mesh_field_initialization_dofs = 0u;
    mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
        system,
        u0,
        system.findFieldByName(options.velocity_field_name),
        "mesh-field velocity initialization");

    if (!init.enabled) {
        mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
            system,
            u0,
            system.findFieldByName(options.pressure_field_name),
            "mesh-field pressure initialization");
        if (mesh_field_initialization_dofs > 0u) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: mesh-field "
                   "initialization diagnostic=mesh_field_initialization"
                << " initialized_dofs=" << mesh_field_initialization_dofs
                << " velocity_field='" << options.velocity_field_name << "'"
                << " pressure_field='" << options.pressure_field_name << "'";
            FE_LOG_INFO(oss.str());
        }
        return;
    }

    if (mesh_field_initialization_dofs > 0u) {
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: mesh-field "
               "initialization diagnostic=mesh_field_initialization"
            << " initialized_dofs=" << mesh_field_initialization_dofs
            << " velocity_field='" << options.velocity_field_name << "'";
        FE_LOG_INFO(oss.str());
    }

    const auto p_id = system.findFieldByName(options.pressure_field_name);
    if (p_id == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not find pressure field '" +
            options.pressure_field_name + "'");
    }

    const auto& pressure_dofs = system.fieldDofHandler(p_id);
    const auto* entity_map = pressure_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization requires vertex DOF metadata");
    }

    const auto& mesh = system.meshAccess();
    const auto pressure_offset = system.fieldDofOffset(p_id);
    const auto n_vertices = mesh.numVertices();
    const auto active_pressure_domain =
        activePressureDomainFor(options.free_surface);
    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume &&
        init.field_name.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: CutVolume active-domain "
            "hydrostatic pressure initialization requires a level-set-aware "
            "mesh vertex field selected by Hydrostatic_pressure_field_name");
    }
    std::optional<LevelSetVertexFieldView> level_set_values;
    if (active_pressure_domain.has_value()) {
        level_set_values =
            activePressureLevelSetField(system, *active_pressure_domain,
                                        n_vertices,
                                        "active-domain hydrostatic pressure initialization");
    }
    std::optional<std::vector<unsigned char>> active_pressure_support;
    if (active_pressure_domain.has_value() && level_set_values.has_value()) {
        active_pressure_support =
            activePressureSupportVertices(
                system, *active_pressure_domain, *level_set_values, n_vertices);
    }
    std::optional<VertexScalarFieldView> pressure_initialization_field;
    if (!init.field_name.empty()) {
        pressure_initialization_field =
            pressureInitializationField(system, init.field_name, n_vertices);
        for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto value = pressure_initialization_field->values[
                static_cast<std::size_t>(vertex) *
                pressure_initialization_field->components];
            if (!std::isfinite(value)) {
                std::ostringstream oss;
                oss << "IncompressibleNavierStokesVMSModule: hydrostatic "
                       "pressure mesh vertex field '"
                    << init.field_name
                    << "' contains a non-finite value at vertex " << vertex;
                throw std::runtime_error(oss.str());
            }
        }
    }

    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    std::size_t active_wet_vertices = 0u;
    std::size_t active_dry_vertices = 0u;
    std::size_t active_support_pressure_vertices = 0u;
    std::size_t dry_sign_active_support_pressure_vertices = 0u;
    FE::Real initialized_pressure_min =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real initialized_pressure_max =
        -std::numeric_limits<FE::Real>::infinity();
    FE::Real wet_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real wet_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    std::size_t checked_gauge_constraints = 0u;
    std::size_t skipped_gauge_constraints = 0u;
    FE::Real gauge_pressure_min = std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_pressure_max = -std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_initialized_pressure_min =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_initialized_pressure_max =
        -std::numeric_limits<FE::Real>::infinity();
    FE::Real gauge_pressure_max_abs_error = FE::Real{0.0};

    for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        if (vertex_dofs.empty()) {
            continue;
        }

        const auto x = mesh.getNodeCoordinates(vertex);
        bool initialize_hydrostatic = true;
        if (active_pressure_domain.has_value()) {
            const auto vertex_offset =
                static_cast<std::size_t>(vertex) * level_set_values->components;
            const auto phi = level_set_values->values[vertex_offset];
            const bool active_side = pressureVertexOnActiveSide(
                phi,
                active_pressure_domain->boundary->level_set_isovalue,
                active_pressure_domain->active_domain);
            initialize_hydrostatic =
                active_pressure_support.has_value()
                    ? ((*active_pressure_support)[static_cast<std::size_t>(vertex)] !=
                       static_cast<unsigned char>(0))
                    : active_side;
            if (active_side) {
                ++active_wet_vertices;
            } else {
                ++active_dry_vertices;
            }
            if (initialize_hydrostatic) {
                ++active_support_pressure_vertices;
                if (!active_side) {
                    ++dry_sign_active_support_pressure_vertices;
                }
            }
        }
        const FE::Real pressure = initialize_hydrostatic
            ? (pressure_initialization_field.has_value()
                   ? pressure_initialization_field->values[
                         static_cast<std::size_t>(vertex) *
                         pressure_initialization_field->components]
                   : hydrostatic_pressure(x))
            : init.reference_pressure;
        if (!std::isfinite(pressure)) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: hydrostatic pressure "
                   "initialization produced a non-finite value at vertex "
                << vertex;
            if (!init.field_name.empty()) {
                oss << " from mesh vertex field '" << init.field_name << "'";
            }
            throw std::runtime_error(oss.str());
        }
        initialized_pressure_min = std::min(initialized_pressure_min, pressure);
        initialized_pressure_max = std::max(initialized_pressure_max, pressure);
        if (initialize_hydrostatic) {
            wet_pressure_min = std::min(wet_pressure_min, pressure);
            wet_pressure_max = std::max(wet_pressure_max, pressure);
        }

        for (const auto local_dof : vertex_dofs) {
            dofs.push_back(pressure_offset + local_dof);
            values.push_back(pressure);
        }
    }

    for (const auto& constraint : options.node_pressure_constraints.values) {
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (!local_vertex.has_value()) {
            ++skipped_gauge_constraints;
            continue;
        }
        if (*local_vertex < 0 || *local_vertex >= n_vertices) {
            ++skipped_gauge_constraints;
            continue;
        }

        const auto vertex = *local_vertex;
        const auto x = mesh.getNodeCoordinates(vertex);
        bool initialize_hydrostatic = true;
        if (active_pressure_domain.has_value()) {
            const auto vertex_offset =
                static_cast<std::size_t>(vertex) * level_set_values->components;
            const auto phi = level_set_values->values[vertex_offset];
            const bool active_side = pressureVertexOnActiveSide(
                phi,
                active_pressure_domain->boundary->level_set_isovalue,
                active_pressure_domain->active_domain);
            initialize_hydrostatic =
                active_pressure_support.has_value()
                    ? ((*active_pressure_support)[static_cast<std::size_t>(vertex)] !=
                       static_cast<unsigned char>(0))
                    : active_side;
        }
        const FE::Real initialized_pressure = initialize_hydrostatic
            ? (pressure_initialization_field.has_value()
                   ? pressure_initialization_field->values[
                         static_cast<std::size_t>(vertex) *
                         pressure_initialization_field->components]
                   : hydrostatic_pressure(x))
            : init.reference_pressure;

        ++checked_gauge_constraints;
        gauge_pressure_min = std::min(gauge_pressure_min, constraint.pressure);
        gauge_pressure_max = std::max(gauge_pressure_max, constraint.pressure);
        gauge_initialized_pressure_min =
            std::min(gauge_initialized_pressure_min, initialized_pressure);
        gauge_initialized_pressure_max =
            std::max(gauge_initialized_pressure_max, initialized_pressure);
        gauge_pressure_max_abs_error =
            std::max(gauge_pressure_max_abs_error,
                     std::abs(initialized_pressure - constraint.pressure));
    }

    if (dofs.empty()) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization found no pressure vertex DOFs");
    }

    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not create a vector view");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();

    if (active_pressure_domain.has_value()) {
        if (!std::isfinite(initialized_pressure_min)) {
            initialized_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(initialized_pressure_max)) {
            initialized_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(wet_pressure_min)) {
            wet_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(wet_pressure_max)) {
            wet_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_pressure_min)) {
            gauge_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_pressure_max)) {
            gauge_pressure_max = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_initialized_pressure_min)) {
            gauge_initialized_pressure_min = FE::Real{0.0};
        }
        if (!std::isfinite(gauge_initialized_pressure_max)) {
            gauge_initialized_pressure_max = FE::Real{0.0};
        }
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: hydrostatic pressure "
            << "initialization diagnostic=hydrostatic_initialization Active_domain="
            << pressureActiveDomainName(active_pressure_domain->active_domain)
            << " wet_pressure_vertices=" << active_wet_vertices
            << " dry_pressure_vertices=" << active_dry_vertices
            << " active_support_pressure_vertices="
            << active_support_pressure_vertices
            << " dry_sign_active_support_pressure_vertices="
            << dry_sign_active_support_pressure_vertices
            << " reference_pressure=" << init.reference_pressure
            << " initialized_pressure_min=" << initialized_pressure_min
            << " initialized_pressure_max=" << initialized_pressure_max
            << " wet_pressure_min=" << wet_pressure_min
            << " wet_pressure_max=" << wet_pressure_max
            << " gauge_constraints="
            << options.node_pressure_constraints.values.size()
            << " checked_gauge_constraints=" << checked_gauge_constraints
            << " skipped_gauge_constraints=" << skipped_gauge_constraints
            << " gauge_pressure_min=" << gauge_pressure_min
            << " gauge_pressure_max=" << gauge_pressure_max
            << " gauge_initialized_pressure_min="
            << gauge_initialized_pressure_min
            << " gauge_initialized_pressure_max="
            << gauge_initialized_pressure_max
            << " gauge_pressure_max_abs_error="
            << gauge_pressure_max_abs_error;
        if (!init.field_name.empty()) {
            oss << " pressure_field='" << init.field_name << "'";
        }
        FE_LOG_INFO(oss.str());
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
