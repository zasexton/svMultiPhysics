/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cstddef>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    IncompressibleNavierStokesVMSOptions options)
    : velocity_space_(std::move(velocity_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

namespace {

using FreeSurfaceBoundary = IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

[[nodiscard]] bool spacesCompatible(const FE::spaces::FunctionSpace& lhs,
                                    const FE::spaces::FunctionSpace& rhs) noexcept
{
    return lhs.space_type() == rhs.space_type() &&
           lhs.field_type() == rhs.field_type() &&
           lhs.value_dimension() == rhs.value_dimension() &&
           lhs.topological_dimension() == rhs.topological_dimension() &&
           lhs.polynomial_order() == rhs.polynomial_order() &&
           lhs.element_type() == rhs.element_type();
}

[[nodiscard]] FE::FieldId ensureCompatibleUnknownField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    const char* context)
{
    const auto existing = system.findFieldByName(spec.name);
    if (existing == FE::INVALID_FIELD_ID) {
        return system.addField(std::move(spec));
    }

    const auto& rec = system.fieldRecord(existing);
    if (rec.source_kind != FE::systems::FieldSourceKind::Unknown) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' must be an unknown field");
    }
    if (rec.components != spec.components) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' has component count " + std::to_string(rec.components) +
            ", expected " + std::to_string(spec.components));
    }
    if (!rec.space || !spec.space || !spacesCompatible(*rec.space, *spec.space)) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' uses an incompatible function space");
    }

    return existing;
}

[[nodiscard]] FE::FieldId ensureCompatiblePrescribedField(
    FE::systems::FESystem& system,
    FE::systems::FieldSpec spec,
    bool auto_register,
    const char* context)
{
    const auto existing = system.findFieldByName(spec.name);
    if (existing == FE::INVALID_FIELD_ID) {
        if (!auto_register) {
            throw std::invalid_argument(
                std::string(context) + ": prescribed field '" + spec.name +
                "' was requested but is not registered");
        }
        return system.addField(std::move(spec));
    }

    const auto& rec = system.fieldRecord(existing);
    if (rec.source_kind != FE::systems::FieldSourceKind::PrescribedData) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' must be a prescribed data field");
    }
    if (rec.components != spec.components) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' has component count " + std::to_string(rec.components) +
            ", expected " + std::to_string(spec.components));
    }
    if (!rec.space || !spec.space || !spacesCompatible(*rec.space, *spec.space)) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' uses an incompatible function space");
    }

    return existing;
}

constexpr FE::Real kPressureGaugeLevelSetMargin = 1.0e-8;

struct ActivePressureDomain {
    const FreeSurfaceBoundary* boundary{nullptr};
    FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
};

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

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] std::optional<ActivePressureDomain>
activePressureDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    std::optional<ActivePressureDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (bc.implementation != FreeSurfaceImplementation::UnfittedLevelSet) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations are only valid for unfitted level-set free surfaces");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain pressure operations require a non-empty level_set_field_name");
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict pressure operations");
        }
        active_domain = ActivePressureDomain{&bc, bc.active_domain};
    }
    return active_domain;
}

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
    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    dofs.reserve(n_field_dofs);
    values.reserve(n_field_dofs);

    bool all_mesh_vertices_have_vertex_dofs = true;
    for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        if (entity_map->getVertexDofs(vertex).size() != components) {
            all_mesh_vertices_have_vertex_dofs = false;
            break;
        }
    }

    if (all_mesh_vertices_have_vertex_dofs) {
        for (FE::GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto vertex_dofs = entity_map->getVertexDofs(vertex);
            const auto v_base = static_cast<std::size_t>(vertex) * mesh_components;
            for (std::size_t c = 0; c < components; ++c) {
                dofs.push_back(field_offset + vertex_dofs[c]);
                values.push_back(mesh_values[v_base + c]);
            }
        }
    } else {
        std::vector<unsigned char> coefficient_written(n_field_dofs, 0u);
        for (index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
            auto [cell_vertices, n_cell_vertices] = local_mesh.cell_vertices_span(cell);
            if (cell_vertices == nullptr || n_cell_vertices == 0u) {
                throw std::runtime_error(
                    "IncompressibleNavierStokesVMSModule: " + std::string(context) +
                    " cannot initialize from empty cell connectivity");
            }

            const auto cell_dofs =
                field_dofs.getCellDofs(static_cast<FE::GlobalIndex>(cell));
            if (cell_dofs.size() != n_cell_vertices * components) {
                throw std::runtime_error(
                    "IncompressibleNavierStokesVMSModule: " + std::string(context) +
                    " cell DOF count does not match mesh point field connectivity");
            }

            for (std::size_t local_node = 0; local_node < n_cell_vertices; ++local_node) {
                const auto vertex = cell_vertices[local_node];
                if (vertex < 0 ||
                    static_cast<std::size_t>(vertex) >= static_cast<std::size_t>(n_vertices)) {
                    throw std::runtime_error(
                        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
                        " found an out-of-range mesh vertex");
                }
                const auto v_base = static_cast<std::size_t>(vertex) * mesh_components;
                for (std::size_t c = 0; c < components; ++c) {
                    const auto cell_dof_position = c * n_cell_vertices + local_node;
                    const auto dof = cell_dofs[cell_dof_position];
                    if (dof < 0 || static_cast<std::size_t>(dof) >= n_field_dofs) {
                        throw std::runtime_error(
                            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
                            " found an out-of-range field DOF");
                    }
                    const auto sdof = static_cast<std::size_t>(dof);
                    if (coefficient_written[sdof] != 0u) {
                        continue;
                    }
                    dofs.push_back(field_offset + dof);
                    values.push_back(mesh_values[v_base + c]);
                    coefficient_written[sdof] = 1u;
                }
            }
        }
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

[[nodiscard]] FE::Real hydrostaticPressureAt(
    const std::array<FE::Real, 3>& x,
    const IncompressibleNavierStokesVMSOptions& options,
    const IncompressibleNavierStokesVMSOptions::
        HydrostaticPressureInitialization& init) noexcept
{
    FE::Real pressure = init.reference_pressure;
    for (std::size_t d = 0; d < options.body_force.size(); ++d) {
        pressure += options.density * options.body_force[d] *
                    (x[d] - init.reference_point[d]);
    }
    return pressure;
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

void validateActiveDomainPressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    if (options.node_pressure_constraints.values.empty()) {
        return;
    }

    const auto active_pressure_domain = activePressureDomainFor(free_surfaces);
    if (!active_pressure_domain.has_value()) {
        return;
    }

    const auto n_vertices = system.meshAccess().numVertices();
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

} // namespace

void IncompressibleNavierStokesVMSModule::applyInitialConditions(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0) const
{
    const auto& init = options_.hydrostatic_pressure_initialization;

    std::size_t mesh_field_initialization_dofs = 0u;
    mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
        system,
        u0,
        system.findFieldByName(options_.velocity_field_name),
        "mesh-field velocity initialization");

    if (!init.enabled) {
        mesh_field_initialization_dofs += initializeStateFieldFromMeshVertexField(
            system,
            u0,
            system.findFieldByName(options_.pressure_field_name),
            "mesh-field pressure initialization");
        if (mesh_field_initialization_dofs > 0u) {
            std::ostringstream oss;
            oss << "IncompressibleNavierStokesVMSModule: mesh-field "
                   "initialization diagnostic=mesh_field_initialization"
                << " initialized_dofs=" << mesh_field_initialization_dofs
                << " velocity_field='" << options_.velocity_field_name << "'"
                << " pressure_field='" << options_.pressure_field_name << "'";
            FE_LOG_INFO(oss.str());
        }
        return;
    }

    if (mesh_field_initialization_dofs > 0u) {
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: mesh-field "
               "initialization diagnostic=mesh_field_initialization"
            << " initialized_dofs=" << mesh_field_initialization_dofs
            << " velocity_field='" << options_.velocity_field_name << "'";
        FE_LOG_INFO(oss.str());
    }

    const auto p_id = system.findFieldByName(options_.pressure_field_name);
    if (p_id == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not find pressure field '" +
            options_.pressure_field_name + "'");
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
        activePressureDomainFor(options_.free_surface);
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
                   : hydrostaticPressureAt(x, options_, init))
            : init.reference_pressure;
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

    for (const auto& constraint : options_.node_pressure_constraints.values) {
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options_.node_pressure_constraints.id_type,
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
                   : hydrostaticPressureAt(x, options_, init))
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
            << options_.node_pressure_constraints.values.size()
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

namespace {

[[nodiscard]] bool isUnfittedLevelSet(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.implementation == FreeSurfaceImplementation::UnfittedLevelSet;
}

[[nodiscard]] int freeSurfaceMarker(const FreeSurfaceBoundary& bc)
{
    return isUnfittedLevelSet(bc) ? bc.interface_marker : bc.boundary_marker;
}

[[nodiscard]] bool useFittedCurrentGeometry(const FreeSurfaceBoundary& bc,
                                            bool ale_enabled) noexcept
{
    return ale_enabled && !isUnfittedLevelSet(bc);
}

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc)
{
    const char* kind = isUnfittedLevelSet(bc) ? "_i" : "_b";
    return std::string(prefix) + kind + std::to_string(freeSurfaceMarker(bc));
}

[[nodiscard]] int contactLineConstraintMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    return contact_line.contact_line_marker >= 0
               ? contact_line.contact_line_marker
               : contact_line.wall_boundary_marker;
}

[[nodiscard]] FE::FieldId resolveLevelSetFieldId(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto phi_id = system.findFieldByName(bc.level_set_field_name);
    if (phi_id == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free surface references unknown level-set field '" +
            bc.level_set_field_name + "'");
    }

    const auto& rec = system.fieldRecord(phi_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: level-set field '" +
            bc.level_set_field_name + "' must be scalar");
    }
    return phi_id;
}

[[nodiscard]] int generatedInterfaceMarkerFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker >= 0) {
        return bc.interface_marker;
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted free surface requires a non-empty generated_interface_domain_id");
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi_id);
    key.domain_id = bc.generated_interface_domain_id;
    key.isovalue = bc.level_set_isovalue;
    key.requested_marker = bc.interface_marker;
    return FE::interfaces::stableGeneratedInterfaceMarker(key);
}

[[nodiscard]] FreeSurfaceBoundary withResolvedInterfaceMarker(
    FreeSurfaceBoundary bc,
    const FE::systems::FESystem& system)
{
    if (isUnfittedLevelSet(bc) && bc.interface_marker < 0) {
        bc.interface_marker = generatedInterfaceMarkerFor(bc, system);
    }
    return bc;
}

[[nodiscard]] FE::Real constantScalarValueOrThrow(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " currently requires a literal scalar value");
    }
    return *real;
}

[[nodiscard]] std::array<FE::Real, 3> normalizedWallNormal(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    std::array<FE::Real, 3> normal{
        constantScalarValueOrThrow(contact_line.wall_normal[0], "contact-line wall_normal"),
        constantScalarValueOrThrow(contact_line.wall_normal[1], "contact-line wall_normal"),
        constantScalarValueOrThrow(contact_line.wall_normal[2], "contact-line wall_normal")};
    const auto norm = std::sqrt(normal[0] * normal[0] +
                                normal[1] * normal[1] +
                                normal[2] * normal[2]);
    if (!(norm > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires a nonzero wall_normal");
    }
    for (auto& component : normal) {
        component /= norm;
    }
    return normal;
}

[[nodiscard]] FE::forms::FormExpr wallNormalExpression(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto wall_normal = normalizedWallNormal(contact_line);
    std::vector<FE::forms::FormExpr> wall_components;
    wall_components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        wall_components.push_back(FE::forms::FormExpr::constant(
            wall_normal[static_cast<std::size_t>(d)]));
    }
    return FE::forms::FormExpr::asVector(std::move(wall_components));
}

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc, bool ale_enabled)
{
    if (isUnfittedLevelSet(bc)) {
        if (bc.interface_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires interface_marker >= 0");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires a non-empty level_set_field_name");
        }
        if (bc.active_domain == FreeSurfaceActiveDomain::None &&
            !bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: UnfittedLevelSet free surfaces require Active_domain=LevelSetNegative or LevelSetPositive; set Allow_full_domain_unfitted_free_surface=true only for deliberate full-domain diagnostic runs");
        }
        if (bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Nitsche) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces are one-sided embedded boundaries; Nitsche free-surface kinematics require the fitted ALE path or a future two-sided CutFEM interface path");
        }
        if (!FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension) &&
            bc.use_level_set_curvature &&
            bc.curvature_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set surface tension with raw level-set curvature is not validated; set Use_level_set_curvature=false and provide Curvature or a projected curvature field");
        }
        const auto& cut = bc.cut_cell_stabilization;
        if (cut.enabled) {
            const auto* velocity_penalty =
                std::get_if<FE::Real>(&cut.velocity_gradient_penalty);
            if (velocity_penalty && *velocity_penalty < FE::Real{0.0}) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: cut-cell velocity-gradient penalty must be nonnegative");
            }
            const auto* pressure_penalty =
                std::get_if<FE::Real>(&cut.pressure_gradient_penalty);
            if (pressure_penalty && *pressure_penalty < FE::Real{0.0}) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: cut-cell pressure-gradient penalty must be nonnegative");
            }
            if (FE::forms::bc::isZeroConstantScalarValue(cut.velocity_gradient_penalty) &&
                FE::forms::bc::isZeroConstantScalarValue(cut.pressure_gradient_penalty)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: enabled cut-cell stabilization requires a nonzero penalty");
            }
        }
        const auto& extension = bc.velocity_extension;
        if (extension.enabled) {
            const auto* diffusivity =
                std::get_if<FE::Real>(&extension.diffusivity);
            if (diffusivity && *diffusivity < FE::Real{0.0}) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: velocity-extension diffusivity must be nonnegative");
            }
            if (FE::forms::bc::isZeroConstantScalarValue(extension.diffusivity)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: enabled velocity extension requires a nonzero diffusivity");
            }
            if (bc.active_domain == FreeSurfaceActiveDomain::None) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: velocity extension requires an active-domain unfitted free surface");
            }
        }
    } else {
        if (bc.boundary_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free surface requires boundary_marker >= 0");
        }
        if (bc.cut_cell_stabilization.enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: cut-cell stabilization is only valid for unfitted level-set free surfaces");
        }
        if (bc.active_domain != FreeSurfaceActiveDomain::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: active-domain free-surface volume integration is only valid for unfitted level-set free surfaces");
        }
        if (bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Allow_full_domain_unfitted_free_surface is only valid for unfitted level-set free surfaces");
        }
        if (bc.velocity_extension.enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: velocity extension is only valid for unfitted level-set free surfaces");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
            !ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free-surface kinematics require ALE to be enabled");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
            bc.normal_kinematic_policy == FreeSurfaceNormalKinematicPolicy::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted free-surface kinematic enforcement requires a normal kinematic policy");
        }
    }

    if (bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty &&
        FE::forms::bc::isZeroConstantScalarValue(bc.kinematic_penalty)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: penalty free-surface kinematics require a nonzero kinematic_penalty");
    }

    for (const auto& contact_line : bc.contact_lines) {
        if (contact_line.model == FreeSurfaceContactLineModel::Pinned) {
            if (isUnfittedLevelSet(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact lines are currently supported only for fitted ALE free surfaces");
            }
            if (!ale_enabled) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require ALE to be enabled");
            }
            if (contactLineConstraintMarker(contact_line) < 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact line requires contact_line_marker or wall_boundary_marker >= 0");
            }
        }
        if (contact_line.model == FreeSurfaceContactLineModel::PrescribedContactAngle) {
            if (!isUnfittedLevelSet(bc) && !ale_enabled) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles require ALE to be enabled");
            }
        }
    }
}

void warnUnfittedRawCurvatureIfNeeded(const FreeSurfaceBoundary& bc)
{
    if (!isUnfittedLevelSet(bc) ||
        FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension) ||
        !bc.use_level_set_curvature ||
        !bc.curvature_field_name.empty()) {
        return;
    }

    FE_LOG_WARNING(
        std::string("IncompressibleNavierStokesVMSModule: unfitted level-set surface tension is using raw level-set curvature") +
        " marker=" + std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " generated_interface_domain_id='" + bc.generated_interface_domain_id + "'" +
        " diagnostic=unfitted_level_set_raw_curvature"
        " recommendation=use zero surface tension, prescribed curvature, or projected/smoothed curvature for verification cases");
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceNormal(
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& phi)
{
    auto n = FE::forms::unitNormalFromLevelSet(phi);
    if (bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive) {
        return -n;
    }
    return n;
}

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;
[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept;
[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept;
[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept;

void applyFreeSurfaceCutCellStabilization(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    FE::Real stabilization_epsilon,
    int velocity_components,
    int velocity_polynomial_order,
    int pressure_polynomial_order)
{
    if (!isUnfittedLevelSet(bc) || !bc.cut_cell_stabilization.enabled) {
        return;
    }

    namespace bc_forms = FE::forms::bc;
    const auto& cut = bc.cut_cell_stabilization;
    constexpr int supported_derivative_order = 2;
    const auto derivative_order_label = [](int max_order) -> const char* {
        return max_order > 1 ? "1,2" : "1";
    };
    const int velocity_derivative_order =
        velocity_polynomial_order > 1 ? supported_derivative_order : 1;
    const int pressure_derivative_order =
        pressure_polynomial_order > 1 ? supported_derivative_order : 1;
    const int max_derivative_order =
        velocity_derivative_order > pressure_derivative_order
            ? velocity_derivative_order
            : pressure_derivative_order;
    const bool has_unsupported_derivative_order =
        velocity_polynomial_order > supported_derivative_order ||
        pressure_polynomial_order > supported_derivative_order;
    const auto cut_scale = cut.use_cut_metadata_scale
        ? FE::forms::cutStabilizationScale()
        : FE::forms::FormExpr::constant(1.0);
    const auto h_f = FE::forms::avg(FE::forms::hNormal());
    const auto h3 = h_f * h_f * h_f;
    const auto h5 = h3 * h_f * h_f;
    const auto interface_side =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
            ? "Plus"
            : (bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                   ? "Minus"
                   : "All");
    const auto active_domain_side =
        bc.active_domain == FreeSurfaceActiveDomain::None
            ? "FullDomain"
            : cutVolumeSideName(activeDomainSide(bc.active_domain));

    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: cut-cell stabilization "
        << "marker=" << bc.interface_marker
        << " level_set_field='" << bc.level_set_field_name << "'"
        << " interface_side=" << interface_side
        << " active_domain=" << activeDomainName(bc.active_domain)
        << " active_domain_side=" << active_domain_side
        << " Active_domain_method="
        << activeDomainMethodName(bc.active_domain_method)
        << " use_cut_metadata_scale="
        << (cut.use_cut_metadata_scale ? "true" : "false")
        << " facet_scope=cut-adjacent"
        << " velocity_polynomial_order=" << velocity_polynomial_order
        << " pressure_polynomial_order=" << pressure_polynomial_order
        << " derivative_orders=" << derivative_order_label(max_derivative_order)
        << " velocity_derivative_orders="
        << derivative_order_label(velocity_derivative_order)
        << " pressure_derivative_orders="
        << derivative_order_label(pressure_derivative_order)
        << " velocity_scaling="
        << (velocity_derivative_order > 1 ? "h,h^3" : "h")
        << " pressure_scaling="
        << (pressure_derivative_order > 1 ? "h^3/mu,h^5/mu" : "h^3/mu");
    FE_LOG_INFO(oss.str());

    if (has_unsupported_derivative_order) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: high-order cut-cell "
            "stabilization currently supports derivative_orders=1,2; "
            "higher-normal-derivative penalties above derivative_order=" +
            std::to_string(supported_derivative_order) +
            " are not yet available");
    }

    if (!bc_forms::isZeroConstantScalarValue(cut.velocity_gradient_penalty)) {
        const auto velocity_penalty = bc_forms::toScalarExpr(
            cut.velocity_gradient_penalty,
            freeSurfaceValueName("ns_free_surface_cut_velocity_penalty", bc));
        auto velocity_jump_term = FE::forms::FormExpr::constant(0.0);
        for (int component = 0; component < velocity_components; ++component) {
            const auto velocity_jump_u =
                FE::forms::cutAdjacentFacetGradientJump(FE::forms::component(u, component));
            const auto velocity_jump_v =
                FE::forms::cutAdjacentFacetGradientJump(FE::forms::component(v, component));
            velocity_jump_term =
                velocity_jump_term + FE::forms::inner(velocity_jump_u, velocity_jump_v);
        }
        momentum_form =
            momentum_form +
            FE::forms::cutAdjacentFacetIntegral(
                cut_scale * velocity_penalty * mu * h_f *
                    velocity_jump_term,
                bc.interface_marker);

        if (velocity_derivative_order > 1) {
            auto velocity_second_normal_jump_term =
                FE::forms::FormExpr::constant(0.0);
            for (int component = 0; component < velocity_components; ++component) {
                const auto velocity_jump_u =
                    FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(
                        FE::forms::component(u, component));
                const auto velocity_jump_v =
                    FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(
                        FE::forms::component(v, component));
                velocity_second_normal_jump_term =
                    velocity_second_normal_jump_term + velocity_jump_u * velocity_jump_v;
            }
            momentum_form =
                momentum_form +
                FE::forms::cutAdjacentFacetIntegral(
                    cut_scale * velocity_penalty * mu * h3 *
                        velocity_second_normal_jump_term,
                    bc.interface_marker);
        }
    }

    if (!bc_forms::isZeroConstantScalarValue(cut.pressure_gradient_penalty)) {
        const auto pressure_penalty = bc_forms::toScalarExpr(
            cut.pressure_gradient_penalty,
            freeSurfaceValueName("ns_free_surface_cut_pressure_penalty", bc));
        const auto pressure_jump_p =
            FE::forms::cutAdjacentFacetGradientJump(p);
        const auto pressure_jump_q =
            FE::forms::cutAdjacentFacetGradientJump(q);
        continuity_form =
            continuity_form +
            FE::forms::cutAdjacentFacetIntegral(
                cut_scale * pressure_penalty * h3 /
                (mu + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                    FE::forms::inner(pressure_jump_p, pressure_jump_q),
                bc.interface_marker);

        if (pressure_derivative_order > 1) {
            const auto pressure_second_jump_p =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(p);
            const auto pressure_second_jump_q =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(q);
            continuity_form =
                continuity_form +
                FE::forms::cutAdjacentFacetIntegral(
                    cut_scale * pressure_penalty * h5 /
                    (mu + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                        pressure_second_jump_p * pressure_second_jump_q,
                    bc.interface_marker);
        }
    }
}

[[nodiscard]] FE::forms::FormExpr integrateOnFreeSurface(
    const FE::forms::FormExpr& integrand,
    const FreeSurfaceBoundary& bc,
    bool ale_enabled)
{
    if (isUnfittedLevelSet(bc)) {
        return integrand.dI(bc.interface_marker);
    }
    const auto weighted_integrand =
        useFittedCurrentGeometry(bc, ale_enabled)
            ? integrand * FE::forms::currentMeasure()
            : integrand;
    return weighted_integrand.ds(bc.boundary_marker);
}

struct ActiveVolumeDomain {
    int interface_marker{-1};
    FE::forms::CutVolumeSide side{FE::forms::CutVolumeSide::Negative};
    FreeSurfaceActiveDomainMethod method{FreeSurfaceActiveDomainMethod::CutVolume};
    FE::forms::FormExpr indicator{};
    FE::forms::FormExpr cut_volume_shape_tangent_factor{};
};

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept
{
    switch (method) {
    case FreeSurfaceActiveDomainMethod::CutVolume:
        return "CutVolume";
    case FreeSurfaceActiveDomainMethod::SmoothedIndicator:
        return "SmoothedIndicator";
    }
    return "Unknown";
}

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept
{
    switch (enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return "None";
    case FreeSurfaceKinematicEnforcement::Penalty:
        return "Penalty";
    case FreeSurfaceKinematicEnforcement::Nitsche:
        return "Nitsche";
    }
    return "Unknown";
}

[[nodiscard]] const char* fieldSourceKindName(
    FE::systems::FieldSourceKind source_kind) noexcept
{
    switch (source_kind) {
    case FE::systems::FieldSourceKind::Unknown:
        return "Unknown";
    case FE::systems::FieldSourceKind::PrescribedData:
        return "PrescribedData";
    case FE::systems::FieldSourceKind::DerivedFromUnknown:
        return "DerivedFromUnknown";
    }
    return "Unknown";
}

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::forms::CutVolumeSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::forms::CutVolumeSide::Positive;
    }
    return FE::forms::CutVolumeSide::Negative;
}

[[nodiscard]] FE::forms::CutVolumeSide oppositeCutVolumeSide(
    FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative
               ? FE::forms::CutVolumeSide::Positive
               : FE::forms::CutVolumeSide::Negative;
}

[[nodiscard]] FE::forms::FormExpr cutVolumeLevelSetShapeTangentFactor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    FE::forms::CutVolumeSide side,
    std::string_view domain_role)
{
    if (!isUnfittedLevelSet(bc) ||
        bc.active_domain == FreeSurfaceActiveDomain::None ||
        bc.active_domain_method != FreeSurfaceActiveDomainMethod::CutVolume) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (rec.source_kind == FE::systems::FieldSourceKind::PrescribedData) {
        return FE::forms::FormExpr{};
    }
    if (!system.fieldParticipatesInUnknownVector(phi_id)) {
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: unfitted cut-volume active domain "
            << "references a non-prescribed level-set field that is not an unknown; "
            << "no level-set domain tangent will be assembled "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << rec.name << "'"
            << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " side=" << cutVolumeSideName(side)
            << " domain_role=" << domain_role
            << " diagnostic=unfitted_free_surface_cut_volume_phi_tangent_unavailable";
        FE_LOG_WARNING(oss.str());
        return FE::forms::FormExpr{};
    }

    const auto sign = side == FE::forms::CutVolumeSide::Negative
        ? FE::Real{-1.0}
        : FE::Real{1.0};
    const auto phi =
        FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    const auto dphi =
        FE::forms::FormExpr::trialFunction(*rec.space, "d" + rec.name);
    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));

    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: adding unfitted cut-volume "
        << "level-set Hadamard shape tangent factor "
        << "marker=" << bc.interface_marker
        << " level_set_field='" << rec.name << "'"
        << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
        << " Active_domain=" << activeDomainName(bc.active_domain)
        << " Active_domain_method="
        << activeDomainMethodName(bc.active_domain_method)
        << " side=" << cutVolumeSideName(side)
        << " domain_role=" << domain_role
        << " sign=" << sign
        << " diagnostic=unfitted_free_surface_cut_volume_phi_shape_tangent";
    FE_LOG_INFO(oss.str());

    return FE::forms::FormExpr::constant(sign) * dphi / grad_norm;
}

[[nodiscard]] FE::forms::FormExpr activeDomainIndicatorFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto signed_phi =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
            ? FE::forms::FormExpr::constant(bc.level_set_isovalue) - phi
            : phi - FE::forms::FormExpr::constant(bc.level_set_isovalue);
    const auto width = bc.active_domain_smoothing_width > FE::Real{0.0}
        ? FE::forms::FormExpr::constant(bc.active_domain_smoothing_width)
        : FE::forms::h();
    return FE::forms::smoothHeaviside(signed_phi, width);
}

[[nodiscard]] std::optional<ActiveVolumeDomain> activeVolumeDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system)
{
    std::optional<ActiveVolumeDomain> active_domain;
    for (const auto& bc : free_surfaces) {
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        if (active_domain.has_value()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: at most one active-domain free surface may restrict Navier-Stokes volume integration");
        }

        const FE::forms::CutVolumeSide side = activeDomainSide(bc.active_domain);
        FE::forms::FormExpr indicator{};
        if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
            indicator = activeDomainIndicatorFor(bc, system);
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: Active_domain_method=SmoothedIndicator is diagnostic and not a final benchmark acceptance path");
        }
        active_domain = ActiveVolumeDomain{
            bc.interface_marker,
            side,
            bc.active_domain_method,
            indicator,
            cutVolumeLevelSetShapeTangentFactor(
                bc, system, side, std::string_view("active"))};

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: active-domain free surface "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << bc.level_set_field_name << "'"
            << " isovalue=" << bc.level_set_isovalue
            << " generated_interface_domain_id='"
            << bc.generated_interface_domain_id << "'"
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " side=" << cutVolumeSideName(side);
        if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
            oss << " smoothing_width="
                << (bc.active_domain_smoothing_width > FE::Real{0.0}
                        ? std::to_string(bc.active_domain_smoothing_width)
                        : std::string("cell_diameter"));
        }
        FE_LOG_INFO(oss.str());
    }
    return active_domain;
}

[[nodiscard]] FE::forms::FormExpr integrateOnActiveVolume(
    const FE::forms::FormExpr& integrand,
    const std::optional<ActiveVolumeDomain>& active_domain)
{
    if (!active_domain.has_value()) {
        return integrand.dx();
    }
    if (active_domain->method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
        return (active_domain->indicator * integrand).dx();
    }
    auto out = integrand.dCutVolume(active_domain->interface_marker,
                                    active_domain->side);
    return out;
}

[[nodiscard]] ActiveVolumeDomain inactiveVolumeDomainFor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    const auto active_side = activeDomainSide(bc.active_domain);
    FE::forms::FormExpr indicator{};
    if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
        indicator = FE::forms::FormExpr::constant(1.0) -
                    activeDomainIndicatorFor(bc, system);
    }
    return ActiveVolumeDomain{
        bc.interface_marker,
        oppositeCutVolumeSide(active_side),
        bc.active_domain_method,
        indicator,
        cutVolumeLevelSetShapeTangentFactor(
            bc,
            system,
            oppositeCutVolumeSide(active_side),
            std::string_view("inactive"))};
}

void appendCutVolumeShapeTangentForm(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& volume_integrand,
    const std::optional<ActiveVolumeDomain>& domain)
{
    if (!domain.has_value() ||
        domain->method != FreeSurfaceActiveDomainMethod::CutVolume ||
        !domain->cut_volume_shape_tangent_factor.isValid()) {
        return;
    }

    auto term =
        (domain->cut_volume_shape_tangent_factor * volume_integrand)
            .dI(domain->interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;
}

void appendCutVolumeShapeTangentForm(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& volume_integrand,
    const ActiveVolumeDomain& domain)
{
    appendCutVolumeShapeTangentForm(
        shape_tangent_form,
        volume_integrand,
        std::optional<ActiveVolumeDomain>(domain));
}

void applyFreeSurfaceVelocityExtension(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& cut_volume_shape_tangent_form,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& v)
{
    if (!isUnfittedLevelSet(bc) || !bc.velocity_extension.enabled) {
        return;
    }

    namespace bc_forms = FE::forms::bc;
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: unfitted free-surface velocity extension marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " active_domain=" + activeDomainName(bc.active_domain) +
        " active_domain_method=" + activeDomainMethodName(bc.active_domain_method));
    const auto diffusivity = bc_forms::toScalarExpr(
        bc.velocity_extension.diffusivity,
        freeSurfaceValueName("ns_free_surface_velocity_extension_diffusivity", bc));
    const auto inactive_domain = inactiveVolumeDomainFor(bc, system);
    const auto extension_integrand =
        diffusivity * FE::forms::inner(FE::forms::grad(u),
                                       FE::forms::grad(v));
    momentum_form =
        momentum_form +
        integrateOnActiveVolume(extension_integrand, inactive_domain);
    appendCutVolumeShapeTangentForm(
        cut_volume_shape_tangent_form,
        extension_integrand,
        inactive_domain);
}

[[nodiscard]] FE::forms::FormExpr freeSurfaceLevelSet(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    }
    return FE::forms::FormExpr::discreteField(phi_id, *rec.space, bc.level_set_field_name);
}

[[nodiscard]] FE::forms::FormExpr freeSurfaceCurvatureField(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (bc.curvature_field_name.empty()) {
        return FE::forms::FormExpr{};
    }

    const auto kappa_id = system.findFieldByName(bc.curvature_field_name);
    if (kappa_id == FE::INVALID_FIELD_ID || !system.hasField(bc.curvature_field_name)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free surface references unknown curvature field '" +
            bc.curvature_field_name + "'");
    }

    const auto& rec = system.fieldRecord(kappa_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: curvature field '" +
            bc.curvature_field_name + "' must be scalar");
    }

    if (system.fieldParticipatesInUnknownVector(kappa_id)) {
        return FE::forms::StateField(kappa_id, *rec.space, bc.curvature_field_name);
    }
    return FE::forms::FormExpr::discreteField(
        kappa_id, *rec.space, bc.curvature_field_name);
}

[[nodiscard]] FE::forms::FormExpr unfittedLevelSetNormalSpeedFactor(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return FE::forms::FormExpr{};
    }

    const auto phi_id = resolveLevelSetFieldId(bc, system);
    const auto& rec = system.fieldRecord(phi_id);
    if (rec.source_kind == FE::systems::FieldSourceKind::PrescribedData ||
        !system.fieldParticipatesInUnknownVector(phi_id)) {
        return FE::forms::FormExpr{};
    }
    if (!rec.space || rec.components != 1 || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unfitted free-surface shape tangent requires a scalar level-set field space");
    }

    const auto phi =
        FE::forms::StateField(phi_id, *rec.space, bc.level_set_field_name);
    const auto dphi =
        FE::forms::FormExpr::trialFunction(*rec.space, "d" + rec.name);
    const auto grad_phi = FE::forms::grad(phi);
    const auto grad_norm =
        FE::forms::sqrt(FE::forms::inner(grad_phi, grad_phi) +
                        FE::forms::FormExpr::constant(FE::Real{1.0e-30}));
    return FE::forms::FormExpr::constant(FE::Real{-1.0}) * dphi / grad_norm;
}

[[nodiscard]] FE::forms::FormExpr unfittedInterfaceMeasureCurvature(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi)
{
    if (!bc.curvature_field_name.empty()) {
        return freeSurfaceCurvatureField(bc, system);
    }
    return FE::forms::meanCurvatureFromLevelSet(phi);
}

void appendUnfittedInterfaceMeasureShapeTangent(
    FE::forms::FormExpr& shape_tangent_form,
    const FE::forms::FormExpr& residual_integrand,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || !residual_integrand.isValid()) {
        return;
    }

    const auto normal_speed = unfittedLevelSetNormalSpeedFactor(bc, system);
    if (!normal_speed.isValid()) {
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto curvature = unfittedInterfaceMeasureCurvature(bc, system, phi);
    const auto term =
        (curvature * normal_speed * residual_integrand)
            .dI(bc.interface_marker);
    shape_tangent_form =
        shape_tangent_form.isValid() ? shape_tangent_form + term : term;

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: adding unfitted free-surface interface measure shape tangent marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " curvature_source=" +
        (bc.curvature_field_name.empty() ? "level_set_geometry" : "curvature_field") +
        " diagnostic=unfitted_free_surface_interface_measure_shape_tangent");
}

[[nodiscard]] bool appendUniqueExtraTrialField(
    FE::systems::FormInstallOptions& install,
    FE::FieldId field)
{
    if (field == FE::INVALID_FIELD_ID) {
        return false;
    }
    if (std::find(install.extra_trial_fields.begin(),
                  install.extra_trial_fields.end(),
                  field) != install.extra_trial_fields.end()) {
        return false;
    }
    install.extra_trial_fields.push_back(field);
    return true;
}

[[nodiscard]] bool unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(
    const FreeSurfaceBoundary& bc) noexcept
{
    if (!isUnfittedLevelSet(bc)) {
        return false;
    }
    if (bc.active_domain != FreeSurfaceActiveDomain::None &&
        bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
        return true;
    }
    const bool has_dynamic_interface_stress =
        !FE::forms::bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
    if (has_dynamic_interface_stress) {
        return true;
    }
    return bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty;
}

void appendUnfittedFreeSurfaceLevelSetTrialFields(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    const FE::systems::FESystem& system,
    FE::systems::FormInstallOptions& install)
{
    for (const auto& bc : free_surfaces) {
        if (!unfittedFreeSurfaceNeedsLevelSetTrialFieldForNavierStokes(bc)) {
            if (isUnfittedLevelSet(bc)) {
                const char* cut_volume_shape_tangent =
                    "not_applicable";
                if (bc.active_domain != FreeSurfaceActiveDomain::None &&
                    bc.active_domain_method ==
                        FreeSurfaceActiveDomainMethod::CutVolume) {
                    const auto phi_id = resolveLevelSetFieldId(bc, system);
                    const auto& rec = system.fieldRecord(phi_id);
                    cut_volume_shape_tangent =
                        system.fieldParticipatesInUnknownVector(phi_id) &&
                                rec.source_kind !=
                                    FE::systems::FieldSourceKind::PrescribedData
                            ? "matrix_only_hadamard"
                            : "not_installed_non_unknown_or_prescribed_level_set";
                }
                std::ostringstream oss;
                oss << "IncompressibleNavierStokesVMSModule: not adding "
                    << "unfitted free-surface level-set field as Navier-Stokes "
                    << "residual extra trial "
                    << "marker=" << bc.interface_marker
                    << " level_set_field='" << bc.level_set_field_name << "'"
                    << " Active_domain=" << activeDomainName(bc.active_domain)
                    << " Active_domain_method="
                    << activeDomainMethodName(bc.active_domain_method)
                    << " diagnostic=unfitted_free_surface_phi_extra_trial_omitted"
                    << " cut_volume_phi_shape_tangent="
                    << cut_volume_shape_tangent
                    << " reason=no_explicit_residual_level_set_dependence";
                FE_LOG_INFO(oss.str());
            }
            continue;
        }
        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& rec = system.fieldRecord(phi_id);
        if (!system.fieldParticipatesInUnknownVector(phi_id)) {
            continue;
        }
        if (!appendUniqueExtraTrialField(install, phi_id)) {
            continue;
        }

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: adding unfitted free-surface "
            << "level-set field as Navier-Stokes extra trial "
            << "marker=" << bc.interface_marker
            << " level_set_field='" << rec.name << "'"
            << " level_set_source_kind=" << fieldSourceKindName(rec.source_kind)
            << " Active_domain=" << activeDomainName(bc.active_domain)
            << " Active_domain_method="
            << activeDomainMethodName(bc.active_domain_method)
            << " diagnostic=unfitted_free_surface_phi_extra_trial";
        FE_LOG_INFO(oss.str());
    }
}

void applyFreeSurfaceContactLineConstraints(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    int dim)
{
    for (const auto& contact_line : bc.contact_lines) {
        switch (contact_line.model) {
        case FreeSurfaceContactLineModel::None:
        case FreeSurfaceContactLineModel::PrescribedContactAngle:
            continue;
        case FreeSurfaceContactLineModel::Pinned:
            break;
        }

        if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require a coupled mesh displacement unknown");
        }

        const int marker = contactLineConstraintMarker(contact_line);
        std::vector<FE::forms::bc::StrongDirichlet> constraints;
        constraints.reserve(static_cast<std::size_t>(dim));
        for (int component = 0; component < dim; ++component) {
            constraints.push_back(FE::forms::bc::StrongDirichlet{
                .field = ale_binding.mesh_displacement_field,
                .boundary_marker = marker,
                .component = component,
                .value = FE::forms::FormExpr::constant(0.0),
                .symbol = "mesh_displacement",
            });
        }
        FE::systems::installStrongDirichlet(system, constraints);
    }
}

void applyFreeSurfaceContactAngleResidual(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    const IncompressibleNavierStokesVMSOptions& options,
    const FE::systems::FormInstallOptions& base_install_options,
    int dim)
{
    for (const auto& contact_line : bc.contact_lines) {
        if (contact_line.model != FreeSurfaceContactLineModel::PrescribedContactAngle) {
            continue;
        }
        const auto wall_n = wallNormalExpression(contact_line, dim);
        const auto desired = FE::forms::FormExpr::constant(std::cos(
            constantScalarValueOrThrow(
                contact_line.contact_angle_radians,
                "contact-line contact_angle_radians")));
        const auto penalty = FE::forms::bc::toScalarExpr(
            contact_line.contact_angle_penalty,
            freeSurfaceValueName("ns_free_surface_contact_angle_penalty", bc));

        if (isUnfittedLevelSet(bc)) {
            const auto phi_id = system.findFieldByName(bc.level_set_field_name);
            if (phi_id == FE::INVALID_FIELD_ID) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle references unknown level-set field '" +
                    bc.level_set_field_name + "'");
            }
            const auto& rec = system.fieldRecord(phi_id);
            if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: level-set field '" +
                    bc.level_set_field_name + "' must be scalar");
            }
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown");
            }

            const auto phi = FE::forms::StateField(
                phi_id,
                *rec.space,
                bc.level_set_field_name);
            const auto eta = FE::forms::FormExpr::testFunction(
                phi_id,
                *rec.space,
                "eta_contact_angle");
            const auto n = FE::forms::unitNormalFromLevelSet(phi);
            const auto angle_gap = FE::forms::dot(n, wall_n) - desired;
            const auto residual =
                (penalty * angle_gap * eta).dI(bc.interface_marker);

            if (!system.hasOperator("level_set")) {
                system.addOperator("level_set");
            }
            (void)FE::systems::installFormulation(
                system,
                "level_set",
                {phi_id},
                residual,
                base_install_options);
            continue;
        }

        if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles require a coupled mesh displacement unknown");
        }

        const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
        if (!rec.space) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: mesh displacement field for prescribed contact angle has no function space");
        }

        const auto psi = FE::forms::FormExpr::testFunction(
            ale_binding.mesh_displacement_field,
            *rec.space,
            "psi_contact_angle");
        const auto n = FE::forms::currentNormal();
        const auto angle_gap = FE::forms::dot(n, wall_n) - desired;
        const auto residual =
            (penalty * angle_gap * FE::forms::normalTrace(psi, n) *
             FE::forms::currentMeasure()).ds(bc.boundary_marker);

        if (!system.hasOperator("mesh_motion")) {
            system.addOperator("mesh_motion");
        }
        auto install = base_install_options;
        install.compiler_options.geometry_sensitivity.mode =
            FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
        install.compiler_options.geometry_sensitivity.mesh_motion_field =
            ale_binding.mesh_displacement_field;
        install.compiler_options.geometry_tangent_path = options.moving_mesh_tangent_path;
        install.compiler_options.use_symbolic_tangent =
            options.moving_mesh_tangent_path != FE::forms::GeometryTangentPath::ADReference;
        (void)FE::systems::installFormulation(
            system,
            "mesh_motion",
            {ale_binding.mesh_displacement_field},
            residual,
            install);
    }
}

void applyFreeSurfaceBoundary(FE::forms::FormExpr& momentum_form,
                              FE::forms::FormExpr& continuity_form,
                              FE::forms::FormExpr& level_set_shape_tangent_form,
                              const FreeSurfaceBoundary& bc,
                              const FE::systems::FESystem& system,
                              const FE::forms::FormExpr& u,
                              const FE::forms::FormExpr& p,
                              const FE::forms::FormExpr& v,
                              const FE::forms::FormExpr& q,
                              const FE::forms::FormExpr& mesh_velocity,
                              const FE::forms::FormExpr& mu,
                              const IncompressibleNavierStokesVMSOptions& options,
                              bool ale_enabled)
{
    using namespace FE::forms;

    validateFreeSurfaceBoundary(bc, ale_enabled);
    warnUnfittedRawCurvatureIfNeeded(bc);

    const bool has_dynamic_stress =
        !bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !bc::isZeroConstantScalarValue(bc.surface_tension);
    const bool needs_surface_normal =
        has_dynamic_stress ||
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None;
    if (isUnfittedLevelSet(bc)) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: unfitted free-surface boundary mode marker=") +
            std::to_string(bc.interface_marker) +
            " level_set_field='" + bc.level_set_field_name + "'" +
            " generated_interface_domain_id='" + bc.generated_interface_domain_id + "'" +
            " active_domain=" + activeDomainName(bc.active_domain) +
            " active_domain_method=" + activeDomainMethodName(bc.active_domain_method) +
            " dynamic_stress=" + (has_dynamic_stress ? "enabled" : "natural_zero") +
            " kinematic_enforcement=" + kinematicEnforcementName(bc.kinematic_enforcement) +
            " cut_cell_stabilization=" + (bc.cut_cell_stabilization.enabled ? "enabled" : "disabled") +
            " velocity_extension=" + (bc.velocity_extension.enabled ? "enabled" : "disabled"));
    }
    if (!needs_surface_normal) {
        if (isUnfittedLevelSet(bc)) {
            FE_LOG_WARNING(
                std::string("IncompressibleNavierStokesVMSModule: unfitted free surface installs no explicit dI boundary residual marker=") +
                std::to_string(bc.interface_marker) +
                " level_set_field='" + bc.level_set_field_name + "'" +
                " dynamic_stress=natural_zero"
                " kinematic_enforcement=None"
                " diagnostic=unfitted_free_surface_natural_mode");
        }
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto n = isUnfittedLevelSet(bc)
                       ? unfittedInterfaceNormal(bc, phi)
                       : (useFittedCurrentGeometry(bc, ale_enabled)
                              ? currentNormal()
                              : FormExpr::normal());
    const auto p_ext = bc::toScalarExpr(
        bc.external_pressure,
        freeSurfaceValueName("ns_free_surface_external_pressure", bc));
    const auto gamma = bc::toScalarExpr(
        bc.surface_tension,
        freeSurfaceValueName("ns_free_surface_surface_tension", bc));
    const auto curvature = [&]() {
        if (bc::isZeroConstantScalarValue(bc.surface_tension)) {
            return FormExpr::constant(0.0);
        }
        if (!bc.curvature_field_name.empty()) {
            return freeSurfaceCurvatureField(bc, system);
        }
        if (isUnfittedLevelSet(bc) && bc.use_level_set_curvature) {
            return meanCurvatureFromLevelSet(phi);
        }
        if (!isUnfittedLevelSet(bc) && bc.use_current_geometry_curvature) {
            return currentMeanCurvature();
        }
        return bc::toScalarExpr(
            bc.curvature,
            freeSurfaceValueName("ns_free_surface_curvature", bc));
    }();

    if (has_dynamic_stress) {
        const auto traction = (-p_ext + gamma * curvature) * n;
        const auto residual_integrand =
            FE::forms::FormExpr::constant(FE::Real{-1.0}) *
            inner(traction, v);
        momentum_form = momentum_form +
                        integrateOnFreeSurface(residual_integrand, bc, ale_enabled);
        appendUnfittedInterfaceMeasureShapeTangent(
            level_set_shape_tangent_form,
            residual_integrand,
            bc,
            system);
    }

    switch (bc.kinematic_enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return;
    case FreeSurfaceKinematicEnforcement::Penalty: {
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        const auto penalty = bc::toScalarExpr(
            bc.kinematic_penalty,
            freeSurfaceValueName("ns_free_surface_kinematic_penalty", bc));
        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto residual_integrand =
            penalty * normal_mismatch * normalTrace(v, n);
        momentum_form = momentum_form + integrateOnFreeSurface(
            residual_integrand, bc, ale_enabled);
        appendUnfittedInterfaceMeasureShapeTangent(
            level_set_shape_tangent_form,
            residual_integrand,
            bc,
            system);
        return;
    }
    case FreeSurfaceKinematicEnforcement::Nitsche: {
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        if (!(options.nitsche_gamma > 0.0)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics require nitsche_gamma > 0");
        }

        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto v_normal = normalTrace(v, n);
        const auto stress_u = FormExpr::constant(2.0) * mu * sym(grad(u));
        const auto stress_v = FormExpr::constant(2.0) * mu * sym(grad(v));
        const auto normal_stress_u = normalTrace(stress_u * n, n);
        const auto normal_stress_v = normalTrace(stress_v * n, n);
        const auto penalty = bc::buildTraceNitschePenalty(
            mu / hNormal(),
            u,
            bc::TraceNitscheOptions{
                .gamma = options.nitsche_gamma,
                .variant = options.nitsche_symmetric
                    ? bc::NitscheVariant::Symmetric
                    : bc::NitscheVariant::Unsymmetric,
                .scale_with_p = options.nitsche_scale_with_p});

        momentum_form = momentum_form + integrateOnFreeSurface(
            (p - normal_stress_u) * v_normal +
            penalty * normal_mismatch * v_normal,
            bc,
            ale_enabled);
        if (options.nitsche_symmetric) {
            momentum_form = momentum_form - integrateOnFreeSurface(
                normal_stress_v * normal_mismatch, bc, ale_enabled);
            continuity_form = continuity_form + integrateOnFreeSurface(
                q * normal_mismatch, bc, ale_enabled);
        } else {
            momentum_form = momentum_form + integrateOnFreeSurface(
                normal_stress_v * normal_mismatch, bc, ale_enabled);
            continuity_form = continuity_form - integrateOnFreeSurface(
                q * normal_mismatch, bc, ale_enabled);
        }
        return;
    }
    }

    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unsupported free-surface kinematic enforcement");
}

void installFittedFreeSurfaceMeshKinematics(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    const FE::forms::FormExpr& u,
    const IncompressibleNavierStokesVMSOptions& options,
    const FE::systems::FormInstallOptions& base_install_options,
    FE::FieldId velocity_field)
{
    using namespace FE::forms;

    if (bc.implementation != FreeSurfaceImplementation::FittedALE ||
        bc.kinematic_enforcement == FreeSurfaceKinematicEnforcement::None ||
        !ale_binding.coupled()) {
        return;
    }
    if (bc.normal_kinematic_policy !=
        FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
        return;
    }
    if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh kinematics require a coupled mesh displacement unknown");
    }

    const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
    if (!rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh displacement field has no function space");
    }

    const auto psi = TestField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "psi_free_surface_mesh");
    const auto d_mesh = StateField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "d_mesh_free_surface");
    const auto n = currentNormal();
    const auto normal_mismatch = normalTrace(dt(d_mesh) - u, n);
    const auto penalty = [&]() {
        switch (bc.kinematic_enforcement) {
        case FreeSurfaceKinematicEnforcement::Penalty:
            return bc::toScalarExpr(
                bc.kinematic_penalty,
                freeSurfaceValueName("ns_free_surface_mesh_kinematic_penalty", bc));
        case FreeSurfaceKinematicEnforcement::Nitsche:
            return FormExpr::constant(options.nitsche_gamma) / hNormal();
        case FreeSurfaceKinematicEnforcement::None:
            break;
        }
        return FormExpr::constant(0.0);
    }();

    auto residual = integrateOnFreeSurface(
        penalty * normal_mismatch * normalTrace(psi, n),
        bc,
        /*ale_enabled=*/true);

    auto install = base_install_options;
    install.compiler_options.use_symbolic_tangent = true;
    ale_binding.configureInstallOptions(install);
    if (velocity_field != FE::INVALID_FIELD_ID) {
        install.extra_trial_fields.push_back(velocity_field);
    }

    (void)FE::systems::installFormulation(
        system,
        "equations",
        {ale_binding.mesh_displacement_field},
        residual,
        install);
}

} // namespace

void IncompressibleNavierStokesVMSModule::registerOn(FE::systems::FESystem& system) const
{
    if (!velocity_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null velocity_space");
    }
    if (!pressure_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null pressure_space");
    }

    const int dim = velocity_space_->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: velocity space must have 1..3 components");
    }
    if (pressure_space_->value_dimension() != 1) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: pressure space must be scalar");
    }
    if (!(options_.density > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: density must be > 0");
    }
    if (options_.viscosity_model == nullptr && !(options_.viscosity > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: viscosity must be > 0 when viscosity_model is not provided");
    }
    if (options_.enable_vms && !(options_.stabilization_epsilon > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: stabilization_epsilon must be > 0 when VMS is enabled");
    }

    FE::systems::FieldSpec u_spec;
    u_spec.name = options_.velocity_field_name;
    u_spec.space = velocity_space_;
    u_spec.components = dim;
    const FE::FieldId u_id = ensureCompatibleUnknownField(
        system,
        std::move(u_spec),
        "IncompressibleNavierStokesVMSModule::registerOn velocity");

    FE::systems::FieldSpec p_spec;
    p_spec.name = options_.pressure_field_name;
    p_spec.space = pressure_space_;
    p_spec.components = 1;
    const FE::FieldId p_id = ensureCompatibleUnknownField(
        system,
        std::move(p_spec),
        "IncompressibleNavierStokesVMSModule::registerOn pressure");

    FE::FieldId body_force_field_id = FE::INVALID_FIELD_ID;
    if (!options_.body_force_field_name.empty()) {
        FE::systems::FieldSpec source_spec;
        source_spec.name = options_.body_force_field_name;
        source_spec.space = velocity_space_;
        source_spec.components = dim;
        source_spec.source_kind = FE::systems::FieldSourceKind::PrescribedData;
        body_force_field_id = ensureCompatiblePrescribedField(
            system,
            std::move(source_spec),
            options_.auto_register_body_force_field,
            "IncompressibleNavierStokesVMSModule::registerOn momentum source");
    }

    std::vector<FreeSurfaceBoundary> effective_free_surfaces;
    effective_free_surfaces.reserve(options_.free_surface.size());
    for (const auto& bc : options_.free_surface) {
        auto effective_bc = withResolvedInterfaceMarker(bc, system);
        validateFreeSurfaceBoundary(effective_bc, options_.enable_ale);
        effective_free_surfaces.push_back(std::move(effective_bc));
    }
    const auto active_pressure_domain =
        activePressureDomainFor(effective_free_surfaces);
    validateActiveDomainPressureConstraints(
        system,
        options_,
        effective_free_surfaces);

    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume) {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        const bool native_mesh_available = system.mesh() != nullptr;
#else
        const bool native_mesh_available = false;
#endif
        if (!native_mesh_available) {
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: skipping active-domain "
                "inactive velocity/pressure constraints because native mesh support "
                "is unavailable in this FESystem");
        } else {
            const auto side =
                active_pressure_domain->active_domain ==
                        FreeSurfaceActiveDomain::LevelSetPositive
                    ? FE::constraints::LevelSetConstraintSide::Positive
                    : FE::constraints::LevelSetConstraintSide::Negative;
            system.addSystemConstraint(
                std::make_unique<
                    FE::constraints::LevelSetActiveSideVertexDirichletConstraint>(
                    u_id,
                    active_pressure_domain->boundary->level_set_field_name,
                    side,
                    active_pressure_domain->boundary->level_set_isovalue,
                    FE::Real{0.0},
                    active_pressure_domain->boundary->interface_marker));
            system.addSystemConstraint(
                std::make_unique<
                    FE::constraints::LevelSetActiveSideVertexDirichletConstraint>(
                    p_id,
                    active_pressure_domain->boundary->level_set_field_name,
                    side,
                    active_pressure_domain->boundary->level_set_isovalue,
                    FE::Real{0.0},
                    active_pressure_domain->boundary->interface_marker));
        }
    }

    if (!options_.node_pressure_constraints.values.empty()) {
        std::vector<FE::constraints::VertexDirichletValue> values;
        values.reserve(options_.node_pressure_constraints.values.size());
        for (const auto& in : options_.node_pressure_constraints.values) {
            values.push_back(FE::constraints::VertexDirichletValue{in.node_id, in.pressure});
        }

        FE::constraints::VertexIdMode mode = FE::constraints::VertexIdMode::GlobalVertexGid;
        switch (options_.node_pressure_constraints.id_type) {
        case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
            mode = FE::constraints::VertexIdMode::GlobalVertexGid;
            break;
        case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
            mode = FE::constraints::VertexIdMode::LocalVertexId;
            break;
        }

        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(p_id, std::move(values), mode));
    }

    const auto ale_binding = FE::systems::resolveALEBinding(
        system,
        FE::systems::ALEBindingOptions{
            .enabled = options_.enable_ale,
            .dimension = dim,
            .mesh_velocity_source =
                options_.mesh_velocity_source == ALEMeshVelocitySource::CoupledDisplacement
                    ? FE::systems::ALEMeshVelocitySource::CoupledDisplacement
                    : FE::systems::ALEMeshVelocitySource::PrescribedData,
            .geometry_tangent_path = options_.moving_mesh_tangent_path,
            .mesh_velocity_field_name = options_.mesh_velocity_field_name,
            .mesh_displacement_field_name = options_.mesh_displacement_field_name,
            .mesh_velocity_space = options_.mesh_velocity_space ? options_.mesh_velocity_space : velocity_space_,
            .mesh_displacement_space = velocity_space_,
            .auto_register_mesh_velocity_field = options_.auto_register_mesh_velocity_field,
            .auto_register_mesh_displacement_field =
                options_.auto_register_mesh_displacement_field,
        });

    if (!system.hasOperator("equations")) {
        system.addOperator("equations");
    }

    using namespace svmp::FE::forms;

    const auto u = StateField(u_id, *velocity_space_, options_.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");

    const auto active_volume_domain =
        activeVolumeDomainFor(effective_free_surfaces, system);

    const auto rho = FormExpr::constant(options_.density);

    // Body force/source acceleration. The optional field is evaluated as
    // prescribed data, so it contributes to both Galerkin forcing and VMS
    // strong residual without adding unknowns to the system.
    std::vector<FormExpr> f_comp;
    f_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        f_comp.push_back(FormExpr::constant(options_.body_force[static_cast<std::size_t>(d)]));
    }
    FormExpr f = FormExpr::asVector(std::move(f_comp));
    if (options_.has_body_force_spacetime) {
        std::vector<FormExpr> source_comp;
        source_comp.reserve(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            source_comp.push_back(bc::toScalarExpr(
                options_.body_force_spacetime[static_cast<std::size_t>(d)],
                "ns_body_force_spacetime_" + std::to_string(d)));
        }
        f = f + FormExpr::asVector(std::move(source_comp));
    }
    if (body_force_field_id != FE::INVALID_FIELD_ID) {
        f = f + StateField(
                    body_force_field_id,
                    *velocity_space_,
                    options_.body_force_field_name);
    }

    const auto eps_for_mu = sym(grad(u));
    const auto gamma_for_mu =
        sqrt(FormExpr::constant(2.0) * inner(eps_for_mu, eps_for_mu));
    FormExpr mu;
    if (options_.viscosity_model) {
        // Variable viscosity remains a tagged constitutive expression so
        // installFormulation() can publish the law from the residual DAG.
        std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model =
            options_.viscosity_model;
        auto viscosity_metadata = FE::analysis::dynamicViscosityMetadata(
            FE::INVALID_FIELD_ID,
            options_.viscosity,
            options_.viscosity_model);
        viscosity_model = FE::constitutive::withConstitutiveLawMetadata(
            std::move(viscosity_model),
            0u,
            std::move(viscosity_metadata));
        mu = constitutive(std::move(viscosity_model), gamma_for_mu).out(0);
    } else {
        mu = FormExpr::constant(options_.viscosity);
    }

    // ALE uses relative convection u - w_mesh. Static/default paths remain unchanged.
    const auto zero = zeroVector(dim);
    const auto w_mesh = meshVelocity();
    const auto mesh_velocity = options_.enable_ale ? w_mesh : zero;
    const auto a = options_.enable_convection
                       ? (options_.enable_ale ? (u - w_mesh) : u)
                       : zero;
    const bool include_mcv =
        options_.enable_ale && options_.include_moving_control_volume_transient;
    const auto moving_volume_strong =
        include_mcv ? (div(mesh_velocity) * u) : zero;

    // Strong momentum residual (full, including dt(u)):
    //   R_m = rho*(dt(u) + grad(u)*a + chi*div(w_mesh)*u - f)
    //         + grad(p) - div(2 mu sym(grad(u)))
    // with a = u - w_mesh for ALE and chi set by the moving-control-volume option.
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a + moving_volume_strong - f) + grad(p) - div(stress);

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto moving_volume =
        include_mcv ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    const auto galerkin_momentum_integrand =
        inertia + moving_volume + convection + viscous + pressure + forcing;
    const auto galerkin_continuity_integrand = q * div(u);

    FormExpr active_momentum_integrand = galerkin_momentum_integrand;
    FormExpr active_continuity_integrand = galerkin_continuity_integrand;

    if (options_.enable_vms) {
        // Residual-based VMS with static subscales:
        //   u' = -tau_M * R_m
        //   p' = -tau_C * (div u)
        // and coarse-scale stabilization terms assembled from (u', p').
        const auto eps = FormExpr::constant(options_.stabilization_epsilon);
        const auto dt_step = FormExpr::effectiveTimeStep();
        const auto ct_m = FormExpr::constant(options_.ct_m);
        const auto ct_c = FormExpr::constant(options_.ct_c);

        // Element metric tensor Kxi = J^{-T} J^{-1}. FE Forms exposes Jinv()
        // with the active physical dimension, so 2D contractions do not include
        // a dummy frame-thickness component.
        const auto Jinv_expr = Jinv();
        const auto K = transpose(Jinv_expr) * Jinv_expr;
        const auto nu = mu / rho;

        // Legacy-inspired tau_M (stored here as tau_M/rho, matching legacy fluid.cpp naming).
        const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
        const auto kU = inner(a, K * a);
        const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
        const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));

        const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

        const auto u_sub = -tau_m * r_m;
        const auto p_sub = -tau_c * div(u);

        // Advection velocity for convection-related terms (disabled for Stokes).
        const auto u_adv = options_.enable_convection ? (u + u_sub - mesh_velocity) : a;
        const auto p_adv = p + p_sub;

        // Momentum: Galerkin + VMS (SUPG-like) + pressure-subscale (LSIC-like).
        const auto convection_adv = rho * inner(grad(u) * u_adv, v);
        const auto pressure_adv = -p_adv * div(v);
        // Legacy-style full VMS: use the subscale-augmented advection velocity in the
        // test-function stabilization term and include the tauB-based cross-stress closure.
        const auto supg = -rho * inner(grad(v) * u_adv, u_sub);

        // tauB cross-stress closure (legacy fluid.cpp):
        //   tauB = rho / sqrt( u'^T Kxi u' )
        // and adds + (u' · ∇v) · ( tauB * (u' · ∇)u ).
        FormExpr cross_stress = FormExpr::constant(0.0);
        if (options_.enable_convection) {
            const auto tau_b = rho / sqrt(inner(u_sub, K * u_sub) + eps);
            const auto rV_tau = tau_b * (grad(u) * u_sub); // (tauB * (u'·∇)u)
            cross_stress = inner(grad(v) * u_sub, rV_tau);
        }

        active_momentum_integrand =
            inertia + moving_volume + convection_adv + viscous + pressure_adv +
            forcing + supg + cross_stress;

        // Continuity: Galerkin + VMS (PSPG-like).
        active_continuity_integrand = q * div(u) - inner(grad(q), u_sub);
    }

    FormExpr momentum_form =
        integrateOnActiveVolume(active_momentum_integrand, active_volume_domain);
    FormExpr continuity_form =
        integrateOnActiveVolume(active_continuity_integrand, active_volume_domain);
    FormExpr level_set_shape_tangent_form;
    appendCutVolumeShapeTangentForm(
        level_set_shape_tangent_form,
        active_momentum_integrand,
        active_volume_domain);
    appendCutVolumeShapeTangentForm(
        level_set_shape_tangent_form,
        active_continuity_integrand,
        active_volume_domain);

    // ---------------------------------------------------------------------
    // Boundary conditions (installer + factories)
    // ---------------------------------------------------------------------

    if (!options_.coupled_outflow_rcr.empty() || !options_.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, options_.jit_policy);
    }

    FE::systems::BoundaryConditionManager bc_manager;

    // Weak velocity Dirichlet is applied directly to the Forms residual (affects both momentum and continuity).
    // Reserve the marker here so validate() catches conflicts with other BC types.
    bc_manager.install(options_.velocity_dirichlet_weak, Factories::reserveMarker);

    auto free_surface_contact_angle_install = physicsInstallOptions(options_.jit_policy);
    for (const auto& effective_bc : effective_free_surfaces) {
        applyFreeSurfaceContactLineConstraints(system, effective_bc, ale_binding, dim);
        applyFreeSurfaceContactAngleResidual(
            system,
            effective_bc,
            ale_binding,
            options_,
            free_surface_contact_angle_install,
            dim);
        if (effective_bc.implementation == FreeSurfaceImplementation::FittedALE) {
            bc_manager.add(std::make_unique<FE::forms::bc::ReservedBC>(effective_bc.boundary_marker));
        }
        applyFreeSurfaceBoundary(
            momentum_form,
            continuity_form,
            level_set_shape_tangent_form,
            effective_bc,
            system,
            u,
            p,
            v,
            q,
            mesh_velocity,
            mu,
            options_,
            options_.enable_ale);
        applyFreeSurfaceVelocityExtension(
            momentum_form,
            level_set_shape_tangent_form,
            effective_bc,
            system,
            u,
            v);
        installFittedFreeSurfaceMeshKinematics(
            system,
            effective_bc,
            ale_binding,
            u,
            options_,
            free_surface_contact_angle_install,
            u_id);
        applyFreeSurfaceCutCellStabilization(
            momentum_form,
            continuity_form,
            effective_bc,
            u,
            p,
            v,
            q,
            mu,
            options_.stabilization_epsilon,
            dim,
            velocity_space_->polynomial_order(),
            pressure_space_->polynomial_order());
    }

    bc_manager.install(options_.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim); });
    bc_manager.install(options_.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim); });
    bc_manager.install(options_.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho); });
    bc_manager.install(options_.coupled_outflow_rcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(bc, system, u, rho);
    });
    bc_manager.install(options_.coupled_outflow_rcrcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(bc, system, u, rho);
    });
    bc_manager.install(options_.velocity_dirichlet,
                       [&](const auto& bc) {
        return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name);
    });

    bc_manager.applyAll(system, momentum_form, u, v, u_id);

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, options_.pressure_field_name); });
    p_bc_manager.applyAll(system, p_id);

    Factories::applyVelocityNitscheBCs(momentum_form, continuity_form, options_, dim, u, p, v, q, mu);

    // Install the complete residual (momentum + continuity) via the unified
    // installFormulation() entry point.  It auto-detects the two-field mixed
    // structure and sets up per-block Jacobian kernels with optimal assembly.
    const auto residual = momentum_form + continuity_form;

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    if (!options_.viscosity_model) {
        install.recordDynamicViscosity(u_id, options_.viscosity);
    }
    appendUnfittedFreeSurfaceLevelSetTrialFields(
        effective_free_surfaces, system, install);
    ale_binding.configureInstallOptions(install);
    (void)FE::systems::installFormulation(system, "equations", {u_id, p_id}, residual, install);
    if (level_set_shape_tangent_form.isValid()) {
        std::array<FE::FieldId, 2> test_fields{{u_id, p_id}};
        std::vector<FE::FieldId> phi_trial_fields;
        for (const auto& bc : effective_free_surfaces) {
            if (!isUnfittedLevelSet(bc)) {
                continue;
            }
            const auto phi_id = resolveLevelSetFieldId(bc, system);
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                continue;
            }
            if (std::find(phi_trial_fields.begin(),
                          phi_trial_fields.end(),
                          phi_id) == phi_trial_fields.end()) {
                phi_trial_fields.push_back(phi_id);
            }
        }
        if (!phi_trial_fields.empty()) {
            (void)FE::systems::installMixedBilinear(
                system,
                "equations",
                std::span<const FE::FieldId>(test_fields.data(),
                                             test_fields.size()),
                std::span<const FE::FieldId>(phi_trial_fields.data(),
                                             phi_trial_fields.size()),
                level_set_shape_tangent_form,
                install);
        }
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
