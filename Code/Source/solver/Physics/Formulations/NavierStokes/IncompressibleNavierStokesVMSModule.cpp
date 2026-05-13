/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Materials/Fluid/NewtonianViscosity.h"

#include "FE/Assembly/GlobalSystemView.h"
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

#include <cstddef>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
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
    for (const auto& constraint : options.node_pressure_constraints.values) {
        const auto local_vertex = pressureConstraintLocalVertex(
            system,
            options.node_pressure_constraints.id_type,
            constraint.node_id);
        if (!local_vertex.has_value()) {
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
}

} // namespace

void IncompressibleNavierStokesVMSModule::applyInitialConditions(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0) const
{
    const auto& init = options_.hydrostatic_pressure_initialization;
    if (!init.enabled) {
        return;
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

    std::vector<FE::GlobalIndex> dofs;
    std::vector<FE::Real> values;
    std::size_t active_wet_vertices = 0u;
    std::size_t active_dry_vertices = 0u;

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
            initialize_hydrostatic = pressureVertexOnActiveSide(
                phi,
                active_pressure_domain->boundary->level_set_isovalue,
                active_pressure_domain->active_domain);
            if (initialize_hydrostatic) {
                ++active_wet_vertices;
            } else {
                ++active_dry_vertices;
            }
        }
        const FE::Real pressure = initialize_hydrostatic
            ? hydrostaticPressureAt(x, options_, init)
            : init.reference_pressure;

        for (const auto local_dof : vertex_dofs) {
            dofs.push_back(pressure_offset + local_dof);
            values.push_back(pressure);
        }
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
        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: hydrostatic pressure "
            << "initialization Active_domain="
            << pressureActiveDomainName(active_pressure_domain->active_domain)
            << " wet_pressure_vertices=" << active_wet_vertices
            << " dry_pressure_vertices=" << active_dry_vertices
            << " reference_pressure=" << init.reference_pressure;
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
    int velocity_components)
{
    if (!isUnfittedLevelSet(bc) || !bc.cut_cell_stabilization.enabled) {
        return;
    }

    namespace bc_forms = FE::forms::bc;
    const auto& cut = bc.cut_cell_stabilization;
    const auto cut_scale = cut.use_cut_metadata_scale
        ? FE::forms::cutStabilizationScale()
        : FE::forms::FormExpr::constant(1.0);
    const auto h_f = FE::forms::avg(FE::forms::hNormal());

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
                velocity_jump_term);
    }

    if (!bc_forms::isZeroConstantScalarValue(cut.pressure_gradient_penalty)) {
        const auto pressure_penalty = bc_forms::toScalarExpr(
            cut.pressure_gradient_penalty,
            freeSurfaceValueName("ns_free_surface_cut_pressure_penalty", bc));
        const auto h3 = h_f * h_f * h_f;
        const auto pressure_jump_p =
            FE::forms::cutAdjacentFacetGradientJump(p);
        const auto pressure_jump_q =
            FE::forms::cutAdjacentFacetGradientJump(q);
        continuity_form =
            continuity_form +
            FE::forms::cutAdjacentFacetIntegral(
                cut_scale * pressure_penalty * h3 /
                (mu + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                FE::forms::inner(pressure_jump_p, pressure_jump_q));
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

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
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

        FE::forms::CutVolumeSide side = FE::forms::CutVolumeSide::Negative;
        switch (bc.active_domain) {
        case FreeSurfaceActiveDomain::None:
            break;
        case FreeSurfaceActiveDomain::LevelSetNegative:
            side = FE::forms::CutVolumeSide::Negative;
            break;
        case FreeSurfaceActiveDomain::LevelSetPositive:
            side = FE::forms::CutVolumeSide::Positive;
            break;
        }
        FE::forms::FormExpr indicator{};
        if (bc.active_domain_method == FreeSurfaceActiveDomainMethod::SmoothedIndicator) {
            const auto phi = freeSurfaceLevelSet(bc, system);
            const auto signed_phi =
                bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                    ? FE::forms::FormExpr::constant(bc.level_set_isovalue) - phi
                    : phi - FE::forms::FormExpr::constant(bc.level_set_isovalue);
            const auto width = bc.active_domain_smoothing_width > FE::Real{0.0}
                ? FE::forms::FormExpr::constant(bc.active_domain_smoothing_width)
                : FE::forms::h();
            indicator = FE::forms::smoothHeaviside(signed_phi, width);
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: Active_domain_method=SmoothedIndicator is diagnostic and not a final benchmark acceptance path");
        }
        active_domain = ActiveVolumeDomain{
            bc.interface_marker,
            side,
            bc.active_domain_method,
            indicator};

        std::ostringstream oss;
        oss << "IncompressibleNavierStokesVMSModule: active-domain free surface "
            << "marker=" << bc.interface_marker
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
    return integrand.dCutVolume(active_domain->interface_marker,
                                active_domain->side);
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
    return FE::forms::FormExpr::discreteField(phi_id, *rec.space, bc.level_set_field_name);
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

    const bool has_dynamic_stress =
        !bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !bc::isZeroConstantScalarValue(bc.surface_tension);
    const bool needs_surface_normal =
        has_dynamic_stress ||
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None;
    if (!needs_surface_normal) {
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    const auto n = isUnfittedLevelSet(bc)
                       ? unitNormalFromLevelSet(phi)
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
        momentum_form = momentum_form -
                        integrateOnFreeSurface(inner(traction, v), bc, ale_enabled);
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
        momentum_form = momentum_form + integrateOnFreeSurface(
            penalty * normal_mismatch * normalTrace(v, n), bc, ale_enabled);
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

    std::vector<FreeSurfaceBoundary> effective_free_surfaces;
    effective_free_surfaces.reserve(options_.free_surface.size());
    for (const auto& bc : options_.free_surface) {
        auto effective_bc = withResolvedInterfaceMarker(bc, system);
        validateFreeSurfaceBoundary(effective_bc, options_.enable_ale);
        effective_free_surfaces.push_back(std::move(effective_bc));
    }
    validateActiveDomainPressureConstraints(
        system,
        options_,
        effective_free_surfaces);

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

    // Body force (constant vector).
    std::vector<FormExpr> f_comp;
    f_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        f_comp.push_back(FormExpr::constant(options_.body_force[static_cast<std::size_t>(d)]));
    }
    const auto f = FormExpr::asVector(std::move(f_comp));

    // Viscosity is represented as a tagged constitutive expression so
    // installFormulation() can publish the law from the residual DAG.
    const auto eps_for_mu = sym(grad(u));
    const auto gamma_for_mu =
        sqrt(FormExpr::constant(2.0) * inner(eps_for_mu, eps_for_mu));
    std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model =
        options_.viscosity_model
            ? options_.viscosity_model
            : std::make_shared<materials::fluid::NewtonianViscosity>(
                  options_.viscosity);
    auto viscosity_metadata = FE::analysis::dynamicViscosityMetadata(
        FE::INVALID_FIELD_ID,
        options_.viscosity,
        options_.viscosity_model);
    viscosity_model = FE::constitutive::withConstitutiveLawMetadata(
        std::move(viscosity_model),
        0u,
        std::move(viscosity_metadata));
    const auto mu = constitutive(std::move(viscosity_model), gamma_for_mu).out(0);

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

    FormExpr momentum_form =
        integrateOnActiveVolume(galerkin_momentum_integrand, active_volume_domain);
    FormExpr continuity_form =
        integrateOnActiveVolume(galerkin_continuity_integrand, active_volume_domain);

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

        const auto vms_momentum_integrand =
            inertia + moving_volume + convection_adv + viscous + pressure_adv +
            forcing + supg + cross_stress;
        momentum_form =
            integrateOnActiveVolume(vms_momentum_integrand, active_volume_domain);

        // Continuity: Galerkin + VMS (PSPG-like).
        const auto vms_continuity_integrand = q * div(u) - inner(grad(q), u_sub);
        continuity_form =
            integrateOnActiveVolume(vms_continuity_integrand, active_volume_domain);
    }

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
            dim);
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
    ale_binding.configureInstallOptions(install);
    (void)FE::systems::installFormulation(system, "equations", {u_id, p_id}, residual, install);
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
