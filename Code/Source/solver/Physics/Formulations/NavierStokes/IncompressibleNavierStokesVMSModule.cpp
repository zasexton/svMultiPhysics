/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file IncompressibleNavierStokesVMSModule.cpp
 * @brief Incompressible Navier-Stokes (residual-based VMS) with free surfaces
 *
 * Find (u, p) on the liquid Omega (the mesh, or the cut side Omega_h of a
 * level-set geometry) such that for all (v, q)
 *
 *   int_Omega rho (dt u + (grad u) a - f) . v + 2 mu eps(u) : eps(v) - p div v + q div u
 *     + SUPG/PSPG/LSIC and cross-stress terms of the static subscales u', p'
 *     + boundary and free-surface terms                                         = 0,
 *
 * with a = u - w the velocity relative to the mesh.  The boundary and
 * free-surface terms are stated in NavierStokesBCFactories.h.  Everything
 * that is not mathematics (option checks, field registration, certificates,
 * logs, diagnostic operators, the configuration artifact, the residual-work
 * ledger) is in the support headers included below.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

// Not mathematics: option queries, preflight, fields and initial state,
// geometry adapters, logs, energy declarations, the residual-work ledger,
// the diagnostic operators and the configuration artifact.
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"

#include "FE/Systems/TwoFluidStageRecords.h"

#include "FE/Analysis/ConstitutiveLawMetadata.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/CoupledFieldGaugeConstraint.h"
#include "FE/Constraints/GaugeRegistry.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/LevelSet/LevelSetSideSupport.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FieldBinding.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/MeshBoundaryKinematics.h"
#include "FE/Systems/VertexFieldInitialization.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Geometry/FrameGeometry.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <array>
#include <unordered_map>
#include <sstream>
#include <limits>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

using namespace svmp::FE::forms;
namespace fs = svmp::FE::systems;
namespace cs = svmp::FE::constraints;
using FE::FieldId;
using FE::Real;
using Options = IncompressibleNavierStokesVMSOptions;

// ---------------------------------------------------------------------------
// Free-surface geometry of the options (formerly FreeSurface/FreeSurfaceGeometry)
// ---------------------------------------------------------------------------

namespace {

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

    return freeSurfaceLevelSetGeometry(bc, system).interfaceMarker();
}

[[nodiscard]] FE::interfaces::GeneratedActiveBoundaryMarkerKey
generatedActiveBoundaryMarkerKey(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc) || bc.interface_marker < 0 ||
        physical_boundary_marker < 0 ||
        bc.active_domain == FreeSurfaceActiveDomain::None) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: sharp active-boundary marker requires a resolved unfitted active domain and physical boundary marker");
    }
    return freeSurfaceLevelSetGeometry(bc, system)
        .side(activeDomainSide(bc.active_domain))
        .boundaryKey(physical_boundary_marker);
}

} // namespace

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

/**
 * Normal from the same geometric rule that supplies dI/dInterfaceBoundary.
 * Generated interface normals point from phi<0 to phi>0.  Flip only where a
 * directed outward-liquid normal is required; the surface projector itself
 * is orientation independent.
 */
[[nodiscard]] FE::forms::FormExpr generatedInterfaceOutwardNormal(
    const FreeSurfaceBoundary& bc)
{
    if (isUnfittedLevelSet(bc)) {
        return FE::level_set::outwardInterfaceNormal(
            activeDomainSide(bc.active_domain));
    }
    return FE::forms::FormExpr::normal();
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

/// The level-set geometry of an unfitted free surface: its field, generated
/// domain id, isovalue and (resolved or requested) interface marker.
[[nodiscard]] FE::level_set::LevelSetGeometry freeSurfaceLevelSetGeometry(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    return FE::level_set::LevelSetGeometry::bind(
        system,
        {.level_set_field = resolveLevelSetFieldId(bc, system),
         .domain_id = bc.generated_interface_domain_id,
         .isovalue = bc.level_set_isovalue,
         .interface_marker = bc.interface_marker});
}

[[nodiscard]] FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
generatedUnfittedContactLineMarkerKey(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated contact marker key requires an unfitted free surface");
    }
    if (bc.interface_marker < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires resolved interface_marker >= 0");
    }
    if (bc.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires a non-empty generated_interface_domain_id");
    }
    if (contactLineWallBoundaryMarker(contact_line) < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle requires Contact_line_wall_marker or Contact_line_wall_face");
    }

    return freeSurfaceLevelSetGeometry(bc, system).contactLineKey(
        contactLineWallBoundaryMarker(contact_line),
        contactLineMarker(contact_line));
}

[[nodiscard]] int generatedUnfittedContactLineMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine&
        contact_line,
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system)
{
    if (!isUnfittedLevelSet(bc)) {
        return contactLineConstraintMarker(contact_line);
    }
    return FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
        generatedUnfittedContactLineMarkerKey(contact_line, bc, system));
}

[[nodiscard]] int generatedActiveBoundaryMarker(
    const FreeSurfaceBoundary& bc,
    int physical_boundary_marker,
    const FE::systems::FESystem& system)
{
    return FE::interfaces::stableGeneratedActiveBoundaryMarker(
        generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system));
}

[[nodiscard]] std::optional<int> sharpActiveBoundaryMarkerFor(
    int physical_boundary_marker,
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system)
{
    std::optional<int> marker;
    std::string owner_key;
    for (const auto& bc : boundaries) {
        if (!isUnfittedLevelSet(bc) ||
            bc.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        const auto key = generatedActiveBoundaryMarkerKey(
            bc, physical_boundary_marker, system);
        const int candidate =
            FE::interfaces::stableGeneratedActiveBoundaryMarker(key);
        if (!marker.has_value()) {
            marker = candidate;
            owner_key = key.stableKey();
            continue;
        }
        if (*marker != candidate || owner_key != key.stableKey()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: physical boundary marker " +
                std::to_string(physical_boundary_marker) +
                " is covered by multiple ambiguous unfitted active domains; select one explicit active-domain owner");
        }
    }
    return marker;
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

[[nodiscard]] std::string freeSurfaceCurvatureTangentPolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "not_applicable";
    }
    // SurfaceStress, the only form (decision D48).
    return isUnfittedLevelSet(bc) ? "generated_geometry_refreshed_frozen"
                                  : "fitted_geometry_surface_variation";
}

[[nodiscard]] std::optional<ActiveVolumeDomain> activeVolumeDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
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
        active_domain = ActiveVolumeDomain{
            bc.interface_marker,
            side,
            bc.active_domain_method};

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
        FE_LOG_INFO(oss.str());
    }
    return active_domain;
}

[[nodiscard]] FE::forms::FormExpr integrateOnActiveVolume(
    const FE::forms::FormExpr& integrand,
    const std::optional<ActiveVolumeDomain>& active_domain)
{
    if (!active_domain.has_value()) {
        return integrand.dx(FE::forms::VolumeDomain::full());
    }
    return integrand.dx(FE::forms::VolumeDomain::cut(
        active_domain->interface_marker, active_domain->side));
}

// u_ref of the semi-implicit capillary term: prescribed data in the velocity
// space, registered by registerOn and refreshed by the application.
[[nodiscard]] FE::forms::FormExpr freeSurfaceSemiImplicitReferenceVelocity(
    const FE::systems::FESystem& system)
{
    const std::string name(kFreeSurfaceSemiImplicitReferenceVelocityFieldName);
    const auto field = system.findFieldByName(name);
    if (field == FE::INVALID_FIELD_ID) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: the semi-implicit "
            "reference velocity field '" + name + "' is not registered");
    }
    const auto& rec = system.fieldRecord(field);
    if (rec.source_kind != FE::systems::FieldSourceKind::PrescribedData ||
        system.fieldParticipatesInUnknownVector(field) || !rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: the semi-implicit "
            "reference velocity field '" + name +
            "' must be prescribed data, not an unknown");
    }
    return FE::forms::FormExpr::discreteField(field, *rec.space, name);
}

FE::forms::FormExpr freeSurfaceScalar(
    const FE::forms::bc::ScalarValue& value,
    std::string_view name,
    const FreeSurfaceBoundary& bc)
{
    return FE::forms::bc::toScalarExpr(value, freeSurfaceValueName(name, bc));
}

std::function<std::optional<int>(int)> wettedBoundaryResolver(
    FE::systems::FESystem& system,
    const std::vector<FreeSurfaceBoundary>& free_surfaces)
{
    return [&system, &free_surfaces](int marker) -> std::optional<int> {
        const auto generated = sharpActiveBoundaryMarkerFor(marker, free_surfaces, system);
        if (generated.has_value()) {
            system.registerGeneratedEmbeddedInterfaceMarker(*generated);
        }
        return generated;
    };
}

// ---------------------------------------------------------------------------
// The initial state (formerly NavierStokesInitialConditions)
// ---------------------------------------------------------------------------

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
    return FE::systems::initializeFieldFromMeshVertexField(
        system,
        u0,
        field_id,
        "IncompressibleNavierStokesVMSModule: " + std::string(context));
}

[[nodiscard]] FE::level_set::LevelSetSupportSide levelSetSupportSide(
    FreeSurfaceActiveDomain active_domain) noexcept
{
    switch (active_domain) {
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::level_set::LevelSetSupportSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::level_set::LevelSetSupportSide::Positive;
    case FreeSurfaceActiveDomain::None:
        break;
    }
    return FE::level_set::LevelSetSupportSide::Everywhere;
}

[[nodiscard]] bool pressureVertexOnActiveSide(
    FE::Real phi,
    FE::Real isovalue,
    FreeSurfaceActiveDomain active_domain) noexcept
{
    return FE::level_set::vertexOnLevelSetSide(
        phi, isovalue, levelSetSupportSide(active_domain));
}

[[nodiscard]] std::vector<unsigned char> activePressureSupportVertices(
    const FE::systems::FESystem& system,
    const ActivePressureDomain& active_pressure_domain,
    const LevelSetVertexFieldView& level_set_values,
    FE::GlobalIndex n_vertices)
{
    const auto& bc = *active_pressure_domain.boundary;
    FE::level_set::LevelSetSideSupportQuery query{};
    query.level_set = {level_set_values.values,
                       level_set_values.components,
                       level_set_values.entity_count};
    query.isovalue = bc.level_set_isovalue;
    query.side = levelSetSupportSide(active_pressure_domain.active_domain);
    query.interface_marker = bc.interface_marker;
    query.use_generated_volume_rules =
        bc.active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume;
    query.message_prefix =
        "IncompressibleNavierStokesVMSModule: active-domain pressure initialization";
    return FE::level_set::levelSetSideSupportVertices(system, query, n_vertices);
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

namespace {

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

} // namespace

namespace {

// The field registration, the generated-boundary Nitsche count and the
// fitted normal kinematics of registerOn (formerly NavierStokesFields,
// NavierStokesGeneratedNitsche and FreeSurface/FreeSurfaceFittedALE).

std::vector<FE::constraints::VertexDirichletValue> nodePressureValues(
    const IncompressibleNavierStokesVMSOptions& options)
{
    std::vector<FE::constraints::VertexDirichletValue> values;
    values.reserve(options.node_pressure_constraints.values.size());
    for (const auto& in : options.node_pressure_constraints.values) {
        values.push_back(FE::constraints::VertexDirichletValue{in.node_id, in.pressure});
    }
    return values;
}

FE::constraints::VertexIdMode nodePressureIdMode(
    const IncompressibleNavierStokesVMSOptions& options)
{
    FE::constraints::VertexIdMode mode = FE::constraints::VertexIdMode::GlobalVertexGid;
    switch (options.node_pressure_constraints.id_type) {
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::GlobalVertexGid:
        mode = FE::constraints::VertexIdMode::GlobalVertexGid;
        break;
    case IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId:
        mode = FE::constraints::VertexIdMode::LocalVertexId;
        break;
    }
    return mode;
}

struct NavierStokesFieldIds {
    FE::FieldId velocity{FE::INVALID_FIELD_ID};
    FE::FieldId pressure{FE::INVALID_FIELD_ID};
    FE::FieldId body_force{FE::INVALID_FIELD_ID};
};

NavierStokesFieldIds registerNavierStokesFields(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    NavierStokesPreflight& preflight)
{
    auto& u_spec = preflight.u_spec;
    auto& p_spec = preflight.p_spec;
    auto& body_force_spec = preflight.body_force_spec;
    auto& semi_implicit_reference_spec = preflight.semi_implicit_reference_spec;

    FE::systems::FieldBinding fields(
        system, "IncompressibleNavierStokesVMSModule::registerOn");
    const FE::FieldId u_id = fields.unknown(std::move(u_spec), "velocity");
    const FE::FieldId p_id = fields.unknown(std::move(p_spec), "pressure");

    FE::FieldId body_force_field_id = FE::INVALID_FIELD_ID;
    if (body_force_spec.has_value()) {
        body_force_field_id = fields.prescribed(
            std::move(*body_force_spec),
            options.auto_register_body_force_field,
            "momentum source");
    }
    if (semi_implicit_reference_spec.has_value()) {
        (void)fields.prescribed(
            std::move(*semi_implicit_reference_spec),
            /*auto_register=*/true,
            "semi-implicit reference velocity");
    }

    return NavierStokesFieldIds{
        .velocity = u_id,
        .pressure = p_id,
        .body_force = body_force_field_id,
    };
}

void requireCertifiedNitscheTraces(
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<std::optional<int>(int)>& generated_active_boundary_for,
    bool small_cut_aggregation,
    const std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>&
        generated_nitsche_trace_bindings)
{
    std::size_t expected_generated_nitsche_trace_bindings = 0u;
    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int physical_marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(
                bc,
                "IncompressibleNavierStokesVMSModule::registerOn "
                "generated-boundary Nitsche trace policy");
        const auto generated_marker =
            generated_active_boundary_for(physical_marker);
        if (!generated_marker.has_value()) {
            continue;
        }
        if (!small_cut_aggregation) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche route reached installation "
                "without its preflighted small-cut aggregation policy");
        }
        ++expected_generated_nitsche_trace_bindings;
        if (options.viscosity_model != nullptr) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                    "certified generated-boundary Nitsche routes require "
                    "constant Newtonian viscosity");
        }
    }
    if (generated_nitsche_trace_bindings.size() !=
        expected_generated_nitsche_trace_bindings) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated-boundary Nitsche form binding count differs "
            "from the certified production routes");
    }
}

std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
fittedNormalKinematicDeclarations(
    const FE::systems::ALEBinding& ale_binding,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces,
    const FE::forms::FormExpr& u,
    FE::FieldId u_id)
{
    using namespace svmp::FE::forms;
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations;
    if (ale_binding.coupled()) {
        normal_declarations.reserve(effective_free_surfaces.size());
        for (const auto& boundary : effective_free_surfaces) {
            const bool installs_normal_relation =
                boundary.implementation ==
                    FreeSurfaceImplementation::FittedALE &&
                boundary.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                boundary.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (!installs_normal_relation) {
                continue;
            }
            normal_declarations.push_back(
                FE::systems::MeshNormalBoundaryConstraintDeclaration{
                    .mesh_displacement_field =
                        ale_binding.mesh_displacement_field,
                    .boundary_marker = boundary.boundary_marker,
                    .quantity = FE::systems::MeshNormalBoundaryQuantity::
                        MeshVelocityTrace,
                    .target_kind =
                        FE::systems::MeshNormalBoundaryTargetKind::
                            FluidNormalVelocity,
                    .target_expression =
                        normalTrace(u, currentNormal()),
                    .enforcement_kind =
                        FE::analysis::EnforcementKind::WeakPenalty,
                    .related_velocity_field = u_id,
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule."
                        "FreeSurfaceBoundary",
                    .requires_mesh_flux_consistency =
                        boundary.kinematic_enforcement ==
                        FreeSurfaceKinematicEnforcement::MeshNitsche,
                    .mesh_flux_consistency_nitsche_gamma =
                        boundary.kinematic_enforcement ==
                                FreeSurfaceKinematicEnforcement::MeshNitsche
                            ? boundary.kinematic_nitsche_gamma
                            : FE::Real{0.0},
                });
        }
    }
    return normal_declarations;
}

void declareFittedNormalKinematics(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces)
{
    FE::systems::declareMeshNormalKinematics(
        system,
        operator_tag,
        std::move(normal_declarations),
        [&](int marker) {
        const auto fitted_boundary = std::find_if(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [&](const auto& candidate) {
                return candidate.implementation ==
                           FreeSurfaceImplementation::FittedALE &&
                       candidate.boundary_marker == marker;
            });
        if (fitted_boundary == effective_free_surfaces.end()) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "measurement lost its owning free-surface boundary");
        }
        std::optional<std::array<FE::Real, 3>>
            prescribed_tangential_mesh_velocity;
        if (fitted_boundary->tangential_mesh_policy ==
            FreeSurfaceTangentialMeshPolicy::Prescribed) {
            std::array<FE::Real, 3> target{};
            for (std::size_t component = 0u;
                 component < target.size();
                 ++component) {
                target[component] = constantScalarValueOrThrow(
                    fitted_boundary
                        ->prescribed_tangential_mesh_velocity[component],
                    "fitted prescribed tangential mesh velocity");
            }
            prescribed_tangential_mesh_velocity = target;
        }
        return FE::systems::MeshNormalKinematicBinding{
            .mesh_row_source =
                "Fitted free-surface mesh normal kinematic row on marker " +
                std::to_string(marker),
            .fluid_row_source =
                fittedFluidNormalDescriptorSource(*fitted_boundary),
            .prescribed_tangential_mesh_velocity =
                prescribed_tangential_mesh_velocity,
        };
        });
}

} // namespace

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    IncompressibleNavierStokesVMSOptions options)
    : velocity_space_(std::move(velocity_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_pressure_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_pressure_space,
    IncompressibleTwoFluidOptions options)
    : velocity_space_(std::move(negative_velocity_space))
    , pressure_space_(std::move(negative_pressure_space))
    , positive_velocity_space_(std::move(positive_velocity_space))
    , positive_pressure_space_(std::move(positive_pressure_space))
    , two_fluid_options_(std::move(options))
{
}

// ---------------------------------------------------------------------------
// Bulk residual
// ---------------------------------------------------------------------------

NavierStokesVMSIntegrands navierStokesVMSIntegrands(const NavierStokesTerms& t,
                                                    const NavierStokesVMSSettings& s)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto zero = zeroVector(dim);
    const auto w_mesh = meshVelocity();
    const auto w = s.ale ? w_mesh : zero;                                 // mesh velocity
    const auto a = s.convection ? (s.ale ? (u - w_mesh) : u) : zero;      // velocity relative to the mesh
    const auto chi = s.moving_control_volume ? (div(w) * u) : zero;       // moving control volume

    // Strong momentum residual without and with the pressure gradient.
    const auto R_np = rho * (dt(u) + grad(u) * a + chi - f) - div(FormExpr::constant(2.0) * mu * sym(grad(u)));
    const auto R_m = R_np + grad(p);

    const auto inertia = rho * inner(dt(u), v);
    const auto mcv = s.moving_control_volume ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    NavierStokesVMSIntegrands r;
    r.galerkin_momentum = inertia + mcv + convection + viscous + pressure + forcing;
    r.galerkin_continuity = q * div(u);
    r.momentum = r.galerkin_momentum;
    r.continuity = r.galerkin_continuity;
    r.convection = mcv + convection;
    r.pressure = pressure;
    r.forcing = forcing;
    if (!s.vms) {
        return r;
    }

    // Static subscales with the element metric K = J^{-T} J^{-1}:
    //   tau_M = 1 / (rho sqrt(4 ct_m^2/dt^2 + a.K a + ct_c K:K nu^2)),  tau_C = 1 / (tau_M tr K).
    const auto eps = FormExpr::constant(s.epsilon);
    const auto dts = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(s.ct_m);
    const auto ct_c = FormExpr::constant(s.ct_c);
    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;
    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dts * dts);
    const auto tau_m = FormExpr::constant(1.0) /
                       (rho * sqrt(kT + inner(a, K * a) + ct_c * doubleContraction(K, K) * (nu * nu) + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);
    const auto u_s = -tau_m * R_m;                                         // u'
    const auto p_s = -tau_c * div(u);                                      // p'
    const auto u_adv = s.convection ? (u + u_s - w) : a;

    const auto convection_adv = rho * inner(grad(u) * u_adv, v);
    const auto pressure_adv = -(p + p_s) * div(v);
    const auto supg = -rho * inner(grad(v) * u_adv, u_s);
    // Cross stress (u'.grad v) . tau_B (u'.grad) u,  tau_B = rho / sqrt(u'.K u').
    const auto cross_stress = s.convection
        ? inner(grad(v) * u_s, (rho / sqrt(inner(u_s, K * u_s) + eps)) * (grad(u) * u_s))
        : FormExpr::constant(0.0);
    r.momentum = inertia + mcv + convection_adv + viscous + pressure_adv + forcing + supg + cross_stress;

    // PSPG, split: the pressure-gradient part is its own kernel group.  The
    // unit factors keep the installed trees of the previous module.
    r.pspg_pressure_gradient = FormExpr::constant(1.0) * FormExpr::constant(1.0) * inner(grad(q), tau_m * grad(p));
    r.pspg_without_pressure_gradient = FormExpr::constant(1.0) * inner(grad(q), tau_m * R_np);
    r.continuity = r.galerkin_continuity + r.pspg_without_pressure_gradient;
    return r;
}

namespace {

// ---------------------------------------------------------------------------
// Material data
// ---------------------------------------------------------------------------

/// mu, or mu(shear rate) with shear rate = sqrt(2 eps(u):eps(u)); the model is
/// tagged so that the installer can publish the viscosity law.
[[nodiscard]] FormExpr viscosity(const Options& o, const FormExpr& u)
{
    const auto eps = sym(grad(u));
    const auto shear_rate = sqrt(FormExpr::constant(2.0) * inner(eps, eps));
    if (!o.viscosity_model) {
        return FormExpr::constant(o.viscosity);
    }
    auto model = FE::constitutive::withConstitutiveLawMetadata(
        o.viscosity_model, 0u,
        FE::analysis::dynamicViscosityMetadata(FE::INVALID_FIELD_ID, o.viscosity, o.viscosity_model));
    return constitutive(std::move(model), shear_rate).out(0);
}

/// f = g + f(x, t) + f_h - 2 Omega x u   (constant, space-time, field and Coriolis parts).
[[nodiscard]] FormExpr bodyForce(const Options& o, const FormExpr& g, int dim, FieldId f_id,
                                 const FE::spaces::FunctionSpace& V, const FormExpr& u)
{
    FormExpr f = g;
    if (o.has_body_force_spacetime) {
        f = f + FormExpr::asVector(bc::toVectorExpr(o.body_force_spacetime, dim, "ns_body_force_spacetime_"));
    }
    if (f_id != FE::INVALID_FIELD_ID) {
        f = f + StateField(f_id, V, o.body_force_field_name);
    }
    if (o.rotating_frame_coriolis_enabled) {
        f = f + FormExpr::constant(-2.0) *
                    cross(FormExpr::asVector(bc::toVectorExpr(o.rotating_frame_angular_velocity, 3,
                                                              "ns_rotating_frame_angular_velocity_")),
                          u);
    }
    return f;
}

// ---------------------------------------------------------------------------
// Two-fluid composition
// ---------------------------------------------------------------------------

[[nodiscard]] int resolvedInterfaceMarker(
    const IncompressibleTwoFluidOptions& options,
    FE::FieldId level_set_field)
{
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(level_set_field);
    key.domain_id = options.generated_interface_domain_id;
    key.isovalue = options.level_set_isovalue;
    key.requested_marker = options.interface_marker;
    return FE::interfaces::stableGeneratedInterfaceMarker(key);
}

void addSharedGaugeEvidence(FE::systems::FESystem& system,
                            FE::FieldId negative_pressure,
                            FE::FieldId positive_pressure)
{
    constexpr std::string_view source =
        "Coupled two-fluid shared pressure gauge constraint";
    for (const auto field : {negative_pressure, positive_pressure}) {
        system.gaugeRegistry().addAnchoring(FE::gauge::AnchoringEvidence{
            .field = field,
            .component = -1,
            .region = -1,
            .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = FE::gauge::AnchoringVerdict::Anchored,
            .source = std::string(source),
        });
    }
}

[[nodiscard]] IncompressibleNavierStokesVMSOptions makePhaseOptions(
    const IncompressibleTwoFluidOptions& owner,
    const IncompressibleTwoFluidPhaseOptions& phase,
    FreeSurfaceActiveDomain active_domain,
    int interface_marker)
{
    IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = phase.velocity_field_name;
    options.pressure_field_name = phase.pressure_field_name;
    options.operator_tag = owner.operator_tag;
    options.density = phase.density;
    options.viscosity = phase.viscosity;
    options.body_force = owner.body_force;
    options.enable_convection = owner.enable_convection;
    options.enable_vms = owner.enable_vms;
    options.ct_m = owner.ct_m;
    options.ct_c = owner.ct_c;
    options.stabilization_epsilon = owner.stabilization_epsilon;
    options.jit_policy = owner.jit_policy;
    options.velocity_dirichlet = phase.velocity_dirichlet;
    options.velocity_dirichlet.insert(
        options.velocity_dirichlet.end(),
        owner.shared_velocity_dirichlet.begin(),
        owner.shared_velocity_dirichlet.end());

    IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary interface;
    interface.role =
        FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
    interface.implementation =
        FreeSurfaceImplementation::UnfittedLevelSet;
    interface.boundary_marker = -1;
    interface.interface_marker = interface_marker;
    interface.level_set_field_name = owner.level_set_field_name;
    interface.generated_interface_domain_id =
        owner.generated_interface_domain_id;
    interface.generated_interface_geometry =
        owner.generated_interface_geometry;
    interface.geometry_tangent_policy = owner.geometry_tangent_policy;
    interface.level_set_isovalue = owner.level_set_isovalue;
    interface.active_domain = active_domain;
    interface.active_domain_method =
        FreeSurfaceActiveDomainMethod::CutVolume;
    interface.active_domain_smoothing_width = FE::Real{0.0};
    interface.allow_full_domain_unfitted_free_surface = false;
    interface.external_pressure = FE::Real{0.0};
    interface.surface_tension = FE::Real{0.0};
    interface.surface_tension_form =
        FreeSurfaceSurfaceTensionForm::Automatic;
    interface.normal_kinematic_policy =
        FreeSurfaceNormalKinematicPolicy::None;
    interface.tangential_mesh_policy =
        FreeSurfaceTangentialMeshPolicy::Free;
    interface.kinematic_enforcement =
        FreeSurfaceKinematicEnforcement::None;
    interface.cut_cell_stabilization.enabled = true;
    interface.cut_cell_stabilization.pressure_gradient_penalty =
        owner.pressure_gradient_penalty;
    interface.cut_cell_stabilization.pressure_policy =
        FreeSurfacePressureStabilizationPolicy::Enabled;
    interface.cut_cell_stabilization.use_cut_metadata_scale =
        owner.use_cut_metadata_scale;
    interface.cut_cell_stabilization.cut_metadata_scale_cap =
        owner.cut_metadata_scale_cap;
    interface.contact_lines.clear();
    interface.small_cut_aggregation = true;
    interface.small_cut_aggregation_guards =
        owner.small_cut_aggregation_guards;
    options.free_surface.push_back(std::move(interface));
    return options;
}

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerOn(fs::FESystem& system) const
{
    if (two_fluid_options_) {
        registerTwoFluidOn(system);
        return;
    }
    const auto& o = options_;
    effective_configuration_artifact_.reset();
    auto preflight = preflightNavierStokes(system, o, velocity_space_, pressure_space_);   // no mutation before
    const int dim = preflight.dim;
    const auto& free_surfaces = preflight.effective_free_surfaces;
    const auto ids = registerNavierStokesFields(system, o, preflight);              // u, p, f_h, u_ref
    const FieldId u_id = ids.velocity;
    const FieldId p_id = ids.pressure;
    post::declareFreeSurfaceDiscreteFunctionals(free_surfaces, system, u_id, o.density, o.body_force,
                                          !o.has_body_force_spacetime && ids.body_force == FE::INVALID_FIELD_ID,
                                          o.viscosity, o.viscosity_model == nullptr);
    // Exterior conditions act on the wetted part of a boundary when the liquid is a cut side.
    const auto wetted = wettedBoundaryResolver(system, free_surfaces);

    // Constraints, in the plan's phase order: vertex values < essential < inactive side < aggregation
    // < contact-point law.  With a cut liquid L the traction on Gamma_h fixes the pressure constant, the dry
    // side is pinned (u = 0, p = 0) and small cut cells are aggregated.
    cs::ConstraintPlan plan(system);
    const auto liquid = activePressureDomainFor(free_surfaces);
    const bool cut_liquid = liquid.has_value() &&
                            liquid->boundary->active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume;
    if (cut_liquid && isExteriorOnePhaseBoundary(*liquid->boundary)) {
        plan.anchorGauge(o.operator_tag, Factories::freeSurfacePressureAnchor(p_id, *liquid->boundary));
    }
    const bool liquid_constraints = cut_liquid && hasNativeMeshOrWarn(system);
    const bool aggregated = liquid_constraints && liquid->boundary->small_cut_aggregation;
    if (liquid_constraints) {
        const auto L =
            freeSurfaceLevelSetGeometry(*liquid->boundary, system).side(activeDomainSide(liquid->active_domain));
        plan.inactiveSide({u_id}, L).inactiveSide({p_id}, L);
        if (aggregated) {
            plan.aggregate({u_id, p_id}, L, liquid->boundary->small_cut_aggregation_guards);
        }
    }
    if (!o.node_pressure_constraints.values.empty()) {
        plan.vertexValues(p_id, nodePressureValues(o), nodePressureIdMode(o));
    }
    plan.commitThrough(cs::ConstraintPhase::VertexValues);

    const auto u = StateField(u_id, *velocity_space_, o.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, o.pressure_field_name);
    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");
    const auto ale = fs::resolveALEBinding(system, preflight.ale_options);
    auto fitted_normal_kinematics = fittedNormalKinematicDeclarations(ale, free_surfaces, u, u_id);
    if (!system.hasOperator(o.operator_tag)) {
        system.addOperator(o.operator_tag);
    }

    // The liquid Omega: the mesh, or the cut side {phi_h < c} or {phi_h > c}; the bulk residual on it.
    const auto active_volume = activeVolumeDomainFor(free_surfaces);
    const auto Omega = active_volume ? VolumeDomain::cut(active_volume->interface_marker, active_volume->side)
                                     : VolumeDomain::full();
    const auto rho = FormExpr::constant(o.density);
    const auto g = constantVector(o.body_force, dim);
    const NavierStokesTerms t{u, p, v, q, rho, viscosity(o, u),
                              bodyForce(o, g, dim, ids.body_force, *velocity_space_, u), dim};
    const bool mcv = o.enable_ale && o.include_moving_control_volume_transient;
    const auto r = navierStokesVMSIntegrands(
        t, {.convection = o.enable_convection, .ale = o.enable_ale, .moving_control_volume = mcv,
            .vms = preflight.enable_vms,
            .ct_m = o.ct_m, .ct_c = o.ct_c, .epsilon = o.stabilization_epsilon});
    FormExpr M = r.momentum.dx(Omega);
    FormExpr C = r.continuity.dx(Omega);
    const auto pspg_grad_p = r.pspg_pressure_gradient.isValid() ? r.pspg_pressure_gradient.dx(Omega) : FormExpr{};

    // Bookkeeping of the opt-in diagnostics and of the free-surface work ledger (exact additive groups of
    // the residual), collected while the terms are stated.
    const auto active_continuity = pspg_grad_p.isValid() ? C + pspg_grad_p : C;
    const bool nonconservative_force = o.has_body_force_spacetime || ids.body_force != FE::INVALID_FIELD_ID ||
                                       o.rotating_frame_coriolis_enabled;
    auto work = post::bulkResidualWork(r, M, C, pspg_grad_p, -rho * inner(g, v), nonconservative_force,
                                 preflight.enable_vms, Omega);
    auto& diagnostics = fs::diagnosticRegistry(system);   // shared with the other formulations of the system
    post::declareDiagnosticSets(diagnostics);
    std::optional<post::ConservativeBalanceForms> balance;
    if (post::conservativeBalanceDiagnosticSupported(diagnostics, free_surfaces, preflight.has_exterior_free_surface,
                                               o.conservative_balance_diagnostics)) {
        balance = post::ConservativeBalanceForms{.pressure = r.pressure.dx(Omega),
                                           .gravitational_potential = (-rho * inner(g, v)).dx(Omega)};
    }
    auto* const split = balance ? &*balance : nullptr;

    // Boundary conditions and free surfaces, in the order their terms enter M and C.
    if (!o.coupled_outflow_rcr.empty() || !o.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, o.jit_policy);
    }
    fs::BoundaryConditionManager bcs;
    bcs.install(o.velocity_dirichlet_weak, Factories::reserveMarker);   // Nitsche below; reserve the marker
    fs::FormulationBuilder F(system, o.operator_tag, physicsInstallOptions(o.jit_policy));
    F.symbolicTangent();
    for (const auto& b : free_surfaces) {
        if (isExteriorOnePhaseBoundary(b) && isUnfittedLevelSet(b)) {
            Factories::addUnfittedFreeSurface(M, system, plan, F, o, b, t, u_id, split);
        } else if (isExteriorOnePhaseBoundary(b)) {
            Factories::addFittedFreeSurface(M, system, plan, bcs, o, b, ale, t, u_id, F, split);
        }
        const auto J = Factories::cutFacetPressurePenalty(b, t, o, velocity_space_->polynomial_order(),
                                                          pressure_space_->polynomial_order(), F);
        if (J.isValid()) {
            C = C + J;
            work.ghost_penalty = sumOf(work.ghost_penalty, J);
        }
    }
    const auto on = [&](const auto& bc) { return wetted(bc.boundary_marker); };
    bcs.install(o.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim, on(bc)); });
    bcs.install(o.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim, on(bc)); });
    bcs.install(o.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho, on(bc)); });
    const auto rcr = [&](const auto& bc) { return Factories::toCoupledOutflowBC(bc, system, u, rho, on(bc)); };
    bcs.install(o.coupled_outflow_rcr, rcr);
    bcs.install(o.coupled_outflow_rcrcr, rcr);
    bcs.install(o.velocity_dirichlet,
                [&](const auto& bc) { return Factories::toVelocityEssentialBC(bc, dim, o.velocity_field_name); });
    plan.essentials([&](fs::FESystem& target) { bcs.applyAll(target, M, u, v, u_id); });   // weak terms into M
    fs::BoundaryConditionManager pressure_bcs;
    pressure_bcs.install(o.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, o.pressure_field_name); });
    plan.essentials([&](fs::FESystem& target) { pressure_bcs.applyAll(target, p_id); });
    F.log(Factories::activeDomainConstraintLogs(liquid_constraints ? liquid->boundary : nullptr, aggregated));
    plan.commit();

    // Weak velocity Dirichlet (Nitsche), certified on generated wetted boundaries.
    std::vector<bc::GeneratedBoundaryNitscheTraceFormBinding> nitsche_traces;
    Factories::applyVelocityNitscheBCs(M, C, o, dim, u, p, v, q, t.mu, wetted,
                                       aggregated ? &nitsche_traces : nullptr,
                                       &work.weak_boundary_momentum, &work.weak_boundary_continuity);
    requireCertifiedNitscheTraces(o, wetted, aggregated, nitsche_traces);

    // Install: the residual, the residual-work channels, then the PSPG pressure-gradient group.
    if (!o.viscosity_model) {
        F.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(u_id, o.viscosity));
    }
    if (aggregated) {
        F.certifyGeneratedBoundaryNitscheTraces(nitsche_traces, liquid->boundary->interface_marker,
                                                o.generated_boundary_nitsche_minimum_energy_ratio);
    }
    F.configure(ale);
    (void)F.install({u_id, p_id}, M + C);
    if (preflight.has_exterior_free_surface) {
        post::declareFreeSurfaceExternalBoundaryEnergy(system, o);
        post::installResidualWork(F, o, u_id, p_id, work,
                                  {.include_mcv = mcv,
                                   .has_nonconservative_body_force = nonconservative_force,
                                   .enable_vms = preflight.enable_vms});
    }
    if (pspg_grad_p.isValid()) {
        (void)F.installGroup({p_id}, pspg_grad_p,
                             {.source_component_tag = "navier_stokes_vms_pspg_pressure_gradient",
                              .extra_trial_fields = {u_id},
                              .drop_generated_boundary_nitsche_trace_requests = true});
    }

    post::installDiagnosticOperators(F, u_id, p_id, r, split, active_continuity, work.ghost_penalty, u, q, Omega);
    declareFittedNormalKinematics(system, o.operator_tag, std::move(fitted_normal_kinematics), free_surfaces);
    effective_configuration_artifact_ = makeEffectiveConfigurationArtifact(
        o, free_surfaces, system, dim, preflight.enable_vms);
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::applyInitialConditions(const fs::FESystem& system,
                                                                 FE::backends::GenericVector& u0) const
{
    if (two_fluid_options_) {
        applyTwoFluidInitialConditions(system, u0);
        return;
    }
    // u0 and p0 from mesh vertex fields where given; hydrostatic initialization sets
    //   p0(x) = p_ref + rho g . (x - x_ref)   on the liquid support (p_ref on dry vertices).
    const auto& init = options_.hydrostatic_pressure_initialization;
    const auto hydrostatic = [this, &init](const std::array<Real, 3>& x) {
        Real p0 = init.reference_pressure;
        for (std::size_t d = 0; d < options_.body_force.size(); ++d) {
            p0 += options_.density * options_.body_force[d] * (x[d] - init.reference_point[d]);
        }
        return p0;
    };
    initializeNavierStokesState(system, u0, options_, hydrostatic);
}

// ---------------------------------------------------------------------------
// The two-fluid mode: the statement above on each side of a generated
// material interface (the negative phase, then the positive phase), coupled
// on the interface (NavierStokesBCFactories.h) and through one shared
// pressure gauge.  velocity_space_ and pressure_space_ are the negative
// phase's.  The messages keep the two-fluid module's texts.
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerTwoFluidOn(fs::FESystem& system) const
{
    const auto& o = *two_fluid_options_;
    effective_configuration_artifact_.reset();
    const auto preflight = preflightIncompressibleTwoFluid(system, o, velocity_space_, pressure_space_,
                                                           positive_velocity_space_, positive_pressure_space_);
    const auto dimension = preflight.dimension;
    const auto level_set = preflight.level_set;

    const int interface_marker =
        resolvedInterfaceMarker(o, level_set);
    const Factories::IncompressibleTwoFluidInterfaceParameters interface_parameters{
        .dimension = dimension,
        .interface_marker = interface_marker,
        .negative_density = o.negative_phase.density,
        .positive_density = o.positive_phase.density,
        .negative_viscosity = o.negative_phase.viscosity,
        .positive_viscosity = o.positive_phase.viscosity,
        .nitsche_gamma = o.interface_nitsche_gamma,
        .surface_tension = o.surface_tension,
        .include_transient_penalty =
            o.include_transient_interface_penalty,
        .prescribed_pressure_jump = o.prescribed_pressure_jump,
        .prescribed_viscous_traction_jump =
            o.prescribed_viscous_traction_jump,
    };
    const auto interface_weights =
        Factories::incompressibleTwoFluidInterfaceWeights(interface_parameters);

    auto negative_options = makePhaseOptions(
        o,
        o.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        o,
        o.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        velocity_space_,
        pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.registerOn(system);
    positive_module.registerOn(system);

    const auto u_negative = system.findFieldByName(
        o.negative_phase.velocity_field_name);
    const auto p_negative = system.findFieldByName(
        o.negative_phase.pressure_field_name);
    const auto u_positive = system.findFieldByName(
        o.positive_phase.velocity_field_name);
    const auto p_positive = system.findFieldByName(
        o.positive_phase.pressure_field_name);
    if (u_negative == FE::INVALID_FIELD_ID ||
        p_negative == FE::INVALID_FIELD_ID ||
        u_positive == FE::INVALID_FIELD_ID ||
        p_positive == FE::INVALID_FIELD_ID) {
        throw std::logic_error(
            "IncompressibleTwoFluidModule: phase registration did not create all four unknown fields");
    }

    using namespace FE::forms;
    const auto forms = Factories::buildIncompressibleTwoFluidInterfaceForms(
        StateField(u_negative,
                   *velocity_space_,
                   o.negative_phase.velocity_field_name),
        StateField(p_negative,
                   *pressure_space_,
                   o.negative_phase.pressure_field_name),
        TestField(u_negative, *velocity_space_, "v_negative"),
        TestField(p_negative, *pressure_space_, "q_negative"),
        StateField(u_positive,
                   *positive_velocity_space_,
                   o.positive_phase.velocity_field_name),
        StateField(p_positive,
                   *positive_pressure_space_,
                   o.positive_phase.pressure_field_name),
        TestField(u_positive, *positive_velocity_space_, "v_positive"),
        TestField(p_positive, *positive_pressure_space_, "q_positive"),
        interface_parameters);

    auto install = physicsInstallOptions(o.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    install.source_component_tag =
        o.operator_tag + "_two_fluid_material_interface";
    install.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(
        u_negative, o.negative_phase.viscosity, {}, "negative_dynamic_viscosity"));
    install.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(
        u_positive, o.positive_phase.viscosity, {}, "positive_dynamic_viscosity"));
    const std::array<FE::FieldId, 4> fields{
        u_negative, p_negative, u_positive, p_positive};
    (void)FE::systems::installFormulation(
        system,
        o.operator_tag,
        std::span<const FE::FieldId>(fields),
        forms.residual,
        install);

    if (o.shared_pressure_gauge.has_value()) {
        const auto& gauge = *o.shared_pressure_gauge;
        std::vector<FE::constraints::VertexDirichletValue> values{
            FE::constraints::VertexDirichletValue{
                .vertex_id = gauge.vertex_gid,
                .value = gauge.pressure,
            }};
        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(
                p_negative,
                std::move(values),
                FE::constraints::VertexIdMode::GlobalVertexGid));
    } else {
        system.addSystemConstraint(
            std::make_unique<FE::constraints::CoupledFieldGaugeConstraint>(
                p_negative, p_positive));
    }
    addSharedGaugeEvidence(system, p_negative, p_positive);

    system.declareMaterialInterfaceTransportVelocity(
        FE::interfaces::MaterialInterfaceTransportVelocityDeclaration{
            .dimension = dimension,
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .positive_velocity_field = u_positive,
            .level_set_isovalue = o.level_set_isovalue,
            .negative_trace_weight =
                interface_weights.negative_complement,
            .positive_trace_weight =
                interface_weights.positive_complement,
            .geometry_domain_id = o.generated_interface_domain_id,
            .owner_component = "incompressible_two_fluid",
        });

    svmp::FE::systems::twoFluidStageRecords(system).declareTwoFluidAcceptedStageDiagnostics(
        FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration{
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .negative_pressure_field = p_negative,
            .positive_velocity_field = u_positive,
            .positive_pressure_field = p_positive,
            .operator_tag = o.operator_tag,
            .geometry_domain_id = o.generated_interface_domain_id,
            .level_set_isovalue = o.level_set_isovalue,
            .require_conservative_phase_momentum_reconciliation =
                o
                    .require_conservative_phase_momentum_reconciliation,
            .require_accepted_stage_numerics = true,
            .pressure_stabilization_coefficient =
                o.pressure_gradient_penalty,
            .pressure_stabilization_use_cut_metadata_scale =
                o.use_cut_metadata_scale,
            .pressure_stabilization_cut_metadata_scale_cap =
                o.cut_metadata_scale_cap,
            .parameters =
                FE::interfaces::IncompressibleTwoFluidDiagnosticParameters{
                    .dimension = dimension,
                    .interface_marker = interface_marker,
                    .negative_density = o.negative_phase.density,
                    .positive_density = o.positive_phase.density,
                    .negative_viscosity = o.negative_phase.viscosity,
                    .positive_viscosity = o.positive_phase.viscosity,
                    .nitsche_gamma = o.interface_nitsche_gamma,
                    .surface_tension = o.surface_tension,
                    .include_transient_penalty =
                        o.include_transient_interface_penalty,
                    .prescribed_pressure_jump =
                        o.prescribed_pressure_jump,
                    .prescribed_viscous_traction_jump =
                        o.prescribed_viscous_traction_jump,
                    .body_force = o.body_force,
                },
            .owner_component = "incompressible_two_fluid",
        });

    effective_configuration_artifact_ = makeIncompressibleTwoFluidConfigurationArtifact(
        o, dimension, interface_marker, interface_weights);
}

void IncompressibleNavierStokesVMSModule::applyTwoFluidInitialConditions(const fs::FESystem& system,
                                                                         FE::backends::GenericVector& u0) const
{
    const auto& o = *two_fluid_options_;
    const auto level_set =
        system.findFieldByName(o.level_set_field_name);
    if (level_set == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: initial conditions require the registered level-set field '" +
            o.level_set_field_name + "'");
    }
    const auto interface_marker =
        resolvedInterfaceMarker(o, level_set);
    auto negative_options = makePhaseOptions(
        o,
        o.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        o,
        o.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        velocity_space_,
        pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.applyInitialConditions(system, u0);
    positive_module.applyInitialConditions(system, u0);
}

std::optional<EffectiveConfigurationArtifact>
IncompressibleNavierStokesVMSModule::effectiveConfigurationArtifact() const
{
    return effective_configuration_artifact_;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
