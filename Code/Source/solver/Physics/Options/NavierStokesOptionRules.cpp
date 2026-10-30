/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#include "Physics/Options/NavierStokesOptionRules.h"

#include "Physics/Options/NavierStokesOptionSchema.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"
#include "FE/Analysis/GeneratedBoundaryAggregateTraceCertificate.h"
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
#include "FE/Core/NumericValidation.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Interfaces/ContactLineGeometryCertificate.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Systems/FieldBinding.h"
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
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/SystemCollectives.h"
#include "FE/Systems/VertexFieldInitialization.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// Fill only the identity needed at this validation point. Contact eligibility
// remains separate from the earlier field/law checks in the caller.
[[nodiscard]] int prepareSurfaceIdentity(
    const FE::systems::FESystem& system,
    const FreeSurfaceBoundary& boundary,
    PreparedSurfaceIdentity& identity,
    std::optional<std::size_t> contact_index = std::nullopt)
{
    if (contact_index.has_value()) {
        if (!isUnfittedLevelSet(boundary)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated contact marker key requires an unfitted free surface");
        }
        if (boundary.interface_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires resolved interface_marker >= 0");
        }
        if (boundary.generated_interface_domain_id.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: generated unfitted contact-line marker requires a non-empty generated_interface_domain_id");
        }
        if (contactLineWallBoundaryMarker(boundary.contact_lines[*contact_index]) < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle requires a contact wall (Contact_line_wall_face)");
        }
    }
    if (!identity.geometry.has_value()) {
        const auto field = FE::level_set::resolveScalarLevelSetField(
            system, boundary.level_set_field_name,
            "IncompressibleNavierStokesVMSModule: unfitted free surface references unknown level-set field '",
            "IncompressibleNavierStokesVMSModule");
        identity.geometry = FE::level_set::LevelSetGeometry::bind(
            system,
            {.level_set_field = field,
             .domain_id = boundary.generated_interface_domain_id,
             .isovalue = boundary.level_set_isovalue,
             .interface_marker = boundary.interface_marker});
    }
    if (!contact_index.has_value()) {
        return identity.geometry->interfaceMarker();
    }
    const auto existing = std::find_if(
        identity.contacts.begin(), identity.contacts.end(),
        [&](const auto& contact) { return contact.contact_index == *contact_index; });
    if (existing != identity.contacts.end()) {
        return existing->domain.marker();
    }
    const auto& contact = boundary.contact_lines[*contact_index];
    const int wall = contactLineWallBoundaryMarker(contact);
    auto key = identity.geometry->contactLineKey(wall, contactLineMarker(contact));
    const int marker =
        FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key);
    identity.contacts.push_back(
        {*contact_index, std::move(key), FE::forms::ContactLineDomain(marker, wall)});
    return marker;
}

constexpr FE::Real kPressureGaugeLevelSetMargin = 1.0e-8;

[[nodiscard]] const FE::systems::LocalVertexMessages& pressureConstraintVertexMessages()
{
    static const FE::systems::LocalVertexMessages messages{
        .negative_id =
            "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id",
        .no_native_mesh =
            "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires a native mesh for global vertex ids",
        .no_mesh_support =
            "IncompressibleNavierStokesVMSModule: active-domain pressure constraint validation requires native mesh support for global vertex ids"};
    return messages;
}

[[nodiscard]] FE::systems::MeshVertexFieldMessages pressureLevelSetMessages(
    std::string_view context)
{
    const std::string owner =
        "IncompressibleNavierStokesVMSModule: " + std::string(context);
    return {
        .no_native_mesh = owner + " requires a native mesh vertex level-set field",
        .missing_field_prefix = owner + " could not find vertex level-set field '",
        .not_float64 = owner + " requires a Float64 vertex level-set field",
        .no_components = owner + " requires at least one level-set component",
        .too_few_entries =
            owner + " level-set field has fewer entries than pressure vertices",
        .empty = owner + " found an empty level-set field",
        .no_mesh_support =
            "IncompressibleNavierStokesVMSModule: active-domain pressure operations require native mesh support"};
}

[[nodiscard]] FE::level_set::LevelSetSupportSide pressureSupportSide(
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

[[nodiscard]] FE::Real finiteForPressureReport(FE::Real value) noexcept
{
    return std::isfinite(value) ? value : FE::Real{0.0};
}

void validateActiveDomainPressureValues(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    std::span<const FE::constraints::VertexDirichletValue> values)
{
    if (values.empty()) {
        return;
    }
    const auto active_domain = activePressureDomainFor(free_surfaces);
    if (!active_domain.has_value()) {
        return;
    }
    const auto& bc = *active_domain->boundary;
    const auto level_set = FE::systems::requireMeshVertexField(
        system, bc.level_set_field_name, system.meshAccess().numVertices(),
        pressureLevelSetMessages("active-domain pressure constraint validation"));
    const auto statistics = FE::level_set::requireVertexValuesOnLevelSetSide(
        system,
        {.values = values,
         .id_mode = nodePressureIdMode(options),
         .messages = pressureConstraintVertexMessages()},
        {.level_set = {level_set.values, level_set.components, level_set.entity_count},
         .isovalue = bc.level_set_isovalue,
         .side = pressureSupportSide(active_domain->active_domain),
         .exclusion_margin = kPressureGaugeLevelSetMargin,
         .failure_message = [&](const FE::level_set::VertexSideFailure& failure) {
             if (failure.reason == FE::level_set::VertexSideViolation::OutsideMesh) {
                 return "IncompressibleNavierStokesVMSModule: active-domain pressure constraint references vertex " +
                        std::to_string(failure.requested_vertex) +
                        " outside the local pressure mesh";
             }
             std::ostringstream message;
             message << "IncompressibleNavierStokesVMSModule: active-domain pressure "
                     << "constraint node_id=" << failure.requested_vertex
                     << " local_vertex=" << failure.local_vertex;
             if (failure.reason == FE::level_set::VertexSideViolation::OutsideSide) {
                 message << " is on the dry side for Active_domain="
                         << activeDomainName(active_domain->active_domain)
                         << " phi=" << failure.sample.phi
                         << " isovalue=" << bc.level_set_isovalue;
             } else {
                 message << " is too close to the level-set interface: |phi-isovalue|="
                         << std::abs(failure.sample.signed_gap)
                         << " margin=" << kPressureGaugeLevelSetMargin;
             }
             return message.str();
         }});
    FE::logRecord(post::navierStokesLog("pressure gauge")
                      .diagnostic("pressure_gauge_check")
                      .integer("constraints", values.size())
                      .integer("checked_local_constraints", statistics.checked)
                      .integer("skipped_nonlocal_constraints", statistics.skipped_nonlocal)
                      .real("constraint_pressure_min", finiteForPressureReport(statistics.input_min))
                      .real("constraint_pressure_max", finiteForPressureReport(statistics.input_max))
                      .text("active_domain", activeDomainName(active_domain->active_domain))
                      .real("isovalue", bc.level_set_isovalue)
                      .real("min_signed_gap", finiteForPressureReport(statistics.signed_gap_min))
                      .real("max_signed_gap", finiteForPressureReport(statistics.signed_gap_max))
                      .real("margin", kPressureGaugeLevelSetMargin));
}

// The contact-line validators (formerly FreeSurface/FreeSurfaceContactCertificate).

constexpr FE::Real kMinimumContactTransverseSine = FE::interfaces::kContactLineMinimumTransverseSine;

[[nodiscard]] int axisAlignedWallNormalAxis(
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto normal = normalizedWallNormal(contact_line);
    const auto law = contactWallLawName(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    int axis = -1;
    for (int component = 0; component < 3; ++component) {
        const auto magnitude =
            std::abs(normal[static_cast<std::size_t>(component)]);
        if (component >= dim) {
            if (magnitude > tolerance) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must lie in the velocity space dimension");
            }
            continue;
        }
        if (std::abs(magnitude - FE::Real{1.0}) <= tolerance) {
            if (axis >= 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " wall_normal must have exactly one axis-aligned component");
            }
            axis = component;
        } else if (magnitude > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " currently requires an axis-aligned wall_normal because general linear-combination normal constraints are unavailable");
        }
    }
    if (axis < 0) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an axis-aligned wall_normal");
    }
    return axis;
}

void validateDynamicContactPlanarWallMarker(
    const FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line)
{
    constexpr FE::Real alignment_tolerance = FE::interfaces::kContactLineAlignmentTolerance;
    const auto configured = normalizedWallNormal(contact_line);
    const auto evidence = FE::interfaces::planarWallEvidence(
        system,
        contactLineWallBoundaryMarker(contact_line),
        configured);

    if (evidence.face_count == 0u || evidence.invalid_face_count != 0u) {
        std::ostringstream message;
        message
            << "IncompressibleNavierStokesVMSModule: "
            << contactWallLawName(contact_line)
            << " requires every face on Contact_line_wall_marker to be a "
               "first-order planar face with the configured physical "
               "outward wall_normal"
            << " wall_boundary_marker="
            << contactLineWallBoundaryMarker(contact_line)
            << " global_owned_faces=" << evidence.face_count
            << " global_invalid_faces=" << evidence.invalid_face_count
            << " minimum_alignment=" << evidence.minimum_alignment
            << " required_alignment>="
            << FE::Real{1.0} - alignment_tolerance
            << " configured_normal=(" << configured[0] << ','
            << configured[1] << ',' << configured[2] << ')';
        if (!evidence.first_failure.empty()) {
            message << " local_first_failure='" << evidence.first_failure
                    << '\'';
        }
        throw std::invalid_argument(message.str());
    }

    FE::logRecord(post::navierStokesLog("validated complete DynamicContactAngle planar wall marker")
                      .diagnostic("dynamic_contact_complete_wall_marker_validation")
                      .integer("wall_boundary_marker", contactLineWallBoundaryMarker(contact_line))
                      .integer("global_owned_faces", evidence.face_count)
                      .real("minimum_alignment", evidence.minimum_alignment)
                      .real("alignment_tolerance", alignment_tolerance)
                      .text("geometry_contract", "first_order_planar_current_physical"));
}

void validateFreeSurfaceContactGeometryPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    std::span<const PreparedSurfaceIdentity> identities)
{
    const auto* context = system.cutIntegrationContext();
    if (context == nullptr) {
        return;
    }
    for (std::size_t index = 0; index < free_surfaces.size(); ++index) {
        const auto& boundary = free_surfaces[index];
        if (!isUnfittedLevelSet(boundary)) {
            continue;
        }
        for (const auto& contact : identities[index].contacts) {
            const auto& contact_line = boundary.contact_lines[contact.contact_index];
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            static_cast<void>(
                FE::interfaces::certifyContactLineRulesCollectively(
                    system,
                    context,
                    contactLineCheck(
                        contact_line,
                        contact.domain.marker()),
                    true,
                    "free_surface_contact_geometry_preflight"));
        }
    }
}

void validateDynamicContactWallEssentialBC(
    const IncompressibleNavierStokesVMSOptions& options,
    const FreeSurfaceContactLine& contact_line,
    int dim)
{
    const int normal_axis = axisAlignedWallNormalAxis(contact_line, dim);
    const auto law = contactWallLawName(contact_line);
    bool normal_is_constrained = false;
    bool saw_wall_bc = false;
    for (const auto& dirichlet : options.velocity_dirichlet) {
        if (dirichlet.boundary_marker != contactLineWallBoundaryMarker(contact_line)) {
            continue;
        }
        saw_wall_bc = true;
        for (int component = 0; component < dim; ++component) {
            if (!dirichlet.active_components[
                    static_cast<std::size_t>(component)]) {
                continue;
            }
            if (!FE::forms::bc::isZeroConstantScalarValue(
                    dirichlet.value[static_cast<std::size_t>(component)])) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " requires a stationary zero-valued normal wall essential condition");
            }
            if (component == normal_axis) {
                normal_is_constrained = true;
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: " + law +
                    " rejects tangential/full no-slip Dirichlet constraints on its Navier-slip wall");
            }
        }
    }
    for (const auto& dirichlet : options.velocity_dirichlet_weak) {
        if (dirichlet.boundary_marker == contactLineWallBoundaryMarker(contact_line)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " + law +
                " requires a strong zero normal-only wall essential condition; weak velocity Dirichlet data on the wall are unsupported");
        }
    }
    if (!saw_wall_bc || !normal_is_constrained) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + law +
            " requires an axis-aligned zero normal-only velocity Dirichlet condition on Contact_line_wall_marker");
    }
}

} // namespace

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context)
{
    const auto normal = normalizedWallNormal(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    for (int component = dim; component < 3; ++component) {
        if (std::abs(normal[static_cast<std::size_t>(component)]) > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) +
                " wall_normal must lie in the velocity space dimension");
        }
    }
}

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    FE::forms::bc::validateScalarLiteral(
        value, FE::numeric::isFinite,
        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be finite");
}

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    FE::forms::bc::validateScalarLiteral(
        value, FE::numeric::isNonnegative,
        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be nonnegative");
}

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    FE::forms::bc::validateScalarLiteral(
        value, FE::numeric::isPositive,
        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be positive");
}

namespace {

[[nodiscard]] bool formulationOwnsFieldRows(
    const FE::analysis::FormulationRecord& record,
    FE::FieldId field)
{
    for (const auto& [test_field, trial_field] : record.block_couplings) {
        (void)trial_field;
        if (test_field == field) {
            return true;
        }
    }
    // Retain compatibility with formulation metadata produced before
    // block-coupling discovery: a single active field is its row owner.
    return record.block_couplings.empty() &&
           record.active_fields.size() == 1u &&
           record.active_fields.front() == field;
}

[[nodiscard]] std::string owningOperatorTagForField(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::string_view context)
{
    std::optional<std::string> owner;
    for (const auto& record : system.formulationRecords()) {
        if (!formulationOwnsFieldRows(record, field)) {
            continue;
        }
        if (record.operator_tag.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) + " has an empty owning operator tag");
        }
        if (owner.has_value() && *owner != record.operator_tag) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) + " has ambiguous owning operators '" +
                *owner + "' and '" + record.operator_tag + "'");
        }
        owner = record.operator_tag;
    }

    if (owner.has_value()) {
        return *owner;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: " + std::string(context) +
        " has no installed owner formulation; install the level-set equation before the free-surface contact condition");
}

// The lagged normal-increment term is qualified only where its derivation
// holds: an exterior unfitted interface whose generated LinearCorner facets
// carry a piecewise-constant normal, refreshed-frozen geometry inside the
// outer fixed point, and the SurfaceStress energy.  The transient outer
// fixed point and the level-set transport are checked by the application,
// which owns them.
void validateFreeSurfaceSemiImplicitSurfaceTension(
    const FreeSurfaceBoundary& bc)
{
    if (bc.surface_tension_semi_implicit ==
        FreeSurfaceSurfaceTensionSemiImplicit::None) {
        return;
    }
    if (!usesSemiImplicitNormalIncrement(bc)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "Surface_tension_semi_implicit value");
    }
    const auto reject = [](std::string_view requirement) {
        throw std::invalid_argument(
            std::string("IncompressibleNavierStokesVMSModule: "
                        "Surface_tension_semi_implicit=NormalIncrement "
                        "requires ") +
            std::string(requirement));
    };
    if (!isExteriorOnePhaseBoundary(bc) || !isUnfittedLevelSet(bc)) {
        reject("an exterior one-phase UnfittedLevelSet free surface");
    }
    if (!FE::forms::bc::isConstantScalarValue(bc.surface_tension) ||
        !(constantScalarValueOrThrow(
              bc.surface_tension,
              "semi-implicit free-surface surface_tension") >
          FE::Real{0.0})) {
        reject("a literal positive Surface_tension");
    }
    if (bc.active_domain == FreeSurfaceActiveDomain::None) {
        reject("Active_domain=LevelSetNegative or LevelSetPositive");
    }
    if (bc.generated_interface_geometry !=
        FreeSurfaceGeneratedInterfaceGeometry::LinearCorner) {
        reject("Generated_interface_geometry=LinearCorner, whose facet "
               "normal is piecewise constant");
    }
    if (!isRefreshedFrozenGeometryTangent(bc)) {
        reject("Geometry_tangent_policy=RefreshedFrozenQuadrature without "
               "level-set shape tangents");
    }
}

} // namespace

std::optional<ActivePressureDomain> activePressureDomainFor(
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

std::optional<std::array<FE::Real, 3>> fittedTangentialMeasurementVelocity(
    const FreeSurfaceBoundary& boundary)
{
    if (boundary.tangential_mesh_policy != FreeSurfaceTangentialMeshPolicy::Prescribed) {
        return std::nullopt;
    }
    std::array<FE::Real, 3> target{};
    for (std::size_t component = 0; component < target.size(); ++component) {
        target[component] = constantScalarValueOrThrow(
            boundary.prescribed_tangential_mesh_velocity[component],
            "fitted prescribed tangential mesh velocity");
    }
    return target;
}

std::vector<FE::constraints::VertexDirichletValue> nodePressureValues(
    const IncompressibleNavierStokesVMSOptions& options)
{
    std::vector<FE::constraints::VertexDirichletValue> values;
    values.reserve(options.node_pressure_constraints.values.size());
    for (const auto& input : options.node_pressure_constraints.values) {
        values.push_back({input.node_id, input.pressure});
    }
    return values;
}

PreparedPressureInitialization hydrostaticPressureInitialization(
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& law,
    std::span<const FE::constraints::VertexDirichletValue> gauges)
{
    const auto& init = options.hydrostatic_pressure_initialization;
    const auto active_domain = activePressureDomainFor(options.free_surface);
    if (active_domain.has_value() && init.field_name.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: CutVolume active-domain "
            "hydrostatic pressure initialization requires a level-set-aware "
            "mesh vertex field (hydrostatic_pressure_initialization.field_name)");
    }

    FE::level_set::SideVertexInitializationDeclaration declaration{};
    if (active_domain.has_value()) {
        const auto& bc = *active_domain->boundary;
        declaration.support = FE::level_set::LevelSetVertexSupportDeclaration{
            .level_set = {
                .field_name = bc.level_set_field_name,
                .messages = pressureLevelSetMessages(
                    "active-domain hydrostatic pressure initialization")},
            .isovalue = bc.level_set_isovalue,
            .side = pressureSupportSide(active_domain->active_domain),
            .interface_marker = bc.interface_marker,
            .use_generated_volume_rules = true,
            .support_message_prefix =
                "IncompressibleNavierStokesVMSModule: active-domain pressure initialization"};
    }
    if (!init.field_name.empty()) {
        const std::string owner =
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure field initialization";
        declaration.source = FE::systems::RequiredMeshVertexScalarSource{
            .field_name = init.field_name,
            .messages = {
                .no_native_mesh = owner + " requires a native mesh",
                .missing_field_prefix = owner + " could not find vertex field '",
                .not_float64 = owner + " requires a Float64 vertex field",
                .no_components = owner + " requires at least one component",
                .too_few_entries = owner + " has fewer entries than pressure vertices",
                .empty = owner + " found an empty vertex field",
                .no_mesh_support = owner + " requires native mesh support",
                .non_finite_prefix =
                    "IncompressibleNavierStokesVMSModule: hydrostatic pressure mesh vertex field '" +
                    init.field_name + "' contains a non-finite value at vertex "}};
    } else {
        declaration.source = law;
    }
    declaration.unsupported_value = init.reference_pressure;
    declaration.comparisons = {
        .values = gauges,
        .id_mode = nodePressureIdMode(options),
        .messages = pressureConstraintVertexMessages()};
    declaration.messages = {
        .non_finite_prefix =
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization produced a non-finite value at vertex ",
        .non_finite_suffix = init.field_name.empty()
            ? std::string()
            : " from mesh vertex field '" + init.field_name + "'",
        .no_entries =
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization found no pressure vertex DOFs",
        .no_view =
            "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not create a vector view"};
    return {std::move(declaration), active_domain};
}

void validateGeneratedFreeSurfaceMarkerUniqueness(
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system,
    std::span<PreparedSurfaceIdentity> identities)
{
    std::unordered_map<int, std::string> owners;
    const auto register_marker = [&](int marker,
                                     std::string key,
                                     std::string_view kind) {
        const auto [it, inserted] = owners.emplace(marker, std::move(key));
        if (!inserted) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: duplicate/colliding generated " +
                std::string(kind) + " marker " + std::to_string(marker) +
                " between keys '" + it->second +
                "' and another configured free-surface key; assign explicit unique markers");
        }
    };

    for (std::size_t index = 0; index < boundaries.size(); ++index) {
        const auto& bc = boundaries[index];
        if (!isUnfittedLevelSet(bc)) {
            continue;
        }
        auto& identity = identities[index];
        (void)prepareSurfaceIdentity(system, bc, identity);
        const auto phi_id = identity.geometry->levelSetField();
        FE::interfaces::GeneratedInterfaceMarkerKey interface_key{};
        interface_key.source =
            FE::interfaces::LevelSetInterfaceSource::fromField(phi_id);
        interface_key.domain_id = bc.generated_interface_domain_id;
        interface_key.isovalue = bc.level_set_isovalue;
        interface_key.requested_marker = bc.interface_marker;
        register_marker(
            bc.interface_marker,
            "interface:" + interface_key.stableKey(),
            "interface/contact");

        for (const auto& contact : identity.contacts) {
            register_marker(
                contact.domain.marker(),
                "contact:" + contact.key.stableKey(),
                "interface/contact");
        }
    }
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

std::optional<FE::GlobalIndex> pressureConstraintLocalVertex(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    FE::GlobalIndex node_id)
{
    return FE::systems::localVertexOf(
        system,
        nodePressureIdMode(options),
        node_id,
        pressureConstraintVertexMessages());
}

void validateNodePressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options)
{
    std::unordered_map<FE::GlobalIndex, FE::Real> pressure_by_node;
    const auto n_vertices = system.meshAccess().numVertices();
    for (const auto& constraint : options.node_pressure_constraints.values) {
        if (constraint.node_id < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references a negative node id");
        }
        FE::numeric::validate(
            constraint.pressure, FE::numeric::isFinite,
            "IncompressibleNavierStokesVMSModule: pressure constraint value must be finite");
        const auto [existing, inserted] = pressure_by_node.emplace(
            constraint.node_id, constraint.pressure);
        if (!inserted && existing->second != constraint.pressure) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: duplicate pressure constraint node has conflicting values");
        }
        const auto local_vertex = pressureConstraintLocalVertex(
            system, options, constraint.node_id);
        if (local_vertex.has_value() &&
            (*local_vertex < 0 || *local_vertex >= n_vertices)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pressure constraint references vertex " +
                std::to_string(constraint.node_id) +
                " outside the local pressure mesh");
        }
    }
}

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc,
                                 const IncompressibleNavierStokesVMSOptions& options,
                                 bool ale_enabled,
                                 const FE::systems::FESystem& system,
                                 int dim,
                                 PreparedSurfaceIdentity& identity)
{
    if (bc.role !=
            FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary &&
        bc.role !=
            FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface boundary role");
    }
    if (bc.implementation != FreeSurfaceImplementation::FittedALE &&
        bc.implementation !=
            FreeSurfaceImplementation::UnfittedLevelSet) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface implementation");
    }
    const bool supported_boundary_policies =
        (bc.active_domain == FreeSurfaceActiveDomain::None ||
         bc.active_domain ==
             FreeSurfaceActiveDomain::LevelSetNegative ||
         bc.active_domain ==
             FreeSurfaceActiveDomain::LevelSetPositive) &&
        (bc.kinematic_enforcement ==
             FreeSurfaceKinematicEnforcement::None ||
         bc.kinematic_enforcement ==
             FreeSurfaceKinematicEnforcement::MeshNitsche) &&
        (bc.normal_kinematic_policy ==
             FreeSurfaceNormalKinematicPolicy::None ||
         bc.normal_kinematic_policy ==
             FreeSurfaceNormalKinematicPolicy::
                 MatchFluidNormalVelocity) &&
        (bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::Free ||
         bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::SmoothingOnly ||
         bc.tangential_mesh_policy ==
             FreeSurfaceTangentialMeshPolicy::Prescribed);
    if (!supported_boundary_policies) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface boundary policy");
    }
    FE::numeric::validate(
        bc.level_set_isovalue, FE::numeric::isFinite,
        "IncompressibleNavierStokesVMSModule: free-surface level_set_isovalue must be finite");
    validateFiniteConstantScalar(
        bc.external_pressure, "free-surface external_pressure");
    // SurfaceStress contains the constant-gamma surface-energy variation,
    // but its contact wall energy and current input contract also assume one
    // literal gamma.  Keep variable surface chemistry/Marangoni data
    // fail-closed until gamma and the wall-energy law are coupled
    // consistently throughout.
    (void)constantScalarValueOrThrow(
        bc.surface_tension,
        "free-surface surface_tension (variable surface tension/Marangoni traction is unsupported)");
    validateNonnegativeConstantScalar(
        bc.surface_tension, "free-surface surface_tension");
    validateFreeSurfaceSemiImplicitSurfaceTension(bc);
    validateNonnegativeConstantScalar(
        bc.cut_cell_stabilization.pressure_gradient_penalty,
        "cut-cell pressure-gradient penalty");
    if (!FE::constraints::validSmallCutAggregationGuards(
            bc.small_cut_aggregation_guards)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: small-cut aggregation "
            "guards require a positive root path, finite nonnegative "
            "reference extrapolation distance, finite coefficient and row "
            "L1 limits at least one, and a row L1 limit no smaller than the "
            "coefficient limit");
    }
    for (const auto& value : bc.prescribed_tangential_mesh_velocity) {
        validateFiniteConstantScalar(
            value, "prescribed tangential mesh velocity");
    }
    validatePositiveConstantScalar(
        bc.tangential_mesh_penalty, "tangential mesh penalty");
    if (bc.tangential_mesh_policy !=
        FreeSurfaceTangentialMeshPolicy::Prescribed) {
        for (const auto& value : bc.prescribed_tangential_mesh_velocity) {
            if (!FE::forms::bc::isZeroConstantScalarValue(value)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: Free and "
                    "SmoothingOnly tangential mesh policies cannot carry a "
                    "prescribed tangential velocity");
            }
        }
    }

    if (isInternalMaterialInterfaceVolume(bc)) {
        if (!isUnfittedLevelSet(bc) || ale_enabled ||
            bc.boundary_marker >= 0 || bc.interface_marker < 0 ||
            bc.level_set_field_name.empty() ||
            bc.generated_interface_domain_id.empty() ||
            bc.active_domain == FreeSurfaceActiveDomain::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume requires fixed Eulerian UnfittedLevelSet CutVolume integration on one explicit active side and no exterior boundary marker");
        }
        if (!FE::forms::bc::isZeroConstantScalarValue(
                bc.external_pressure) ||
            !FE::forms::bc::isZeroConstantScalarValue(
                bc.surface_tension)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume cannot own exterior pressure, surface tension, curvature, or an interface-traction representation");
        }
        if (bc.kinematic_enforcement !=
                FreeSurfaceKinematicEnforcement::None ||
            bc.normal_kinematic_policy !=
                FreeSurfaceNormalKinematicPolicy::None ||
            bc.tangential_mesh_policy !=
                FreeSurfaceTangentialMeshPolicy::Free ||
            !bc.contact_lines.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume cannot own interface kinematics, tangential mesh motion, or contact laws");
        }
        (void)prepareSurfaceIdentity(system, bc, identity);
        const auto phi_id = identity.geometry->levelSetField();
        const auto& phi_record = system.fieldRecord(phi_id);
        if (phi_record.components != 1 || !phi_record.space ||
            phi_record.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume requires a scalar level-set field");
        }
        return;
    }

    if (isUnfittedLevelSet(bc)) {
        if (bc.tangential_mesh_policy !=
            FreeSurfaceTangentialMeshPolicy::SmoothingOnly) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: tangential mesh "
                "policies apply only to fitted-ALE free surfaces");
        }
        if (ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces cannot currently be combined with ALE; the level-set transport must use the relative velocity u-w before this combination is supported");
        }
        if (bc.interface_marker < 0) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires interface_marker >= 0");
        }
        if (bc.level_set_field_name.empty()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted free surface requires a non-empty level_set_field_name");
        }
        if (bc.active_domain == FreeSurfaceActiveDomain::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: UnfittedLevelSet free surfaces require Active_domain=LevelSetNegative or LevelSetPositive");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces use Eulerian level-set transport for kinematics; a mesh kinematic enforcement (MeshNitsche) needs a fitted ALE boundary and is therefore unsupported");
        }
        const bool has_surface_stress_residual =
            !FE::forms::bc::isZeroConstantScalarValue(
                bc.external_pressure) ||
            !FE::forms::bc::isZeroConstantScalarValue(
                bc.surface_tension);
        if (has_surface_stress_residual &&
            !isRefreshedFrozenGeometryTangent(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: SurfaceStress currently requires Geometry_tangent_policy=RefreshedFrozenQuadrature; DifferentiatedQuadrature does not yet contain the complete projector-normal, point-location, and measure derivative");
        }
        const auto& cut = bc.cut_cell_stabilization;
        if (cut.enabled) {
            (void)constantScalarValueOrThrow(
                cut.pressure_gradient_penalty,
                "enabled cut-cell pressure-gradient penalty");
        }
    } else {
        // SurfaceStress is the fitted Automatic form and needs no opt-in
        // (decision D47 Q2(b)).  Its checks apply only when it installs the
        // surface energy (gamma != 0); with gamma = 0 it is the exterior
        // pressure load alone.
        if (surfaceStressActive(bc)) {
            // The fitted Laplace-Beltrami form gamma (I - n n) : grad(v)
            // is evaluated in the assembly (active) frame with the current
            // boundary normal.  The two frames coincide for a static mesh
            // and for coupled mesh displacement, whose nonlinear geometry
            // transaction moves the active coordinates to the current
            // configuration at every trial state.  Prescribed mesh motion
            // may assemble on the reference configuration, where grad(v)
            // would not be the current-frame gradient.
            if (ale_enabled &&
                options.mesh_velocity_source !=
                    ALEMeshVelocitySource::CoupledDisplacement) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted SurfaceStress requires a static mesh or ALE with a coupled mesh-displacement unknown, so that the assembly frame is the current configuration");
            }
            (void)constantScalarValueOrThrow(
                bc.surface_tension, "fitted SurfaceStress surface_tension");
            if (!bc.contact_lines.empty()) {
                for (const auto& contact : bc.contact_lines) {
                    if (contactLineKind(contact) != ContactLineKind::None) {
                        throw std::invalid_argument(
                            "IncompressibleNavierStokesVMSModule: fitted SurfaceStress is admitted only without a fitted contact-line model; its conormal line force would need a codimension-two integration entity");
                    }
                }
            }
        }
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
        // The qualified fitted-ALE contract, the only fitted configuration
        // (the schema-1 legacy mode that waived it was retired).
        if (!ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: the qualified "
                "fitted-ALE free-surface contract requires ALE to be "
                "enabled");
        }
        if (options.mesh_velocity_source !=
            ALEMeshVelocitySource::CoupledDisplacement) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: the qualified "
                "fitted-ALE free-surface contract requires mesh velocity "
                "derived from a coupled mesh-displacement unknown");
        }
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::
                MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: the qualified "
                "fitted-ALE free-surface contract requires "
                "MatchFluidNormalVelocity");
        }
        if (bc.kinematic_enforcement !=
            FreeSurfaceKinematicEnforcement::MeshNitsche) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: the qualified "
                "fitted-ALE free-surface contract requires MeshNitsche "
                "normal enforcement; kinematic_enforcement None is not "
                "accepted for a fitted free surface");
        }
        for (int component = dim; component < 3; ++component) {
            if (!FE::forms::bc::isZeroConstantScalarValue(
                    bc.prescribed_tangential_mesh_velocity[
                        static_cast<std::size_t>(component)])) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed "
                    "tangential mesh velocity must lie in the mesh "
                    "dimension");
            }
        }
    }

    if (bc.kinematic_enforcement ==
            FreeSurfaceKinematicEnforcement::MeshNitsche &&
        (!std::isfinite(bc.kinematic_nitsche_gamma) ||
         !(bc.kinematic_nitsche_gamma > FE::Real{0.0}))) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: MeshNitsche free-surface kinematics require a finite positive boundary-local kinematic_nitsche_gamma");
    }

    for (std::size_t i = 0; i < bc.contact_lines.size(); ++i) {
        const auto& candidate = bc.contact_lines[i];
        const auto owns_wall_law = [](ContactLineKind model) {
            return model == ContactLineKind::PrescribedAngle ||
                   model == ContactLineKind::DynamicRenE;
        };
        if (!owns_wall_law(contactLineKind(candidate)) ||
            contactLineWallBoundaryMarker(candidate) < 0) {
            continue;
        }
        for (std::size_t j = i + 1; j < bc.contact_lines.size(); ++j) {
            const auto& other = bc.contact_lines[j];
            if (owns_wall_law(contactLineKind(other)) &&
                contactLineWallBoundaryMarker(other) ==
                    contactLineWallBoundaryMarker(candidate)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: duplicate or conflicting contact-line wall laws on wall marker " +
                    std::to_string(contactLineWallBoundaryMarker(candidate)) +
                    "; exactly one contact-line model may own each wall footprint");
            }
        }
    }

    for (std::size_t contact_index = 0; contact_index < bc.contact_lines.size(); ++contact_index) {
        const auto& contact_line = bc.contact_lines[contact_index];
        if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
            if (isUnfittedLevelSet(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact lines are currently supported only for fitted ALE free surfaces");
            }
            if (contactLineConstraintMarker(contact_line) < 0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: pinned contact line requires contact_line_marker or wall_boundary_marker >= 0");
            }
        }
        if (contactLineKind(contact_line) == ContactLineKind::PrescribedAngle) {
            const auto angle = constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "contact-line contact_angle_radians");
            constexpr FE::Real pi = FE::Real{3.14159265358979323846};
            if (!std::isfinite(angle) || !(angle > FE::Real{0.0}) ||
                !(angle < pi)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: contact-line contact_angle_radians must be finite and strictly in (0, pi); complete-wetting endpoints are not a transverse codimension-two contact configuration");
            }
            if (contactLineHasNavierSlip(contact_line)) {
                if (options.viscosity_model) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed-angle Navier slip requires literal Newtonian viscosity so accepted sharp-wall dissipation can be evaluated from the identical operator stage");
                }
                (void)constantScalarValueOrThrow(
                    contactLineSlipLength(contact_line),
                    "PrescribedContactAngle Navier slip_length");
                validatePositiveConstantScalar(
                    contactLineSlipLength(contact_line),
                    "PrescribedContactAngle Navier slip_length");
            }
            if (isUnfittedLevelSet(bc)) {
                if (bc.generated_interface_geometry !=
                    FreeSurfaceGeneratedInterfaceGeometry::LinearCorner) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles currently support only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
                }
                (void)prepareSurfaceIdentity(system, bc, identity);
                const auto phi_id = identity.geometry->levelSetField();
                const auto& phi_record = system.fieldRecord(phi_id);
                if (phi_record.components != 1 || !phi_record.space ||
                    phi_record.space->value_dimension() != 1) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact-angle level-set field must be scalar");
                }
                if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown so its normal is coupled");
                }
                (void)owningOperatorTagForField(
                    system,
                    phi_id,
                    "prescribed contact-angle level-set field '" +
                        bc.level_set_field_name + "'");
                if (phi_record.space->polynomial_order() != 1) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles currently require an order-1 level-set space; generated contact geometry is reconstructed from linear corner values");
                }
                if (phi_record.space->continuity() != FE::Continuity::C0) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require a continuous C0 level-set space; discontinuous element-local corner values do not define one wall contact geometry");
                }
                const int generated_marker =
                    prepareSurfaceIdentity(
                        system, bc, identity, contact_index);
                if (contactLineMarker(contact_line) >= 0 &&
                    contactLineMarker(contact_line) != generated_marker) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angle uses a generated interface-boundary marker; Contact_line_marker must be omitted or match the generated marker " +
                        std::to_string(generated_marker));
                }
                validateWallNormalDimension(
                    contact_line,
                    dim,
                    "prescribed contact angle");
                // Decision D4: the Young term in momentum is the only
                // contact-angle mechanism.  On a no-slip wall the velocity
                // test functions vanish, so that term does no work and
                // nothing would impose the angle; the contact line moves
                // through Navier slip on the wetted wall, held by a strong
                // normal-only condition, exactly as for DynamicContactAngle.
                if (!contactLineHasNavierSlip(contact_line)) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: unfitted PrescribedContactAngle requires Navier slip on the wetted wall (Wall_slip_model=Navier with a positive literal Wall_slip_length) and a stationary zero normal-only strong velocity condition on Contact_line_wall_marker, as DynamicContactAngle; on a no-slip wall the Young term does no work and nothing imposes the contact angle (decision D4)");
                }
                validateDynamicContactWallEssentialBC(
                    options, contact_line, dim);
                validateDynamicContactPlanarWallMarker(
                    system, contact_line);
            } else {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles are unsupported until a true fitted contact-line (codimension-two) integration entity is available; the condition must not be integrated over the complete free-surface boundary");
            }
        }
        if (contactLineKind(contact_line) == ContactLineKind::DynamicRenE) {
            if (options.viscosity_model) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires literal Newtonian viscosity so accepted sharp-wall slip dissipation can be evaluated from the identical operator stage; constitutive viscosity models are unsupported for this contact law");
            }
            if (!isUnfittedLevelSet(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle is currently supported only for sharp unfitted level-set free surfaces");
            }
            if (bc.generated_interface_geometry !=
                FreeSurfaceGeneratedInterfaceGeometry::LinearCorner) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently supports only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
            }

            (void)prepareSurfaceIdentity(system, bc, identity);
            const auto phi_id = identity.geometry->levelSetField();
            const auto& phi_record = system.fieldRecord(phi_id);
            if (phi_record.components != 1 || !phi_record.space ||
                phi_record.space->value_dimension() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle level-set field must be scalar");
            }
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires the level-set field to be an unknown so the authoritative interface, contact, and sharp wetted-wall geometry can be refreshed from the accepted state");
            }
            if (phi_record.space->polynomial_order() != 1) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently requires an order-1 level-set space; high-order contact-line geometry is unsupported");
            }
            if (phi_record.space->continuity() != FE::Continuity::C0) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires a continuous C0 level-set space; discontinuous element-local corner values do not define one wall contact geometry");
            }

            const auto angle = constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "DynamicContactAngle equilibrium contact_angle_radians");
            constexpr FE::Real pi = FE::Real{3.14159265358979323846};
            if (!std::isfinite(angle) || !(angle > FE::Real{0.0}) ||
                !(angle < pi)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle equilibrium contact_angle_radians must be finite and strictly in (0, pi); complete-wetting endpoints make the footprint direction singular");
            }
            const auto equilibrium_transverse_sine = std::sin(angle);
            if (!std::isfinite(equilibrium_transverse_sine) ||
                equilibrium_transverse_sine <
                    kMinimumContactTransverseSine) {
                std::ostringstream message;
                message
                    << "IncompressibleNavierStokesVMSModule: "
                       "DynamicContactAngle equilibrium angle is too close "
                       "to a complete-wetting endpoint for the supported "
                       "transverse generated contact geometry"
                    << " contact_angle_radians=" << angle
                    << " sin(theta_e)=" << equilibrium_transverse_sine
                    << " required_sin(theta_e)>="
                    << kMinimumContactTransverseSine
                    << " minimum_transverse_sine="
                    << kMinimumContactTransverseSine;
                throw std::invalid_argument(message.str());
            }
            (void)constantScalarValueOrThrow(
                bc.surface_tension,
                "DynamicContactAngle surface_tension");
            validatePositiveConstantScalar(
                bc.surface_tension,
                "DynamicContactAngle surface_tension");
            (void)constantScalarValueOrThrow(
                contactLineMobility(contact_line),
                "DynamicContactAngle contact-line mobility");
            validatePositiveConstantScalar(
                contactLineMobility(contact_line),
                "DynamicContactAngle contact-line mobility");
            (void)constantScalarValueOrThrow(
                contactLineSlipLength(contact_line),
                "DynamicContactAngle Navier slip_length");
            validatePositiveConstantScalar(
                contactLineSlipLength(contact_line),
                "DynamicContactAngle Navier slip_length");

            const int generated_marker = prepareSurfaceIdentity(
                system, bc, identity, contact_index);
            if (contactLineMarker(contact_line) >= 0 &&
                contactLineMarker(contact_line) != generated_marker) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle uses a generated interface-boundary marker; Contact_line_marker must be omitted or match the generated marker " +
                    std::to_string(generated_marker));
            }
            (void)normalizedWallNormal(contact_line);
            validateDynamicContactWallEssentialBC(options, contact_line, dim);
            validateDynamicContactPlanarWallMarker(system, contact_line);

        }
    }
}

void validateNavierStokesBoundaryConfiguration(
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    const FE::spaces::FunctionSpace& velocity_space,
    const FE::spaces::FunctionSpace& pressure_space,
    int dim,
    const std::function<std::optional<int>(int)>& generated_active_boundary_for,
    std::vector<PreparedWeakVelocityBoundary>* prepared_weak_velocity,
    std::vector<FE::forms::BoundaryDomain>* required_routes)
{
    using FE::forms::FormExpr;

    bool uses_sharp_active_boundary_form = false;
    const auto inspect_boundary_markers =
        [&](const auto& conditions) {
            for (const auto& condition : conditions) {
                uses_sharp_active_boundary_form =
                    uses_sharp_active_boundary_form ||
                    generated_active_boundary_for(
                        condition.boundary_marker).has_value();
            }
        };
    inspect_boundary_markers(options.traction_neumann);
    inspect_boundary_markers(options.traction_robin);
    inspect_boundary_markers(options.pressure_outflow);
    inspect_boundary_markers(options.coupled_outflow_rcr);
    inspect_boundary_markers(options.coupled_outflow_rcrcr);
    inspect_boundary_markers(options.velocity_dirichlet_weak);
    for (const auto& free_surface : free_surfaces) {
        if (!isUnfittedLevelSet(free_surface) ||
            free_surface.active_domain == FreeSurfaceActiveDomain::None) {
            continue;
        }
        uses_sharp_active_boundary_form =
            uses_sharp_active_boundary_form ||
            std::any_of(
                free_surface.contact_lines.begin(),
                free_surface.contact_lines.end(),
                [](const auto& contact_line) {
                    return contactLineHasNavierSlip(contact_line);
                });
    }
    if (uses_sharp_active_boundary_form) {
        if (velocity_space.polynomial_order() != 1 ||
            pressure_space.polynomial_order() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: sharp generated exterior-boundary forms currently require order-1 velocity and pressure spaces; high-order active-boundary quadrature and entity measures are not qualified");
        }
        for (const auto& free_surface : free_surfaces) {
            if (isUnfittedLevelSet(free_surface) &&
                free_surface.active_domain != FreeSurfaceActiveDomain::None &&
                free_surface.generated_interface_geometry !=
                    FreeSurfaceGeneratedInterfaceGeometry::LinearCorner) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: sharp generated exterior-boundary forms currently require Generated_interface_geometry=LinearCorner; high-order active-boundary quadrature and entity measures are not qualified");
            }
        }
    }

    FE::systems::BoundaryConditionManager velocity_conditions;
    velocity_conditions.install(options.traction_neumann, [&](const auto& bc) {
        for (const auto& value : bc.traction) {
            validateFiniteConstantScalar(value, "traction value");
        }
        return Factories::toTractionBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    velocity_conditions.install(options.traction_robin, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.alpha, "Robin coefficient");
        for (const auto& value : bc.rhs) {
            validateFiniteConstantScalar(value, "Robin right-hand side");
        }
        return Factories::toTractionRobinBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });

    const auto u = FE::forms::StateField(
        FE::FieldId{0}, velocity_space, "u_boundary_preflight");
    const auto p = FE::forms::StateField(
        FE::FieldId{1}, pressure_space, "p_boundary_preflight");
    const auto v = FE::forms::TestField(
        FE::FieldId{0}, velocity_space, "v_boundary_preflight");
    const auto q = FE::forms::TestField(
        FE::FieldId{1}, pressure_space, "q_boundary_preflight");
    const auto rho = FormExpr::constant(options.density);

    velocity_conditions.install(options.pressure_outflow, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.pressure, "outflow pressure");
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "outflow backflow coefficient");
        return Factories::toOutflowBC(
            bc,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });

    for (const auto& bc : options.coupled_outflow_rcr) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule coupled RCR preflight");
        for (const auto [value, label] : {
                 std::pair{bc.Rp, std::string_view{"RCR Rp"}},
                 std::pair{bc.C, std::string_view{"RCR C"}},
                 std::pair{bc.Rd, std::string_view{"RCR Rd"}},
                 std::pair{bc.Pd, std::string_view{"RCR Pd"}},
                 std::pair{bc.X0, std::string_view{"RCR initial state"}}}) {
            FE::numeric::validate(
                value, FE::numeric::isFinite,
                "IncompressibleNavierStokesVMSModule: " +
                    std::string(label) + " must be finite");
        }
        if (bc.Rd == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCROutflowBC: Rd must be nonzero");
        }
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "RCR backflow coefficient");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }
    for (const auto& bc : options.coupled_outflow_rcrcr) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule coupled RCRCR preflight");
        for (const auto [value, label] : {
                 std::pair{bc.Rp, std::string_view{"RCRCR Rp"}},
                 std::pair{bc.C1, std::string_view{"RCRCR C1"}},
                 std::pair{bc.Rm, std::string_view{"RCRCR Rm"}},
                 std::pair{bc.C2, std::string_view{"RCRCR C2"}},
                 std::pair{bc.Rd, std::string_view{"RCRCR Rd"}},
                 std::pair{bc.Pd, std::string_view{"RCRCR Pd"}},
                 std::pair{bc.P10, std::string_view{"RCRCR P1 initial state"}},
                 std::pair{bc.P20, std::string_view{"RCRCR P2 initial state"}}}) {
            FE::numeric::validate(
                value, FE::numeric::isFinite,
                "IncompressibleNavierStokesVMSModule: " +
                    std::string(label) + " must be finite");
        }
        if (bc.C1 == FE::Real{0.0} || bc.C2 == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCRCROutflowBC: C1 and C2 must be nonzero");
        }
        if (bc.Rm == FE::Real{0.0} || bc.Rd == FE::Real{0.0}) {
            throw std::invalid_argument(
                "CoupledRCRCROutflowBC: Rm and Rd must be nonzero");
        }
        validateNonnegativeConstantScalar(
            bc.backflow_beta, "RCRCR backflow coefficient");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }

    velocity_conditions.install(options.velocity_dirichlet, [&](const auto& bc) {
        for (const auto& value : bc.value) {
            validateFiniteConstantScalar(value, "velocity Dirichlet value");
        }
        return Factories::toVelocityEssentialBC(
            bc, dim, options.velocity_field_name);
    });
    for (const auto& bc : options.velocity_dirichlet_weak) {
        for (const auto& value : bc.value) {
            validateFiniteConstantScalar(
                value, "weak velocity Dirichlet value");
        }
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc,
            "IncompressibleNavierStokesVMSModule weak velocity preflight");
        velocity_conditions.add(
            std::make_unique<FE::forms::bc::ReservedBC>(marker));
    }
    for (const auto& bc : free_surfaces) {
        if (bc.implementation == FreeSurfaceImplementation::FittedALE) {
            velocity_conditions.add(
                std::make_unique<FE::forms::bc::ReservedBC>(
                    bc.boundary_marker));
        }
    }
    velocity_conditions.validate();

    FE::systems::BoundaryConditionManager pressure_conditions;
    pressure_conditions.install(options.pressure_dirichlet, [&](const auto& bc) {
        validateFiniteConstantScalar(bc.value, "pressure Dirichlet value");
        return Factories::toPressureEssentialBC(
            bc, options.pressure_field_name);
    });
    pressure_conditions.validate();

    FormExpr momentum_preflight;
    FormExpr continuity_preflight;
    std::vector<PreparedWeakVelocityBoundary> prepared_boundaries;
    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>
        generated_nitsche_trace_binding_preflight;
    Factories::applyVelocityNitscheBCs(
        momentum_preflight,
        continuity_preflight,
        options,
        dim,
        u,
        p,
        v,
        q,
        FormExpr::constant(options.viscosity),
        generated_active_boundary_for,
        &generated_nitsche_trace_binding_preflight,
        nullptr,
        nullptr,
        &prepared_boundaries);
    std::vector<FE::forms::BoundaryDomain> routes;
    for (const auto& boundary : prepared_boundaries) {
        if (boundary.generated_marker.has_value()) {
            routes.push_back(FE::forms::BoundaryDomain::generatedActiveSubset(
                boundary.physical_marker, *boundary.generated_marker));
        }
    }
    FE::systems::requireGeneratedBoundaryNitscheTraceBindings(
        routes, generated_nitsche_trace_binding_preflight,
        "IncompressibleNavierStokesVMSModule: generated-boundary "
        "Nitsche binding preflight count differs from the configured "
        "production routes");
    if (required_routes != nullptr) {
        *required_routes = std::move(routes);
    }
    if (prepared_weak_velocity != nullptr) {
        *prepared_weak_velocity = std::move(prepared_boundaries);
    }
}

NavierStokesPreflight preflightNavierStokes(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& pressure_space)
{
    if (options.mesh_velocity_source !=
            ALEMeshVelocitySource::PrescribedData &&
        options.mesh_velocity_source !=
            ALEMeshVelocitySource::CoupledDisplacement) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported ALE mesh-velocity source");
    }
    if (freeSurfacePhysicalModelName(options.free_surface_physical_model) ==
        nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "unsupported_free_surface_physical_model");
    }
    if (!velocity_space) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null velocity_space");
    }
    if (!pressure_space) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null pressure_space");
    }
    if (options.operator_tag.empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: operator_tag must be non-empty");
    }

    const int dim = velocity_space->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: velocity space must have 1..3 components");
    }
    if (options.rotating_frame_coriolis_enabled && dim != 3) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: rotating-frame Coriolis forcing requires a 3D velocity space");
    }
    if (pressure_space->value_dimension() != 1) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: pressure space must be scalar");
    }
    FE::numeric::validate(
        options.density, FE::numeric::isPositive,
        "IncompressibleNavierStokesVMSModule::registerOn: density must be > 0");
    if (options.viscosity_model == nullptr) {
        FE::numeric::validate(
            options.viscosity, FE::numeric::isPositive,
            "IncompressibleNavierStokesVMSModule::registerOn: viscosity must be > 0 when viscosity_model is not provided");
    }
    const bool enable_vms = options.enable_vms;
    if (enable_vms) {
        FE::numeric::validate(
            options.stabilization_epsilon, FE::numeric::isPositive,
            "IncompressibleNavierStokesVMSModule::registerOn: stabilization_epsilon must be > 0 when VMS is enabled");
    }

    FE::systems::FieldSpec u_spec;
    u_spec.name = options.velocity_field_name;
    u_spec.space = velocity_space;
    u_spec.components = dim;

    FE::systems::FieldSpec p_spec;
    p_spec.name = options.pressure_field_name;
    p_spec.space = pressure_space;
    p_spec.components = 1;

    std::optional<FE::systems::FieldSpec> body_force_spec;
    if (!options.body_force_field_name.empty()) {
        FE::systems::FieldSpec source_spec;
        source_spec.name = options.body_force_field_name;
        source_spec.space = velocity_space;
        source_spec.components = dim;
        source_spec.source_kind =
            FE::systems::FieldSourceKind::PrescribedData;
        body_force_spec = std::move(source_spec);
    }

    const FE::systems::ALEBindingOptions ale_options{
        .enabled = options.enable_ale,
        .dimension = dim,
        .mesh_velocity_source =
            options.mesh_velocity_source ==
                    ALEMeshVelocitySource::CoupledDisplacement
                ? FE::systems::ALEMeshVelocitySource::CoupledDisplacement
                : FE::systems::ALEMeshVelocitySource::PrescribedData,
        .geometry_tangent_path = options.moving_mesh_tangent_path,
        .mesh_velocity_field_name = options.mesh_velocity_field_name,
        .mesh_displacement_field_name =
            options.mesh_displacement_field_name,
        .mesh_velocity_space = options.mesh_velocity_space
            ? options.mesh_velocity_space
            : velocity_space,
        .mesh_displacement_space = velocity_space,
        .auto_register_mesh_velocity_field =
            options.auto_register_mesh_velocity_field,
        .auto_register_mesh_displacement_field =
            options.auto_register_mesh_displacement_field,
    };

    // Preflight every free-surface field, policy, ownership relation, and
    // generated-marker request before adding Navier--Stokes fields, forms, or
    // constraints.  A rejected configuration must leave the system definition
    // unchanged.
    std::vector<FreeSurfaceBoundary> effective_free_surfaces;
    effective_free_surfaces.reserve(options.free_surface.size());
    std::vector<PreparedSurfaceIdentity> surface_identities;
    surface_identities.reserve(options.free_surface.size());
    std::set<int> prospective_fitted_normal_markers;
    for (const auto& bc : options.free_surface) {
        auto effective_bc = bc;
        PreparedSurfaceIdentity identity;
        if (isUnfittedLevelSet(bc) && bc.interface_marker < 0) {
            if (bc.generated_interface_domain_id.empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: generated unfitted free surface requires a non-empty generated_interface_domain_id");
            }
            effective_bc.interface_marker = prepareSurfaceIdentity(system, bc, identity);
        }
        validateFreeSurfaceBoundary(effective_bc,
                                    options,
                                    options.enable_ale,
                                    system,
                                    dim,
                                    identity);
        // A validated fitted boundary has ALE with a coupled
        // mesh-displacement unknown.
        if (effective_bc.implementation ==
                FreeSurfaceImplementation::FittedALE) {
            if (!system.meshTangentialBoundaryPolicyHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policies cannot be installed "
                    "after accepted tangential-policy history has begun");
            }
            auto displacement_field = system.meshMotionField(
                FE::systems::MeshMotionFieldRole::Displacement);
            if (!displacement_field.has_value()) {
                const auto named = system.findFieldByName(
                    options.mesh_displacement_field_name);
                if (named != FE::INVALID_FIELD_ID) {
                    displacement_field = named;
                }
            }
            const bool installs_normal_relation =
                effective_bc.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                effective_bc.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (installs_normal_relation &&
                !system.meshNormalBoundaryConstraintHistory().empty()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface normal relations cannot be installed "
                    "after accepted normal-constraint history has begun");
            }
            if (installs_normal_relation &&
                !prospective_fitted_normal_markers
                     .insert(effective_bc.boundary_marker)
                     .second) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: duplicate fitted "
                    "free-surface normal relation on marker " +
                    std::to_string(effective_bc.boundary_marker));
            }
            if (installs_normal_relation &&
                effective_bc.kinematic_enforcement ==
                    FreeSurfaceKinematicEnforcement::MeshNitsche &&
                displacement_field.has_value()) {
                // The mesh-motion module adds the Nitsche consistency term of
                // its own operator on the boundaries this relation declares,
                // so it must register after the declaration exists.
                const bool mesh_operator_installed = std::any_of(
                    system.formulationRecords().begin(),
                    system.formulationRecords().end(),
                    [&](const auto& record) {
                        const bool volume = std::find(
                            record.active_domains.begin(),
                            record.active_domains.end(),
                            FE::analysis::DomainKind::Cell) !=
                            record.active_domains.end();
                        const bool mesh_rows = std::any_of(
                            record.block_couplings.begin(),
                            record.block_couplings.end(),
                            [&](const auto& coupling) {
                                return coupling.first == *displacement_field &&
                                       coupling.second == *displacement_field;
                            });
                        return volume && mesh_rows;
                    });
                if (mesh_operator_installed) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: MeshNitsche "
                        "free-surface kinematics on marker " +
                        std::to_string(effective_bc.boundary_marker) +
                        " require the fluid equation to be registered before "
                        "the mesh-motion equation, which adds the Nitsche "
                        "consistency term of its operator on the declared "
                        "boundary");
                }
            }
            if (installs_normal_relation &&
                displacement_field.has_value()) {
                const auto& displacement_record =
                    system.fieldRecord(*displacement_field);
                if (displacement_record.space == nullptr ||
                    velocity_space->polynomial_order() >
                        displacement_record.space->polynomial_order()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted-ALE "
                        "normal measurement quadrature requires velocity "
                        "order " +
                        std::to_string(
                            velocity_space->polynomial_order()) +
                        " <= displacement-primary order " +
                        (displacement_record.space != nullptr
                             ? std::to_string(
                                   displacement_record.space
                                       ->polynomial_order())
                             : std::string("unavailable")) +
                        " on marker " +
                        std::to_string(
                            effective_bc.boundary_marker));
                }
                const auto normal_conflict = std::find_if(
                    system.meshNormalBoundaryConstraints().begin(),
                    system.meshNormalBoundaryConstraints().end(),
                    [&](const auto& declaration) {
                        return declaration.boundary_marker ==
                                   effective_bc.boundary_marker &&
                               declaration.mesh_displacement_field ==
                                   *displacement_field;
                    });
                if (normal_conflict !=
                    system.meshNormalBoundaryConstraints().end()) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: fitted "
                        "free-surface normal relation on marker " +
                        std::to_string(effective_bc.boundary_marker) +
                        " conflicts with existing mesh-motion owner '" +
                        normal_conflict->owner_component + "'");
                }
            }
            const auto conflict = std::find_if(
                system.meshTangentialBoundaryPolicies().begin(),
                system.meshTangentialBoundaryPolicies().end(),
                [&](const auto& declaration) {
                    return declaration.boundary_marker ==
                               effective_bc.boundary_marker &&
                           (!displacement_field.has_value() ||
                            declaration.mesh_displacement_field ==
                                *displacement_field);
                });
            if (conflict !=
                system.meshTangentialBoundaryPolicies().end()) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface tangential policy on marker " +
                    std::to_string(effective_bc.boundary_marker) +
                    " conflicts with existing mesh-motion owner '" +
                    conflict->owner_component + "'");
            }
        }
        effective_free_surfaces.push_back(std::move(effective_bc));
        surface_identities.push_back(std::move(identity));
    }
    std::optional<FE::systems::FieldSpec> semi_implicit_reference_spec;
    if (std::any_of(effective_free_surfaces.begin(),
                    effective_free_surfaces.end(),
                    usesSemiImplicitNormalIncrement)) {
        if (!FE::analysis::isCertifiedGeneratedBoundaryTraceSpace(*velocity_space, dim)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: Surface_tension_semi_implicit=NormalIncrement requires an affine P1 Product H1 velocity space on Triangle3 or Tetra4 cells");
        }
        FE::systems::FieldSpec reference_spec;
        reference_spec.name =
            std::string(kFreeSurfaceSemiImplicitReferenceVelocityFieldName);
        reference_spec.space = velocity_space;
        reference_spec.components = dim;
        reference_spec.source_kind =
            FE::systems::FieldSourceKind::PrescribedData;
        FE::systems::FieldBinding::check(
            system,
            reference_spec,
            FE::systems::FieldSourceKind::PrescribedData,
            /*allow_missing=*/true,
            "IncompressibleNavierStokesVMSModule::registerOn semi-implicit reference velocity");
        semi_implicit_reference_spec = std::move(reference_spec);
    }
    validateGeneratedFreeSurfaceMarkerUniqueness(
        effective_free_surfaces, system, surface_identities);
    validateNodePressureConstraints(system, options);
    auto node_pressure_values = nodePressureValues(options);
    validateActiveDomainPressureValues(
        system,
        options,
        effective_free_surfaces,
        node_pressure_values);
    const bool has_exterior_free_surface = std::any_of(
        effective_free_surfaces.begin(),
        effective_free_surfaces.end(),
        isExteriorOnePhaseBoundary);
    const auto active_pressure_domain =
        activePressureDomainFor(effective_free_surfaces);
    const std::optional<std::size_t> active_boundary_index = active_pressure_domain
        ? std::optional<std::size_t>(static_cast<std::size_t>(
              active_pressure_domain->boundary - effective_free_surfaces.data()))
        : std::nullopt;
    if (!(options
              .generated_boundary_nitsche_minimum_energy_ratio >
          FE::Real{0.0}) ||
        !(options
              .generated_boundary_nitsche_minimum_energy_ratio <
          FE::Real{1.0}) ||
        !std::isfinite(
            options
                .generated_boundary_nitsche_minimum_energy_ratio)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "generated_boundary_nitsche_minimum_energy_ratio must be "
            "finite and strictly between zero and one");
    }
    if (!options.velocity_dirichlet_weak.empty() &&
        (!(options.nitsche_gamma > FE::Real{0.0}) ||
         !std::isfinite(options.nitsche_gamma))) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule::registerOn: "
            "nitsche_gamma must be finite and > 0 when weak velocity "
            "Dirichlet conditions are configured");
    }
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const bool native_mesh_available_for_trace_certificate =
        system.mesh() != nullptr;
#else
    const bool native_mesh_available_for_trace_certificate =
        false;
#endif
    std::vector<FE::level_set::CutSide> active_sides;
    if (active_boundary_index.has_value()) {
        const auto index = *active_boundary_index;
        active_sides.push_back(surface_identities[index].geometry->side(
            activeDomainSide(effective_free_surfaces[index].active_domain)));
    }
    auto generated_active_boundary_for = FE::level_set::generatedActiveBoundaryResolver(
        std::move(active_sides), "IncompressibleNavierStokesVMSModule",
        "IncompressibleNavierStokesVMSModule: sharp active-boundary marker requires a resolved unfitted active domain and physical boundary marker");
    const bool has_generated_boundary_nitsche_route =
        std::any_of(
            options.velocity_dirichlet_weak.begin(),
            options.velocity_dirichlet_weak.end(),
            [&](const auto& bc) {
                return generated_active_boundary_for(
                           FE::forms::bc::detail::
                               boundaryMarkerOrThrow(
                                   bc,
                                   "IncompressibleNavierStokesVMSModule::"
                                   "registerOn generated-boundary "
                                   "Nitsche trace preflight"))
                    .has_value();
            });

    FE::systems::FieldBinding::requireDistinctNames(
        "IncompressibleNavierStokesVMSModule::registerOn", {
        {"velocity", u_spec.name},
        {"pressure", p_spec.name},
        {"body-force data",
         body_force_spec.has_value() ? std::string_view(body_force_spec->name)
                                     : std::string_view{}},
        {"mesh velocity",
         options.enable_ale
             ? std::string_view(ale_options.mesh_velocity_field_name)
             : std::string_view{}},
        {"mesh displacement",
         options.enable_ale &&
                 ale_options.mesh_velocity_source ==
                     FE::systems::ALEMeshVelocitySource::CoupledDisplacement
             ? std::string_view(ale_options.mesh_displacement_field_name)
             : std::string_view{}},
        {"semi-implicit reference velocity",
         semi_implicit_reference_spec.has_value()
             ? std::string_view(semi_implicit_reference_spec->name)
             : std::string_view{}},
    });
    FE::systems::FieldBinding::check(
        system,
        u_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn velocity");
    FE::systems::FieldBinding::check(
        system,
        p_spec,
        FE::systems::FieldSourceKind::Unknown,
        /*allow_missing=*/true,
        "IncompressibleNavierStokesVMSModule::registerOn pressure");
    if (body_force_spec.has_value()) {
        FE::systems::FieldBinding::check(
            system,
            *body_force_spec,
            FE::systems::FieldSourceKind::PrescribedData,
            options.auto_register_body_force_field,
            "IncompressibleNavierStokesVMSModule::registerOn momentum source");
    }
    FE::systems::validateALEBinding(system, ale_options);
    if (ale_options.mesh_velocity_source !=
        FE::systems::ALEMeshVelocitySource::CoupledDisplacement) {
        for (const auto& bc : effective_free_surfaces) {
            for (const auto& contact_line : bc.contact_lines) {
                if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require ALE mesh velocity to be derived from a coupled mesh displacement unknown");
                }
            }
        }
    }
    std::vector<PreparedWeakVelocityBoundary> prepared_weak_velocity_boundaries;
    std::vector<FE::forms::BoundaryDomain> required_nitsche_routes;
    validateNavierStokesBoundaryConfiguration(
        options,
        effective_free_surfaces,
        system,
        *velocity_space,
        *pressure_space,
        dim,
        generated_active_boundary_for,
        &prepared_weak_velocity_boundaries,
        &required_nitsche_routes);
    if (has_generated_boundary_nitsche_route) {
        if (!active_pressure_domain.has_value() ||
            !active_pressure_domain->boundary
                 ->small_cut_aggregation) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require CutVolume "
                "small-cut aggregation and aggregate-trace certification");
        }
        if (!native_mesh_available_for_trace_certificate) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "generated-boundary Nitsche routes require a native mesh "
                "for aggregate-trace certification");
        }
        if (options.viscosity_model != nullptr ||
            !(options.viscosity > FE::Real{0.0}) ||
            !std::isfinite(options.viscosity)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "finite positive constant Newtonian viscosity");
        }
        const bool supported_trace_space =
            FE::analysis::isCertifiedGeneratedBoundaryTraceSpace(
                *velocity_space, dim);
        const auto existing_velocity =
            system.findFieldByName(
                options.velocity_field_name);
        bool supported_existing_velocity = true;
        if (existing_velocity != FE::INVALID_FIELD_ID) {
            const auto& record =
                system.fieldRecord(existing_velocity);
            supported_existing_velocity =
                record.space &&
                record.components == dim &&
                FE::analysis::isCertifiedGeneratedBoundaryTraceSpace(
                    *record.space, dim);
        }
        if (!supported_trace_space) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "certified generated-boundary Nitsche routes require "
                "an affine P1 Product H1 velocity space on Triangle3 "
                "or Tetra4 cells");
        }
        if (!supported_existing_velocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: "
                "the existing velocity field is outside the affine P1 "
                "Product H1 Triangle3/Tetra4 generated-boundary trace "
                "certificate envelope");
        }
    }
    validateFreeSurfaceContactGeometryPreflight(
        effective_free_surfaces, system, surface_identities);
    post::validateFreeSurfaceDiscreteFunctionalPreflight(
        effective_free_surfaces, system);

    std::vector<PreparedFittedNormalBoundary> fitted_normal_boundaries;
    for (std::size_t index = 0; index < effective_free_surfaces.size(); ++index) {
        const auto& boundary = effective_free_surfaces[index];
        if (boundary.implementation != FreeSurfaceImplementation::FittedALE ||
            boundary.kinematic_enforcement == FreeSurfaceKinematicEnforcement::None ||
            boundary.normal_kinematic_policy !=
                FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            continue;
        }
        fitted_normal_boundaries.push_back({
            .boundary_index = index,
            .binding = {
                .mesh_row_source =
                    "Fitted free-surface mesh normal kinematic row on marker " +
                    std::to_string(boundary.boundary_marker),
                .fluid_row_source = fittedFluidNormalDescriptorSource(boundary)}});
    }
    auto bulk_volume = FE::forms::VolumeDomain::full();
    std::optional<FE::LogRecord> active_domain_log;
    if (active_boundary_index.has_value()) {
        const auto index = *active_boundary_index;
        const auto& boundary = effective_free_surfaces[index];
        const auto side = activeDomainSide(boundary.active_domain);
        bulk_volume = surface_identities[index].geometry->side(side).volume();
        active_domain_log = post::navierStokesLog("active-domain free surface")
            .diagnostic("active_domain_free_surface")
            .integer("marker", boundary.interface_marker)
            .quoted("level_set_field", boundary.level_set_field_name)
            .real("isovalue", boundary.level_set_isovalue)
            .quoted("generated_interface_domain_id", boundary.generated_interface_domain_id)
            .text("active_domain", activeDomainName(boundary.active_domain))
            .text("active_domain_method", "CutVolume")
            .text("side", cutVolumeSideName(side));
    }

    return NavierStokesPreflight{
        .dim = dim,
        .enable_vms = enable_vms,
        .u_spec = std::move(u_spec),
        .p_spec = std::move(p_spec),
        .body_force_spec = std::move(body_force_spec),
        .semi_implicit_reference_spec = std::move(semi_implicit_reference_spec),
        .ale_options = ale_options,
        .effective_free_surfaces = std::move(effective_free_surfaces),
        .node_pressure_values = std::move(node_pressure_values),
        .prepared_weak_velocity_boundaries = std::move(prepared_weak_velocity_boundaries),
        .surface_identities = std::move(surface_identities),
        .active_boundary_index = active_boundary_index,
        .bulk_volume = std::move(bulk_volume),
        .active_domain_log = std::move(active_domain_log),
        .generated_active_boundary_for = std::move(generated_active_boundary_for),
        .required_nitsche_routes = std::move(required_nitsche_routes),
        .fitted_normal_boundaries = std::move(fitted_normal_boundaries),
        .has_exterior_free_surface = has_exterior_free_surface,
    };
}

bool hasNativeMeshOrWarn(const FE::systems::FESystem& system)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const bool native_mesh_available = system.mesh() != nullptr;
#else
    (void)system;
    const bool native_mesh_available = false;
#endif
    if (!native_mesh_available) {
        FE::logRecord(post::navierStokesLog("active-domain inactive constraints skipped", FE::LogLevel::WARNING)
                          .diagnostic("active_domain_inactive_constraints_skipped")
                          .text("reason", "native_mesh_unavailable"));
    }
    return native_mesh_available;
}


// ---------------------------------------------------------------------------
// The two-fluid mode (was IncompressibleTwoFluidModule)
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] bool twoFluidSpacesCompatible(
    const FE::spaces::FunctionSpace& lhs,
    const FE::spaces::FunctionSpace& rhs) noexcept
{
    return lhs.element_type() == rhs.element_type() &&
           lhs.polynomial_order() == rhs.polynomial_order() &&
           lhs.continuity() == rhs.continuity() &&
           lhs.value_dimension() == rhs.value_dimension();
}

void requireFinitePositive(FE::Real value, std::string_view name)
{
    FE::numeric::validate(
        value,
        [](FE::Real candidate) {
            return FE::numeric::isFinite(candidate) &&
                   FE::numeric::isPositive(candidate);
        },
        "IncompressibleTwoFluidModule: " + std::string(name) +
            " must be finite and positive");
}

enum class VelocityBoundaryDataPolicy {
    HomogeneousPhaseLocal,
    SharedExternalData,
};

void validateSharedVelocityBoundaryValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view scope)
{
    if (const auto* literal = std::get_if<FE::Real>(&value)) {
        FE::numeric::validate(
            *literal, FE::numeric::isFinite,
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_shared_velocity_boundary:" +
                std::string(scope));
        return;
    }
    if (const auto* coefficient =
            std::get_if<FE::forms::ScalarCoefficient>(&value)) {
        if (!*coefficient) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_empty_shared_velocity_coefficient:" +
                std::string(scope));
        }
        return;
    }
    if (const auto* coefficient =
            std::get_if<FE::forms::TimeScalarCoefficient>(&value)) {
        if (!*coefficient) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_empty_shared_velocity_coefficient:" +
                std::string(scope));
        }
        return;
    }
    throw std::invalid_argument(
        "[svMultiPhysics::Physics] "
        "unsupported_two_fluid_shared_velocity_form_expression:" +
        std::string(scope));
}

void validateVelocityBoundaryData(
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        boundaries,
    std::string_view scope,
    VelocityBoundaryDataPolicy policy)
{
    std::set<int> markers;
    for (const auto& boundary : boundaries) {
        if (boundary.boundary_marker < 0) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_negative_velocity_boundary_marker:" +
                std::string(scope));
        }
        if (!markers.insert(boundary.boundary_marker).second) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_duplicate_velocity_boundary_marker:" +
                std::string(scope));
        }
        for (std::size_t component = 0u;
             component < boundary.active_components.size(); ++component) {
            if (!boundary.active_components[component]) {
                continue;
            }
            if (policy == VelocityBoundaryDataPolicy::SharedExternalData) {
                validateSharedVelocityBoundaryValue(
                    boundary.value[component], scope);
            } else if (!FE::forms::bc::isZeroConstantScalarValue(
                           boundary.value[component])) {
                throw std::invalid_argument(
                    "[svMultiPhysics::Physics] "
                    "unsupported_two_fluid_nonhomogeneous_velocity_boundary:" +
                    std::string(scope));
            }
        }
    }
}

void validateNoSharedVelocityBoundaryOverlap(
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        shared,
    std::span<const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        phase_local,
    std::string_view phase)
{
    std::set<int> shared_markers;
    for (const auto& boundary : shared) {
        shared_markers.insert(boundary.boundary_marker);
    }
    for (const auto& boundary : phase_local) {
        if (shared_markers.find(boundary.boundary_marker) !=
            shared_markers.end()) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_overlapping_shared_velocity_boundary_marker:" +
                std::string(phase));
        }
    }
}

void validateVelocitySpace(
    const std::shared_ptr<const FE::spaces::FunctionSpace>& space,
    std::string_view phase)
{
    if (!space) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: null " + std::string(phase) +
            " velocity space");
    }
    const auto product = std::dynamic_pointer_cast<
        const FE::spaces::ProductSpace>(space);
    const auto dim = space->value_dimension();
    const bool supported_cell =
        (dim == 2 && space->element_type() == FE::ElementType::Triangle3) ||
        (dim == 3 && space->element_type() == FE::ElementType::Tetra4);
    if (!product || !supported_cell || space->polynomial_order() != 1 ||
        space->continuity() != FE::Continuity::C0) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: phase velocity spaces must be affine C0 P1 Product H1 spaces on Triangle3 or Tetra4 cells");
    }
}

void validatePressureSpace(
    const std::shared_ptr<const FE::spaces::FunctionSpace>& space,
    const FE::spaces::FunctionSpace& velocity_space,
    std::string_view phase)
{
    if (!space || space->value_dimension() != 1 ||
        space->space_type() != FE::spaces::SpaceType::H1 ||
        space->element_type() != velocity_space.element_type() ||
        space->polynomial_order() != 1 ||
        space->continuity() != FE::Continuity::C0) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: " + std::string(phase) +
            " pressure space must be a matching scalar affine C0 P1 H1 space");
    }
}

void validateExistingUnknownField(
    const FE::systems::FESystem& system,
    std::string_view name,
    const FE::spaces::FunctionSpace& space,
    int components)
{
    const auto field = system.findFieldByName(name);
    if (field == FE::INVALID_FIELD_ID) {
        return;
    }
    const auto& record = system.fieldRecord(field);
    if (record.source_kind != FE::systems::FieldSourceKind::Unknown ||
        record.components != components || !record.space ||
        !twoFluidSpacesCompatible(*record.space, space)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: existing field '" +
            std::string(name) +
            "' is incompatible with the requested phase unknown");
    }
}

} // namespace

void validateIncompressibleTwoFluidConfigurationSemantics(
    const IncompressibleTwoFluidOptions& options)
{
    for (const auto acceleration : options.body_force) {
        if (!std::isfinite(acceleration)) {
            throw std::invalid_argument(
                "[svMultiPhysics::Physics] "
                "unsupported_two_fluid_nonfinite_body_force");
        }
    }
    if (options.prescribed_pressure_jump.has_value() &&
        !std::isfinite(*options.prescribed_pressure_jump)) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_prescribed_pressure_jump");
    }
    if (options.prescribed_viscous_traction_jump.has_value() &&
        !std::all_of(
            options.prescribed_viscous_traction_jump->begin(),
            options.prescribed_viscous_traction_jump->end(),
            [](FE::Real value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_prescribed_viscous_traction_jump");
    }
    if (options.shared_pressure_gauge.has_value() &&
        options.shared_pressure_gauge->vertex_gid < 0) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_negative_shared_pressure_gauge_vertex_gid");
    }
    if (options.shared_pressure_gauge.has_value() &&
        !std::isfinite(options.shared_pressure_gauge->pressure)) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_nonfinite_shared_pressure_gauge_value");
    }
    validateVelocityBoundaryData(
        options.negative_phase.velocity_dirichlet,
        "negative-phase",
        VelocityBoundaryDataPolicy::HomogeneousPhaseLocal);
    validateVelocityBoundaryData(
        options.positive_phase.velocity_dirichlet,
        "positive-phase",
        VelocityBoundaryDataPolicy::HomogeneousPhaseLocal);
    validateVelocityBoundaryData(
        options.shared_velocity_dirichlet,
        "shared",
        VelocityBoundaryDataPolicy::SharedExternalData);
    validateNoSharedVelocityBoundaryOverlap(
        options.shared_velocity_dirichlet,
        options.negative_phase.velocity_dirichlet,
        "negative-phase");
    validateNoSharedVelocityBoundaryOverlap(
        options.shared_velocity_dirichlet,
        options.positive_phase.velocity_dirichlet,
        "positive-phase");
}

IncompressibleTwoFluidPreflight preflightIncompressibleTwoFluid(
    const FE::systems::FESystem& system,
    const IncompressibleTwoFluidOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& negative_velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& negative_pressure_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& positive_velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& positive_pressure_space)
{
    validateVelocitySpace(negative_velocity_space, "negative-phase");
    validateVelocitySpace(positive_velocity_space, "positive-phase");
    validatePressureSpace(
        negative_pressure_space, *negative_velocity_space, "negative-phase");
    validatePressureSpace(
        positive_pressure_space, *positive_velocity_space, "positive-phase");
    const auto dimension = negative_velocity_space->value_dimension();
    if (positive_velocity_space->value_dimension() != dimension ||
        !twoFluidSpacesCompatible(*negative_velocity_space,
                          *positive_velocity_space) ||
        !twoFluidSpacesCompatible(*negative_pressure_space,
                          *positive_pressure_space)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: both phases must use matching velocity and pressure spaces");
    }
    if (dimension == 2 && options.body_force[2] != FE::Real{0.0}) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_out_of_plane_body_force");
    }
    if (options.prescribed_viscous_traction_jump.has_value() &&
        dimension == 2 &&
        (*options.prescribed_viscous_traction_jump)[2] != FE::Real{0.0}) {
        throw std::invalid_argument(
            "[svMultiPhysics::Physics] "
            "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
    }

    const std::array<std::string_view, 5> field_names{
        options.negative_phase.velocity_field_name,
        options.negative_phase.pressure_field_name,
        options.positive_phase.velocity_field_name,
        options.positive_phase.pressure_field_name,
        options.level_set_field_name,
    };
    for (const auto name : field_names) {
        if (name.empty()) {
            throw std::invalid_argument(
                "IncompressibleTwoFluidModule: field names must be nonempty");
        }
    }
    for (std::size_t i = 0; i < field_names.size(); ++i) {
        for (std::size_t j = i + 1u; j < field_names.size(); ++j) {
            if (field_names[i] == field_names[j]) {
                throw std::invalid_argument(
                    "IncompressibleTwoFluidModule: all phase and level-set field names must be distinct");
            }
        }
    }
    if (options.operator_tag.empty() ||
        options.generated_interface_domain_id.empty()) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: operator tag and generated-interface domain must be nonempty");
    }
    if (options.generated_interface_geometry !=
            FreeSurfaceGeneratedInterfaceGeometry::LinearCorner ||
        options.geometry_tangent_policy !=
            FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: the initial envelope requires LinearCorner generated geometry with RefreshedFrozenQuadrature tangents");
    }
    if (!std::isfinite(options.level_set_isovalue) ||
        !std::isfinite(options.surface_tension) ||
        options.surface_tension < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set isovalue and nonnegative surface tension must be finite");
    }
    if (!options.enable_vms) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: the affine equal-order initial envelope requires phasewise VMS/PSPG stabilization");
    }
    requireFinitePositive(
        options.negative_phase.density, "negative density");
    requireFinitePositive(
        options.positive_phase.density, "positive density");
    requireFinitePositive(
        options.negative_phase.viscosity, "negative viscosity");
    requireFinitePositive(
        options.positive_phase.viscosity, "positive viscosity");
    requireFinitePositive(
        options.interface_nitsche_gamma, "interface Nitsche gamma");
    requireFinitePositive(options.ct_m, "VMS ct_m");
    requireFinitePositive(options.ct_c, "VMS ct_c");
    requireFinitePositive(
        options.stabilization_epsilon, "stabilization epsilon");
    requireFinitePositive(
        options.pressure_gradient_penalty,
        "pressure-gradient ghost penalty");
    if (!FE::constraints::validSmallCutAggregationGuards(
            options.small_cut_aggregation_guards)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: invalid small-cut aggregation guards");
    }
    validateIncompressibleTwoFluidConfigurationSemantics(options);

    const auto level_set =
        system.findFieldByName(options.level_set_field_name);
    if (level_set == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set field '" +
            options.level_set_field_name +
            "' must be registered before the coupled fluid owner");
    }
    const auto& level_set_record = system.fieldRecord(level_set);
    const bool supported_level_set_source =
        level_set_record.source_kind ==
            FE::systems::FieldSourceKind::Unknown ||
        level_set_record.source_kind ==
            FE::systems::FieldSourceKind::PrescribedData;
    if (!supported_level_set_source || level_set_record.components != 1 ||
        !level_set_record.space ||
        level_set_record.space->space_type() !=
            FE::spaces::SpaceType::H1 ||
        !twoFluidSpacesCompatible(*level_set_record.space,
                          *negative_pressure_space)) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: level-set field must be a prescribed or unknown scalar field in the matching affine C0 P1 space");
    }

    validateExistingUnknownField(
        system,
        options.negative_phase.velocity_field_name,
        *negative_velocity_space,
        dimension);
    validateExistingUnknownField(
        system,
        options.negative_phase.pressure_field_name,
        *negative_pressure_space,
        1);
    validateExistingUnknownField(
        system,
        options.positive_phase.velocity_field_name,
        *positive_velocity_space,
        dimension);
    validateExistingUnknownField(
        system,
        options.positive_phase.pressure_field_name,
        *positive_pressure_space,
        1);
    return IncompressibleTwoFluidPreflight{.dimension = dimension, .level_set = level_set};
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

namespace svmp::Physics::options {

void validate_fitted_surface_contact_capability(
    const svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::
      FreeSurfaceImplementation;
  using ContactLine = svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

  for (const auto& boundary : options.free_surface) {
    if (boundary.implementation != FreeSurfaceImplementation::FittedALE) {
      continue;
    }
    // Fitted SurfaceStress needs no opt-in (decision D47 Q2(b)); the module
    // checks its frame, its surface tension and its contact lines.
    for (const auto& contact_line : boundary.contact_lines) {
      if (std::holds_alternative<ContactLine::PrescribedAngle>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed fitted contact angles are unsupported until a true fitted contact-line (codimension-two) integration entity is available; the condition must not be integrated over the complete free-surface boundary");
      }
      if (std::holds_alternative<ContactLine::DynamicRenE>(
              contact_line.configuration)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: DynamicContactAngle is currently supported only for sharp unfitted level-set free surfaces");
      }
    }
  }
}

} // namespace svmp::Physics::options
