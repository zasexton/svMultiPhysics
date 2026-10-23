/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"

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
    if (bc.active_domain == FreeSurfaceActiveDomain::None ||
        bc.allow_full_domain_unfitted_free_surface) {
        reject("Active_domain=LevelSetNegative or LevelSetPositive");
    }
    if (bc.active_domain_method !=
            FreeSurfaceActiveDomainMethod::CutVolume ||
        bc.active_domain_smoothing_width != FE::Real{0.0}) {
        reject("Active_domain_method=CutVolume with zero smoothing width");
    }
    if (normalizedFreeSurfaceOptionToken(bc.generated_interface_geometry) !=
        "linearcorner") {
        reject("Generated_interface_geometry=LinearCorner, whose facet "
               "normal is piecewise constant");
    }
    if (!isRefreshedFrozenGeometryTangent(bc)) {
        reject("Geometry_tangent_policy=RefreshedFrozenQuadrature without "
               "level-set shape tangents");
    }
}

} // namespace

void validateGeneratedFreeSurfaceMarkerUniqueness(
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system)
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

    for (const auto& bc : boundaries) {
        if (!isUnfittedLevelSet(bc)) {
            continue;
        }
        const auto phi_id = resolveLevelSetFieldId(bc, system);
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

        for (const auto& contact_line : bc.contact_lines) {
            if (contactLineKind(contact_line) !=
                    ContactLineKind::PrescribedAngle &&
                contactLineKind(contact_line) !=
                    ContactLineKind::DynamicRenE) {
                continue;
            }
            const auto key = generatedUnfittedContactLineMarkerKey(
                contact_line, bc, system);
            const auto marker =
                FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
                    key);
            register_marker(
                marker,
                "contact:" + key.stableKey(),
                "interface/contact");
        }
    }
}

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc,
                                 const IncompressibleNavierStokesVMSOptions& options,
                                 bool ale_enabled,
                                 const FE::systems::FESystem& system,
                                 int dim)
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
        bc.active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume &&
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
             FreeSurfaceTangentialMeshPolicy::Prescribed) &&
        (bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Enabled ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Incremental ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::Disabled ||
         bc.cut_cell_stabilization.pressure_policy ==
             FreeSurfacePressureStabilizationPolicy::
                 DisabledForRefreshedFrozenHighOrder) &&
        (bc.surface_tension_form ==
             FreeSurfaceSurfaceTensionForm::Automatic ||
         bc.surface_tension_form ==
             FreeSurfaceSurfaceTensionForm::SurfaceStress);
    if (!supported_boundary_policies) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: unsupported "
            "free-surface boundary policy");
    }
    if (!std::isfinite(bc.level_set_isovalue)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface level_set_isovalue must be finite");
    }
    if (!std::isfinite(bc.active_domain_smoothing_width) ||
        bc.active_domain_smoothing_width < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface active_domain_smoothing_width must be finite and nonnegative");
    }
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
    const auto& aggregation_guards = bc.small_cut_aggregation_guards;
    if (aggregation_guards.maximum_root_path_length == 0u ||
        !(aggregation_guards.maximum_reference_extrapolation_distance >=
          FE::Real{0.0}) ||
        !std::isfinite(
            aggregation_guards.maximum_reference_extrapolation_distance) ||
        !(aggregation_guards.maximum_absolute_coefficient >= FE::Real{1.0}) ||
        !std::isfinite(aggregation_guards.maximum_absolute_coefficient) ||
        !(aggregation_guards.maximum_row_l1_norm >= FE::Real{1.0}) ||
        !std::isfinite(aggregation_guards.maximum_row_l1_norm) ||
        aggregation_guards.maximum_row_l1_norm <
            aggregation_guards.maximum_absolute_coefficient) {
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
            bc.active_domain == FreeSurfaceActiveDomain::None ||
            bc.active_domain_method !=
                FreeSurfaceActiveDomainMethod::CutVolume ||
            bc.active_domain_smoothing_width != FE::Real{0.0} ||
            bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume requires fixed Eulerian UnfittedLevelSet CutVolume integration on one explicit active side, no exterior boundary marker, and no smoothing/full-domain fallback");
        }
        if (!FE::forms::bc::isZeroConstantScalarValue(
                bc.external_pressure) ||
            !FE::forms::bc::isZeroConstantScalarValue(
                bc.surface_tension) ||
            bc.surface_tension_form !=
                FreeSurfaceSurfaceTensionForm::Automatic) {
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
        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& phi_record = system.fieldRecord(phi_id);
        if (phi_record.components != 1 || !phi_record.space ||
            phi_record.space->value_dimension() != 1) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: an internal material-interface volume requires a scalar level-set field");
        }
        const auto& cut = bc.cut_cell_stabilization;
        if (cut.enabled && cut.cut_metadata_scale_cap.has_value() &&
            (!std::isfinite(*cut.cut_metadata_scale_cap) ||
             *cut.cut_metadata_scale_cap < FE::Real{1.0})) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: cut-cell metadata scale cap must be finite and at least 1");
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
        if (bc.active_domain == FreeSurfaceActiveDomain::None &&
            !bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: UnfittedLevelSet free surfaces require Active_domain=LevelSetNegative or LevelSetPositive; set Allow_full_domain_unfitted_free_surface=true only for deliberate full-domain diagnostic runs");
        }
        if (bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unfitted level-set free surfaces use Eulerian level-set transport for kinematics; a mesh kinematic enforcement (MeshNitsche) needs a fitted ALE boundary and is therefore unsupported");
        }
        const bool has_surface_stress_residual =
            usesSurfaceStress(bc) &&
            (!FE::forms::bc::isZeroConstantScalarValue(
                 bc.external_pressure) ||
             !FE::forms::bc::isZeroConstantScalarValue(
                 bc.surface_tension));
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
            if (cut.cut_metadata_scale_cap.has_value() &&
                (!std::isfinite(*cut.cut_metadata_scale_cap) ||
                 *cut.cut_metadata_scale_cap < FE::Real{1.0})) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: cut-cell metadata scale cap must be finite and at least 1");
            }
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
        if (bc.allow_full_domain_unfitted_free_surface) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Allow_full_domain_unfitted_free_surface is only valid for unfitted level-set free surfaces");
        }
        if (!options.explicit_legacy_configuration) {
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
                    "normal enforcement (the input default); None is "
                    "accepted only by the explicit schema-1 legacy mode");
            }
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
        if (bc.tangential_mesh_policy ==
                FreeSurfaceTangentialMeshPolicy::Prescribed &&
            !ale_enabled) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed fitted "
                "free-surface tangential mesh velocity requires ALE");
        }
        if (bc.tangential_mesh_policy ==
                FreeSurfaceTangentialMeshPolicy::Prescribed &&
            options.mesh_velocity_source !=
                ALEMeshVelocitySource::CoupledDisplacement) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: prescribed fitted "
                "free-surface tangential mesh velocity requires a coupled "
                "mesh-displacement unknown");
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
    if (bc.kinematic_enforcement ==
            FreeSurfaceKinematicEnforcement::MeshNitsche &&
        (isUnfittedLevelSet(bc) || !ale_enabled ||
         options.mesh_velocity_source !=
             ALEMeshVelocitySource::CoupledDisplacement)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: MeshNitsche free-surface kinematics require a fitted ALE boundary with a coupled mesh-displacement unknown");
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

    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) == ContactLineKind::Pinned) {
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
                if (bc.active_domain_method !=
                    FreeSurfaceActiveDomainMethod::CutVolume) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed-angle Navier slip requires Active_domain_method=CutVolume for a sharp wetted-wall domain");
                }
                if (bc.active_domain_smoothing_width != FE::Real{0.0}) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed-angle Navier slip requires active_domain_smoothing_width=0 because the sharp wall operator has no smoothing-width parameter");
                }
                (void)constantScalarValueOrThrow(
                    contactLineSlipLength(contact_line),
                    "PrescribedContactAngle Navier slip_length");
                validatePositiveConstantScalar(
                    contactLineSlipLength(contact_line),
                    "PrescribedContactAngle Navier slip_length");
            }
            if (isUnfittedLevelSet(bc)) {
                if (bc.active_domain == FreeSurfaceActiveDomain::None) {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require an active liquid side (LevelSetNegative or LevelSetPositive)");
                }
                if (normalizedFreeSurfaceOptionToken(
                        bc.generated_interface_geometry) != "linearcorner") {
                    throw std::invalid_argument(
                        "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles currently support only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
                }
                const auto phi_id = resolveLevelSetFieldId(bc, system);
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
                    generatedUnfittedContactLineMarker(
                        contact_line,
                        bc,
                        system);
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
            if (bc.active_domain == FreeSurfaceActiveDomain::None) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires an active liquid side (LevelSetNegative or LevelSetPositive)");
            }
            if (bc.active_domain_method !=
                FreeSurfaceActiveDomainMethod::CutVolume) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle requires Active_domain_method=CutVolume for a sharp liquid domain");
            }
            if (bc.active_domain_smoothing_width != FE::Real{0.0}) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle with Active_domain_method=CutVolume requires active_domain_smoothing_width=0 because the sharp operator has no smoothing-width parameter");
            }
            if (normalizedFreeSurfaceOptionToken(
                    bc.generated_interface_geometry) != "linearcorner") {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: DynamicContactAngle currently supports only Generated_interface_geometry=LinearCorner; high-order implicit contact geometry is not yet validated");
            }

            const auto phi_id = resolveLevelSetFieldId(bc, system);
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

            const int generated_marker = generatedUnfittedContactLineMarker(
                contact_line,
                bc,
                system);
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
    int dim)
{
    using FE::forms::FormExpr;

    const auto generated_active_boundary_for =
        [&system, free_surfaces](int physical_boundary_marker) {
            return sharpActiveBoundaryMarkerFor(
                physical_boundary_marker, free_surfaces, system);
        };

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
                normalizedFreeSurfaceOptionToken(
                    free_surface.generated_interface_geometry) !=
                    "linearcorner") {
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

    std::vector<FormExpr> zero_components;
    zero_components.reserve(static_cast<std::size_t>(dim));
    for (int component = 0; component < dim; ++component) {
        zero_components.push_back(FormExpr::constant(FE::Real{0.0}));
    }
    const auto zero_flux = FormExpr::asVector(std::move(zero_components));
    const auto validate_finite_real = [](FE::Real value,
                                         std::string_view label) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(label) + " must be finite");
        }
    };
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
            validate_finite_real(value, label);
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
            validate_finite_real(value, label);
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
        &generated_nitsche_trace_binding_preflight);
    std::size_t expected_generated_nitsche_trace_bindings = 0u;
    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(
                bc,
                "IncompressibleNavierStokesVMSModule generated-boundary "
                "Nitsche binding preflight");
        if (generated_active_boundary_for(marker).has_value()) {
            ++expected_generated_nitsche_trace_bindings;
        }
    }
    if (generated_nitsche_trace_binding_preflight.size() !=
        expected_generated_nitsche_trace_bindings) {
        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: generated-boundary "
            "Nitsche binding preflight count differs from the configured "
            "production routes");
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
