/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#ifndef SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H
#define SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H

/**
 * @file NavierStokesOptionRules.h
 * @brief Validation rules of the Navier-Stokes options
 *
 * Checks of option values against each other and against the installation
 * (decisions D47 and D52), moved unchanged from the free-surface support
 * files of Physics/Formulations/NavierStokes:
 *  - the capability rules of the free-surface and boundary options
 *    (formerly FreeSurface/FreeSurfaceValidation): which combinations of
 *    implementation, active domain, surface-tension form, kinematic
 *    enforcement, contact laws, cut stabilization and aggregation are
 *    supported, the standard boundary conditions against the free
 *    surfaces, the uniqueness of generated markers, and the envelope of the
 *    semi-implicit capillary term, with the messages tests check;
 *  - the preflight of a registration (formerly NavierStokesPreflight):
 *    every check that runs before the system is changed, and what it
 *    resolves for the registration that follows;
 *  - the fitted contact capability of a translated input.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Core/Types.h"
#include "FE/Core/LogRecord.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/LevelSet/LevelSetSideSupport.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/MeshBoundaryKinematics.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

struct PreparedContactIdentity {
    std::size_t contact_index;
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key;
    FE::forms::ContactLineDomain domain;
};

/// Immutable geometry identities, filled only at the first required lookup.
struct PreparedSurfaceIdentity {
    std::optional<FE::level_set::LevelSetGeometry> geometry;
    std::vector<PreparedContactIdentity> contacts;
};

struct PreparedFittedNormalBoundary {
    std::size_t boundary_index;
    FE::systems::MeshNormalKinematicBinding binding;
};

/// The constant target required by fitted stage measurement, checked at its
/// late declaration point without evaluating a coefficient.
[[nodiscard]] std::optional<std::array<FE::Real, 3>> fittedTangentialMeasurementVelocity(
    const FreeSurfaceBoundary& boundary);

/// The selected active-side pressure policy; the boundary storage is borrowed.
struct ActivePressureDomain {
    const FreeSurfaceBoundary* boundary{nullptr};
    FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
};

[[nodiscard]] std::optional<ActivePressureDomain> activePressureDomainFor(
    const std::vector<FreeSurfaceBoundary>& free_surfaces);

/// Ordered input values for vertex constraints and initialization comparisons.
[[nodiscard]] std::vector<FE::constraints::VertexDirichletValue> nodePressureValues(
    const IncompressibleNavierStokesVMSOptions& options);

struct PreparedPressureInitialization {
    FE::level_set::SideVertexInitializationDeclaration field;
    std::optional<ActivePressureDomain> active_domain;
};

/// Declare the pressure source and support from the original options without
/// reading fields or evaluating the law. The options and gauge storage must
/// remain valid until initialization finishes.
[[nodiscard]] PreparedPressureInitialization hydrostaticPressureInitialization(
    const IncompressibleNavierStokesVMSOptions& options,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& law,
    std::span<const FE::constraints::VertexDirichletValue> gauges);

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context);

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context);

void validateGeneratedFreeSurfaceMarkerUniqueness(
    std::span<const FreeSurfaceBoundary> boundaries,
    const FE::systems::FESystem& system,
    std::span<PreparedSurfaceIdentity> identities);

/// The vertex-id mode of the node pressure constraints.
[[nodiscard]] FE::constraints::VertexIdMode nodePressureIdMode(
    const IncompressibleNavierStokesVMSOptions& options);

/// The local pressure vertex of node pressure constraint id @p node_id, or
/// nothing when its global vertex is not on this rank.
[[nodiscard]] std::optional<FE::GlobalIndex> pressureConstraintLocalVertex(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    FE::GlobalIndex node_id);

/// The node pressure constraints have nonnegative ids, finite values, one
/// value per node, and local vertices inside the local pressure mesh.
void validateNodePressureConstraints(
    const FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options);

void validateFreeSurfaceBoundary(const FreeSurfaceBoundary& bc,
                                 const IncompressibleNavierStokesVMSOptions& options,
                                 bool ale_enabled,
                                 const FE::systems::FESystem& system,
                                 int dim,
                                 PreparedSurfaceIdentity& identity);

/// Resolved weak-boundary input for this registration transaction. Certified
/// route bindings are built separately with the actual registered fields.
struct PreparedWeakVelocityBoundary {
    int physical_marker{-1};
    std::optional<int> generated_marker{};
    FE::forms::FormExpr prescribed_velocity{};
};

void validateNavierStokesBoundaryConfiguration(
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    const FE::spaces::FunctionSpace& velocity_space,
    const FE::spaces::FunctionSpace& pressure_space,
    int dim,
    const std::function<std::optional<int>(int)>& generated_active_boundary_for,
    std::vector<PreparedWeakVelocityBoundary>* prepared_weak_velocity = nullptr,
    std::vector<FE::forms::BoundaryDomain>* required_routes = nullptr);

/// What preflightNavierStokes resolved for the registration that follows.
struct NavierStokesPreflight {
    int dim{0};
    bool enable_vms{true};
    FE::systems::FieldSpec u_spec;
    FE::systems::FieldSpec p_spec;
    std::optional<FE::systems::FieldSpec> body_force_spec;
    std::optional<FE::systems::FieldSpec> semi_implicit_reference_spec;
    FE::systems::ALEBindingOptions ale_options;
    std::vector<FreeSurfaceBoundary> effective_free_surfaces;   // generated interface markers resolved
    std::vector<FE::constraints::VertexDirichletValue> node_pressure_values;
    std::vector<PreparedWeakVelocityBoundary> prepared_weak_velocity_boundaries;
    std::vector<PreparedSurfaceIdentity> surface_identities;
    std::optional<std::size_t> active_boundary_index;
    FE::forms::VolumeDomain bulk_volume{FE::forms::VolumeDomain::full()};
    std::optional<FE::LogRecord> active_domain_log;
    std::function<std::optional<int>(int)> generated_active_boundary_for;
    std::vector<FE::forms::BoundaryDomain> required_nitsche_routes;
    std::vector<PreparedFittedNormalBoundary> fitted_normal_boundaries;
    bool has_exterior_free_surface{false};
};

/// Every check of a Navier-Stokes registration that runs before the system is
/// changed: options, spaces, free surfaces (capability rules, fitted
/// mesh-motion ownership, generated markers, contact walls, energy
/// declarations), generated-boundary Nitsche routes, field compatibility.  A
/// rejected configuration leaves the system definition unchanged.
[[nodiscard]] NavierStokesPreflight preflightNavierStokes(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& pressure_space);

/// The system has a native mesh, which the active-side constraints need;
/// warns (and the constraints are skipped) otherwise.
[[nodiscard]] bool hasNativeMeshOrWarn(const FE::systems::FESystem& system);

/** Validate parser-visible semantics shared by preflight and registration. */
void validateIncompressibleTwoFluidConfigurationSemantics(
    const IncompressibleTwoFluidOptions& options);

/// What preflightIncompressibleTwoFluid resolved for the two-fluid
/// registration that follows.
struct IncompressibleTwoFluidPreflight {
    int dimension{0};
    FE::FieldId level_set{FE::INVALID_FIELD_ID};
};

/// Every check of a two-fluid registration that runs before the system is
/// changed: the phase spaces, the options of the initial envelope (with
/// validateIncompressibleTwoFluidConfigurationSemantics), the registered
/// level-set field and any existing phase unknowns.  The messages are the
/// two-fluid module's ("IncompressibleTwoFluidModule: ...").
[[nodiscard]] IncompressibleTwoFluidPreflight preflightIncompressibleTwoFluid(
    const FE::systems::FESystem& system,
    const IncompressibleTwoFluidOptions& options,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& negative_velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& negative_pressure_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& positive_velocity_space,
    const std::shared_ptr<const FE::spaces::FunctionSpace>& positive_pressure_space);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

namespace svmp::Physics::options {

/// The fitted free-surface contact capability of a translated input (fails
/// closed on fitted prescribed or dynamic contact angles).
void validate_fitted_surface_contact_capability(
    const svmp::Physics::formulations::navier_stokes::
        IncompressibleNavierStokesVMSOptions& options);

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_RULES_H
