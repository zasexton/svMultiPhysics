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
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
    const FE::systems::FESystem& system);

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
                                 int dim);

void validateNavierStokesBoundaryConfiguration(
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    const FE::spaces::FunctionSpace& velocity_space,
    const FE::spaces::FunctionSpace& pressure_space,
    int dim);

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
