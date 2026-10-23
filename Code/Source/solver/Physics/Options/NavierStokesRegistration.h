/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#ifndef SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_REGISTRATION_H
#define SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_REGISTRATION_H

/**
 * @file NavierStokesRegistration.h
 * @brief Registration of the Navier-Stokes equations: the register API and
 *        the resolution of the physical model
 *
 * The register API (formerly NavierStokesRegister.h) is what application
 * orchestration calls before installing the "fluid" and "stokes" equations:
 * the fitted contact preflight, the two-fluid dependency and the early
 * registration of fields.  Below it, in svmp::Physics::options, are the
 * scope and schema-contract checks, the physical-model resolution, the
 * two-fluid parameter readers and the two-fluid preflight, moved unchanged
 * from NavierStokesRegister.cpp (decision D52).  The register API functions
 * are still defined in NavierStokesRegister.cpp, which owns the two-fluid
 * translation until the option schema replaces it (D47 P2, batch B).
 */

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Core/Types.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace svmp {
class MeshBase;
}

namespace svmp::FE::spaces {
class FunctionSpace;
}

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::Physics {
struct EquationModuleInput;
}

namespace svmp::Physics::formulations::navier_stokes {

/** Pure dependency advertised by the sharp incompressible two-fluid owner. */
struct IncompressibleTwoFluidDependency {
  const svmp::MeshBase* mesh{nullptr};
  std::string mesh_name{};
  int dimension{0};
  int interface_marker{-1};
  std::string level_set_field_name{};
  std::string negative_velocity_field_name{};
  std::string positive_velocity_field_name{};
  std::string negative_pressure_field_name{};
  std::string positive_pressure_field_name{};
  std::string operator_tag{};
  std::string generated_interface_domain_id{};
};

/**
 * Validate the fitted free-surface/contact capability encoded by an input.
 *
 * This typed translation preflight does not mutate an FE system.  Application
 * orchestration uses it before cross-equation field pre-registration so an
 * excluded fitted request cannot leave a partially declared system behind.
 */
void preflightFittedSurfaceContactCapability(
    const EquationModuleInput& input);

/** Return the validated two-fluid dependency, or nullopt for another model. */
[[nodiscard]] std::optional<IncompressibleTwoFluidDependency>
incompressibleTwoFluidDependency(
    const EquationModuleInput& input,
    const FE::systems::FESystem& system);

/**
 * Predeclare phase fields in velocity-pair/pressure-pair order and publish the
 * matching material-interface velocity owner before equation installation.
 */
void preRegisterIncompressibleTwoFluidDependencyFields(
    const EquationModuleInput& input,
    FE::systems::FESystem& system);

/**
 * Predeclare the primary velocity unknown described by a fluid/stokes input.
 *
 * This field-only hook supports coupled modules that are intentionally listed
 * before the owning Navier--Stokes equation and need its velocity FieldId while
 * installing their forms.  The regular Navier--Stokes registration remains the
 * owner and revalidates the field's source kind, components, and function space.
 */
[[nodiscard]] FE::FieldId preRegisterPrimaryVelocityField(
    const EquationModuleInput& input,
    FE::systems::FESystem& system);

void forceLink_NavierStokesRegister();

} // namespace svmp::Physics::formulations::navier_stokes

namespace svmp::Physics::options {

enum class NavierStokesPhysicalModel {
  OnePhaseLiquidPrescribedExteriorPressure,
  IncompressibleTwoFluid,
};

struct FreeSurfaceSchemaContract {
  int version{
      svmp::Physics::formulations::navier_stokes::
          IncompressibleNavierStokesVMSOptions::
              current_configuration_schema_version};
  bool explicit_legacy{false};
};

FreeSurfaceSchemaContract resolve_free_surface_schema_contract(
    const svmp::Physics::ParameterMap& params);

NavierStokesPhysicalModel
validate_and_resolve_free_surface_physical_model(
    const svmp::Physics::EquationModuleInput& input);

std::string require_two_fluid_string(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name);
svmp::FE::Real require_two_fluid_real(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name);
int require_two_fluid_int(
    const svmp::Physics::ParameterMap& params,
    std::initializer_list<std::string_view> keys,
    std::string_view canonical_name);
void reject_unsupported_two_fluid_default_domain_parameters(
    const svmp::Physics::ParameterMap& params);
void reject_unsupported_two_fluid_equation_parameters(
    const svmp::Physics::ParameterMap& params);
void reject_unsupported_two_fluid_boundary_parameters(
    std::span<const svmp::Physics::BoundaryConditionInput> boundary_conditions);

struct TranslatedIncompressibleTwoFluidInput {
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> velocity_space{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> pressure_space{};
  svmp::Physics::formulations::navier_stokes::
      IncompressibleTwoFluidOptions options{};
};

[[nodiscard]] bool dependencySpacesCompatible(
    const svmp::FE::spaces::FunctionSpace& lhs,
    const svmp::FE::spaces::FunctionSpace& rhs) noexcept;
void preflightTwoFluidUnknownField(
    const svmp::FE::systems::FESystem& system,
    std::string_view name,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space,
    int components);
[[nodiscard]] svmp::FE::FieldId ensureTwoFluidUnknownField(
    svmp::FE::systems::FESystem& system,
    std::string name,
    const std::shared_ptr<const svmp::FE::spaces::FunctionSpace>& space,
    int components);
void preflightTwoFluidTranslationBeforeMutation(
    const TranslatedIncompressibleTwoFluidInput& translated,
    const svmp::FE::systems::FESystem& system);

/// The one-phase options and spaces of a fluid or stokes input.
struct ResolvedOnePhaseNavierStokesInput {
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> velocity_space{};
  std::shared_ptr<const svmp::FE::spaces::FunctionSpace> pressure_space{};
  svmp::Physics::formulations::navier_stokes::
      IncompressibleNavierStokesVMSOptions options{};
};

/// A resolved fluid or stokes input: the physical model and, for it, the
/// one-phase or the two-fluid options and spaces.
struct ResolvedNavierStokesInput {
  NavierStokesPhysicalModel physical_model{
      NavierStokesPhysicalModel::OnePhaseLiquidPrescribedExteriorPressure};
  std::optional<ResolvedOnePhaseNavierStokesInput> one_phase{};
  std::optional<TranslatedIncompressibleTwoFluidInput> two_fluid{};
};

/// Resolve a fluid or stokes equation input: the fitted contact preflight,
/// the physical model, then the one-phase translation (equation, domain and
/// boundary keys, boundary data files) or the two-fluid translation.  The
/// register constructs the module from the result.
[[nodiscard]] ResolvedNavierStokesInput resolveNavierStokesInput(
    const svmp::Physics::EquationModuleInput& input,
    svmp::FE::systems::FESystem& system);

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_REGISTRATION_H
