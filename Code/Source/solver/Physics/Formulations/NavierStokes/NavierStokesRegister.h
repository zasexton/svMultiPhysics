#pragma once

#include "FE/Core/Types.h"

#include <optional>
#include <string>

namespace svmp {
class MeshBase;
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
