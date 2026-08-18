#pragma once

#include "FE/Core/Types.h"

namespace svmp::FE::systems {
class FESystem;
}

namespace svmp::Physics {
struct EquationModuleInput;
}

namespace svmp::Physics::formulations::navier_stokes {

/**
 * Validate the fitted free-surface/contact capability encoded by an input.
 *
 * This typed translation preflight does not mutate an FE system.  Application
 * orchestration uses it before cross-equation field pre-registration so an
 * excluded fitted request cannot leave a partially declared system behind.
 */
void preflightFittedSurfaceContactCapability(
    const EquationModuleInput& input);

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
