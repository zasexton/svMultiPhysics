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
