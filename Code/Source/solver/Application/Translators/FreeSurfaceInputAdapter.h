#pragma once

#include "Physics/Core/EquationModuleInput.h"

namespace svmp {
class DistributedMesh;
using Mesh = DistributedMesh;
}

namespace application::translators::free_surface_input {

bool usesLegacyUnfittedFaceBinding(const svmp::Physics::ParameterMap& params);
void resolveLegacyContactWallFaces(svmp::Physics::ParameterMap& params,
                                   const svmp::Mesh& mesh);

} // namespace application::translators::free_surface_input
