/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceLegacyTerms.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <stdexcept>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

FE::forms::FormExpr unfittedCurvatureTraction(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi,
    const FE::forms::FormExpr& p_ext,
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& n,
    const FE::forms::FormExpr& v)
{
    using namespace FE::forms;
    const auto curvature = bc::isZeroConstantScalarValue(bc.surface_tension)
        ? FormExpr::constant(0.0)
        : unfittedTractionCurvature(bc, system, phi);
    const auto traction_scalar = -p_ext - gamma * curvature;
    const auto traction = traction_scalar * n;
    return FE::forms::FormExpr::constant(FE::Real{-1.0}) *
           inner(traction, v);
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
