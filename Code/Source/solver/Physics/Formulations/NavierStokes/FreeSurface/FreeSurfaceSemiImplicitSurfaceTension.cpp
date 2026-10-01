#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept
{
    switch (value) {
    case FreeSurfaceSurfaceTensionSemiImplicit::None:
        return "None";
    case FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement:
        return "NormalIncrement";
    }
    return "Unsupported";
}

bool usesSemiImplicitNormalIncrement(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.surface_tension_semi_implicit ==
           FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement;
}

FE::forms::FormExpr normalComponentSurfaceGradient(
    const FE::forms::FormExpr& w,
    const FE::forms::FormExpr& n)
{
    using namespace FE::forms;
    const auto normal_trace_gradient = transpose(grad(w)) * n;
    return normal_trace_gradient - inner(normal_trace_gradient, n) * n;
}

FE::forms::FormExpr semiImplicitNormalIncrementIntegrand(
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& u_ref,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& n)
{
    using namespace FE::forms;
    const auto increment_trace_gradient =
        transpose(grad(u) - grad(u_ref)) * n;
    const auto increment_surface_gradient =
        increment_trace_gradient -
        inner(increment_trace_gradient, n) * n;
    return gamma * FormExpr::effectiveTimeStep() *
           inner(increment_surface_gradient,
                 normalComponentSurfaceGradient(v, n));
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
