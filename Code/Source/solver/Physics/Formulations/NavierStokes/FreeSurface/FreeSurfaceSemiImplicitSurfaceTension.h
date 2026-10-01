#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_SEMI_IMPLICIT_SURFACE_TENSION_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_SEMI_IMPLICIT_SURFACE_TENSION_H

/**
 * @file FreeSurfaceSemiImplicitSurfaceTension.h
 * @brief Lagged normal-increment capillary term for unfitted free surfaces
 *
 * The term (design note free_surface_semi_implicit_surface_tension_design.md,
 * section 3.3) is
 *
 *     R_SI(u; v) = gamma dt_eff int_{Gamma_h}
 *                  grad_Gamma((u - u_ref).n_h) . grad_Gamma(v.n_h) dGamma,
 *     grad_Gamma(w.n_h) = P_h (grad w)^T n_h,   P_h = I - n_h (x) n_h,
 *
 * where dt_eff = 1/a0 is the effective step of the time integrator and
 * u_ref is the velocity of the iterate that generated Gamma_h.  The
 * application overwrites u_ref at every generated-state refresh, so R_SI is
 * zero in every freshly refreshed residual: the accepted state and the
 * acceptance test of the outer fixed point are unchanged, and only the
 * Jacobian of the frozen inner solves gains the constant, symmetric,
 * positive-semidefinite velocity block
 *
 *     gamma dt_eff int_{Gamma_h} grad_Gamma(du.n_h) . grad_Gamma(v.n_h).
 *
 * n_h is constant on each LinearCorner facet, so the block is the
 * Laplace--Beltrami part of the omitted geometry Jacobian.  The coefficient
 * is a physical input times an integrator constant; nothing is tuned.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceOptions.h"

#include "FE/Forms/FormExpr.h"

#include <string_view>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// Prescribed vector field (velocity space) holding u_ref.  The
/// Navier--Stokes module registers it when any free surface requests
/// NormalIncrement; the application refreshes it from the velocity unknown.
inline constexpr std::string_view
    kFreeSurfaceSemiImplicitReferenceVelocityFieldName =
        "ns_free_surface_semi_implicit_reference_velocity";

[[nodiscard]] const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept;

[[nodiscard]] bool usesSemiImplicitNormalIncrement(
    const FreeSurfaceBoundary& bc) noexcept;

/// P_h (grad w)^T n_h for a vector field w: the surface gradient of the
/// normal component w.n_h on a facet with constant normal n_h.
[[nodiscard]] FE::forms::FormExpr normalComponentSurfaceGradient(
    const FE::forms::FormExpr& w,
    const FE::forms::FormExpr& n);

/// gamma dt_eff P_h (grad u - grad u_ref)^T n_h . P_h (grad v)^T n_h.
/// The velocity difference is formed before the projection so that the
/// integrand is exactly zero where u and u_ref carry identical coefficients.
[[nodiscard]] FE::forms::FormExpr semiImplicitNormalIncrementIntegrand(
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& u_ref,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& n);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_SEMI_IMPLICIT_SURFACE_TENSION_H
