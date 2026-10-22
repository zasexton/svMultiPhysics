/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_VMS_FORMULATION_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_VMS_FORMULATION_H

/**
 * @file NavierStokesVMSFormulation.h
 * @brief The bulk residual of the incompressible Navier-Stokes VMS formulation
 *
 *   R_m = rho (dt u + (grad u) a + chi - f) - div(2 mu eps(u)) + grad p,   a = u - w,
 *   u' = -tau_M R_m,   p' = -tau_C div u          (static subscales)
 *
 * IncompressibleNavierStokesVMSModule integrates these integrands over the
 * liquid (the mesh, or the cut side of a level-set geometry); the two-fluid
 * module can integrate them over each phase.  The integrands are returned
 * unintegrated so that the caller chooses the volume domain.
 */

#include "FE/Core/Types.h"
#include "FE/Forms/FormExpr.h"

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// The unknowns, test functions and material data of one fluid.
struct NavierStokesTerms {
    FE::forms::FormExpr u, p;   // velocity, pressure
    FE::forms::FormExpr v, q;   // their test functions
    FE::forms::FormExpr rho;    // density
    FE::forms::FormExpr mu;     // dynamic viscosity, mu or mu(shear rate)
    FE::forms::FormExpr f;      // body acceleration
    int dim{3};
};

struct NavierStokesVMSSettings {
    bool convection{true};
    bool ale{false};
    bool moving_control_volume{false};      // chi = div(w) u (ALE only)
    bool vms{true};
    bool incremental_pspg_pressure{false};  // PSPG on grad(dt_eff dt p) instead of grad p
    FE::Real ct_m{1.0};
    FE::Real ct_c{36.0};
    FE::Real epsilon{1.0e-12};
};

/// Integrands of the bulk residual.
struct NavierStokesVMSIntegrands {
    FE::forms::FormExpr momentum;                // Galerkin + SUPG + LSIC + cross stress
    FE::forms::FormExpr continuity;              // q div u + PSPG without the pressure gradient
    FE::forms::FormExpr pspg_pressure_gradient;  // tau_M grad q . grad p (its own kernel group)
    FE::forms::FormExpr pspg_without_pressure_gradient;   // tau_M grad q . (R_m - grad p)
    FE::forms::FormExpr galerkin_momentum;
    FE::forms::FormExpr galerkin_continuity;
    // Galerkin parts named for the free-surface residual-work ledger.
    FE::forms::FormExpr convection;              // moving control volume + convection
    FE::forms::FormExpr pressure;                // -p div v
    FE::forms::FormExpr forcing;                 // -rho f . v
};

[[nodiscard]] NavierStokesVMSIntegrands navierStokesVMSIntegrands(
    const NavierStokesTerms& terms,
    const NavierStokesVMSSettings& settings);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_VMS_FORMULATION_H
