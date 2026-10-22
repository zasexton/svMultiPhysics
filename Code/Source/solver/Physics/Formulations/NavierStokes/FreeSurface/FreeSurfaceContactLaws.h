/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_LAWS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_LAWS_H

/**
 * @file FreeSurfaceContactLaws.h
 * @brief The contact-line laws of an unfitted free surface, with their coefficients as forms
 *
 * Reads the contact-line options of one free surface into the data its
 * formulation uses: the wall, the law (Young wall energy, weak or strong
 * Ren-E), the wall normal, cos(theta_e), the mobility and the Navier slip
 * length, under the coefficient names the module has always used.  The
 * options are validated by the preflight.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Core/Types.h"
#include "FE/Forms/FormExpr.h"

#include <array>
#include <functional>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// One PrescribedAngle or DynamicRenE contact line of a free surface.
struct ContactLaw {
    const FreeSurfaceContactLine* line{nullptr};
    int wall{-1};             ///< wall boundary marker
    int marker{-1};           ///< requested contact-line marker (-1: generated)
    bool young{false};        ///< PrescribedAngle: Young wall energy
    bool dynamic{false};      ///< DynamicRenE
    bool strong{false};       ///< DynamicRenE as a contact-point law
    FE::forms::FormExpr n_w{};           ///< wall normal, out of the liquid
    FE::forms::FormExpr cos_e{};         ///< cos(theta_e)
    FE::forms::FormExpr mobility{};      ///< M of the weak Ren-E law
    FE::forms::FormExpr slip_length{};   ///< l_s of the Navier slip; empty for a no-slip wall
    std::array<FE::Real, 3> wall_normal{};           ///< strong law: unit wall normal
    std::function<FE::Real(FE::Real)> contact_speed{};   ///< strong law: cos(theta_d) -> M gamma (cos(theta_e) - cos(theta_d))
};

/// The PrescribedAngle and DynamicRenE contact lines of @p bc, in order.
[[nodiscard]] std::vector<ContactLaw> contactLaws(const FreeSurfaceBoundary& bc, int dim);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_CONTACT_LAWS_H
