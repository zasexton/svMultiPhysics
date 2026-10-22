/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactLaws.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include <cmath>
#include <utility>
#include <variant>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

std::vector<ContactLaw> contactLaws(const FreeSurfaceBoundary& bc, int dim)
{
    std::vector<ContactLaw> laws;
    for (const auto& c : bc.contact_lines) {
        const auto kind = contactLineKind(c);
        if (kind != ContactLineKind::PrescribedAngle && kind != ContactLineKind::DynamicRenE) {
            continue;
        }
        ContactLaw law;
        law.line = &c;
        law.wall = contactLineWallBoundaryMarker(c);
        law.marker = contactLineMarker(c);
        law.young = kind == ContactLineKind::PrescribedAngle;
        law.dynamic = kind == ContactLineKind::DynamicRenE;
        law.strong = law.dynamic && std::get<FreeSurfaceContactLine::DynamicRenE>(c.configuration).strong_law;
        law.n_w = wallNormalExpression(c, dim);
        law.cos_e = FE::forms::FormExpr::constant(
            std::cos(constantScalarValueOrThrow(contactLineAngleRadians(c), "contact_angle_radians")));
        if (law.dynamic && !law.strong) {
            law.mobility = freeSurfaceScalar(contactLineMobility(c), "ns_dynamic_contact_angle_mobility", bc);
        }
        if (contactLineHasNavierSlip(c)) {
            law.slip_length = freeSurfaceScalar(contactLineSlipLength(c), "ns_contact_wall_slip_length", bc);
        }
        if (law.strong) {
            law.wall_normal = normalizedWallNormal(c);
            const FE::Real mobility =
                constantScalarValueOrThrow(contactLineMobility(c), "Strong DynamicContactAngle mobility");
            const FE::Real gamma =
                constantScalarValueOrThrow(bc.surface_tension, "Strong DynamicContactAngle surface tension");
            const FE::Real cos_e = std::cos(constantScalarValueOrThrow(
                contactLineAngleRadians(c), "Strong DynamicContactAngle equilibrium angle"));
            law.contact_speed = [mobility, gamma, cos_e](FE::Real cos_d) {
                return mobility * gamma * (cos_e - cos_d);
            };
        }
        laws.push_back(std::move(law));
    }
    return laws;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
