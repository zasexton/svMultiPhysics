/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_NITSCHE_TRACE_BILINEAR_FORM_H
#define SVMP_FE_FORMS_NITSCHE_TRACE_BILINEAR_FORM_H

/**
 * @file NitscheTraceBilinearForm.h
 * @brief The bilinear forms a generated-boundary Nitsche route can be consistent with
 *
 * Shared by the route's form helper (BoundaryConditions.h), the trace policy
 * of the system and the aggregate trace certificate.
 */

#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * The bilinear form a generated-boundary Nitsche route is consistent with.
 *
 * SymmetricGradient: a(u, v) = integral 2 k eps(u) : eps(v) for a finite
 * positive constant coefficient k; its boundary flux is
 * sigma(u) n = 2 k eps(u) n.  The route adds the consistency terms of that
 * flux and the penalty (gamma p^2 k / h_n) (u - g) . v, and the aggregate
 * trace certificate bounds the flux by the energy of the same form.
 */
enum class NitscheTraceBilinearForm : std::uint8_t {
    SymmetricGradient = 1u,
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_NITSCHE_TRACE_BILINEAR_FORM_H
