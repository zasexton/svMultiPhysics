/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_REFERENCEMONOMIALINTEGRALS_H
#define SVMP_FE_BASIS_REFERENCEMONOMIALINTEGRALS_H

#include "Quadrature/ReferenceMonomialIntegrals.h"

namespace svmp {
namespace FE {
namespace basis {
namespace detail {
namespace reference_integrals {

using ::svmp::FE::quadrature::reference_integrals::integral_monomial_1d;
using ::svmp::FE::quadrature::reference_integrals::integral_pyramid_z;
using ::svmp::FE::quadrature::reference_integrals::integral_tetra_monomial;
using ::svmp::FE::quadrature::reference_integrals::integral_triangle_monomial;
using ::svmp::FE::quadrature::reference_integrals::integral_wedge_monomial;

} // namespace reference_integrals
} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_REFERENCEMONOMIALINTEGRALS_H
