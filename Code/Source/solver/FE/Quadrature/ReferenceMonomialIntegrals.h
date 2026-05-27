/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_REFERENCEMONOMIALINTEGRALS_H
#define SVMP_FE_QUADRATURE_REFERENCEMONOMIALINTEGRALS_H

#include "Core/Types.h"

namespace svmp {
namespace FE {
namespace quadrature {
namespace reference_integrals {

// Integral of x^p over the symmetric reference interval [-1, 1].
Real integral_monomial_1d(int p);

// Integral of x^px y^py over the unit reference triangle
// {(x,y): x >= 0, y >= 0, x + y <= 1}.
Real integral_triangle_monomial(int px, int py);

// Integral of x^px y^py z^pz over the unit reference tetrahedron.
Real integral_tetra_monomial(int px, int py, int pz);

// Integral of z^power over the reference pyramid apex coordinate z in [0, 1].
Real integral_pyramid_z(int power);

// Product-domain integral over unit triangle(x,y) x symmetric interval [-1, 1](z).
Real integral_wedge_monomial(int px, int py, int pz);

} // namespace reference_integrals
} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_REFERENCEMONOMIALINTEGRALS_H
