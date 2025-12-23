/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_SINGULARQUADRATURE_H
#define SVMP_FE_QUADRATURE_SINGULARQUADRATURE_H

/**
 * @file SingularQuadrature.h
 * @brief Special quadrature constructions for weak singularities
 */

#include "QuadratureRule.h"
#include "GaussQuadrature.h"
#include <memory>

namespace svmp {
namespace FE {
namespace quadrature {

class SingularQuadrature {
public:
    /// Duffy-transformed rule targeting a corner singularity on a triangle
    static std::unique_ptr<QuadratureRule> duffy_triangle(int order);

    /// Duffy-transformed rule targeting a vertex singularity on a tetrahedron
    static std::unique_ptr<QuadratureRule> duffy_tetrahedron(int order);
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_SINGULARQUADRATURE_H
