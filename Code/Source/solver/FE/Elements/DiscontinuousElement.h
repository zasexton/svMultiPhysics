/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_DISCONTINUOUSELEMENT_H
#define SVMP_FE_ELEMENTS_DISCONTINUOUSELEMENT_H

/**
 * @file DiscontinuousElement.h
 * @brief L²-discontinuous scalar elements (DG-style)
 */

#include "Elements/LagrangeElement.h"

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Scalar discontinuous (L²) element built from a Lagrange basis
 *
 * This is a thin wrapper around LagrangeElement that simply changes the
 * continuity classification to L². The local approximation space and
 * quadrature are identical; global DOF management determines whether DOFs
 * are shared across element boundaries.
 */
class DiscontinuousElement : public LagrangeElement {
public:
    DiscontinuousElement(ElementType element_type,
                         int order,
                         FieldType field_type = FieldType::Scalar)
        : LagrangeElement(element_type, order, field_type, Continuity::L2) {}
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_DISCONTINUOUSELEMENT_H

