/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/LagrangeElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::basis::BasisFactory;
using svmp::FE::basis::BasisRequest;
using svmp::FE::quadrature::QuadratureFactory;

LagrangeElement::LagrangeElement(ElementType element_type,
                                 int order,
                                 FieldType field_type,
                                 Continuity continuity) {
    if (field_type != FieldType::Scalar) {
        throw FEException("LagrangeElement currently supports scalar fields only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    info_.element_type = element_type;
    info_.field_type   = field_type;
    info_.continuity   = continuity;
    info_.order        = order;

    if (info_.order < 0) {
        throw FEException("LagrangeElement requires non-negative polynomial order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    dimension_ = element_dimension(element_type);
    if (dimension_ < 0) {
        throw FEException("LagrangeElement: unknown element dimension",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    // Build scalar Lagrange basis via BasisFactory
    BasisRequest req;
    req.element_type = element_type;
    req.basis_type   = BasisType::Lagrange;
    req.order        = info_.order;
    req.continuity   = continuity;
    req.field_type   = FieldType::Scalar;

    basis_ = BasisFactory::create(req);
    if (!basis_) {
        throw FEException("LagrangeElement: BasisFactory returned null basis",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    num_dofs_ = basis_->size(); // scalar field: one DOF per basis function

    // Choose a quadrature order suitable for stiffness matrices by default
    const int quad_order = QuadratureFactory::recommended_order(info_.order, /*is_mass_matrix=*/false);
    quad_ = QuadratureFactory::create(element_type, quad_order);
    if (!quad_) {
        throw FEException("LagrangeElement: QuadratureFactory returned null rule",
                          __FILE__, __LINE__, __func__, FEStatus::QuadratureError);
    }
}

} // namespace elements
} // namespace FE
} // namespace svmp

