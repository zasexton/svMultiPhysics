/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/VectorElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::basis::BasisFactory;
using svmp::FE::basis::BasisRequest;
using svmp::FE::quadrature::QuadratureFactory;

VectorElement::VectorElement(ElementType element_type,
                             int order,
                             Continuity continuity)
    : VectorElement(element_type, order, continuity, BasisType::Lagrange) {}

VectorElement::VectorElement(ElementType element_type,
                             int order,
                             Continuity continuity,
                             BasisType basis_type) {
    if (continuity != Continuity::H_div && continuity != Continuity::H_curl) {
        throw FEException("VectorElement requires H_div or H_curl continuity",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    info_.element_type = element_type;
    info_.field_type   = FieldType::Vector;
    info_.continuity   = continuity;
    info_.order        = order;
    basis_type_        = basis_type;

    if (info_.order < 0) {
        throw FEException("VectorElement requires non-negative polynomial order",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    dimension_ = element_dimension(element_type);
    if (dimension_ < 0) {
        throw FEException("VectorElement: unknown element dimension",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    BasisRequest req;
    req.element_type = element_type;
    req.basis_type   = basis_type_;
    req.order        = info_.order;
    req.continuity   = continuity;
    req.field_type   = FieldType::Vector;

    basis_ = BasisFactory::create(req);
    if (!basis_) {
        throw FEException("VectorElement: BasisFactory returned null basis",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    num_dofs_ = basis_->size();

    const int quad_order = QuadratureFactory::recommended_order(info_.order, /*is_mass_matrix=*/false);
    quad_ = QuadratureFactory::create(element_type, quad_order);
    if (!quad_) {
        throw FEException("VectorElement: QuadratureFactory returned null rule",
                          __FILE__, __LINE__, __func__, FEStatus::QuadratureError);
    }
}

} // namespace elements
} // namespace FE
} // namespace svmp
