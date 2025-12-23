/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/SpectralElement.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::basis::BasisFactory;
using svmp::FE::basis::BasisRequest;
using svmp::FE::quadrature::QuadratureFactory;
using svmp::FE::QuadratureType;

SpectralElement::SpectralElement(ElementType element_type,
                                 int order,
                                 FieldType field_type) {
    if (field_type != FieldType::Scalar) {
        throw FEException("SpectralElement currently supports scalar fields only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    info_.element_type = element_type;
    info_.field_type   = field_type;
    info_.continuity   = Continuity::C0;
    info_.order        = order;

    // Spectral elements require at least order 1 for well-defined Gauss-Lobatto nodes
    if (info_.order < 1) {
        throw FEException("SpectralElement requires polynomial order >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    dimension_ = element_dimension(element_type);
    if (dimension_ < 0) {
        throw FEException("SpectralElement: unknown element dimension",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    BasisRequest req;
    req.element_type = element_type;
    req.basis_type   = BasisType::Spectral;
    req.order        = info_.order;
    req.continuity   = Continuity::C0;
    req.field_type   = FieldType::Scalar;

    basis_ = BasisFactory::create(req);
    if (!basis_) {
        throw FEException("SpectralElement: BasisFactory returned null basis",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    num_dofs_ = basis_->size();

    // For collocation (diagonal mass matrix), use Gauss-Lobatto quadrature with
    // (p+1) points per coordinate, i.e. polynomial exactness 2p-1.
    const int quad_order = 2 * info_.order - 1;
    quad_ = QuadratureFactory::create(element_type, quad_order, QuadratureType::GaussLobatto);
    if (!quad_) {
        throw FEException("SpectralElement: QuadratureFactory returned null rule",
                          __FILE__, __LINE__, __func__, FEStatus::QuadratureError);
    }
}

} // namespace elements
} // namespace FE
} // namespace svmp
