/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ElementFactory.h"

#include "Core/FEException.h"
#include "Basis/BasisFactory.h"
#include "Elements/LagrangeElement.h"
#include "Elements/DiscontinuousElement.h"
#include "Elements/GeneralBasisElement.h"
#include "Elements/VectorElement.h"
#include "Elements/SpectralElement.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace elements {

namespace {

int require_element_order(const ElementRequest& req,
                          const char* missing_message) {
    if (!req.order.has_value()) {
        throw FEException(missing_message,
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return *req.order;
}

bool uses_generic_scalar_basis_host(BasisType basis_type) {
    switch (basis_type) {
        case BasisType::Hierarchical:
        case BasisType::Bernstein:
        case BasisType::Serendipity:
        case BasisType::Hermite:
        case BasisType::Bubble:
        case BasisType::BSpline:
        case BasisType::NURBS:
        case BasisType::Custom:
            return true;
        default:
            return false;
    }
}

Continuity effective_scalar_continuity(const ElementRequest& req) {
    if (req.continuity == Continuity::H_div || req.continuity == Continuity::H_curl) {
        throw FEException("ElementFactory: scalar basis families cannot be tagged as H(div)/H(curl)",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (req.basis_type == BasisType::Hermite) {
        if (req.continuity == Continuity::C0 || req.continuity == Continuity::C1) {
            return Continuity::C1;
        }
        throw FEException("ElementFactory: BasisType::Hermite requires Continuity::C1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (req.continuity == Continuity::C1) {
        throw FEException("ElementFactory: requested scalar basis family is not exposed through Continuity::C1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    return req.continuity;
}

int scalar_basis_quadrature_order(const basis::BasisFunction& basis) {
    switch (basis.basis_type()) {
        case BasisType::BSpline:
            return std::max(2, 2 * basis.order() + 1);
        case BasisType::NURBS:
            // Rational integrands are not polynomially exact; use mild over-integration.
            return std::max(2, 2 * basis.order() + 3);
        case BasisType::Bubble:
        case BasisType::Hermite:
        case BasisType::Hierarchical:
        case BasisType::Bernstein:
        case BasisType::Serendipity:
        default:
            return quadrature::QuadratureFactory::recommended_order(basis.order(), true);
    }
}

int vector_basis_quadrature_order(const basis::BasisFunction& basis) {
    switch (basis.basis_type()) {
        case BasisType::BSpline:
            return std::max(2, 2 * basis.order() + 1);
        case BasisType::NURBS:
            return std::max(2, 2 * basis.order() + 3);
        default:
            return quadrature::QuadratureFactory::recommended_order(basis.order(), false);
    }
}

std::shared_ptr<Element> create_generic_scalar_element(const ElementRequest& req) {
    if (req.field_type != FieldType::Scalar) {
        throw FEException("ElementFactory: generic scalar basis elements require FieldType::Scalar",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const Continuity continuity = effective_scalar_continuity(req);

    basis::BasisRequest basis_req;
    basis_req.element_type = req.element_type;
    basis_req.basis_type = req.basis_type;
    basis_req.order = req.order;
    basis_req.continuity = continuity;
    basis_req.field_type = FieldType::Scalar;
    basis_req.knot_vector = req.knot_vector;
    basis_req.weights = req.weights;
    basis_req.axis_orders = req.axis_orders;
    basis_req.axis_knot_vectors = req.axis_knot_vectors;
    basis_req.axis_weights = req.axis_weights;
    basis_req.tensor_extents = req.tensor_extents;
    basis_req.custom_id = req.custom_id;

    auto basis = basis::BasisFactory::create(basis_req);
    if (!basis || basis->is_vector_valued()) {
        throw FEException("ElementFactory: BasisFactory did not return a scalar basis",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const int quad_order = scalar_basis_quadrature_order(*basis);
    auto quad = quadrature::QuadratureFactory::create(
        req.element_type, quad_order, QuadratureType::GaussLegendre, true);

    return std::make_shared<GeneralBasisElement>(
        std::move(basis), std::move(quad), FieldType::Scalar, continuity);
}

std::shared_ptr<Element> create_generic_vector_element(const ElementRequest& req,
                                                       Continuity continuity) {
    FE_CHECK_ARG(req.field_type == FieldType::Vector,
                 "ElementFactory: generic vector basis elements require FieldType::Vector");
    FE_CHECK_ARG(continuity == Continuity::H_div || continuity == Continuity::H_curl,
                 "ElementFactory: generic vector basis elements require H(div) or H(curl) continuity");

    basis::BasisRequest basis_req;
    basis_req.element_type = req.element_type;
    basis_req.basis_type = req.basis_type;
    basis_req.order = req.order;
    basis_req.continuity = continuity;
    basis_req.field_type = FieldType::Vector;
    basis_req.knot_vector = req.knot_vector;
    basis_req.weights = req.weights;
    basis_req.axis_orders = req.axis_orders;
    basis_req.axis_knot_vectors = req.axis_knot_vectors;
    basis_req.axis_weights = req.axis_weights;
    basis_req.tensor_extents = req.tensor_extents;
    basis_req.custom_id = req.custom_id;

    auto basis = basis::BasisFactory::create(basis_req);
    if (!basis || !basis->is_vector_valued()) {
        throw FEException("ElementFactory: BasisFactory did not return a vector basis",
                          __FILE__, __LINE__, __func__, FEStatus::AssemblyError);
    }

    const int quad_order = vector_basis_quadrature_order(*basis);
    auto quad = quadrature::QuadratureFactory::create(
        req.element_type, quad_order, QuadratureType::GaussLegendre, true);

    return std::make_shared<GeneralBasisElement>(
        std::move(basis), std::move(quad), FieldType::Vector, continuity);
}

} // namespace

std::shared_ptr<Element> ElementFactory::create(const ElementRequest& req) {
    if (req.element_type == ElementType::Unknown) {
        throw FEException("ElementFactory: ElementType::Unknown is not allowed",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Validate polynomial order early
    if (req.order.has_value() && *req.order < 0) {
        throw FEException("ElementFactory: negative polynomial order is not allowed",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // Mixed and composite elements are typically constructed explicitly by
    // higher-level code and are not created by this factory.

    // ---------------------------------------------------------------------
    // Vector-valued conforming elements (H(div) / H(curl))
    //
    // The VectorElement internally delegates basis selection to BasisFactory
    // using (continuity, order). For user convenience, allow callers to
    // specify either:
    //   - req.continuity = H_div / H_curl (req.basis_type may be Lagrange), or
    //   - req.basis_type = RaviartThomas / BDM / Nedelec (continuity inferred).
    //
    // For the vector-specific basis types, we validate that the requested
    // order matches the currently implemented families to avoid silently
    // returning a different basis than requested.
    // ---------------------------------------------------------------------

    Continuity effective_continuity = req.continuity;
    if (req.basis_type == BasisType::Nedelec) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_curl) {
            throw FEException("ElementFactory: BasisType::Nedelec conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_curl;
    } else if (req.basis_type == BasisType::RaviartThomas) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_div) {
            throw FEException("ElementFactory: BasisType::RaviartThomas conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_div;
    } else if (req.basis_type == BasisType::BDM) {
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::H_div) {
            throw FEException("ElementFactory: BasisType::BDM conflicts with requested continuity",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        effective_continuity = Continuity::H_div;
    }

    if (effective_continuity == Continuity::H_div || effective_continuity == Continuity::H_curl) {
        if (req.field_type != FieldType::Vector) {
            throw FEException("ElementFactory: H(div)/H(curl) elements require FieldType::Vector",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        const int order = require_element_order(
            req, "ElementFactory: H(div)/H(curl) elements require an explicit order");

        if (effective_continuity == Continuity::H_div) {
            const bool ok_basis =
                (req.basis_type == BasisType::Lagrange) ||
                (req.basis_type == BasisType::RaviartThomas) ||
                (req.basis_type == BasisType::BDM) ||
                (req.basis_type == BasisType::BSpline) ||
                (req.basis_type == BasisType::NURBS);
            if (!ok_basis) {
                throw FEException("ElementFactory: H(div) elements require BasisType::Lagrange/::RaviartThomas/::BDM/::BSpline/::NURBS",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
        } else {
            const bool ok_basis =
                (req.basis_type == BasisType::Lagrange) ||
                (req.basis_type == BasisType::Nedelec) ||
                (req.basis_type == BasisType::BSpline) ||
                (req.basis_type == BasisType::NURBS);
            if (!ok_basis) {
                throw FEException("ElementFactory: H(curl) elements require BasisType::Lagrange/::Nedelec/::BSpline/::NURBS",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
        }

        if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
            return create_generic_vector_element(req, effective_continuity);
        }

        return std::make_shared<VectorElement>(req.element_type, order, effective_continuity, req.basis_type);
    }

    if (req.basis_type == BasisType::Lagrange) {
        const int order = require_element_order(
            req, "ElementFactory: Lagrange element requests require an explicit order");
        if (req.continuity == Continuity::L2) {
            // DG scalar element
            if (req.field_type != FieldType::Scalar) {
                throw FEException("ElementFactory: DG elements currently support scalar fields only",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }
            return std::make_shared<DiscontinuousElement>(req.element_type, order, req.field_type);
        }

        // Standard scalar H¹ Lagrange element
        if (req.field_type != FieldType::Scalar) {
            throw FEException("ElementFactory: scalar LagrangeElement requires FieldType::Scalar",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<LagrangeElement>(req.element_type, order, req.field_type, req.continuity);
    }

    if (uses_generic_scalar_basis_host(req.basis_type)) {
        return create_generic_scalar_element(req);
    }

    if (req.basis_type == BasisType::Spectral) {
        const int order = require_element_order(
            req, "ElementFactory: Spectral element requests require an explicit order");
        if (req.field_type != FieldType::Scalar) {
            throw FEException("ElementFactory: SpectralElement currently supports scalar fields only",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (req.continuity != Continuity::C0 && req.continuity != Continuity::L2) {
            throw FEException("ElementFactory: Spectral elements require Continuity::C0 or Continuity::L2",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<SpectralElement>(
            req.element_type, order, req.field_type, req.continuity);
    }

    throw FEException("ElementFactory: requested basis type not supported by factory",
                      __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
}

} // namespace elements
} // namespace FE
} // namespace svmp
