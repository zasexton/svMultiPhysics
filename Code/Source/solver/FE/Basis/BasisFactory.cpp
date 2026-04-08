/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisFactory.h"

namespace svmp {
namespace FE {
namespace basis {

namespace {

int spline_tensor_dimension(ElementType element_type) {
    if (element_type == ElementType::Line2) {
        return 1;
    }
    if (element_type == ElementType::Quad4) {
        return 2;
    }
    if (element_type == ElementType::Hex8) {
        return 3;
    }
    return 0;
}

void validate_scalar_spline_request(const BasisRequest& req) {
    if (req.field_type != FieldType::Scalar) {
        throw FEException("BasisFactory: spline bases currently support scalar fields only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (req.continuity == Continuity::H_div || req.continuity == Continuity::H_curl) {
        throw FEException("BasisFactory: spline bases do not support H(div)/H(curl) continuity",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (req.continuity == Continuity::C1) {
        throw FEException("BasisFactory: spline bases are not exposed through the C1 continuity path",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

BSplineBasis make_bspline_axis(const BasisRequest& req,
                               std::size_t axis) {
    const bool use_axis_knots = !req.axis_knot_vectors.empty();
    const bool use_axis_orders = !req.axis_orders.empty();

    const int degree = use_axis_orders
        ? req.axis_orders[axis]
        : req.order;
    const std::vector<Real>& knots = use_axis_knots
        ? req.axis_knot_vectors[axis]
        : req.knot_vector;

    if (knots.empty()) {
        throw FEException("BasisFactory: spline knot vectors are required for BSpline creation",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return BSplineBasis(degree, knots);
}

std::shared_ptr<BasisFunction> create_bspline_basis(const BasisRequest& req) {
    validate_scalar_spline_request(req);

    if (!req.weights.empty()) {
        throw FEException("BasisFactory: non-rational BSpline request must not provide weights",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (!req.axis_weights.empty()) {
        throw FEException("BasisFactory: non-rational BSpline request must not provide axis weights",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    const int dim = spline_tensor_dimension(req.element_type);
    if (dim == 0) {
        throw FEException("BasisFactory: BSpline currently supports Line2, Quad4, and Hex8 only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    const bool use_axis_data = !req.axis_orders.empty() || !req.axis_knot_vectors.empty();
    if (use_axis_data) {
        if (!req.axis_orders.empty() && req.axis_orders.size() != static_cast<std::size_t>(dim)) {
            throw FEException("BasisFactory: axis_orders size must match spline tensor dimension",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        if (!req.axis_knot_vectors.empty() && req.axis_knot_vectors.size() != static_cast<std::size_t>(dim)) {
            throw FEException("BasisFactory: axis_knot_vectors size must match spline tensor dimension",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
    } else if (req.knot_vector.empty()) {
        throw FEException("BasisFactory: spline knot vectors are required for BSpline creation",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (dim == 1) {
        return std::make_shared<BSplineBasis>(make_bspline_axis(req, 0));
    }

    if (dim == 2) {
        if (use_axis_data) {
            return std::make_shared<TensorProductBasis<BSplineBasis>>(
                make_bspline_axis(req, 0),
                make_bspline_axis(req, 1));
        }
        return std::make_shared<TensorProductBasis<BSplineBasis>>(
            BSplineBasis(req.order, req.knot_vector),
            2);
    }

    if (use_axis_data) {
        return std::make_shared<TensorProductBasis<BSplineBasis>>(
            make_bspline_axis(req, 0),
            make_bspline_axis(req, 1),
            make_bspline_axis(req, 2));
    }
    return std::make_shared<TensorProductBasis<BSplineBasis>>(
        BSplineBasis(req.order, req.knot_vector),
        3);
}

std::shared_ptr<BasisFunction> create_nurbs_basis(const BasisRequest& req) {
    validate_scalar_spline_request(req);

    if (req.element_type != ElementType::Line2) {
        throw FEException(
            "BasisFactory: NURBS currently supports Line2 only; multi-dimensional rational tensor-product support requires control-point weights beyond the current separable basis API",
            __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }
    if (req.knot_vector.empty()) {
        throw FEException("BasisFactory: NURBS creation requires a knot vector",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (req.weights.empty()) {
        throw FEException("BasisFactory: NURBS creation requires weights",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    if (!req.axis_orders.empty() || !req.axis_knot_vectors.empty() || !req.axis_weights.empty()) {
        throw FEException("BasisFactory: Line2 NURBS does not accept per-axis spline descriptors",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
    return std::make_shared<BSplineBasis>(req.order, req.knot_vector, req.weights);
}

} // namespace

std::shared_ptr<BasisFunction> BasisFactory::create(const BasisRequest& req) {
    // Vector-valued conforming bases take precedence
    if (req.continuity == Continuity::H_div) {
        // Allow explicit family selection via BasisType.
        if (req.basis_type == BasisType::RaviartThomas) {
            return std::make_shared<RaviartThomasBasis>(req.element_type, req.order);
        }
        if (req.basis_type == BasisType::BDM) {
            return std::make_shared<BDMBasis>(req.element_type, req.order);
        }

        // Default selection (BasisType::Lagrange): keep the historical choice of BDM(1) on 2D elements,
        // but fall back to Raviart-Thomas when BDM is not applicable (e.g., 3D tensor-product RT(1)+).
        const int dim = element_dimension(req.element_type);
        if (dim == 2 && req.order == 1) {
            return std::make_shared<BDMBasis>(req.element_type, 1);
        }
        return std::make_shared<RaviartThomasBasis>(req.element_type, req.order);
    }

    if (req.continuity == Continuity::H_curl) {
        if (req.basis_type != BasisType::Lagrange && req.basis_type != BasisType::Nedelec) {
            throw FEException("BasisFactory: H(curl) bases require BasisType::Lagrange or BasisType::Nedelec",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }
        return std::make_shared<NedelecBasis>(req.element_type, req.order);
    }

    // L² (discontinuous) uses the same reference-space shape functions as C0.
    // DOF ownership (element-local vs shared) is managed at the Space/Element level.
    // Fall through to the BasisType switch intentionally.
    if (req.continuity == Continuity::L2) {
        // Intentional fall-through to BasisType dispatch below
    }

    // C¹ scalar bases currently use the Hermite family.
    else if (req.continuity == Continuity::C1) {
        if (req.field_type == FieldType::Scalar) {
            return std::make_shared<HermiteBasis>(req.element_type, req.order);
        }
        throw FEException("BasisFactory: C1 continuity currently supports scalar fields only",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    switch (req.basis_type) {
        case BasisType::Lagrange:
            return std::make_shared<LagrangeBasis>(req.element_type, req.order);
        case BasisType::Hierarchical:
            return std::make_shared<HierarchicalBasis>(req.element_type, req.order);
        case BasisType::Bernstein:
            return std::make_shared<BernsteinBasis>(req.element_type, req.order);
        case BasisType::BSpline:
            return create_bspline_basis(req);
        case BasisType::NURBS:
            return create_nurbs_basis(req);
        case BasisType::Spectral:
            return std::make_shared<SpectralBasis>(req.element_type, req.order);
        case BasisType::Serendipity:
            return std::make_shared<SerendipityBasis>(req.element_type, req.order);
        case BasisType::Hermite:
            return std::make_shared<HermiteBasis>(req.element_type, req.order);
        case BasisType::Bubble:
            return std::make_shared<BubbleBasis>(req.element_type);
        default:
            throw FEException("Unsupported basis type in BasisFactory",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
