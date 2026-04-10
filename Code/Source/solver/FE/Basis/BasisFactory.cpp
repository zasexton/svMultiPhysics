/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisFactory.h"
#include "Core/FEException.h"

#include <array>
#include <mutex>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace basis {

namespace {

int require_basis_order(const BasisRequest& req,
                        const char* missing_message,
                        const char* negative_message) {
    if (!req.order.has_value()) {
        throw BasisConfigurationException(missing_message,
                                          __FILE__, __LINE__, __func__);
    }
    if (*req.order < 0) {
        throw BasisConfigurationException(negative_message,
                                          __FILE__, __LINE__, __func__);
    }
    return *req.order;
}

void reject_explicit_basis_order(const BasisRequest& req,
                                 const char* message) {
    if (req.order.has_value()) {
        throw BasisConfigurationException(message,
                                          __FILE__, __LINE__, __func__);
    }
}

using CustomRegistryMap = std::unordered_map<std::string, BasisFactory::CustomFactory>;

CustomRegistryMap& custom_registry() {
    static CustomRegistryMap registry;
    return registry;
}

std::mutex& custom_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

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

void validate_vector_factory_request(const BasisRequest& req,
                                     Continuity continuity) {
    if (req.field_type != FieldType::Vector) {
        throw BasisConfigurationException("BasisFactory: H(div)/H(curl) bases require FieldType::Vector",
                                          __FILE__, __LINE__, __func__);
    }

    if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
        if (!req.axis_orders.empty()) {
            for (int axis_order : req.axis_orders) {
                if (axis_order < 1) {
                    throw BasisConfigurationException("BasisFactory: compatible spline/NURBS vector bases require axis order >= 1",
                                                      __FILE__, __LINE__, __func__);
                }
            }
        } else {
            (void)require_basis_order(req,
                                      "BasisFactory: vector-conforming bases require an explicit order when axis_orders are omitted",
                                      "BasisFactory: vector-conforming bases require non-negative order");
            if (*req.order < 1) {
                throw BasisConfigurationException("BasisFactory: compatible spline/NURBS vector bases require order >= 1",
                                                  __FILE__, __LINE__, __func__);
            }
        }
    } else {
        (void)require_basis_order(req,
                                  "BasisFactory: vector-conforming bases require an explicit order",
                                  "BasisFactory: vector-conforming bases require non-negative order");
    }

    if (continuity == Continuity::H_div) {
        if (req.basis_type != BasisType::Lagrange &&
            req.basis_type != BasisType::RaviartThomas &&
            req.basis_type != BasisType::BDM &&
            req.basis_type != BasisType::BSpline &&
            req.basis_type != BasisType::NURBS) {
            throw BasisConfigurationException("BasisFactory: H(div) bases require BasisType::Lagrange, BasisType::RaviartThomas, BasisType::BDM, BasisType::BSpline, or BasisType::NURBS",
                                              __FILE__, __LINE__, __func__);
        }
        return;
    }

    if (req.basis_type != BasisType::Lagrange &&
        req.basis_type != BasisType::Nedelec &&
        req.basis_type != BasisType::BSpline &&
        req.basis_type != BasisType::NURBS) {
        throw BasisConfigurationException("BasisFactory: H(curl) bases require BasisType::Lagrange, BasisType::Nedelec, BasisType::BSpline, or BasisType::NURBS",
                                          __FILE__, __LINE__, __func__);
    }
}

void validate_scalar_spline_request(const BasisRequest& req) {
    if (req.field_type != FieldType::Scalar) {
        throw BasisConfigurationException("BasisFactory: spline bases currently support scalar fields only",
                                          __FILE__, __LINE__, __func__);
    }
    if (req.continuity == Continuity::H_div || req.continuity == Continuity::H_curl) {
        throw BasisConfigurationException("BasisFactory: spline bases do not support H(div)/H(curl) continuity",
                                          __FILE__, __LINE__, __func__);
    }
    if (req.continuity == Continuity::C1) {
        throw BasisConfigurationException("BasisFactory: spline bases are not exposed through the C1 continuity path",
                                          __FILE__, __LINE__, __func__);
    }
}

BSplineBasis make_bspline_axis(const BasisRequest& req,
                               std::size_t axis) {
    const bool use_axis_knots = !req.axis_knot_vectors.empty();
    const bool use_axis_orders = !req.axis_orders.empty();

    const int degree = use_axis_orders
        ? req.axis_orders[axis]
        : require_basis_order(req,
                              "BasisFactory: BSpline creation requires an explicit order when axis_orders are omitted",
                              "BasisFactory: BSpline requires non-negative order");
    const std::vector<Real>& knots = use_axis_knots
        ? req.axis_knot_vectors[axis]
        : req.knot_vector;

    if (knots.empty()) {
        throw BasisConfigurationException("BasisFactory: spline knot vectors are required for BSpline creation",
                                          __FILE__, __LINE__, __func__);
    }
    return BSplineBasis(degree, knots);
}

std::shared_ptr<BasisFunction> create_bspline_basis(const BasisRequest& req) {
    validate_scalar_spline_request(req);

    if (!req.weights.empty()) {
        throw BasisConfigurationException("BasisFactory: non-rational BSpline request must not provide weights",
                                          __FILE__, __LINE__, __func__);
    }
    if (!req.axis_weights.empty()) {
        throw BasisConfigurationException("BasisFactory: non-rational BSpline request must not provide axis weights",
                                          __FILE__, __LINE__, __func__);
    }

    const int dim = spline_tensor_dimension(req.element_type);
    if (dim == 0) {
        throw BasisElementCompatibilityException("BasisFactory: BSpline currently supports Line2, Quad4, and Hex8 only",
                                                 __FILE__, __LINE__, __func__);
    }

    const bool use_axis_data = !req.axis_orders.empty() || !req.axis_knot_vectors.empty();
    if (use_axis_data) {
        if (!req.axis_orders.empty() && req.axis_orders.size() != static_cast<std::size_t>(dim)) {
            throw BasisConfigurationException("BasisFactory: axis_orders size must match spline tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
        if (!req.axis_knot_vectors.empty() && req.axis_knot_vectors.size() != static_cast<std::size_t>(dim)) {
            throw BasisConfigurationException("BasisFactory: axis_knot_vectors size must match spline tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
    } else if (req.knot_vector.empty()) {
        throw BasisConfigurationException("BasisFactory: spline knot vectors are required for BSpline creation",
                                          __FILE__, __LINE__, __func__);
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
            BSplineBasis(require_basis_order(req,
                                             "BasisFactory: BSpline creation requires an explicit order when axis_orders are omitted",
                                             "BasisFactory: BSpline requires non-negative order"),
                         req.knot_vector),
            2);
    }

    if (use_axis_data) {
        return std::make_shared<TensorProductBasis<BSplineBasis>>(
            make_bspline_axis(req, 0),
            make_bspline_axis(req, 1),
            make_bspline_axis(req, 2));
    }
    return std::make_shared<TensorProductBasis<BSplineBasis>>(
        BSplineBasis(require_basis_order(req,
                                         "BasisFactory: BSpline creation requires an explicit order when axis_orders are omitted",
                                         "BasisFactory: BSpline requires non-negative order"),
                     req.knot_vector),
        3);
}

std::shared_ptr<BasisFunction> create_nurbs_basis(const BasisRequest& req) {
    validate_scalar_spline_request(req);

    if (req.weights.empty()) {
        throw BasisConfigurationException("BasisFactory: NURBS creation requires weights",
                                          __FILE__, __LINE__, __func__);
    }
    if (!req.axis_weights.empty()) {
        throw BasisConfigurationException("BasisFactory: NURBS uses a flattened control-net weight array; per-axis axis_weights are not supported",
                                          __FILE__, __LINE__, __func__);
    }

    const int dim = spline_tensor_dimension(req.element_type);
    if (dim == 0) {
        throw BasisElementCompatibilityException("BasisFactory: NURBS currently supports Line2, Quad4, and Hex8 only",
                                                 __FILE__, __LINE__, __func__);
    }

    const bool use_axis_data = !req.axis_orders.empty() || !req.axis_knot_vectors.empty();
    if (use_axis_data) {
        if (!req.axis_orders.empty() && req.axis_orders.size() != static_cast<std::size_t>(dim)) {
            throw BasisConfigurationException("BasisFactory: axis_orders size must match spline tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
        if (!req.axis_knot_vectors.empty() && req.axis_knot_vectors.size() != static_cast<std::size_t>(dim)) {
            throw BasisConfigurationException("BasisFactory: axis_knot_vectors size must match spline tensor dimension",
                                              __FILE__, __LINE__, __func__);
        }
    } else if (req.knot_vector.empty()) {
        throw BasisConfigurationException("BasisFactory: NURBS creation requires knot vectors",
                                          __FILE__, __LINE__, __func__);
    }

    if (dim == 1) {
        if (!req.tensor_extents.empty()) {
            if (req.tensor_extents.size() != 1u) {
                throw BasisConfigurationException("BasisFactory: 1D NURBS tensor_extents must have size 1",
                                                  __FILE__, __LINE__, __func__);
            }
        }
        if (use_axis_data) {
            if (req.axis_orders.empty() || req.axis_knot_vectors.empty()) {
                throw BasisConfigurationException("BasisFactory: 1D NURBS axis descriptors require both axis_orders and axis_knot_vectors",
                                                  __FILE__, __LINE__, __func__);
            }
            return std::make_shared<BSplineBasis>(req.axis_orders[0],
                                                  req.axis_knot_vectors[0],
                                                  req.weights);
        }
        return std::make_shared<BSplineBasis>(
            require_basis_order(req,
                                "BasisFactory: 1D NURBS creation requires an explicit order when axis_orders are omitted",
                                "BasisFactory: NURBS requires non-negative order"),
            req.knot_vector,
            req.weights);
    }

    if (dim == 2) {
        return std::make_shared<NURBSTensorBasis>(
            make_bspline_axis(req, 0),
            make_bspline_axis(req, 1),
            req.weights,
            req.tensor_extents);
    }

    return std::make_shared<NURBSTensorBasis>(
        make_bspline_axis(req, 0),
        make_bspline_axis(req, 1),
        make_bspline_axis(req, 2),
        req.weights,
        req.tensor_extents);
}

std::vector<BSplineBasis> make_tensor_axes(const BasisRequest& req,
                                           int dim) {
    FE_CHECK_ARG(dim == 2,
                 "BasisFactory: compatible spline/NURBS vector bases currently support quadrilateral elements only");

    const bool use_axis_data = !req.axis_orders.empty() || !req.axis_knot_vectors.empty();
    if (use_axis_data) {
        FE_CHECK_ARG(req.axis_orders.empty() || req.axis_orders.size() == static_cast<std::size_t>(dim),
                     "BasisFactory: vector spline/NURBS axis_orders size must match tensor dimension");
        FE_CHECK_ARG(req.axis_knot_vectors.empty() || req.axis_knot_vectors.size() == static_cast<std::size_t>(dim),
                     "BasisFactory: vector spline/NURBS axis_knot_vectors size must match tensor dimension");
    } else {
        FE_CHECK_ARG(!req.knot_vector.empty(),
                     "BasisFactory: vector spline/NURBS construction requires knot vectors");
    }

    std::vector<BSplineBasis> axes;
    axes.reserve(static_cast<std::size_t>(dim));
    for (int axis = 0; axis < dim; ++axis) {
        axes.push_back(make_bspline_axis(req, static_cast<std::size_t>(axis)));
    }
    return axes;
}

BSplineBasis make_reduced_bspline_axis(const BSplineBasis& axis) {
    FE_CHECK_ARG(axis.order() >= 1,
                 "BasisFactory: compatible spline/NURBS vector bases require axis order >= 1");

    auto knots = axis.knots();
    FE_CHECK_ARG(knots.size() >= 2u,
                 "BasisFactory: spline knot vector too short for compatible vector basis");
    knots.erase(knots.begin());
    knots.pop_back();
    return BSplineBasis(axis.order() - 1, std::move(knots));
}

std::array<int, 2> infer_quad_tensor_extents(const std::vector<BSplineBasis>& axes) {
    FE_CHECK_ARG(axes.size() == 2u,
                 "BasisFactory: compatible tensor extents require two axes");
    return {static_cast<int>(axes[0].size()), static_cast<int>(axes[1].size())};
}

std::vector<Real> reduce_quad_weights(const std::vector<Real>& weights,
                                      const std::array<int, 2>& extents,
                                      int axis) {
    FE_CHECK_ARG(extents[0] > 1 && extents[1] > 1,
                 "BasisFactory: compatible spline/NURBS vector bases require at least two control points per axis");
    FE_CHECK_ARG(axis == 0 || axis == 1,
                 "BasisFactory: invalid reduced tensor axis");

    const int nx = extents[0];
    const int ny = extents[1];
    FE_CHECK_ARG(weights.size() == static_cast<std::size_t>(nx * ny),
                 "BasisFactory: NURBS weights size does not match tensor extents");

    if (axis == 0) {
        std::vector<Real> reduced(static_cast<std::size_t>((nx - 1) * ny), Real(0));
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                const std::size_t dst = static_cast<std::size_t>(j * (nx - 1) + i);
                const std::size_t src0 = static_cast<std::size_t>(j * nx + i);
                const std::size_t src1 = static_cast<std::size_t>(j * nx + (i + 1));
                reduced[dst] = Real(0.5) * (weights[src0] + weights[src1]);
            }
        }
        return reduced;
    }

    std::vector<Real> reduced(static_cast<std::size_t>(nx * (ny - 1)), Real(0));
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::size_t dst = static_cast<std::size_t>(j * nx + i);
            const std::size_t src0 = static_cast<std::size_t>(j * nx + i);
            const std::size_t src1 = static_cast<std::size_t>((j + 1) * nx + i);
            reduced[dst] = Real(0.5) * (weights[src0] + weights[src1]);
        }
    }
    return reduced;
}

std::shared_ptr<BasisFunction> make_compatible_quad_component_basis(
    BasisType semantic_basis_type,
    const std::vector<BSplineBasis>& base_axes,
    const std::array<int, 2>& base_extents,
    const std::vector<Real>& weights,
    bool reduce_x,
    bool reduce_y) {
    FE_CHECK_ARG(base_axes.size() == 2u,
                 "BasisFactory: compatible quad component basis requires two axes");

    const BSplineBasis ax = reduce_x ? make_reduced_bspline_axis(base_axes[0]) : base_axes[0];
    const BSplineBasis ay = reduce_y ? make_reduced_bspline_axis(base_axes[1]) : base_axes[1];

    if (semantic_basis_type == BasisType::BSpline) {
        return std::make_shared<TensorProductBasis<BSplineBasis>>(ax, ay);
    }

    FE_CHECK_ARG(semantic_basis_type == BasisType::NURBS,
                 "BasisFactory: compatible quad component basis requires BSpline or NURBS semantics");

    std::vector<Real> component_weights = weights;
    std::array<int, 2> extents = base_extents;
    if (reduce_x) {
        component_weights = reduce_quad_weights(component_weights, extents, 0);
        extents[0] -= 1;
    }
    if (reduce_y) {
        component_weights = reduce_quad_weights(component_weights, extents, 1);
        extents[1] -= 1;
    }

    return std::make_shared<NURBSTensorBasis>(
        ax,
        ay,
        std::move(component_weights),
        std::vector<int>{extents[0], extents[1]});
}

std::vector<DofAssociation> build_quad_compatible_vector_associations(
    CompatibleTensorVectorBasis::Family family,
    const std::array<int, 2>& first_extents,
    const std::array<int, 2>& second_extents) {
    std::vector<DofAssociation> associations;
    associations.reserve(static_cast<std::size_t>(first_extents[0] * first_extents[1] +
                                                  second_extents[0] * second_extents[1]));

    int cell_moment = 0;

    if (family == CompatibleTensorVectorBasis::Family::HCurl) {
        const int nx = first_extents[0];
        const int ny = first_extents[1];
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                DofAssociation assoc{};
                if (j == 0) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = 0;
                    assoc.moment_index = i;
                } else if (j == ny - 1) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = 2;
                    assoc.moment_index = (nx - 1) - i;
                } else {
                    assoc.entity_type = DofEntity::Interior;
                    assoc.entity_id = 0;
                    assoc.moment_index = cell_moment++;
                }
                associations.push_back(assoc);
            }
        }

        const int mx = second_extents[0];
        const int my = second_extents[1];
        for (int j = 0; j < my; ++j) {
            for (int i = 0; i < mx; ++i) {
                DofAssociation assoc{};
                if (i == mx - 1) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = 1;
                    assoc.moment_index = j;
                } else if (i == 0) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = 3;
                    assoc.moment_index = (my - 1) - j;
                } else {
                    assoc.entity_type = DofEntity::Interior;
                    assoc.entity_id = 0;
                    assoc.moment_index = cell_moment++;
                }
                associations.push_back(assoc);
            }
        }
        return associations;
    }

    FE_CHECK_ARG(family == CompatibleTensorVectorBasis::Family::HDiv,
                 "BasisFactory: unsupported compatible vector family");

    const int nx = first_extents[0];
    const int ny = first_extents[1];
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            DofAssociation assoc{};
            if (i == nx - 1) {
                assoc.entity_type = DofEntity::Edge;
                assoc.entity_id = 1;
                assoc.moment_index = j;
            } else if (i == 0) {
                assoc.entity_type = DofEntity::Edge;
                assoc.entity_id = 3;
                assoc.moment_index = (ny - 1) - j;
            } else {
                assoc.entity_type = DofEntity::Interior;
                assoc.entity_id = 0;
                assoc.moment_index = cell_moment++;
            }
            associations.push_back(assoc);
        }
    }

    const int mx = second_extents[0];
    const int my = second_extents[1];
    for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
            DofAssociation assoc{};
            if (j == 0) {
                assoc.entity_type = DofEntity::Edge;
                assoc.entity_id = 0;
                assoc.moment_index = i;
            } else if (j == my - 1) {
                assoc.entity_type = DofEntity::Edge;
                assoc.entity_id = 2;
                assoc.moment_index = (mx - 1) - i;
            } else {
                assoc.entity_type = DofEntity::Interior;
                assoc.entity_id = 0;
                assoc.moment_index = cell_moment++;
            }
            associations.push_back(assoc);
        }
    }

    return associations;
}

std::shared_ptr<BasisFunction> create_compatible_tensor_vector_basis(
    const BasisRequest& req,
    Continuity continuity) {
    FE_CHECK_ARG(req.element_type == ElementType::Quad4,
                 "BasisFactory: compatible spline/NURBS H(div)/H(curl) bases currently support Quad4 only");

    const auto axes = make_tensor_axes(req, /*dim=*/2);
    const auto base_extents = infer_quad_tensor_extents(axes);
    FE_CHECK_ARG(base_extents[0] >= 2 && base_extents[1] >= 2,
                 "BasisFactory: compatible spline/NURBS vector bases require at least two basis functions per axis");

    if (req.basis_type == BasisType::NURBS && req.weights.empty()) {
        throw BasisConfigurationException("BasisFactory: vector NURBS construction requires weights",
                                          __FILE__, __LINE__, __func__);
    }

    const auto family = (continuity == Continuity::H_curl)
        ? CompatibleTensorVectorBasis::Family::HCurl
        : CompatibleTensorVectorBasis::Family::HDiv;

    const bool first_reduce_x =
        (family == CompatibleTensorVectorBasis::Family::HCurl);
    const bool first_reduce_y =
        (family == CompatibleTensorVectorBasis::Family::HDiv);
    const bool second_reduce_x =
        (family == CompatibleTensorVectorBasis::Family::HDiv);
    const bool second_reduce_y =
        (family == CompatibleTensorVectorBasis::Family::HCurl);

    auto first_basis = make_compatible_quad_component_basis(
        req.basis_type, axes, base_extents, req.weights, first_reduce_x, first_reduce_y);
    auto second_basis = make_compatible_quad_component_basis(
        req.basis_type, axes, base_extents, req.weights, second_reduce_x, second_reduce_y);

    const std::array<int, 2> first_extents = {
        base_extents[0] - (first_reduce_x ? 1 : 0),
        base_extents[1] - (first_reduce_y ? 1 : 0)
    };
    const std::array<int, 2> second_extents = {
        base_extents[0] - (second_reduce_x ? 1 : 0),
        base_extents[1] - (second_reduce_y ? 1 : 0)
    };

    auto associations = build_quad_compatible_vector_associations(family,
                                                                  first_extents,
                                                                  second_extents);

    const int order = std::max(axes[0].order(), axes[1].order());
    return std::make_shared<CompatibleTensorVectorBasis>(family,
                                                         req.basis_type,
                                                         std::move(first_basis),
                                                         std::move(second_basis),
                                                         std::move(associations),
                                                         order,
                                                         ElementType::Quad4);
}

} // namespace

std::shared_ptr<BasisFunction> BasisFactory::create(const BasisRequest& req) {
    // Vector-valued conforming bases take precedence
    if (req.continuity == Continuity::H_div) {
        validate_vector_factory_request(req, Continuity::H_div);

        if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
            return create_compatible_tensor_vector_basis(req, Continuity::H_div);
        }

        const int order = *req.order;

        // Allow explicit family selection via BasisType.
        if (req.basis_type == BasisType::RaviartThomas) {
            return std::make_shared<RaviartThomasBasis>(req.element_type, order);
        }
        if (req.basis_type == BasisType::BDM) {
            return std::make_shared<BDMBasis>(req.element_type, order);
        }

        // Default selection (BasisType::Lagrange): keep the historical choice of BDM(1) on 2D elements,
        // but fall back to Raviart-Thomas when BDM is not applicable (e.g., 3D tensor-product RT(1)+).
        const int dim = element_dimension(req.element_type);
        if (dim == 2 && order == 1) {
            return std::make_shared<BDMBasis>(req.element_type, 1);
        }
        return std::make_shared<RaviartThomasBasis>(req.element_type, order);
    }

    if (req.continuity == Continuity::H_curl) {
        validate_vector_factory_request(req, Continuity::H_curl);
        if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
            return create_compatible_tensor_vector_basis(req, Continuity::H_curl);
        }
        return std::make_shared<NedelecBasis>(req.element_type, *req.order);
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
            return std::make_shared<HermiteBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: C1/Hermite creation requires an explicit order",
                                    "BasisFactory: C1/Hermite requires non-negative order"));
        }
        throw BasisConfigurationException("BasisFactory: C1 continuity currently supports scalar fields only",
                                          __FILE__, __LINE__, __func__);
    }

    switch (req.basis_type) {
        case BasisType::Lagrange:
            return std::make_shared<LagrangeBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Lagrange creation requires an explicit order",
                                    "BasisFactory: Lagrange requires non-negative order"));
        case BasisType::Hierarchical:
            return std::make_shared<HierarchicalBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Hierarchical creation requires an explicit order",
                                    "BasisFactory: Hierarchical requires non-negative order"));
        case BasisType::Bernstein:
            return std::make_shared<BernsteinBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Bernstein creation requires an explicit order",
                                    "BasisFactory: Bernstein requires non-negative order"));
        case BasisType::BSpline:
            return create_bspline_basis(req);
        case BasisType::NURBS:
            return create_nurbs_basis(req);
        case BasisType::Spectral:
            return std::make_shared<SpectralBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Spectral creation requires an explicit order",
                                    "BasisFactory: Spectral requires non-negative order"));
        case BasisType::Serendipity:
            return std::make_shared<SerendipityBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Serendipity creation requires an explicit order",
                                    "BasisFactory: Serendipity requires non-negative order"));
        case BasisType::Hermite:
            return std::make_shared<HermiteBasis>(
                req.element_type,
                require_basis_order(req,
                                    "BasisFactory: Hermite creation requires an explicit order",
                                    "BasisFactory: Hermite requires non-negative order"));
        case BasisType::Bubble:
            reject_explicit_basis_order(req,
                                        "BasisFactory: Bubble requests must omit order; order is intrinsic to the topology");
            return std::make_shared<BubbleBasis>(req.element_type);
        case BasisType::Custom: {
            if (req.custom_id.empty()) {
                throw BasisConfigurationException("BasisFactory: BasisType::Custom requires a non-empty custom_id",
                                                  __FILE__, __LINE__, __func__);
            }

            CustomFactory factory;
            {
                std::lock_guard<std::mutex> lock(custom_registry_mutex());
                const auto it = custom_registry().find(req.custom_id);
                if (it == custom_registry().end()) {
                    throw BasisConfigurationException("BasisFactory: unknown custom basis id \"" + req.custom_id + "\"",
                                                      __FILE__, __LINE__, __func__);
                }
                factory = it->second;
            }

            auto basis = factory(req);
            if (!basis) {
                throw BasisConstructionException("BasisFactory: custom basis factory returned null",
                                                 __FILE__, __LINE__, __func__);
            }
            return basis;
        }
        default:
            throw BasisConfigurationException("Unsupported basis type in BasisFactory",
                                              __FILE__, __LINE__, __func__);
    }
}

void BasisFactory::register_custom(std::string custom_id,
                                   CustomFactory factory) {
    if (custom_id.empty()) {
        throw BasisConfigurationException("BasisFactory: custom_id must not be empty",
                                          __FILE__, __LINE__, __func__);
    }
    if (!factory) {
        throw BasisConfigurationException("BasisFactory: custom basis factory must be valid",
                                          __FILE__, __LINE__, __func__);
    }

    std::lock_guard<std::mutex> lock(custom_registry_mutex());
    const auto [it, inserted] = custom_registry().emplace(std::move(custom_id), std::move(factory));
    if (!inserted) {
        throw BasisConfigurationException("BasisFactory: duplicate custom basis id \"" + it->first + "\"",
                                          __FILE__, __LINE__, __func__);
    }
}

void BasisFactory::unregister_custom(const std::string& custom_id) {
    if (custom_id.empty()) {
        throw BasisConfigurationException("BasisFactory: custom_id must not be empty",
                                          __FILE__, __LINE__, __func__);
    }

    std::lock_guard<std::mutex> lock(custom_registry_mutex());
    if (custom_registry().erase(custom_id) == 0u) {
        throw BasisConfigurationException("BasisFactory: unknown custom basis id \"" + custom_id + "\"",
                                          __FILE__, __LINE__, __func__);
    }
}

void BasisFactory::clear_custom_registry_for_tests() {
    std::lock_guard<std::mutex> lock(custom_registry_mutex());
    custom_registry().clear();
}

} // namespace basis
} // namespace FE
} // namespace svmp
