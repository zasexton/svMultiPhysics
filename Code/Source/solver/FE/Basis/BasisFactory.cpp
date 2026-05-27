/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisFactory.h"

#include "BernsteinBasis.h"
#include "BSplineBasis.h"
#include "BubbleBasis.h"
#include "CompatibleTensorVectorBasis.h"
#include "HermiteBasis.h"
#include "HierarchicalBasis.h"
#include "LagrangeBasis.h"
#include "NURBSTensorBasis.h"
#include "SerendipityBasis.h"
#include "SpectralBasis.h"
#include "TensorBasis.h"
#include "VectorBasis.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <unordered_map>
#include <utility>

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
                                 const char* message);

enum class DescriptorOrderPolicy {
    RequiredNonNegative,
    MustOmit
};

using DescriptorFactory =
    std::shared_ptr<BasisFunction> (*)(const BasisRequest&, int);

struct BasisDescriptor {
    BasisType canonical_type;
    std::array<BasisType, 1> aliases;
    DescriptorOrderPolicy order_policy;
    const char* missing_order_message;
    const char* negative_order_message;
    const char* explicit_order_message;
    DescriptorFactory factory;
};

template <typename BasisT>
std::shared_ptr<BasisFunction> create_ordered_descriptor_basis(const BasisRequest& req,
                                                               int order) {
    return std::make_shared<BasisT>(req.element_type, order);
}

std::shared_ptr<BasisFunction> create_bubble_descriptor_basis(const BasisRequest& req,
                                                              int /*order*/) {
    return std::make_shared<BubbleBasis>(req.element_type);
}

constexpr std::array<BasisDescriptor, 7> scalar_basis_descriptors() {
    return {{
        {BasisType::Lagrange,
         {BasisType::Lagrange},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Lagrange creation requires an explicit order",
         "BasisFactory: Lagrange requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<LagrangeBasis>},
        {BasisType::Hierarchical,
         {BasisType::Hierarchical},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Hierarchical creation requires an explicit order",
         "BasisFactory: Hierarchical requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<HierarchicalBasis>},
        {BasisType::Bernstein,
         {BasisType::Bernstein},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Bernstein creation requires an explicit order",
         "BasisFactory: Bernstein requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<BernsteinBasis>},
        {BasisType::Spectral,
         {BasisType::Spectral},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Spectral creation requires an explicit order",
         "BasisFactory: Spectral requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<SpectralBasis>},
        {BasisType::Serendipity,
         {BasisType::Serendipity},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Serendipity creation requires an explicit order",
         "BasisFactory: Serendipity requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<SerendipityBasis>},
        {BasisType::Hermite,
         {BasisType::Hermite},
         DescriptorOrderPolicy::RequiredNonNegative,
         "BasisFactory: Hermite creation requires an explicit order",
         "BasisFactory: Hermite requires non-negative order",
         nullptr,
         &create_ordered_descriptor_basis<HermiteBasis>},
        {BasisType::Bubble,
         {BasisType::Bubble},
         DescriptorOrderPolicy::MustOmit,
         nullptr,
         nullptr,
         "BasisFactory: Bubble requests must omit order; order is intrinsic to the topology",
         &create_bubble_descriptor_basis},
    }};
}

const BasisDescriptor* find_scalar_basis_descriptor(BasisType basis_type) {
    static constexpr auto descriptors = scalar_basis_descriptors();
    for (const BasisDescriptor& descriptor : descriptors) {
        if (descriptor.canonical_type == basis_type ||
            std::find(descriptor.aliases.begin(), descriptor.aliases.end(), basis_type) !=
                descriptor.aliases.end()) {
            return &descriptor;
        }
    }
    return nullptr;
}

std::shared_ptr<BasisFunction> create_from_descriptor(const BasisDescriptor& descriptor,
                                                      const BasisRequest& req) {
    int order = 0;
    if (descriptor.order_policy == DescriptorOrderPolicy::RequiredNonNegative) {
        order = require_basis_order(req,
                                    descriptor.missing_order_message,
                                    descriptor.negative_order_message);
    } else {
        reject_explicit_basis_order(req, descriptor.explicit_order_message);
    }
    return descriptor.factory(req, order);
}

void reject_explicit_basis_order(const BasisRequest& req,
                                 const char* message) {
    if (req.order.has_value()) {
        throw BasisConfigurationException(message,
                                          __FILE__, __LINE__, __func__);
    }
}

using CustomRegistryMap = std::unordered_map<std::string, basis_factory::CustomFactory>;

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
        throw BasisElementCompatibilityException("BasisFactory: scalar BSpline is tensor-product only and currently supports Line2, Quad4, and Hex8",
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
        throw BasisElementCompatibilityException("BasisFactory: scalar NURBS is tensor-product only and currently supports Line2, Quad4, and Hex8",
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
    const bool use_axis_data = !req.axis_orders.empty() || !req.axis_knot_vectors.empty();
    if (use_axis_data) {
        BASIS_CHECK_CONFIG(req.axis_orders.empty() || req.axis_orders.size() == static_cast<std::size_t>(dim),
                     "BasisFactory: vector spline/NURBS axis_orders size must match tensor dimension");
        BASIS_CHECK_CONFIG(req.axis_knot_vectors.empty() || req.axis_knot_vectors.size() == static_cast<std::size_t>(dim),
                     "BasisFactory: vector spline/NURBS axis_knot_vectors size must match tensor dimension");
    } else {
        BASIS_CHECK_CONFIG(!req.knot_vector.empty(),
                     "BasisFactory: vector spline/NURBS construction requires knot vectors");
    }

    std::vector<BSplineBasis> axes;
    axes.reserve(static_cast<std::size_t>(dim));
    for (int axis = 0; axis < dim; ++axis) {
        axes.push_back(make_bspline_axis(req, static_cast<std::size_t>(axis)));
    }
    return axes;
}

std::size_t tensor_product_size(const std::vector<int>& extents) {
    std::size_t size = 1u;
    for (int extent : extents) {
        BASIS_CHECK_CONFIG(extent > 0, "BasisFactory: tensor extents must be positive");
        size *= static_cast<std::size_t>(extent);
    }
    return size;
}

BSplineBasis make_reduced_bspline_axis(const BSplineBasis& axis) {
    BASIS_CHECK_CONFIG(axis.order() >= 1,
                 "BasisFactory: compatible spline/NURBS vector bases require axis order >= 1");

    auto knots = axis.knots();
    BASIS_CHECK_CONFIG(knots.size() >= 2u,
                 "BasisFactory: spline knot vector too short for compatible vector basis");
    knots.erase(knots.begin());
    knots.pop_back();
    return BSplineBasis(axis.order() - 1, std::move(knots));
}

std::vector<int> infer_tensor_extents(const std::vector<BSplineBasis>& axes) {
    BASIS_CHECK_CONFIG(axes.size() == 2u || axes.size() == 3u,
                 "BasisFactory: compatible tensor extents require two or three axes");
    std::vector<int> extents;
    extents.reserve(axes.size());
    for (const auto& axis : axes) {
        extents.push_back(static_cast<int>(axis.size()));
    }
    return extents;
}

std::vector<Real> reduce_tensor_weights(const std::vector<Real>& weights,
                                      const std::vector<int>& extents,
                                      int axis) {
    BASIS_CHECK_CONFIG(axis >= 0 && axis < static_cast<int>(extents.size()),
                 "BasisFactory: invalid reduced tensor axis");
    BASIS_CHECK_CONFIG(weights.size() == tensor_product_size(extents),
                 "BasisFactory: NURBS weights size does not match tensor extents");
    BASIS_CHECK_CONFIG(extents[static_cast<std::size_t>(axis)] > 1,
                 "BasisFactory: compatible spline/NURBS vector bases require at least two control points per axis");

    std::vector<int> reduced_extents = extents;
    --reduced_extents[static_cast<std::size_t>(axis)];
    const std::size_t reduced_size = tensor_product_size(reduced_extents);
    std::vector<Real> reduced(reduced_size, Real(0));

    const std::size_t dim = extents.size();
    std::array<std::size_t, 3> input_strides{1u, 1u, 1u};
    for (std::size_t d = 1; d < dim; ++d) {
        input_strides[d] = input_strides[d - 1u] *
                           static_cast<std::size_t>(extents[d - 1u]);
    }

    for (std::size_t linear = 0; linear < reduced_size; ++linear) {
        std::array<int, 3> index{0, 0, 0};
        std::size_t remaining = linear;
        for (std::size_t d = 0; d < dim; ++d) {
            const auto extent = static_cast<std::size_t>(reduced_extents[d]);
            index[d] = static_cast<int>(remaining % extent);
            remaining /= extent;
        }

        std::size_t input0 = 0u;
        for (std::size_t d = 0; d < dim; ++d) {
            input0 += static_cast<std::size_t>(index[d]) * input_strides[d];
        }
        const std::size_t input1 = input0 + input_strides[static_cast<std::size_t>(axis)];
        reduced[linear] = Real(0.5) * (weights[input0] + weights[input1]);
    }

    return reduced;
}

std::shared_ptr<BasisFunction> make_compatible_component_basis(
    BasisType semantic_basis_type,
    const std::vector<BSplineBasis>& base_axes,
    const std::vector<int>& base_extents,
    const std::vector<Real>& weights,
    const std::vector<bool>& reduce_axes) {
    BASIS_CHECK_CONFIG(base_axes.size() == base_extents.size(),
                 "BasisFactory: compatible component basis extent mismatch");
    BASIS_CHECK_CONFIG(base_axes.size() == reduce_axes.size(),
                 "BasisFactory: compatible component basis axis mask mismatch");

    std::vector<BSplineBasis> axes;
    axes.reserve(base_axes.size());
    for (std::size_t axis = 0; axis < base_axes.size(); ++axis) {
        axes.push_back(reduce_axes[axis] ? make_reduced_bspline_axis(base_axes[axis]) : base_axes[axis]);
    }

    if (semantic_basis_type == BasisType::BSpline) {
        if (axes.size() == 2u) {
            return std::make_shared<TensorProductBasis<BSplineBasis>>(axes[0], axes[1]);
        }
        return std::make_shared<TensorProductBasis<BSplineBasis>>(axes[0], axes[1], axes[2]);
    }

    BASIS_CHECK_CONFIG(semantic_basis_type == BasisType::NURBS,
                 "BasisFactory: compatible component basis requires BSpline or NURBS semantics");

    std::vector<Real> component_weights = weights;
    std::vector<int> extents = base_extents;
    for (std::size_t axis = 0; axis < reduce_axes.size(); ++axis) {
        if (!reduce_axes[axis]) {
            continue;
        }
        component_weights = reduce_tensor_weights(component_weights, extents, static_cast<int>(axis));
        --extents[axis];
    }

    if (axes.size() == 2u) {
        return std::make_shared<NURBSTensorBasis>(
            axes[0],
            axes[1],
            std::move(component_weights),
            extents);
    }

    return std::make_shared<NURBSTensorBasis>(
        axes[0],
        axes[1],
        axes[2],
        std::move(component_weights),
        extents);
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

    BASIS_CHECK_CONFIG(family == CompatibleTensorVectorBasis::Family::HDiv,
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

std::vector<DofAssociation> build_hex_compatible_vector_associations(
    CompatibleTensorVectorBasis::Family family,
    const std::array<std::array<int, 3>, 3>& component_extents) {
    auto component_size = [](const std::array<int, 3>& extents) {
        return static_cast<std::size_t>(extents[0]) *
               static_cast<std::size_t>(extents[1]) *
               static_cast<std::size_t>(extents[2]);
    };

    std::vector<DofAssociation> associations;
    associations.reserve(component_size(component_extents[0]) +
                         component_size(component_extents[1]) +
                         component_size(component_extents[2]));

    std::array<int, 12> edge_moment{};
    std::array<int, 6> face_moment{};
    int cell_moment = 0;

    auto push_face = [&](int face_id) {
        DofAssociation assoc{};
        assoc.entity_type = DofEntity::Face;
        assoc.entity_id = face_id;
        assoc.moment_index = face_moment[static_cast<std::size_t>(face_id)]++;
        associations.push_back(assoc);
    };
    auto push_edge = [&](int edge_id, int moment_index) {
        DofAssociation assoc{};
        assoc.entity_type = DofEntity::Edge;
        assoc.entity_id = edge_id;
        assoc.moment_index = moment_index;
        edge_moment[static_cast<std::size_t>(edge_id)] =
            std::max(edge_moment[static_cast<std::size_t>(edge_id)], moment_index + 1);
        associations.push_back(assoc);
    };
    auto push_interior = [&]() {
        DofAssociation assoc{};
        assoc.entity_type = DofEntity::Interior;
        assoc.entity_id = 0;
        assoc.moment_index = cell_moment++;
        associations.push_back(assoc);
    };

    if (family == CompatibleTensorVectorBasis::Family::HDiv) {
        const auto& ex = component_extents[0];
        for (int k = 0; k < ex[2]; ++k) {
            for (int j = 0; j < ex[1]; ++j) {
                for (int i = 0; i < ex[0]; ++i) {
                    if (i == 0) {
                        push_face(5);
                    } else if (i == ex[0] - 1) {
                        push_face(3);
                    } else {
                        push_interior();
                    }
                }
            }
        }

        const auto& ey = component_extents[1];
        for (int k = 0; k < ey[2]; ++k) {
            for (int j = 0; j < ey[1]; ++j) {
                for (int i = 0; i < ey[0]; ++i) {
                    if (j == 0) {
                        push_face(2);
                    } else if (j == ey[1] - 1) {
                        push_face(4);
                    } else {
                        push_interior();
                    }
                }
            }
        }

        const auto& ez = component_extents[2];
        for (int k = 0; k < ez[2]; ++k) {
            for (int j = 0; j < ez[1]; ++j) {
                for (int i = 0; i < ez[0]; ++i) {
                    if (k == 0) {
                        push_face(0);
                    } else if (k == ez[2] - 1) {
                        push_face(1);
                    } else {
                        push_interior();
                    }
                }
            }
        }

        return associations;
    }

    BASIS_CHECK_CONFIG(family == CompatibleTensorVectorBasis::Family::HCurl,
                 "BasisFactory: unsupported compatible vector family");

    const auto& ex = component_extents[0];
    for (int k = 0; k < ex[2]; ++k) {
        for (int j = 0; j < ex[1]; ++j) {
            for (int i = 0; i < ex[0]; ++i) {
                const bool on_y = (j == 0 || j == ex[1] - 1);
                const bool on_z = (k == 0 || k == ex[2] - 1);
                if (on_y && on_z) {
                    int edge_id = 0;
                    int moment_index = i;
                    if (j == 0 && k == 0) {
                        edge_id = 0;
                    } else if (j == ex[1] - 1 && k == 0) {
                        edge_id = 2;
                        moment_index = (ex[0] - 1) - i;
                    } else if (j == 0 && k == ex[2] - 1) {
                        edge_id = 4;
                    } else {
                        edge_id = 6;
                        moment_index = (ex[0] - 1) - i;
                    }
                    push_edge(edge_id, moment_index);
                } else if (on_y) {
                    push_face(j == 0 ? 2 : 4);
                } else if (on_z) {
                    push_face(k == 0 ? 0 : 1);
                } else {
                    push_interior();
                }
            }
        }
    }

    const auto& ey = component_extents[1];
    for (int k = 0; k < ey[2]; ++k) {
        for (int j = 0; j < ey[1]; ++j) {
            for (int i = 0; i < ey[0]; ++i) {
                const bool on_x = (i == 0 || i == ey[0] - 1);
                const bool on_z = (k == 0 || k == ey[2] - 1);
                if (on_x && on_z) {
                    int edge_id = 0;
                    int moment_index = j;
                    if (i == ey[0] - 1 && k == 0) {
                        edge_id = 1;
                    } else if (i == 0 && k == 0) {
                        edge_id = 3;
                        moment_index = (ey[1] - 1) - j;
                    } else if (i == ey[0] - 1 && k == ey[2] - 1) {
                        edge_id = 5;
                    } else {
                        edge_id = 7;
                        moment_index = (ey[1] - 1) - j;
                    }
                    push_edge(edge_id, moment_index);
                } else if (on_x) {
                    push_face(i == ey[0] - 1 ? 3 : 5);
                } else if (on_z) {
                    push_face(k == 0 ? 0 : 1);
                } else {
                    push_interior();
                }
            }
        }
    }

    const auto& ez = component_extents[2];
    for (int k = 0; k < ez[2]; ++k) {
        for (int j = 0; j < ez[1]; ++j) {
            for (int i = 0; i < ez[0]; ++i) {
                const bool on_x = (i == 0 || i == ez[0] - 1);
                const bool on_y = (j == 0 || j == ez[1] - 1);
                if (on_x && on_y) {
                    int edge_id = 0;
                    if (i == 0 && j == 0) {
                        edge_id = 8;
                    } else if (i == ez[0] - 1 && j == 0) {
                        edge_id = 9;
                    } else if (i == ez[0] - 1 && j == ez[1] - 1) {
                        edge_id = 10;
                    } else {
                        edge_id = 11;
                    }
                    push_edge(edge_id, k);
                } else if (on_x) {
                    push_face(i == ez[0] - 1 ? 3 : 5);
                } else if (on_y) {
                    push_face(j == 0 ? 2 : 4);
                } else {
                    push_interior();
                }
            }
        }
    }

    return associations;
}

std::shared_ptr<BasisFunction> create_compatible_tensor_vector_basis(
    const BasisRequest& req,
    Continuity continuity) {
    const int dim = spline_tensor_dimension(req.element_type);
    BASIS_CHECK_CONFIG(req.element_type == ElementType::Quad4 || req.element_type == ElementType::Hex8,
                 "BasisFactory: compatible spline/NURBS H(div)/H(curl) bases are intentionally limited to Quad4 and Hex8");

    const auto axes = make_tensor_axes(req, dim);
    const auto base_extents = infer_tensor_extents(axes);
    BASIS_CHECK_CONFIG(std::all_of(base_extents.begin(),
                             base_extents.end(),
                             [](int extent) { return extent >= 2; }),
                 "BasisFactory: compatible spline/NURBS vector bases require at least two basis functions per axis");

    if (req.basis_type == BasisType::NURBS && req.weights.empty()) {
        throw BasisConfigurationException("BasisFactory: vector NURBS construction requires weights",
                                          __FILE__, __LINE__, __func__);
    }

    const auto family = (continuity == Continuity::H_curl)
        ? CompatibleTensorVectorBasis::Family::HCurl
        : CompatibleTensorVectorBasis::Family::HDiv;

    const int order = std::max_element(
        axes.begin(),
        axes.end(),
        [](const BSplineBasis& a, const BSplineBasis& b) {
            return a.order() < b.order();
        })->order();

    if (dim == 2) {
        const std::vector<bool> first_reduce = {
            family == CompatibleTensorVectorBasis::Family::HCurl,
            family == CompatibleTensorVectorBasis::Family::HDiv
        };
        const std::vector<bool> second_reduce = {
            family == CompatibleTensorVectorBasis::Family::HDiv,
            family == CompatibleTensorVectorBasis::Family::HCurl
        };

        auto first_basis = make_compatible_component_basis(
            req.basis_type, axes, base_extents, req.weights, first_reduce);
        auto second_basis = make_compatible_component_basis(
            req.basis_type, axes, base_extents, req.weights, second_reduce);

        const std::array<int, 2> first_extents = {
            base_extents[0] - (first_reduce[0] ? 1 : 0),
            base_extents[1] - (first_reduce[1] ? 1 : 0)
        };
        const std::array<int, 2> second_extents = {
            base_extents[0] - (second_reduce[0] ? 1 : 0),
            base_extents[1] - (second_reduce[1] ? 1 : 0)
        };

        auto associations = build_quad_compatible_vector_associations(family,
                                                                      first_extents,
                                                                      second_extents);

        return std::make_shared<CompatibleTensorVectorBasis>(family,
                                                             req.basis_type,
                                                             std::move(first_basis),
                                                             std::move(second_basis),
                                                             std::move(associations),
                                                             order,
                                                             ElementType::Quad4);
    }

    std::array<std::vector<bool>, 3> reduce_masks{};
    if (family == CompatibleTensorVectorBasis::Family::HCurl) {
        reduce_masks = {std::vector<bool>{true, false, false},
                        std::vector<bool>{false, true, false},
                        std::vector<bool>{false, false, true}};
    } else {
        reduce_masks = {std::vector<bool>{false, true, true},
                        std::vector<bool>{true, false, true},
                        std::vector<bool>{true, true, false}};
    }

    std::vector<std::shared_ptr<BasisFunction>> component_bases;
    component_bases.reserve(3u);
    std::array<std::array<int, 3>, 3> component_extents{};
    for (std::size_t c = 0; c < 3u; ++c) {
        component_bases.push_back(make_compatible_component_basis(
            req.basis_type, axes, base_extents, req.weights, reduce_masks[c]));
        for (std::size_t axis = 0; axis < 3u; ++axis) {
            component_extents[c][axis] =
                base_extents[axis] - (reduce_masks[c][axis] ? 1 : 0);
        }
    }

    auto associations = build_hex_compatible_vector_associations(family, component_extents);
    return std::make_shared<CompatibleTensorVectorBasis>(family,
                                                         req.basis_type,
                                                         std::move(component_bases),
                                                         std::move(associations),
                                                         order,
                                                         ElementType::Hex8);
}

} // namespace

std::shared_ptr<BasisFunction> basis_factory::create(const BasisRequest& req) {
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

        // Default selection (BasisType::Lagrange): use Raviart-Thomas unless the
        // caller explicitly requests BDM.
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

    // Intentional narrow contract: C¹ scalar bases route only through the
    // cubic Hermite family on Line2/Quad4/Hex8.
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

    if (req.basis_type == BasisType::BSpline) {
        return create_bspline_basis(req);
    }
    if (req.basis_type == BasisType::NURBS) {
        return create_nurbs_basis(req);
    }
    if (req.basis_type == BasisType::Custom) {
        if (req.custom_id.empty()) {
            throw BasisConfigurationException("BasisFactory: BasisType::Custom requires a non-empty custom_id",
                                              __FILE__, __LINE__, __func__);
        }

        basis_factory::CustomFactory factory;
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

    if (const BasisDescriptor* descriptor = find_scalar_basis_descriptor(req.basis_type)) {
        return create_from_descriptor(*descriptor, req);
    }

    throw BasisConfigurationException("Unsupported basis type in BasisFactory",
                                      __FILE__, __LINE__, __func__);
}

void basis_factory::register_custom(std::string custom_id,
                                    basis_factory::CustomFactory factory) {
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

void basis_factory::unregister_custom(const std::string& custom_id) {
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

void basis_factory::clear_custom_registry_for_tests() {
    std::lock_guard<std::mutex> lock(custom_registry_mutex());
    custom_registry().clear();
}

} // namespace basis
} // namespace FE
} // namespace svmp
