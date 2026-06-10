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
using RequestPredicate = bool (*)(const BasisRequest&);
using RequestValidator = void (*)(const BasisRequest&);
using RequestFactory = std::shared_ptr<BasisFunction> (*)(const BasisRequest&);

struct BasisDescriptor {
    BasisType canonical_type;
    std::array<BasisType, 1> aliases;
    DescriptorOrderPolicy order_policy;
    const char* missing_order_message;
    const char* negative_order_message;
    const char* explicit_order_message;
    DescriptorFactory factory;
};

struct BasisCreationDescriptor {
    const char* name;
    RequestPredicate matches;
    RequestValidator validate;
    RequestFactory factory;
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
    struct QuadComponentRule {
        int boundary_axis;
        int low_entity;
        int high_entity;
        bool reverse_low;
        bool reverse_high;
    };

    constexpr std::array<QuadComponentRule, 2> hcurl_rules{{
        {1, 0, 2, false, true},
        {0, 3, 1, true, false},
    }};
    constexpr std::array<QuadComponentRule, 2> hdiv_rules{{
        {0, 3, 1, true, false},
        {1, 0, 2, false, true},
    }};

    std::vector<DofAssociation> associations;
    associations.reserve(static_cast<std::size_t>(first_extents[0] * first_extents[1] +
                                                  second_extents[0] * second_extents[1]));

    int cell_moment = 0;

    auto append_component = [&](const std::array<int, 2>& extents,
                                const QuadComponentRule& rule) {
        const int moment_axis = 1 - rule.boundary_axis;
        for (int j = 0; j < extents[1]; ++j) {
            for (int i = 0; i < extents[0]; ++i) {
                const std::array<int, 2> coord{i, j};
                const int boundary = coord[static_cast<std::size_t>(rule.boundary_axis)];
                const int moment_coord = coord[static_cast<std::size_t>(moment_axis)];
                const int moment_extent = extents[static_cast<std::size_t>(moment_axis)];

                DofAssociation assoc{};
                if (boundary == 0) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = rule.low_entity;
                    assoc.moment_index =
                        rule.reverse_low ? (moment_extent - 1) - moment_coord : moment_coord;
                } else if (boundary == extents[static_cast<std::size_t>(rule.boundary_axis)] - 1) {
                    assoc.entity_type = DofEntity::Edge;
                    assoc.entity_id = rule.high_entity;
                    assoc.moment_index =
                        rule.reverse_high ? (moment_extent - 1) - moment_coord : moment_coord;
                } else {
                    assoc.entity_type = DofEntity::Interior;
                    assoc.entity_id = 0;
                    assoc.moment_index = cell_moment++;
                }
                associations.push_back(assoc);
            }
        }
    };

    if (family == CompatibleTensorVectorBasis::Family::HCurl) {
        append_component(first_extents, hcurl_rules[0]);
        append_component(second_extents, hcurl_rules[1]);
        return associations;
    }

    BASIS_CHECK_CONFIG(family == CompatibleTensorVectorBasis::Family::HDiv,
                 "BasisFactory: unsupported compatible vector family");

    append_component(first_extents, hdiv_rules[0]);
    append_component(second_extents, hdiv_rules[1]);

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

    struct HcurlEdgeRule {
        int first_side;
        int second_side;
        int edge_id;
        bool reverse;
    };

    struct HcurlComponentRule {
        std::array<int, 2> boundary_axes;
        std::array<int, 2> first_axis_faces;
        std::array<int, 2> second_axis_faces;
        std::array<HcurlEdgeRule, 4> edge_rules;
    };

    constexpr std::array<std::array<int, 2>, 3> hdiv_component_faces{{
        {{5, 3}},
        {{2, 4}},
        {{0, 1}},
    }};

    constexpr std::array<HcurlComponentRule, 3> hcurl_component_rules{{
        {{{1, 2}}, {{2, 4}}, {{0, 1}},
         {{{0, 0, 0, false}, {1, 0, 2, true}, {0, 1, 4, false}, {1, 1, 6, true}}}},
        {{{0, 2}}, {{5, 3}}, {{0, 1}},
         {{{1, 0, 1, false}, {0, 0, 3, true}, {1, 1, 5, false}, {0, 1, 7, true}}}},
        {{{0, 1}}, {{5, 3}}, {{2, 4}},
         {{{0, 0, 8, false}, {1, 0, 9, false}, {1, 1, 10, false}, {0, 1, 11, false}}}},
    }};

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
        for (std::size_t component = 0; component < component_extents.size(); ++component) {
            const auto& extents = component_extents[component];
            const auto& faces = hdiv_component_faces[component];
            for (int k = 0; k < extents[2]; ++k) {
                for (int j = 0; j < extents[1]; ++j) {
                    for (int i = 0; i < extents[0]; ++i) {
                        const std::array<int, 3> coord{i, j, k};
                        const int axis_coord = coord[component];
                        if (axis_coord == 0) {
                            push_face(faces[0]);
                        } else if (axis_coord == extents[component] - 1) {
                            push_face(faces[1]);
                        } else {
                            push_interior();
                        }
                    }
                }
            }
        }
        return associations;
    }

    BASIS_CHECK_CONFIG(family == CompatibleTensorVectorBasis::Family::HCurl,
                 "BasisFactory: unsupported compatible vector family");

    auto boundary_side = [](int coord, int extent) {
        if (coord == 0) {
            return 0;
        }
        if (coord == extent - 1) {
            return 1;
        }
        return -1;
    };

    for (std::size_t component = 0; component < component_extents.size(); ++component) {
        const auto& extents = component_extents[component];
        const auto& rule = hcurl_component_rules[component];
        for (int k = 0; k < extents[2]; ++k) {
            for (int j = 0; j < extents[1]; ++j) {
                for (int i = 0; i < extents[0]; ++i) {
                    const std::array<int, 3> coord{i, j, k};
                    const int first_axis = rule.boundary_axes[0];
                    const int second_axis = rule.boundary_axes[1];
                    const int first_side =
                        boundary_side(coord[static_cast<std::size_t>(first_axis)],
                                      extents[static_cast<std::size_t>(first_axis)]);
                    const int second_side =
                        boundary_side(coord[static_cast<std::size_t>(second_axis)],
                                      extents[static_cast<std::size_t>(second_axis)]);

                    if (first_side >= 0 && second_side >= 0) {
                        const auto edge = std::find_if(
                            rule.edge_rules.begin(),
                            rule.edge_rules.end(),
                            [&](const HcurlEdgeRule& candidate) {
                                return candidate.first_side == first_side &&
                                       candidate.second_side == second_side;
                            });
                        BASIS_CHECK_CONFIG(edge != rule.edge_rules.end(),
                                     "BasisFactory: missing compatible HCurl edge association rule");
                        const int moment_coord = coord[component];
                        const int moment_extent = extents[component];
                        push_edge(edge->edge_id,
                                  edge->reverse
                                      ? (moment_extent - 1) - moment_coord
                                      : moment_coord);
                    } else if (first_side >= 0) {
                        push_face(rule.first_axis_faces[static_cast<std::size_t>(first_side)]);
                    } else if (second_side >= 0) {
                        push_face(rule.second_axis_faces[static_cast<std::size_t>(second_side)]);
                    } else {
                        push_interior();
                    }
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

bool matches_hdiv_request(const BasisRequest& req) {
    return req.continuity == Continuity::H_div;
}

bool matches_hcurl_request(const BasisRequest& req) {
    return req.continuity == Continuity::H_curl;
}

bool matches_c1_request(const BasisRequest& req) {
    return req.continuity == Continuity::C1;
}

bool matches_bspline_request(const BasisRequest& req) {
    return req.basis_type == BasisType::BSpline;
}

bool matches_nurbs_request(const BasisRequest& req) {
    return req.basis_type == BasisType::NURBS;
}

bool matches_custom_request(const BasisRequest& req) {
    return req.basis_type == BasisType::Custom;
}

bool matches_scalar_descriptor_request(const BasisRequest& req) {
    return find_scalar_basis_descriptor(req.basis_type) != nullptr;
}

void validate_hdiv_request(const BasisRequest& req) {
    validate_vector_factory_request(req, Continuity::H_div);
}

void validate_hcurl_request(const BasisRequest& req) {
    validate_vector_factory_request(req, Continuity::H_curl);
}

std::shared_ptr<BasisFunction> create_hdiv_basis(const BasisRequest& req) {
    if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
        return create_compatible_tensor_vector_basis(req, Continuity::H_div);
    }

    const int order = *req.order;
    if (req.basis_type == BasisType::RaviartThomas) {
        return std::make_shared<RaviartThomasBasis>(req.element_type, order);
    }
    if (req.basis_type == BasisType::BDM) {
        return std::make_shared<BDMBasis>(req.element_type, order);
    }

    return std::make_shared<RaviartThomasBasis>(req.element_type, order);
}

std::shared_ptr<BasisFunction> create_hcurl_basis(const BasisRequest& req) {
    if (req.basis_type == BasisType::BSpline || req.basis_type == BasisType::NURBS) {
        return create_compatible_tensor_vector_basis(req, Continuity::H_curl);
    }
    return std::make_shared<NedelecBasis>(req.element_type, *req.order);
}

std::shared_ptr<BasisFunction> create_c1_basis(const BasisRequest& req) {
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

std::shared_ptr<BasisFunction> create_custom_basis(const BasisRequest& req) {
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

std::shared_ptr<BasisFunction> create_scalar_descriptor_basis(const BasisRequest& req) {
    const BasisDescriptor* descriptor = find_scalar_basis_descriptor(req.basis_type);
    if (descriptor == nullptr) {
        throw BasisConfigurationException("Unsupported basis type in BasisFactory",
                                          __FILE__, __LINE__, __func__);
    }
    return create_from_descriptor(*descriptor, req);
}

constexpr std::array<BasisCreationDescriptor, 7> basis_creation_descriptors() {
    return {{
        {"H(div)", &matches_hdiv_request, &validate_hdiv_request, &create_hdiv_basis},
        {"H(curl)", &matches_hcurl_request, &validate_hcurl_request, &create_hcurl_basis},
        {"C1", &matches_c1_request, nullptr, &create_c1_basis},
        {"BSpline", &matches_bspline_request, nullptr, &create_bspline_basis},
        {"NURBS", &matches_nurbs_request, nullptr, &create_nurbs_basis},
        {"Custom", &matches_custom_request, nullptr, &create_custom_basis},
        {"Scalar", &matches_scalar_descriptor_request, nullptr, &create_scalar_descriptor_basis},
    }};
}

} // namespace

std::shared_ptr<BasisFunction> basis_factory::create(const BasisRequest& req) {
    // L² (discontinuous) uses the same reference-space shape functions as C0.
    // DOF ownership (element-local vs shared) is managed at the Space/Element level.
    // The descriptor order is semantically meaningful: vector-valued conforming
    // continuities take precedence, C1 has its narrow Hermite contract, and L2
    // intentionally falls through to the scalar basis-family descriptor.
    static constexpr auto descriptors = basis_creation_descriptors();
    for (const BasisCreationDescriptor& descriptor : descriptors) {
        if (!descriptor.matches(req)) {
            continue;
        }
        if (descriptor.validate != nullptr) {
            descriptor.validate(req);
        }
        return descriptor.factory(req);
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
