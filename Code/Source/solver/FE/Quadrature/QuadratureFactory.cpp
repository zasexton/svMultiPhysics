/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "QuadratureFactory.h"
#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace quadrature {

namespace {

/**
 * @brief Map higher-order element types to their canonical (linear) base type
 */
ElementType canonical(ElementType type) {
    switch (type) {
        case ElementType::Line3:     return ElementType::Line2;
        case ElementType::Triangle6: return ElementType::Triangle3;
        case ElementType::Quad8:
        case ElementType::Quad9:     return ElementType::Quad4;
        case ElementType::Tetra10:   return ElementType::Tetra4;
        case ElementType::Hex20:
        case ElementType::Hex27:     return ElementType::Hex8;
        case ElementType::Wedge15:
        case ElementType::Wedge18:   return ElementType::Wedge6;
        case ElementType::Pyramid13:
        case ElementType::Pyramid14: return ElementType::Pyramid5;
        default:                     return type;
    }
}

/**
 * @brief Compute number of 1D Gauss points needed for given order
 */
int points_for_order(int order, QuadratureType type) {
    if (order < 1) {
        order = 1;
    }
    switch (type) {
        case QuadratureType::GaussLobatto:
            // Need 2n-3 >= order  ->  n >= ceil((order+3)/2)
            return std::max(2, (order + 4) / 2);
        default:
            return std::max(1, (order + 2) / 2);
    }
}

/**
 * @brief Build an order-based quadrature rule
 */
std::shared_ptr<QuadratureRule> build_rule(ElementType element_type,
                                           int order,
                                           QuadratureType type) {
    if (type == QuadratureType::Newton ||
        type == QuadratureType::Composite ||
        type == QuadratureType::Custom) {
        throw FEException("QuadratureFactory: requested QuadratureType is not supported by create(); "
                          "construct the rule explicitly (e.g., CompositeQuadrature / custom points)",
                          __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
    }

    const bool reduced = type == QuadratureType::Reduced;
    const int effective_order = reduced ? std::max(1, order - 1) : order;

    switch (canonical(element_type)) {
        case ElementType::Line2:
            if (type == QuadratureType::GaussLobatto) {
                return std::make_shared<GaussLobattoQuadrature1D>(points_for_order(effective_order, type));
            }
            // Reduced integration lowers the requested exactness (fewer points).
            return std::make_shared<GaussQuadrature1D>(points_for_order(effective_order, QuadratureType::GaussLegendre));
        case ElementType::Triangle3:
            return std::make_shared<TriangleQuadrature>(effective_order);
        case ElementType::Quad4:
            return std::make_shared<QuadrilateralQuadrature>(order, type);
        case ElementType::Tetra4:
            return std::make_shared<TetrahedronQuadrature>(effective_order);
        case ElementType::Hex8:
            return std::make_shared<HexahedronQuadrature>(order, type);
        case ElementType::Wedge6:
            return std::make_shared<WedgeQuadrature>(order, type);
        case ElementType::Pyramid5:
            if (type == QuadratureType::GaussLobatto) {
                throw FEException("QuadratureFactory: PyramidQuadrature does not support Gauss-Lobatto; "
                                  "use PyramidQuadrature (Gauss-Legendre) or request GaussLegendre/Reduced",
                                  __FILE__, __LINE__, __func__, FEStatus::NotImplemented);
            }
            return std::make_shared<PyramidQuadrature>(effective_order);
        case ElementType::Point1:
            class PointQuadrature : public QuadratureRule {
            public:
                explicit PointQuadrature(int ord)
                    : QuadratureRule(svmp::CellFamily::Point, 0, std::max(0, ord)) {
                    set_data({QuadPoint{Real(0), Real(0), Real(0)}}, {Real(1)});
                }
            };
            return std::make_shared<PointQuadrature>(order);
        default:
            throw FEException("QuadratureFactory: unsupported element type",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

/**
 * @brief Build a position-based quadrature rule
 */
std::shared_ptr<QuadratureRule> build_position_based_rule(
    ElementType element_type,
    const PositionBasedParams& params) {

    switch (canonical(element_type)) {
        case ElementType::Triangle3:
            return std::make_shared<PositionBasedTriangleQuadrature>(params);
        case ElementType::Tetra4:
            return std::make_shared<PositionBasedTetrahedronQuadrature>(params);
        default:
            throw FEException(
                "QuadratureFactory: position-based quadrature only supported for Triangle3 and Tetra4",
                __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

/**
 * @brief Compute effective order for position-based rules (for cache key)
 *
 * This is a heuristic since position-based rules don't have a well-defined
 * polynomial exactness except at specific positions.
 */
int effective_order_for_position_based(ElementType element_type, Real modifier) {
    switch (canonical(element_type)) {
        case ElementType::Triangle3: {
            // Gaussian position for TRI3 is 2/3
            const Real gaussian = 2.0 / 3.0;
            if (std::abs(modifier - gaussian) < 0.01) {
                return 2;
            }
            return 1;
        }
        case ElementType::Tetra4: {
            // Gaussian position for TET4 is (5+3*sqrt(5))/20
            const Real gaussian = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
            if (std::abs(modifier - gaussian) < 0.01) {
                return 2;
            }
            return 1;
        }
        default:
            return 1;
    }
}

} // anonymous namespace

// =============================================================================
// Order-based quadrature (standard approach)
// =============================================================================

std::shared_ptr<const QuadratureRule> QuadratureFactory::create(
    ElementType element_type,
    int order,
    QuadratureType type,
    bool use_cache) {

    if (order < 1) {
        throw FEException("QuadratureFactory: order must be >= 1",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    if (type == QuadratureType::PositionBased) {
        // Provide a convenient entry-point through the standard factory API.
        // This uses the legacy-default modifier (Gaussian placement).
        return create_position_based(element_type,
                                     PositionBasedParams{default_legacy_modifier(element_type)},
                                     use_cache);
    }

    if (!use_cache) {
        return build_rule(element_type, order, type);
    }

    QuadratureKey key{element_type, order, type};
    return QuadratureCache::instance().get_or_create(
        key, [=]() { return build_rule(element_type, order, type); });
}

int QuadratureFactory::recommended_order(int basis_order, bool is_mass_matrix) {
    int order = std::max(1, 2 * basis_order);
    if (is_mass_matrix) {
        order = std::max(order, 2 * basis_order + 1);
    }
    return order;
}

// =============================================================================
// Position-based quadrature (legacy compatible approach)
// =============================================================================

std::shared_ptr<const QuadratureRule> QuadratureFactory::create_position_based(
    ElementType element_type,
    const PositionBasedParams& params,
    bool use_cache) {

    // Validate element type
    if (!supports_position_based(element_type)) {
        throw FEException(
            "QuadratureFactory: position-based quadrature only supported for Triangle3 and Tetra4",
            __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }

    if (!use_cache) {
        return build_position_based_rule(element_type, params);
    }

    // Create cache key with position modifier
    int effective_order = effective_order_for_position_based(element_type, params.modifier);
    QuadratureKey key{element_type, effective_order, QuadratureType::PositionBased, params.modifier};

    return QuadratureCache::instance().get_or_create(
        key, [=]() { return build_position_based_rule(element_type, params); });
}

std::shared_ptr<const QuadratureRule> QuadratureFactory::create_legacy_compatible(
    ElementType element_type,
    Real legacy_modifier,
    bool use_cache) {

    // The legacy qmTET4 and qmTRI3 parameters use the same parameterization
    // as our PositionBasedParams, so we can pass through directly
    PositionBasedParams params{legacy_modifier};
    return create_position_based(element_type, params, use_cache);
}

// =============================================================================
// Convenience methods for common position-based configurations
// =============================================================================

std::shared_ptr<const QuadratureRule> QuadratureFactory::create_central(
    ElementType element_type,
    bool use_cache) {

    switch (canonical(element_type)) {
        case ElementType::Triangle3:
            return create_position_based(element_type,
                                         PositionBasedParams::tri3_central(),
                                         use_cache);
        case ElementType::Tetra4:
            return create_position_based(element_type,
                                         PositionBasedParams::tet4_central(),
                                         use_cache);
        default:
            throw FEException(
                "QuadratureFactory: central quadrature only supported for Triangle3 and Tetra4",
                __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

std::shared_ptr<const QuadratureRule> QuadratureFactory::create_nodal(
    ElementType element_type,
    bool use_cache) {

    switch (canonical(element_type)) {
        case ElementType::Triangle3:
            return create_position_based(element_type,
                                         PositionBasedParams::tri3_nodal(),
                                         use_cache);
        case ElementType::Tetra4:
            return create_position_based(element_type,
                                         PositionBasedParams::tet4_nodal(),
                                         use_cache);
        default:
            throw FEException(
                "QuadratureFactory: nodal quadrature only supported for Triangle3 and Tetra4",
                __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

// =============================================================================
// Query methods
// =============================================================================

bool QuadratureFactory::supports_position_based(ElementType element_type) noexcept {
    switch (canonical(element_type)) {
        case ElementType::Triangle3:
        case ElementType::Tetra4:
            return true;
        default:
            return false;
    }
}

Real QuadratureFactory::default_legacy_modifier(ElementType element_type) {
    switch (canonical(element_type)) {
        case ElementType::Triangle3:
            // Default qmTRI3 value from legacy solver
            return 2.0 / 3.0;
        case ElementType::Tetra4:
            // Default qmTET4 value from legacy solver: (5+3*sqrt(5))/20
            return (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
        default:
            throw FEException(
                "QuadratureFactory: default_legacy_modifier only supported for Triangle3 and Tetra4",
                __FILE__, __LINE__, __func__, FEStatus::InvalidElement);
    }
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
