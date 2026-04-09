/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ELEMENTFACTORY_H
#define SVMP_FE_ELEMENTS_ELEMENTFACTORY_H

/**
 * @file ElementFactory.h
 * @brief Runtime factory for creating finite element objects
 */

#include "Elements/Element.h"

#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Request describing the desired element configuration
 */
struct ElementRequest {
    ElementType element_type{ElementType::Unknown};
    BasisType   basis_type{BasisType::Lagrange};
    FieldType   field_type{FieldType::Scalar};
    Continuity  continuity{Continuity::C0};
    std::optional<int> order{};
    std::vector<Real> knot_vector{};
    std::vector<Real> weights{};
    std::vector<int> axis_orders{};
    std::vector<std::vector<Real>> axis_knot_vectors{};
    std::vector<std::vector<Real>> axis_weights{};
    std::vector<int> tensor_extents{};
    std::string custom_id{};
};

/**
 * @brief Factory for creating concrete Element instances
 *
 * The factory encapsulates the selection of concrete element classes
 * (Lagrange, DG, vector, spectral, generic basis-backed, etc.) based on a
 * simple request structure. More specialized elements (e.g., mixed,
 * composite) are typically constructed directly by higher-level code.
 */
class ElementFactory {
public:
    static std::shared_ptr<Element> create(const ElementRequest& req);
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ELEMENTFACTORY_H
