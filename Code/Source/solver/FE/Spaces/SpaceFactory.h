/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACEFACTORY_H
#define SVMP_FE_SPACES_SPACEFACTORY_H

/**
 * @file SpaceFactory.h
 * @brief Factory for creating basic function spaces
 *
 * SpaceFactory provides convenience helpers for constructing common function
 * spaces (H¹, L², H(curl), H(div), vector-valued H¹) without depending on
 * any Mesh types. Higher-level modules pass element types and polynomial
 * orders; mesh topology is handled elsewhere.
 */

#include "Spaces/FunctionSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/IsogeometricSpace.h"
#include "Spaces/C1Space.h"
#include <memory>

namespace svmp {
namespace FE {
namespace spaces {

class SpaceFactory {
public:
    /// Generic factory for core scalar/vector spaces
    static std::shared_ptr<FunctionSpace> create(SpaceType type,
                                                 ElementType element_type,
                                                 int order);

    static std::shared_ptr<H1Space> create_h1(ElementType element_type,
                                              int order) {
        return std::make_shared<H1Space>(element_type, order);
    }

    static std::shared_ptr<C1Space> create_c1(ElementType element_type,
                                              int order = 3) {
        return std::make_shared<C1Space>(element_type, order);
    }

    static std::shared_ptr<L2Space> create_l2(ElementType element_type,
                                              int order) {
        return std::make_shared<L2Space>(element_type, order);
    }

    static std::shared_ptr<HCurlSpace> create_hcurl(ElementType element_type,
                                                    int order) {
        return std::make_shared<HCurlSpace>(element_type, order);
    }

    static std::shared_ptr<HDivSpace> create_hdiv(ElementType element_type,
                                                  int order) {
        return std::make_shared<HDivSpace>(element_type, order);
    }

    /// Convenience: vector-valued H¹ space as product of scalar H¹
    static std::shared_ptr<ProductSpace> create_vector_h1(ElementType element_type,
                                                          int order,
                                                          int components);

    /// Factory for isogeometric spaces from external basis/quadrature
    static std::shared_ptr<IsogeometricSpace> create_isogeometric(
        std::shared_ptr<basis::BasisFunction> basis,
        std::shared_ptr<const quadrature::QuadratureRule> quadrature,
        FieldType field_type = FieldType::Scalar,
        Continuity continuity = Continuity::C0) {
        return std::make_shared<IsogeometricSpace>(
            std::move(basis), std::move(quadrature), field_type, continuity);
    }
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACEFACTORY_H
