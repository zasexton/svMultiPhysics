/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceFactory.h"

namespace svmp {
namespace FE {
namespace spaces {

std::shared_ptr<FunctionSpace> SpaceFactory::create(SpaceType type,
                                                    ElementType element_type,
                                                    int order) {
    switch (type) {
        case SpaceType::H1:
            return create_h1(element_type, order);
        case SpaceType::C1:
            return create_c1(element_type, order);
        case SpaceType::L2:
            return create_l2(element_type, order);
        case SpaceType::HCurl:
            return create_hcurl(element_type, order);
        case SpaceType::HDiv:
            return create_hdiv(element_type, order);
        case SpaceType::Product:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: Product spaces require component count");
        case SpaceType::Isogeometric:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: Isogeometric spaces require external basis and quadrature");
        case SpaceType::Mixed:
        case SpaceType::Trace:
        case SpaceType::Composite:
        case SpaceType::Enriched:
        case SpaceType::Adaptive:
            FE_THROW(NotImplementedException,
                     "SpaceFactory::create: use specialized constructors for composite/enriched/adaptive spaces");
        default:
            FE_THROW(InvalidArgumentException,
                     "SpaceFactory::create: unknown SpaceType");
    }
}

std::shared_ptr<ProductSpace> SpaceFactory::create_vector_h1(ElementType element_type,
                                                             int order,
                                                             int components) {
    auto base = create_h1(element_type, order);
    return std::make_shared<ProductSpace>(base, components);
}

} // namespace spaces
} // namespace FE
} // namespace svmp
