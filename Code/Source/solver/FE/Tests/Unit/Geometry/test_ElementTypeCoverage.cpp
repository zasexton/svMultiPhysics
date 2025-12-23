/**
 * @file test_ElementTypeCoverage.cpp
 * @brief Geometry coverage tests across supported element variants.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "FE/Geometry/MappingFactory.h"
#include "FE/Geometry/GeometryValidator.h"
#include "FE/Basis/LagrangeBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

math::Vector<Real,3> reference_point(ElementType e) {
    switch (e) {
        case ElementType::Line2:
        case ElementType::Line3:
            return math::Vector<Real,3>{Real(0.3), Real(0), Real(0)};
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return math::Vector<Real,3>{Real(0.2), Real(0.1), Real(0)};
        case ElementType::Quad4:
        case ElementType::Quad9:
            return math::Vector<Real,3>{Real(0.2), Real(-0.3), Real(0)};
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return math::Vector<Real,3>{Real(0.1), Real(0.2), Real(0.1)};
        case ElementType::Hex8:
        case ElementType::Hex27:
            return math::Vector<Real,3>{Real(0.1), Real(-0.2), Real(0.3)};
        case ElementType::Wedge6:
        case ElementType::Wedge18:
            return math::Vector<Real,3>{Real(0.2), Real(0.1), Real(-0.3)};
        case ElementType::Pyramid5:
        case ElementType::Pyramid14:
            return math::Vector<Real,3>{Real(0.1), Real(-0.2), Real(0.4)};
        default:
            return math::Vector<Real,3>{Real(0), Real(0), Real(0)};
    }
}

} // namespace

TEST(GeometryElementTypes, LagrangeVariantsHaveValidMappings) {
    using ET = ElementType;
    const ET types[] = {
        ET::Line2, ET::Line3,
        ET::Triangle3, ET::Triangle6,
        ET::Quad4, ET::Quad9,
        ET::Tetra4, ET::Tetra10,
        ET::Hex8, ET::Hex27,
        ET::Wedge6, ET::Wedge18,
        ET::Pyramid5, ET::Pyramid14
    };

    for (ET e : types) {
        int geometry_order = 1;
        switch (e) {
            case ET::Line3:
            case ET::Triangle6:
            case ET::Quad9:
            case ET::Tetra10:
            case ET::Hex27:
            case ET::Wedge18:
            case ET::Pyramid14:
                geometry_order = 2;
                break;
            default:
                break;
        }

        auto basis = std::make_shared<basis::LagrangeBasis>(e, geometry_order);
        auto nodes = basis->nodes();

        MappingRequest req{e, geometry_order, false, nullptr};
        auto mapping = MappingFactory::create(req, nodes);

        const auto xi = reference_point(e);
        auto q = GeometryValidator::evaluate(*mapping, xi);
        EXPECT_TRUE(q.positive_jacobian) << "ElementType index: " << static_cast<int>(e);
        EXPECT_TRUE(std::isfinite(q.condition_number)) << "ElementType index: " << static_cast<int>(e);
    }
}
