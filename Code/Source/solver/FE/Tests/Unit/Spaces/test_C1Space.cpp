/**
 * @file test_C1Space.cpp
 * @brief Unit tests for C¹ function space
 */

#include <gtest/gtest.h>

#include "FE/Spaces/C1Space.h"
#include "FE/Spaces/SpaceFactory.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

TEST(C1Space, MetadataAndConstructionAcrossSupportedTopologies) {
    struct Case {
        ElementType element_type;
        int dimension;
        std::size_t dofs;
    };

    const Case cases[] = {
        {ElementType::Line2, 1, 4u},
        {ElementType::Quad4, 2, 16u},
        {ElementType::Hex8, 3, 64u},
    };

    for (const auto& c : cases) {
        C1Space space(c.element_type, 3);

        EXPECT_EQ(space.space_type(), SpaceType::C1);
        EXPECT_EQ(space.field_type(), FieldType::Scalar);
        EXPECT_EQ(space.continuity(), Continuity::C1);
        EXPECT_EQ(space.value_dimension(), 1);
        EXPECT_EQ(space.topological_dimension(), c.dimension);
        EXPECT_EQ(space.polynomial_order(), 3);
        EXPECT_EQ(space.element_type(), c.element_type);
        EXPECT_EQ(space.dofs_per_element(), c.dofs);

        const auto& elem = space.element();
        EXPECT_EQ(elem.element_type(), c.element_type);
        EXPECT_EQ(elem.continuity(), Continuity::C1);
        EXPECT_EQ(elem.field_type(), FieldType::Scalar);
        EXPECT_EQ(elem.polynomial_order(), 3);
        ASSERT_TRUE(elem.quadrature());
        EXPECT_GT(elem.quadrature()->num_points(), 0u);
    }
}

TEST(C1Space, InterpolatesCubicPolynomialExactly) {
    C1Space space(ElementType::Line2, 3);

    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& xi) {
        FunctionSpace::Value out{};
        const Real x = xi[0];
        out[0] = x * x * x + Real(2) * x * x - x + Real(1);
        return out;
    };

    std::vector<Real> coeffs;
    space.interpolate(f, coeffs);
    EXPECT_EQ(coeffs.size(), space.dofs_per_element());

    auto quad = space.element().quadrature();
    ASSERT_TRUE(quad);

    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        auto xi = quad->point(q);
        const Real expected = f(xi)[0];
        const Real approx = space.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(approx, expected, 1e-10);
    }
}

TEST(C1Space, SpaceFactoryCreatesC1Space) {
    for (ElementType element_type : {ElementType::Line2, ElementType::Quad4, ElementType::Hex8}) {
        auto space = SpaceFactory::create(SpaceType::C1, element_type, 3);
        ASSERT_TRUE(space);
        EXPECT_EQ(space->space_type(), SpaceType::C1);
        EXPECT_EQ(space->continuity(), Continuity::C1);
        EXPECT_EQ(space->element_type(), element_type);
        EXPECT_EQ(space->polynomial_order(), 3);
    }
}

TEST(C1Space, UnsupportedElementTypesAndOrdersThrow) {
    EXPECT_THROW(C1Space(ElementType::Triangle3, 3), svmp::FE::FEException);
    EXPECT_THROW(C1Space(ElementType::Tetra4, 3), svmp::FE::FEException);
    EXPECT_THROW(C1Space(ElementType::Wedge6, 3), svmp::FE::FEException);
    EXPECT_THROW(C1Space(ElementType::Line2, 2), svmp::FE::FEException);
    EXPECT_THROW(C1Space(ElementType::Quad4, 4), svmp::FE::FEException);
    EXPECT_THROW(C1Space(ElementType::Hex8, 5), svmp::FE::FEException);
}
