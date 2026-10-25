/**
 * @file test_GlobalSpace.cpp
 * @brief The space of a global scalar: its basis, elements and the quadrature of its rows
 */

#include <gtest/gtest.h>

#include "FE/Core/FEException.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/GlobalSpace.h"

#include <array>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::spaces;

TEST(GlobalSpace, DescribesOneScalarWithoutGeometry)
{
    GlobalSpace space;
    EXPECT_EQ(space.space_type(), SpaceType::Global);
    EXPECT_EQ(space.field_type(), FieldType::Scalar);
    EXPECT_EQ(space.continuity(), Continuity::L2);
    EXPECT_EQ(space.value_dimension(), 1);
    EXPECT_EQ(space.polynomial_order(), 0);
    EXPECT_EQ(space.topological_dimension(), 0);
    EXPECT_EQ(space.dofs_per_element(), 1u);
    EXPECT_EQ(space.element_type(), ElementType::Point1);
    EXPECT_EQ(space.element().num_dofs(), 1u);
    EXPECT_EQ(space.explicitQuadratureOrder(), 0);
    EXPECT_EQ(space.quadratureOrder(), 0);
}

TEST(GlobalSpace, RefusesSeveralScalarsAndNegativeOrders)
{
    EXPECT_THROW(GlobalSpace(2), FEException);
    EXPECT_THROW(GlobalSpace(1, -1), FEException);
}

TEST(GlobalSpace, CellElementsHaveTheConstantBasisAndTheBoundRule)
{
    GlobalSpace space;
    EXPECT_FALSE(space.hasBoundQuadrature(ElementType::Quad4));
    // Unbound cell types get the prototype.
    EXPECT_EQ(&space.getElement(ElementType::Quad4, 0), &space.element());

    const std::array<ElementType, 2> types{ElementType::Quad4, ElementType::Triangle3};
    space.bindQuadrature(types, 4);
    EXPECT_EQ(space.quadratureOrder(), 4);

    for (const auto type : types) {
        ASSERT_TRUE(space.hasBoundQuadrature(type));
        const auto& element = space.getElement(type, 7);
        EXPECT_EQ(element.element_type(), type);
        EXPECT_EQ(element.num_dofs(), 1u);
        EXPECT_EQ(element.polynomial_order(), 0);
        const auto rule = element.quadrature();
        ASSERT_TRUE(rule);
        const auto expected = quadrature::QuadratureFactory::create(type, 4);
        EXPECT_EQ(rule->num_points(), expected->num_points());

        std::vector<Real> values;
        std::vector<basis::Gradient> gradients;
        const math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0.0)};
        element.basis().evaluate_values(xi, values);
        element.basis().evaluate_gradients(xi, gradients);
        ASSERT_EQ(values.size(), 1u);
        EXPECT_DOUBLE_EQ(values[0], 1.0);
        ASSERT_EQ(gradients.size(), 1u);
        for (int d = 0; d < 3; ++d) {
            EXPECT_DOUBLE_EQ(gradients[0][static_cast<std::size_t>(d)], 0.0);
        }
    }
    // The same order again, and further cell types, bind; another automatic order does not.
    EXPECT_NO_THROW(space.bindQuadrature(types, 4));
    const std::array<ElementType, 1> hex{ElementType::Hex8};
    EXPECT_NO_THROW(space.bindQuadrature(hex, 4));
    EXPECT_THROW(space.bindQuadrature(types, 2), FEException);
}

TEST(GlobalSpace, HigherOrderCellsTakeTheConstantOfTheirLinearShape)
{
    GlobalSpace space;
    const std::array<ElementType, 2> types{ElementType::Quad9, ElementType::Tetra10};
    space.bindQuadrature(types, 4);
    const auto& quad = space.getElement(ElementType::Quad9, 0);
    EXPECT_EQ(quad.element_type(), ElementType::Quad4);
    EXPECT_EQ(quad.num_dofs(), 1u);
    EXPECT_EQ(quad.quadrature()->num_points(),
              quadrature::QuadratureFactory::create(ElementType::Quad9, 4)->num_points());
    const auto& tet = space.getElement(ElementType::Tetra10, 0);
    EXPECT_EQ(tet.element_type(), ElementType::Tetra4);
    EXPECT_EQ(tet.num_dofs(), 1u);
}

TEST(GlobalSpace, AnExplicitOrderWinsOverTheAutomaticOne)
{
    GlobalSpace space(1, 3);
    const std::array<ElementType, 1> types{ElementType::Triangle3};
    space.bindQuadrature(types, 8);
    EXPECT_EQ(space.quadratureOrder(), 3);
    EXPECT_EQ(space.getElement(ElementType::Triangle3, 0).quadrature()->num_points(),
              quadrature::QuadratureFactory::create(ElementType::Triangle3, 3)->num_points());
    // Another system's automatic order does not conflict with an explicit one.
    EXPECT_NO_THROW(space.bindQuadrature(types, 2));
}
