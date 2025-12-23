/**
 * @file test_Order0Elements.cpp
 * @brief Behavioral validation for order-0 (constant) Lagrange elements
 */

#include <gtest/gtest.h>

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Elements/DiscontinuousElement.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Geometry/IsoparametricMapping.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

namespace {

Real compute_scalar_mass_entry(const basis::BasisFunction& basis_fn,
                               const quadrature::QuadratureRule& quad,
                               const geometry::GeometryMapping& mapping) {
    std::vector<Real> values;
    values.resize(basis_fn.size());

    Real mass = Real(0);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& xi = quad.point(q);
        const Real w = quad.weight(q);
        basis_fn.evaluate_values(xi, values);
        const Real detJ = std::abs(mapping.jacobian_determinant(xi));
        mass += w * detJ * values[0] * values[0];
    }
    return mass;
}

void check_order0_element(ElementType type) {
    LagrangeElement elem(type, 0);
    EXPECT_EQ(elem.num_dofs(), 1u);
    EXPECT_EQ(elem.basis().size(), 1u);

    // Constant basis: value == 1 and gradients == 0 at representative points.
    std::vector<Real> values;
    std::vector<basis::Gradient> grads;
    const math::Vector<Real, 3> xi0{};

    elem.basis().evaluate_values(xi0, values);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_NEAR(static_cast<double>(values[0]), 1.0, 1e-14);

    elem.basis().evaluate_gradients(xi0, grads);
    ASSERT_EQ(grads.size(), 1u);
    EXPECT_NEAR(static_cast<double>(grads[0][0]), 0.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(grads[0][1]), 0.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(grads[0][2]), 0.0, 1e-12);

    // Use a linear geometry mapping for the reference element shape.
    auto geom_basis = std::make_shared<basis::LagrangeBasis>(type, 1);
    auto geom_nodes = geom_basis->nodes();
    geometry::IsoparametricMapping mapping(geom_basis, geom_nodes);

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);
    const Real m00 = compute_scalar_mass_entry(elem.basis(), *quad, mapping);

    const Real expected = ReferenceElement::create(type).reference_measure();
    EXPECT_NEAR(static_cast<double>(m00), static_cast<double>(expected), 1e-12);
}

} // namespace

TEST(Order0Elements, ConstantBasisAndMassMatchesReferenceMeasure) {
    // Cover the major element families where order-0 Lagrange is supported.
    check_order0_element(ElementType::Line2);
    check_order0_element(ElementType::Triangle3);
    check_order0_element(ElementType::Quad4);
    check_order0_element(ElementType::Tetra4);
    check_order0_element(ElementType::Hex8);
    check_order0_element(ElementType::Wedge6);
}

TEST(Order0Elements, DGElementAlsoBehavesConstant) {
    DiscontinuousElement elem(ElementType::Quad4, 0);
    EXPECT_EQ(elem.continuity(), Continuity::L2);
    EXPECT_EQ(elem.num_dofs(), 1u);
    EXPECT_EQ(elem.basis().size(), 1u);

    std::vector<Real> values;
    elem.basis().evaluate_values(math::Vector<Real, 3>{Real(0.3), Real(-0.2), Real(0)}, values);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_NEAR(static_cast<double>(values[0]), 1.0, 1e-14);
}

