/**
 * @file test_GenericBasisSpace.cpp
 * @brief Unit tests for GenericBasisSpace wrapper
 *
 * These tests exercise the GenericBasisSpace FunctionSpace façade that
 * wraps an elements::GeneralBasisElement using an externally supplied
 * BasisFunction and QuadratureRule. The actual B-spline/NURBS basis is
 * supplied externally; we use a simple B-spline basis implementation.
 */

#include <gtest/gtest.h>

#include "FE/Spaces/GenericBasisSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/NURBSTensorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/HexahedronQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

class GenericBasisSpaceTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

TEST_F(GenericBasisSpaceTest, MetadataMatchesUnderlyingElement) {
    // Quadratic B-spline basis with an interior knot (4 basis functions)
    auto basis = std::make_shared<basis::BSplineBasis>(
        2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)});
    auto quad  = std::make_shared<quadrature::GaussQuadrature1D>(4);

    GenericBasisSpace space(basis, quad, FieldType::Scalar, Continuity::C0);

    EXPECT_EQ(space.space_type(), SpaceType::GenericBasis);
    EXPECT_EQ(space.field_type(), FieldType::Scalar);
    EXPECT_EQ(space.continuity(), Continuity::C0);
    EXPECT_EQ(space.element_type(), ElementType::Line2);
    EXPECT_EQ(space.polynomial_order(), 2);

    const auto& elem = space.element();
    EXPECT_EQ(elem.element_type(), ElementType::Line2);
    EXPECT_EQ(elem.num_dofs(), basis->size());
}

TEST_F(GenericBasisSpaceTest, InterpolateAndEvaluateConstant) {
    auto basis = std::make_shared<basis::BSplineBasis>(
        2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)});
    auto quad  = std::make_shared<quadrature::GaussQuadrature1D>(4);

    GenericBasisSpace space(basis, quad, FieldType::Scalar, Continuity::C0);

    // Interpolate constant function f(x̂) = 3.0 in reference space
    std::vector<Real> coeffs;
    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& xi) {
        (void)xi;
        FunctionSpace::Value out{};
        out[0] = Real(3.0);
        return out;
    };
    space.interpolate(f, coeffs);

    ASSERT_EQ(coeffs.size(), space.dofs_per_element());

    auto q = space.element().quadrature();
    ASSERT_TRUE(q);
    for (std::size_t i = 0; i < q->num_points(); ++i) {
        auto xi = q->point(i);
        Real val = space.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(val, 3.0, tol);
    }
}

TEST_F(GenericBasisSpaceTest, RationalQuadAndHexMetadataArePreserved) {
    auto quad_basis = std::make_shared<basis::NURBSTensorBasis>(
        basis::BSplineBasis(2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)}),
        basis::BSplineBasis(1, std::vector<Real>{Real(0), Real(0), Real(0.35), Real(1), Real(1)}),
        std::vector<Real>{
            Real(1), Real(1), Real(0.8), Real(1),
            Real(1), Real(1.2), Real(1), Real(1),
            Real(1), Real(1), Real(1), Real(1)
        },
        std::vector<int>{4, 3});
    auto quad_rule = std::make_shared<quadrature::QuadrilateralQuadrature>(4);

    GenericBasisSpace quad_space(quad_basis, quad_rule, FieldType::Scalar, Continuity::C0);
    EXPECT_EQ(quad_space.element_type(), ElementType::Quad4);
    EXPECT_EQ(quad_space.polynomial_order(), 2);
    EXPECT_EQ(quad_space.dofs_per_element(), quad_basis->size());

    auto hex_basis = std::make_shared<basis::NURBSTensorBasis>(
        basis::BSplineBasis(2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)}),
        basis::BSplineBasis(1, std::vector<Real>{Real(0), Real(0), Real(0.6), Real(1), Real(1)}),
        basis::BSplineBasis(1, std::vector<Real>{Real(0), Real(0), Real(0.4), Real(1), Real(1)}),
        std::vector<Real>(36u, Real(1)),
        std::vector<int>{4, 3, 3});
    auto hex_rule = std::make_shared<quadrature::HexahedronQuadrature>(4);

    GenericBasisSpace hex_space(hex_basis, hex_rule, FieldType::Scalar, Continuity::C0);
    EXPECT_EQ(hex_space.element_type(), ElementType::Hex8);
    EXPECT_EQ(hex_space.polynomial_order(), 2);
    EXPECT_EQ(hex_space.dofs_per_element(), hex_basis->size());
}

TEST_F(GenericBasisSpaceTest, SpaceFactoryRequestCreatesBsplineSpace) {
    SpaceRequest req;
    req.space_type = SpaceType::GenericBasis;
    req.element.element_type = ElementType::Quad4;
    req.element.basis_type = BasisType::BSpline;
    req.element.field_type = FieldType::Scalar;
    req.element.continuity = Continuity::C0;
    req.element.order = 2;
    req.element.axis_orders = {2, 1};
    req.element.axis_knot_vectors = {
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
        {Real(0), Real(0), Real(0.3), Real(0.7), Real(1), Real(1)}
    };

    auto space = SpaceFactory::create(req);
    auto generic = std::dynamic_pointer_cast<GenericBasisSpace>(space);
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic->space_type(), SpaceType::GenericBasis);
    EXPECT_EQ(generic->element_type(), ElementType::Quad4);
    EXPECT_EQ(generic->polynomial_order(), 2);
    ASSERT_TRUE(generic->element().basis_ptr());
    EXPECT_EQ(generic->element().basis().basis_type(), BasisType::BSpline);
}

TEST_F(GenericBasisSpaceTest, SpaceFactoryRequestWrapsSpectralElements) {
    SpaceRequest req;
    req.space_type = SpaceType::GenericBasis;
    req.element.element_type = ElementType::Line2;
    req.element.basis_type = BasisType::Spectral;
    req.element.field_type = FieldType::Scalar;
    req.element.continuity = Continuity::C0;
    req.element.order = 3;

    auto space = SpaceFactory::create(req);
    auto generic = std::dynamic_pointer_cast<GenericBasisSpace>(space);
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic->element_type(), ElementType::Line2);
    EXPECT_EQ(generic->polynomial_order(), 3);
    ASSERT_TRUE(generic->element().basis_ptr());
    EXPECT_EQ(generic->element().basis().basis_type(), BasisType::Spectral);
}

TEST_F(GenericBasisSpaceTest, SpaceFactoryRequestCreatesBubbleSpaceWithoutOrder) {
    SpaceRequest req;
    req.space_type = SpaceType::GenericBasis;
    req.element.element_type = ElementType::Triangle3;
    req.element.basis_type = BasisType::Bubble;
    req.element.field_type = FieldType::Scalar;
    req.element.continuity = Continuity::L2;

    auto space = SpaceFactory::create(req);
    auto generic = std::dynamic_pointer_cast<GenericBasisSpace>(space);
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic->element_type(), ElementType::Triangle3);
    EXPECT_EQ(generic->polynomial_order(), 3);
    ASSERT_TRUE(generic->element().basis_ptr());
    EXPECT_EQ(generic->element().basis().basis_type(), BasisType::Bubble);
}

TEST_F(GenericBasisSpaceTest, LegacyFactoryOverloadRejectsUnderSpecifiedGenericBasisRequest) {
    EXPECT_THROW(SpaceFactory::create(SpaceType::GenericBasis, ElementType::Line2, 2),
                 FEException);
}
