/**
 * @file test_IsogeometricSpace.cpp
 * @brief Unit tests for IsogeometricSpace wrapper
 *
 * These tests exercise the IsogeometricSpace FunctionSpace façade that
 * wraps an elements::IsogeometricElement using an externally supplied
 * BasisFunction and QuadratureRule. The actual B-spline/NURBS basis is
 * supplied externally; we use a simple B-spline basis implementation.
 */

#include <gtest/gtest.h>

#include "FE/Spaces/IsogeometricSpace.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

class IsogeometricSpaceTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

TEST_F(IsogeometricSpaceTest, MetadataMatchesUnderlyingElement) {
    // Quadratic B-spline basis with an interior knot (4 basis functions)
    auto basis = std::make_shared<basis::BSplineBasis>(
        2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)});
    auto quad  = std::make_shared<quadrature::GaussQuadrature1D>(4);

    IsogeometricSpace space(basis, quad, FieldType::Scalar, Continuity::C0);

    EXPECT_EQ(space.space_type(), SpaceType::Isogeometric);
    EXPECT_EQ(space.field_type(), FieldType::Scalar);
    EXPECT_EQ(space.continuity(), Continuity::C0);
    EXPECT_EQ(space.element_type(), ElementType::Line2);
    EXPECT_EQ(space.polynomial_order(), 2);

    const auto& elem = space.element();
    EXPECT_EQ(elem.element_type(), ElementType::Line2);
    EXPECT_EQ(elem.num_dofs(), basis->size());
}

TEST_F(IsogeometricSpaceTest, InterpolateAndEvaluateConstant) {
    auto basis = std::make_shared<basis::BSplineBasis>(
        2, std::vector<Real>{Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)});
    auto quad  = std::make_shared<quadrature::GaussQuadrature1D>(4);

    IsogeometricSpace space(basis, quad, FieldType::Scalar, Continuity::C0);

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
