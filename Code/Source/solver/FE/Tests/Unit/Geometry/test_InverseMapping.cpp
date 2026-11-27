/**
 * @file test_InverseMapping.cpp
 * @brief Tests for inverse mapping robustness
 */

#include <gtest/gtest.h>
#include "FE/Geometry/InverseMapping.h"
#include "FE/Geometry/LinearMapping.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(InverseMapping, SucceedsForAffineTriangle) {
    // Triangle mapping: scale and shift
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(1), Real(1), Real(0)},
        {Real(3), Real(1), Real(0)},
        {Real(1), Real(4), Real(0)}
    };
    LinearMapping map(ElementType::Triangle3, nodes);
    math::Vector<Real,3> xi{Real(0.2), Real(0.3), Real(0)};
    auto x = map.map_to_physical(xi);
    auto xi_back = InverseMapping::solve(map, x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
}

TEST(InverseMapping, SingularJacobianThrows) {
    // Degenerate triangle (all nodes collinear)
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Triangle3, nodes);
    math::Vector<Real,3> x_phys{Real(0.5), Real(0.0), Real(0.0)};
    EXPECT_THROW((void)InverseMapping::solve(map, x_phys), FEException);
}

TEST(InverseMapping, NonConvergenceThrows) {
    // Valid mapping but force non-convergence via tiny max_iters
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> x_phys{Real(0.3), Real(-0.2), Real(0)};
    EXPECT_THROW((void)InverseMapping::solve(map, x_phys, math::Vector<Real,3>{}, 1, Real(1e-16)), FEException);
}

TEST(InverseMapping, RecoversReferenceOnTiltedQuadSurface) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Reference square mapped to the plane z = x + y.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(-1), Real(-1), Real(-2)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(2)},
        {Real(-1), Real(1),  Real(0)}
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(-0.3), Real(0)};
    const auto x = map.map_to_physical(xi);
    const auto xi_back = InverseMapping::solve(map, x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-10);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-10);
}

TEST(InverseMapping, RequireInsideRejectsOutsidePoint) {
    // Identity triangle on unit right reference triangle (x>=0,y>=0,x+y<=1)
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)}
    };
    LinearMapping map(ElementType::Triangle3, nodes);

    InverseMappingOptions opts;
    opts.require_inside = true;

    math::Vector<Real,3> x_out{Real(1.2), Real(0.1), Real(0)};
    EXPECT_THROW((void)InverseMapping::solve_with_options(map, x_out, math::Vector<Real,3>{}, opts), FEException);
}

TEST(InverseMapping, AnInteriorProjectionDoesNotInvertAnOffSurfacePoint) {
    const std::vector<math::Vector<Real, 3>> nodes{{0.,0.,0.}, {1.,0.,0.}, {0.,1.,0.}};
    LinearMapping map(ElementType::Triangle3, nodes);
    const math::Vector<Real, 3> outside{0.25, 0.25, 1.};
    EXPECT_THROW((void)map.map_to_reference(outside), FEException);
    for (bool line_search : {false, true}) {
        for (bool inside : {false, true}) {
            InverseMappingOptions options;
            options.use_line_search = line_search;
            options.require_inside = inside;
            EXPECT_THROW((void)InverseMapping::solve_with_options(map, outside, {}, options), FEException);
            const auto near = InverseMapping::solve_with_options(map, {0.25, 0.25, 0.5 * options.tol}, {}, options);
            EXPECT_NEAR(near[0], 0.25, options.tol);
            EXPECT_NEAR(near[1], 0.25, options.tol);
        }
    }
}

TEST(InverseMapping, RequireInsideAcceptsBoundaryPoint) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)}
    };
    LinearMapping map(ElementType::Triangle3, nodes);

    InverseMappingOptions opts;
    opts.require_inside = true;

    // Point on the hypotenuse (x+y=1) should be accepted.
    const math::Vector<Real,3> xi{Real(0.7), Real(0.3), Real(0)};
    const auto x = map.map_to_physical(xi);
    const auto xi_back = InverseMapping::solve_with_options(map, x, math::Vector<Real,3>{}, opts);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-10);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-10);
}
