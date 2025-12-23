/**
 * @file test_MetricSurfacePushForward.cpp
 * @brief Tests for metric tensor, surface geometry, and push-forward operations
 */

#include <gtest/gtest.h>
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Geometry/MetricTensor.h"
#include "FE/Geometry/SurfaceGeometry.h"
#include "FE/Geometry/CurveGeometry.h"
#include "FE/Geometry/PushForward.h"
#include "FE/Geometry/LinearMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(MetricTensor, IdentityMappingMetrics) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    auto J = map.jacobian({Real(0), Real(0), Real(0)});
    auto G = MetricTensor::covariant(J, map.dimension());
    auto Ginv = MetricTensor::contravariant(J, map.dimension());

    EXPECT_NEAR(G(0,0), 1.0, 1e-12);
    EXPECT_NEAR(G(1,1), 1.0, 1e-12);
    EXPECT_NEAR(Ginv(0,0), 1.0, 1e-12);
    EXPECT_NEAR(Ginv(1,1), 1.0, 1e-12);
}

TEST(MetricTensor, Scaled3DMappingMetrics) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    // Scaling factors: dx=2 => Jx=1, dy=3 => Jy=1.5, dz=4 => Jz=2
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},      // (-1,-1,-1)
        {Real(2), Real(0), Real(0)},      // (1,-1,-1)
        {Real(2), Real(3), Real(0)},      // (1,1,-1)
        {Real(0), Real(3), Real(0)},      // (-1,1,-1)
        {Real(0), Real(0), Real(4)},      // (-1,-1,1)
        {Real(2), Real(0), Real(4)},      // (1,-1,1)
        {Real(2), Real(3), Real(4)},      // (1,1,1)
        {Real(0), Real(3), Real(4)}       // (-1,1,1)
    };
    IsoparametricMapping map(basis, nodes);
    auto J = map.jacobian({Real(0), Real(0), Real(0)});
    auto G = MetricTensor::covariant(J, 3);
    auto Ginv = MetricTensor::contravariant(J, 3);

    EXPECT_NEAR(G(0,0), 1.0, 1e-12);
    EXPECT_NEAR(G(1,1), 2.25, 1e-12);
    EXPECT_NEAR(G(2,2), 4.0, 1e-12);
    EXPECT_NEAR(Ginv(0,0), 1.0, 1e-12);
    EXPECT_NEAR(Ginv(1,1), 4.0/9.0, 1e-12);
    EXPECT_NEAR(Ginv(2,2), 0.25, 1e-12);
}

TEST(SurfaceGeometry, PlanarQuadNormals) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Map to plane z=2
    std::vector<math::Vector<Real, 3>> nodes;
    for (const auto& n : basis->nodes()) {
        nodes.push_back({n[0], n[1], Real(2)});
    }
    IsoparametricMapping map(basis, nodes);

    auto surf = SurfaceGeometry::evaluate(map, math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    EXPECT_NEAR(surf.normal[0], 0.0, 1e-12);
    EXPECT_NEAR(surf.normal[1], 0.0, 1e-12);
    EXPECT_NEAR(surf.normal[2], 1.0, 1e-12);
    EXPECT_NEAR(surf.area_element, 1.0, 1e-12); // |J_u x J_v| at the center for identity map
}

TEST(SurfaceGeometry, TiltedQuadNormalAndArea) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Reference square mapped to the plane z = x + y.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(-1), Real(-1), Real(-2)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(2)},
        {Real(-1), Real(1),  Real(0)}
    };
    IsoparametricMapping map(basis, nodes);

    const auto surf = SurfaceGeometry::evaluate(map, math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    const Real inv_sqrt3 = Real(1) / std::sqrt(Real(3));

    EXPECT_NEAR(surf.area_element, std::sqrt(Real(3)), 1e-12);
    EXPECT_NEAR(surf.normal.norm(), 1.0, 1e-12);
    EXPECT_NEAR(surf.normal[0], -inv_sqrt3, 1e-12);
    EXPECT_NEAR(surf.normal[1], -inv_sqrt3, 1e-12);
    EXPECT_NEAR(surf.normal[2], inv_sqrt3, 1e-12);
}

TEST(SurfaceGeometry, InvalidDimensionThrows) {
    // 1D line mapping should be rejected by SurfaceGeometry
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)}
    };
    LinearMapping line(ElementType::Line2, nodes);
    EXPECT_THROW((void)SurfaceGeometry::evaluate(line, math::Vector<Real,3>{Real(0), Real(0), Real(0)}), FEException);
}

TEST(CurveGeometry, LineTangentAndNormalsFormOrthonormalFrame) {
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(0), Real(2), Real(0)}
    };
    LinearMapping line(ElementType::Line2, nodes);

    const auto curve = CurveGeometry::evaluate(line, math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    EXPECT_NEAR(curve.line_element, 1.0, 1e-12);
    EXPECT_NEAR(curve.unit_tangent.norm(), 1.0, 1e-12);
    EXPECT_NEAR(curve.normal_1.norm(), 1.0, 1e-12);
    EXPECT_NEAR(curve.normal_2.norm(), 1.0, 1e-12);

    EXPECT_NEAR(curve.unit_tangent.dot(curve.normal_1), 0.0, 1e-12);
    EXPECT_NEAR(curve.unit_tangent.dot(curve.normal_2), 0.0, 1e-12);
    EXPECT_NEAR(curve.normal_1.dot(curve.normal_2), 0.0, 1e-12);
}

TEST(PushForward, GradientAndPiolaTransforms) {
    // Simple scaling map x=2*xi, y=3*eta with node ordering matching basis
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)}, // ref node (-1,-1)
        {Real(2), Real(0), Real(0)}, // ref node (1,-1)
        {Real(2), Real(3), Real(0)}, // ref node (1,1)
        {Real(0), Real(3), Real(0)}  // ref node (-1,1)
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> grad_ref{Real(1), Real(0.5), Real(0)};
    auto grad_phys = PushForward::gradient(map, grad_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(grad_phys[0], 1.0, 1e-12);
    EXPECT_NEAR(grad_phys[1], 1.0/3.0, 1e-9);

    math::Vector<Real,3> v_ref{Real(1), Real(2), Real(0)};
    auto v_hdiv = PushForward::hdiv_vector(map, v_ref, math::Vector<Real,3>{});
    const Real det = map.jacobian_determinant(math::Vector<Real,3>{});
    EXPECT_NEAR(v_hdiv[0], (1.0) / det, 1e-12);
    EXPECT_NEAR(v_hdiv[1], (3.0) / det, 1e-12);

    auto v_hcurl = PushForward::hcurl_vector(map, v_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(v_hcurl[0], 1.0, 1e-12);
    EXPECT_NEAR(v_hcurl[1], 1.3333333333, 1e-9);
}

TEST(PushForward, GradientTransformOnTiltedQuadSurface) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Reference square mapped to the plane z = x + y.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(-1), Real(-1), Real(-2)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(2)},
        {Real(-1), Real(1),  Real(0)}
    };
    IsoparametricMapping map(basis, nodes);

    // Surface gradient of u_hat(xi,eta) = xi + eta at the center.
    // For this mapping, J = [ [1,0],[0,1],[1,1] ] (as columns), so
    // grad_phys = J (J^T J)^{-1} [1,1] = [1/3, 1/3, 2/3].
    math::Vector<Real,3> grad_ref{Real(1), Real(1), Real(0)};
    const auto grad_phys = PushForward::gradient(map, grad_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(grad_phys[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(grad_phys[1], 1.0/3.0, 1e-12);
    EXPECT_NEAR(grad_phys[2], 2.0/3.0, 1e-12);
}

TEST(PushForward, GradientAndPiolaTransforms3D) {
    // 3D scaling map x=2*xi, y=3*eta, z=4*zeta
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},      // (-1,-1,-1)
        {Real(2), Real(0), Real(0)},      // (1,-1,-1)
        {Real(2), Real(3), Real(0)},      // (1,1,-1)
        {Real(0), Real(3), Real(0)},      // (-1,1,-1)
        {Real(0), Real(0), Real(4)},      // (-1,-1,1)
        {Real(2), Real(0), Real(4)},      // (1,-1,1)
        {Real(2), Real(3), Real(4)},      // (1,1,1)
        {Real(0), Real(3), Real(4)}       // (-1,1,1)
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> grad_ref{Real(1), Real(2), Real(3)};
    auto grad_phys = PushForward::gradient(map, grad_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(grad_phys[0], 1.0, 1e-12);      // Jx = 1
    EXPECT_NEAR(grad_phys[1], 4.0/3.0, 1e-9);   // Jy = 1.5 -> inverse 2/3 times 2
    EXPECT_NEAR(grad_phys[2], 1.5, 1e-12);      // Jz = 2 -> inverse 0.5 times 3

    math::Vector<Real,3> v_ref{Real(1), Real(2), Real(3)};
    const Real det = map.jacobian_determinant(math::Vector<Real,3>{});
    auto v_hdiv = PushForward::hdiv_vector(map, v_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(v_hdiv[0], (1.0) / det, 1e-12); // J*v_ref = {1,3,6}, det=3
    EXPECT_NEAR(v_hdiv[1], (3.0) / det, 1e-12);
    EXPECT_NEAR(v_hdiv[2], (6.0) / det, 1e-12);

    auto v_hcurl = PushForward::hcurl_vector(map, v_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(v_hcurl[0], 1.0, 1e-12);
    EXPECT_NEAR(v_hcurl[1], 1.3333333333, 1e-9);
    EXPECT_NEAR(v_hcurl[2], 1.5, 1e-12);
}

TEST(PushForward, HDivAndHCurlReduceIn1D) {
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);
    math::Vector<Real,3> v_ref{Real(1), Real(0), Real(0)};
    auto v_hdiv = PushForward::hdiv_vector(map, v_ref, math::Vector<Real,3>{});
    auto v_hcurl = PushForward::hcurl_vector(map, v_ref, math::Vector<Real,3>{});

    EXPECT_NEAR(v_hdiv[0], 1.0, 1e-12); // J=1, det=1
    EXPECT_NEAR(v_hcurl[0], 1.0, 1e-12);
    EXPECT_NEAR(v_hcurl[1], 0.0, 1e-12);
    EXPECT_NEAR(v_hcurl[2], 0.0, 1e-12);
}

TEST(PushForward, WedgeIdentityMappingConsistency) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(0.1), Real(-0.3)};
    math::Vector<Real,3> grad_ref{Real(1), Real(2), Real(3)};
    auto grad_phys = PushForward::gradient(map, grad_ref, xi);
    EXPECT_NEAR(grad_phys[0], grad_ref[0], 1e-12);
    EXPECT_NEAR(grad_phys[1], grad_ref[1], 1e-12);
    EXPECT_NEAR(grad_phys[2], grad_ref[2], 1e-12);

    math::Vector<Real,3> v_ref{Real(1), Real(2), Real(3)};
    const Real det = map.jacobian_determinant(xi);
    auto v_hdiv = PushForward::hdiv_vector(map, v_ref, xi);
    EXPECT_NEAR(v_hdiv[0], v_ref[0] / det, 1e-12);
    EXPECT_NEAR(v_hdiv[1], v_ref[1] / det, 1e-12);
    EXPECT_NEAR(v_hdiv[2], v_ref[2] / det, 1e-12);

    auto v_hcurl = PushForward::hcurl_vector(map, v_ref, xi);
    EXPECT_NEAR(v_hcurl[0], v_ref[0], 1e-12);
    EXPECT_NEAR(v_hcurl[1], v_ref[1], 1e-12);
    EXPECT_NEAR(v_hcurl[2], v_ref[2], 1e-12);
}

TEST(PushForward, PyramidIdentityMappingConsistency) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.2), Real(0.4)};
    math::Vector<Real,3> grad_ref{Real(1), Real(2), Real(3)};
    auto grad_phys = PushForward::gradient(map, grad_ref, xi);
    EXPECT_NEAR(grad_phys[0], grad_ref[0], 1e-9);
    EXPECT_NEAR(grad_phys[1], grad_ref[1], 1e-9);
    EXPECT_NEAR(grad_phys[2], grad_ref[2], 1e-9);

    math::Vector<Real,3> v_ref{Real(1), Real(2), Real(3)};
    const Real det = map.jacobian_determinant(xi);
    auto v_hdiv = PushForward::hdiv_vector(map, v_ref, xi);
    EXPECT_NEAR(v_hdiv[0], v_ref[0] / det, 1e-9);
    EXPECT_NEAR(v_hdiv[1], v_ref[1] / det, 1e-9);
    EXPECT_NEAR(v_hdiv[2], v_ref[2] / det, 1e-9);

    auto v_hcurl = PushForward::hcurl_vector(map, v_ref, xi);
    EXPECT_NEAR(v_hcurl[0], v_ref[0], 1e-9);
    EXPECT_NEAR(v_hcurl[1], v_ref[1], 1e-9);
    EXPECT_NEAR(v_hcurl[2], v_ref[2], 1e-9);
}
