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
#include <functional>
#include <initializer_list>
#include <string>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

class ManufacturedCurvedVolumeMapping final : public GeometryMapping {
public:
    ElementType element_type() const noexcept override { return ElementType::Hex27; }
    int dimension() const noexcept override { return 3; }
    std::size_t num_nodes() const noexcept override { return nodes_.size(); }
    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept override { return nodes_; }
    bool isAffine() const noexcept override { return false; }

    math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const override {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        return math::Vector<Real, 3>{
            x + Real(0.20) * x * y + Real(0.05) * z * z,
            y + Real(0.15) * y * z + Real(0.04) * x * x,
            z + Real(0.10) * x * z + Real(0.03) * y * y};
    }

    math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>&,
                                           const math::Vector<Real, 3>& = math::Vector<Real, 3>{}) const override {
        throw FEException("ManufacturedCurvedVolumeMapping: inverse map not needed by this test");
    }

    math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const override {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        math::Matrix<Real, 3, 3> J{};
        J(0, 0) = Real(1) + Real(0.20) * y;
        J(0, 1) = Real(0.20) * x;
        J(0, 2) = Real(0.10) * z;
        J(1, 0) = Real(0.08) * x;
        J(1, 1) = Real(1) + Real(0.15) * z;
        J(1, 2) = Real(0.15) * y;
        J(2, 0) = Real(0.10) * z;
        J(2, 1) = Real(0.06) * y;
        J(2, 2) = Real(1) + Real(0.10) * x;
        return J;
    }

    MappingHessian mapping_hessian(const math::Vector<Real, 3>&) const override {
        MappingHessian H{};
        H[0](0, 1) = Real(0.20);
        H[0](1, 0) = Real(0.20);
        H[0](2, 2) = Real(0.10);
        H[1](0, 0) = Real(0.08);
        H[1](1, 2) = Real(0.15);
        H[1](2, 1) = Real(0.15);
        H[2](0, 2) = Real(0.10);
        H[2](2, 0) = Real(0.10);
        H[2](1, 1) = Real(0.06);
        return H;
    }

private:
    std::vector<math::Vector<Real, 3>> nodes_{};
};

math::Vector<Real, 3> manufacturedReferenceVector(const math::Vector<Real, 3>& xi)
{
    return math::Vector<Real, 3>{
        Real(1.0) + Real(0.30) * xi[0] - Real(0.20) * xi[1] + Real(0.10) * xi[2],
        Real(-0.4) - Real(0.05) * xi[0] + Real(0.50) * xi[1] + Real(0.10) * xi[2],
        Real(0.7) - Real(0.30) * xi[0] + Real(0.04) * xi[1] + Real(0.20) * xi[2]};
}

math::Matrix<Real, 3, 3> manufacturedReferenceJacobian()
{
    math::Matrix<Real, 3, 3> jac{};
    jac(0, 0) = Real(0.30);
    jac(0, 1) = Real(-0.20);
    jac(0, 2) = Real(0.10);
    jac(1, 0) = Real(-0.05);
    jac(1, 1) = Real(0.50);
    jac(1, 2) = Real(0.10);
    jac(2, 0) = Real(-0.30);
    jac(2, 1) = Real(0.04);
    jac(2, 2) = Real(0.20);
    return jac;
}

void expectPhysicalGradientMatchesReferenceDirectionalDerivative(
    const ManufacturedCurvedVolumeMapping& map,
    const math::Vector<Real, 3>& xi,
    const math::Matrix<Real, 3, 3>& grad_x,
    const std::function<math::Vector<Real, 3>(const math::Vector<Real, 3>&)>& transformed_value)
{
    const auto J = map.jacobian(xi);
    constexpr Real eps = Real(1e-6);
    for (std::size_t b = 0; b < 3; ++b) {
        math::Vector<Real, 3> xip = xi;
        math::Vector<Real, 3> xim = xi;
        xip[b] += eps;
        xim[b] -= eps;
        const auto vp = transformed_value(xip);
        const auto vm = transformed_value(xim);
        for (std::size_t r = 0; r < 3; ++r) {
            const Real fd_dv_dxi = (vp[r] - vm[r]) / (Real(2) * eps);
            Real analytic_dv_dxi = Real(0);
            for (std::size_t c = 0; c < 3; ++c) {
                analytic_dv_dxi += grad_x(r, c) * J(c, b);
            }
            EXPECT_NEAR(analytic_dv_dxi, fd_dv_dxi, 2e-7);
        }
    }
}

template <typename Callable>
void expectFEExceptionContains(Callable&& callable,
                               std::initializer_list<const char*> substrings)
{
    try {
        callable();
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        const std::string msg = e.what();
        for (const char* substring : substrings) {
            EXPECT_NE(msg.find(substring), std::string::npos)
                << "missing substring '" << substring << "' in: " << msg;
        }
    } catch (...) {
        FAIL() << "Expected FEException";
    }
}

} // namespace

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

TEST(PushForward, AffineVectorJacobianPiolaTransforms3D) {
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)},
        {Real(0), Real(3), Real(0)},
        {Real(0), Real(0), Real(4)}
    };
    LinearMapping map(ElementType::Tetra4, nodes);
    ASSERT_TRUE(map.isAffine());

    math::Matrix<Real, 3, 3> jac_ref{};
    jac_ref(0, 0) = Real(2);
    jac_ref(0, 1) = Real(-1);
    jac_ref(1, 0) = Real(3);
    jac_ref(1, 1) = Real(4);
    jac_ref(2, 2) = Real(-2);

    const auto xi = math::Vector<Real,3>{Real(0.25), Real(0.25), Real(0.25)};
    const auto J = map.jacobian(xi);
    const auto Jinv = map.jacobian_inverse(xi);
    const Real det = map.jacobian_determinant(xi);

    const auto ordinary = PushForward::vector_jacobian(map, jac_ref, xi);
    const auto hdiv = PushForward::hdiv_vector_jacobian(map, jac_ref, xi);
    const auto hcurl = PushForward::hcurl_vector_jacobian(map, jac_ref, xi);

    const math::Matrix<Real, 3, 3> expected_ordinary = jac_ref * Jinv;
    const math::Matrix<Real, 3, 3> expected_hdiv = (J * jac_ref * Jinv) * (Real(1) / det);
    const math::Matrix<Real, 3, 3> expected_hcurl = Jinv.transpose() * jac_ref * Jinv;

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(ordinary(r, c), expected_ordinary(r, c), 1e-12);
            EXPECT_NEAR(hdiv(r, c), expected_hdiv(r, c), 1e-12);
            EXPECT_NEAR(hcurl(r, c), expected_hcurl(r, c), 1e-12);
        }
    }
}

TEST(PushForward, CurvedHDivVectorJacobianMatchesManufacturedReferenceDerivative3D) {
    ManufacturedCurvedVolumeMapping map;
    const auto xi = math::Vector<Real, 3>{Real(0.17), Real(-0.22), Real(0.13)};
    const auto v_ref = manufacturedReferenceVector(xi);
    const auto jac_ref = manufacturedReferenceJacobian();

    const auto grad_x = PushForward::hdiv_vector_jacobian(map, v_ref, jac_ref, xi);

    expectPhysicalGradientMatchesReferenceDirectionalDerivative(
        map,
        xi,
        grad_x,
        [&](const math::Vector<Real, 3>& xiq) {
            return PushForward::hdiv_vector(map, manufacturedReferenceVector(xiq), xiq);
        });
}

TEST(PushForward, CurvedHCurlVectorJacobianMatchesManufacturedReferenceDerivative3D) {
    ManufacturedCurvedVolumeMapping map;
    const auto xi = math::Vector<Real, 3>{Real(-0.11), Real(0.19), Real(0.24)};
    const auto v_ref = manufacturedReferenceVector(xi);
    const auto jac_ref = manufacturedReferenceJacobian();

    const auto grad_x = PushForward::hcurl_vector_jacobian(map, v_ref, jac_ref, xi);

    expectPhysicalGradientMatchesReferenceDirectionalDerivative(
        map,
        xi,
        grad_x,
        [&](const math::Vector<Real, 3>& xiq) {
            return PushForward::hcurl_vector(map, manufacturedReferenceVector(xiq), xiq);
        });
}

TEST(PushForward, CurvedPiolaVectorGradientGeometryDataIsReusable) {
    ManufacturedCurvedVolumeMapping map;
    const auto xi = math::Vector<Real, 3>{Real(0.09), Real(0.07), Real(-0.16)};
    const auto data = PushForward::piola_vector_gradient_geometry_data(map, xi);
    ASSERT_FALSE(data.affine);
    EXPECT_GT(std::abs(data.jacobian_derivatives_x[0](0, 1)), Real(1e-3));
    EXPECT_NEAR(data.inverse_transpose_jacobian_derivatives_x[1](2, 0),
                data.inverse_jacobian_derivatives_x[1](0, 2),
                1e-15);

    const auto jac_ref = manufacturedReferenceJacobian();
    const auto v0 = manufacturedReferenceVector(xi);
    const math::Vector<Real, 3> v1{Real(-0.2), Real(0.8), Real(0.35)};

    const auto hdiv_direct = PushForward::hdiv_vector_jacobian(map, v0, jac_ref, xi);
    const auto hdiv_reused = PushForward::hdiv_vector_jacobian(v0, jac_ref, data);
    const auto hcurl_direct = PushForward::hcurl_vector_jacobian(map, v1, jac_ref, xi);
    const auto hcurl_reused = PushForward::hcurl_vector_jacobian(v1, jac_ref, data);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(hdiv_direct(r, c), hdiv_reused(r, c), 1e-13);
            EXPECT_NEAR(hcurl_direct(r, c), hcurl_reused(r, c), 1e-13);
        }
    }
}

TEST(PushForward, CurvedPiolaVectorJacobianOverloadsReduceToAffinePath) {
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)},
        {Real(0), Real(3), Real(0)},
        {Real(0), Real(0), Real(4)}
    };
    LinearMapping map(ElementType::Tetra4, nodes);
    const auto xi = math::Vector<Real,3>{Real(0.2), Real(0.3), Real(0.1)};
    const math::Vector<Real, 3> v_ref{Real(0.7), Real(-0.1), Real(0.4)};
    const auto jac_ref = manufacturedReferenceJacobian();

    const auto hdiv_affine = PushForward::hdiv_vector_jacobian(map, jac_ref, xi);
    const auto hdiv_curved_api = PushForward::hdiv_vector_jacobian(map, v_ref, jac_ref, xi);
    const auto hcurl_affine = PushForward::hcurl_vector_jacobian(map, jac_ref, xi);
    const auto hcurl_curved_api = PushForward::hcurl_vector_jacobian(map, v_ref, jac_ref, xi);

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(hdiv_affine(r, c), hdiv_curved_api(r, c), 1e-14);
            EXPECT_NEAR(hcurl_affine(r, c), hcurl_curved_api(r, c), 1e-14);
        }
    }
}

TEST(PushForward, TensorProductAffineMappingAllowsPiolaVectorJacobians) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)},
        {Real(2), Real(3), Real(0)},
        {Real(0), Real(3), Real(0)}
    };
    IsoparametricMapping map(basis, nodes);
    ASSERT_TRUE(map.isAffine());

    math::Matrix<Real, 3, 3> jac_ref{};
    jac_ref(0, 0) = Real(1);
    jac_ref(0, 1) = Real(2);
    jac_ref(1, 0) = Real(-3);
    jac_ref(1, 1) = Real(4);

    const auto xi = math::Vector<Real,3>{Real(0.1), Real(-0.2), Real(0)};
    const auto hdiv = PushForward::hdiv_vector_jacobian(map, jac_ref, xi);
    const auto hcurl = PushForward::hcurl_vector_jacobian(map, jac_ref, xi);

    const auto J = map.jacobian(xi);
    const auto Jinv = map.jacobian_inverse(xi);
    const Real det = map.jacobian_determinant(xi);
    const math::Matrix<Real, 3, 3> expected_hdiv = (J * jac_ref * Jinv) * (Real(1) / det);
    const math::Matrix<Real, 3, 3> expected_hcurl = Jinv.transpose() * jac_ref * Jinv;

    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            EXPECT_NEAR(hdiv(r, c), expected_hdiv(r, c), 1e-12);
            EXPECT_NEAR(hcurl(r, c), expected_hcurl(r, c), 1e-12);
        }
    }
}

TEST(PushForward, NonAffinePiolaVectorJacobiansThrow) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)},
        {Real(2.4), Real(3), Real(0)},
        {Real(0), Real(3), Real(0)}
    };
    IsoparametricMapping map(basis, nodes);
    ASSERT_FALSE(map.isAffine());

    math::Matrix<Real, 3, 3> jac_ref{};
    jac_ref(0, 0) = Real(1);
    jac_ref(1, 1) = Real(1);
    const auto xi = math::Vector<Real,3>{Real(0), Real(0), Real(0)};

    expectFEExceptionContains(
        [&]() { (void)PushForward::hdiv_vector_jacobian(map, jac_ref, xi); },
        {"H(div)", "affine geometry mapping", "non-affine curved Piola gradients", "reference vector value"});
    expectFEExceptionContains(
        [&]() { (void)PushForward::hcurl_vector_jacobian(map, jac_ref, xi); },
        {"H(curl)", "affine geometry mapping", "non-affine curved Piola gradients", "reference vector value"});

    const math::Vector<Real, 3> v_ref{Real(0.2), Real(-0.1), Real(0.0)};
    expectFEExceptionContains(
        [&]() { (void)PushForward::hdiv_vector_jacobian(map, v_ref, jac_ref, xi); },
        {"curved Piola vector-gradient derivatives", "non-affine 3D volume mappings",
         "lower-dimensional curved mappings"});
    expectFEExceptionContains(
        [&]() { (void)PushForward::hcurl_vector_jacobian(map, v_ref, jac_ref, xi); },
        {"curved Piola vector-gradient derivatives", "non-affine 3D volume mappings",
         "lower-dimensional curved mappings"});
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
