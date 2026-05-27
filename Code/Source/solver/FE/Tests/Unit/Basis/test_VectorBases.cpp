/**
 * @file test_VectorBases.cpp
 * @brief Tests for H(div)/H(curl) vector bases
 */

#include <gtest/gtest.h>
#include "FE/Basis/CompatibleTensorVectorBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Basis/VectorBasisEvaluationHelpers.h"
#include "FE/Core/FEException.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/ReferenceMonomialIntegrals.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;

// Forward declarations for helper integrators used in edge-based DOF tests.
static double integrate_wedge_edge_dof(const NedelecBasis& basis,
                                       int edge_id,
                                       int func_id,
                                       int quad_order = 4);

static double integrate_pyramid_edge_dof(const NedelecBasis& basis,
                                         int edge_id,
                                         int func_id,
                                         int quad_order = 4);

static double integrate_triangle_edge_tangential_moment(const NedelecBasis& basis,
                                                        int edge_id,
                                                        int mode_id,
                                                        int func_id,
                                                        int order,
                                                        int quad_order = 8);

static double integrate_triangle_edge_normal_moment(const RaviartThomasBasis& basis,
                                                    int edge_id,
                                                    int mode_id,
                                                    int func_id,
                                                    int order,
                                                    int quad_order = 8);

static Real evaluate_quad_bdm_edge_normal_flux(const BDMBasis& basis,
                                               int edge_id,
                                               int func_id,
                                               Real s);

static Real evaluate_triangle_bdm_edge_scaled_normal_flux(const BDMBasis& basis,
                                                          int edge_id,
                                                          int func_id,
                                                          Real s);

static double integrate_quad_bdm_edge_flux(const BDMBasis& basis,
                                           int edge_id,
                                           int func_id,
                                           int quad_order = 8);

static double integrate_quad_bdm_edge_moment(const BDMBasis& basis,
                                             int edge_id,
                                             int mode_id,
                                             int func_id,
                                             int quad_order = 8);

static double integrate_triangle_bdm_edge_flux(const BDMBasis& basis,
                                               int edge_id,
                                               int func_id,
                                               int quad_order = 8);

static double integrate_triangle_bdm_edge_moment(const BDMBasis& basis,
                                                 int edge_id,
                                                 int mode_id,
                                                 int func_id,
                                                 int quad_order = 8);

static double integrate_quad_bdm_divergence(const BDMBasis& basis,
                                            int func_id,
                                            int quad_order = 8);

static double integrate_triangle_bdm_divergence(const BDMBasis& basis,
                                                int func_id,
                                                int quad_order = 8);

static double integrate_triangle_bdm_edge_moment_exact(const BDMBasis& basis,
                                                       int edge_id,
                                                       int mode_id,
                                                       int func_id,
                                                       int order,
                                                       int quad_order = 8);

static double integrate_triangle_bdm_interior_moment(const BDMBasis& basis,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order = 8);

static double integrate_tetra_bdm_face_moment(const BDMBasis& basis,
                                              int face_id,
                                              int mode_id,
                                              int func_id,
                                              int order,
                                              int quad_order = 8);

static double integrate_tetra_bdm_face_moment_with_vertex_permutation(
    const BDMBasis& basis,
    int face_id,
    const std::array<int, 3>& vertex_permutation,
    int mode_id,
    int func_id,
    int order,
    int quad_order = 8);

static double integrate_tetra_bdm_interior_moment(const BDMBasis& basis,
                                                  int mode_id,
                                                  int func_id,
                                                  int order,
                                                  int quad_order = 8);

static double integrate_tetra_edge_tangential_moment(const NedelecBasis& basis,
                                                     int edge_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order = 8);

static double integrate_tetra_face_tangential_moment(const NedelecBasis& basis,
                                                     int face_id,
                                                     int dir_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order = 8);

using Vec3 = svmp::FE::math::Vector<Real, 3>;

static Vec3 curl_from_jacobian(const VectorJacobian& J) {
    return Vec3{J(2u, 1u) - J(1u, 2u),
                J(0u, 2u) - J(2u, 0u),
                J(1u, 0u) - J(0u, 1u)};
}

static Real divergence_from_jacobian(const VectorJacobian& J) {
    return J(0u, 0u) + J(1u, 1u) + J(2u, 2u);
}

class FallbackPlanarVectorBasis final : public VectorBasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Quad4; }
    int dimension() const noexcept override { return 2; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }

    void evaluate_vector_values(const Vec3& xi,
                                std::vector<Vec3>& values) const override {
        values = {
            Vec3{xi[0] + Real(2) * xi[1], Real(3) * xi[0] - xi[1], Real(0)},
            Vec3{-Real(2) * xi[0], xi[1] * xi[1], Real(0)}
        };
    }

    void evaluate_vector_jacobians(const Vec3& xi,
                                   std::vector<VectorJacobian>& jacobians) const override {
        jacobians.assign(size(), VectorJacobian{});
        jacobians[0](0u, 0u) = Real(1);
        jacobians[0](0u, 1u) = Real(2);
        jacobians[0](1u, 0u) = Real(3);
        jacobians[0](1u, 1u) = Real(-1);
        jacobians[1](0u, 0u) = Real(-2);
        jacobians[1](1u, 1u) = Real(2) * xi[1];
    }
};

template <typename BasisT>
static void expect_vector_strided_all_outputs_match_pointwise(const BasisT& basis,
                                                              const std::vector<Vec3>& points) {
    const std::size_t stride = points.size() + 3u;
    const Real sentinel = Real(-123);
    std::vector<Real> values(basis.size() * 3u * stride, sentinel);
    std::vector<Real> jacobians(basis.size() * 9u * stride, sentinel);
    std::vector<Real> curls(basis.size() * 3u * stride, sentinel);
    std::vector<Real> divergence(basis.size() * stride, sentinel);

    basis.evaluate_vector_at_quadrature_points_strided(points,
                                                       stride,
                                                       values.data(),
                                                       jacobians.data(),
                                                       curls.data(),
                                                       divergence.data());

    std::vector<Vec3> expected_values;
    std::vector<VectorJacobian> expected_jacobians;
    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_vector_values(points[q], expected_values);
        basis.evaluate_vector_jacobians(points[q], expected_jacobians);
        ASSERT_EQ(expected_values.size(), basis.size());
        ASSERT_EQ(expected_jacobians.size(), basis.size());

        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            const Vec3 expected_curl = curl_from_jacobian(expected_jacobians[dof]);
            const Real expected_divergence = divergence_from_jacobian(expected_jacobians[dof]);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(values[(dof * 3u + component) * stride + q],
                            expected_values[dof][component],
                            1e-12);
                EXPECT_NEAR(curls[(dof * 3u + component) * stride + q],
                            expected_curl[component],
                            1e-12);
                for (std::size_t derivative = 0; derivative < 3u; ++derivative) {
                    EXPECT_NEAR(jacobians[(dof * 9u + component * 3u + derivative) *
                                          stride + q],
                                expected_jacobians[dof](component, derivative),
                                1e-12);
                }
            }
            EXPECT_NEAR(divergence[dof * stride + q], expected_divergence, 1e-12);
        }
    }

    for (std::size_t dof = 0; dof < basis.size(); ++dof) {
        for (std::size_t pad = points.size(); pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(divergence[dof * stride + pad], sentinel);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_DOUBLE_EQ(values[(dof * 3u + component) * stride + pad], sentinel);
                EXPECT_DOUBLE_EQ(curls[(dof * 3u + component) * stride + pad], sentinel);
            }
            for (std::size_t component = 0; component < 9u; ++component) {
                EXPECT_DOUBLE_EQ(jacobians[(dof * 9u + component) * stride + pad], sentinel);
            }
        }
    }
}

template <typename BasisT>
static void expect_vector_strided_values_only_match_pointwise(const BasisT& basis,
                                                              const std::vector<Vec3>& points) {
    const std::size_t stride = points.size() + 1u;
    const Real sentinel = Real(-77);
    std::vector<Real> values(basis.size() * 3u * stride, sentinel);
    basis.evaluate_vector_at_quadrature_points_strided(
        points, stride, values.data(), nullptr, nullptr, nullptr);

    std::vector<Vec3> expected_values;
    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_vector_values(points[q], expected_values);
        ASSERT_EQ(expected_values.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(values[(dof * 3u + component) * stride + q],
                            expected_values[dof][component],
                            1e-12);
                EXPECT_DOUBLE_EQ(values[(dof * 3u + component) * stride + points.size()],
                                 sentinel);
            }
        }
    }
}

template <typename BasisT>
static void expect_vector_strided_divergence_only_match_pointwise(
    const BasisT& basis,
    const std::vector<Vec3>& points) {
    const std::size_t stride = points.size() + 2u;
    const Real sentinel = Real(-55);
    std::vector<Real> divergence(basis.size() * stride, sentinel);
    basis.evaluate_vector_at_quadrature_points_strided(
        points, stride, nullptr, nullptr, nullptr, divergence.data());

    std::vector<Real> expected_divergence;
    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_divergence(points[q], expected_divergence);
        ASSERT_EQ(expected_divergence.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            EXPECT_NEAR(divergence[dof * stride + q], expected_divergence[dof], 1e-12);
            for (std::size_t pad = points.size(); pad < stride; ++pad) {
                EXPECT_DOUBLE_EQ(divergence[dof * stride + pad], sentinel);
            }
        }
    }
}

template <typename BasisT>
static void expect_vector_strided_curl_only_match_pointwise(const BasisT& basis,
                                                           const std::vector<Vec3>& points) {
    const std::size_t stride = points.size() + 2u;
    const Real sentinel = Real(-44);
    std::vector<Real> curls(basis.size() * 3u * stride, sentinel);
    basis.evaluate_vector_at_quadrature_points_strided(
        points, stride, nullptr, nullptr, curls.data(), nullptr);

    std::vector<Vec3> expected_curl;
    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_curl(points[q], expected_curl);
        ASSERT_EQ(expected_curl.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(curls[(dof * 3u + component) * stride + q],
                            expected_curl[dof][component],
                            1e-12);
                for (std::size_t pad = points.size(); pad < stride; ++pad) {
                    EXPECT_DOUBLE_EQ(curls[(dof * 3u + component) * stride + pad],
                                     sentinel);
                }
            }
        }
    }
}

TEST(VectorBasisStrided, RaviartThomasOverrideMatchesPointwise) {
    const RaviartThomasBasis basis(ElementType::Triangle3, 1);
    const std::vector<Vec3> points = {
        {Real(0.15), Real(0.2), Real(0)},
        {Real(0.45), Real(0.1), Real(0)}
    };

    expect_vector_strided_all_outputs_match_pointwise(basis, points);
    expect_vector_strided_values_only_match_pointwise(basis, points);
    expect_vector_strided_divergence_only_match_pointwise(basis, points);
}

TEST(VectorBasisStrided, NedelecOverrideMatchesPointwise) {
    const NedelecBasis basis(ElementType::Triangle3, 1);
    const std::vector<Vec3> points = {
        {Real(0.2), Real(0.25), Real(0)},
        {Real(0.55), Real(0.15), Real(0)}
    };

    expect_vector_strided_all_outputs_match_pointwise(basis, points);
    expect_vector_strided_values_only_match_pointwise(basis, points);
    expect_vector_strided_curl_only_match_pointwise(basis, points);
}

TEST(VectorBasisStrided, BDMOverrideMatchesPointwise) {
    const BDMBasis basis(ElementType::Triangle3, 2);
    const std::vector<Vec3> points = {
        {Real(0.1), Real(0.35), Real(0)},
        {Real(0.5), Real(0.2), Real(0)}
    };

    expect_vector_strided_all_outputs_match_pointwise(basis, points);
    expect_vector_strided_values_only_match_pointwise(basis, points);
    expect_vector_strided_divergence_only_match_pointwise(basis, points);
}

TEST(VectorBasisStrided, TransformedSeedOverridesMatchPointwise) {
    const std::vector<Vec3> wedge_points = {
        {Real(0.2), Real(0.2), Real(-0.3)},
        {Real(0.5), Real(0.1), Real(0.4)}
    };

    const RaviartThomasBasis rt_wedge(ElementType::Wedge6, 1);
    expect_vector_strided_all_outputs_match_pointwise(rt_wedge, wedge_points);
    expect_vector_strided_divergence_only_match_pointwise(rt_wedge, wedge_points);

    const NedelecBasis nd_wedge(ElementType::Wedge6, 1);
    expect_vector_strided_all_outputs_match_pointwise(nd_wedge, wedge_points);
    expect_vector_strided_curl_only_match_pointwise(nd_wedge, wedge_points);
}

TEST(VectorBasisStrided, BaseFallbackOverwritesPlanarComponents) {
    const FallbackPlanarVectorBasis concrete_basis;
    const VectorBasisFunction& basis = concrete_basis;
    const std::vector<Vec3> points = {
        {Real(0.25), Real(-0.5), Real(0)},
        {Real(-0.3), Real(0.4), Real(0)}
    };
    const std::size_t stride = points.size() + 2u;
    const Real sentinel = Real(37);
    std::vector<Real> values(basis.size() * 3u * stride, sentinel);
    std::vector<Real> jacobians(basis.size() * 9u * stride, sentinel);
    std::vector<Real> curls(basis.size() * 3u * stride, sentinel);
    std::vector<Real> divergence(basis.size() * stride, sentinel);

    basis.evaluate_vector_at_quadrature_points_strided(points,
                                                       stride,
                                                       values.data(),
                                                       jacobians.data(),
                                                       curls.data(),
                                                       divergence.data());

    std::vector<Vec3> expected_values;
    std::vector<VectorJacobian> expected_jacobians;
    for (std::size_t q = 0; q < points.size(); ++q) {
        basis.evaluate_vector_values(points[q], expected_values);
        basis.evaluate_vector_jacobians(points[q], expected_jacobians);
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            const Vec3 expected_curl = curl_from_jacobian(expected_jacobians[dof]);
            const Real expected_divergence = divergence_from_jacobian(expected_jacobians[dof]);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(values[(dof * 3u + component) * stride + q],
                            expected_values[dof][component],
                            1e-14);
                EXPECT_NEAR(curls[(dof * 3u + component) * stride + q],
                            expected_curl[component],
                            1e-14);
            }
            for (std::size_t component = 0; component < 9u; ++component) {
                const std::size_t row = component / 3u;
                const std::size_t col = component % 3u;
                EXPECT_NEAR(jacobians[(dof * 9u + component) * stride + q],
                            expected_jacobians[dof](row, col),
                            1e-14);
            }
            EXPECT_NEAR(divergence[dof * stride + q], expected_divergence, 1e-14);
            EXPECT_DOUBLE_EQ(values[(dof * 3u + 2u) * stride + q], Real(0));
            EXPECT_DOUBLE_EQ(curls[(dof * 3u + 0u) * stride + q], Real(0));
            EXPECT_DOUBLE_EQ(curls[(dof * 3u + 1u) * stride + q], Real(0));
        }
    }

    for (std::size_t dof = 0; dof < basis.size(); ++dof) {
        for (std::size_t pad = points.size(); pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(divergence[dof * stride + pad], sentinel);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_DOUBLE_EQ(values[(dof * 3u + component) * stride + pad], sentinel);
                EXPECT_DOUBLE_EQ(curls[(dof * 3u + component) * stride + pad], sentinel);
            }
            for (std::size_t component = 0; component < 9u; ++component) {
                EXPECT_DOUBLE_EQ(jacobians[(dof * 9u + component) * stride + pad], sentinel);
            }
        }
    }
}

TEST(ReferenceMonomialIntegrals, KnownLowOrderValuesAndProductDomains) {
    using namespace svmp::FE::quadrature::reference_integrals;

    EXPECT_NEAR(integral_monomial_1d(0), Real(2), 1e-15);
    EXPECT_NEAR(integral_monomial_1d(1), Real(0), 1e-15);
    EXPECT_NEAR(integral_monomial_1d(2), Real(2) / Real(3), 1e-15);

    EXPECT_NEAR(integral_triangle_monomial(0, 0), Real(0.5), 1e-15);
    EXPECT_NEAR(integral_triangle_monomial(1, 0), Real(1) / Real(6), 1e-15);
    EXPECT_NEAR(integral_triangle_monomial(1, 1), Real(1) / Real(24), 1e-15);

    EXPECT_NEAR(integral_tetra_monomial(0, 0, 0), Real(1) / Real(6), 1e-15);
    EXPECT_NEAR(integral_tetra_monomial(1, 0, 0), Real(1) / Real(24), 1e-15);
    EXPECT_NEAR(integral_tetra_monomial(1, 1, 0), Real(1) / Real(120), 1e-15);

    EXPECT_NEAR(integral_pyramid_z(2), Real(1) / Real(3), 1e-15);
    EXPECT_NEAR(integral_wedge_monomial(1, 1, 2),
                integral_triangle_monomial(1, 1) * integral_monomial_1d(2),
                1e-15);

    EXPECT_THROW((void)integral_triangle_monomial(-1, 0), FEException);
    EXPECT_THROW((void)integral_tetra_monomial(0, -1, 0), FEException);
    EXPECT_THROW((void)integral_wedge_monomial(0, 0, -1), FEException);
}

TEST(VectorBasisModalPolynomial, AppendUniqueSkipsExactDuplicates) {
    std::vector<VectorBasisModalPolynomial> polynomials;

    VectorBasisModalPolynomial x{};
    x.num_terms = 1;
    x.terms[0] = VectorBasisModalTerm{0, 1, 0, 0, Real(1)};

    VectorBasisModalPolynomial y = x;
    y.terms[0].component = 1;

    EXPECT_TRUE(append_unique_modal_polynomial(polynomials, x));
    EXPECT_FALSE(append_unique_modal_polynomial(polynomials, x));
    EXPECT_TRUE(append_unique_modal_polynomial(polynomials, y));
    EXPECT_EQ(polynomials.size(), 2u);
}

TEST(VectorBasisSparseCoefficients, PrunesRelativeRoundoffNoise) {
    using svmp::FE::basis::detail::vector_common::build_sparse_modal_coefficients;

    constexpr Real max_abs = Real(2);
    const Real below_threshold =
        std::numeric_limits<Real>::epsilon() * Real(16) * max_abs;
    const Real above_threshold =
        std::numeric_limits<Real>::epsilon() * Real(1024) * max_abs;

    const std::vector<Real> dense = {
        Real(1), below_threshold, -above_threshold, Real(0),
        -below_threshold, -max_abs, Real(0), Real(0.5)
    };

    const auto sparse = build_sparse_modal_coefficients(dense, 2u, 4u);

    EXPECT_EQ(sparse.rows, 2u);
    EXPECT_EQ(sparse.cols, 4u);
    EXPECT_EQ(sparse.row_offsets, (std::vector<std::size_t>{0u, 2u, 4u}));
    EXPECT_EQ(sparse.dofs, (std::vector<std::size_t>{0u, 2u, 1u, 3u}));

    ASSERT_EQ(sparse.coefficients.size(), 4u);
    EXPECT_EQ(sparse.coefficients[0], Real(1));
    EXPECT_EQ(sparse.coefficients[1], -above_threshold);
    EXPECT_EQ(sparse.coefficients[2], -max_abs);
    EXPECT_EQ(sparse.coefficients[3], Real(0.5));
}

TEST(RaviartThomasBasis, DivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    ASSERT_EQ(basis.size(), 4u);

    const svmp::FE::math::Vector<Real, 3> xi0{Real(0.0), Real(0.0), Real(0.0)};
    const svmp::FE::math::Vector<Real, 3> xi1{Real(0.37), Real(-0.22), Real(0.0)};

    std::vector<Real> div0;
    std::vector<Real> div1;
    basis.evaluate_divergence(xi0, div0);
    basis.evaluate_divergence(xi1, div1);

    ASSERT_EQ(div0.size(), basis.size());
    ASSERT_EQ(div1.size(), basis.size());

    for (std::size_t i = 0; i < basis.size(); ++i) {
        EXPECT_NEAR(div0[i], div1[i], 1e-12);
        EXPECT_NEAR(div0[i], 0.25, 1e-12);
    }
}

TEST(NedelecBasis, CurlSigns) {
    NedelecBasis basis(ElementType::Quad4, 0);
    svmp::FE::math::Vector<Real, 3> xi{0.1, -0.2, 0.0};
    EXPECT_TRUE(basis.is_vector_valued());

    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    basis.evaluate_curl(xi, curl);
    ASSERT_EQ(curl.size(), 4u);
    for (const auto& c : curl) {
        EXPECT_NEAR(c[0], 0.0, 1e-14);
        EXPECT_NEAR(c[1], 0.0, 1e-14);
        EXPECT_NEAR(c[2], 0.25, 1e-12);
    }
}

TEST(VectorBasisJacobians, ScalarBasisThrowsClearException) {
    LagrangeBasis basis(ElementType::Triangle3, 1);
    std::vector<VectorJacobian> jacobians;

    EXPECT_THROW(basis.evaluate_vector_jacobians({Real(0.2), Real(0.3), Real(0)}, jacobians),
                 BasisEvaluationException);
}

TEST(VectorBasisJacobians, HDivDivergenceMatchesJacobianTrace) {
    const std::array<std::pair<ElementType, int>, 5> rt_cases = {{
        {ElementType::Triangle3, 0},
        {ElementType::Quad4, 0},
        {ElementType::Tetra4, 0},
        {ElementType::Hex8, 0},
        {ElementType::Wedge6, 0},
    }};

    const auto xi = svmp::FE::math::Vector<Real, 3>{Real(0.2), Real(0.15), Real(0.1)};
    for (const auto& [type, order] : rt_cases) {
        RaviartThomasBasis basis(type, order);
        std::vector<VectorJacobian> jacobians;
        std::vector<Real> divergence;
        basis.evaluate_vector_jacobians(xi, jacobians);
        basis.evaluate_divergence(xi, divergence);

        ASSERT_EQ(jacobians.size(), divergence.size()) << static_cast<int>(type);
        for (std::size_t i = 0; i < jacobians.size(); ++i) {
            Real trace = Real(0);
            for (int d = 0; d < basis.dimension(); ++d) {
                trace += jacobians[i](static_cast<std::size_t>(d), static_cast<std::size_t>(d));
            }
            EXPECT_NEAR(trace, divergence[i], 1e-11)
                << "type=" << static_cast<int>(type) << ", i=" << i;
        }
    }

    for (ElementType type : {ElementType::Triangle3, ElementType::Quad4, ElementType::Tetra4}) {
        BDMBasis basis(type, 1);
        std::vector<VectorJacobian> jacobians;
        std::vector<Real> divergence;
        basis.evaluate_vector_jacobians(xi, jacobians);
        basis.evaluate_divergence(xi, divergence);

        ASSERT_EQ(jacobians.size(), divergence.size()) << static_cast<int>(type);
        for (std::size_t i = 0; i < jacobians.size(); ++i) {
            Real trace = Real(0);
            for (int d = 0; d < basis.dimension(); ++d) {
                trace += jacobians[i](static_cast<std::size_t>(d), static_cast<std::size_t>(d));
            }
            EXPECT_NEAR(trace, divergence[i], 1e-11)
                << "type=" << static_cast<int>(type) << ", i=" << i;
        }
    }
}

TEST(VectorBasisJacobians, HCurlCurlMatchesJacobianSkewPart) {
    const std::array<std::pair<ElementType, int>, 4> cases = {{
        {ElementType::Triangle3, 0},
        {ElementType::Quad4, 0},
        {ElementType::Tetra4, 0},
        {ElementType::Hex8, 0},
    }};

    const auto xi = svmp::FE::math::Vector<Real, 3>{Real(0.2), Real(-0.15), Real(0.1)};
    for (const auto& [type, order] : cases) {
        NedelecBasis basis(type, order);
        std::vector<VectorJacobian> jacobians;
        std::vector<svmp::FE::math::Vector<Real, 3>> curl;
        basis.evaluate_vector_jacobians(xi, jacobians);
        basis.evaluate_curl(xi, curl);

        ASSERT_EQ(jacobians.size(), curl.size()) << static_cast<int>(type);
        for (std::size_t i = 0; i < jacobians.size(); ++i) {
            const auto& J = jacobians[i];
            svmp::FE::math::Vector<Real, 3> from_J{
                J(2, 1) - J(1, 2),
                J(0, 2) - J(2, 0),
                J(1, 0) - J(0, 1),
            };
            for (int d = 0; d < 3; ++d) {
                EXPECT_NEAR(from_J[static_cast<std::size_t>(d)], curl[i][static_cast<std::size_t>(d)], 1e-11)
                    << "type=" << static_cast<int>(type) << ", i=" << i << ", d=" << d;
            }
        }
    }
}

TEST(VectorBasisJacobians, ReferenceJacobiansMatchValueFiniteDifferences) {
    const auto xi = svmp::FE::math::Vector<Real, 3>{Real(0.21), Real(-0.17), Real(0.13)};
    constexpr Real h = Real(1e-6);

    auto check_basis = [&](const VectorBasisFunction& basis) {
        std::vector<VectorJacobian> jacobians;
        basis.evaluate_vector_jacobians(xi, jacobians);
        ASSERT_EQ(jacobians.size(), basis.size());

        for (int d = 0; d < basis.dimension(); ++d) {
            auto xp = xi;
            auto xm = xi;
            xp[static_cast<std::size_t>(d)] += h;
            xm[static_cast<std::size_t>(d)] -= h;

            std::vector<svmp::FE::math::Vector<Real, 3>> vp;
            std::vector<svmp::FE::math::Vector<Real, 3>> vm;
            basis.evaluate_vector_values(xp, vp);
            basis.evaluate_vector_values(xm, vm);
            ASSERT_EQ(vp.size(), jacobians.size());
            ASSERT_EQ(vm.size(), jacobians.size());

            for (std::size_t i = 0; i < jacobians.size(); ++i) {
                for (int c = 0; c < 3; ++c) {
                    const Real fd = (vp[i][static_cast<std::size_t>(c)] -
                                     vm[i][static_cast<std::size_t>(c)]) / (Real(2) * h);
                    EXPECT_NEAR(jacobians[i](static_cast<std::size_t>(c), static_cast<std::size_t>(d)),
                                fd,
                                2e-7)
                        << "basis=" << static_cast<int>(basis.basis_type())
                        << ", type=" << static_cast<int>(basis.element_type())
                        << ", i=" << i << ", c=" << c << ", d=" << d;
                }
            }
        }
    };

    RaviartThomasBasis rt(ElementType::Tetra4, 0);
    NedelecBasis nd(ElementType::Tetra4, 0);
    BDMBasis bdm(ElementType::Triangle3, 1);

    check_basis(rt);
    check_basis(nd);
    check_basis(bdm);
}

TEST(VectorBasisJacobians, CompatibleTensorJacobiansMatchComponentGradientsAndOperators) {
    auto bx = std::make_shared<LagrangeBasis>(ElementType::Quad4, 1);
    auto by = std::make_shared<LagrangeBasis>(ElementType::Quad4, 1);
    const std::vector<DofAssociation> associations(bx->size() + by->size());

    CompatibleTensorVectorBasis hdiv(CompatibleTensorVectorBasis::Family::HDiv,
                                     BasisType::Lagrange,
                                     bx,
                                     by,
                                     associations,
                                     1,
                                     ElementType::Quad4);
    CompatibleTensorVectorBasis hcurl(CompatibleTensorVectorBasis::Family::HCurl,
                                      BasisType::Lagrange,
                                      std::make_shared<LagrangeBasis>(ElementType::Quad4, 1),
                                      std::make_shared<LagrangeBasis>(ElementType::Quad4, 1),
                                      associations,
                                      1,
                                      ElementType::Quad4);

    const auto xi = svmp::FE::math::Vector<Real, 3>{Real(0.2), Real(-0.3), Real(0)};

    std::vector<Gradient> gradients;
    bx->evaluate_gradients(xi, gradients);

    std::vector<VectorJacobian> jacobians;
    hdiv.evaluate_vector_jacobians(xi, jacobians);
    ASSERT_EQ(jacobians.size(), associations.size());
    for (std::size_t i = 0; i < bx->size(); ++i) {
        EXPECT_NEAR(jacobians[i](0, 0), gradients[i][0], 1e-14);
        EXPECT_NEAR(jacobians[i](0, 1), gradients[i][1], 1e-14);
        EXPECT_NEAR(jacobians[bx->size() + i](1, 0), gradients[i][0], 1e-14);
        EXPECT_NEAR(jacobians[bx->size() + i](1, 1), gradients[i][1], 1e-14);
    }

    std::vector<Real> divergence;
    hdiv.evaluate_divergence(xi, divergence);
    ASSERT_EQ(divergence.size(), jacobians.size());
    for (std::size_t i = 0; i < jacobians.size(); ++i) {
        EXPECT_NEAR(jacobians[i](0, 0) + jacobians[i](1, 1), divergence[i], 1e-14);
    }

    hcurl.evaluate_vector_jacobians(xi, jacobians);
    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    hcurl.evaluate_curl(xi, curl);
    ASSERT_EQ(curl.size(), jacobians.size());
    for (std::size_t i = 0; i < jacobians.size(); ++i) {
        EXPECT_NEAR(jacobians[i](1, 0) - jacobians[i](0, 1), curl[i][2], 1e-14);
    }

    const std::vector<svmp::FE::math::Vector<Real, 3>> points = {
        {Real(0.2), Real(-0.3), Real(0)},
        {Real(-0.4), Real(0.1), Real(0)}
    };
    const std::size_t stride = points.size() + 2u;

    std::vector<Real> values(hdiv.size() * 3u * stride, Real(-17));
    std::vector<Real> jacs(hdiv.size() * 9u * stride, Real(-17));
    std::vector<Real> divs(hdiv.size() * stride, Real(-17));
    hdiv.evaluate_vector_at_quadrature_points_strided(points,
                                                      stride,
                                                      values.data(),
                                                      jacs.data(),
                                                      nullptr,
                                                      divs.data());

    std::vector<svmp::FE::math::Vector<Real, 3>> expected_values;
    for (std::size_t q = 0; q < points.size(); ++q) {
        hdiv.evaluate_vector_values(points[q], expected_values);
        hdiv.evaluate_vector_jacobians(points[q], jacobians);
        hdiv.evaluate_divergence(points[q], divergence);
        for (std::size_t i = 0; i < hdiv.size(); ++i) {
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(values[(i * 3u + component) * stride + q],
                            expected_values[i][component],
                            1e-14);
                for (std::size_t derivative = 0; derivative < 3u; ++derivative) {
                    EXPECT_NEAR(jacs[(i * 9u + component * 3u + derivative) * stride + q],
                                jacobians[i](component, derivative),
                                1e-14);
                }
            }
            EXPECT_NEAR(divs[i * stride + q], divergence[i], 1e-14);
        }
    }

    std::vector<Real> curl_values(hcurl.size() * 3u * stride, Real(-19));
    std::vector<Real> curl_jacs(hcurl.size() * 9u * stride, Real(-19));
    std::vector<Real> curls(hcurl.size() * 3u * stride, Real(-19));
    hcurl.evaluate_vector_at_quadrature_points_strided(points,
                                                       stride,
                                                       curl_values.data(),
                                                       curl_jacs.data(),
                                                       curls.data(),
                                                       nullptr);
    for (std::size_t q = 0; q < points.size(); ++q) {
        hcurl.evaluate_vector_values(points[q], expected_values);
        hcurl.evaluate_vector_jacobians(points[q], jacobians);
        hcurl.evaluate_curl(points[q], curl);
        for (std::size_t i = 0; i < hcurl.size(); ++i) {
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(curl_values[(i * 3u + component) * stride + q],
                            expected_values[i][component],
                            1e-14);
                EXPECT_NEAR(curls[(i * 3u + component) * stride + q],
                            curl[i][component],
                            1e-14);
                for (std::size_t derivative = 0; derivative < 3u; ++derivative) {
                    EXPECT_NEAR(curl_jacs[(i * 9u + component * 3u + derivative) * stride + q],
                                jacobians[i](component, derivative),
                                1e-14);
                }
            }
        }
    }

    for (std::size_t i = 0; i < hdiv.size(); ++i) {
        for (std::size_t pad = points.size(); pad < stride; ++pad) {
            EXPECT_DOUBLE_EQ(values[(i * 3u) * stride + pad], Real(-17));
            EXPECT_DOUBLE_EQ(divs[i * stride + pad], Real(-17));
        }
    }
}

TEST(BDMBasis, DivergenceLinear) {
    BDMBasis basis(ElementType::Quad4, 1);
    svmp::FE::math::Vector<Real, 3> xi{0.3, -0.4, 0.0};
    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 8u);
    EXPECT_NEAR(div[0], 0.5, 1e-12);
    EXPECT_NEAR(div[1], 0.5 * xi[0], 1e-12);
    EXPECT_NEAR(div[2], 0.5, 1e-12);
    EXPECT_NEAR(div[3], 0.5 * xi[1], 1e-12);
    EXPECT_NEAR(div[4], 0.5, 1e-12);
    EXPECT_NEAR(div[5], 0.5 * xi[0], 1e-12);
    EXPECT_NEAR(div[6], 0.5, 1e-12);
    EXPECT_NEAR(div[7], 0.5 * xi[1], 1e-12);
}

TEST(BDMBasis, TriangleDivergenceLinear) {
    BDMBasis basis(ElementType::Triangle3, 1);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.4), Real(0)};
    std::vector<Real> div;
    basis.evaluate_divergence(xi, div);
    ASSERT_EQ(div.size(), 6u);
    EXPECT_NEAR(div[0], 2.0, 1e-12);
    EXPECT_NEAR(div[1], 0.0, 1e-12);
    EXPECT_NEAR(div[2], 2.0, 1e-12);
    EXPECT_NEAR(div[3], 0.0, 1e-12);
    EXPECT_NEAR(div[4], 2.0, 1e-12);
    EXPECT_NEAR(div[5], 0.0, 1e-12);
}

TEST(BDMBasis, QuadEdgeNormalTracesMatchConstruction) {
    BDMBasis basis(ElementType::Quad4, 1);
    const Real samples[] = {Real(-0.75), Real(0), Real(0.6)};

    for (int edge = 0; edge < 4; ++edge) {
        const int constant_func = 2 * edge;
        const int linear_func = constant_func + 1;

        for (Real s : samples) {
            for (int func = 0; func < static_cast<int>(basis.size()); ++func) {
                const Real flux = evaluate_quad_bdm_edge_normal_flux(basis, edge, func, s);
                if (func == constant_func) {
                    EXPECT_NEAR(flux, Real(1), 1e-12)
                        << "edge=" << edge << ", s=" << s;
                } else if (func == linear_func) {
                    EXPECT_NEAR(flux, s, 1e-12)
                        << "edge=" << edge << ", s=" << s;
                } else {
                    EXPECT_NEAR(flux, Real(0), 1e-12)
                        << "edge=" << edge << ", s=" << s << ", func=" << func;
                }
            }
        }
    }
}

TEST(BDMBasis, TriangleEdgeNormalTracesMatchConstruction) {
    BDMBasis basis(ElementType::Triangle3, 1);
    const Real samples[] = {Real(-0.7), Real(0), Real(0.4)};

    for (int edge = 0; edge < 3; ++edge) {
        const int constant_func = 2 * edge;
        const int linear_func = constant_func + 1;

        for (Real s : samples) {
            for (int func = 0; func < static_cast<int>(basis.size()); ++func) {
                const Real flux = evaluate_triangle_bdm_edge_scaled_normal_flux(basis, edge, func, s);
                if (func == constant_func) {
                    EXPECT_NEAR(flux, Real(1), 1e-12)
                        << "edge=" << edge << ", s=" << s;
                } else if (func == linear_func) {
                    EXPECT_NEAR(flux, Real(3) * s, 1e-12)
                        << "edge=" << edge << ", s=" << s;
                } else {
                    EXPECT_NEAR(flux, Real(0), 1e-12)
                        << "edge=" << edge << ", s=" << s << ", func=" << func;
                }
            }
        }
    }
}

TEST(BDMBasis, QuadDivergenceMatchesBoundaryFlux) {
    BDMBasis basis(ElementType::Quad4, 1);
    ASSERT_EQ(basis.size(), 8u);

    for (int func = 0; func < static_cast<int>(basis.size()); ++func) {
        const double volume = integrate_quad_bdm_divergence(basis, func);
        double boundary = 0.0;
        for (int edge = 0; edge < 4; ++edge) {
            boundary += integrate_quad_bdm_edge_flux(basis, edge, func);
        }
        EXPECT_NEAR(volume, boundary, 1e-11) << "func=" << func;
    }
}

TEST(BDMBasis, TriangleDivergenceMatchesBoundaryFlux) {
    BDMBasis basis(ElementType::Triangle3, 1);
    ASSERT_EQ(basis.size(), 6u);

    for (int func = 0; func < static_cast<int>(basis.size()); ++func) {
        const double volume = integrate_triangle_bdm_divergence(basis, func);
        double boundary = 0.0;
        for (int edge = 0; edge < 3; ++edge) {
            boundary += integrate_triangle_bdm_edge_flux(basis, edge, func);
        }
        EXPECT_NEAR(volume, boundary, 1e-11) << "func=" << func;
    }
}

TEST(BDMBasis, QuadEdgeMomentsAreKroneckerWithDofAssociations) {
    BDMBasis basis(ElementType::Quad4, 1);
    const auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), basis.size());

    for (std::size_t func = 0; func < basis.size(); ++func) {
        ASSERT_EQ(assoc[func].entity_type, DofEntity::Edge);
        for (int edge = 0; edge < 4; ++edge) {
            for (int mode = 0; mode < 2; ++mode) {
                const double moment =
                    integrate_quad_bdm_edge_moment(basis, edge, mode, static_cast<int>(func));
                const double expected =
                    (assoc[func].entity_id == edge && assoc[func].moment_index == mode) ? 1.0 : 0.0;
                EXPECT_NEAR(moment, expected, 1e-11)
                    << "func=" << func << ", edge=" << edge << ", mode=" << mode;
            }
        }
    }
}

TEST(BDMBasis, TriangleEdgeMomentsAreKroneckerWithDofAssociations) {
    BDMBasis basis(ElementType::Triangle3, 1);
    const auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), basis.size());

    for (std::size_t func = 0; func < basis.size(); ++func) {
        ASSERT_EQ(assoc[func].entity_type, DofEntity::Edge);
        for (int edge = 0; edge < 3; ++edge) {
            for (int mode = 0; mode < 2; ++mode) {
                const double moment =
                    integrate_triangle_bdm_edge_moment(basis, edge, mode, static_cast<int>(func));
                const double expected =
                    (assoc[func].entity_id == edge && assoc[func].moment_index == mode) ? 1.0 : 0.0;
                EXPECT_NEAR(moment, expected, 1e-11)
                    << "func=" << func << ", edge=" << edge << ", mode=" << mode;
            }
        }
    }
}

TEST(BDMBasis, TriangleHigherOrderEdgeAndInteriorMomentsAreKronecker) {
    for (const int order : {2, 3}) {
        BDMBasis basis(ElementType::Triangle3, order);
        const auto assoc = basis.dof_associations();
        ASSERT_EQ(assoc.size(), basis.size());

        const std::size_t edge_dofs = static_cast<std::size_t>(3 * (order + 1));
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            for (std::size_t func = 0; func < basis.size(); ++func) {
                double moment = 0.0;
                if (assoc[dof].entity_type == DofEntity::Edge) {
                    moment = integrate_triangle_bdm_edge_moment_exact(
                        basis,
                        assoc[dof].entity_id,
                        assoc[dof].moment_index,
                        static_cast<int>(func),
                        order);
                } else {
                    ASSERT_EQ(assoc[dof].entity_type, DofEntity::Interior);
                    moment = integrate_triangle_bdm_interior_moment(
                        basis,
                        assoc[dof].moment_index,
                        static_cast<int>(func),
                        order);
                }

                const double expected = (dof == func) ? 1.0 : 0.0;
                const double tol = (dof < edge_dofs && func < edge_dofs) ? 2e-10 : 2e-9;
                EXPECT_NEAR(moment, expected, tol)
                    << "order=" << order << ", dof=" << dof << ", func=" << func;
            }
        }
    }
}

TEST(BDMBasis, TetraHigherOrderFaceAndInteriorMomentsAreKronecker) {
    for (const int order : {1, 2, 3}) {
        BDMBasis basis(ElementType::Tetra4, order);
        const auto assoc = basis.dof_associations();
        ASSERT_EQ(assoc.size(), basis.size());

        const std::size_t face_dofs =
            4u * static_cast<std::size_t>((order + 1) * (order + 2) / 2);
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            for (std::size_t func = 0; func < basis.size(); ++func) {
                double moment = 0.0;
                if (assoc[dof].entity_type == DofEntity::Face) {
                    moment = integrate_tetra_bdm_face_moment(
                        basis,
                        assoc[dof].entity_id,
                        assoc[dof].moment_index,
                        static_cast<int>(func),
                        order);
                } else {
                    ASSERT_EQ(assoc[dof].entity_type, DofEntity::Interior);
                    moment = integrate_tetra_bdm_interior_moment(
                        basis,
                        assoc[dof].moment_index,
                        static_cast<int>(func),
                        order);
                }

                const double expected = (dof == func) ? 1.0 : 0.0;
                const double tol = (dof < face_dofs && func < face_dofs) ? 5e-9 : 1e-8;
                EXPECT_NEAR(moment, expected, tol)
                    << "order=" << order << ", dof=" << dof << ", func=" << func;
            }
        }
    }
}

TEST(BDMBasis, RepeatedHigherOrderConstructionIsNumericallyStable) {
    const struct Case {
        ElementType type;
        int order;
        math::Vector<Real, 3> point;
        Real tolerance;
    } cases[] = {
        {ElementType::Triangle3, 3, {Real(0.21), Real(0.27), Real(0)}, Real(1e-12)},
        {ElementType::Tetra4, 3, {Real(0.16), Real(0.19), Real(0.23)}, Real(1e-11)},
    };

    for (const auto& c : cases) {
        BDMBasis first(c.type, c.order);
        BDMBasis second(c.type, c.order);
        ASSERT_EQ(first.size(), second.size());

        std::vector<math::Vector<Real, 3>> first_values;
        std::vector<math::Vector<Real, 3>> second_values;
        first.evaluate_vector_values(c.point, first_values);
        second.evaluate_vector_values(c.point, second_values);
        ASSERT_EQ(first_values.size(), second_values.size());

        for (std::size_t i = 0; i < first_values.size(); ++i) {
            for (std::size_t d = 0; d < 3u; ++d) {
                EXPECT_NEAR(first_values[i][d], second_values[i][d], c.tolerance)
                    << "basis=" << i << ", dim=" << d;
            }
        }

        std::vector<Real> first_divergence;
        std::vector<Real> second_divergence;
        first.evaluate_divergence(c.point, first_divergence);
        second.evaluate_divergence(c.point, second_divergence);
        ASSERT_EQ(first_divergence.size(), second_divergence.size());
        for (std::size_t i = 0; i < first_divergence.size(); ++i) {
            EXPECT_NEAR(first_divergence[i], second_divergence[i], c.tolerance)
                << "divergence basis=" << i;
        }
    }
}

TEST(BDMBasis, TriangleDivergenceLiesInPkMinusOne) {
    using svmp::FE::math::Vector;

    const std::array<Vector<Real, 3>, 3> sample_points = {
        Vector<Real, 3>{Real(0.17), Real(0.21), Real(0)},
        Vector<Real, 3>{Real(0.32), Real(0.18), Real(0)},
        Vector<Real, 3>{Real(0.14), Real(0.41), Real(0)}
    };

    for (const int order : {1, 2, 3}) {
        BDMBasis basis(ElementType::Triangle3, order);

        if (order == 1) {
            for (std::size_t func = 0; func < basis.size(); ++func) {
                std::vector<Real> div_ref;
                basis.evaluate_divergence(sample_points[0], div_ref);
                for (std::size_t p = 1; p < sample_points.size(); ++p) {
                    std::vector<Real> div;
                    basis.evaluate_divergence(sample_points[p], div);
                    EXPECT_NEAR(div[func], div_ref[func], 1e-12)
                        << "order=" << order << ", func=" << func;
                }
            }
            continue;
        }

        LagrangeBasis scalar_basis(ElementType::Triangle3, order - 1);
        const auto nodes = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Triangle3, order - 1);
        ASSERT_EQ(nodes.size(), scalar_basis.size());

        for (std::size_t func = 0; func < basis.size(); ++func) {
            std::vector<Real> nodal_values(nodes.size(), Real(0));
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                std::vector<Real> div;
                basis.evaluate_divergence(nodes[i], div);
                nodal_values[i] = div[func];
            }

            for (const auto& xi : sample_points) {
                std::vector<Real> phi;
                scalar_basis.evaluate_values(xi, phi);
                Real interpolated = Real(0);
                for (std::size_t i = 0; i < phi.size(); ++i) {
                    interpolated += phi[i] * nodal_values[i];
                }

                std::vector<Real> div;
                basis.evaluate_divergence(xi, div);
                EXPECT_NEAR(div[func], interpolated, 1e-10)
                    << "order=" << order << ", func=" << func;
            }
        }
    }
}

TEST(BDMBasis, TetraDivergenceLiesInPkMinusOne) {
    using svmp::FE::math::Vector;

    const std::array<Vector<Real, 3>, 4> sample_points = {
        Vector<Real, 3>{Real(0.12), Real(0.17), Real(0.18)},
        Vector<Real, 3>{Real(0.24), Real(0.11), Real(0.16)},
        Vector<Real, 3>{Real(0.18), Real(0.22), Real(0.09)},
        Vector<Real, 3>{Real(0.09), Real(0.16), Real(0.14)}
    };

    for (const int order : {1, 2, 3}) {
        BDMBasis basis(ElementType::Tetra4, order);

        if (order == 1) {
            for (std::size_t func = 0; func < basis.size(); ++func) {
                std::vector<Real> div_ref;
                basis.evaluate_divergence(sample_points[0], div_ref);
                for (std::size_t p = 1; p < sample_points.size(); ++p) {
                    std::vector<Real> div;
                    basis.evaluate_divergence(sample_points[p], div);
                    EXPECT_NEAR(div[func], div_ref[func], 1e-12)
                        << "order=" << order << ", func=" << func;
                }
            }
            continue;
        }

        LagrangeBasis scalar_basis(ElementType::Tetra4, order - 1);
        const auto nodes = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Tetra4, order - 1);
        ASSERT_EQ(nodes.size(), scalar_basis.size());

        for (std::size_t func = 0; func < basis.size(); ++func) {
            std::vector<Real> nodal_values(nodes.size(), Real(0));
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                std::vector<Real> div;
                basis.evaluate_divergence(nodes[i], div);
                nodal_values[i] = div[func];
            }

            for (const auto& xi : sample_points) {
                std::vector<Real> phi;
                scalar_basis.evaluate_values(xi, phi);
                Real interpolated = Real(0);
                for (std::size_t i = 0; i < phi.size(); ++i) {
                    interpolated += phi[i] * nodal_values[i];
                }

                std::vector<Real> div;
                basis.evaluate_divergence(xi, div);
                EXPECT_NEAR(div[func], interpolated, 1e-10)
                    << "order=" << order << ", func=" << func;
            }
        }
    }
}

TEST(BDMBasis, HigherOrderDofAssociationsMatchExpectedBoundaryInteriorSplit) {
    {
        const int order = 2;
        BDMBasis basis(ElementType::Triangle3, order);
        const auto assoc = basis.dof_associations();
        const int edge_dofs = std::count_if(assoc.begin(), assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Edge;
        });
        const int interior_dofs = std::count_if(assoc.begin(), assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Interior;
        });
        EXPECT_EQ(edge_dofs, 3 * (order + 1));
        EXPECT_EQ(interior_dofs, static_cast<int>(basis.size()) - edge_dofs);
        EXPECT_TRUE(std::all_of(assoc.begin() + edge_dofs, assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Interior;
        }));
    }

    {
        const int order = 3;
        BDMBasis basis(ElementType::Tetra4, order);
        const auto assoc = basis.dof_associations();
        const int face_dofs = std::count_if(assoc.begin(), assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Face;
        });
        const int interior_dofs = std::count_if(assoc.begin(), assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Interior;
        });
        const int expected_face_dofs = 4 * ((order + 1) * (order + 2) / 2);
        EXPECT_EQ(face_dofs, expected_face_dofs);
        EXPECT_EQ(interior_dofs, static_cast<int>(basis.size()) - face_dofs);
        EXPECT_TRUE(std::all_of(assoc.begin() + face_dofs, assoc.end(), [](const DofAssociation& a) {
            return a.entity_type == DofEntity::Interior;
        }));
    }
}

TEST(BDMBasis, TetraFaceMomentOrderingIsInvariantUnderCyclicFaceParameterization) {
    using svmp::FE::math::Vector;

    const int order = 2;
    const std::array<int, 3> cyclic_perm = {1, 2, 0};
    const auto face_nodes = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Triangle3, order);
    BDMBasis basis(ElementType::Tetra4, order);

    auto map_mode_under_permutation = [&](std::size_t mode) -> std::size_t {
        const Vector<Real, 3>& xi = face_nodes[mode];
        const Real l0 = Real(1) - xi[0] - xi[1];
        const std::array<Real, 3> bary = {l0, xi[0], xi[1]};
        const Vector<Real, 3> permuted_xi{bary[static_cast<std::size_t>(cyclic_perm[1])],
                                          bary[static_cast<std::size_t>(cyclic_perm[2])],
                                          Real(0)};
        for (std::size_t i = 0; i < face_nodes.size(); ++i) {
            if (std::abs(face_nodes[i][0] - permuted_xi[0]) < 1e-12 &&
                std::abs(face_nodes[i][1] - permuted_xi[1]) < 1e-12) {
                return i;
            }
        }
        ADD_FAILURE() << "Failed to map face mode under cyclic permutation";
        return std::size_t(0);
    };

    for (int face = 0; face < 4; ++face) {
        for (std::size_t mode = 0; mode < face_nodes.size(); ++mode) {
            const std::size_t permuted_mode = map_mode_under_permutation(mode);
            for (std::size_t func = 0; func < basis.size(); ++func) {
                const double ref_moment =
                    integrate_tetra_bdm_face_moment(basis, face, static_cast<int>(mode), static_cast<int>(func), order);
                const double permuted_moment =
                    integrate_tetra_bdm_face_moment_with_vertex_permutation(
                        basis,
                        face,
                        cyclic_perm,
                        static_cast<int>(permuted_mode),
                        static_cast<int>(func),
                        order);
                EXPECT_NEAR(permuted_moment, ref_moment, 5e-10)
                    << "face=" << face << ", mode=" << mode << ", func=" << func;
            }
        }
    }
}

TEST(VectorBasis, ScalarEvaluateThrows) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};
    std::vector<Real> vals;
    EXPECT_THROW(basis.evaluate_values(xi, vals), svmp::FE::basis::BasisEvaluationException);
}

TEST(VectorBasis, HexahedronInstantiationAndSize) {
    RaviartThomasBasis rt_hex(ElementType::Hex8, 0);
    NedelecBasis ned_hex(ElementType::Hex8, 0);
    EXPECT_EQ(rt_hex.dimension(), 3);
    EXPECT_EQ(ned_hex.dimension(), 3);
    EXPECT_EQ(rt_hex.size(), 6u);
    EXPECT_EQ(ned_hex.size(), 12u);
}

TEST(VectorBasis, WedgeAndPyramidInstantiationAndSize) {
    RaviartThomasBasis rt_wedge(ElementType::Wedge6, 0);
    EXPECT_EQ(rt_wedge.dimension(), 3);
    EXPECT_EQ(rt_wedge.size(), 5u);

    NedelecBasis ned_wedge(ElementType::Wedge6, 0);
    EXPECT_EQ(ned_wedge.dimension(), 3);
    EXPECT_EQ(ned_wedge.size(), 9u);

    RaviartThomasBasis rt_pyr(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt_pyr.dimension(), 3);
    EXPECT_EQ(rt_pyr.size(), 5u);

    NedelecBasis ned_pyr(ElementType::Pyramid5, 0);
    EXPECT_EQ(ned_pyr.dimension(), 3);
    EXPECT_EQ(ned_pyr.size(), 8u);
}

TEST(RaviartThomasBasis, PyramidRT0DivergenceConstant) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    svmp::FE::math::Vector<Real, 3> xi1{Real(0.1), Real(0.2), Real(0.3)};
    svmp::FE::math::Vector<Real, 3> xi2{Real(-0.3), Real(0.1), Real(0.5)};
    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);
    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-14);
    }
}

TEST(VectorBasis, TriangleAndTetraInstantiationAndSize) {
    RaviartThomasBasis rt_tri(ElementType::Triangle3, 0);
    EXPECT_EQ(rt_tri.dimension(), 2);
    EXPECT_EQ(rt_tri.size(), 3u);

    RaviartThomasBasis rt_tet(ElementType::Tetra4, 0);
    EXPECT_EQ(rt_tet.dimension(), 3);
    EXPECT_EQ(rt_tet.size(), 4u);

    NedelecBasis ned_tri(ElementType::Triangle3, 0);
    EXPECT_EQ(ned_tri.dimension(), 2);
    EXPECT_EQ(ned_tri.size(), 3u);

    NedelecBasis ned_tet(ElementType::Tetra4, 0);
    EXPECT_EQ(ned_tet.dimension(), 3);
    EXPECT_EQ(ned_tet.size(), 6u);
}

TEST(RaviartThomasBasis, TriangleAndTetraDivergenceConstants) {
    RaviartThomasBasis rt_tri(ElementType::Triangle3, 0);
    svmp::FE::math::Vector<Real, 3> xi_tri1{Real(0.2), Real(0.3), Real(0)};
    svmp::FE::math::Vector<Real, 3> xi_tri2{Real(0.1), Real(0.2), Real(0)};
    std::vector<Real> div_tri1, div_tri2;
    rt_tri.evaluate_divergence(xi_tri1, div_tri1);
    rt_tri.evaluate_divergence(xi_tri2, div_tri2);
    ASSERT_EQ(div_tri1.size(), 3u);
    ASSERT_EQ(div_tri2.size(), 3u);
    for (std::size_t i = 0; i < div_tri1.size(); ++i) {
        EXPECT_NEAR(div_tri1[i], div_tri2[i], 1e-14);
    }

    RaviartThomasBasis rt_tet(ElementType::Tetra4, 0);
    svmp::FE::math::Vector<Real, 3> xi_tet1{Real(0.1), Real(0.2), Real(0.1)};
    svmp::FE::math::Vector<Real, 3> xi_tet2{Real(0.2), Real(0.1), Real(0.2)};
    std::vector<Real> div_tet1, div_tet2;
    rt_tet.evaluate_divergence(xi_tet1, div_tet1);
    rt_tet.evaluate_divergence(xi_tet2, div_tet2);
    ASSERT_EQ(div_tet1.size(), 4u);
    ASSERT_EQ(div_tet2.size(), 4u);
    for (std::size_t i = 0; i < div_tet1.size(); ++i) {
        EXPECT_NEAR(div_tet1[i], div_tet2[i], 1e-14);
    }
}

TEST(RaviartThomasBasis, WedgeDivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    svmp::FE::math::Vector<Real, 3> xi1{Real(0.2), Real(0.3), Real(-0.4)};
    svmp::FE::math::Vector<Real, 3> xi2{Real(0.1), Real(0.1), Real(0.5)};
    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);
    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-14);
    }
}

TEST(NedelecBasis, TriangleAndTetraCurls) {
    NedelecBasis ned_tri(ElementType::Triangle3, 0);
    svmp::FE::math::Vector<Real, 3> xi_tri{Real(0.2), Real(0.3), Real(0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> curl_tri;
    ned_tri.evaluate_curl(xi_tri, curl_tri);
    ASSERT_EQ(curl_tri.size(), 3u);
    for (const auto& c : curl_tri) {
        EXPECT_GT(c[2], 1.5); // all have positive z-curl
    }

    NedelecBasis ned_tet(ElementType::Tetra4, 0);
    svmp::FE::math::Vector<Real, 3> xi_tet{Real(0.1), Real(0.2), Real(0.1)};
    std::vector<svmp::FE::math::Vector<Real, 3>> curl_tet;
    ned_tet.evaluate_curl(xi_tet, curl_tet);
    ASSERT_EQ(curl_tet.size(), 6u);
    // Reuse existing expectations on hex-like patterns: non-zero components present
    int nonzero = 0;
    for (const auto& c : curl_tet) {
        if (std::abs(c[0]) > 1e-14 || std::abs(c[1]) > 1e-14 || std::abs(c[2]) > 1e-14) {
            ++nonzero;
        }
    }
    EXPECT_GE(nonzero, 3);
}

TEST(NedelecBasis, TriangleOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Triangle3, k);
    ASSERT_EQ(basis.size(), 8u);

    const int edge_dofs = 3 * (k + 1);
    for (int e = 0; e < 3; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_triangle_edge_tangential_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 1e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 1e-10);
                }
            }
        }
    }
}

TEST(NedelecBasis, TetraOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Tetra4, k);
    ASSERT_EQ(basis.size(), 20u);

    const int edge_dofs = 6 * (k + 1);
    for (int e = 0; e < 6; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_tetra_edge_tangential_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 5e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 5e-10);
                }
            }
        }
    }
}

TEST(NedelecBasis, TetraOrderOneFaceDofsAreKronecker) {
    constexpr int k = 1;
    NedelecBasis basis(ElementType::Tetra4, k);
    ASSERT_EQ(basis.size(), 20u);

    const int edge_dofs = 6 * (k + 1);
    const int face_dofs = 4 * 2; // 4 faces, 2 tangential DOFs each for k=1

    for (int f = 0; f < 4; ++f) {
        for (int dir = 0; dir < 2; ++dir) {
            const int dof_idx = edge_dofs + 2 * f + dir;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_tetra_face_tangential_moment(
                    basis, f, dir, /*mode_id=*/0, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 5e-9);
                } else if (static_cast<int>(j) >= edge_dofs &&
                           static_cast<int>(j) < edge_dofs + face_dofs) {
                    EXPECT_NEAR(dof, 0.0, 5e-9);
                }
            }
        }
    }
}

TEST(RaviartThomasBasis, TriangleOrderOneEdgeDofsAreKronecker) {
    constexpr int k = 1;
    RaviartThomasBasis basis(ElementType::Triangle3, k);
    ASSERT_EQ(basis.size(), 8u);

    const int edge_dofs = 3 * (k + 1);
    for (int e = 0; e < 3; ++e) {
        for (int a = 0; a <= k; ++a) {
            const int dof_idx = e * (k + 1) + a;
            for (std::size_t j = 0; j < basis.size(); ++j) {
                const double dof = integrate_triangle_edge_normal_moment(
                    basis, e, a, static_cast<int>(j), k);
                if (static_cast<int>(j) == dof_idx) {
                    EXPECT_NEAR(dof, 1.0, 1e-10);
                } else if (static_cast<int>(j) < edge_dofs) {
                    EXPECT_NEAR(dof, 0.0, 1e-10);
                }
            }
        }
    }
}

TEST(RaviartThomasBasis, HexDivergenceConstants) {
    RaviartThomasBasis basis(ElementType::Hex8, 0);
    const svmp::FE::math::Vector<Real, 3> xi0{Real(0.2), Real(-0.1), Real(0.3)};
    const svmp::FE::math::Vector<Real, 3> xi1{Real(-0.33), Real(0.27), Real(-0.19)};

    std::vector<Real> div0;
    std::vector<Real> div1;
    basis.evaluate_divergence(xi0, div0);
    basis.evaluate_divergence(xi1, div1);
    ASSERT_EQ(div0.size(), 6u);
    ASSERT_EQ(div1.size(), 6u);

    for (std::size_t i = 0; i < div0.size(); ++i) {
        EXPECT_NEAR(div0[i], div1[i], 1e-12);
        EXPECT_NEAR(std::abs(div0[i]), 0.125, 1e-12);
    }
}

TEST(NedelecBasis, HexCurlConstants) {
    NedelecBasis basis(ElementType::Hex8, 0);
    svmp::FE::math::Vector<Real, 3> xi{Real(0.15), Real(-0.05), Real(0.4)};

    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    basis.evaluate_curl(xi, curl);
    ASSERT_EQ(curl.size(), 12u);
    // Spot-check a few curls against the analytical expressions.
    // Edge 0-1: v = (1/8)(1-y)(1-z) e_x
    EXPECT_NEAR(curl[0][1], -0.125 * (Real(1) - xi[1]), 1e-12);
    EXPECT_NEAR(curl[0][2],  0.125 * (Real(1) - xi[2]), 1e-12);
    // Edge 1-2: v = (1/8)(1+x)(1-z) e_y
    EXPECT_NEAR(curl[1][0],  0.125 * (Real(1) + xi[0]), 1e-12);
    EXPECT_NEAR(curl[1][2],  0.125 * (Real(1) - xi[2]), 1e-12);
    // Vertical edge 0-4: v = (1/8)(1-x)(1-y) e_z
    EXPECT_NEAR(curl[8][0], -0.125 * (Real(1) - xi[0]), 1e-12);
    EXPECT_NEAR(curl[8][1],  0.125 * (Real(1) - xi[1]), 1e-12);

    std::vector<svmp::FE::math::Vector<Real, 3>> vals;
    basis.evaluate_vector_values(xi, vals);
    ASSERT_EQ(vals.size(), 12u);
    // Each edge function has exactly one nonzero component.
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const int nonzero =
            (std::abs(vals[i][0]) > Real(1e-14)) +
            (std::abs(vals[i][1]) > Real(1e-14)) +
            (std::abs(vals[i][2]) > Real(1e-14));
        EXPECT_EQ(nonzero, 1);
    }
}

TEST(NedelecBasis, WedgeMinimalEdgeDofs) {
    NedelecBasis basis(ElementType::Wedge6, 0);

    for (int i = 0; i < 9; ++i) {
        for (int e = 0; e < 9; ++e) {
            double dof = integrate_wedge_edge_dof(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(dof, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(dof, 0.0, 1e-10);
            }
        }
    }
}

TEST(NedelecBasis, PyramidMinimalEdgeDofs) {
    NedelecBasis basis(ElementType::Pyramid5, 0);

    for (int i = 0; i < 8; ++i) {
        for (int e = 0; e < 8; ++e) {
            double dof = integrate_pyramid_edge_dof(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(dof, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(dof, 0.0, 1e-10);
            }
        }
    }
}

TEST(NedelecBasis, PyramidCurlsAreDivergenceFree) {
    std::unique_ptr<NedelecBasis> basis = std::make_unique<NedelecBasis>(ElementType::Pyramid5, 0);

    using svmp::FE::math::Vector;
    const Real h = Real(1e-3);
    // Choose an interior point well inside the pyramid
    Vector<Real,3> xc{Real(0), Real(0), Real(0.25)};

    for (int i = 0; i < 8; ++i) {
        // Approximate divergence of curl φ_i at xc via finite differences
        // ∂curl_x/∂x + ∂curl_y/∂y + ∂curl_z/∂z
        auto eval_curl = [&](const Vector<Real,3>& p) {
            std::vector<Vector<Real,3>> c;
            basis->evaluate_curl(p, c);
            return c[static_cast<std::size_t>(i)];
        };

        Vector<Real,3> c0 = eval_curl(xc);

        Vector<Real,3> pxp{xc[0] + h, xc[1], xc[2]};
        Vector<Real,3> pxm{xc[0] - h, xc[1], xc[2]};
        Vector<Real,3> cxp = eval_curl(pxp);
        Vector<Real,3> cxm = eval_curl(pxm);

        Vector<Real,3> pyp{xc[0], xc[1] + h, xc[2]};
        Vector<Real,3> pym{xc[0], xc[1] - h, xc[2]};
        Vector<Real,3> cyp = eval_curl(pyp);
        Vector<Real,3> cym = eval_curl(pym);

        Vector<Real,3> pzp{xc[0], xc[1], xc[2] + h};
        Vector<Real,3> pzm{xc[0], xc[1], xc[2] - h};
        Vector<Real,3> czp = eval_curl(pzp);
        Vector<Real,3> czm = eval_curl(pzm);

        double dCx_dx = (static_cast<double>(cxp[0] - cxm[0]) / (2.0 * static_cast<double>(h)));
        double dCy_dy = (static_cast<double>(cyp[1] - cym[1]) / (2.0 * static_cast<double>(h)));
        double dCz_dz = (static_cast<double>(czp[2] - czm[2]) / (2.0 * static_cast<double>(h)));

        double div_curl = dCx_dx + dCy_dy + dCz_dz;
        EXPECT_NEAR(div_curl, 0.0, 1e-6);
        (void)c0;
    }
}

static double integrate_edge_normal_flux_quad4(const RaviartThomasBasis& basis,
                                               int edge_id,
                                               int func_id,
                                               int quad_order = 6) {
    using svmp::FE::quadrature::GaussQuadrature1D;
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Quad4);
    const auto& en = ref.edge_nodes(static_cast<std::size_t>(edge_id));
    EXPECT_EQ(en.size(), 2u);
    const Vector<Real, 3> p0 = ReferenceNodeLayout::get_node_coords(ElementType::Quad4, static_cast<std::size_t>(en[0]));
    const Vector<Real, 3> p1 = ReferenceNodeLayout::get_node_coords(ElementType::Quad4, static_cast<std::size_t>(en[1]));

    const Vector<Real, 3> tvec = p1 - p0;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }

    const Vector<Real, 3> t = tvec / len;
    const Vector<Real, 3> nrm{t[1], -t[0], Real(0)};
    const Real J = len * Real(0.5);

    GaussQuadrature1D quad(quad_order);
    double flux = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real, 3> xi = p0 * (Real(1) - tpar) + p1 * tpar;

        std::vector<Vector<Real, 3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];

        flux += static_cast<double>(quad.weight(q) * (J * v.dot(nrm)));
    }
    return flux;
}

static Real evaluate_quad_bdm_edge_normal_flux(const BDMBasis& basis,
                                               int edge_id,
                                               int func_id,
                                               Real s) {
    using svmp::FE::math::Vector;

    Vector<Real, 3> xi{};
    Vector<Real, 3> normal{};
    switch (edge_id) {
        case 0:
            xi = Vector<Real, 3>{s, Real(-1), Real(0)};
            normal = Vector<Real, 3>{Real(0), Real(-1), Real(0)};
            break;
        case 1:
            xi = Vector<Real, 3>{Real(1), s, Real(0)};
            normal = Vector<Real, 3>{Real(1), Real(0), Real(0)};
            break;
        case 2:
            xi = Vector<Real, 3>{s, Real(1), Real(0)};
            normal = Vector<Real, 3>{Real(0), Real(1), Real(0)};
            break;
        default:
            xi = Vector<Real, 3>{Real(-1), s, Real(0)};
            normal = Vector<Real, 3>{Real(-1), Real(0), Real(0)};
            break;
    }

    std::vector<Vector<Real, 3>> values;
    basis.evaluate_vector_values(xi, values);
    return values[static_cast<std::size_t>(func_id)].dot(normal);
}

static Real evaluate_triangle_bdm_edge_scaled_normal_flux(const BDMBasis& basis,
                                                          int edge_id,
                                                          int func_id,
                                                          Real s) {
    using svmp::FE::math::Vector;

    Vector<Real, 3> xi{};
    Vector<Real, 3> normal{};
    switch (edge_id) {
        case 0:
            xi = Vector<Real, 3>{(s + Real(1)) * Real(0.5), Real(0), Real(0)};
            normal = Vector<Real, 3>{Real(0), Real(-1), Real(0)};
            break;
        case 1:
            xi = Vector<Real, 3>{(Real(1) - s) * Real(0.5), (Real(1) + s) * Real(0.5), Real(0)};
            normal = Vector<Real, 3>{Real(1), Real(1), Real(0)};
            break;
        default:
            xi = Vector<Real, 3>{Real(0), (s + Real(1)) * Real(0.5), Real(0)};
            normal = Vector<Real, 3>{Real(-1), Real(0), Real(0)};
            break;
    }

    std::vector<Vector<Real, 3>> values;
    basis.evaluate_vector_values(xi, values);
    return values[static_cast<std::size_t>(func_id)].dot(normal);
}

static double integrate_quad_bdm_edge_flux(const BDMBasis& basis,
                                           int edge_id,
                                           int func_id,
                                           int quad_order) {
    using svmp::FE::quadrature::GaussQuadrature1D;

    GaussQuadrature1D quad(quad_order);
    double flux = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        flux += static_cast<double>(quad.weight(q) *
                                    evaluate_quad_bdm_edge_normal_flux(basis, edge_id, func_id, s));
    }
    return flux;
}

static double integrate_quad_bdm_edge_moment(const BDMBasis& basis,
                                             int edge_id,
                                             int mode_id,
                                             int func_id,
                                             int quad_order) {
    using svmp::FE::quadrature::GaussQuadrature1D;

    GaussQuadrature1D quad(quad_order);
    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real flux = evaluate_quad_bdm_edge_normal_flux(basis, edge_id, func_id, s);
        const Real basis_mode = (mode_id == 0) ? Real(1) : s;
        moment += static_cast<double>(quad.weight(q) * flux * basis_mode);
    }
    const double scale = (mode_id == 0) ? 0.5 : 1.5;
    return scale * moment;
}

static double integrate_triangle_bdm_edge_flux(const BDMBasis& basis,
                                               int edge_id,
                                               int func_id,
                                               int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    struct Edge {
        Vector<Real, 3> a;
        Vector<Real, 3> b;
        Vector<Real, 3> normal;
    };

    const Edge edges[3] = {
        {Vector<Real, 3>{Real(0), Real(0), Real(0)},
         Vector<Real, 3>{Real(1), Real(0), Real(0)},
         Vector<Real, 3>{Real(0), Real(-1), Real(0)}},
        {Vector<Real, 3>{Real(1), Real(0), Real(0)},
         Vector<Real, 3>{Real(0), Real(1), Real(0)},
         Vector<Real, 3>{Real(1), Real(1), Real(0)}},
        {Vector<Real, 3>{Real(0), Real(0), Real(0)},
         Vector<Real, 3>{Real(0), Real(1), Real(0)},
         Vector<Real, 3>{Real(-1), Real(0), Real(0)}},
    };

    const Edge& edge = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real, 3> tvec = edge.b - edge.a;
    const Real length = tvec.norm();

    GaussQuadrature1D quad(quad_order);
    double flux = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real t = (s + Real(1)) * Real(0.5);
        const Vector<Real, 3> xi = edge.a * (Real(1) - t) + edge.b * t;

        std::vector<Vector<Real, 3>> values;
        basis.evaluate_vector_values(xi, values);
        const Real normal_flux =
            values[static_cast<std::size_t>(func_id)].dot(edge.normal) / length;

        flux += static_cast<double>(quad.weight(q) * (Real(0.5) * length * normal_flux));
    }
    return flux;
}

static double integrate_triangle_bdm_edge_moment(const BDMBasis& basis,
                                                 int edge_id,
                                                 int mode_id,
                                                 int func_id,
                                                 int quad_order) {
    using svmp::FE::quadrature::GaussQuadrature1D;

    GaussQuadrature1D quad(quad_order);
    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real flux = evaluate_triangle_bdm_edge_scaled_normal_flux(basis, edge_id, func_id, s);
        const Real basis_mode = (mode_id == 0) ? Real(1) : s;
        moment += static_cast<double>(quad.weight(q) * flux * basis_mode);
    }
    return 0.5 * moment;
}

static double integrate_quad_bdm_divergence(const BDMBasis& basis,
                                            int func_id,
                                            int quad_order) {
    using svmp::FE::quadrature::QuadrilateralQuadrature;

    QuadrilateralQuadrature quad(quad_order, quad_order);
    double volume = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> divergence;
        basis.evaluate_divergence(quad.point(q), divergence);
        volume += static_cast<double>(quad.weight(q) * divergence[static_cast<std::size_t>(func_id)]);
    }
    return volume;
}

static double integrate_triangle_bdm_divergence(const BDMBasis& basis,
                                                int func_id,
                                                int quad_order) {
    using svmp::FE::quadrature::TriangleQuadrature;

    TriangleQuadrature quad(quad_order);
    double volume = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        std::vector<Real> divergence;
        basis.evaluate_divergence(quad.point(q), divergence);
        volume += static_cast<double>(quad.weight(q) * divergence[static_cast<std::size_t>(func_id)]);
    }
    return volume;
}

static double integrate_triangle_bdm_edge_moment_exact(const BDMBasis& basis,
                                                       int edge_id,
                                                       int mode_id,
                                                       int func_id,
                                                       int order,
                                                       int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Triangle3);
    const auto& edge_nodes = ref.edge_nodes(static_cast<std::size_t>(edge_id));
    const Vector<Real, 3> a = ReferenceNodeLayout::get_node_coords(ElementType::Triangle3, static_cast<std::size_t>(edge_nodes[0]));
    const Vector<Real, 3> b = ReferenceNodeLayout::get_node_coords(ElementType::Triangle3, static_cast<std::size_t>(edge_nodes[1]));

    const Vector<Real, 3> tvec = b - a;
    const Real len = tvec.norm();
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real, 3> t = tvec / len;
    const Vector<Real, 3> nrm{t[1], -t[0], Real(0)};
    const Real J = len * Real(0.5);

    LagrangeBasis line(ElementType::Line2, order);
    const auto quad = QuadratureFactory::create(
        ElementType::Line2, std::max(2 * order + 2, quad_order), QuadratureType::GaussLegendre, /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Real s = quad->point(q)[0];
        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real, 3> xi = a * (Real(1) - tpar) + b * tpar;

        std::vector<Real> line_values;
        line.evaluate_values(Vector<Real, 3>{s, Real(0), Real(0)}, line_values);

        std::vector<Vector<Real, 3>> values;
        basis.evaluate_vector_values(xi, values);
        moment += static_cast<double>(quad->weight(q) *
                                      (J * line_values[static_cast<std::size_t>(mode_id)] *
                                       values[static_cast<std::size_t>(func_id)].dot(nrm)));
    }
    return moment;
}

static double integrate_triangle_bdm_interior_moment(const BDMBasis& basis,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    if (order < 2) {
        return 0.0;
    }

    NedelecBasis interior_basis(ElementType::Triangle3, order - 2);
    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, std::max(2 * order + 2, quad_order), QuadratureType::GaussLegendre, /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Vector<Real, 3> xi = quad->point(q);
        std::vector<Vector<Real, 3>> values;
        std::vector<Vector<Real, 3>> test_values;
        basis.evaluate_vector_values(xi, values);
        interior_basis.evaluate_vector_values(xi, test_values);
        moment += static_cast<double>(quad->weight(q) *
                                      values[static_cast<std::size_t>(func_id)].dot(
                                          test_values[static_cast<std::size_t>(mode_id)]));
    }
    return moment;
}

static double integrate_tetra_bdm_face_moment(const BDMBasis& basis,
                                              int face_id,
                                              int mode_id,
                                              int func_id,
                                              int order,
                                              int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Tetra4);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));
    const Vector<Real, 3> a = ReferenceNodeLayout::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(face_nodes[0]));
    const Vector<Real, 3> b = ReferenceNodeLayout::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(face_nodes[1]));
    const Vector<Real, 3> c = ReferenceNodeLayout::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(face_nodes[2]));

    const Vector<Real, 3> e01 = b - a;
    const Vector<Real, 3> e02 = c - a;
    const Vector<Real, 3> cross = e01.cross(e02);
    const Real scale = cross.norm();
    if (scale <= Real(0)) {
        return 0.0;
    }
    const Vector<Real, 3> nrm = cross / scale;

    LagrangeBasis face_basis(ElementType::Triangle3, order);
    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, std::max(2 * order + 2, quad_order), QuadratureType::GaussLegendre, /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto uv = quad->point(q);
        const Real u = uv[0];
        const Real v = uv[1];
        const Vector<Real, 3> xi = a + e01 * u + e02 * v;

        std::vector<Real> basis_values;
        face_basis.evaluate_values(Vector<Real, 3>{u, v, Real(0)}, basis_values);

        std::vector<Vector<Real, 3>> values;
        basis.evaluate_vector_values(xi, values);
        moment += static_cast<double>(quad->weight(q) *
                                      (scale * basis_values[static_cast<std::size_t>(mode_id)] *
                                       values[static_cast<std::size_t>(func_id)].dot(nrm)));
    }
    return moment;
}

static double integrate_tetra_bdm_face_moment_with_vertex_permutation(
    const BDMBasis& basis,
    int face_id,
    const std::array<int, 3>& vertex_permutation,
    int mode_id,
    int func_id,
    int order,
    int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Tetra4);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));
    const Vector<Real, 3> a = ReferenceNodeLayout::get_node_coords(
        ElementType::Tetra4, static_cast<std::size_t>(face_nodes[static_cast<std::size_t>(vertex_permutation[0])]));
    const Vector<Real, 3> b = ReferenceNodeLayout::get_node_coords(
        ElementType::Tetra4, static_cast<std::size_t>(face_nodes[static_cast<std::size_t>(vertex_permutation[1])]));
    const Vector<Real, 3> c = ReferenceNodeLayout::get_node_coords(
        ElementType::Tetra4, static_cast<std::size_t>(face_nodes[static_cast<std::size_t>(vertex_permutation[2])]));

    const Vector<Real, 3> e01 = b - a;
    const Vector<Real, 3> e02 = c - a;
    const Vector<Real, 3> cross = e01.cross(e02);
    const Real scale = cross.norm();
    if (scale <= Real(0)) {
        return 0.0;
    }
    const Vector<Real, 3> nrm = cross / scale;

    LagrangeBasis face_basis(ElementType::Triangle3, order);
    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, std::max(2 * order + 2, quad_order), QuadratureType::GaussLegendre, /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto uv = quad->point(q);
        const Real u = uv[0];
        const Real v = uv[1];
        const Vector<Real, 3> xi = a + e01 * u + e02 * v;

        std::vector<Real> basis_values;
        face_basis.evaluate_values(Vector<Real, 3>{u, v, Real(0)}, basis_values);

        std::vector<Vector<Real, 3>> values;
        basis.evaluate_vector_values(xi, values);
        moment += static_cast<double>(quad->weight(q) *
                                      (scale * basis_values[static_cast<std::size_t>(mode_id)] *
                                       values[static_cast<std::size_t>(func_id)].dot(nrm)));
    }
    return moment;
}

static double integrate_tetra_bdm_interior_moment(const BDMBasis& basis,
                                                  int mode_id,
                                                  int func_id,
                                                  int order,
                                                  int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    if (order < 2) {
        return 0.0;
    }

    NedelecBasis interior_basis(ElementType::Tetra4, order - 2);
    const auto quad = QuadratureFactory::create(
        ElementType::Tetra4, std::max(2 * order + 2, quad_order), QuadratureType::GaussLegendre, /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Vector<Real, 3> xi = quad->point(q);
        std::vector<Vector<Real, 3>> values;
        std::vector<Vector<Real, 3>> test_values;
        basis.evaluate_vector_values(xi, values);
        interior_basis.evaluate_vector_values(xi, test_values);
        moment += static_cast<double>(quad->weight(q) *
                                      values[static_cast<std::size_t>(func_id)].dot(
                                          test_values[static_cast<std::size_t>(mode_id)]));
    }
    return moment;
}

TEST(RaviartThomasBasis, QuadEdgeFluxes) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    ASSERT_EQ(basis.size(), 4u);

    for (int edge = 0; edge < 4; ++edge) {
        for (int func = 0; func < 4; ++func) {
            const double flux = integrate_edge_normal_flux_quad4(basis, edge, func);
            if (edge == func) {
                EXPECT_NEAR(flux, 1.0, 1e-12);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-12);
            }
        }
    }
}

static double integrate_wedge_face_flux(const RaviartThomasBasis& basis,
                                        int face_id,
                                        int func_id,
                                        int tri_order = 4,
                                        int quad_order = 4) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    if (face_id == 0 || face_id == 1) {
        TriangleQuadrature tri(tri_order);
        const double z = (face_id == 0) ? -1.0 : 1.0;
        const double nz = (face_id == 0) ? -1.0 : 1.0;
        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            Vector<Real,3> xi{pt[0], pt[1], static_cast<Real>(z)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = static_cast<double>(v[2]) * nz;
            flux += static_cast<double>(tri.weight(q)) * integrand;
        }
        return flux;
    }

    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const double xi = pt[0];
        const double eta = pt[1];

        Vector<Real,3> x_ref{};
        Vector<Real,3> cross{};

        if (face_id == 2) { // y=0
            const double t = 0.5 * (xi + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{static_cast<Real>(t), Real(0), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0), Real(-0.5), Real(0)};
        } else if (face_id == 3) { // x=0
            const double t = 0.5 * (xi + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{Real(0), static_cast<Real>(t), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0.5), Real(0), Real(0)};
        } else { // face_id == 4, x+y=1
            const double t = 0.5 * (xi + 1.0);
            const double z = eta;
            const double x = t;
            const double y = 1.0 - t;
            x_ref = Vector<Real,3>{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(-0.5), Real(-0.5), Real(0)};
        }

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(x_ref, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(quad.weight(q)) * integrand;
    }

    return flux;
}

static double integrate_triangle_edge_tangential_moment(const NedelecBasis& basis,
                                                        int edge_id,
                                                        int mode_id,
                                                        int func_id,
                                                        int order,
                                                        int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[3] = {{v0, v1}, {v1, v2}, {v2, v0}};
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Real J = len * Real(0.5);

    LagrangeBasis line(ElementType::Line2, order);
    GaussQuadrature1D quad(quad_order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad.weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_triangle_edge_normal_moment(const RaviartThomasBasis& basis,
                                                    int edge_id,
                                                    int mode_id,
                                                    int func_id,
                                                    int order,
                                                    int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[3] = {{v0, v1}, {v1, v2}, {v2, v0}};
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Vector<Real,3> n{t[1], -t[0], Real(0)}; // outward for CCW edge ordering
    const Real J = len * Real(0.5);

    LagrangeBasis line(ElementType::Line2, order);
    GaussQuadrature1D quad(quad_order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(n);

        moment += static_cast<double>(quad.weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_tetra_edge_tangential_moment(const NedelecBasis& basis,
                                                     int edge_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};

    struct Edge { Vector<Real,3> a; Vector<Real,3> b; };
    const Edge edges[6] = {
        {v0, v1}, {v1, v2}, {v2, v0},
        {v0, v3}, {v1, v3}, {v2, v3}
    };
    const Edge& e = edges[static_cast<std::size_t>(edge_id)];

    const Vector<Real,3> tvec = e.b - e.a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real,3> t = tvec / len;
    const Real J = len * Real(0.5);

    const int q_order = std::max(2 * order + 2, quad_order);
    const auto quad = QuadratureFactory::create(
        ElementType::Line2, q_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    LagrangeBasis line(ElementType::Line2, order);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Real s = quad->point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real l = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = e.a * (Real(1) - tpar) + e.b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad->weight(q) * (J * l * dot));
    }
    return moment;
}

static double integrate_tetra_face_tangential_moment(const NedelecBasis& basis,
                                                     int face_id,
                                                     int dir_id,
                                                     int mode_id,
                                                     int func_id,
                                                     int order,
                                                     int quad_order) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};

    struct Face { Vector<Real,3> a; Vector<Real,3> b; Vector<Real,3> c; };
    const Face faces[4] = {
        {v0, v1, v2},
        {v0, v1, v3},
        {v1, v2, v3},
        {v0, v2, v3}
    };
    const Face& f = faces[static_cast<std::size_t>(face_id)];

    const Vector<Real,3> tu = f.b - f.a;
    const Vector<Real,3> tv = f.c - f.a;

    Vector<Real,3> cross{};
    cross[0] = tu[1] * tv[2] - tu[2] * tv[1];
    cross[1] = tu[2] * tv[0] - tu[0] * tv[2];
    cross[2] = tu[0] * tv[1] - tu[1] * tv[0];
    const Real scale = cross.norm();
    EXPECT_GT(scale, Real(0));
    if (scale <= Real(0)) {
        return 0.0;
    }

    const int q_order = std::max(2 * order + 2, quad_order);
    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, q_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    LagrangeBasis face_basis(ElementType::Triangle3, order - 1);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto pt = quad->point(q);
        const Real u = pt[0];
        const Real v = pt[1];

        std::vector<Real> bvals;
        face_basis.evaluate_values(Vector<Real,3>{u, v, Real(0)}, bvals);
        const Real phi = bvals[static_cast<std::size_t>(mode_id)];

        const Vector<Real,3> xi = f.a + tu * u + tv * v;
        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const Vector<Real,3>& t = (dir_id == 0) ? tu : tv;
        const Real dot = w.dot(t);

        moment += static_cast<double>(quad->weight(q) * (scale * phi * dot));
    }
    return moment;
}

static double integrate_wedge_edge_dof(const NedelecBasis& basis,
                                       int edge_id,
                                       int func_id,
                                       int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    // Reference wedge nodes (matching NodeOrderingConventions and RT0 wedge tests)
    const Vector<Real,3> v0{Real(0), Real(0), Real(-1)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(-1)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(-1)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};
    const Vector<Real,3> v4{Real(1), Real(0), Real(1)};
    const Vector<Real,3> v5{Real(0), Real(1), Real(1)};

    struct Edge {
        Vector<Real,3> a;
        Vector<Real,3> b;
    };
    const Edge edges[9] = {
        {v0, v1}, // bottom
        {v1, v2},
        {v2, v0},
        {v3, v4}, // top
        {v4, v5},
        {v5, v3},
        {v0, v3}, // vertical
        {v1, v4},
        {v2, v5}
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0]; // in [-1,1]
        const Real s = (tq + Real(1)) * Real(0.5); // map to [0,1]
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const Real wx = w[0];
        const Real wy = w[1];
        const Real wz = w[2];
        const Real tx = t_vec[0];
        const Real ty = t_vec[1];
        const Real tz = t_vec[2];

        const double integrand = static_cast<double>(wx*tx + wy*ty + wz*tz);
        const double ds = 0.5 * quad.weight(q);
        dof += ds * integrand;
    }
    return dof;
}

static double integrate_pyramid_edge_dof(const NedelecBasis& basis,
                                         int edge_id,
                                         int func_id,
                                         int quad_order) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    // Reference Pyramid5 vertices (matching NodeOrderingConventions)
    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    struct Edge {
        Vector<Real,3> a;
        Vector<Real,3> b;
    };

    const Edge edges[8] = {
        {v0, v1}, // base edges
        {v1, v2},
        {v2, v3},
        {v3, v0},
        {v0, v4}, // vertical edges
        {v1, v4},
        {v2, v4},
        {v3, v4}
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0];   // [-1,1]
        const Real s  = (tq + Real(1)) * Real(0.5); // [0,1]
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const Real wx = w[0];
        const Real wy = w[1];
        const Real wz = w[2];
        const Real tx = t_vec[0];
        const Real ty = t_vec[1];
        const Real tz = t_vec[2];

        const double integrand = static_cast<double>(wx*tx + wy*ty + wz*tz);
        const double dt = 0.5 * quad.weight(q);
        dof += dt * integrand;
    }

    return dof;
}

TEST(RaviartThomasBasis, WedgeMinimalFaceFluxDofs) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    constexpr int n_faces = 5;
    constexpr int n_funcs = 5;

    for (int i = 0; i < n_funcs; ++i) {
        for (int j = 0; j < n_faces; ++j) {
            double flux = integrate_wedge_face_flux(basis, j, i);
            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}


static double integrate_pyramid_face_flux(const RaviartThomasBasis& basis,
                                          int face_id,
                                          int func_id,
                                          int tri_order = 4,
                                          int quad_order = 4) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    // Reference pyramid vertices
    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    if (face_id == 0) {
        QuadrilateralQuadrature quad(quad_order, quad_order);
        for (std::size_t q = 0; q < quad.num_points(); ++q) {
            const auto& pt = quad.point(q);
            Vector<Real,3> xi{pt[0], pt[1], Real(0)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = -static_cast<double>(v[2]); // outward -z
            flux += static_cast<double>(quad.weight(q)) * integrand;
        }
        return flux;
    }

    TriangleQuadrature tri(tri_order);
    const Vector<Real,3>* a = nullptr;
    const Vector<Real,3>* b = nullptr;
    const Vector<Real,3>* c = nullptr;
    switch (face_id) {
        case 1: a = &v0; b = &v1; c = &v4; break;
        case 2: a = &v1; b = &v2; c = &v4; break;
        case 3: a = &v2; b = &v3; c = &v4; break;
        default: a = &v3; b = &v0; c = &v4; break;
    }
    Vector<Real,3> e1 = *b - *a;
    Vector<Real,3> e2 = *c - *a;
    Vector<Real,3> cross = e1.cross(e2);

    for (std::size_t q = 0; q < tri.num_points(); ++q) {
        const auto& pt = tri.point(q);
        const double r = pt[0];
        const double s = pt[1];
        const double l0 = 1.0 - r - s;
        const double l1 = r;
        const double l2 = s;
        Vector<Real,3> xi =
            (*a) * static_cast<Real>(l0) +
            (*b) * static_cast<Real>(l1) +
            (*c) * static_cast<Real>(l2);

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(tri.weight(q)) * integrand;
    }

    return flux;
}

TEST(RaviartThomasBasis, PyramidMinimalFaceFluxDofs) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    constexpr int n_faces = 5;
    constexpr int n_funcs = 5;

    for (int i = 0; i < n_funcs; ++i) {
        for (int j = 0; j < n_faces; ++j) {
            double flux = integrate_pyramid_face_flux(basis, j, i);
            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}

// =============================================================================
// Higher-order Wedge/Pyramid Vector Element Tests
// =============================================================================

TEST(RaviartThomasBasis, WedgeHigherOrderConstruction) {
    // Test that RT(k) for wedge can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 24u);  // RT(1) on wedge: 18 face + 6 interior
    });
}

TEST(RaviartThomasBasis, PyramidHigherOrderConstruction) {
    // Test that RT(k) for pyramid can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 19u);  // RT(1) on pyramid: 16 face + 3 interior
    });
}

TEST(RaviartThomasBasis, WedgeRT1Size) {
    // Wedge RT(k) DOF counts
    // RT(0): 5 faces (2 tri + 3 quad with 1 DOF each)
    RaviartThomasBasis rt0(ElementType::Wedge6, 0);
    EXPECT_EQ(rt0.size(), 5u);

    // RT(1): 24 DOFs (18 face + 6 interior)
    RaviartThomasBasis rt1(ElementType::Wedge6, 1);
    EXPECT_EQ(rt1.size(), 24u);
}

TEST(RaviartThomasBasis, PyramidRT1Size) {
    // Pyramid RT(k) DOF counts
    // RT(0): 5 faces (1 quad + 4 tri)
    RaviartThomasBasis rt0(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt0.size(), 5u);

    // RT(1): 19 DOFs (16 face + 3 interior)
    RaviartThomasBasis rt1(ElementType::Pyramid5, 1);
    EXPECT_EQ(rt1.size(), 19u);
}

TEST(RaviartThomasBasis, WedgeRT1Evaluation) {
    // Test that RT(1) on wedge can be evaluated
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 24u);

    // Test divergence evaluation
    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 24u);
}

TEST(RaviartThomasBasis, PyramidRT1Evaluation) {
    // Test that RT(1) on pyramid can be evaluated
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    // Evaluate at a test point (inside pyramid)
    svmp::FE::math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 19u);

    // Test divergence evaluation
    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 19u);
}

TEST(NedelecBasis, WedgeHigherOrderConstruction) {
    // Test that Nedelec(k) for wedge can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 9u);  // 9 edges
    });

    // k=1 now works with direct DOF-based construction
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        EXPECT_EQ(basis.size(), 34u);  // 18 edge + 16 face
    });
}

TEST(NedelecBasis, PyramidHigherOrderConstruction) {
    // Test that Nedelec(k) for pyramid can be constructed
    // k=0 works with analytical formulas
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 8u);  // 8 edges
    });

    // k=1 construction and evaluation are both implemented
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 1);
        // Evaluation coverage is exercised in the dedicated higher-order tests.
    });
}

TEST(NedelecBasis, WedgeND1Size) {
    // Wedge Nedelec(k) DOF counts
    // ND(0): 9 edges
    NedelecBasis nd0(ElementType::Wedge6, 0);
    EXPECT_EQ(nd0.size(), 9u);

    // ND(1): 34 DOFs (18 edge + 16 face)
    NedelecBasis nd1(ElementType::Wedge6, 1);
    EXPECT_EQ(nd1.size(), 34u);
}

TEST(NedelecBasis, PyramidND1Size) {
    // Pyramid Nedelec(k) DOF counts
    // ND(0): 8 edges
    NedelecBasis nd0(ElementType::Pyramid5, 0);
    EXPECT_EQ(nd0.size(), 8u);

    // ND(1) construction and evaluation are both implemented
    NedelecBasis nd1(ElementType::Pyramid5, 1);
    EXPECT_EQ(nd1.size(), 28u);  // 16 edge + 12 face
}

TEST(NedelecBasis, WedgeND1Evaluation) {
    // Test that Nedelec(1) on wedge can be evaluated
    NedelecBasis basis(ElementType::Wedge6, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 34u);

    // Test curl evaluation (via finite differences)
    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 34u);
}

TEST(NedelecBasis, PyramidND1Evaluation) {
    // Test that Nedelec(1) on pyramid can be evaluated (direct DOF construction)
    NedelecBasis basis(ElementType::Pyramid5, 1);

    // Evaluate at a test point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};
    std::vector<svmp::FE::math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 28u);  // Pyramid ND(1) has 28 DOFs

    // Test curl evaluation (via finite differences)
    std::vector<svmp::FE::math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 28u);
}

// =============================================================================
// Dimension Verification Tests
// =============================================================================
// The dimension formulas for higher-order wedge/pyramid RT and Nedelec elements.
// NOTE: Higher-order (k>=1) tests now enabled via direct DOF construction.
// See test_HigherOrderWedgePyramid.cpp for comprehensive k=0,1,2 coverage.

TEST(RaviartThomasBasis, WedgeDimensionFormulas) {
    // Wedge RT(k) dimensions (theoretical):
    // Face DOFs: 2*(k+1)(k+2)/2 + 3*(k+1)^2 = (k+1)(k+2) + 3(k+1)^2
    // Interior DOFs: 3*k*(k+1)^2/2 for k >= 1
    // RT(0): 5 (5 faces, no interior)
    // RT(1): 18 face + 6 interior = 24
    // RT(2): 39 face + 24 interior = 63

    RaviartThomasBasis rt0(ElementType::Wedge6, 0);
    EXPECT_EQ(rt0.size(), 5u);
    // Higher-order construction/evaluation coverage lives in test_HigherOrderWedgePyramid.cpp.
}

TEST(RaviartThomasBasis, PyramidDimensionFormulas) {
    // Pyramid RT(k) dimensions (theoretical):
    // Face DOFs: (k+1)^2 + 4*(k+1)(k+2)/2 = (k+1)^2 + 2(k+1)(k+2)
    // Interior DOFs: 3*k^3 for k >= 1
    // RT(0): 5 (5 faces, no interior)
    // RT(1): 16 face + 3 interior = 19
    // RT(2): 33 face + 24 interior = 57

    RaviartThomasBasis rt0(ElementType::Pyramid5, 0);
    EXPECT_EQ(rt0.size(), 5u);
    // Higher-order construction/evaluation coverage lives in test_HigherOrderWedgePyramid.cpp.
}

TEST(NedelecBasis, WedgeDimensionFormulas) {
    // Wedge Nedelec(k) dimensions (theoretical):
    // Edge DOFs: 9*(k+1)
    // Face DOFs: 2*k(k+1) + 3*2k(k+1) = 8k(k+1) for k >= 1
    // Interior DOFs: 3*k*(k-1)*(k+1)/2 for k >= 2
    // ND(0): 9 edges
    // ND(1): 18 edge + 16 face + 0 interior = 34
    // ND(2): 27 edge + 48 face + 9 interior = 84

    NedelecBasis nd0(ElementType::Wedge6, 0);
    EXPECT_EQ(nd0.size(), 9u);
    // Higher-order construction/evaluation coverage lives in test_HigherOrderWedgePyramid.cpp.
}

TEST(NedelecBasis, PyramidDimensionFormulas) {
    // Pyramid Nedelec(k) dimensions (theoretical):
    // Edge DOFs: 8*(k+1)
    // Face DOFs: 2k(k+1) + 4*k(k+1) = 6k(k+1) for k >= 1
    // Interior DOFs: 3*k*(k-1)*(k+1)/6 for k >= 2
    // ND(0): 8 edges
    // ND(1): 16 edge + 12 face + 0 interior = 28
    // ND(2): 24 edge + 36 face + 3 interior = 63

    NedelecBasis nd0(ElementType::Pyramid5, 0);
    EXPECT_EQ(nd0.size(), 8u);
    // Higher-order construction/evaluation coverage lives in test_HigherOrderWedgePyramid.cpp.
}

// =============================================================================
// DOF Association Tests
// =============================================================================

TEST(RaviartThomasBasis, DofAssociationsTriangle) {
    RaviartThomasBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For RT(1) on triangle: 3 edges x 2 DOFs per edge + interior
    int edge_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        } else if (a.entity_type == DofEntity::Interior) {
            ++interior_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 6); // 3 edges x 2 DOFs
    EXPECT_GE(interior_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsTriangle) {
    NedelecBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For Nedelec(1) on triangle: 3 edges x 2 DOFs per edge + interior
    int edge_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 6); // 3 edges x 2 DOFs
}

TEST(RaviartThomasBasis, DofAssociationsTetra) {
    RaviartThomasBasis basis(ElementType::Tetra4, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For RT(1) on tetrahedron: 4 faces with face DOFs
    int face_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) {
            ++face_dofs;
        }
    }
    EXPECT_GT(face_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsTetra) {
    NedelecBasis basis(ElementType::Tetra4, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());

    // For Nedelec(1) on tetrahedron: 6 edges with edge DOFs + face DOFs
    int edge_dofs = 0;
    int face_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) {
            ++edge_dofs;
        } else if (a.entity_type == DofEntity::Face) {
            ++face_dofs;
        }
    }
    EXPECT_EQ(edge_dofs, 12); // 6 edges x 2 DOFs
    EXPECT_GT(face_dofs, 0);
}

TEST(BDMBasis, DofAssociationsTriangle) {
    BDMBasis basis(ElementType::Triangle3, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 6u); // BDM1 on triangle has 6 DOFs

    // All should be edge DOFs for BDM1
    for (const auto& a : assoc) {
        EXPECT_EQ(a.entity_type, DofEntity::Edge);
    }
}

TEST(RaviartThomasBasis, DofAssociationsWedge) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 24u);

    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(face_dofs, 18);
    EXPECT_EQ(interior_dofs, 6);
}

TEST(RaviartThomasBasis, DofAssociationsWedgeOrderTwo) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 66u);

    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(face_dofs, 39);
    EXPECT_EQ(interior_dofs, 27);
}

TEST(RaviartThomasBasis, DofAssociationsPyramid) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 19u);

    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(face_dofs, 16);
    EXPECT_EQ(interior_dofs, 3);
}

TEST(RaviartThomasBasis, DofAssociationsPyramidOrderTwo) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 57u);

    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(face_dofs, 33);
    EXPECT_EQ(interior_dofs, 24);
}

TEST(NedelecBasis, DofAssociationsPyramid) {
    NedelecBasis basis(ElementType::Pyramid5, 1);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 28u);

    int edge_dofs = 0;
    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) edge_dofs++;
        else if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(edge_dofs, 16);
    EXPECT_EQ(face_dofs, 12);
    EXPECT_EQ(interior_dofs, 0);
}

TEST(NedelecBasis, DofAssociationsPyramidOrderTwo) {
    NedelecBasis basis(ElementType::Pyramid5, 2);
    auto assoc = basis.dof_associations();
    EXPECT_EQ(assoc.size(), basis.size());
    EXPECT_EQ(assoc.size(), 63u);

    int edge_dofs = 0;
    int face_dofs = 0;
    int interior_dofs = 0;
    for (const auto& a : assoc) {
        if (a.entity_type == DofEntity::Edge) edge_dofs++;
        else if (a.entity_type == DofEntity::Face) face_dofs++;
        else if (a.entity_type == DofEntity::Interior) interior_dofs++;
    }
    EXPECT_EQ(edge_dofs, 24);
    EXPECT_EQ(face_dofs, 36);
    EXPECT_EQ(interior_dofs, 3);
}
