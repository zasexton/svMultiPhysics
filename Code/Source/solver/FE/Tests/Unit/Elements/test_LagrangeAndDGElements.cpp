/**
 * @file test_LagrangeAndDGElements.cpp
 * @brief Tests for LagrangeElement and DiscontinuousElement
 */

#include <gtest/gtest.h>

#include "FE/Elements/LagrangeElement.h"
#include "FE/Elements/DiscontinuousElement.h"
#include "FE/Elements/ElementTransform.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Quadrature/PyramidQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

namespace {

template<typename BasisType, typename QuadType, typename MappingType>
std::vector<Real> compute_laplacian_stiffness_direct(
    const BasisType& basis,
    const QuadType& quad,
    const MappingType& mapping) {

    const std::size_t n = basis.size();
    std::vector<Real> K(n * n, Real(0));

    std::vector<basis::Gradient> grads_ref(n);

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& qp = quad.point(q);
        const Real w = quad.weight(q);

        basis.evaluate_gradients(qp, grads_ref);

        std::vector<math::Vector<Real,3>> grads_phys;
        ElementTransform::gradients_to_physical(mapping, qp, grads_ref, grads_phys);

        const Real detJ = std::abs(mapping.jacobian_determinant(qp));
        const int dim = mapping.dimension();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Real dot = Real(0);
                for (int d = 0; d < dim; ++d) {
                    dot += grads_phys[i][static_cast<std::size_t>(d)] *
                           grads_phys[j][static_cast<std::size_t>(d)];
                }
                K[i * n + j] += w * detJ * dot;
            }
        }
    }

    return K;
}

std::vector<Real> compute_laplacian_stiffness_element(
    const LagrangeElement& elem,
    const geometry::GeometryMapping& mapping) {

    auto quad = elem.quadrature();
    const auto& basis_fn = elem.basis();

    const std::size_t n = basis_fn.size();
    std::vector<Real> K(n * n, Real(0));

    std::vector<basis::Gradient> grads_ref(n);

    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto& qp = quad->point(q);
        const Real w = quad->weight(q);

        basis_fn.evaluate_gradients(qp, grads_ref);

        std::vector<math::Vector<Real,3>> grads_phys;
        ElementTransform::gradients_to_physical(mapping, qp, grads_ref, grads_phys);

        const Real detJ = std::abs(mapping.jacobian_determinant(qp));
        const int dim = mapping.dimension();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Real dot = Real(0);
                for (int d = 0; d < dim; ++d) {
                    dot += grads_phys[i][static_cast<std::size_t>(d)] *
                           grads_phys[j][static_cast<std::size_t>(d)];
                }
                K[i * n + j] += w * detJ * dot;
            }
        }
    }

    return K;
}

} // namespace

TEST(LagrangeElement, Quad4MetadataAndBasis) {
    LagrangeElement elem(ElementType::Quad4, 1);

    EXPECT_EQ(elem.element_type(), ElementType::Quad4);
    EXPECT_EQ(elem.field_type(), FieldType::Scalar);
    EXPECT_EQ(elem.continuity(), Continuity::C0);
    EXPECT_EQ(elem.polynomial_order(), 1);
    EXPECT_EQ(elem.dimension(), 2);

    EXPECT_EQ(elem.num_nodes(), 4u);
    EXPECT_EQ(elem.num_dofs(), 4u);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis_ptr->size(), 4u);

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);
    EXPECT_GT(quad->num_points(), 0u);
}

TEST(DiscontinuousElement, UsesL2Continuity) {
    DiscontinuousElement elem(ElementType::Quad4, 1);
    EXPECT_EQ(elem.continuity(), Continuity::L2);
    EXPECT_EQ(elem.num_nodes(), 4u);
    EXPECT_EQ(elem.num_dofs(), 4u);
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyQuad4) {
    // Reference Quad4 element on [-1,1]^2
    LagrangeElement elem(ElementType::Quad4, 1);

    // Build an explicit Lagrange basis and use it to define geometry
    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis_ptr->nodes();
    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    // Direct assembly using basis + explicit quadrature
    quadrature::QuadrilateralQuadrature quad_direct(2);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    // Element-based assembly using the element's quadrature
    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyQuad4SkewedMapping) {
    LagrangeElement elem(ElementType::Quad4, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis_ptr->nodes();

    // Apply an affine shear transform in physical space to exercise
    // non-orthogonal Jacobians while keeping the mapping affine.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        n[0] = Real(1.7) * x + Real(0.3) * y;
        n[1] = Real(0.2) * x + Real(1.1) * y;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::QuadrilateralQuadrature quad_direct(6);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyTetra4) {
    LagrangeElement elem(ElementType::Tetra4, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 1);
    auto nodes = basis_ptr->nodes();

    // Stretch the reference tetrahedron to exercise non-identity Jacobians.
    for (auto& n : nodes) {
        n[0] *= Real(2);
        n[1] *= Real(3);
        n[2] *= Real(4);
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    // Higher-order direct quadrature (element default is order 2 for p=1)
    quadrature::TetrahedronQuadrature quad_direct(4);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyHex8) {
    LagrangeElement elem(ElementType::Hex8, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis_ptr->nodes();

    // Anisotropic scaling to exercise Jacobian transforms.
    for (auto& n : nodes) {
        n[0] *= Real(2);
        n[1] *= Real(0.5);
        n[2] *= Real(1.5);
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::HexahedronQuadrature quad_direct(4);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyHex8SkewedMapping) {
    LagrangeElement elem(ElementType::Hex8, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis_ptr->nodes();

    // Apply a 3x3 affine transform with off-diagonal terms.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        const Real z = n[2];
        n[0] = Real(1.3) * x + Real(0.15) * y + Real(0.1) * z;
        n[1] = Real(0.05) * x + Real(1.1) * y + Real(0.2) * z;
        n[2] = Real(0.1) * x + Real(0.05) * y + Real(0.9) * z;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::HexahedronQuadrature quad_direct(6);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyQuad9Order2) {
    LagrangeElement elem(ElementType::Quad9, 2);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Quad9, 2);
    auto nodes = basis_ptr->nodes();
    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    // Use a higher-order direct quadrature than the element default to ensure
    // the element-selected rule is not under-integrating.
    quadrature::QuadrilateralQuadrature quad_direct(6);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyTriangle6Order2) {
    LagrangeElement elem(ElementType::Triangle6, 2);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle6, 2);
    auto nodes = basis_ptr->nodes();

    // Affine transform in 2D to exercise Jacobians.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        n[0] = Real(2.0) * x + Real(0.25) * y;
        n[1] = Real(0.1) * x + Real(1.6) * y;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::TriangleQuadrature quad_direct(8);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyTetra10Order2) {
    LagrangeElement elem(ElementType::Tetra10, 2);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra10, 2);
    auto nodes = basis_ptr->nodes();

    // Affine transform in 3D.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        const Real z = n[2];
        n[0] = Real(1.8) * x + Real(0.2) * y + Real(0.1) * z;
        n[1] = Real(0.1) * x + Real(2.1) * y + Real(0.15) * z;
        n[2] = Real(0.05) * x + Real(0.1) * y + Real(1.6) * z;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::TetrahedronQuadrature quad_direct(8);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyHex27Order2) {
    LagrangeElement elem(ElementType::Hex27, 2);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Hex27, 2);
    auto nodes = basis_ptr->nodes();

    // Mild affine transform in 3D to avoid degeneracy.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        const Real z = n[2];
        n[0] = Real(1.4) * x + Real(0.1) * y;
        n[1] = Real(0.05) * x + Real(1.2) * y + Real(0.1) * z;
        n[2] = Real(0.08) * y + Real(1.1) * z;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::HexahedronQuadrature quad_direct(8);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyWedge6) {
    LagrangeElement elem(ElementType::Wedge6, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    auto nodes = basis_ptr->nodes();

    // Affine transform in 3D.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        const Real z = n[2];
        n[0] = Real(1.6) * x + Real(0.15) * y + Real(0.05) * z;
        n[1] = Real(0.1) * x + Real(1.3) * y + Real(0.1) * z;
        n[2] = Real(0.2) * x + Real(0.1) * y + Real(1.2) * z;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::WedgeQuadrature quad_direct(8);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-10);
    }
}

TEST(LagrangeElement, StiffnessMatchesDirectAssemblyPyramid5) {
    LagrangeElement elem(ElementType::Pyramid5, 1);

    auto basis_ptr = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
    auto nodes = basis_ptr->nodes();

    // Mild affine transform to avoid relying on identity mapping.
    for (auto& n : nodes) {
        const Real x = n[0];
        const Real y = n[1];
        const Real z = n[2];
        n[0] = Real(1.2) * x + Real(0.1) * y;
        n[1] = Real(0.05) * x + Real(1.1) * y;
        n[2] = Real(0.9) * z;
    }

    geometry::IsoparametricMapping mapping(basis_ptr, nodes);

    quadrature::PyramidQuadrature quad_direct(10);
    std::vector<Real> K_direct =
        compute_laplacian_stiffness_direct(*basis_ptr, quad_direct, mapping);

    std::vector<Real> K_elem = compute_laplacian_stiffness_element(elem, mapping);

    ASSERT_EQ(K_direct.size(), K_elem.size());
    for (std::size_t i = 0; i < K_direct.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(K_direct[i]),
                    static_cast<double>(K_elem[i]),
                    1e-9);
    }
}
