/**
 * @file test_VectorSpectralAndComposite.cpp
 * @brief Tests for VectorElement, SpectralElement, IsogeometricElement, CompositeElement
 */

#include <gtest/gtest.h>

#include "FE/Elements/VectorElement.h"
#include "FE/Elements/SpectralElement.h"
#include "FE/Elements/IsogeometricElement.h"
#include "FE/Elements/CompositeElement.h"
#include "FE/Elements/MixedElement.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Elements/ElementTransform.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

TEST(VectorElement, HdivOnQuad4) {
    VectorElement elem(ElementType::Quad4, 0, Continuity::H_div);

    EXPECT_EQ(elem.field_type(), FieldType::Vector);
    EXPECT_EQ(elem.continuity(), Continuity::H_div);
    EXPECT_EQ(elem.dimension(), 2);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    EXPECT_TRUE(basis_ptr->is_vector_valued());
    EXPECT_EQ(basis_ptr->size(), 4u); // RT0 on Quad4

    EXPECT_EQ(elem.num_dofs(), basis_ptr->size());

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);
    EXPECT_GT(quad->num_points(), 0u);
}

TEST(VectorElement, HcurlOnTetra4) {
    VectorElement elem(ElementType::Tetra4, 0, Continuity::H_curl);
    EXPECT_EQ(elem.field_type(), FieldType::Vector);
    EXPECT_EQ(elem.continuity(), Continuity::H_curl);
    EXPECT_EQ(elem.dimension(), 3);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    EXPECT_TRUE(basis_ptr->is_vector_valued());
    EXPECT_EQ(elem.num_dofs(), basis_ptr->size());
}

TEST(SpectralElement, Line2Metadata) {
    SpectralElement elem(ElementType::Line2, 3);
    EXPECT_EQ(elem.element_type(), ElementType::Line2);
    EXPECT_EQ(elem.field_type(), FieldType::Scalar);
    EXPECT_EQ(elem.dimension(), 1);
    EXPECT_GE(elem.polynomial_order(), 1);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    EXPECT_EQ(basis_ptr->basis_type(), BasisType::Spectral);
    EXPECT_EQ(basis_ptr->size(), static_cast<std::size_t>(elem.polynomial_order() + 1));

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);
    EXPECT_GT(quad->num_points(), 0u);
}

TEST(IsogeometricElement, WrapsExternalBasisAndQuadrature) {
    // Use a simple Lagrange basis as a stand-in for a NURBS basis
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    svmp::FE::quadrature::GaussQuadrature1D quad_1d(2);
    auto quad = std::make_shared<svmp::FE::quadrature::QuadrilateralQuadrature>(2);

    IsogeometricElement elem(basis, quad, FieldType::Scalar, Continuity::C0);

    EXPECT_EQ(elem.element_type(), ElementType::Quad4);
    EXPECT_EQ(elem.dimension(), 2);
    EXPECT_EQ(elem.num_nodes(), basis->size());
    EXPECT_EQ(elem.num_dofs(), basis->size());
}

TEST(CompositeElement, AggregatesComponents) {
    auto e1 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    auto e2 = std::make_shared<LagrangeElement>(ElementType::Triangle3, 2);

    CompositeElement comp({e1, e2});

    EXPECT_EQ(comp.element_type(), ElementType::Triangle3);
    EXPECT_EQ(comp.dimension(), 2);
    EXPECT_EQ(comp.field_type(), FieldType::Mixed);
    EXPECT_EQ(comp.components().size(), 2u);

    const std::size_t expected_dofs = e1->num_dofs() + e2->num_dofs();
    EXPECT_EQ(comp.num_dofs(), expected_dofs);
}

TEST(ElementTransform, HdivAndHcurlIdentityOnHex) {
    // Identity geometry on Hex8 via isoparametric mapping
    auto geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto geom_nodes = geom_basis->nodes();
    svmp::FE::geometry::IsoparametricMapping mapping(geom_basis, geom_nodes);

    // H(div) and H(curl) bases on Hex8
    basis::RaviartThomasBasis rt(ElementType::Hex8, 0);
    basis::NedelecBasis        ned(ElementType::Hex8, 0);

    svmp::FE::math::Vector<Real,3> xi{Real(0.2), Real(-0.1), Real(0.3)};

    // H(div) Piola: with identity mapping, should be identity transform
    std::vector<svmp::FE::math::Vector<Real,3>> vals_rt, vals_rt_phys;
    rt.evaluate_vector_values(xi, vals_rt);
    elements::ElementTransform::hdiv_vectors_to_physical(mapping, xi, vals_rt, vals_rt_phys);
    ASSERT_EQ(vals_rt.size(), vals_rt_phys.size());
    for (std::size_t i = 0; i < vals_rt.size(); ++i) {
        EXPECT_NEAR(vals_rt[i][0], vals_rt_phys[i][0], 1e-12);
        EXPECT_NEAR(vals_rt[i][1], vals_rt_phys[i][1], 1e-12);
        EXPECT_NEAR(vals_rt[i][2], vals_rt_phys[i][2], 1e-12);
    }

    // H(curl) Piola: with identity mapping, should also be identity transform
    std::vector<svmp::FE::math::Vector<Real,3>> vals_ned, vals_ned_phys;
    ned.evaluate_vector_values(xi, vals_ned);
    elements::ElementTransform::hcurl_vectors_to_physical(mapping, xi, vals_ned, vals_ned_phys);
    ASSERT_EQ(vals_ned.size(), vals_ned_phys.size());
    for (std::size_t i = 0; i < vals_ned.size(); ++i) {
        EXPECT_NEAR(vals_ned[i][0], vals_ned_phys[i][0], 1e-12);
        EXPECT_NEAR(vals_ned[i][1], vals_ned_phys[i][1], 1e-12);
        EXPECT_NEAR(vals_ned[i][2], vals_ned_phys[i][2], 1e-12);
    }
}

TEST(MixedElementTests, ThrowsOnIncompatibleSubElements) {
    auto tri = std::make_shared<LagrangeElement>(ElementType::Triangle3, 1);
    auto tet = std::make_shared<LagrangeElement>(ElementType::Tetra4, 1);

    // Different dimensions / element types should trigger an error
    elements::MixedSubElement se1{tri, 0};
    elements::MixedSubElement se2{tet, 1};
    std::vector<elements::MixedSubElement> subs{se1, se2};

    EXPECT_THROW(elements::MixedElement bad(subs), svmp::FE::FEException);
}
