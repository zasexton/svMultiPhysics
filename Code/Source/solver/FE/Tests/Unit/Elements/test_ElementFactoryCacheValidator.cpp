/**
 * @file test_ElementFactoryCacheValidator.cpp
 * @brief Tests for ElementFactory, ElementCache, and ElementValidator
 */

#include <gtest/gtest.h>

#include "FE/Elements/ElementFactory.h"
#include "FE/Elements/ElementCache.h"
#include "FE/Elements/ElementValidator.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

TEST(ElementFactory, CreatesLagrangeAndDGAndVector) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.basis_type   = BasisType::Lagrange;
    req.field_type   = FieldType::Scalar;
    req.order        = 1;
    req.continuity   = Continuity::C0;

    auto lag = ElementFactory::create(req);
    ASSERT_TRUE(lag);
    EXPECT_EQ(lag->element_type(), ElementType::Quad4);
    EXPECT_EQ(lag->field_type(), FieldType::Scalar);
    EXPECT_EQ(lag->continuity(), Continuity::C0);

    req.continuity = Continuity::L2;
    auto dg = ElementFactory::create(req);
    ASSERT_TRUE(dg);
    EXPECT_EQ(dg->continuity(), Continuity::L2);

    req.field_type = FieldType::Vector;
    req.continuity = Continuity::H_div;
    auto hdiv = ElementFactory::create(req);
    ASSERT_TRUE(hdiv);
    EXPECT_EQ(hdiv->field_type(), FieldType::Vector);
    EXPECT_EQ(hdiv->continuity(), Continuity::H_div);
}

TEST(ElementFactory, CreatesVectorElementsFromExplicitVectorBasisTypes) {
    ElementRequest req;
    req.element_type = ElementType::Quad4;
    req.field_type   = FieldType::Vector;

    // H(curl) via explicit BasisType::Nedelec (continuity inferred)
    req.basis_type = BasisType::Nedelec;
    req.order      = 0;
    req.continuity = Continuity::C0; // allow inference
    auto hcurl = ElementFactory::create(req);
    ASSERT_TRUE(hcurl);
    EXPECT_EQ(hcurl->field_type(), FieldType::Vector);
    EXPECT_EQ(hcurl->continuity(), Continuity::H_curl);

    // H(div) via explicit BasisType::RaviartThomas (continuity inferred)
    req.basis_type = BasisType::RaviartThomas;
    req.order      = 0;
    req.continuity = Continuity::C0;
    auto rt = ElementFactory::create(req);
    ASSERT_TRUE(rt);
    EXPECT_EQ(rt->continuity(), Continuity::H_div);

    // H(div) via explicit BasisType::BDM (order 1 only)
    req.basis_type = BasisType::BDM;
    req.order      = 1;
    req.continuity = Continuity::C0;
    auto bdm = ElementFactory::create(req);
    ASSERT_TRUE(bdm);
    EXPECT_EQ(bdm->continuity(), Continuity::H_div);

    // Conflicting requests are rejected
    req.basis_type = BasisType::Nedelec;
    req.order      = 0;
    req.continuity = Continuity::H_div;
    EXPECT_THROW(ElementFactory::create(req), svmp::FE::FEException);
}

TEST(ElementCache, AggregatesBasisAndJacobianCaches) {
    LagrangeElement elem(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    svmp::FE::geometry::IsoparametricMapping mapping(basis, nodes);

    ElementCache& cache = ElementCache::instance();
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);

    ElementCacheEntry entry1 = cache.get(elem, mapping);
    ASSERT_NE(entry1.basis, nullptr);
    ASSERT_NE(entry1.jacobian, nullptr);
    EXPECT_GT(cache.size(), 0u);

    ElementCacheEntry entry2 = cache.get(elem, mapping);
    EXPECT_EQ(entry1.basis, entry2.basis);
    EXPECT_EQ(entry1.jacobian, entry2.jacobian);
}

TEST(ElementValidator, ReportsPositiveJacobianForIdentityQuad) {
    LagrangeElement elem(ElementType::Quad4, 1);
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    svmp::FE::geometry::IsoparametricMapping mapping(basis, nodes);

    ElementQuality q = ElementValidator::validate(elem, mapping);
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_GT(q.min_detJ, 0.0);
    EXPECT_GT(q.max_condition_number, 0.0);
}
