/**
 * @file test_ElementFactoryCacheValidator.cpp
 * @brief Tests for ElementFactory, ElementCache, and ElementValidator
 */

#include <gtest/gtest.h>

#include "FE/Elements/ElementFactory.h"
#include "FE/Elements/ElementCache.h"
#include "FE/Elements/GeneralBasisElement.h"
#include "FE/Elements/ElementValidator.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include <optional>

using namespace svmp::FE;
using namespace svmp::FE::elements;

namespace {

class TestCustomElementBasis final : public basis::BasisFunction {
public:
    explicit TestCustomElementBasis(int order)
        : order_(order) {}

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return 2u; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        values[0] = Real(0.5) * (Real(1) - xi[0]);
        values[1] = Real(0.5) * (Real(1) + xi[0]);
    }

private:
    int order_{1};
};

} // namespace

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

TEST(ElementFactory, CreatesGenericScalarBasisBackedElements) {
    struct Case {
        ElementType element_type;
        BasisType basis_type;
        std::optional<int> order;
        Continuity requested_continuity;
        Continuity expected_continuity;
    };

    const std::vector<Case> cases = {
        {ElementType::Line2, BasisType::Hierarchical, 3, Continuity::C0, Continuity::C0},
        {ElementType::Triangle3, BasisType::Bernstein, 2, Continuity::C0, Continuity::C0},
        {ElementType::Quad4, BasisType::Serendipity, 2, Continuity::C0, Continuity::C0},
        {ElementType::Quad4, BasisType::Hermite, 3, Continuity::C0, Continuity::C1},
        {ElementType::Triangle3, BasisType::Bubble, std::nullopt, Continuity::L2, Continuity::L2},
        {ElementType::Quad4, BasisType::BSpline, 2, Continuity::C0, Continuity::C0},
        {ElementType::Line2, BasisType::NURBS, 2, Continuity::C0, Continuity::C0},
        {ElementType::Quad4, BasisType::NURBS, 2, Continuity::C0, Continuity::C0},
        {ElementType::Hex8, BasisType::NURBS, 2, Continuity::C0, Continuity::C0},
    };

    for (const auto& c : cases) {
        ElementRequest req;
        req.element_type = c.element_type;
        req.basis_type = c.basis_type;
        req.field_type = FieldType::Scalar;
        req.continuity = c.requested_continuity;
        req.order = c.order;
        if (c.basis_type == BasisType::BSpline) {
            req.axis_orders = {2, 1};
            req.axis_knot_vectors = {
                {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
                {Real(0), Real(0), Real(0.3), Real(0.7), Real(1), Real(1)}
            };
        } else if (c.basis_type == BasisType::NURBS) {
            if (c.element_type == ElementType::Line2) {
                req.knot_vector = {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)};
                req.weights = {Real(1), Real(0.75), Real(1.5), Real(1)};
            } else if (c.element_type == ElementType::Quad4) {
                req.axis_orders = {2, 1};
                req.axis_knot_vectors = {
                    {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
                    {Real(0), Real(0), Real(0.35), Real(0.7), Real(1), Real(1)}
                };
                req.tensor_extents = {4, 4};
                req.weights.assign(16u, Real(1));
                req.weights[5] = Real(0.8);
                req.weights[10] = Real(1.4);
            } else {
                req.axis_orders = {2, 1, 1};
                req.axis_knot_vectors = {
                    {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
                    {Real(0), Real(0), Real(0.4), Real(1), Real(1)},
                    {Real(0), Real(0), Real(0.6), Real(1), Real(1)}
                };
                req.tensor_extents = {4, 3, 3};
                req.weights.assign(36u, Real(1));
                req.weights[4] = Real(0.65);
                req.weights[18] = Real(1.35);
                req.weights[31] = Real(0.9);
            }
        }

        auto elem = ElementFactory::create(req);
        ASSERT_TRUE(elem);
        EXPECT_EQ(elem->element_type(), c.element_type);
        EXPECT_EQ(elem->field_type(), FieldType::Scalar);
        EXPECT_EQ(elem->continuity(), c.expected_continuity);

        auto generic = std::dynamic_pointer_cast<GeneralBasisElement>(elem);
        ASSERT_TRUE(generic);

        auto basis = elem->basis_ptr();
        ASSERT_TRUE(basis);
        EXPECT_EQ(basis->basis_type(), c.basis_type);
        EXPECT_EQ(basis->element_type(), c.element_type);
        EXPECT_GT(elem->quadrature()->num_points(), 0u);
    }
}

TEST(ElementFactory, CreatesRegisteredCustomScalarElement) {
    basis::BasisFactory::clear_custom_registry_for_tests();
    basis::BasisFactory::register_custom(
        "test-element-custom",
        [](const basis::BasisRequest& req) -> std::shared_ptr<basis::BasisFunction> {
            return std::make_shared<TestCustomElementBasis>(req.order.value_or(1));
        });

    ElementRequest req;
    req.element_type = ElementType::Line2;
    req.basis_type = BasisType::Custom;
    req.field_type = FieldType::Scalar;
    req.continuity = Continuity::C0;
    req.order = 2;
    req.custom_id = "test-element-custom";

    auto elem = ElementFactory::create(req);
    ASSERT_TRUE(elem);
    EXPECT_EQ(elem->element_type(), ElementType::Line2);
    EXPECT_EQ(elem->field_type(), FieldType::Scalar);
    EXPECT_EQ(elem->continuity(), Continuity::C0);

    auto generic = std::dynamic_pointer_cast<GeneralBasisElement>(elem);
    ASSERT_TRUE(generic);
    ASSERT_TRUE(elem->basis_ptr());
    EXPECT_EQ(elem->basis_ptr()->basis_type(), BasisType::Custom);
    EXPECT_EQ(elem->basis_ptr()->order(), 2);

    basis::BasisFactory::clear_custom_registry_for_tests();
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
