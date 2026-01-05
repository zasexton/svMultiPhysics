/**
 * @file test_FunctionSpaces.cpp
 * @brief Unit tests for core function space types
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/MixedSpace.h"
#include "FE/Spaces/TraceSpace.h"
#include "FE/Spaces/MortarSpace.h"
#include "FE/Spaces/CompositeSpace.h"
#include "FE/Spaces/EnrichedSpace.h"
#include "FE/Spaces/AdaptiveSpace.h"
#include "FE/Spaces/SpaceFactory.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

TEST(FunctionSpaces, H1SpaceMetadataAndInterpolation) {
    H1Space space(ElementType::Line2, 1);

    EXPECT_EQ(space.space_type(), SpaceType::H1);
    EXPECT_EQ(space.field_type(), FieldType::Scalar);
    EXPECT_EQ(space.continuity(), Continuity::C0);
    EXPECT_EQ(space.value_dimension(), 1);
    EXPECT_EQ(space.topological_dimension(), 1);
    EXPECT_EQ(space.polynomial_order(), 1);
    EXPECT_EQ(space.element_type(), ElementType::Line2);

    const auto ndofs = space.dofs_per_element();
    EXPECT_GT(ndofs, 0u);

    // Interpolate a linear function f(xi) = 1 + xi on [-1,1]
    std::vector<Real> coeffs;
    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& xi) {
        FunctionSpace::Value out{};
        out[0] = Real(1) + xi[0];
        return out;
    };
    space.interpolate(f, coeffs);
    EXPECT_EQ(coeffs.size(), ndofs);

    auto quad = space.element().quadrature();
    ASSERT_TRUE(quad);

    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        auto xi = quad->point(q);
        const Real expected = Real(1) + xi[0];
        const Real approx = space.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(approx, expected, 1e-10);
    }
}

TEST(FunctionSpaces, L2SpaceMetadata) {
    L2Space space(ElementType::Triangle3, 1);
    EXPECT_EQ(space.space_type(), SpaceType::L2);
    EXPECT_EQ(space.field_type(), FieldType::Scalar);
    EXPECT_EQ(space.continuity(), Continuity::L2);
    EXPECT_EQ(space.topological_dimension(), 2);
    EXPECT_EQ(space.element_type(), ElementType::Triangle3);
    EXPECT_GT(space.dofs_per_element(), 0u);
}

TEST(FunctionSpaces, VectorSpacesMetadata) {
    HCurlSpace hcurl(ElementType::Quad4, 0);
    EXPECT_EQ(hcurl.space_type(), SpaceType::HCurl);
    EXPECT_EQ(hcurl.field_type(), FieldType::Vector);
    EXPECT_EQ(hcurl.continuity(), Continuity::H_curl);
    EXPECT_EQ(hcurl.topological_dimension(), 2);
    EXPECT_GT(hcurl.dofs_per_element(), 0u);

    HDivSpace hdiv(ElementType::Quad4, 0);
    EXPECT_EQ(hdiv.space_type(), SpaceType::HDiv);
    EXPECT_EQ(hdiv.field_type(), FieldType::Vector);
    EXPECT_EQ(hdiv.continuity(), Continuity::H_div);
    EXPECT_EQ(hdiv.topological_dimension(), 2);
    EXPECT_GT(hdiv.dofs_per_element(), 0u);
}

TEST(FunctionSpaces, ProductSpaceVectorValuedEvaluation) {
    auto base = std::make_shared<H1Space>(ElementType::Line2, 1);
    ProductSpace space(base, 2); // 2D vector-valued H1

    EXPECT_EQ(space.space_type(), SpaceType::Product);
    EXPECT_EQ(space.field_type(), FieldType::Vector);
    EXPECT_EQ(space.value_dimension(), 2);

    const std::size_t per_comp = base->dofs_per_element();
    std::vector<Real> coeffs(2 * per_comp, Real(0));
    // First component: constant 1, second: linear xi
    for (std::size_t i = 0; i < per_comp; ++i) {
        coeffs[i] = Real(1);
    }

    // Use base space quadrature points for evaluation
    auto quad = base->element().quadrature();
    ASSERT_TRUE(quad);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        auto xi = quad->point(q);
        FunctionSpace::Value v = space.evaluate(xi, coeffs);
        EXPECT_NEAR(v[0], 1.0, 1e-12);
    }
}

TEST(FunctionSpaces, ProductSpaceInterpolationIsComponentWise) {
    auto base = std::make_shared<H1Space>(ElementType::Line2, 1);
    ProductSpace space(base, 2);

    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& xi) {
        FunctionSpace::Value out{};
        out[0] = Real(1);
        out[1] = xi[0];
        return out;
    };

    std::vector<Real> coeffs;
    space.interpolate(f, coeffs);
    ASSERT_EQ(coeffs.size(), space.dofs_per_element());

    // Verify each component block matches base-space scalar interpolation.
    const std::size_t per_comp = base->dofs_per_element();
    std::vector<Real> c0;
    std::vector<Real> c1;
    base->interpolate_scalar([](const FunctionSpace::Value&) { return Real(1); }, c0);
    base->interpolate_scalar([](const FunctionSpace::Value& xi) { return xi[0]; }, c1);
    ASSERT_EQ(c0.size(), per_comp);
    ASSERT_EQ(c1.size(), per_comp);

    for (std::size_t i = 0; i < per_comp; ++i) {
        EXPECT_NEAR(coeffs[i], c0[i], 1e-12);
        EXPECT_NEAR(coeffs[per_comp + i], c1[i], 1e-12);
    }

    // Sanity: evaluate at quadrature points.
    auto quad = base->element().quadrature();
    ASSERT_TRUE(quad);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        auto xi = quad->point(q);
        const auto v = space.evaluate(xi, coeffs);
        EXPECT_NEAR(v[0], 1.0, 1e-12);
        EXPECT_NEAR(v[1], xi[0], 1e-12);
    }
}

TEST(FunctionSpaces, MixedSpaceMetadataAndOffsets) {
    MixedSpace mixed;
    auto v = std::make_shared<ProductSpace>(
        SpaceFactory::create_h1(ElementType::Triangle3, 1), 2);
    auto p = SpaceFactory::create_l2(ElementType::Triangle3, 1);

    mixed.add_component("velocity", v);
    mixed.add_component("pressure", p);

    EXPECT_EQ(mixed.space_type(), SpaceType::Mixed);
    EXPECT_EQ(mixed.num_components(), 2u);
    EXPECT_GT(mixed.dofs_per_element(), 0u);

    const auto off_v = mixed.component_offset(0);
    const auto off_p = mixed.component_offset(1);
    EXPECT_EQ(off_v, 0u);
    EXPECT_GT(off_p, 0u);
}

TEST(FunctionSpaces, TraceSpaceReducesDimension) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 1);
    TraceSpace trace(h1, 0);

    EXPECT_EQ(trace.space_type(), SpaceType::Trace);
    EXPECT_EQ(trace.field_type(), FieldType::Scalar);
    EXPECT_EQ(trace.topological_dimension(), 1); // Quad4 volume dim is 2
    EXPECT_EQ(trace.element_type(), ElementType::Line2);

    // Evaluation uses face reference coordinates and face coefficients; compare
    // against volume evaluation at the embedded face point.
    std::vector<Real> elem_coeffs(h1->dofs_per_element(), Real(1));
    const auto face_coeffs = trace.restrict(elem_coeffs);
    EXPECT_EQ(face_coeffs.size(), trace.dofs_per_element());

    FunctionSpace::Value xi_face{};
    xi_face[0] = Real(0); // midpoint of Line2 reference face

    const auto xi_volume = trace.embed_face_point(xi_face);

    auto val_trace = trace.evaluate(xi_face, face_coeffs);
    auto val_base  = h1->evaluate(xi_volume, elem_coeffs);
    EXPECT_NEAR(val_trace[0], val_base[0], 1e-14);
}

TEST(FunctionSpaces, MortarSpaceWrapsInterfaceSpace) {
    auto iface = std::make_shared<H1Space>(ElementType::Triangle3, 1);
    MortarSpace mortar(iface);

    EXPECT_EQ(mortar.space_type(), SpaceType::Mortar);
    EXPECT_EQ(mortar.field_type(), iface->field_type());
    EXPECT_EQ(mortar.continuity(), iface->continuity());
    EXPECT_EQ(mortar.element_type(), iface->element_type());
    EXPECT_EQ(mortar.topological_dimension(), iface->topological_dimension());
    EXPECT_EQ(mortar.dofs_per_element(), iface->dofs_per_element());

    FunctionSpace::Value xi{};
    xi[0] = Real(0.2);
    xi[1] = Real(0.3);

    std::vector<Real> coeffs(mortar.dofs_per_element(), Real(1));
    const auto val_mortar = mortar.evaluate(xi, coeffs);
    const auto val_iface  = iface->evaluate(xi, coeffs);
    EXPECT_NEAR(val_mortar[0], val_iface[0], 1e-14);
}

TEST(FunctionSpaces, EnrichedSpaceCombinesBaseAndEnrichment) {
    auto base = std::make_shared<H1Space>(ElementType::Line2, 1);
    auto enr  = std::make_shared<L2Space>(ElementType::Line2, 1);

    EnrichedSpace space(base, enr);
    EXPECT_EQ(space.space_type(), SpaceType::Enriched);
    EXPECT_EQ(space.dofs_per_element(),
              base->dofs_per_element() + enr->dofs_per_element());

    // Build coefficients: base contributes 1, enrichment contributes 0
    const std::size_t base_dofs = base->dofs_per_element();
    const std::size_t enr_dofs  = enr->dofs_per_element();
    std::vector<Real> coeffs(base_dofs + enr_dofs, Real(0));
    for (std::size_t i = 0; i < base_dofs; ++i) {
        coeffs[i] = Real(1);
    }

    FunctionSpace::Value xi{};
    xi[0] = Real(0.25);
    auto val_base = base->evaluate(xi, std::vector<Real>(base_dofs, Real(1)));
    auto val_enr  = space.evaluate(xi, coeffs);
    EXPECT_NEAR(val_base[0], val_enr[0], 1e-14);

    // Interpolation should fill base part and zero enrichment
    std::vector<Real> interp_coeffs;
    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& x) {
        FunctionSpace::Value out{};
        out[0] = Real(2) + x[0];
        return out;
    };
    space.interpolate(f, interp_coeffs);
    EXPECT_EQ(interp_coeffs.size(), base_dofs + enr_dofs);
    for (std::size_t i = base_dofs; i < interp_coeffs.size(); ++i) {
        EXPECT_NEAR(interp_coeffs[i], 0.0, 1e-14);
    }
}

TEST(FunctionSpaces, AdaptiveSpaceSelectsActiveLevel) {
    AdaptiveSpace adaptive;
    auto p1 = std::make_shared<H1Space>(ElementType::Line2, 1);
    auto p2 = std::make_shared<H1Space>(ElementType::Line2, 2);

    adaptive.add_level(1, p1);
    adaptive.add_level(2, p2);

    EXPECT_EQ(adaptive.space_type(), SpaceType::Adaptive);
    EXPECT_EQ(adaptive.polynomial_order(), 2); // highest order active by default

    adaptive.set_active_level_by_order(1);
    EXPECT_EQ(adaptive.polynomial_order(), 1);
    EXPECT_EQ(adaptive.dofs_per_element(), p1->dofs_per_element());

    // Evaluation delegates to active level
    std::vector<Real> coeffs(p1->dofs_per_element(), Real(1));
    FunctionSpace::Value xi{};
    xi[0] = Real(0.0);
    auto val1 = adaptive.evaluate(xi, coeffs);
    EXPECT_NEAR(val1[0], 1.0, 1e-14);
}

TEST(FunctionSpaces, SpaceFactoryCreatesCoreSpaces) {
    auto h1 = SpaceFactory::create(SpaceType::H1, ElementType::Triangle3, 1);
    auto l2 = SpaceFactory::create(SpaceType::L2, ElementType::Triangle3, 1);
    auto hd = SpaceFactory::create(SpaceType::HDiv, ElementType::Quad4, 0);
    auto hc = SpaceFactory::create(SpaceType::HCurl, ElementType::Quad4, 0);

    EXPECT_EQ(h1->space_type(), SpaceType::H1);
    EXPECT_EQ(l2->space_type(), SpaceType::L2);
    EXPECT_EQ(hd->space_type(), SpaceType::HDiv);
    EXPECT_EQ(hc->space_type(), SpaceType::HCurl);

    EXPECT_THROW(SpaceFactory::create(SpaceType::Mixed, ElementType::Triangle3, 1),
                 svmp::FE::FEException);
    EXPECT_THROW(SpaceFactory::create(SpaceType::Mortar, ElementType::Triangle3, 1),
                 svmp::FE::FEException);
}

TEST(FunctionSpaces, CompositeSpaceRegionMappingAndMetadata) {
    CompositeSpace composite;

    auto reg1 = SpaceFactory::create_h1(ElementType::Tetra4, 1);
    auto reg2 = SpaceFactory::create_l2(ElementType::Tetra4, 0);

    composite.add_region(1, reg1);
    composite.add_region(2, reg2);

    EXPECT_EQ(composite.space_type(), SpaceType::Composite);
    EXPECT_EQ(composite.num_regions(), 2u);

    // First region drives metadata
    EXPECT_EQ(composite.element_type(), reg1->element_type());
    EXPECT_EQ(composite.topological_dimension(), reg1->topological_dimension());

    // Region lookups
    const FunctionSpace& s1 = composite.space_for_region(1);
    const FunctionSpace& s2 = composite.space_for_region(2);
    EXPECT_EQ(s1.space_type(), SpaceType::H1);
    EXPECT_EQ(s2.space_type(), SpaceType::L2);

    auto opt = composite.try_space_for_region(2);
    ASSERT_TRUE(opt);
    EXPECT_EQ(opt->space_type(), SpaceType::L2);

    EXPECT_THROW(composite.space_for_region(99), svmp::FE::FEException);
}
