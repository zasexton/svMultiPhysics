/**
 * @file test_SpaceInterpolation.cpp
 * @brief Tests for SpaceInterpolation utilities
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceInterpolation.h"
#include "FE/Basis/LagrangeBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

TEST(SpaceInterpolation, ProjectsLinearSplinesAcrossNonmatchingKnotSpans) {
    elements::ElementRequest req;
    req.element_type = ElementType::Line2;
    req.basis_type = BasisType::BSpline;
    req.order = 1;
    req.knot_vector = {-1, -1, -0.5, 0.25, 1, 1};
    const H1Space fine(req);
    req.knot_vector = {-1, -1, 0, 1, 1};
    const H1Space coarse(req);
    const L2Space constant(ElementType::Line2, 0);
    const std::vector<Real> hat{0, 1, 0, 0};
    std::vector<Real> projected;
    SpaceInterpolation::l2_projection(fine, hat, constant, projected);
    ASSERT_EQ(projected.size(), 1u);
    EXPECT_NEAR(projected[0], 5.0 / 16, 2e-14);

    // Integrating the two sets of hats gives the cross-mass column (76,103,1)/288.
    SpaceInterpolation::l2_projection(fine, hat, coarse, projected);
    ASSERT_EQ(projected.size(), 3u);
    EXPECT_NEAR(projected[0], 109.0 / 192, 2e-14);
    EXPECT_NEAR(projected[1], 43.0 / 96, 2e-14);
    EXPECT_NEAR(projected[2], -41.0 / 192, 2e-14);
}

TEST(SpaceInterpolation, L2ProjectionSelfConsistency) {
    H1Space space(ElementType::Line2, 1);

    // Arbitrary coefficients representing some field
    std::vector<Real> src_coeffs(space.dofs_per_element());
    for (std::size_t i = 0; i < src_coeffs.size(); ++i) {
        src_coeffs[i] = Real(1) + Real(i);
    }

    std::vector<Real> dst_coeffs;
    SpaceInterpolation::l2_projection(space, src_coeffs, space, dst_coeffs);

    ASSERT_EQ(dst_coeffs.size(), src_coeffs.size());
    for (std::size_t i = 0; i < src_coeffs.size(); ++i) {
        EXPECT_NEAR(dst_coeffs[i], src_coeffs[i], 1e-12);
    }
}

TEST(SpaceInterpolation, NodalInterpolationLagrangeToLagrange) {
    H1Space src(ElementType::Line2, 1);
    H1Space dst(ElementType::Line2, 2);

    // Build arbitrary source coefficients
    std::vector<Real> src_coeffs(src.dofs_per_element());
    for (std::size_t i = 0; i < src_coeffs.size(); ++i) {
        src_coeffs[i] = Real(0.5) + Real(i);
    }

    std::vector<Real> dst_coeffs;
    SpaceInterpolation::nodal_interpolation(src, src_coeffs, dst, dst_coeffs);

    // Evaluate at destination nodes and compare with source evaluation
    const auto& dst_basis = dst.element().basis();
    auto dst_lagrange = dynamic_cast<const svmp::FE::basis::LagrangeBasis*>(&dst_basis);
    ASSERT_NE(dst_lagrange, nullptr);
    const auto& dst_nodes = dst_lagrange->nodes();

    ASSERT_EQ(dst_coeffs.size(), dst_nodes.size());

    for (std::size_t i = 0; i < dst_nodes.size(); ++i) {
        const auto& xi = dst_nodes[i];
        const Real src_val = src.evaluate_scalar(xi, src_coeffs);
        const Real dst_val = dst.evaluate_scalar(xi, dst_coeffs);
        EXPECT_NEAR(dst_val, src_val, 1e-12);
    }
}

TEST(SpaceInterpolation, RestrictionSatisfiesReferenceMassMoments) {
    H1Space fine(ElementType::Line2, 3);
    L2Space constant(ElementType::Line2, 0);
    H1Space linear(ElementType::Line2, 1);
    std::vector<Real> field, out;
    fine.interpolate_scalar([](const auto& p) { return p[0] * p[0] * (1. + p[0]); }, field);
    // On [-1,1], x^2 projects to 1/3 and x^3 projects to 3x/5.
    SpaceInterpolation::l2_projection(fine, field, constant, out);
    EXPECT_NEAR(constant.evaluate_scalar({}, out), 1. / 3., 2e-14);
    SpaceInterpolation::conservative_interpolation(fine, field, linear, out);
    for (Real x : {-1., 0., 1.})
        EXPECT_NEAR(linear.evaluate_scalar({x, 0., 0.}, out), 1. / 3. + 0.6 * x, 3e-14);

    const auto reverse = [](const FunctionSpace::Value& p) { return FunctionSpace::Value{-p[0], 0., 0.}; };
    const auto scale = [](const FunctionSpace::Value& value) { return value * 2.; };
    const auto restriction = SpaceInterpolation::restriction_operator(fine, linear, reverse, scale);
    SpaceInterpolation::apply_transfer(restriction, field, out);
    EXPECT_NEAR(linear.evaluate_scalar({0.5, 0., 0.}, out), 2. * (1. / 3. - 0.3), 4e-14);
    SpaceInterpolation::restrict_coefficients(fine, field, constant, field);
    EXPECT_NEAR(constant.evaluate_scalar({}, field), 1. / 3., 2e-14);
}

TEST(SpaceInterpolation, L2ProjectionPreservesSeparateVectorMeans) {
    ProductSpace fine(std::make_shared<H1Space>(ElementType::Line2, 2), 2);
    ProductSpace coarse(std::make_shared<L2Space>(ElementType::Line2, 0), 2);
    std::vector<Real> field, out;
    fine.interpolate([](const auto& p) {
        return FunctionSpace::Value{p[0] * p[0], 1. - 3. * p[0] * p[0], 0.};
    }, field);
    SpaceInterpolation::l2_projection(fine, field, coarse, out);
    const auto mean = coarse.evaluate({}, out);
    EXPECT_NEAR(mean[0], 1. / 3., 3e-14);
    EXPECT_NEAR(mean[1], 0., 3e-14);
}

TEST(SpaceInterpolation, L2ProjectionH1ToL2PreservesCoefficientsForSameOrder) {
    H1Space src(ElementType::Line2, 1);
    L2Space dst(ElementType::Line2, 1);

    std::vector<Real> src_coeffs = {Real(0), Real(2)}; // u(x)=1+x

    std::vector<Real> dst_coeffs;
    SpaceInterpolation::l2_projection(src, src_coeffs, dst, dst_coeffs);

    ASSERT_EQ(dst_coeffs.size(), dst.dofs_per_element());
    EXPECT_NEAR(dst_coeffs[0], src_coeffs[0], 1e-12);
    EXPECT_NEAR(dst_coeffs[1], src_coeffs[1], 1e-12);
}

TEST(SpaceInterpolation, L2ProjectionSupportsPRefinementTransfer) {
    H1Space src(ElementType::Line2, 1);
    H1Space dst(ElementType::Line2, 2);

    std::vector<Real> src_coeffs = {Real(0), Real(2)}; // u(x)=1+x

    std::vector<Real> dst_coeffs;
    SpaceInterpolation::l2_projection(src, src_coeffs, dst, dst_coeffs);

    ASSERT_EQ(dst_coeffs.size(), dst.dofs_per_element());

    const std::vector<Real> eval_pts = {Real(-1), Real(-0.25), Real(0), Real(0.75), Real(1)};
    for (Real x : eval_pts) {
        FunctionSpace::Value xi{};
        xi[0] = x;
        const Real u_src = src.evaluate_scalar(xi, src_coeffs);
        const Real u_dst = dst.evaluate_scalar(xi, dst_coeffs);
        EXPECT_NEAR(u_dst, u_src, 1e-12);
    }
}

TEST(SpaceInterpolation, ConservativeInterpolationMatchesL2Projection) {
    H1Space src(ElementType::Line2, 1);
    H1Space dst(ElementType::Line2, 2);

    std::vector<Real> src_coeffs = {Real(0), Real(2)};

    std::vector<Real> proj_coeffs;
    SpaceInterpolation::l2_projection(src, src_coeffs, dst, proj_coeffs);

    std::vector<Real> cons_coeffs;
    SpaceInterpolation::conservative_interpolation(src, src_coeffs, dst, cons_coeffs);

    ASSERT_EQ(cons_coeffs.size(), proj_coeffs.size());
    for (std::size_t i = 0; i < proj_coeffs.size(); ++i) {
        EXPECT_NEAR(cons_coeffs[i], proj_coeffs[i], 1e-12);
    }
}

TEST(SpaceInterpolation, ProlongationOperatorMatchesDirectTransfer) {
    H1Space coarse(ElementType::Line2, 1);
    H1Space fine(ElementType::Line2, 3);

    std::vector<Real> coarse_coeffs = {Real(-1), Real(2)};

    const auto op = SpaceInterpolation::prolongation_operator(coarse, fine);

    std::vector<Real> fine_coeffs;
    SpaceInterpolation::apply_transfer(op, coarse_coeffs, fine_coeffs);

    std::vector<Real> direct_coeffs;
    SpaceInterpolation::prolongate(coarse, coarse_coeffs, fine, direct_coeffs);

    ASSERT_EQ(fine_coeffs.size(), fine.dofs_per_element());
    ASSERT_EQ(direct_coeffs.size(), fine_coeffs.size());
    for (std::size_t i = 0; i < fine_coeffs.size(); ++i) {
        EXPECT_NEAR(fine_coeffs[i], direct_coeffs[i], 1e-12);
    }
}

TEST(SpaceInterpolation, RestrictionOperatorMatchesDirectTransfer) {
    H1Space fine(ElementType::Line2, 3);
    H1Space coarse(ElementType::Line2, 1);

    std::vector<Real> fine_coeffs = {Real(-1), Real(0.25), Real(1.0), Real(2.0)};

    const auto op = SpaceInterpolation::restriction_operator(fine, coarse);

    std::vector<Real> coarse_coeffs;
    SpaceInterpolation::apply_transfer(op, fine_coeffs, coarse_coeffs);

    std::vector<Real> direct_coeffs;
    SpaceInterpolation::restrict_coefficients(fine, fine_coeffs, coarse, direct_coeffs);

    ASSERT_EQ(coarse_coeffs.size(), coarse.dofs_per_element());
    ASSERT_EQ(direct_coeffs.size(), coarse_coeffs.size());
    for (std::size_t i = 0; i < coarse_coeffs.size(); ++i) {
        EXPECT_NEAR(coarse_coeffs[i], direct_coeffs[i], 1e-12);
    }
}

TEST(SpaceInterpolation, PointMapSupportsReversedReferenceOrientation) {
    H1Space src(ElementType::Line2, 1);
    H1Space dst(ElementType::Line2, 1);

    std::vector<Real> src_coeffs = {Real(2), Real(-1)};

    auto reverse = [](const FunctionSpace::Value& xi) {
        FunctionSpace::Value mapped = xi;
        mapped[0] = -mapped[0];
        return mapped;
    };

    std::vector<Real> dst_coeffs;
    SpaceInterpolation::prolongate(src, src_coeffs, dst, dst_coeffs, reverse);

    FunctionSpace::Value left{};
    left[0] = Real(-1);
    FunctionSpace::Value right{};
    right[0] = Real(1);

    EXPECT_NEAR(dst.evaluate_scalar(left, dst_coeffs), src.evaluate_scalar(right, src_coeffs), 1e-12);
    EXPECT_NEAR(dst.evaluate_scalar(right, dst_coeffs), src.evaluate_scalar(left, src_coeffs), 1e-12);
}
