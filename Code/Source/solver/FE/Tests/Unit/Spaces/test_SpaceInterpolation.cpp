/**
 * @file test_SpaceInterpolation.cpp
 * @brief Tests for SpaceInterpolation utilities
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Spaces/SpaceInterpolation.h"
#include "FE/Basis/LagrangeBasis.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

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
