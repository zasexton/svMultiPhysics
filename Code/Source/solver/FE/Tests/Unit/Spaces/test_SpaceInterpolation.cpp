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
