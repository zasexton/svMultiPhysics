/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ConstraintTools.cpp
 * @brief Unit tests for ConstraintTools
 */

#include <gtest/gtest.h>
#include "Basis/NodeOrderingConventions.h"
#include "Constraints/ConstraintTools.h"
#include "Constraints/AffineConstraints.h"
#include "Spaces/HDivSpace.h"

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

using basis::ReferenceNodeLayout;
using ReferencePoint = math::Vector<Real, 3>;

void expectNodalInterpolation(ElementType type, int order, const std::vector<ReferencePoint>& nodes)
{
    SCOPED_TRACE(::testing::Message() << "element " << static_cast<int>(type) << ", degree " << order);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        const auto weights = computeInterpolationWeights({node[0], node[1], node[2]}, type, order);
        ASSERT_EQ(weights.size(), nodes.size());
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            EXPECT_NEAR(weights[j], i == j ? 1.0 : 0.0, 2e-12) << "row " << i << ", column " << j;
        }
    }

    const std::array<double, 3> point{0.13, 0.21, 0.17};
    const auto weights = computeInterpolationWeights(point, type, order);
    ASSERT_EQ(weights.size(), nodes.size());
    EXPECT_NEAR(std::accumulate(weights.begin(), weights.end(), 0.0), 1.0, 2e-12);
    for (int d = 0; order > 0 && d < element_dimension(type); ++d) {
        double affine = 0.0, polynomial = 0.0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            affine += weights[i] * nodes[i][d];
            polynomial += weights[i] * std::pow(nodes[i][d], order);
        }
        EXPECT_NEAR(affine, point[d], 2e-12);
        EXPECT_NEAR(polynomial, std::pow(point[d], order), 2e-12);
    }
}

TEST(ConstraintToolsTest, Triangle6InterpolatesInVertexThenEdgeOrder)
{
    const auto vertex = computeInterpolationWeights({0., 0., 0.}, ElementType::Triangle6, 2);
    EXPECT_EQ(vertex, (std::vector<double>{1., 0., 0., 0., 0., 0.}));

    const auto interior = computeInterpolationWeights({0.2, 0.3, 0.}, ElementType::Triangle6, 2);
    const std::array<double, 6> expected{0.0, -0.12, -0.12, 0.4, 0.24, 0.6};
    ASSERT_EQ(interior.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(interior[i], expected[i], 1e-14);
    }
}

TEST(ConstraintToolsTest, CompleteCellInterpolationRespectsTheFieldDegree)
{
    for (const auto [linear, quadratic] : {
        std::pair{ElementType::Line2, ElementType::Line3}, {ElementType::Triangle3, ElementType::Triangle6},
        {ElementType::Quad4, ElementType::Quad9}, {ElementType::Tetra4, ElementType::Tetra10},
        {ElementType::Hex8, ElementType::Hex27}, {ElementType::Wedge6, ElementType::Wedge18},
        {ElementType::Pyramid5, ElementType::Pyramid14}}) {
        for (int order = 0; order <= 3; ++order) {
            const auto nodes = ReferenceNodeLayout::get_lagrange_node_coords(linear, order);
            expectNodalInterpolation(linear, order, nodes);
            expectNodalInterpolation(quadratic, order, nodes);
        }
    }
    expectNodalInterpolation(ElementType::Point1, 0, {{0., 0., 0.}});
}

TEST(ConstraintToolsTest, SerendipityInterpolationUsesTheReducedFieldNodeSet)
{
    for (const auto [reduced, linear] : {
        std::pair{ElementType::Quad8, ElementType::Quad4}, {ElementType::Hex20, ElementType::Hex8},
        {ElementType::Wedge15, ElementType::Wedge6}, {ElementType::Pyramid13, ElementType::Pyramid5}}) {
        for (int order = 0; order <= 1; ++order) {
            expectNodalInterpolation(reduced, order, ReferenceNodeLayout::get_lagrange_node_coords(linear, order));
        }
        std::vector<ReferencePoint> nodes;
        for (std::size_t i = 0; i < ReferenceNodeLayout::num_nodes(reduced); ++i) {
            nodes.push_back(ReferenceNodeLayout::get_node_coords(reduced, i));
        }
        expectNodalInterpolation(reduced, 2, nodes);
    }
    auto cubic_quad = ReferenceNodeLayout::get_lagrange_node_coords(ElementType::Quad4, 3);
    cubic_quad.resize(12); // A cubic serendipity quad has four vertices and two nodes on each edge.
    expectNodalInterpolation(ElementType::Quad8, 3, cubic_quad);
}

TEST(ConstraintToolsTest, InterpolationRejectsInvalidOrUnavailableBases)
{
    const std::array<double, 3> point{};
    EXPECT_THROW((void)computeInterpolationWeights(point, ElementType::Triangle6, -1), FEException);
    EXPECT_THROW((void)computeInterpolationWeights(point, ElementType::Unknown, 1), FEException);
    for (auto type : {ElementType::Hex20, ElementType::Wedge15, ElementType::Pyramid13}) {
        EXPECT_THROW((void)computeInterpolationWeights(point, type, 3), FEException);
    }
}

TEST(ConstraintToolsTest, EdgeMidpointUsesTheSuppliedParentField)
{
    const std::array<GlobalIndex, 4> parents{10, 11, 12, 13};
    AffineConstraints constraints;
    makeHangingNodeConstraint2D(0, std::span(parents).first<2>(), 3, constraints);
    makeHangingNodeConstraint2D(1, std::span(parents).first<3>(), 2, constraints);
    makeHangingNodeConstraint2D(2, parents, 3, constraints);
    constraints.close();

    const auto linear = constraints.getConstraint(0);
    ASSERT_TRUE(linear);
    ASSERT_EQ(linear->entries.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(linear->entries[i].master_dof, parents[i]);
        EXPECT_DOUBLE_EQ(linear->entries[i].weight, 0.5);
    }
    const auto quadratic = constraints.getConstraint(1);
    ASSERT_TRUE(quadratic);
    ASSERT_EQ(quadratic->entries.size(), 1u);
    EXPECT_EQ(quadratic->entries[0].master_dof, parents[2]);
    EXPECT_DOUBLE_EQ(quadratic->entries[0].weight, 1.0);

    // Cubic edge nodes occur at -1, 1, -1/3 and 1/3 in FE order.
    const std::array<double, 4> at_midpoint{-1./16., -1./16., 9./16., 9./16.};
    const auto cubic = constraints.getConstraint(2);
    ASSERT_TRUE(cubic);
    ASSERT_EQ(cubic->entries.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(cubic->entries[i].master_dof, parents[i]);
        EXPECT_NEAR(cubic->entries[i].weight, at_midpoint[i], 1e-14);
    }
}

TEST(ConstraintToolsTest, SharedAndMalformedEdgeMidpointsLeaveConstraintsIntact)
{
    const std::array<GlobalIndex, 3> parents{10, 11, 12};
    AffineConstraints constraints;
    EXPECT_NO_THROW(makeHangingNodeConstraint2D(12, parents, 2, constraints));
    EXPECT_THROW(makeHangingNodeConstraint2D(20, parents, 3, constraints), ConstraintException);
    EXPECT_THROW(makeHangingNodeConstraint2D(21, parents, 0, constraints), ConstraintException);
    EXPECT_THROW(makeHangingNodeConstraint2D(22, {}, 2, constraints), ConstraintException);
    EXPECT_THROW(makeHangingNodeConstraint2D(10, std::span(parents).first<2>(), 1, constraints),
                 ConstraintException);
    const std::array<GlobalIndex, 2> invalid{10, -1};
    EXPECT_THROW(makeHangingNodeConstraint2D(23, invalid, 1, constraints), ConstraintException);
    EXPECT_TRUE(constraints.getConstrainedDofs().empty());
}

TEST(ConstraintToolsTest, MakeDirichletConstraints_Constant) {
    std::vector<GlobalIndex> dofs = {0, 5, 10};
    AffineConstraints constraints;
    
    makeDirichletConstraints(dofs, 3.14, constraints);
    constraints.close();

    EXPECT_EQ(constraints.numConstraints(), 3);
    
    auto c0 = constraints.getConstraint(0);
    ASSERT_TRUE(c0.has_value());
    EXPECT_DOUBLE_EQ(c0->inhomogeneity, 3.14);
    EXPECT_TRUE(c0->entries.empty());

    auto c5 = constraints.getConstraint(5);
    ASSERT_TRUE(c5.has_value());
    EXPECT_DOUBLE_EQ(c5->inhomogeneity, 3.14);
}

TEST(ConstraintToolsTest, ExtractComponentDofs) {
    // 3 DOFs per node, 2 nodes
    // Node 0: 0(u), 1(v), 2(w)
    // Node 1: 3(u), 4(v), 5(w)
    std::vector<GlobalIndex> all_dofs = {0, 1, 2, 3, 4, 5};
    
    auto u_dofs = extractComponentDofs(all_dofs, 0, 3);
    EXPECT_EQ(u_dofs.size(), 2);
    EXPECT_EQ(u_dofs[0], 0);
    EXPECT_EQ(u_dofs[1], 3);

    auto v_dofs = extractComponentDofs(all_dofs, 1, 3);
    EXPECT_EQ(v_dofs.size(), 2);
    EXPECT_EQ(v_dofs[0], 1);
    EXPECT_EQ(v_dofs[1], 4);
}

TEST(ConstraintToolsTest, MergeConstraints) {
    AffineConstraints c1;
    c1.addLine(0);
    c1.setInhomogeneity(0, 1.0);

    AffineConstraints c2;
    c2.addLine(1);
    c2.setInhomogeneity(1, 2.0);

    AffineConstraints target;
    mergeConstraints(target, c1);
    mergeConstraints(target, c2);
    target.close();

    EXPECT_EQ(target.numConstraints(), 2);
    EXPECT_TRUE(target.isConstrained(0));
    EXPECT_TRUE(target.isConstrained(1));
}

TEST(ConstraintToolsTest, MakePeriodicConstraints_Translation) {
    // Simple 1D periodic: u_0 (x=0) = u_1 (x=1)
    std::vector<GlobalIndex> slave = {0};
    std::vector<std::array<double, 3>> slave_coords = {{{0.0, 0.0, 0.0}}};
    
    std::vector<GlobalIndex> master = {1};
    std::vector<std::array<double, 3>> master_coords = {{{1.0, 0.0, 0.0}}};

    AffineConstraints constraints;
    std::array<double, 3> translation = {{1.0, 0.0, 0.0}}; // slave + trans = master

    makePeriodicConstraintsTranslation(slave, slave_coords, master, master_coords, translation, constraints);
    constraints.close();

    // Check u_0 = u_1
    auto c = constraints.getConstraint(0);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c->entries.size(), 1);
    EXPECT_EQ(c->entries[0].master_dof, 1);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, 1.0);
}

TEST(ConstraintToolsTest, MakeHDivTracePeriodicPairsTranslationUsesOppositeBoundaryNormals)
{
    spaces::HDivSpace space(ElementType::Quad4, /*order=*/1);

    const TraceBoundaryEntity slave{
        .dofs = {0, 1},
        .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
        .outward_normal = {{-1.0, 0.0, 0.0}},
    };
    const TraceBoundaryEntity master{
        .dofs = {10, 11},
        .vertices = {{{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}},
        .outward_normal = {{1.0, 0.0, 0.0}},
    };

    const auto pairs = makeHDivTracePeriodicPairsTranslation(
        space,
        std::span<const TraceBoundaryEntity>(&slave, 1),
        std::span<const TraceBoundaryEntity>(&master, 1),
        {{1.0, 0.0, 0.0}});

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].slave_dof, 0);
    EXPECT_EQ(pairs[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(pairs[0].weight, -1.0);
    EXPECT_EQ(pairs[1].slave_dof, 1);
    EXPECT_EQ(pairs[1].master_dof, 11);
    EXPECT_DOUBLE_EQ(pairs[1].weight, -1.0);
}

TEST(ConstraintToolsTest, MakeHDivTracePeriodicPairsTranslationAlignsReversedEdgeTraceOrdering)
{
    spaces::HDivSpace space(ElementType::Quad4, /*order=*/1);

    const TraceBoundaryEntity slave{
        .dofs = {0, 1},
        .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
        .outward_normal = {{-1.0, 0.0, 0.0}},
    };
    const TraceBoundaryEntity master{
        .dofs = {10, 11},
        .vertices = {{{1.0, 1.0, 0.0}, {1.0, 0.0, 0.0}}},
        .outward_normal = {{1.0, 0.0, 0.0}},
    };

    const auto pairs = makeHDivTracePeriodicPairsTranslation(
        space,
        std::span<const TraceBoundaryEntity>(&slave, 1),
        std::span<const TraceBoundaryEntity>(&master, 1),
        {{1.0, 0.0, 0.0}});

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].slave_dof, 1);
    EXPECT_EQ(pairs[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(pairs[0].weight, 1.0);
    EXPECT_EQ(pairs[1].slave_dof, 0);
    EXPECT_EQ(pairs[1].master_dof, 11);
    EXPECT_DOUBLE_EQ(pairs[1].weight, 1.0);
}

TEST(ConstraintToolsTest, MakeHDivTracePeriodicPairsTranslationPermutesQuadrilateralFaceTrace)
{
    spaces::HDivSpace space(ElementType::Hex8, /*order=*/1);

    const TraceBoundaryEntity slave{
        .dofs = {0, 1, 2, 3},
        .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}},
        .outward_normal = {{-1.0, 0.0, 0.0}},
    };
    const TraceBoundaryEntity master{
        .dofs = {10, 11, 12, 13},
        .vertices = {{{1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 0.0, 0.0}}},
        .outward_normal = {{1.0, 0.0, 0.0}},
    };

    const auto pairs = makeHDivTracePeriodicPairsTranslation(
        space,
        std::span<const TraceBoundaryEntity>(&slave, 1),
        std::span<const TraceBoundaryEntity>(&master, 1),
        {{1.0, 0.0, 0.0}});

    ASSERT_EQ(pairs.size(), 4u);
    EXPECT_EQ(pairs[0].slave_dof, 3);
    EXPECT_EQ(pairs[0].master_dof, 10);
    EXPECT_DOUBLE_EQ(pairs[0].weight, -1.0);
    EXPECT_EQ(pairs[1].slave_dof, 0);
    EXPECT_EQ(pairs[1].master_dof, 11);
    EXPECT_DOUBLE_EQ(pairs[1].weight, -1.0);
    EXPECT_EQ(pairs[2].slave_dof, 1);
    EXPECT_EQ(pairs[2].master_dof, 12);
    EXPECT_DOUBLE_EQ(pairs[2].weight, -1.0);
    EXPECT_EQ(pairs[3].slave_dof, 2);
    EXPECT_EQ(pairs[3].master_dof, 13);
    EXPECT_DOUBLE_EQ(pairs[3].weight, -1.0);
}

TEST(ConstraintToolsTest, MakeHDivTracePeriodicMPCTranslationBuildsExplicitConstraints)
{
    spaces::HDivSpace space(ElementType::Quad4, /*order=*/0);

    const TraceBoundaryEntity slave{
        .dofs = {2},
        .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
        .outward_normal = {{-1.0, 0.0, 0.0}},
    };
    const TraceBoundaryEntity master{
        .dofs = {7},
        .vertices = {{{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}},
        .outward_normal = {{1.0, 0.0, 0.0}},
    };

    auto mpc = makeHDivTracePeriodicMPCTranslation(
        space,
        std::span<const TraceBoundaryEntity>(&slave, 1),
        std::span<const TraceBoundaryEntity>(&master, 1),
        {{1.0, 0.0, 0.0}});

    AffineConstraints constraints;
    mpc.apply(constraints);
    constraints.close();

    const auto c = constraints.getConstraint(2);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c->entries.size(), 1u);
    EXPECT_EQ(c->entries[0].master_dof, 7);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, -1.0);
}

TEST(ConstraintToolsTest, MakeHDivTracePeriodicBCTranslationBuildsPeriodicConstraint)
{
    spaces::HDivSpace space(ElementType::Quad4, /*order=*/0);

    const TraceBoundaryEntity slave{
        .dofs = {3},
        .vertices = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
        .outward_normal = {{-1.0, 0.0, 0.0}},
    };
    const TraceBoundaryEntity master{
        .dofs = {8},
        .vertices = {{{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}},
        .outward_normal = {{1.0, 0.0, 0.0}},
    };

    auto bc = makeHDivTracePeriodicBCTranslation(
        space,
        std::span<const TraceBoundaryEntity>(&slave, 1),
        std::span<const TraceBoundaryEntity>(&master, 1),
        {{1.0, 0.0, 0.0}});

    AffineConstraints constraints;
    bc.apply(constraints);
    constraints.close();

    const auto c = constraints.getConstraint(3);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c->entries.size(), 1u);
    EXPECT_EQ(c->entries[0].master_dof, 8);
    EXPECT_DOUBLE_EQ(c->entries[0].weight, -1.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
