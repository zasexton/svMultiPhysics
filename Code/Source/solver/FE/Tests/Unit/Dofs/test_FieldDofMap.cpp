/**
 * @file test_FieldDofMap.cpp
 * @brief Unit tests for FieldDofMap
 */

#include <gtest/gtest.h>

#include "FE/Dofs/FieldDofMap.h"
#include "FE/Dofs/SubspaceView.h"
#include "FE/Core/FEException.h"

#include <set>
#include <vector>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::dofs::FieldDofMap;
using svmp::FE::dofs::FieldLayout;

TEST(FieldDofMap, BasicFieldRegistrationAndOffsets) {
    FieldDofMap map;
    map.addVectorField("velocity", /*n_components=*/3, /*n_dofs_per_component=*/4); // 12
    map.addScalarField("pressure", /*n_dofs=*/4); // 4
    map.finalize();

    EXPECT_EQ(map.numFields(), 2u);
    EXPECT_EQ(map.totalDofs(), 16);

    auto vel_range = map.getFieldDofRange("velocity");
    auto p_range = map.getFieldDofRange("pressure");
    EXPECT_EQ(vel_range.first, 0);
    EXPECT_EQ(vel_range.second, 12);
    EXPECT_EQ(p_range.first, 12);
    EXPECT_EQ(p_range.second, 16);

    EXPECT_EQ(map.numComponents("velocity"), 3);
    EXPECT_EQ(map.numComponents("pressure"), 1);
}

TEST(FieldDofMap, ComponentDofsInterleavedLayout) {
    FieldDofMap map;
    map.setLayout(FieldLayout::Interleaved);
    map.addVectorField("velocity", 3, 4); // 4 nodes per component
    map.finalize();

    auto cx = map.getComponentDofs("velocity", 0);
    auto cy = map.getComponentDofs("velocity", 1);
    auto cz = map.getComponentDofs("velocity", 2);

    EXPECT_EQ(cx.size(), 4);
    EXPECT_EQ(cy.size(), 4);
    EXPECT_EQ(cz.size(), 4);

    // Interleaved: (x0,y0,z0,x1,y1,z1,...)
    auto cx_vec = cx.toVector();
    ASSERT_EQ(cx_vec.size(), 4u);
    EXPECT_EQ(cx_vec[0], 0);
    EXPECT_EQ(cx_vec[1], 3);

    // DOF->(field,component)
    auto fc = map.getComponentOfDof(5);
    ASSERT_TRUE(fc.has_value());
    EXPECT_EQ(fc->first, 0);
    EXPECT_EQ(fc->second, static_cast<LocalIndex>(2)); // z component at node 1
}

TEST(FieldDofMap, ComponentDofsBlockLayout) {
    FieldDofMap map;
    map.setLayout(FieldLayout::Block);
    map.addVectorField("velocity", 3, 4);
    map.finalize();

    auto cx = map.getComponentDofs("velocity", 0).toVector();
    auto cy = map.getComponentDofs("velocity", 1).toVector();
    auto cz = map.getComponentDofs("velocity", 2).toVector();

    EXPECT_EQ(cx.front(), 0);
    EXPECT_EQ(cy.front(), 4);
    EXPECT_EQ(cz.front(), 8);

    // In block layout, DOF 5 should be in component 1 (y), local index 1.
    auto fc = map.getComponentOfDof(5);
    ASSERT_TRUE(fc.has_value());
    EXPECT_EQ(fc->first, 0);
    EXPECT_EQ(fc->second, static_cast<LocalIndex>(1));
}

TEST(FieldDofMap, FieldView) {
    FieldDofMap map;
    map.addVectorField("velocity", 3, 2); // 6
    map.addScalarField("pressure", 2); // 2
    map.finalize();

    auto vel_view = map.getFieldView("velocity");
    ASSERT_NE(vel_view, nullptr);
    EXPECT_EQ(vel_view->getLocalSize(), 6);
    EXPECT_TRUE(vel_view->contains(0));
    EXPECT_TRUE(vel_view->contains(5));
    EXPECT_FALSE(vel_view->contains(6));
}

TEST(FieldDofMap, RequiresFinalizeForQueries) {
    FieldDofMap map;
    map.addScalarField("p", 3);
    EXPECT_THROW(map.getFieldDofRange("p"), FEException);
}

TEST(FieldDofMap, VectorBasisFieldDisallowsComponentQueries) {
    FieldDofMap map;
    map.addVectorBasisField("E", /*value_dimension=*/3, /*n_dofs=*/6);
    map.finalize();

    EXPECT_EQ(map.numFields(), 1u);
    EXPECT_EQ(map.totalDofs(), 6);
    EXPECT_EQ(map.numComponents("E"), 3);

    auto range = map.getFieldDofRange("E");
    EXPECT_EQ(range.first, 0);
    EXPECT_EQ(range.second, 6);

    EXPECT_EQ(map.fieldToGlobal(0, 5), 5);
    EXPECT_THROW(map.getComponentDofs("E", 0), FEException);
    EXPECT_THROW(map.componentToGlobal(0, 0, 0), FEException);

    EXPECT_FALSE(map.getComponentOfDof(0).has_value());
}
