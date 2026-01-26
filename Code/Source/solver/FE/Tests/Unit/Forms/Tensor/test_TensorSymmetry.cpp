/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorSymmetry.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorSymmetry, Symmetric2IndependentCount)
{
    const auto sym = TensorSymmetry::symmetric2();
    EXPECT_EQ(sym.numIndependentComponents(3), 6);

    const auto comps = sym.independentComponents(3);
    ASSERT_EQ(comps.size(), 6u);
    for (const auto& mi : comps) {
        ASSERT_EQ(mi.rank(), 2);
        ASSERT_TRUE(mi.indices[0].isFixed());
        ASSERT_TRUE(mi.indices[1].isFixed());
        // canonical representation uses fixed i<=j ordering
        EXPECT_LE(*mi.indices[0].fixed_value, *mi.indices[1].fixed_value);
    }
}

TEST(TensorSymmetry, Antisymmetric2IndependentCount)
{
    const auto asym = TensorSymmetry::antisymmetric2();
    EXPECT_EQ(asym.numIndependentComponents(3), 3);

    const auto comps = asym.independentComponents(3);
    ASSERT_EQ(comps.size(), 3u);
    for (const auto& mi : comps) {
        ASSERT_EQ(mi.rank(), 2);
        EXPECT_LT(*mi.indices[0].fixed_value, *mi.indices[1].fixed_value);
    }
}

TEST(TensorSymmetry, ElasticityIndependentCount3D)
{
    const auto el = TensorSymmetry::elasticity();
    EXPECT_EQ(el.numIndependentComponents(3), 21);

    const auto comps = el.independentComponents(3);
    EXPECT_EQ(comps.size(), 21u);
    for (const auto& mi : comps) {
        EXPECT_EQ(mi.rank(), 4);
        for (const auto& idx : mi.indices) {
            EXPECT_TRUE(idx.isFixed());
            ASSERT_TRUE(idx.fixed_value.has_value());
            EXPECT_GE(*idx.fixed_value, 0);
            EXPECT_LT(*idx.fixed_value, 3);
        }
    }
}

} // namespace svmp::FE::forms::tensor

