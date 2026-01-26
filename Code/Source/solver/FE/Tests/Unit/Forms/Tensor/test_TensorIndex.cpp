/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorIndex.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorIndex, VarianceRaiseLower)
{
    TensorIndex i;
    i.id = 1;
    i.name = "i";
    i.variance = IndexVariance::Lower;
    i.role = IndexRole::Free;
    i.dimension = 3;

    const auto iu = i.raised();
    EXPECT_EQ(iu.id, 1);
    EXPECT_EQ(iu.name, "i");
    EXPECT_EQ(iu.variance, IndexVariance::Upper);

    const auto il = iu.lowered();
    EXPECT_EQ(il.variance, IndexVariance::Lower);
}

TEST(TensorIndex, MultiIndexFreeAndContractions)
{
    MultiIndex mi;
    mi.indices = {
        TensorIndex{.id = 10, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Free, .dimension = 3},
        TensorIndex{.id = 11, .name = "j", .variance = IndexVariance::None, .role = IndexRole::Free, .dimension = 3},
    };

    const auto free = mi.freeIndices();
    ASSERT_EQ(free.size(), 2u);
    EXPECT_EQ(free[0], 10);
    EXPECT_EQ(free[1], 11);
    EXPECT_TRUE(mi.contractionPairs().empty());

    // A(i,i) -> one contraction pair.
    MultiIndex tr;
    tr.indices = {
        TensorIndex{.id = 7, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Dummy, .dimension = 3},
        TensorIndex{.id = 7, .name = "i", .variance = IndexVariance::None, .role = IndexRole::Dummy, .dimension = 3},
    };
    const auto pairs = tr.contractionPairs();
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].first, 0);
    EXPECT_EQ(pairs[0].second, 1);
    EXPECT_TRUE(tr.isFullyContracted());
}

} // namespace svmp::FE::forms::tensor

