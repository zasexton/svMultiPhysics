/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/SpecialTensors.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(SpecialTensors, Delta)
{
    EXPECT_EQ(special::delta(0, 0), 1);
    EXPECT_EQ(special::delta(0, 1), 0);
    EXPECT_EQ(special::delta(2, 1), 0);
    EXPECT_EQ(special::delta(2, 2), 1);
}

TEST(SpecialTensors, LeviCivita3D)
{
    EXPECT_EQ(special::levicivita(0, 1, 2), 1);
    EXPECT_EQ(special::levicivita(1, 2, 0), 1);
    EXPECT_EQ(special::levicivita(2, 0, 1), 1);

    EXPECT_EQ(special::levicivita(0, 2, 1), -1);
    EXPECT_EQ(special::levicivita(2, 1, 0), -1);
    EXPECT_EQ(special::levicivita(1, 0, 2), -1);

    EXPECT_EQ(special::levicivita(0, 0, 1), 0);
    EXPECT_EQ(special::levicivita(1, 2, 2), 0);
}

TEST(SpecialTensors, IdentityMetric)
{
    const auto g3 = special::identityMetric(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const Real expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(g3[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], expected);
        }
    }

    const auto g2 = special::identityMetric(2);
    EXPECT_DOUBLE_EQ(g2[0][0], 1.0);
    EXPECT_DOUBLE_EQ(g2[1][1], 1.0);
    EXPECT_DOUBLE_EQ(g2[2][2], 0.0);
    EXPECT_DOUBLE_EQ(g2[0][1], 0.0);
    EXPECT_DOUBLE_EQ(g2[1][2], 0.0);
}

} // namespace svmp::FE::forms::tensor

