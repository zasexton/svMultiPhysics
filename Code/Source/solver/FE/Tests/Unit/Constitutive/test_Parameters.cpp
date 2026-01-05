/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/Parameters.h"

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

TEST(ParametersTest, ReadsOptionalValues)
{
    Parameters params([](std::string_view key) -> std::optional<Real> {
        if (key == "a") return 2.0;
        return std::nullopt;
    });

    EXPECT_TRUE(params.getReal("a").has_value());
    EXPECT_DOUBLE_EQ(*params.getReal("a"), 2.0);
    EXPECT_FALSE(params.getReal("missing").has_value());
    EXPECT_DOUBLE_EQ(params.getRealOr("missing", 3.0), 3.0);
}

TEST(ParametersTest, ThrowsOnMissingRequiredValue)
{
    Parameters params;
    EXPECT_THROW((void)params.requireReal("a"), InvalidArgumentException);
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp

