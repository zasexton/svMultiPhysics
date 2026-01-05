/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendKind.h"

namespace svmp::FE::backends {

TEST(BackendKind, StringRoundTrip)
{
    EXPECT_EQ(backendKindFromString("eigen"), BackendKind::Eigen);
    EXPECT_EQ(backendKindFromString("EIGEN"), BackendKind::Eigen);
    EXPECT_EQ(backendKindToString(BackendKind::Eigen), "eigen");

    EXPECT_EQ(backendKindFromString("fsils"), BackendKind::FSILS);
    EXPECT_EQ(backendKindFromString("FSILS"), BackendKind::FSILS);
    EXPECT_EQ(backendKindToString(BackendKind::FSILS), "fsils");

    EXPECT_EQ(backendKindFromString("petsc"), BackendKind::PETSc);
    EXPECT_EQ(backendKindToString(BackendKind::PETSc), "petsc");

    EXPECT_EQ(backendKindFromString("trilinos"), BackendKind::Trilinos);
    EXPECT_EQ(backendKindToString(BackendKind::Trilinos), "trilinos");

    EXPECT_FALSE(backendKindFromString("nope").has_value());
}

} // namespace svmp::FE::backends

