/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_BackendCapability.cpp
 * @brief Unit tests for BackendCapability.h
 */

#include <gtest/gtest.h>

#include "Backends/Utils/BackendCapability.h"

namespace svmp::FE::backends {

// ============================================================================
// Capability flag operations
// ============================================================================

TEST(BackendCapability, SingleCapabilityCheck)
{
    const CapabilitySet set = Capability::MonolithicSolve | Capability::DistributedSolve;
    EXPECT_TRUE(hasCapability(set, Capability::MonolithicSolve));
    EXPECT_TRUE(hasCapability(set, Capability::DistributedSolve));
    EXPECT_FALSE(hasCapability(set, Capability::GenericBlockSolve));
    EXPECT_FALSE(hasCapability(set, Capability::DirectSolve));
}

TEST(BackendCapability, EmptyCapabilitySet)
{
    constexpr CapabilitySet empty = 0u;
    EXPECT_FALSE(hasCapability(empty, Capability::MonolithicSolve));
    EXPECT_FALSE(hasCapability(empty, Capability::DistributedSolve));
}

TEST(BackendCapability, MultipleOrChain)
{
    const CapabilitySet set = Capability::MonolithicSolve
                            | Capability::GenericBlockSolve
                            | Capability::SaddlePointSolve
                            | Capability::DirectSolve;

    EXPECT_TRUE(hasCapability(set, Capability::MonolithicSolve));
    EXPECT_TRUE(hasCapability(set, Capability::GenericBlockSolve));
    EXPECT_TRUE(hasCapability(set, Capability::SaddlePointSolve));
    EXPECT_TRUE(hasCapability(set, Capability::DirectSolve));
    EXPECT_FALSE(hasCapability(set, Capability::MatrixFreeApplySingle));
    EXPECT_FALSE(hasCapability(set, Capability::AMGPreconditioner));
}

// ============================================================================
// BackendDescriptor
// ============================================================================

TEST(BackendDescriptor, SupportsCheck)
{
    BackendDescriptor desc{"test", Capability::MonolithicSolve | Capability::DirectSolve};

    EXPECT_TRUE(desc.supports(Capability::MonolithicSolve));
    EXPECT_TRUE(desc.supports(Capability::DirectSolve));
    EXPECT_FALSE(desc.supports(Capability::DistributedSolve));
}

TEST(BackendDescriptor, SupportsAllCheck)
{
    BackendDescriptor desc{"test",
                           Capability::MonolithicSolve
                         | Capability::DirectSolve
                         | Capability::DistributedSolve};

    const CapabilitySet req1 = Capability::MonolithicSolve | Capability::DirectSolve;
    EXPECT_TRUE(desc.supportsAll(req1));

    const CapabilitySet req2 = Capability::MonolithicSolve | Capability::AMGPreconditioner;
    EXPECT_FALSE(desc.supportsAll(req2));

    // Empty requirement always satisfied
    EXPECT_TRUE(desc.supportsAll(0u));
}

// ============================================================================
// Well-known profiles
// ============================================================================

TEST(BackendProfiles, FSILS)
{
    const auto fsils = profiles::fsils();
    EXPECT_EQ(fsils.name, "FSILS");
    EXPECT_TRUE(fsils.supports(Capability::MonolithicSolve));
    EXPECT_TRUE(fsils.supports(Capability::SaddlePointSolve));
    EXPECT_TRUE(fsils.supports(Capability::DistributedSolve));
    // FSILS does NOT have generic block, matrix-free, or direct solve
    EXPECT_FALSE(fsils.supports(Capability::GenericBlockSolve));
    EXPECT_FALSE(fsils.supports(Capability::MatrixFreeApplySingle));
    EXPECT_FALSE(fsils.supports(Capability::MatrixFreeApplyBlock));
    EXPECT_FALSE(fsils.supports(Capability::DirectSolve));
    EXPECT_FALSE(fsils.supports(Capability::AMGPreconditioner));
    EXPECT_FALSE(fsils.supports(Capability::FieldSplitPreconditioner));
    EXPECT_FALSE(fsils.supports(Capability::VariableBlockLayout));
}

TEST(BackendProfiles, PETSc)
{
    const auto petsc = profiles::petsc();
    EXPECT_EQ(petsc.name, "PETSc");
    // PETSc has everything
    EXPECT_TRUE(petsc.supports(Capability::MonolithicSolve));
    EXPECT_TRUE(petsc.supports(Capability::GenericBlockSolve));
    EXPECT_TRUE(petsc.supports(Capability::SaddlePointSolve));
    EXPECT_TRUE(petsc.supports(Capability::MatrixFreeApplySingle));
    EXPECT_TRUE(petsc.supports(Capability::MatrixFreeApplyBlock));
    EXPECT_TRUE(petsc.supports(Capability::FieldSplitPreconditioner));
    EXPECT_TRUE(petsc.supports(Capability::GenericBlockPreconditioner));
    EXPECT_TRUE(petsc.supports(Capability::DistributedSolve));
    EXPECT_TRUE(petsc.supports(Capability::DirectSolve));
    EXPECT_TRUE(petsc.supports(Capability::AMGPreconditioner));
    EXPECT_TRUE(petsc.supports(Capability::VariableBlockLayout));
}

TEST(BackendProfiles, Trilinos)
{
    const auto trilinos = profiles::trilinos();
    EXPECT_EQ(trilinos.name, "Trilinos");
    // Same as PETSc
    EXPECT_TRUE(trilinos.supports(Capability::MonolithicSolve));
    EXPECT_TRUE(trilinos.supports(Capability::GenericBlockSolve));
    EXPECT_TRUE(trilinos.supports(Capability::MatrixFreeApplyBlock));
    EXPECT_TRUE(trilinos.supports(Capability::DistributedSolve));
    EXPECT_TRUE(trilinos.supports(Capability::AMGPreconditioner));
}

TEST(BackendProfiles, Eigen3)
{
    const auto eigen3 = profiles::eigen3();
    EXPECT_EQ(eigen3.name, "Eigen3");
    EXPECT_TRUE(eigen3.supports(Capability::MonolithicSolve));
    EXPECT_TRUE(eigen3.supports(Capability::GenericBlockSolve));
    EXPECT_TRUE(eigen3.supports(Capability::SaddlePointSolve));
    EXPECT_TRUE(eigen3.supports(Capability::DirectSolve));
    EXPECT_TRUE(eigen3.supports(Capability::FieldSplitPreconditioner));
    EXPECT_TRUE(eigen3.supports(Capability::VariableBlockLayout));
    // Eigen3 is serial and has no matrix-free or AMG
    EXPECT_FALSE(eigen3.supports(Capability::DistributedSolve));
    EXPECT_FALSE(eigen3.supports(Capability::MatrixFreeApplySingle));
    EXPECT_FALSE(eigen3.supports(Capability::MatrixFreeApplyBlock));
    EXPECT_FALSE(eigen3.supports(Capability::AMGPreconditioner));
}

// ============================================================================
// Compatibility checking
// ============================================================================

TEST(BackendCompatibility, FullyCompatible)
{
    const auto petsc = profiles::petsc();
    const CapabilitySet required = Capability::MonolithicSolve
                                 | Capability::DistributedSolve
                                 | Capability::DirectSolve;
    const auto msg = checkCompatibility(petsc, required);
    EXPECT_TRUE(msg.empty()) << "Expected compatible, got: " << msg;
}

TEST(BackendCompatibility, EmptyRequirement)
{
    const auto fsils = profiles::fsils();
    const auto msg = checkCompatibility(fsils, 0u);
    EXPECT_TRUE(msg.empty());
}

TEST(BackendCompatibility, MissingCapability)
{
    const auto fsils = profiles::fsils();
    const CapabilitySet required = Capability::MonolithicSolve
                                 | Capability::GenericBlockSolve;  // FSILS lacks this
    const auto msg = checkCompatibility(fsils, required);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("GenericBlockSolve"), std::string::npos)
        << "Error message should mention missing capability: " << msg;
    EXPECT_NE(msg.find("FSILS"), std::string::npos)
        << "Error message should mention backend name: " << msg;
}

TEST(BackendCompatibility, MultipleMissing)
{
    const auto fsils = profiles::fsils();
    const CapabilitySet required = Capability::MatrixFreeApplySingle
                                 | Capability::AMGPreconditioner
                                 | Capability::DirectSolve;
    const auto msg = checkCompatibility(fsils, required);
    EXPECT_FALSE(msg.empty());
    // All three should be mentioned
    EXPECT_NE(msg.find("MatrixFreeApplySingle"), std::string::npos);
    EXPECT_NE(msg.find("AMGPreconditioner"), std::string::npos);
    EXPECT_NE(msg.find("DirectSolve"), std::string::npos);
}

TEST(BackendCompatibility, FsilsCanHandleSaddlePoint)
{
    const auto fsils = profiles::fsils();
    const CapabilitySet ns_req = Capability::MonolithicSolve
                               | Capability::SaddlePointSolve
                               | Capability::DistributedSolve;
    const auto msg = checkCompatibility(fsils, ns_req);
    EXPECT_TRUE(msg.empty()) << "FSILS should support NS: " << msg;
}

TEST(BackendCompatibility, Eigen3CannotDoDistributed)
{
    const auto eigen3 = profiles::eigen3();
    const CapabilitySet parallel_req = Capability::MonolithicSolve
                                     | Capability::DistributedSolve;
    const auto msg = checkCompatibility(eigen3, parallel_req);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("DistributedSolve"), std::string::npos);
}

} // namespace svmp::FE::backends
