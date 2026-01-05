/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Core/FEException.h"
#include "Sparsity/SparsityPattern.h"

namespace svmp::FE::backends {

TEST(BackendFactory, UnknownBackendNameThrows)
{
    EXPECT_THROW((void)BackendFactory::create("not-a-backend"), InvalidArgumentException);
}

TEST(BackendFactory, CreateEigenWhenEnabled)
{
#if !defined(FE_HAS_EIGEN)
    EXPECT_THROW((void)BackendFactory::create("eigen"), NotImplementedException);
#else
    auto factory = BackendFactory::create(BackendKind::Eigen);
    ASSERT_TRUE(factory);
    EXPECT_EQ(factory->backendKind(), BackendKind::Eigen);

    auto factory2 = BackendFactory::create("eigen");
    ASSERT_TRUE(factory2);
    EXPECT_EQ(factory2->backendKind(), BackendKind::Eigen);
#endif
}

TEST(BackendFactory, CreatePetscOptional)
{
#if !defined(FE_HAS_PETSC)
    EXPECT_THROW((void)BackendFactory::create("petsc"), NotImplementedException);
#else
    auto factory = BackendFactory::create("petsc");
    ASSERT_TRUE(factory);
    EXPECT_EQ(factory->backendKind(), BackendKind::PETSc);
#endif
}

TEST(BackendFactory, CreateTrilinosOptional)
{
#if !defined(FE_HAS_TRILINOS)
    EXPECT_THROW((void)BackendFactory::create("trilinos"), NotImplementedException);
#else
    auto factory = BackendFactory::create("trilinos");
    ASSERT_TRUE(factory);
    EXPECT_EQ(factory->backendKind(), BackendKind::Trilinos);
#endif
}

TEST(BackendFactory, CreateFsilsOptional)
{
    auto factory = BackendFactory::create("fsils");
    ASSERT_TRUE(factory);
    EXPECT_EQ(factory->backendKind(), BackendKind::FSILS);
}

TEST(BackendFactory, CreateFsilsWithDofPerNode)
{
    BackendFactory::CreateOptions opts;
    opts.dof_per_node = 2;
    auto factory = BackendFactory::create("fsils", opts);
    ASSERT_TRUE(factory);
    EXPECT_EQ(factory->backendKind(), BackendKind::FSILS);

    sparsity::SparsityPattern p(4, 4);
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            p.addEntry(i, j);
        }
    }
    p.finalize();

    auto A = factory->createMatrix(p);
    ASSERT_TRUE(A);
    const auto* fsils_A = dynamic_cast<const FsilsMatrix*>(A.get());
    ASSERT_TRUE(fsils_A);
    EXPECT_EQ(fsils_A->fsilsDof(), 2);
}

} // namespace svmp::FE::backends
