/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Sparsity/SparsityPattern.h"

#if defined(FE_HAS_EIGEN)
#include "Backends/Eigen/EigenFactory.h"
#endif

namespace svmp::FE::backends {

namespace {

sparsity::SparsityPattern make_1x1_pattern()
{
    sparsity::SparsityPattern p(1, 1);
    p.addEntry(0, 0);
    p.finalize();
    return p;
}

} // namespace

TEST(BlockSystems, BlockVectorAssembly)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    std::vector<std::unique_ptr<GenericVector>> blocks;
    blocks.push_back(factory.createVector(1));
    blocks.push_back(factory.createVector(1));

    BlockVector v(std::move(blocks));
    auto view = v.createAssemblyView();

    view->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real vals[2] = {11.0, 13.0};
    view->addVectorEntries(dofs, vals, assembly::AddMode::Insert);
    view->finalizeAssembly();

    EXPECT_DOUBLE_EQ(view->getVectorEntry(0), 11.0);
    EXPECT_DOUBLE_EQ(view->getVectorEntry(1), 13.0);
    EXPECT_DOUBLE_EQ(v.block(0).localSpan()[0], 11.0);
    EXPECT_DOUBLE_EQ(v.block(1).localSpan()[0], 13.0);
#endif
}

TEST(BlockSystems, BlockMatrixAssemblyAndMult)
{
#if !defined(FE_HAS_EIGEN)
    GTEST_SKIP() << "FE_HAS_EIGEN not enabled";
#else
    EigenFactory factory;
    const auto pat = make_1x1_pattern();

    std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks(2);
    blocks[0].push_back(factory.createMatrix(pat));
    blocks[0].push_back(factory.createMatrix(pat));
    blocks[1].push_back(factory.createMatrix(pat));
    blocks[1].push_back(factory.createMatrix(pat));

    BlockMatrix A({1, 1}, {1, 1}, std::move(blocks));

    // Assemble monolithic 2x2 dense contributions into the block structure.
    auto viewA = A.createAssemblyView();
    viewA->beginAssemblyPhase();
    const GlobalIndex dofs[2] = {0, 1};
    const Real Ke[4] = {2.0, 3.0,
                        5.0, 7.0};
    viewA->addMatrixEntries(dofs, Ke, assembly::AddMode::Insert);
    viewA->finalizeAssembly();
    A.finalizeAssembly();

    EXPECT_DOUBLE_EQ(A.getEntry(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(A.getEntry(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(A.getEntry(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(A.getEntry(1, 1), 7.0);

    // y = A x, with x = [11, 13].
    std::vector<std::unique_ptr<GenericVector>> x_blocks;
    x_blocks.push_back(factory.createVector(1));
    x_blocks.push_back(factory.createVector(1));
    BlockVector x(std::move(x_blocks));
    x.block(0).localSpan()[0] = 11.0;
    x.block(1).localSpan()[0] = 13.0;

    std::vector<std::unique_ptr<GenericVector>> y_blocks;
    y_blocks.push_back(factory.createVector(1));
    y_blocks.push_back(factory.createVector(1));
    BlockVector y(std::move(y_blocks));

    A.mult(x, y);
    EXPECT_DOUBLE_EQ(y.block(0).localSpan()[0], 61.0);
    EXPECT_DOUBLE_EQ(y.block(1).localSpan()[0], 146.0);

    // Zero global row 1 and set diagonal.
    const GlobalIndex rows_to_zero[1] = {1};
    viewA->zeroRows(rows_to_zero, true);
    EXPECT_DOUBLE_EQ(A.getEntry(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(A.getEntry(1, 1), 1.0);
#endif
}

} // namespace svmp::FE::backends

