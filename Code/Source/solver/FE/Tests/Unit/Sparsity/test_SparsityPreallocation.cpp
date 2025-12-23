/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "Sparsity/SparsityPreallocation.h"
#include <vector>
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// SparsityPreallocation Construction Tests
// ============================================================================

TEST(SparsityPreallocationTest, DefaultConstruction) {
    SparsityPreallocation prealloc;
    EXPECT_EQ(prealloc.numRows(), 0);
    EXPECT_EQ(prealloc.totalNnz(), 0);
    EXPECT_TRUE(prealloc.empty());
}

TEST(SparsityPreallocationTest, UniformConstruction) {
    SparsityPreallocation prealloc(10, 5);  // 10 rows, 5 NNZ per row

    EXPECT_EQ(prealloc.numRows(), 10);
    EXPECT_EQ(prealloc.totalNnz(), 50);
    EXPECT_EQ(prealloc.maxRowNnz(), 5);
    EXPECT_EQ(prealloc.minRowNnz(), 5);
    EXPECT_NEAR(prealloc.avgRowNnz(), 5.0, 1e-10);
    EXPECT_TRUE(prealloc.isUniform());
}

TEST(SparsityPreallocationTest, PerRowConstruction) {
    std::vector<GlobalIndex> nnz = {2, 5, 3, 8, 1};
    SparsityPreallocation prealloc(std::move(nnz));

    EXPECT_EQ(prealloc.numRows(), 5);
    EXPECT_EQ(prealloc.totalNnz(), 19);
    EXPECT_EQ(prealloc.maxRowNnz(), 8);
    EXPECT_EQ(prealloc.minRowNnz(), 1);
    EXPECT_FALSE(prealloc.isUniform());
}

TEST(SparsityPreallocationTest, FromPattern) {
    SparsityPattern pattern(5, 5);
    pattern.addEntry(0, 0);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 1);
    pattern.addEntry(1, 2);
    pattern.addEntry(1, 3);
    pattern.addEntry(2, 2);
    pattern.finalize();

    SparsityPreallocation prealloc(pattern);

    EXPECT_EQ(prealloc.numRows(), 5);
    EXPECT_EQ(prealloc.getRowNnz(0), 2);
    EXPECT_EQ(prealloc.getRowNnz(1), 3);
    EXPECT_EQ(prealloc.getRowNnz(2), 1);
    EXPECT_EQ(prealloc.getRowNnz(3), 0);
    EXPECT_EQ(prealloc.getRowNnz(4), 0);
}

// ============================================================================
// Query Tests
// ============================================================================

TEST(SparsityPreallocationTest, GetRowNnz) {
    std::vector<GlobalIndex> nnz = {1, 2, 3, 4, 5};
    SparsityPreallocation prealloc(std::move(nnz));

    EXPECT_EQ(prealloc.getRowNnz(0), 1);
    EXPECT_EQ(prealloc.getRowNnz(2), 3);
    EXPECT_EQ(prealloc.getRowNnz(4), 5);
}

TEST(SparsityPreallocationTest, GetNnzPerRow) {
    std::vector<GlobalIndex> nnz = {1, 2, 3};
    SparsityPreallocation prealloc(std::move(nnz));

    auto span = prealloc.getNnzPerRow();
    EXPECT_EQ(span.size(), 3);
    EXPECT_EQ(span[0], 1);
    EXPECT_EQ(span[1], 2);
    EXPECT_EQ(span[2], 3);
}

TEST(SparsityPreallocationTest, GetNnzPerRowData) {
    std::vector<GlobalIndex> nnz = {5, 3, 7};
    SparsityPreallocation prealloc(std::move(nnz));

    const GlobalIndex* data = prealloc.getNnzPerRowData();
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data[0], 5);
    EXPECT_EQ(data[1], 3);
    EXPECT_EQ(data[2], 7);
}

TEST(SparsityPreallocationTest, GetNnzPerRowVector) {
    std::vector<GlobalIndex> nnz = {10, 20, 30};
    SparsityPreallocation prealloc(std::move(nnz));

    auto int_vec = prealloc.getNnzPerRowVector<int>();
    EXPECT_EQ(int_vec.size(), 3);
    EXPECT_EQ(int_vec[0], 10);
    EXPECT_EQ(int_vec[1], 20);
    EXPECT_EQ(int_vec[2], 30);

    auto size_vec = prealloc.getNnzPerRowVector<std::size_t>();
    EXPECT_EQ(size_vec[1], 20);
}

// ============================================================================
// Modification Tests
// ============================================================================

TEST(SparsityPreallocationTest, ApplySafetyFactor) {
    SparsityPreallocation prealloc(10, 5);

    prealloc.applySafetyFactor(1.5);

    EXPECT_EQ(prealloc.maxRowNnz(), 8);  // ceil(5 * 1.5) = 8
    EXPECT_EQ(prealloc.totalNnz(), 80);
}

TEST(SparsityPreallocationTest, AddExtraPerRow) {
    std::vector<GlobalIndex> nnz = {1, 2, 3};
    SparsityPreallocation prealloc(std::move(nnz));

    prealloc.addExtraPerRow(5);

    EXPECT_EQ(prealloc.getRowNnz(0), 6);
    EXPECT_EQ(prealloc.getRowNnz(1), 7);
    EXPECT_EQ(prealloc.getRowNnz(2), 8);
}

TEST(SparsityPreallocationTest, ClampToMax) {
    std::vector<GlobalIndex> nnz = {5, 10, 15, 20};
    SparsityPreallocation prealloc(std::move(nnz));

    prealloc.clampToMax(12);

    EXPECT_EQ(prealloc.getRowNnz(0), 5);
    EXPECT_EQ(prealloc.getRowNnz(1), 10);
    EXPECT_EQ(prealloc.getRowNnz(2), 12);
    EXPECT_EQ(prealloc.getRowNnz(3), 12);
}

TEST(SparsityPreallocationTest, EnsureMinimum) {
    std::vector<GlobalIndex> nnz = {0, 1, 2, 5};
    SparsityPreallocation prealloc(std::move(nnz));

    prealloc.ensureMinimum(3);

    EXPECT_EQ(prealloc.getRowNnz(0), 3);
    EXPECT_EQ(prealloc.getRowNnz(1), 3);
    EXPECT_EQ(prealloc.getRowNnz(2), 3);
    EXPECT_EQ(prealloc.getRowNnz(3), 5);
}

// ============================================================================
// Combination Tests
// ============================================================================

TEST(SparsityPreallocationTest, Combine) {
    std::vector<GlobalIndex> nnz1 = {2, 5, 3};
    std::vector<GlobalIndex> nnz2 = {4, 3, 6};

    SparsityPreallocation p1(std::move(nnz1));
    SparsityPreallocation p2(std::move(nnz2));

    auto combined = p1.combine(p2);

    EXPECT_EQ(combined.getRowNnz(0), 4);  // max(2, 4)
    EXPECT_EQ(combined.getRowNnz(1), 5);  // max(5, 3)
    EXPECT_EQ(combined.getRowNnz(2), 6);  // max(3, 6)
}

TEST(SparsityPreallocationTest, Add) {
    std::vector<GlobalIndex> nnz1 = {2, 5, 3};
    std::vector<GlobalIndex> nnz2 = {4, 3, 6};

    SparsityPreallocation p1(std::move(nnz1));
    SparsityPreallocation p2(std::move(nnz2));

    auto sum = p1.add(p2);

    EXPECT_EQ(sum.getRowNnz(0), 6);  // 2 + 4
    EXPECT_EQ(sum.getRowNnz(1), 8);  // 5 + 3
    EXPECT_EQ(sum.getRowNnz(2), 9);  // 3 + 6
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(SparsityPreallocationTest, Validate) {
    std::vector<GlobalIndex> nnz = {1, 2, 3};
    SparsityPreallocation prealloc(std::move(nnz));

    EXPECT_TRUE(prealloc.validate());
}

TEST(SparsityPreallocationTest, MemoryUsage) {
    std::vector<GlobalIndex> nnz = {1, 2, 3, 4, 5};
    SparsityPreallocation prealloc(std::move(nnz));

    std::size_t mem = prealloc.memoryUsageBytes();
    EXPECT_GT(mem, 0);
    EXPECT_GE(mem, 5 * sizeof(GlobalIndex));
}

// ============================================================================
// Copy/Move Tests
// ============================================================================

TEST(SparsityPreallocationTest, Copy) {
    std::vector<GlobalIndex> nnz = {1, 2, 3};
    SparsityPreallocation original(std::move(nnz));

    SparsityPreallocation copy(original);

    EXPECT_EQ(copy.numRows(), 3);
    EXPECT_EQ(copy.getRowNnz(0), 1);
    EXPECT_EQ(copy.getRowNnz(2), 3);
}

TEST(SparsityPreallocationTest, Move) {
    std::vector<GlobalIndex> nnz = {1, 2, 3};
    SparsityPreallocation original(std::move(nnz));

    SparsityPreallocation moved(std::move(original));

    EXPECT_EQ(moved.numRows(), 3);
    EXPECT_EQ(moved.getRowNnz(1), 2);
}

// ============================================================================
// DistributedSparsityPreallocation Tests
// ============================================================================

TEST(DistributedSparsityPreallocationTest, DefaultConstruction) {
    DistributedSparsityPreallocation prealloc;
    EXPECT_EQ(prealloc.numOwnedRows(), 0);
    EXPECT_TRUE(prealloc.empty());
}

TEST(DistributedSparsityPreallocationTest, UniformConstruction) {
    DistributedSparsityPreallocation prealloc(10, 5, 3);  // 10 rows, 5 diag, 3 offdiag

    EXPECT_EQ(prealloc.numOwnedRows(), 10);
    EXPECT_EQ(prealloc.totalDiagNnz(), 50);
    EXPECT_EQ(prealloc.totalOffdiagNnz(), 30);
    EXPECT_EQ(prealloc.totalLocalNnz(), 80);
}

TEST(DistributedSparsityPreallocationTest, PerRowConstruction) {
    std::vector<GlobalIndex> diag = {2, 3, 4};
    std::vector<GlobalIndex> offdiag = {1, 2, 1};

    DistributedSparsityPreallocation prealloc(
        std::move(diag), std::move(offdiag));

    EXPECT_EQ(prealloc.numOwnedRows(), 3);
    EXPECT_EQ(prealloc.getDiagRowNnz(0), 2);
    EXPECT_EQ(prealloc.getOffdiagRowNnz(0), 1);
    EXPECT_EQ(prealloc.getRowNnz(0), 3);
}

TEST(DistributedSparsityPreallocationTest, DiagOffdiagArrays) {
    std::vector<GlobalIndex> diag = {5, 3, 7};
    std::vector<GlobalIndex> offdiag = {2, 4, 1};

    DistributedSparsityPreallocation prealloc(
        std::move(diag), std::move(offdiag));

    const GlobalIndex* d_nnz = prealloc.getDiagNnzData();
    const GlobalIndex* o_nnz = prealloc.getOffdiagNnzData();

    EXPECT_EQ(d_nnz[0], 5);
    EXPECT_EQ(d_nnz[1], 3);
    EXPECT_EQ(o_nnz[0], 2);
    EXPECT_EQ(o_nnz[2], 1);
}

TEST(DistributedSparsityPreallocationTest, MaxNnz) {
    std::vector<GlobalIndex> diag = {2, 5, 3};
    std::vector<GlobalIndex> offdiag = {4, 1, 6};

    DistributedSparsityPreallocation prealloc(
        std::move(diag), std::move(offdiag));

    EXPECT_EQ(prealloc.maxDiagRowNnz(), 5);
    EXPECT_EQ(prealloc.maxOffdiagRowNnz(), 6);
}

TEST(DistributedSparsityPreallocationTest, ApplySafetyFactor) {
    DistributedSparsityPreallocation prealloc(5, 10, 5);

    prealloc.applySafetyFactor(2.0);

    EXPECT_EQ(prealloc.maxDiagRowNnz(), 20);
    EXPECT_EQ(prealloc.maxOffdiagRowNnz(), 10);
}

TEST(DistributedSparsityPreallocationTest, ApplySeparateSafetyFactors) {
    DistributedSparsityPreallocation prealloc(5, 10, 5);

    prealloc.applySafetyFactors(1.5, 2.0);

    EXPECT_EQ(prealloc.maxDiagRowNnz(), 15);
    EXPECT_EQ(prealloc.maxOffdiagRowNnz(), 10);
}

TEST(DistributedSparsityPreallocationTest, GetCombinedPreallocation) {
    std::vector<GlobalIndex> diag = {2, 3};
    std::vector<GlobalIndex> offdiag = {1, 2};

    DistributedSparsityPreallocation dist_prealloc(
        std::move(diag), std::move(offdiag));

    SparsityPreallocation combined = dist_prealloc.getCombinedPreallocation();

    EXPECT_EQ(combined.numRows(), 2);
    EXPECT_EQ(combined.getRowNnz(0), 3);  // 2 + 1
    EXPECT_EQ(combined.getRowNnz(1), 5);  // 3 + 2
}

TEST(DistributedSparsityPreallocationTest, GetSeparatePreallocations) {
    std::vector<GlobalIndex> diag = {2, 3};
    std::vector<GlobalIndex> offdiag = {1, 2};

    DistributedSparsityPreallocation dist_prealloc(
        std::move(diag), std::move(offdiag));

    auto diag_prealloc = dist_prealloc.getDiagPreallocation();
    auto offdiag_prealloc = dist_prealloc.getOffdiagPreallocation();

    EXPECT_EQ(diag_prealloc.getRowNnz(0), 2);
    EXPECT_EQ(offdiag_prealloc.getRowNnz(0), 1);
}

TEST(DistributedSparsityPreallocationTest, Validate) {
    std::vector<GlobalIndex> diag = {2, 3};
    std::vector<GlobalIndex> offdiag = {1, 2};

    DistributedSparsityPreallocation prealloc(
        std::move(diag), std::move(offdiag));

    EXPECT_TRUE(prealloc.validate());
}

TEST(DistributedSparsityPreallocationTest, TypedVectors) {
    std::vector<GlobalIndex> diag = {100, 200};
    std::vector<GlobalIndex> offdiag = {50, 75};

    DistributedSparsityPreallocation prealloc(
        std::move(diag), std::move(offdiag));

    auto diag_int = prealloc.getDiagNnzVector<int>();
    auto offdiag_size_t = prealloc.getOffdiagNnzVector<std::size_t>();

    EXPECT_EQ(diag_int[0], 100);
    EXPECT_EQ(offdiag_size_t[1], 75);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST(SparsityPreallocationTest, CreateFromPattern) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 0);
    pattern.addEntry(1, 1);
    pattern.addEntry(2, 0);
    pattern.addEntry(2, 1);
    pattern.addEntry(2, 2);
    pattern.finalize();

    auto prealloc = createPreallocation(pattern);

    EXPECT_EQ(prealloc.numRows(), 3);
    EXPECT_EQ(prealloc.getRowNnz(0), 1);
    EXPECT_EQ(prealloc.getRowNnz(1), 1);
    EXPECT_EQ(prealloc.getRowNnz(2), 3);
}

TEST(SparsityPreallocationTest, UniformPreallocation) {
    auto prealloc = uniformPreallocation(100, 10);

    EXPECT_EQ(prealloc.numRows(), 100);
    EXPECT_EQ(prealloc.maxRowNnz(), 10);
    EXPECT_TRUE(prealloc.isUniform());
}

TEST(SparsityPreallocationTest, EstimatePreallocation) {
    auto prealloc = estimatePreallocation(1000, 500, 8, 4.0);

    EXPECT_EQ(prealloc.numRows(), 1000);
    EXPECT_GT(prealloc.maxRowNnz(), 0);
    EXPECT_LE(prealloc.maxRowNnz(), 1000);  // Can't exceed full row
}
