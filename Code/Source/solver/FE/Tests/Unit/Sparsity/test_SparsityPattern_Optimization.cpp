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
#include "Sparsity/SparsityPattern.h"

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Optimization Tests - Non-Finalized Patterns
// ============================================================================

TEST(SparsityPatternOptimizationTest, PatternUnionNonFinalized) {
    SparsityPattern a(3, 3);
    a.addEntry(0, 0);
    a.addEntry(0, 1);
    // Not calling finalize()

    SparsityPattern b(3, 3);
    b.addEntry(0, 1);
    b.addEntry(1, 1);
    // Not calling finalize()

    // This triggers the optimized path using friend access to row_sets_
    auto c = patternUnion(a, b);
    
    EXPECT_TRUE(c.isFinalized());
    EXPECT_EQ(c.getNnz(), 3);  // Unique entries: (0,0), (0,1), (1,1)
    EXPECT_TRUE(c.hasEntry(0, 0));
    EXPECT_TRUE(c.hasEntry(0, 1));
    EXPECT_TRUE(c.hasEntry(1, 1));
}

TEST(SparsityPatternOptimizationTest, PatternIntersectionNonFinalized) {
    SparsityPattern a(3, 3);
    a.addEntry(0, 0);
    a.addEntry(0, 1);
    a.addEntry(1, 1);
    // Not calling finalize()

    SparsityPattern b(3, 3);
    b.addEntry(0, 1);
    b.addEntry(1, 1);
    b.addEntry(2, 2);
    // Not calling finalize()

    auto c = patternIntersection(a, b);
    
    EXPECT_TRUE(c.isFinalized());
    EXPECT_EQ(c.getNnz(), 2);  // Intersection: (0,1), (1,1)
    EXPECT_FALSE(c.hasEntry(0, 0));
    EXPECT_TRUE(c.hasEntry(0, 1));
    EXPECT_TRUE(c.hasEntry(1, 1));
    EXPECT_FALSE(c.hasEntry(2, 2));
}

TEST(SparsityPatternOptimizationTest, SymmetrizeNonFinalized) {
    SparsityPattern pattern(3, 3);
    pattern.addEntry(0, 1);
    pattern.addEntry(1, 2);
    // Not calling finalize()

    auto symmetric = symmetrize(pattern);
    
    EXPECT_TRUE(symmetric.isFinalized());
    EXPECT_EQ(symmetric.getNnz(), 4);
    EXPECT_TRUE(symmetric.hasEntry(0, 1));
    EXPECT_TRUE(symmetric.hasEntry(1, 0));
    EXPECT_TRUE(symmetric.hasEntry(1, 2));
    EXPECT_TRUE(symmetric.hasEntry(2, 1));
}

TEST(SparsityPatternOptimizationTest, MixedStateUnion) {
    SparsityPattern a(3, 3);
    a.addEntry(0, 0);
    a.finalize(); // a is Finalized

    SparsityPattern b(3, 3);
    b.addEntry(1, 1);
    // b is Building

    auto c = patternUnion(a, b);
    
    EXPECT_TRUE(c.hasEntry(0, 0));
    EXPECT_TRUE(c.hasEntry(1, 1));
}
