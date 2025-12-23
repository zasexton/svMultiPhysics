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
#include "Sparsity/GraphSparsity.h"
#include "Sparsity/SparsityPattern.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>

using namespace svmp::FE;
using namespace svmp::FE::sparsity;

// ============================================================================
// Helper to create symmetric test patterns
// ============================================================================

SparsityPattern createSymmetricPattern(GlobalIndex n, const std::vector<std::pair<GlobalIndex, GlobalIndex>>& edges) {
    SparsityPattern pattern(n, n);
    for (const auto& [i, j] : edges) {
        pattern.addEntry(i, j);
        pattern.addEntry(j, i);  // Symmetric
    }
    // Add diagonal
    for (GlobalIndex i = 0; i < n; ++i) {
        pattern.addEntry(i, i);
    }
    pattern.finalize();
    return pattern;
}

// Create a path graph: 0-1-2-...-n-1
SparsityPattern createPathGraph(GlobalIndex n) {
    std::vector<std::pair<GlobalIndex, GlobalIndex>> edges;
    for (GlobalIndex i = 0; i < n - 1; ++i) {
        edges.push_back({i, i + 1});
    }
    return createSymmetricPattern(n, edges);
}

// Create a star graph: 0 connected to all others
SparsityPattern createStarGraph(GlobalIndex n) {
    std::vector<std::pair<GlobalIndex, GlobalIndex>> edges;
    for (GlobalIndex i = 1; i < n; ++i) {
        edges.push_back({0, i});
    }
    return createSymmetricPattern(n, edges);
}

// Create a complete graph
SparsityPattern createCompleteGraph(GlobalIndex n) {
    SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            pattern.addEntry(i, j);
        }
    }
    pattern.finalize();
    return pattern;
}

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(GraphSparsityTest, DefaultConstruction) {
    GraphSparsity graph;
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.numVertices(), 0);
}

TEST(GraphSparsityTest, ConstructFromPattern) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.numVertices(), 5);
    EXPECT_FALSE(graph.empty());
}

TEST(GraphSparsityTest, SetPattern) {
    GraphSparsity graph;
    auto pattern = createPathGraph(5);
    graph.setPattern(pattern);

    EXPECT_EQ(graph.numVertices(), 5);
}

// ============================================================================
// Graph Statistics Tests
// ============================================================================

TEST(GraphSparsityTest, ComputeStatsPathGraph) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto stats = graph.computeStats();

    EXPECT_EQ(stats.n_vertices, 5);
    EXPECT_EQ(stats.bandwidth, 1);  // Path graph has bandwidth 1
    EXPECT_TRUE(stats.is_connected);
    EXPECT_EQ(stats.n_components, 1);
}

TEST(GraphSparsityTest, ComputeStatsStarGraph) {
    auto pattern = createStarGraph(5);
    GraphSparsity graph(pattern);

    auto stats = graph.computeStats();

    EXPECT_EQ(stats.n_vertices, 5);
    EXPECT_EQ(stats.max_degree, 4);  // Center vertex
    EXPECT_EQ(stats.min_degree, 1);  // Leaf vertices (just connected to center)
    EXPECT_TRUE(stats.is_connected);
}

TEST(GraphSparsityTest, ComputeStatsCompleteGraph) {
    auto pattern = createCompleteGraph(4);
    GraphSparsity graph(pattern);

    auto stats = graph.computeStats();

    EXPECT_EQ(stats.n_vertices, 4);
    EXPECT_EQ(stats.max_degree, 3);
    EXPECT_EQ(stats.min_degree, 3);
}

TEST(GraphSparsityTest, ComputeBandwidth) {
    auto pattern = createPathGraph(10);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.computeBandwidth(), 1);
}

TEST(GraphSparsityTest, ComputeBandwidthLarge) {
    // Arrow pattern has large bandwidth
    SparsityPattern pattern(10, 10);
    for (GlobalIndex i = 0; i < 10; ++i) {
        pattern.addEntry(0, i);
        pattern.addEntry(i, 0);
        pattern.addEntry(i, i);
    }
    pattern.finalize();

    GraphSparsity graph(pattern);
    EXPECT_EQ(graph.computeBandwidth(), 9);
}

TEST(GraphSparsityTest, ComputeProfile) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    GlobalIndex profile = graph.computeProfile();
    EXPECT_GT(profile, 0);
}

TEST(GraphSparsityTest, ComputeDegrees) {
    auto pattern = createStarGraph(5);
    GraphSparsity graph(pattern);

    auto degrees = graph.computeDegrees();

    EXPECT_EQ(degrees.size(), 5);
    EXPECT_EQ(degrees[0], 4);  // Center connected to all
    for (GlobalIndex i = 1; i < 5; ++i) {
        EXPECT_EQ(degrees[i], 1);  // Leaves connected only to center
    }
}

TEST(GraphSparsityTest, GetDegree) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.getDegree(0), 1);  // End vertex
    EXPECT_EQ(graph.getDegree(2), 2);  // Middle vertex
    EXPECT_EQ(graph.getDegree(4), 1);  // End vertex
}

TEST(GraphSparsityTest, GetNeighbors) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto neighbors = graph.getNeighbors(2);
    std::set<GlobalIndex> neighbor_set(neighbors.begin(), neighbors.end());

    EXPECT_EQ(neighbor_set.size(), 2);
    EXPECT_TRUE(neighbor_set.count(1) > 0);
    EXPECT_TRUE(neighbor_set.count(3) > 0);
}

// ============================================================================
// Graph Coloring Tests
// ============================================================================

TEST(GraphSparsityTest, GreedyColoringPath) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto coloring = graph.greedyColoring();

    EXPECT_TRUE(coloring.is_valid);
    EXPECT_EQ(coloring.num_colors, 2);  // Path graph is 2-colorable
    EXPECT_EQ(coloring.colors.size(), 5);

    // Verify coloring is valid
    EXPECT_TRUE(graph.verifyColoring(coloring.colors));
}

TEST(GraphSparsityTest, GreedyColoringComplete) {
    auto pattern = createCompleteGraph(4);
    GraphSparsity graph(pattern);

    auto coloring = graph.greedyColoring();

    EXPECT_TRUE(coloring.is_valid);
    EXPECT_EQ(coloring.num_colors, 4);  // Complete graph needs n colors

    EXPECT_TRUE(graph.verifyColoring(coloring.colors));
}

TEST(GraphSparsityTest, GreedyColoringStar) {
    auto pattern = createStarGraph(5);
    GraphSparsity graph(pattern);

    auto coloring = graph.greedyColoring();

    EXPECT_TRUE(coloring.is_valid);
    EXPECT_EQ(coloring.num_colors, 2);  // Star graph is 2-colorable

    EXPECT_TRUE(graph.verifyColoring(coloring.colors));
}

TEST(GraphSparsityTest, DegreeBasedColoring) {
    auto pattern = createPathGraph(10);
    GraphSparsity graph(pattern);

    auto coloring = graph.degreeBasedColoring(true);

    EXPECT_TRUE(coloring.is_valid);
    EXPECT_LE(coloring.num_colors, 3);

    EXPECT_TRUE(graph.verifyColoring(coloring.colors));
}

TEST(GraphSparsityTest, GetVerticesOfColor) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto coloring = graph.greedyColoring();

    for (GlobalIndex c = 0; c < coloring.num_colors; ++c) {
        auto vertices = coloring.getVerticesOfColor(c);
        EXPECT_FALSE(vertices.empty());

        // Verify all returned vertices have correct color
        for (GlobalIndex v : vertices) {
            EXPECT_EQ(coloring.colors[v], c);
        }
    }
}

TEST(GraphSparsityTest, VerifyInvalidColoring) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    // Create invalid coloring: adjacent vertices same color
    std::vector<GlobalIndex> bad_colors = {0, 0, 0, 0, 0};

    EXPECT_FALSE(graph.verifyColoring(bad_colors));
}

// ============================================================================
// Reordering Tests
// ============================================================================

TEST(GraphSparsityTest, CuthillMcKee) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto perm = graph.cuthillMcKee();

    EXPECT_EQ(perm.size(), 5);

    // Verify permutation is valid
    std::vector<GlobalIndex> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    std::vector<GlobalIndex> expected = {0, 1, 2, 3, 4};
    EXPECT_EQ(sorted_perm, expected);
}

TEST(GraphSparsityTest, ReverseCuthillMcKee) {
    auto pattern = createPathGraph(10);
    GraphSparsity graph(pattern);

    auto perm = graph.reverseCuthillMcKee();

    EXPECT_EQ(perm.size(), 10);

    // Verify valid permutation
    std::vector<GlobalIndex> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (GlobalIndex i = 0; i < 10; ++i) {
        EXPECT_EQ(sorted_perm[i], i);
    }
}

TEST(GraphSparsityTest, RCMReducesBandwidth) {
    // Create pattern with poor ordering
    SparsityPattern pattern(10, 10);
    // Connect 0-9, 1-8, 2-7, etc. (anti-diagonal connections)
    for (GlobalIndex i = 0; i < 5; ++i) {
        pattern.addEntry(i, 9 - i);
        pattern.addEntry(9 - i, i);
        pattern.addEntry(i, i);
        pattern.addEntry(9 - i, 9 - i);
    }
    pattern.finalize();

    GraphSparsity graph(pattern);
    GlobalIndex original_bw = graph.computeBandwidth();

    auto perm = graph.reverseCuthillMcKee();
    auto reordered = pattern.permute(perm, perm);

    GraphSparsity reordered_graph(reordered);
    GlobalIndex new_bw = reordered_graph.computeBandwidth();

    EXPECT_LE(new_bw, original_bw);
}

TEST(GraphSparsityTest, FindPseudoPeripheral) {
    auto pattern = createPathGraph(10);
    GraphSparsity graph(pattern);

    GlobalIndex peripheral = graph.findPseudoPeripheral();

    // For path graph, should be one of the endpoints
    EXPECT_TRUE(peripheral == 0 || peripheral == 9);
}

TEST(GraphSparsityTest, NaturalOrdering) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto perm = graph.naturalOrdering();

    std::vector<GlobalIndex> expected = {0, 1, 2, 3, 4};
    EXPECT_EQ(perm, expected);
}

TEST(GraphSparsityTest, ApproximateMinimumDegree) {
    auto pattern = createCompleteGraph(5);
    GraphSparsity graph(pattern);

    auto perm = graph.approximateMinimumDegree();

    EXPECT_EQ(perm.size(), 5);

    // Verify valid permutation
    std::vector<GlobalIndex> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (GlobalIndex i = 0; i < 5; ++i) {
        EXPECT_EQ(sorted_perm[i], i);
    }
}

// ============================================================================
// Structural Analysis Tests
// ============================================================================

TEST(GraphSparsityTest, ConnectedComponents) {
    // Create two disconnected paths
    SparsityPattern pattern(6, 6);
    // Path 0-1-2
    pattern.addEntry(0, 1); pattern.addEntry(1, 0);
    pattern.addEntry(1, 2); pattern.addEntry(2, 1);
    // Path 3-4-5
    pattern.addEntry(3, 4); pattern.addEntry(4, 3);
    pattern.addEntry(4, 5); pattern.addEntry(5, 4);
    // Diagonal
    for (GlobalIndex i = 0; i < 6; ++i) pattern.addEntry(i, i);
    pattern.finalize();

    GraphSparsity graph(pattern);
    auto components = graph.computeConnectedComponents();

    EXPECT_EQ(components.num_components, 2);
    EXPECT_EQ(components.component_sizes.size(), 2);
    EXPECT_EQ(components.component_sizes[0], 3);
    EXPECT_EQ(components.component_sizes[1], 3);
}

TEST(GraphSparsityTest, IsConnected) {
    auto connected = createPathGraph(5);
    GraphSparsity graph1(connected);
    EXPECT_TRUE(graph1.isConnected());

    // Disconnected graph
    SparsityPattern disconnected(4, 4);
    disconnected.addEntry(0, 1); disconnected.addEntry(1, 0);
    disconnected.addEntry(2, 3); disconnected.addEntry(3, 2);
    for (GlobalIndex i = 0; i < 4; ++i) disconnected.addEntry(i, i);
    disconnected.finalize();

    GraphSparsity graph2(disconnected);
    EXPECT_FALSE(graph2.isConnected());
}

TEST(GraphSparsityTest, ComputeLevelSets) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto levels = graph.computeLevelSets(0);

    EXPECT_EQ(levels.num_levels, 5);  // Path from 0: each vertex is new level
    EXPECT_EQ(levels.root, 0);
    EXPECT_EQ(levels.levels[0], 0);
    EXPECT_EQ(levels.levels[1], 1);
    EXPECT_EQ(levels.levels[4], 4);
}

TEST(GraphSparsityTest, ComputeLevelSetsFromMiddle) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto levels = graph.computeLevelSets(2);

    EXPECT_EQ(levels.num_levels, 3);  // Max distance is 2 (to 0 or 4)
    EXPECT_EQ(levels.levels[2], 0);
    EXPECT_EQ(levels.levels[1], 1);
    EXPECT_EQ(levels.levels[3], 1);
    EXPECT_EQ(levels.levels[0], 2);
    EXPECT_EQ(levels.levels[4], 2);
}

TEST(GraphSparsityTest, ComputeDiameter) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.computeDiameter(), 4);  // Distance 0 to 4
}

TEST(GraphSparsityTest, ComputeDiameterStar) {
    auto pattern = createStarGraph(5);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.computeDiameter(), 2);  // Any leaf to any other leaf through center
}

TEST(GraphSparsityTest, ComputeEccentricity) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.computeEccentricity(0), 4);  // End vertex
    EXPECT_EQ(graph.computeEccentricity(2), 2);  // Middle vertex
    EXPECT_EQ(graph.computeEccentricity(4), 4);  // End vertex
}

// ============================================================================
// Fill-in Analysis Tests
// ============================================================================

TEST(GraphSparsityTest, PredictCholeskyFillIn) {
    auto pattern = createPathGraph(5);
    GraphSparsity graph(pattern);

    auto prediction = graph.predictCholeskyFillIn();

    // Path graph should have minimal fill-in since it's already well-ordered
    EXPECT_EQ(prediction.original_nnz, pattern.getNnz());
    // Note: total_factor_nnz for lower/upper triangle only, original_nnz is full symmetric pattern
    // For a symmetric pattern, the Cholesky factor has approximately half the entries (lower or upper)
    // The prediction is for the factor, which may be less than the full pattern NNZ
    EXPECT_GT(prediction.total_factor_nnz, 0);
}

TEST(GraphSparsityTest, PredictLUFillIn) {
    auto pattern = createCompleteGraph(4);
    GraphSparsity graph(pattern);

    auto prediction = graph.predictLUFillIn();

    // Complete graph has no fill-in
    EXPECT_EQ(prediction.predicted_fill, 0);
    EXPECT_NEAR(prediction.fill_ratio, 1.0, 1e-10);
}

TEST(GraphSparsityTest, SymbolicCholesky) {
    auto pattern = createPathGraph(4);
    GraphSparsity graph(pattern);

    auto L = graph.symbolicCholesky();

    EXPECT_EQ(L.numRows(), 4);
    EXPECT_EQ(L.numCols(), 4);

    // Lower triangular
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_TRUE(L.hasEntry(i, i));  // Diagonal
        for (GlobalIndex j = i + 1; j < 4; ++j) {
            EXPECT_FALSE(L.hasEntry(i, j));  // No upper entries
        }
    }
}

TEST(GraphSparsityTest, SymbolicLU) {
    auto pattern = createPathGraph(4);
    GraphSparsity graph(pattern);

    auto [L, U] = graph.symbolicLU();

    EXPECT_EQ(L.numRows(), 4);
    EXPECT_EQ(U.numRows(), 4);

    // L should be lower triangular, U should be upper triangular
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_TRUE(L.hasEntry(i, i));
        EXPECT_TRUE(U.hasEntry(i, i));
    }
}

// ============================================================================
// Free Function Tests
// ============================================================================

TEST(GraphSparsityTest, ComputePatternBandwidth) {
    auto pattern = createPathGraph(10);
    EXPECT_EQ(computePatternBandwidth(pattern), 1);
}

TEST(GraphSparsityTest, ComputePatternProfile) {
    auto pattern = createPathGraph(5);
    GlobalIndex profile = computePatternProfile(pattern);
    EXPECT_GT(profile, 0);
}

TEST(GraphSparsityTest, ApplyRCM) {
    // Arrow pattern
    SparsityPattern pattern(5, 5);
    for (GlobalIndex i = 0; i < 5; ++i) {
        pattern.addEntry(0, i);
        pattern.addEntry(i, 0);
        pattern.addEntry(i, i);
    }
    pattern.finalize();

    auto reordered = applyRCM(pattern);

    EXPECT_EQ(reordered.numRows(), 5);
    EXPECT_EQ(reordered.getNnz(), pattern.getNnz());

    // Bandwidth should not increase
    EXPECT_LE(computePatternBandwidth(reordered), computePatternBandwidth(pattern));
}

TEST(GraphSparsityTest, GetRCMPermutation) {
    auto pattern = createPathGraph(5);
    auto perm = getRCMPermutation(pattern);

    EXPECT_EQ(perm.size(), 5);
}

TEST(GraphSparsityTest, ColorPattern) {
    auto pattern = createPathGraph(5);
    auto coloring = colorPattern(pattern);

    EXPECT_TRUE(coloring.is_valid);
    EXPECT_EQ(coloring.num_colors, 2);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST(GraphSparsityTest, EmptyGraph) {
    SparsityPattern pattern(0, 0);
    pattern.finalize();

    GraphSparsity graph(pattern);
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.numVertices(), 0);
}

TEST(GraphSparsityTest, SingleVertex) {
    SparsityPattern pattern(1, 1);
    pattern.addEntry(0, 0);
    pattern.finalize();

    GraphSparsity graph(pattern);

    EXPECT_EQ(graph.numVertices(), 1);
    EXPECT_TRUE(graph.isConnected());
    EXPECT_EQ(graph.computeBandwidth(), 0);
    EXPECT_EQ(graph.computeDiameter(), 0);
}

TEST(GraphSparsityTest, DiagonalOnly) {
    SparsityPattern pattern(5, 5);
    for (GlobalIndex i = 0; i < 5; ++i) {
        pattern.addEntry(i, i);
    }
    pattern.finalize();

    GraphSparsity graph(pattern);

    auto components = graph.computeConnectedComponents();
    EXPECT_EQ(components.num_components, 5);  // Each vertex isolated
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST(GraphSparsityTest, DeterministicColoring) {
    auto create_and_color = []() {
        auto pattern = createPathGraph(10);
        GraphSparsity graph(pattern);
        return graph.greedyColoring();
    };

    auto c1 = create_and_color();
    auto c2 = create_and_color();

    EXPECT_EQ(c1.num_colors, c2.num_colors);
    EXPECT_EQ(c1.colors, c2.colors);
}

TEST(GraphSparsityTest, DeterministicRCM) {
    auto create_and_rcm = []() {
        auto pattern = createCompleteGraph(5);
        GraphSparsity graph(pattern);
        return graph.reverseCuthillMcKee();
    };

    auto p1 = create_and_rcm();
    auto p2 = create_and_rcm();

    EXPECT_EQ(p1, p2);
}
