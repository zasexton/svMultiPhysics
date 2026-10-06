/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_FSILS_DIRECT_SOLVE_TEST_UTILS_H
#define SVMP_FE_TESTS_FSILS_DIRECT_SOLVE_TEST_UTILS_H

// Block chain operators for the FSILS gathered direct solve tests: n_nodes
// nodes with kChainDof components, nodal blocks coupling each node with its
// two neighbours, row magnitudes varying by node, and an exactly known
// solution that is zero on the Dirichlet DOFs.  The same builder produces a
// serial operator (SparsityPattern) or the owned rows of a contiguous
// partition (DistributedSparsityPattern with ghost rows).

#include "Assembly/GlobalSystemView.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace svmp::FE::backends::direct_test {

constexpr int kChainDof = 3;

struct ChainOptions {
    GlobalIndex n_nodes{29};
    Real scale{1.0};
    bool couple_10{false};          ///< make component pair (1,0) nonzero (zero otherwise)
    GlobalIndex zero_row{-1};       ///< scalar row whose entries are all zero (singular operator)
    std::vector<GlobalIndex> dirichlet{};
};

struct ChainSystem {
    std::unique_ptr<GenericMatrix> A;
    std::unique_ptr<GenericVector> b;
    std::unique_ptr<GenericVector> x;
    GlobalIndex first{0};  ///< first owned node
    GlobalIndex last{0};   ///< last owned node
};

inline Real rowMagnitude(GlobalIndex node)
{
    static constexpr Real kMag[3] = {1.0, 10.0, 40.0};
    return kMag[node % 3];
}

inline Real blockValue(const ChainOptions& o, GlobalIndex i, GlobalIndex j, int r, int c)
{
    static constexpr Real kDiag[9] = {6.0, 1.0, 0.5, 0.0, 6.0, 0.5, 0.5, 0.5, 1.0};
    static constexpr Real kLeft[9] = {-1.0, 0.2, 0.0, 0.0, -1.0, 0.2, -0.3, 0.2, -0.2};
    static constexpr Real kRight[9] = {-1.0, 0.1, 0.3, 0.0, -1.0, -0.2, 0.0, -0.2, -0.2};
    const Real* blk = (j == i) ? kDiag : (j < i ? kLeft : kRight);
    Real v = blk[r * kChainDof + c];
    if (r == 1 && c == 0) {
        v = o.couple_10 ? (j == i ? 0.4 : 0.15) : 0.0;
    }
    if (i * kChainDof + r == o.zero_row) {
        v = 0.0;
    }
    return o.scale * rowMagnitude(i) * v;
}

inline bool isDirichlet(const ChainOptions& o, GlobalIndex dof)
{
    return std::find(o.dirichlet.begin(), o.dirichlet.end(), dof) != o.dirichlet.end();
}

/// Exact solution (zero on the Dirichlet DOFs).
inline Real exactValue(const ChainOptions& o, GlobalIndex dof)
{
    if (isDirichlet(o, dof)) {
        return 0.0;
    }
    const auto node = dof / kChainDof;
    const auto comp = dof % kChainDof;
    return 1.0 + std::cos(0.37 * static_cast<Real>(node)) + 0.25 * static_cast<Real>(comp);
}

inline std::vector<GlobalIndex> chainNeighbours(GlobalIndex n_nodes, GlobalIndex i)
{
    std::vector<GlobalIndex> nb;
    for (GlobalIndex j = std::max<GlobalIndex>(0, i - 1); j <= std::min(n_nodes - 1, i + 1); ++j) {
        nb.push_back(j);
    }
    return nb;
}

/// Contiguous node partition: [first, last] owned by `rank` of `size`.
inline void chainPartition(GlobalIndex n_nodes, int rank, int size, GlobalIndex& first, GlobalIndex& last)
{
    const GlobalIndex base = n_nodes / size;
    const GlobalIndex rem = n_nodes % size;
    first = rank * base + std::min<GlobalIndex>(rank, rem);
    last = first + base + (rank < rem ? 1 : 0) - 1;
}

inline void assembleChain(const ChainOptions& o, ChainSystem& sys)
{
    auto viewA = sys.A->createAssemblyView();
    auto viewb = sys.b->createAssemblyView();
    viewA->beginAssemblyPhase();
    viewb->beginAssemblyPhase();
    for (GlobalIndex i = sys.first; i <= sys.last; ++i) {
        const auto nb = chainNeighbours(o.n_nodes, i);
        std::vector<GlobalIndex> rows;
        std::vector<GlobalIndex> cols;
        for (int r = 0; r < kChainDof; ++r) {
            rows.push_back(i * kChainDof + r);
        }
        for (const auto j : nb) {
            for (int c = 0; c < kChainDof; ++c) {
                cols.push_back(j * kChainDof + c);
            }
        }
        std::vector<Real> values(rows.size() * cols.size(), 0.0);
        std::vector<Real> rhs(kChainDof, 0.0);
        for (int r = 0; r < kChainDof; ++r) {
            for (std::size_t k = 0; k < nb.size(); ++k) {
                for (int c = 0; c < kChainDof; ++c) {
                    const Real v = blockValue(o, i, nb[k], r, c);
                    values[static_cast<std::size_t>(r) * cols.size() + k * kChainDof + static_cast<std::size_t>(c)] = v;
                    rhs[static_cast<std::size_t>(r)] += v * exactValue(o, nb[k] * kChainDof + c);
                }
            }
            if (isDirichlet(o, i * kChainDof + r)) {
                rhs[static_cast<std::size_t>(r)] = 7.0;  // must be ignored by the solve
            }
        }
        viewA->addMatrixEntries(rows, cols, values, assembly::AddMode::Insert);
        viewb->addVectorEntries(rows, rhs, assembly::AddMode::Insert);
    }
    viewA->finalizeAssembly();
    viewb->finalizeAssembly();
    sys.A->finalizeAssembly();
}

/// Serial operator on the factory's communicator (which must have one rank).
inline ChainSystem buildSerialChain(const FsilsFactory& factory, const ChainOptions& o)
{
    const GlobalIndex n = o.n_nodes * kChainDof;
    sparsity::SparsityPattern pattern(n, n);
    for (GlobalIndex i = 0; i < o.n_nodes; ++i) {
        for (int r = 0; r < kChainDof; ++r) {
            for (const auto j : chainNeighbours(o.n_nodes, i)) {
                for (int c = 0; c < kChainDof; ++c) {
                    pattern.addEntry(i * kChainDof + r, j * kChainDof + c);
                }
            }
        }
    }
    pattern.finalize();
    ChainSystem sys;
    sys.first = 0;
    sys.last = o.n_nodes - 1;
    sys.A = factory.createMatrix(pattern);
    sys.b = factory.createVector(n);
    sys.x = factory.createVector(n);
    assembleChain(o, sys);
    return sys;
}

/// Owned rows of a contiguous partition over the factory's communicator.
inline ChainSystem buildDistributedChain(const FsilsFactory& factory, const ChainOptions& o, int rank, int size)
{
    const GlobalIndex n = o.n_nodes * kChainDof;
    ChainSystem sys;
    chainPartition(o.n_nodes, rank, size, sys.first, sys.last);
    const sparsity::IndexRange owned{sys.first * kChainDof, (sys.last + 1) * kChainDof};
    sparsity::DistributedSparsityPattern pattern(owned, owned, n, n);
    for (GlobalIndex i = sys.first; i <= sys.last; ++i) {
        for (int r = 0; r < kChainDof; ++r) {
            for (const auto j : chainNeighbours(o.n_nodes, i)) {
                for (int c = 0; c < kChainDof; ++c) {
                    pattern.addEntry(i * kChainDof + r, j * kChainDof + c);
                }
            }
        }
    }
    pattern.ensureDiagonal();
    pattern.finalize();

    std::vector<GlobalIndex> ghost_rows;
    std::vector<GlobalIndex> ghost_ptr{0};
    std::vector<GlobalIndex> ghost_cols;
    auto addGhost = [&](GlobalIndex ghost, GlobalIndex a, GlobalIndex b) {
        for (int r = 0; r < kChainDof; ++r) {
            ghost_rows.push_back(ghost * kChainDof + r);
            for (const auto j : {a, b}) {
                for (int c = 0; c < kChainDof; ++c) {
                    ghost_cols.push_back(j * kChainDof + c);
                }
            }
            ghost_ptr.push_back(static_cast<GlobalIndex>(ghost_cols.size()));
        }
    };
    if (sys.first > 0) {
        addGhost(sys.first - 1, sys.first - 1, sys.first);
    }
    if (sys.last < o.n_nodes - 1) {
        addGhost(sys.last + 1, sys.last, sys.last + 1);
    }
    if (!ghost_rows.empty()) {
        pattern.setGhostRows(std::move(ghost_rows), std::move(ghost_ptr), std::move(ghost_cols));
    }
    sys.A = factory.createMatrix(pattern);
    sys.b = factory.createVector(n);
    sys.x = factory.createVector(n);
    assembleChain(o, sys);
    return sys;
}

} // namespace svmp::FE::backends::direct_test

#endif // SVMP_FE_TESTS_FSILS_DIRECT_SOLVE_TEST_UTILS_H
