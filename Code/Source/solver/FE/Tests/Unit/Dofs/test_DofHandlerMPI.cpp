/**
 * @file test_DofHandlerMPI.cpp
 * @brief MPI unit tests for distributed DOF ownership/numbering and ghost exchange
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofHandler.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Dofs/GhostDofManager.h"

#include <mpi.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <span>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::dofs::DofDistributionOptions;
using svmp::FE::dofs::DofHandler;
using svmp::FE::dofs::DofLayoutInfo;
using svmp::FE::dofs::GlobalNumberingMode;
using svmp::FE::dofs::MeshTopologyInfo;
using svmp::FE::dofs::OwnershipStrategy;

namespace {

int mpiRank(MPI_Comm comm) {
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

int mpiSize(MPI_Comm comm) {
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}

svmp::FE::dofs::gid_t sparseVertexGid(svmp::FE::dofs::gid_t logical_gid) {
    // Monotone, intentionally sparse mapping to test dense compaction.
    return logical_gid * static_cast<svmp::FE::dofs::gid_t>(1000) + static_cast<svmp::FE::dofs::gid_t>(7);
}

void remapVertexGidsSparse(MeshTopologyInfo& topo) {
    for (auto& gid : topo.vertex_gids) {
        gid = sparseVertexGid(gid);
    }
}

std::vector<GlobalIndex> allgatherGlobalIndices(std::span<const GlobalIndex> local, MPI_Comm comm) {
    const int size = mpiSize(comm);
    std::vector<GlobalIndex> gathered(static_cast<std::size_t>(size) * local.size(), GlobalIndex{-1});
    MPI_Allgather(local.data(),
                  static_cast<int>(local.size()),
                  MPI_INT64_T,
                  gathered.data(),
                  static_cast<int>(local.size()),
                  MPI_INT64_T,
                  comm);
    return gathered;
}

template <typename T>
std::vector<T> allgathervInt64(std::span<const T> local, MPI_Comm comm) {
    static_assert(sizeof(T) == sizeof(std::int64_t), "allgathervInt64 expects 64-bit integer-like values");
    const int size = mpiSize(comm);

    const int local_n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(size) + 1u, 0);
    for (int r = 0; r < size; ++r) {
        displs[static_cast<std::size_t>(r) + 1u] =
            displs[static_cast<std::size_t>(r)] + counts[static_cast<std::size_t>(r)];
    }
    const int total = displs[static_cast<std::size_t>(size)];
    std::vector<T> gathered(static_cast<std::size_t>(total));

    MPI_Allgatherv(local.data(),
                   local_n,
                   MPI_INT64_T,
                   gathered.data(),
                   counts.data(),
                   displs.data(),
                   MPI_INT64_T,
                   comm);

    return gathered;
}

MeshTopologyInfo makeTwoRankTriangleP1(int rank, GlobalIndex gid_cell) {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.dim = 2;

    // Rank 0: (0,1,2). Rank 1: (0,1,3).
    if (rank == 0) {
        topo.vertex_gids = {0, 1, 2};
    } else {
        topo.vertex_gids = {0, 1, 3};
    }
    topo.n_vertices = static_cast<GlobalIndex>(topo.vertex_gids.size());

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2}; // local vertex ids (0..2)

    topo.cell_gids = {static_cast<svmp::FE::dofs::gid_t>(gid_cell)};
    topo.cell_owner_ranks = {rank};
    topo.neighbor_ranks = {1 - rank};

    return topo;
}

MeshTopologyInfo makeTwoRankDisjointTrianglesP1(int rank, GlobalIndex n_cells, svmp::FE::dofs::gid_t gid_cell_base) {
    MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = n_cells;

    const GlobalIndex n_vertices = 3 * n_cells;
    topo.n_vertices = n_vertices;
    topo.vertex_gids.resize(static_cast<std::size_t>(n_vertices));
    for (GlobalIndex i = 0; i < n_vertices; ++i) {
        topo.vertex_gids[static_cast<std::size_t>(i)] = static_cast<svmp::FE::dofs::gid_t>(rank) *
                                                            static_cast<svmp::FE::dofs::gid_t>(n_vertices) +
                                                        static_cast<svmp::FE::dofs::gid_t>(i);
    }

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1u, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(n_cells) * 3u, 0);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        const auto sc = static_cast<std::size_t>(c);
        topo.cell2vertex_offsets[sc] = static_cast<GlobalIndex>(3u * sc);
        topo.cell2vertex_data[3u * sc + 0u] = 3 * c + 0;
        topo.cell2vertex_data[3u * sc + 1u] = 3 * c + 1;
        topo.cell2vertex_data[3u * sc + 2u] = 3 * c + 2;
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = static_cast<GlobalIndex>(topo.cell2vertex_data.size());

    topo.cell_gids.resize(static_cast<std::size_t>(n_cells));
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(n_cells), rank);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] =
            gid_cell_base + static_cast<svmp::FE::dofs::gid_t>(rank) * 1000000 + static_cast<svmp::FE::dofs::gid_t>(c);
    }

    topo.neighbor_ranks = {1 - rank};
    return topo;
}

MeshTopologyInfo makeTwoRankTriangleP2(int rank, GlobalIndex gid_cell) {
    MeshTopologyInfo topo = makeTwoRankTriangleP1(rank, gid_cell);

    topo.n_edges = 3;
    topo.cell2edge_offsets = {0, 3};
    topo.cell2edge_data = {0, 1, 2};
    topo.edge2vertex_data = {
        0, 1, // edge 0
        1, 2, // edge 1
        2, 0  // edge 2
    };

    return topo;
}

MeshTopologyInfo makeTwoRankTriangleP2Sparse(int rank, GlobalIndex gid_cell) {
    auto topo = makeTwoRankTriangleP2(rank, gid_cell);
    remapVertexGidsSparse(topo);
    return topo;
}

MeshTopologyInfo makeTwoRankLineDGWithGhostCells(int rank,
                                                svmp::FE::dofs::gid_t gid_cell0,
                                                svmp::FE::dofs::gid_t gid_cell1) {
    MeshTopologyInfo topo;
    topo.dim = 1;
    topo.n_cells = 2;
    topo.n_vertices = 0;

    topo.cell_gids = {gid_cell0, gid_cell1};
    topo.cell_owner_ranks = {0, 1}; // cell0 owned by rank0, cell1 owned by rank1
    topo.neighbor_ranks = {1 - rank};

    return topo;
}

MeshTopologyInfo makeFourRankTriangleP2(int rank, GlobalIndex gid_cell) {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.dim = 2;

    // Rank0: (0,1,2)
    // Rank1: (0,1,3)
    // Rank2: (0,2,4)
    // Rank3: (1,2,5)
    switch (rank) {
        case 0:
            topo.vertex_gids = {0, 1, 2};
            break;
        case 1:
            topo.vertex_gids = {0, 1, 3};
            break;
        case 2:
            topo.vertex_gids = {0, 2, 4};
            break;
        case 3:
            topo.vertex_gids = {1, 2, 5};
            break;
        default:
            topo.vertex_gids = {0, 1, 2};
            break;
    }
    topo.n_vertices = static_cast<GlobalIndex>(topo.vertex_gids.size());

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};

    topo.n_edges = 3;
    topo.cell2edge_offsets = {0, 3};
    topo.cell2edge_data = {0, 1, 2};
    topo.edge2vertex_data = {
        0, 1, // edge 0
        1, 2, // edge 1
        2, 0  // edge 2
    };

    topo.cell_gids = {static_cast<svmp::FE::dofs::gid_t>(gid_cell)};
    topo.cell_owner_ranks = {rank};

    switch (rank) {
        case 0:
            topo.neighbor_ranks = {1, 2, 3};
            break;
        case 1:
            topo.neighbor_ranks = {0, 2, 3};
            break;
        case 2:
            topo.neighbor_ranks = {0, 1, 3};
            break;
        case 3:
            topo.neighbor_ranks = {0, 1, 2};
            break;
        default:
            topo.neighbor_ranks.clear();
            break;
    }

    return topo;
}

MeshTopologyInfo makeFourRankTriangleP2Sparse(int rank, GlobalIndex gid_cell) {
    auto topo = makeFourRankTriangleP2(rank, gid_cell);
    remapVertexGidsSparse(topo);
    return topo;
}

std::optional<GlobalIndex> findLocalVertexByGid(const MeshTopologyInfo& topo, svmp::FE::dofs::gid_t gid) {
    for (std::size_t i = 0; i < topo.vertex_gids.size(); ++i) {
        if (topo.vertex_gids[i] == gid) {
            return static_cast<GlobalIndex>(i);
        }
    }
    return std::nullopt;
}

std::optional<GlobalIndex> findLocalEdgeByVertexGids(const MeshTopologyInfo& topo,
                                                     svmp::FE::dofs::gid_t gid_a,
                                                     svmp::FE::dofs::gid_t gid_b) {
    if (topo.n_edges <= 0 || topo.edge2vertex_data.size() < static_cast<std::size_t>(2 * topo.n_edges)) {
        return std::nullopt;
    }
    const auto lo = std::min(gid_a, gid_b);
    const auto hi = std::max(gid_a, gid_b);
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        const auto idx = static_cast<std::size_t>(2 * e);
        const auto v0 = topo.edge2vertex_data[idx];
        const auto v1 = topo.edge2vertex_data[idx + 1];
        const auto gv0 = topo.vertex_gids[static_cast<std::size_t>(v0)];
        const auto gv1 = topo.vertex_gids[static_cast<std::size_t>(v1)];
        if (std::min(gv0, gv1) == lo && std::max(gv0, gv1) == hi) {
            return e;
        }
    }
    return std::nullopt;
}

svmp::FE::dofs::gid_t globalEdgeGid(svmp::FE::dofs::gid_t gid_a, svmp::FE::dofs::gid_t gid_b) {
    const auto lo = std::min(gid_a, gid_b);
    const auto hi = std::max(gid_a, gid_b);

    // Global mesh edges for tests:
    // (0,1),(0,2),(1,2),(0,3),(1,3),(0,4),(2,4),(1,5),(2,5)
    if (lo == 0 && hi == 1) return 0;
    if (lo == 0 && hi == 2) return 1;
    if (lo == 1 && hi == 2) return 2;
    if (lo == 0 && hi == 3) return 3;
    if (lo == 1 && hi == 3) return 4;
    if (lo == 0 && hi == 4) return 5;
    if (lo == 2 && hi == 4) return 6;
    if (lo == 1 && hi == 5) return 7;
    if (lo == 2 && hi == 5) return 8;

    throw std::runtime_error("globalEdgeGid: unexpected edge vertex GIDs");
}

void setTriangleEdgeGidsFromVertices(MeshTopologyInfo& topo) {
    if (topo.n_edges <= 0 || topo.edge2vertex_data.size() < static_cast<std::size_t>(2 * topo.n_edges)) {
        throw std::runtime_error("setTriangleEdgeGidsFromVertices: missing edge2vertex_data");
    }
    topo.edge_gids.resize(static_cast<std::size_t>(topo.n_edges), svmp::FE::dofs::gid_t{-1});
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        const auto idx = static_cast<std::size_t>(2 * e);
        const auto v0 = topo.edge2vertex_data[idx];
        const auto v1 = topo.edge2vertex_data[idx + 1];
        const auto gv0 = topo.vertex_gids[static_cast<std::size_t>(v0)];
        const auto gv1 = topo.vertex_gids[static_cast<std::size_t>(v1)];
        topo.edge_gids[static_cast<std::size_t>(e)] = globalEdgeGid(gv0, gv1);
    }
}

MeshTopologyInfo makeTwoRankFourTriangleMeshP2(int rank) {
    MeshTopologyInfo topo;
    topo.dim = 2;

    struct Cell {
        std::array<svmp::FE::dofs::gid_t, 3> verts;
        svmp::FE::dofs::gid_t gid;
    };

    std::vector<Cell> cells;
    if (rank == 0) {
        cells.push_back(Cell{{0, 1, 2}, 100});
        cells.push_back(Cell{{0, 2, 4}, 102});
    } else {
        cells.push_back(Cell{{0, 1, 3}, 101});
        cells.push_back(Cell{{1, 2, 5}, 103});
    }

    topo.n_cells = static_cast<GlobalIndex>(cells.size());
    topo.cell_gids.reserve(cells.size());
    topo.cell_owner_ranks.assign(cells.size(), rank);
    topo.neighbor_ranks = {1 - rank};

    // Build local vertex list.
    std::vector<svmp::FE::dofs::gid_t> vset;
    vset.reserve(8);
    for (const auto& c : cells) {
        vset.insert(vset.end(), c.verts.begin(), c.verts.end());
        topo.cell_gids.push_back(c.gid);
    }
    std::sort(vset.begin(), vset.end());
    vset.erase(std::unique(vset.begin(), vset.end()), vset.end());

    topo.vertex_gids = vset;
    topo.n_vertices = static_cast<GlobalIndex>(topo.vertex_gids.size());

    std::unordered_map<svmp::FE::dofs::gid_t, GlobalIndex> gid_to_lvid;
    gid_to_lvid.reserve(topo.vertex_gids.size());
    for (std::size_t i = 0; i < topo.vertex_gids.size(); ++i) {
        gid_to_lvid.emplace(topo.vertex_gids[i], static_cast<GlobalIndex>(i));
    }

    // Cell-to-vertex connectivity.
    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(topo.n_cells) + 1u);
    topo.cell2vertex_data.clear();
    topo.cell2vertex_data.reserve(static_cast<std::size_t>(topo.n_cells) * 3u);
    topo.cell2vertex_offsets[0] = 0;
    for (std::size_t c = 0; c < cells.size(); ++c) {
        for (std::size_t j = 0; j < 3u; ++j) {
            topo.cell2vertex_data.push_back(gid_to_lvid.at(cells[c].verts[j]));
        }
        topo.cell2vertex_offsets[c + 1] = static_cast<GlobalIndex>(topo.cell2vertex_data.size());
    }

    // Build edges and cell-to-edge connectivity (triangles).
    struct EdgeKey {
        svmp::FE::dofs::gid_t a;
        svmp::FE::dofs::gid_t b;
        bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey& k) const noexcept {
            const auto h1 = static_cast<std::size_t>(k.a);
            const auto h2 = static_cast<std::size_t>(k.b);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_to_id;
    std::vector<std::pair<GlobalIndex, GlobalIndex>> edges;

    topo.cell2edge_offsets.resize(static_cast<std::size_t>(topo.n_cells) + 1u);
    topo.cell2edge_data.clear();
    topo.cell2edge_data.reserve(static_cast<std::size_t>(topo.n_cells) * 3u);
    topo.cell2edge_offsets[0] = 0;

    auto edge_local_id = [&](svmp::FE::dofs::gid_t ga, svmp::FE::dofs::gid_t gb) -> GlobalIndex {
        const auto lo = std::min(ga, gb);
        const auto hi = std::max(ga, gb);
        EdgeKey key{lo, hi};
        auto it = edge_to_id.find(key);
        if (it != edge_to_id.end()) {
            return it->second;
        }
        const GlobalIndex id = static_cast<GlobalIndex>(edges.size());
        edge_to_id.emplace(key, id);
        edges.emplace_back(gid_to_lvid.at(lo), gid_to_lvid.at(hi));
        return id;
    };

    for (std::size_t c = 0; c < cells.size(); ++c) {
        const auto a = cells[c].verts[0];
        const auto b = cells[c].verts[1];
        const auto d = cells[c].verts[2];
        topo.cell2edge_data.push_back(edge_local_id(a, b));
        topo.cell2edge_data.push_back(edge_local_id(b, d));
        topo.cell2edge_data.push_back(edge_local_id(d, a));
        topo.cell2edge_offsets[c + 1] = static_cast<GlobalIndex>(topo.cell2edge_data.size());
    }

    topo.n_edges = static_cast<GlobalIndex>(edges.size());
    topo.edge2vertex_data.resize(static_cast<std::size_t>(2) * static_cast<std::size_t>(topo.n_edges));
    topo.edge_gids.resize(static_cast<std::size_t>(topo.n_edges), svmp::FE::dofs::gid_t{-1});
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        const auto se = static_cast<std::size_t>(e);
        const auto [lv0, lv1] = edges[se];
        topo.edge2vertex_data[2 * se + 0] = lv0;
        topo.edge2vertex_data[2 * se + 1] = lv1;
        topo.edge_gids[se] = globalEdgeGid(topo.vertex_gids[static_cast<std::size_t>(lv0)],
                                           topo.vertex_gids[static_cast<std::size_t>(lv1)]);
    }

    return topo;
}

MeshTopologyInfo makeTwoRankFourTriangleMeshP2Sparse(int rank) {
    auto topo = makeTwoRankFourTriangleMeshP2(rank);
    remapVertexGidsSparse(topo);
    return topo;
}

} // namespace

TEST(DofHandlerMPI, DistributedCG_P1_TwoRank_GlobalIDsAndOwnershipLowestRank) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP1(rank, /*gid_cell=*/10 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;
    opts.validate_parallel = true;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 4);
    EXPECT_EQ(dh.getNumLocalDofs(), (rank == 0) ? 3 : 1);

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);

    const auto lv0 = findLocalVertexByGid(topo, 0);
    const auto lv1 = findLocalVertexByGid(topo, 1);
    ASSERT_TRUE(lv0.has_value());
    ASSERT_TRUE(lv1.has_value());

    const auto v0_dofs = entity->getVertexDofs(*lv0);
    const auto v1_dofs = entity->getVertexDofs(*lv1);
    ASSERT_EQ(v0_dofs.size(), 1u);
    ASSERT_EQ(v1_dofs.size(), 1u);

    const std::array<GlobalIndex, 2> local_ids = {v0_dofs[0], v1_dofs[0]};
    const auto gathered = allgatherGlobalIndices(local_ids, comm);
    ASSERT_EQ(gathered.size(), 4u);
    EXPECT_EQ(gathered[0], gathered[2]); // v0 equal across ranks
    EXPECT_EQ(gathered[1], gathered[3]); // v1 equal across ranks

    EXPECT_EQ(dh.getDofMap().getDofOwner(v0_dofs[0]), 0);
    EXPECT_EQ(dh.getDofMap().getDofOwner(v1_dofs[0]), 0);

    const auto& part = dh.getPartition();
    if (rank == 0) {
        EXPECT_TRUE(part.isOwned(v0_dofs[0]));
        EXPECT_TRUE(part.isOwned(v1_dofs[0]));
        EXPECT_EQ(part.ghostSize(), 0);
    } else {
        EXPECT_TRUE(part.isGhost(v0_dofs[0]));
        EXPECT_TRUE(part.isGhost(v1_dofs[0]));
        EXPECT_EQ(part.ghostSize(), 2);
    }
}

TEST(DofHandlerMPI, ValidateParallel_DetectsMismatchWhenOwnershipOptionsDiffer) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP1(rank, /*gid_cell=*/70 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.validate_parallel = true;
    opts.ownership = (rank == 0) ? OwnershipStrategy::LowestRank : OwnershipStrategy::HighestRank;

    bool threw = false;
    try {
        DofHandler dh;
        dh.distributeDofs(topo, layout, opts);
        dh.finalize();
    } catch (const std::exception&) {
        threw = true;
    }

    const int local = threw ? 1 : 0;
    int global_sum = 0;
    MPI_Allreduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, comm);
    EXPECT_EQ(global_sum, size);
}

TEST(DofHandlerMPI, DistributedCG_P2_TwoRank_SharedEdgeAndGhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/20 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 9);
    EXPECT_EQ(dh.getNumLocalDofs(), (rank == 0) ? 6 : 3);

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);

    const auto le = findLocalEdgeByVertexGids(topo, /*gid_a=*/0, /*gid_b=*/1);
    ASSERT_TRUE(le.has_value());
    const auto edofs = entity->getEdgeDofs(*le);
    ASSERT_EQ(edofs.size(), 1u);

    const GlobalIndex edge_dof = edofs[0];
    const auto gathered = allgatherGlobalIndices(std::span<const GlobalIndex>(&edge_dof, 1), comm);
    ASSERT_EQ(gathered.size(), 2u);
    EXPECT_EQ(gathered[0], gathered[1]);
    EXPECT_EQ(dh.getDofMap().getDofOwner(edge_dof), 0);

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);

    dh.syncGhostValuesMPI(owned_values, ghost_values);

    // Verify ghost values equal owner-provided values in ghost ordering.
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
    }
}

TEST(DofHandlerMPI, DistributedCG_P2_TwoRank_Vector2_SharedEdgeAndGhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/120 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3, /*num_components=*/2);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 18);
    EXPECT_EQ(dh.getNumLocalDofs(), (rank == 0) ? 12 : 6);

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);

    const auto le = findLocalEdgeByVertexGids(topo, /*gid_a=*/0, /*gid_b=*/1);
    ASSERT_TRUE(le.has_value());
    const auto edofs = entity->getEdgeDofs(*le);
    ASSERT_EQ(edofs.size(), 2u);

    std::array<GlobalIndex, 2> edge_dofs{edofs[0], edofs[1]};
    const auto gathered = allgatherGlobalIndices(edge_dofs, comm);
    ASSERT_EQ(gathered.size(), 4u);
    EXPECT_EQ(gathered[0], gathered[2]);
    EXPECT_EQ(gathered[1], gathered[3]);
    EXPECT_EQ(dh.getDofMap().getDofOwner(edge_dofs[0]), 0);
    EXPECT_EQ(dh.getDofMap().getDofOwner(edge_dofs[1]), 0);

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);

    dh.syncGhostValuesMPI(owned_values, ghost_values);

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
    }
}

TEST(DofHandlerMPI, DistributedCG_P2_TwoRank_Vector2_GhostSync_MatchesReferenceExchange) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/125 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3, /*num_components=*/2);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_ref(static_cast<std::size_t>(ghost_dofs.size()), -1.0);
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -2.0);
    std::vector<double> ghost_persistent(static_cast<std::size_t>(ghost_dofs.size()), -3.0);

    const auto* ghost_mgr = dh.getGhostManager();
    ASSERT_NE(ghost_mgr, nullptr);

    ghost_mgr->syncGhostValues(owned_values, ghost_ref,
                              [&](int send_rank,
                                  std::span<const double> send_data,
                                  int recv_rank,
                                  std::span<double> recv_data) {
                                  constexpr int tag = 9902;
                                  MPI_Sendrecv(send_data.empty() ? nullptr : send_data.data(),
                                               static_cast<int>(send_data.size()),
                                               MPI_DOUBLE,
                                               send_rank,
                                               tag,
                                               recv_data.empty() ? nullptr : recv_data.data(),
                                               static_cast<int>(recv_data.size()),
                                               MPI_DOUBLE,
                                               recv_rank,
                                               tag,
                                               comm,
                                               MPI_STATUS_IGNORE);
                              });

    dh.syncGhostValuesMPI(owned_values, ghost_values);
    dh.syncGhostValuesMPIPersistent(owned_values, ghost_persistent);

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_ref[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], ghost_ref[i]);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], ghost_ref[i]);
    }
}

TEST(DofHandlerMPI, DistributedDG_Line_TwoRank_Vector2_GhostCellNumberingAndSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankLineDGWithGhostCells(rank, /*gid_cell0=*/900, /*gid_cell1=*/901);
    const auto layout = DofLayoutInfo::DG(/*order=*/1, /*num_verts_per_cell=*/2, /*components=*/2);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 8);
    EXPECT_EQ(dh.getNumLocalDofs(), 4);

    const auto cell0_dofs = dh.getCellDofs(0);
    const auto cell1_dofs = dh.getCellDofs(1);
    ASSERT_EQ(cell0_dofs.size(), 4u);
    ASSERT_EQ(cell1_dofs.size(), 4u);

    std::array<GlobalIndex, 4> local0{};
    std::array<GlobalIndex, 4> local1{};
    std::copy(cell0_dofs.begin(), cell0_dofs.end(), local0.begin());
    std::copy(cell1_dofs.begin(), cell1_dofs.end(), local1.begin());

    const auto g0 = allgatherGlobalIndices(local0, comm);
    const auto g1 = allgatherGlobalIndices(local1, comm);
    ASSERT_EQ(g0.size(), 8u);
    ASSERT_EQ(g1.size(), 8u);
    EXPECT_EQ(std::vector<GlobalIndex>(g0.begin(), g0.begin() + 4),
              std::vector<GlobalIndex>(g0.begin() + 4, g0.end()));
    EXPECT_EQ(std::vector<GlobalIndex>(g1.begin(), g1.begin() + 4),
              std::vector<GlobalIndex>(g1.begin() + 4, g1.end()));

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);
    dh.syncGhostValuesMPI(owned_values, ghost_values);

    ASSERT_EQ(ghost_values.size(), ghost_dofs.size());
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
    }
}

TEST(DofHandlerMPI, DistributedCG_P2_TwoRank_GhostSync_MatchesReferenceExchange) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/25 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_ref(static_cast<std::size_t>(ghost_dofs.size()), -1.0);
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -2.0);
    std::vector<double> ghost_persistent(static_cast<std::size_t>(ghost_dofs.size()), -3.0);

    const auto* ghost_mgr = dh.getGhostManager();
    ASSERT_NE(ghost_mgr, nullptr);

    ghost_mgr->syncGhostValues(owned_values, ghost_ref,
                              [&](int send_rank,
                                  std::span<const double> send_data,
                                  int recv_rank,
                                  std::span<double> recv_data) {
                                  constexpr int tag = 9901;
                                  MPI_Sendrecv(send_data.empty() ? nullptr : send_data.data(),
                                               static_cast<int>(send_data.size()),
                                               MPI_DOUBLE,
                                               send_rank,
                                               tag,
                                               recv_data.empty() ? nullptr : recv_data.data(),
                                               static_cast<int>(recv_data.size()),
                                               MPI_DOUBLE,
                                               recv_rank,
                                               tag,
                                               comm,
                                               MPI_STATUS_IGNORE);
                              });

    dh.syncGhostValuesMPI(owned_values, ghost_values);
    dh.syncGhostValuesMPIPersistent(owned_values, ghost_persistent);

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_ref[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], ghost_ref[i]);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], ghost_ref[i]);
    }
}

TEST(DofHandlerMPI, DistributedCG_P2_TwoRank_PersistentGhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/40 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();
    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);
    std::vector<double> ghost_values_mpi(static_cast<std::size_t>(ghost_dofs.size()), -2.0);

    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }
    dh.syncGhostValuesMPIPersistent(owned_values, ghost_values);
    dh.syncGhostValuesMPI(owned_values, ghost_values_mpi);
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values_mpi[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values_mpi[i], ghost_values[i]);
    }

    owned_values.clear();
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(2000 * rank) + static_cast<double>(dof));
    }
    dh.syncGhostValuesMPIPersistent(owned_values, ghost_values);
    dh.syncGhostValuesMPI(owned_values, ghost_values_mpi);
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(2000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values_mpi[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values_mpi[i], ghost_values[i]);
    }
}

TEST(DofHandlerMPI, DistributedCG_CellOwnerStrategy_TwoRank_UsesMinCellGid) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    // Swap cell gids so rank 1 has the smaller cell gid.
    const GlobalIndex cell_gid = (rank == 0) ? 11 : 10;
    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, cell_gid);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::CellOwner;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto lv0 = findLocalVertexByGid(topo, 0);
    ASSERT_TRUE(lv0.has_value());
    const auto v0_dofs = entity->getVertexDofs(*lv0);
    ASSERT_EQ(v0_dofs.size(), 1u);

    // Owner should be rank 1 due to smaller cell gid.
    EXPECT_EQ(dh.getDofMap().getDofOwner(v0_dofs[0]), 1);
}

TEST(DofHandlerMPI, DistributedCG_VertexGIDOwnership_IsTouchingAndConsistent) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP1(rank, /*gid_cell=*/30 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::VertexGID;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto lv0 = findLocalVertexByGid(topo, 0);
    ASSERT_TRUE(lv0.has_value());
    const auto v0_dofs = entity->getVertexDofs(*lv0);
    ASSERT_EQ(v0_dofs.size(), 1u);

    const int owner_local = dh.getDofMap().getDofOwner(v0_dofs[0]);
    const std::array<GlobalIndex, 1> local_owner = {static_cast<GlobalIndex>(owner_local)};
    const auto gathered = allgatherGlobalIndices(local_owner, comm);
    ASSERT_EQ(gathered.size(), 2u);
    EXPECT_EQ(gathered[0], gathered[1]); // all ranks agree

    EXPECT_TRUE(owner_local == 0 || owner_local == 1); // owner must touch (both touch)
}

TEST(DofHandlerMPI, DistributedCG_P2_FourRank_MultiNeighborGhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 4) {
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }

    MeshTopologyInfo topo = makeFourRankTriangleP2(rank, /*gid_cell=*/100 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::HighestRank;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 15);

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);

    dh.syncGhostValuesMPI(owned_values, ghost_values);

    // Rank 0 should have ghosts from multiple owners; validate by value matching.
    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
    }
}

TEST(DofHandlerMPI, DistributedCG_P2_FourRank_NoGlobalCollectives_GhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 4) {
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }

    MeshTopologyInfo topo = makeFourRankTriangleP2(rank, /*gid_cell=*/200 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;
    opts.no_global_collectives = true;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 15);

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_ref(static_cast<std::size_t>(ghost_dofs.size()), -1.0);
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -2.0);
    std::vector<double> ghost_persistent(static_cast<std::size_t>(ghost_dofs.size()), -3.0);

    const auto* ghost_mgr = dh.getGhostManager();
    ASSERT_NE(ghost_mgr, nullptr);

    ghost_mgr->syncGhostValues(owned_values, ghost_ref,
                              [&](int send_rank,
                                  std::span<const double> send_data,
                                  int recv_rank,
                                  std::span<double> recv_data) {
                                  constexpr int tag = 9902;
                                  MPI_Sendrecv(send_data.empty() ? nullptr : send_data.data(),
                                               static_cast<int>(send_data.size()),
                                               MPI_DOUBLE,
                                               send_rank,
                                               tag,
                                               recv_data.empty() ? nullptr : recv_data.data(),
                                               static_cast<int>(recv_data.size()),
                                               MPI_DOUBLE,
                                               recv_rank,
                                               tag,
                                               comm,
                                               MPI_STATUS_IGNORE);
                              });

    dh.syncGhostValuesMPI(owned_values, ghost_values);
    dh.syncGhostValuesMPIPersistent(owned_values, ghost_persistent);

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_ref[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], expected);
        EXPECT_DOUBLE_EQ(ghost_values[i], ghost_ref[i]);
        EXPECT_DOUBLE_EQ(ghost_persistent[i], ghost_ref[i]);
    }
}

TEST(DofHandlerMPI, GlobalIds_NoGlobalCollectives_TwoRank_NumberingAndGhostSync) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2(rank, /*gid_cell=*/300 + rank);
    setTriangleEdgeGidsFromVertices(topo);

    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;
    opts.global_numbering = GlobalNumberingMode::GlobalIds;
    opts.no_global_collectives = true;
    opts.validate_parallel = true;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 9);

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);

    const auto lv0 = findLocalVertexByGid(topo, 0);
    ASSERT_TRUE(lv0.has_value());
    const auto v0_dofs = entity->getVertexDofs(*lv0);
    ASSERT_EQ(v0_dofs.size(), 1u);
    EXPECT_EQ(v0_dofs[0], 0);

    const auto le01 = findLocalEdgeByVertexGids(topo, 0, 1);
    ASSERT_TRUE(le01.has_value());
    const auto e01_dofs = entity->getEdgeDofs(*le01);
    ASSERT_EQ(e01_dofs.size(), 1u);
    EXPECT_EQ(e01_dofs[0], 4);

    dh.buildScatterContexts();

    const auto owned = dh.getPartition().locallyOwned().toVector();
    std::vector<double> owned_values;
    owned_values.reserve(owned.size());
    for (auto dof : owned) {
        owned_values.push_back(static_cast<double>(1000 * rank) + static_cast<double>(dof));
    }

    const auto ghost_dofs = dh.getGhostDofs();
    std::vector<double> ghost_values(static_cast<std::size_t>(ghost_dofs.size()), -1.0);

    dh.syncGhostValuesMPI(owned_values, ghost_values);

    for (std::size_t i = 0; i < ghost_dofs.size(); ++i) {
        const auto dof = ghost_dofs[i];
        const int owner = dh.getDofMap().getDofOwner(dof);
        const double expected = static_cast<double>(1000 * owner) + static_cast<double>(dof);
        EXPECT_DOUBLE_EQ(ghost_values[i], expected);
    }
}

TEST(DofHandlerMPI, GlobalIds_ProcessCountIndependent_Numbering) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int world_rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size != 4) {
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }

    // 4-rank partition: one triangle per rank.
    const GlobalIndex cell_gid = 100 + world_rank;
    MeshTopologyInfo topo4 = makeFourRankTriangleP2(world_rank, cell_gid);
    setTriangleEdgeGidsFromVertices(topo4);

    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts4;
    opts4.my_rank = world_rank;
    opts4.world_size = world_size;
    opts4.mpi_comm = comm;
    opts4.ownership = OwnershipStrategy::LowestRank;
    opts4.global_numbering = GlobalNumberingMode::GlobalIds;
    opts4.validate_parallel = true;

    DofHandler dh4;
    dh4.distributeDofs(topo4, layout, opts4);
    dh4.finalize();

    EXPECT_EQ(dh4.getNumDofs(), 15);

    GlobalIndex v0_dof4 = -1;
    GlobalIndex e01_dof4 = -1;
    if (world_rank < 2) {
        const auto* entity4 = dh4.getEntityDofMap();
        ASSERT_NE(entity4, nullptr);
        const auto lv0 = findLocalVertexByGid(topo4, 0);
        ASSERT_TRUE(lv0.has_value());
        const auto v0_dofs = entity4->getVertexDofs(*lv0);
        ASSERT_EQ(v0_dofs.size(), 1u);
        v0_dof4 = v0_dofs[0];

        const auto le01 = findLocalEdgeByVertexGids(topo4, 0, 1);
        ASSERT_TRUE(le01.has_value());
        const auto e_dofs = entity4->getEdgeDofs(*le01);
        ASSERT_EQ(e_dofs.size(), 1u);
        e01_dof4 = e_dofs[0];
    }

    // 2-rank sub-communicator partition: two triangles per rank.
    MPI_Comm comm2 = MPI_COMM_NULL;
    MPI_Comm_split(comm, (world_rank < 2) ? 0 : MPI_UNDEFINED, world_rank, &comm2);

    if (world_rank < 2) {
        const int rank2 = mpiRank(comm2);
        const int size2 = mpiSize(comm2);
        ASSERT_EQ(size2, 2);

        MeshTopologyInfo topo2 = makeTwoRankFourTriangleMeshP2(rank2);

        DofDistributionOptions opts2;
        opts2.my_rank = rank2;
        opts2.world_size = size2;
        opts2.mpi_comm = comm2;
        opts2.ownership = OwnershipStrategy::LowestRank;
        opts2.global_numbering = GlobalNumberingMode::GlobalIds;
        opts2.validate_parallel = true;

        DofHandler dh2;
        dh2.distributeDofs(topo2, layout, opts2);
        dh2.finalize();

        EXPECT_EQ(dh2.getNumDofs(), 15);

        const auto* entity2 = dh2.getEntityDofMap();
        ASSERT_NE(entity2, nullptr);
        const auto lv0 = findLocalVertexByGid(topo2, 0);
        ASSERT_TRUE(lv0.has_value());
        const auto v0_dofs = entity2->getVertexDofs(*lv0);
        ASSERT_EQ(v0_dofs.size(), 1u);
        const GlobalIndex v0_dof2 = v0_dofs[0];

        const auto le01 = findLocalEdgeByVertexGids(topo2, 0, 1);
        ASSERT_TRUE(le01.has_value());
        const auto e_dofs = entity2->getEdgeDofs(*le01);
        ASSERT_EQ(e_dofs.size(), 1u);
        const GlobalIndex e01_dof2 = e_dofs[0];

        // Vertex DOFs use vertex_gid directly; edge block starts at (max_vertex_gid+1)=6.
        EXPECT_EQ(v0_dof2, 0);
        EXPECT_EQ(e01_dof2, 6);

        // Must match the 4-rank partition numbering on the overlapping ranks.
        EXPECT_EQ(v0_dof2, v0_dof4);
        EXPECT_EQ(e01_dof2, e01_dof4);

        MPI_Comm_free(&comm2);
    }
}

TEST(DofHandlerMPI, DenseGlobalIds_SparseGids_ProducesDenseRanges) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 4) {
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }

    MeshTopologyInfo topo = makeFourRankTriangleP2Sparse(rank, /*gid_cell=*/200 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts;
    opts.my_rank = rank;
    opts.world_size = size;
    opts.mpi_comm = comm;
    opts.ownership = OwnershipStrategy::LowestRank;
    opts.global_numbering = GlobalNumberingMode::DenseGlobalIds;
    opts.validate_parallel = true;

    DofHandler dh;
    dh.distributeDofs(topo, layout, opts);
    dh.finalize();

    EXPECT_EQ(dh.getNumDofs(), 15);

    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);

    std::vector<GlobalIndex> local_vertex_dofs;
    local_vertex_dofs.reserve(static_cast<std::size_t>(topo.n_vertices));
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        const auto dofs = entity->getVertexDofs(v);
        ASSERT_EQ(dofs.size(), 1u);
        local_vertex_dofs.push_back(dofs[0]);
    }

    const auto gathered_vertices = allgatherGlobalIndices(local_vertex_dofs, comm);
    std::vector<GlobalIndex> unique_vertices = gathered_vertices;
    std::sort(unique_vertices.begin(), unique_vertices.end());
    unique_vertices.erase(std::unique(unique_vertices.begin(), unique_vertices.end()), unique_vertices.end());
    ASSERT_EQ(unique_vertices.size(), 6u);
    for (GlobalIndex i = 0; i < 6; ++i) {
        EXPECT_EQ(unique_vertices[static_cast<std::size_t>(i)], i);
    }

    std::vector<GlobalIndex> local_edge_dofs;
    local_edge_dofs.reserve(static_cast<std::size_t>(topo.n_edges));
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        const auto dofs = entity->getEdgeDofs(e);
        ASSERT_EQ(dofs.size(), 1u);
        local_edge_dofs.push_back(dofs[0]);
    }

    const auto gathered_edges = allgatherGlobalIndices(local_edge_dofs, comm);
    std::vector<GlobalIndex> unique_edges = gathered_edges;
    std::sort(unique_edges.begin(), unique_edges.end());
    unique_edges.erase(std::unique(unique_edges.begin(), unique_edges.end()), unique_edges.end());
    ASSERT_EQ(unique_edges.size(), 9u);
    for (GlobalIndex i = 0; i < 9; ++i) {
        EXPECT_EQ(unique_edges[static_cast<std::size_t>(i)], 6 + i);
    }

    // Auto-selected key-ordered dense IDs: vertex DOFs are monotone in vertex_gid.
    const auto gathered_vertex_gids =
        allgathervInt64(std::span<const svmp::FE::dofs::gid_t>(topo.vertex_gids.data(), topo.vertex_gids.size()), comm);
    const auto gathered_vertex_dofs =
        allgathervInt64(std::span<const GlobalIndex>(local_vertex_dofs.data(), local_vertex_dofs.size()), comm);
    ASSERT_EQ(gathered_vertex_gids.size(), gathered_vertex_dofs.size());

    struct Pair {
        svmp::FE::dofs::gid_t gid{0};
        GlobalIndex dof{-1};
    };
    std::vector<Pair> pairs;
    pairs.reserve(gathered_vertex_gids.size());
    for (std::size_t i = 0; i < gathered_vertex_gids.size(); ++i) {
        pairs.push_back(Pair{gathered_vertex_gids[i], gathered_vertex_dofs[i]});
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.gid < b.gid; });
    pairs.erase(std::unique(pairs.begin(),
                            pairs.end(),
                            [](const Pair& a, const Pair& b) {
                                return a.gid == b.gid && a.dof == b.dof;
                            }),
                pairs.end());
    for (std::size_t i = 1; i < pairs.size(); ++i) {
        EXPECT_LT(pairs[i - 1].dof, pairs[i].dof);
    }
}

TEST(DofHandlerMPI, DenseGlobalIds_ProcessCountIndependent_NumberingSparse) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int world_rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size != 4) {
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }

    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    MeshTopologyInfo topo4 = makeFourRankTriangleP2Sparse(world_rank, /*gid_cell=*/300 + world_rank);

    DofDistributionOptions opts4;
    opts4.my_rank = world_rank;
    opts4.world_size = world_size;
    opts4.mpi_comm = comm;
    opts4.ownership = OwnershipStrategy::LowestRank;
    opts4.global_numbering = GlobalNumberingMode::DenseGlobalIds;
    opts4.validate_parallel = true;

    DofHandler dh4;
    dh4.distributeDofs(topo4, layout, opts4);
    dh4.finalize();

    EXPECT_EQ(dh4.getNumDofs(), 15);

    GlobalIndex v0_dof4 = -1;
    GlobalIndex e01_dof4 = -1;
    if (world_rank < 2) {
        const auto* entity4 = dh4.getEntityDofMap();
        ASSERT_NE(entity4, nullptr);

        const auto gv0 = sparseVertexGid(0);
        const auto gv1 = sparseVertexGid(1);

        const auto lv0 = findLocalVertexByGid(topo4, gv0);
        ASSERT_TRUE(lv0.has_value());
        const auto v0_dofs = entity4->getVertexDofs(*lv0);
        ASSERT_EQ(v0_dofs.size(), 1u);
        v0_dof4 = v0_dofs[0];

        const auto le01 = findLocalEdgeByVertexGids(topo4, gv0, gv1);
        ASSERT_TRUE(le01.has_value());
        const auto e_dofs = entity4->getEdgeDofs(*le01);
        ASSERT_EQ(e_dofs.size(), 1u);
        e01_dof4 = e_dofs[0];
    }

    MPI_Comm comm2 = MPI_COMM_NULL;
    MPI_Comm_split(comm, (world_rank < 2) ? 0 : MPI_UNDEFINED, world_rank, &comm2);

    if (world_rank < 2) {
        const int rank2 = mpiRank(comm2);
        const int size2 = mpiSize(comm2);
        ASSERT_EQ(size2, 2);

        MeshTopologyInfo topo2 = makeTwoRankFourTriangleMeshP2Sparse(rank2);

        DofDistributionOptions opts2;
        opts2.my_rank = rank2;
        opts2.world_size = size2;
        opts2.mpi_comm = comm2;
        opts2.ownership = OwnershipStrategy::LowestRank;
        opts2.global_numbering = GlobalNumberingMode::DenseGlobalIds;
        opts2.validate_parallel = true;

        DofHandler dh2;
        dh2.distributeDofs(topo2, layout, opts2);
        dh2.finalize();

        EXPECT_EQ(dh2.getNumDofs(), 15);

        const auto* entity2 = dh2.getEntityDofMap();
        ASSERT_NE(entity2, nullptr);

        const auto gv0 = sparseVertexGid(0);
        const auto gv1 = sparseVertexGid(1);

        const auto lv0 = findLocalVertexByGid(topo2, gv0);
        ASSERT_TRUE(lv0.has_value());
        const auto v0_dofs = entity2->getVertexDofs(*lv0);
        ASSERT_EQ(v0_dofs.size(), 1u);
        const GlobalIndex v0_dof2 = v0_dofs[0];

        const auto le01 = findLocalEdgeByVertexGids(topo2, gv0, gv1);
        ASSERT_TRUE(le01.has_value());
        const auto e_dofs = entity2->getEdgeDofs(*le01);
        ASSERT_EQ(e_dofs.size(), 1u);
        const GlobalIndex e01_dof2 = e_dofs[0];

        EXPECT_EQ(v0_dof2, v0_dof4);
        EXPECT_EQ(e01_dof2, e01_dof4);

        MPI_Comm_free(&comm2);
    }
}

TEST(DofHandlerMPI, DenseGlobalIds_NoGlobalCollectives_MatchesCollectives) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MeshTopologyInfo topo = makeTwoRankTriangleP2Sparse(rank, /*gid_cell=*/400 + rank);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/3);

    auto run = [&](bool no_global_collectives) {
        DofDistributionOptions opts;
        opts.my_rank = rank;
        opts.world_size = size;
        opts.mpi_comm = comm;
        opts.ownership = OwnershipStrategy::LowestRank;
        opts.global_numbering = GlobalNumberingMode::DenseGlobalIds;
        opts.validate_parallel = true;
        opts.no_global_collectives = no_global_collectives;

        DofHandler dh;
        dh.distributeDofs(topo, layout, opts);
        dh.finalize();

        const auto* entity = dh.getEntityDofMap();
        if (!entity) {
            throw std::runtime_error("missing EntityDofMap");
        }

        std::vector<GlobalIndex> result;
        result.reserve(static_cast<std::size_t>(topo.n_vertices + topo.n_edges));
        for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
            const auto dofs = entity->getVertexDofs(v);
            if (dofs.size() != 1u) throw std::runtime_error("unexpected vertex dof size");
            result.push_back(dofs[0]);
        }
        for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
            const auto dofs = entity->getEdgeDofs(e);
            if (dofs.size() != 1u) throw std::runtime_error("unexpected edge dof size");
            result.push_back(dofs[0]);
        }
        return std::make_pair(dh.getNumDofs(), result);
    };

    const auto [n_collective, dofs_collective] = run(false);
    const auto [n_nocollective, dofs_nocollective] = run(true);

    EXPECT_EQ(n_collective, n_nocollective);
    EXPECT_EQ(dofs_collective, dofs_nocollective);
}

TEST(DofHandlerMPI, DenseGlobalIds_AutoSelectHashOrdering_LargeVertexCount) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    // Global unique vertices = 2 * (3*n_cells). Choose just above the library default threshold.
    constexpr GlobalIndex n_cells = 10000; // per rank -> 30000 vertices per rank, 60000 global.
    MeshTopologyInfo topo = makeTwoRankDisjointTrianglesP1(rank, n_cells, /*gid_cell_base=*/500);
    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    auto run = [&](bool no_global_collectives) {
        DofDistributionOptions opts;
        opts.my_rank = rank;
        opts.world_size = size;
        opts.mpi_comm = comm;
        opts.ownership = OwnershipStrategy::LowestRank;
        opts.global_numbering = GlobalNumberingMode::DenseGlobalIds;
        opts.no_global_collectives = no_global_collectives;
        opts.validate_parallel = false;

        DofHandler dh;
        dh.distributeDofs(topo, layout, opts);
        dh.finalize();

        const auto* entity = dh.getEntityDofMap();
        if (!entity) {
            throw std::runtime_error("missing EntityDofMap");
        }
        std::vector<GlobalIndex> vertex_dofs;
        vertex_dofs.reserve(static_cast<std::size_t>(topo.n_vertices));
        for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
            const auto dofs = entity->getVertexDofs(v);
            if (dofs.size() != 1u) throw std::runtime_error("unexpected vertex dof size");
            vertex_dofs.push_back(dofs[0]);
        }
        return std::make_pair(dh.getNumDofs(), vertex_dofs);
    };

    const auto [n_collective, dofs_collective] = run(false);
    const auto [n_nocollective, dofs_nocollective] = run(true);

    EXPECT_EQ(n_collective, n_nocollective);
    EXPECT_EQ(dofs_collective, dofs_nocollective);
    EXPECT_EQ(n_collective, static_cast<GlobalIndex>(2 * 3 * n_cells));

    const auto gathered_vertex_gids =
        allgathervInt64(std::span<const svmp::FE::dofs::gid_t>(topo.vertex_gids.data(), topo.vertex_gids.size()), comm);
    const auto gathered_vertex_dofs =
        allgathervInt64(std::span<const GlobalIndex>(dofs_collective.data(), dofs_collective.size()), comm);
    ASSERT_EQ(gathered_vertex_gids.size(), gathered_vertex_dofs.size());

    struct Pair {
        svmp::FE::dofs::gid_t gid{0};
        GlobalIndex dof{-1};
    };
    std::vector<Pair> pairs;
    pairs.reserve(gathered_vertex_gids.size());
    for (std::size_t i = 0; i < gathered_vertex_gids.size(); ++i) {
        pairs.push_back(Pair{gathered_vertex_gids[i], gathered_vertex_dofs[i]});
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.gid < b.gid; });

    bool has_inversion = false;
    for (std::size_t i = 1; i < pairs.size(); ++i) {
        if (pairs[i].dof <= pairs[i - 1].dof) {
            has_inversion = true;
            break;
        }
    }
    EXPECT_TRUE(has_inversion);
}

TEST(DofHandlerMPI, ReproducibleAcrossCommunicators_ReversedRanks_SameNumbering) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int world_rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MPI_Comm rev_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm, /*color=*/0, /*key=*/world_size - 1 - world_rank, &rev_comm);
    const int rev_rank = mpiRank(rev_comm);

    MeshTopologyInfo topo_world = makeTwoRankTriangleP1(world_rank, /*gid_cell=*/60 + world_rank);
    MeshTopologyInfo topo_rev = makeTwoRankTriangleP1(world_rank, /*gid_cell=*/60 + world_rank);
    topo_rev.cell_owner_ranks = {rev_rank};
    topo_rev.neighbor_ranks = {1 - rev_rank};

    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts_world;
    opts_world.my_rank = world_rank;
    opts_world.world_size = world_size;
    opts_world.mpi_comm = comm;
    opts_world.ownership = OwnershipStrategy::LowestRank;
    opts_world.reproducible_across_communicators = true;

    DofHandler dh_world;
    dh_world.distributeDofs(topo_world, layout, opts_world);
    dh_world.finalize();

    const auto* entity_world = dh_world.getEntityDofMap();
    ASSERT_NE(entity_world, nullptr);
    const auto lv0_world = findLocalVertexByGid(topo_world, 0);
    const auto lv1_world = findLocalVertexByGid(topo_world, 1);
    ASSERT_TRUE(lv0_world.has_value());
    ASSERT_TRUE(lv1_world.has_value());
    const auto v0_world = entity_world->getVertexDofs(*lv0_world);
    const auto v1_world = entity_world->getVertexDofs(*lv1_world);
    ASSERT_EQ(v0_world.size(), 1u);
    ASSERT_EQ(v1_world.size(), 1u);

    DofDistributionOptions opts_rev = opts_world;
    opts_rev.my_rank = rev_rank;
    opts_rev.mpi_comm = rev_comm;

    DofHandler dh_rev;
    dh_rev.distributeDofs(topo_rev, layout, opts_rev);
    dh_rev.finalize();

    const auto* entity_rev = dh_rev.getEntityDofMap();
    ASSERT_NE(entity_rev, nullptr);
    const auto lv0_rev = findLocalVertexByGid(topo_rev, 0);
    const auto lv1_rev = findLocalVertexByGid(topo_rev, 1);
    ASSERT_TRUE(lv0_rev.has_value());
    ASSERT_TRUE(lv1_rev.has_value());
    const auto v0_rev = entity_rev->getVertexDofs(*lv0_rev);
    const auto v1_rev = entity_rev->getVertexDofs(*lv1_rev);
    ASSERT_EQ(v0_rev.size(), 1u);
    ASSERT_EQ(v1_rev.size(), 1u);

    EXPECT_EQ(v0_world[0], v0_rev[0]);
    EXPECT_EQ(v1_world[0], v1_rev[0]);

    MPI_Comm_free(&rev_comm);
}

TEST(DofHandlerMPI, ReproducibleAcrossCommunicators_NoGlobalCollectives_ReversedRanks_SameNumbering) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const int world_rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size != 2) {
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }

    MPI_Comm rev_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm, /*color=*/0, /*key=*/world_size - 1 - world_rank, &rev_comm);
    const int rev_rank = mpiRank(rev_comm);

    MeshTopologyInfo topo_world = makeTwoRankTriangleP1(world_rank, /*gid_cell=*/70 + world_rank);
    MeshTopologyInfo topo_rev = makeTwoRankTriangleP1(world_rank, /*gid_cell=*/70 + world_rank);
    topo_rev.cell_owner_ranks = {rev_rank};
    topo_rev.neighbor_ranks = {1 - rev_rank};

    const auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions opts_world;
    opts_world.my_rank = world_rank;
    opts_world.world_size = world_size;
    opts_world.mpi_comm = comm;
    opts_world.ownership = OwnershipStrategy::LowestRank;
    opts_world.reproducible_across_communicators = true;
    opts_world.no_global_collectives = true;

    DofHandler dh_world;
    dh_world.distributeDofs(topo_world, layout, opts_world);
    dh_world.finalize();

    const auto* entity_world = dh_world.getEntityDofMap();
    ASSERT_NE(entity_world, nullptr);
    const auto lv0_world = findLocalVertexByGid(topo_world, 0);
    const auto lv1_world = findLocalVertexByGid(topo_world, 1);
    ASSERT_TRUE(lv0_world.has_value());
    ASSERT_TRUE(lv1_world.has_value());
    const auto v0_world = entity_world->getVertexDofs(*lv0_world);
    const auto v1_world = entity_world->getVertexDofs(*lv1_world);
    ASSERT_EQ(v0_world.size(), 1u);
    ASSERT_EQ(v1_world.size(), 1u);

    DofDistributionOptions opts_rev = opts_world;
    opts_rev.my_rank = rev_rank;
    opts_rev.mpi_comm = rev_comm;

    DofHandler dh_rev;
    dh_rev.distributeDofs(topo_rev, layout, opts_rev);
    dh_rev.finalize();

    const auto* entity_rev = dh_rev.getEntityDofMap();
    ASSERT_NE(entity_rev, nullptr);
    const auto lv0_rev = findLocalVertexByGid(topo_rev, 0);
    const auto lv1_rev = findLocalVertexByGid(topo_rev, 1);
    ASSERT_TRUE(lv0_rev.has_value());
    ASSERT_TRUE(lv1_rev.has_value());
    const auto v0_rev = entity_rev->getVertexDofs(*lv0_rev);
    const auto v1_rev = entity_rev->getVertexDofs(*lv1_rev);
    ASSERT_EQ(v0_rev.size(), 1u);
    ASSERT_EQ(v1_rev.size(), 1u);

    EXPECT_EQ(v0_world[0], v0_rev[0]);
    EXPECT_EQ(v1_world[0], v1_rev[0]);

    MPI_Comm_free(&rev_comm);
}
