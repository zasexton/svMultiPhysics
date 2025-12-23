/**
 * @file test_DofHandler.cpp
 * @brief Unit tests for DofHandler (mesh-independent DOF distribution)
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofHandler.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Core/FEException.h"
#include "FE/Elements/ReferenceElement.h"

#include <numeric>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

using svmp::FE::FEException;
using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::MeshIndex;
using svmp::FE::dofs::DofDistributionOptions;
using svmp::FE::dofs::DofHandler;
using svmp::FE::dofs::DofLayoutInfo;
using svmp::FE::dofs::MeshTopologyInfo;
using svmp::FE::dofs::TopologyCompletion;

namespace {

MeshTopologyInfo makeLineMesh(GlobalIndex n_cells) {
    MeshTopologyInfo topo;
    topo.n_cells = n_cells;
    topo.n_vertices = n_cells + 1;
    topo.dim = 1;

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(n_cells) + 1);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(2 * n_cells));
    topo.vertex_gids.resize(static_cast<std::size_t>(n_cells) + 1);

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = 2 * c;
        topo.cell2vertex_data[static_cast<std::size_t>(2 * c + 0)] = c;
        topo.cell2vertex_data[static_cast<std::size_t>(2 * c + 1)] = c + 1;
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(n_cells)] = 2 * n_cells;

    std::iota(topo.vertex_gids.begin(), topo.vertex_gids.end(), 0);
    return topo;
}

MeshTopologyInfo make2x2QuadMesh() {
    MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 9;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4, 8, 12, 16};
    topo.cell2vertex_data = {
        0, 1, 4, 3, // cell 0
        1, 2, 5, 4, // cell 1
        3, 4, 7, 6, // cell 2
        4, 5, 8, 7  // cell 3
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    return topo;
}

MeshTopologyInfo makeTwoTriangleMeshWithEdges() {
    // Two triangles sharing the (0,1) edge:
    // cell0: (0,1,2)
    // cell1: (0,3,1)  (shared edge is local edge (2->0) in reference ordering)
    MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 4;
    topo.n_edges = 5;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 3, 6};
    topo.cell2vertex_data = {0, 1, 2, 0, 3, 1};
    topo.vertex_gids = {0, 1, 2, 3};

    topo.cell2edge_offsets = {0, 3, 6};
    topo.cell2edge_data = {
        0, 1, 2, // cell 0 edges: (0-1), (1-2), (2-0)
        3, 4, 0  // cell 1 edges: (0-3), (3-1), (0-1)
    };
    topo.edge2vertex_data = {
        0, 1, // edge 0
        1, 2, // edge 1
        2, 0, // edge 2
        0, 3, // edge 3
        3, 1  // edge 4
    };

    return topo;
}

MeshTopologyInfo makeSingleQuadWithEdges() {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 4;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};

    topo.cell2edge_offsets = {0, 4};
    topo.cell2edge_data = {0, 1, 2, 3};
    topo.edge2vertex_data = {
        0, 1,
        1, 2,
        2, 3,
        3, 0
    };
    return topo;
}

MeshTopologyInfo makeSingleQuadVerticesOnly() {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    return topo;
}

MeshTopologyInfo makeSingleHexWithEdgesAndFaces() {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 8;
    topo.n_edges = 12;
    topo.n_faces = 6;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};

    topo.cell2edge_offsets = {0, 12};
    topo.cell2edge_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    topo.edge2vertex_data = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };

    topo.cell2face_offsets = {0, 6};
    topo.cell2face_data = {0, 1, 2, 3, 4, 5};
    topo.face2vertex_offsets = {0, 4, 8, 12, 16, 20, 24};
    topo.face2vertex_data = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7
    };

    return topo;
}

MeshTopologyInfo makeSingleHexWithEdgesOnly() {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 8;
    topo.n_edges = 12;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};

    topo.cell2edge_offsets = {0, 12};
    topo.cell2edge_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    topo.edge2vertex_data = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };

    return topo;
}

MeshTopologyInfo makeSingleHexVerticesOnly() {
    MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 8;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
    return topo;
}

MeshTopologyInfo makeTwoHexMeshWithSharedFaceFlipped() {
    // Two Hex8 cells sharing the face {1,2,6,5}. Cell 1's local face ordering
    // is reflected relative to cell 0 on the shared face, exercising face-DOF permutations.
    MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 12;
    topo.dim = 3;

    const std::vector<GlobalIndex> cell0 = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<GlobalIndex> cell1 = {5, 8, 9, 1, 6, 11, 10, 2};

    topo.cell2vertex_offsets = {0, 8, 16};
    topo.cell2vertex_data.insert(topo.cell2vertex_data.end(), cell0.begin(), cell0.end());
    topo.cell2vertex_data.insert(topo.cell2vertex_data.end(), cell1.begin(), cell1.end());
    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices));
    std::iota(topo.vertex_gids.begin(), topo.vertex_gids.end(), 0);

    const auto ref = svmp::FE::elements::ReferenceElement::create(ElementType::Hex8);

    struct EdgeKey {
        svmp::FE::dofs::gid_t a;
        svmp::FE::dofs::gid_t b;
        bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey& k) const noexcept {
            const std::size_t h1 = std::hash<svmp::FE::dofs::gid_t>{}(k.a);
            const std::size_t h2 = std::hash<svmp::FE::dofs::gid_t>{}(k.b);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    struct FaceKey {
        std::array<svmp::FE::dofs::gid_t, 4> gids{};
        bool operator==(const FaceKey& other) const noexcept { return gids == other.gids; }
    };
    struct FaceKeyHash {
        std::size_t operator()(const FaceKey& k) const noexcept {
            std::size_t seed = 0;
            for (auto g : k.gids) {
                const std::size_t h = std::hash<svmp::FE::dofs::gid_t>{}(g);
                seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    auto canonical_cycle = [&](const std::vector<GlobalIndex>& verts) -> std::vector<GlobalIndex> {
        const std::size_t n = verts.size();
        if (n < 3u) return verts;

        std::size_t start = 0;
        svmp::FE::dofs::gid_t best_gid = topo.vertex_gids[static_cast<std::size_t>(verts[0])];
        GlobalIndex best_vid = verts[0];
        for (std::size_t i = 1; i < n; ++i) {
            const auto v = verts[i];
            const auto gid = topo.vertex_gids[static_cast<std::size_t>(v)];
            if (gid < best_gid || (gid == best_gid && v < best_vid)) {
                best_gid = gid;
                best_vid = v;
                start = i;
            }
        }

        const std::size_t next = (start + 1u) % n;
        const std::size_t prev = (start + n - 1u) % n;
        const auto gid_next = topo.vertex_gids[static_cast<std::size_t>(verts[next])];
        const auto gid_prev = topo.vertex_gids[static_cast<std::size_t>(verts[prev])];

        const bool forward =
            (gid_next < gid_prev) || (gid_next == gid_prev && verts[next] < verts[prev]);

        std::vector<GlobalIndex> out;
        out.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t idx = forward ? ((start + k) % n) : ((start + n - k) % n);
            out.push_back(verts[idx]);
        }
        return out;
    };

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
    std::vector<std::array<GlobalIndex, 2>> edges;

    std::unordered_map<FaceKey, GlobalIndex, FaceKeyHash> face_ids;
    std::vector<std::vector<GlobalIndex>> faces;

    topo.cell2edge_offsets.resize(3, 0);
    topo.cell2face_offsets.resize(3, 0);

    for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
        topo.cell2edge_offsets[static_cast<std::size_t>(c)] =
            static_cast<GlobalIndex>(topo.cell2edge_data.size());
        topo.cell2face_offsets[static_cast<std::size_t>(c)] =
            static_cast<GlobalIndex>(topo.cell2face_data.size());

        const auto cell_verts = topo.getCellVertices(c);

        for (std::size_t le = 0; le < ref.num_edges(); ++le) {
            const auto& en = ref.edge_nodes(le);
            const GlobalIndex gv0 = cell_verts[static_cast<std::size_t>(en[0])];
            const GlobalIndex gv1 = cell_verts[static_cast<std::size_t>(en[1])];
            const auto gid0 = topo.vertex_gids[static_cast<std::size_t>(gv0)];
            const auto gid1 = topo.vertex_gids[static_cast<std::size_t>(gv1)];
            EdgeKey key{std::min(gid0, gid1), std::max(gid0, gid1)};
            auto it = edge_ids.find(key);
            GlobalIndex eid = -1;
            if (it == edge_ids.end()) {
                eid = static_cast<GlobalIndex>(edges.size());
                edge_ids.emplace(key, eid);
                if (gid0 <= gid1) {
                    edges.push_back({gv0, gv1});
                } else {
                    edges.push_back({gv1, gv0});
                }
            } else {
                eid = it->second;
            }
            topo.cell2edge_data.push_back(eid);
        }

        for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
            const auto& fn = ref.face_nodes(lf);
            std::vector<GlobalIndex> fv;
            fv.reserve(fn.size());
            FaceKey key{};
            for (std::size_t i = 0; i < 4u; ++i) {
                const GlobalIndex gv = cell_verts[static_cast<std::size_t>(fn[i])];
                fv.push_back(gv);
                key.gids[i] = topo.vertex_gids[static_cast<std::size_t>(gv)];
            }
            std::sort(key.gids.begin(), key.gids.end());

            auto it = face_ids.find(key);
            GlobalIndex fid = -1;
            if (it == face_ids.end()) {
                fid = static_cast<GlobalIndex>(faces.size());
                face_ids.emplace(key, fid);
                faces.push_back(canonical_cycle(fv));
            } else {
                fid = it->second;
            }
            topo.cell2face_data.push_back(fid);
        }
    }

    topo.cell2edge_offsets[2] = static_cast<GlobalIndex>(topo.cell2edge_data.size());
    topo.cell2face_offsets[2] = static_cast<GlobalIndex>(topo.cell2face_data.size());

    topo.n_edges = static_cast<GlobalIndex>(edges.size());
    topo.edge2vertex_data.resize(static_cast<std::size_t>(2) * static_cast<std::size_t>(topo.n_edges));
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        topo.edge2vertex_data[static_cast<std::size_t>(2 * e + 0)] = edges[static_cast<std::size_t>(e)][0];
        topo.edge2vertex_data[static_cast<std::size_t>(2 * e + 1)] = edges[static_cast<std::size_t>(e)][1];
    }

    topo.n_faces = static_cast<GlobalIndex>(faces.size());
    topo.face2vertex_offsets.clear();
    topo.face2vertex_data.clear();
    topo.face2vertex_offsets.reserve(static_cast<std::size_t>(topo.n_faces) + 1u);
    topo.face2vertex_offsets.push_back(0);
    for (const auto& fv : faces) {
        topo.face2vertex_data.insert(topo.face2vertex_data.end(), fv.begin(), fv.end());
        topo.face2vertex_offsets.push_back(static_cast<GlobalIndex>(topo.face2vertex_data.size()));
    }

    return topo;
}

} // namespace

TEST(DofHandler, DefaultConstruction) {
    DofHandler handler;
    EXPECT_FALSE(handler.isFinalized());
    EXPECT_EQ(handler.getNumDofs(), 0);
}

TEST(DofHandler, DGDistributionLineMesh) {
    DofHandler handler;
    auto topo = makeLineMesh(3);
    auto layout = DofLayoutInfo::DG(/*order=*/1, /*num_verts_per_cell=*/2, /*components=*/1);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    // 3 cells, 2 dofs per cell -> 6 dofs total.
    EXPECT_EQ(handler.getNumDofs(), 6);

    std::set<GlobalIndex> all;
    for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
        auto dofs = handler.getCellDofs(c);
        EXPECT_EQ(dofs.size(), 2u);
        all.insert(dofs.begin(), dofs.end());
    }
    EXPECT_EQ(all.size(), 6u);
}

TEST(DofHandler, CGDistributionQuadMeshP1) {
    DofHandler handler;
    auto topo = make2x2QuadMesh();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/2, /*num_verts_per_cell=*/4);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    // 9 vertex dofs for P1 quad mesh.
    EXPECT_EQ(handler.getNumDofs(), 9);

    // Center vertex id 4 appears in all four cells.
    for (GlobalIndex c = 0; c < topo.n_cells; ++c) {
        auto dofs = handler.getCellDofs(c);
        EXPECT_TRUE(std::find(dofs.begin(), dofs.end(), GlobalIndex{4}) != dofs.end());
    }
}

TEST(DofHandler, EntityDofMapIsPopulated) {
    DofHandler handler;
    auto topo = makeLineMesh(2);
    auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/1, /*num_verts_per_cell=*/2);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    const auto* edm = handler.getEntityDofMap();
    ASSERT_NE(edm, nullptr);
    EXPECT_TRUE(edm->isFinalized());
    EXPECT_TRUE(edm->hasReverseMapping());

    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        auto vdofs = edm->getVertexDofs(v);
        ASSERT_EQ(vdofs.size(), 1u);
        EXPECT_EQ(vdofs[0], v);
    }
}

TEST(DofHandler, LagrangeP3TriangleEdgeOrientationReversesInteriorEdgeDofs) {
    DofHandler handler;
    auto topo = makeTwoTriangleMeshWithEdges();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/3, /*dim=*/2, /*num_verts_per_cell=*/3);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    // Total DOFs:
    //  - 4 vertex DOFs
    //  - 5 edges * 2 interior dofs = 10
    //  - 2 cell interior dofs = 2
    EXPECT_EQ(handler.getNumDofs(), 16);

    const auto c0 = handler.getCellDofs(0);
    const auto c1 = handler.getCellDofs(1);
    ASSERT_EQ(c0.size(), 10u);
    ASSERT_EQ(c1.size(), 10u);

    // Expected edge-DOF ordering (see comments in helper mesh).
    const std::vector<GlobalIndex> expected_c0 = {
        0, 1, 2,          // vertices
        4, 5,             // edge 0 (0->1), forward
        6, 7,             // edge 1 (1->2), forward
        9, 8,             // edge 2 (2->0), reversed
        14                // cell interior
    };
    const std::vector<GlobalIndex> expected_c1 = {
        0, 3, 1,          // vertices
        10, 11,           // edge 3 (0->3), forward
        13, 12,           // edge 4 (3->1), reversed
        5, 4,             // edge 0 (1->0), reversed
        15                // cell interior
    };

    EXPECT_EQ(std::vector<GlobalIndex>(c0.begin(), c0.end()), expected_c0);
    EXPECT_EQ(std::vector<GlobalIndex>(c1.begin(), c1.end()), expected_c1);
}

TEST(DofHandler, LagrangeP3TriangleEdgeOrientationRespectsCanonicalOrderingFlag) {
    DofHandler handler;
    auto topo = makeTwoTriangleMeshWithEdges();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/3, /*dim=*/2, /*num_verts_per_cell=*/3);

    DofDistributionOptions options;
    options.use_canonical_ordering = false;

    handler.distributeDofs(topo, layout, options);
    handler.finalize();

    const auto c0 = handler.getCellDofs(0);
    const auto c1 = handler.getCellDofs(1);
    ASSERT_EQ(c0.size(), 10u);
    ASSERT_EQ(c1.size(), 10u);

    // With canonical ordering disabled, edge-interior DOFs follow the reference-element
    // direction without reversal.
    const std::vector<GlobalIndex> expected_c0 = {
        0, 1, 2,          // vertices
        4, 5,             // edge 0 (0->1)
        6, 7,             // edge 1 (1->2)
        8, 9,             // edge 2 (2->0), no reversal
        14                // cell interior
    };
    const std::vector<GlobalIndex> expected_c1 = {
        0, 3, 1,          // vertices
        10, 11,           // edge 3 (0->3)
        12, 13,           // edge 4 (3->1), no reversal
        4, 5,             // edge 0 (1->0), no reversal
        15                // cell interior
    };

    EXPECT_EQ(std::vector<GlobalIndex>(c0.begin(), c0.end()), expected_c0);
    EXPECT_EQ(std::vector<GlobalIndex>(c1.begin(), c1.end()), expected_c1);
}

TEST(DofHandler, CGDistributionSingleQuadQ2CountsAndOrdering) {
    DofHandler handler;
    auto topo = makeSingleQuadWithEdges();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/4);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 9);
    const auto dofs = handler.getCellDofs(0);
    EXPECT_EQ(dofs.size(), 9u);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()),
              (std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(DofHandler, CGDistributionSingleQuadQ2DerivesEdgesFromCell2Vertex) {
    DofHandler handler;
    auto topo = makeSingleQuadVerticesOnly();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/4);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 9);
    const auto dofs = handler.getCellDofs(0);
    EXPECT_EQ(dofs.size(), 9u);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()),
              (std::vector<GlobalIndex>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(DofHandler, CGDistributionSingleQuadQ2RequireCompleteThrowsWhenEdgesMissing) {
    DofHandler handler;
    auto topo = makeSingleQuadVerticesOnly();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/4);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    try {
        handler.distributeDofs(topo, layout, opts);
        FAIL() << "Expected FEException for missing edge connectivity";
    } catch (const FEException& ex) {
        EXPECT_NE(std::string(ex.what()).find("edge-interior DOFs require"), std::string::npos);
    }
}

TEST(DofHandler, CGDistributionSingleQuadQ2RequireCompleteSucceedsWithEdges) {
    DofHandler handler;
    auto topo = makeSingleQuadWithEdges();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/2, /*num_verts_per_cell=*/4);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    handler.distributeDofs(topo, layout, opts);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 9);
}

TEST(DofHandler, CGDistributionSingleHexQ2CountsAndOrdering) {
    DofHandler handler;
    auto topo = makeSingleHexWithEdgesAndFaces();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/3, /*num_verts_per_cell=*/8);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 27);
    const auto dofs = handler.getCellDofs(0);
    EXPECT_EQ(dofs.size(), 27u);

    std::vector<GlobalIndex> expected;
    expected.reserve(27);
    for (GlobalIndex i = 0; i < 27; ++i) expected.push_back(i);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()), expected);
}

TEST(DofHandler, CGDistributionSingleHexQ2RequireCompleteThrowsWhenFacesMissing) {
    DofHandler handler;
    auto topo = makeSingleHexWithEdgesOnly();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/3, /*num_verts_per_cell=*/8);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    try {
        handler.distributeDofs(topo, layout, opts);
        FAIL() << "Expected FEException for missing face connectivity";
    } catch (const FEException& ex) {
        EXPECT_NE(std::string(ex.what()).find("face-interior DOFs require"), std::string::npos);
    }
}

TEST(DofHandler, CGDistributionSingleHexQ2RequireCompleteSucceedsWithEdgesAndFaces) {
    DofHandler handler;
    auto topo = makeSingleHexWithEdgesAndFaces();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/3, /*num_verts_per_cell=*/8);

    DofDistributionOptions opts;
    opts.topology_completion = TopologyCompletion::RequireComplete;

    handler.distributeDofs(topo, layout, opts);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 27);
}

TEST(DofHandler, CGDistributionSingleHexQ2DerivesEdgesAndFacesFromCell2Vertex) {
    DofHandler handler;
    auto topo = makeSingleHexVerticesOnly();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/2, /*dim=*/3, /*num_verts_per_cell=*/8);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 27);
    const auto dofs = handler.getCellDofs(0);
    EXPECT_EQ(dofs.size(), 27u);

    std::vector<GlobalIndex> expected;
    expected.reserve(27);
    for (GlobalIndex i = 0; i < 27; ++i) expected.push_back(i);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()), expected);
}

TEST(DofHandler, LagrangeQ3HexSharedFacePermutesFaceInteriorDofs) {
    DofHandler handler;
    auto topo = makeTwoHexMeshWithSharedFaceFlipped();
    auto layout = DofLayoutInfo::Lagrange(/*order=*/3, /*dim=*/3, /*num_verts_per_cell=*/8);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    const auto* edm = handler.getEntityDofMap();
    ASSERT_NE(edm, nullptr);

    const auto cell0_faces = topo.getCellFaces(0);
    const auto cell1_faces = topo.getCellFaces(1);
    MeshIndex shared_face = -1;
    for (auto f0 : cell0_faces) {
        if (std::find(cell1_faces.begin(), cell1_faces.end(), f0) != cell1_faces.end()) {
            shared_face = f0;
            break;
        }
    }
    ASSERT_GE(shared_face, 0);

    const auto canon = edm->getFaceDofs(shared_face);
    ASSERT_EQ(canon.size(), 4u);

    auto find_face_slot = [](std::span<const MeshIndex> faces, MeshIndex face_id) -> int {
        for (std::size_t i = 0; i < faces.size(); ++i) {
            if (faces[i] == face_id) return static_cast<int>(i);
        }
        return -1;
    };

    const int lf0 = find_face_slot(cell0_faces, shared_face);
    const int lf1 = find_face_slot(cell1_faces, shared_face);
    ASSERT_GE(lf0, 0);
    ASSERT_GE(lf1, 0);

    const auto c0 = handler.getCellDofs(0);
    const auto c1 = handler.getCellDofs(1);
    ASSERT_EQ(c0.size(), 64u);
    ASSERT_EQ(c1.size(), 64u);

    const std::size_t vertex_dofs = 8u;
    const std::size_t edge_dofs = 12u * 2u;
    const std::size_t face_base = vertex_dofs + edge_dofs;

    std::vector<GlobalIndex> face0;
    std::vector<GlobalIndex> face1;
    for (std::size_t i = 0; i < 4u; ++i) {
        face0.push_back(c0[face_base + static_cast<std::size_t>(lf0) * 4u + i]);
        face1.push_back(c1[face_base + static_cast<std::size_t>(lf1) * 4u + i]);
    }

    // Cell 0 sees the shared face with canonical vertex ordering -> identity permutation.
    EXPECT_EQ(face0, (std::vector<GlobalIndex>{canon[0], canon[1], canon[2], canon[3]}));

    // Cell 1 sees a reflected face ordering on the shared face -> swap interior nodes (1 <-> 2).
    EXPECT_EQ(face1, (std::vector<GlobalIndex>{canon[0], canon[2], canon[1], canon[3]}));
}

TEST(DofHandler, DGDistributionInterleavedRenumberingUsesOptions) {
    DofHandler handler;
    auto topo = makeLineMesh(1);
    auto layout = DofLayoutInfo::DG(/*order=*/1, /*num_verts_per_cell=*/2, /*components=*/2);
    svmp::FE::dofs::DofDistributionOptions opts;
    opts.numbering = svmp::FE::dofs::DofNumberingStrategy::Interleaved;

    handler.distributeDofs(topo, layout, opts);
    handler.finalize();

    const auto dofs = handler.getCellDofs(0);
    ASSERT_EQ(dofs.size(), 4u);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()),
              (std::vector<GlobalIndex>{0, 2, 1, 3}));
}

TEST(DofHandler, CGDistributionVectorComponentsBlockLayout) {
    DofHandler handler;
    auto topo = makeLineMesh(1);
    auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/1, /*num_verts_per_cell=*/2, /*num_components=*/2);

    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_EQ(handler.getNumDofs(), 4);
    EXPECT_EQ(handler.getNumLocalDofs(), 4);

    const auto dofs = handler.getCellDofs(0);
    ASSERT_EQ(dofs.size(), 4u);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()),
              (std::vector<GlobalIndex>{0, 1, 2, 3}));

    const auto* entity = handler.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    EXPECT_EQ(std::vector<GlobalIndex>(entity->getVertexDofs(0).begin(), entity->getVertexDofs(0).end()),
              (std::vector<GlobalIndex>{0, 2}));
    EXPECT_EQ(std::vector<GlobalIndex>(entity->getVertexDofs(1).begin(), entity->getVertexDofs(1).end()),
              (std::vector<GlobalIndex>{1, 3}));
}

TEST(DofHandler, CGDistributionVectorComponentsInterleavedRenumbering) {
    DofHandler handler;
    auto topo = makeLineMesh(1);
    auto layout = DofLayoutInfo::Lagrange(/*order=*/1, /*dim=*/1, /*num_verts_per_cell=*/2, /*num_components=*/2);
    svmp::FE::dofs::DofDistributionOptions opts;
    opts.numbering = svmp::FE::dofs::DofNumberingStrategy::Interleaved;

    handler.distributeDofs(topo, layout, opts);
    handler.finalize();

    const auto dofs = handler.getCellDofs(0);
    ASSERT_EQ(dofs.size(), 4u);
    EXPECT_EQ(std::vector<GlobalIndex>(dofs.begin(), dofs.end()),
              (std::vector<GlobalIndex>{0, 2, 1, 3}));

    const auto* entity = handler.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    EXPECT_EQ(std::vector<GlobalIndex>(entity->getVertexDofs(0).begin(), entity->getVertexDofs(0).end()),
              (std::vector<GlobalIndex>{0, 1}));
    EXPECT_EQ(std::vector<GlobalIndex>(entity->getVertexDofs(1).begin(), entity->getVertexDofs(1).end()),
              (std::vector<GlobalIndex>{2, 3}));
}

TEST(DofHandler, CannotDistributeAfterFinalize) {
    DofHandler handler;
    auto topo = makeLineMesh(1);
    auto layout = DofLayoutInfo::Lagrange(1, 1, 2);
    handler.distributeDofs(topo, layout);
    handler.finalize();

    EXPECT_THROW(handler.distributeDofs(topo, layout), FEException);
}
