/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NormalComponentConstraintMPI.cpp
 * @brief u . n = g strongly on curved and faceted boundaries, at one rank and in parallel
 *
 * The meshes are the structured grid of [-1, 1]^d with n cells per side, as
 * quadrilaterals or hexahedra or split into triangles or (Kuhn) tetrahedra,
 * either as they are (the faceted box: walls, edges and corners) or mapped to
 * the unit ball by x |x|_inf / |x|_2 (a faceted circle or sphere whose
 * vertices lie on the curve).  Every rank sees every cell; the cells are
 * split into contiguous blocks over the ranks, and the DOFs are numbered by
 * global ids, so the numbering does not depend on the rank count.  Each test
 * builds the system on MPI_COMM_SELF (the one-rank reference, where it also
 * checks the lines) and on MPI_COMM_WORLD, and requires the lines of the
 * owned slaves, gathered over the ranks, to equal the reference bit for bit.
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/NormalComponentConstraint.h"
#include "Constraints/StrongDirichletConstraint.h"
#include "Elements/ReferenceElement.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::constraints::test_normal_component {
namespace {

using forms::FormExpr;

constexpr int kWall = 1;
constexpr int kSide = 2;

struct StructuredMesh {
    int dim{2};
    ElementType type{ElementType::Quad4};
    std::vector<std::array<Real, 3>> x{};
    std::vector<std::vector<GlobalIndex>> cells{};
    struct Face {
        GlobalIndex cell{-1};
        LocalIndex local{0};
        int marker{-1};
    };
    std::vector<Face> faces{};
};

/// The grid of [-1, 1]^dim with n cells per side; boundary faces get kWall,
/// or kSide on x = -1 when @p mark_side.
StructuredMesh structuredMesh(int dim, int n, bool simplex, bool curved, bool mark_side = false)
{
    StructuredMesh m;
    m.dim = dim;
    m.type = dim == 2 ? (simplex ? ElementType::Triangle3 : ElementType::Quad4)
                      : (simplex ? ElementType::Tetra4 : ElementType::Hex8);
    const int nz = dim == 3 ? n : 0;
    const auto id = [&](int i, int j, int k) {
        return static_cast<GlobalIndex>((k * (n + 1) + j) * (n + 1) + i);
    };
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                std::array<Real, 3> p{-1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n, dim == 3 ? -1.0 + 2.0 * k / n : 0.0};
                if (curved) {
                    const Real inf = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
                    const Real two = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                    if (two > 0.0) {
                        for (auto& c : p) {
                            c *= inf / two;
                        }
                    }
                }
                m.x.push_back(p);
            }
        }
    }
    const auto volume = [&](const std::vector<GlobalIndex>& t) {
        const auto& a = m.x[static_cast<std::size_t>(t[0])];
        std::array<std::array<Real, 3>, 3> e{};
        for (std::size_t r = 1; r < t.size(); ++r) {
            for (int c = 0; c < 3; ++c) {
                e[r - 1][static_cast<std::size_t>(c)] = m.x[static_cast<std::size_t>(t[r])][static_cast<std::size_t>(c)] -
                                                        a[static_cast<std::size_t>(c)];
            }
        }
        if (t.size() == 3) {
            return e[0][0] * e[1][1] - e[0][1] * e[1][0];
        }
        return e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
               e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    };
    for (int k = 0; k < std::max(nz, 1); ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const auto v = [&](int a, int b, int c) { return id(i + a, j + b, dim == 3 ? k + c : 0); };
                if (dim == 2 && !simplex) {
                    m.cells.push_back({v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)});
                } else if (dim == 2) {
                    m.cells.push_back({v(0, 0, 0), v(1, 0, 0), v(1, 1, 0)});
                    m.cells.push_back({v(0, 0, 0), v(1, 1, 0), v(0, 1, 0)});
                } else if (!simplex) {
                    m.cells.push_back({v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0), v(0, 0, 1), v(1, 0, 1),
                                       v(1, 1, 1), v(0, 1, 1)});
                } else {
                    // Kuhn: the six paths from (0,0,0) to (1,1,1) along the axes.
                    const std::array<std::array<int, 3>, 6> orders{
                        {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
                    for (const auto& order : orders) {
                        std::array<int, 3> b{0, 0, 0};
                        std::vector<GlobalIndex> t{v(0, 0, 0)};
                        for (const int axis : order) {
                            b[static_cast<std::size_t>(axis)] = 1;
                            t.push_back(v(b[0], b[1], b[2]));
                        }
                        if (volume(t) < 0.0) {
                            std::swap(t[1], t[2]);
                        }
                        m.cells.push_back(std::move(t));
                    }
                }
            }
        }
    }
    // Boundary faces: the reference faces held by one cell.
    const auto& reference = elements::ReferenceElement::shared(m.type);
    std::map<std::vector<GlobalIndex>, std::vector<std::pair<GlobalIndex, LocalIndex>>> incidence;
    for (GlobalIndex c = 0; c < static_cast<GlobalIndex>(m.cells.size()); ++c) {
        for (std::size_t f = 0; f < reference.num_faces(); ++f) {
            std::vector<GlobalIndex> key;
            for (const auto node : reference.face_nodes(f)) {
                key.push_back(m.cells[static_cast<std::size_t>(c)][static_cast<std::size_t>(node)]);
            }
            std::sort(key.begin(), key.end());
            incidence[key].emplace_back(c, static_cast<LocalIndex>(f));
        }
    }
    for (const auto& [key, holders] : incidence) {
        if (holders.size() != 1u) {
            continue;
        }
        const bool side = std::all_of(key.begin(), key.end(), [&](GlobalIndex vtx) {
            return (vtx % (n + 1)) == 0;   // grid index i = 0, the face x = -1
        });
        m.faces.push_back({holders[0].first, holders[0].second, mark_side && side ? kSide : kWall});
    }
    return m;
}

/// The mesh on one rank: every cell visible, contiguous blocks owned.
class ReplicatedMeshAccess final : public assembly::IMeshAccess {
public:
    ReplicatedMeshAccess(std::shared_ptr<const StructuredMesh> mesh, int rank, int size)
        : mesh_(std::move(mesh))
        , rank_(rank)
        , size_(size)
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                owned_.push_back(c);
            }
        }
    }

    [[nodiscard]] int owner(GlobalIndex cell) const { return static_cast<int>((cell * size_) / numCells()); }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(mesh_->cells.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return static_cast<GlobalIndex>(owned_.size()); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return static_cast<GlobalIndex>(mesh_->faces.size()); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return mesh_->dim; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override { return true; }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell) const override { return owner(cell); }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override { return owner(cell) == rank_; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return mesh_->type; }

    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        nodes = mesh_->cells.at(static_cast<std::size_t>(cell));
    }
    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node) const override
    {
        return mesh_->x.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(GlobalIndex cell, std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& nodes = mesh_->cells.at(static_cast<std::size_t>(cell));
        coords.resize(nodes.size());
        for (std::size_t k = 0; k < nodes.size(); ++k) {
            coords[k] = mesh_->x[static_cast<std::size_t>(nodes[k])];
        }
    }
    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face, GlobalIndex) const override
    {
        return mesh_->faces.at(static_cast<std::size_t>(face)).local;
    }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face) const override
    {
        return mesh_->faces.at(static_cast<std::size_t>(face)).marker;
    }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }
    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (const auto c : owned_) {
            callback(c);
        }
    }
    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        for (GlobalIndex f = 0; f < numBoundaryFaces(); ++f) {
            const auto& face = mesh_->faces[static_cast<std::size_t>(f)];
            if ((marker < 0 || face.marker == marker) && isOwnedCell(face.cell)) {
                callback(f, face.cell);
            }
        }
    }
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

    /// Cells, vertices, edges and faces with their ids as global ids.
    [[nodiscard]] dofs::MeshTopologyInfo topology() const
    {
        dofs::MeshTopologyInfo topo;
        topo.dim = mesh_->dim;
        topo.n_cells = numCells();
        topo.n_vertices = static_cast<GlobalIndex>(mesh_->x.size());
        topo.cell2vertex_offsets.push_back(0);
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            for (const auto node : mesh_->cells[static_cast<std::size_t>(c)]) {
                topo.cell2vertex_data.push_back(static_cast<MeshIndex>(node));
            }
            topo.cell2vertex_offsets.push_back(static_cast<MeshOffset>(topo.cell2vertex_data.size()));
            topo.cell_gids.push_back(c);
            topo.cell_owner_ranks.push_back(owner(c));
        }
        for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
            topo.vertex_gids.push_back(v);
        }
        const auto& reference = elements::ReferenceElement::shared(mesh_->type);
        std::map<std::pair<MeshIndex, MeshIndex>, MeshIndex> edges;
        std::map<std::vector<MeshIndex>, MeshIndex> faces;
        topo.cell2edge_offsets.push_back(0);
        topo.cell2face_offsets.push_back(0);
        topo.face2vertex_offsets.push_back(0);
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            const auto& nodes = mesh_->cells[static_cast<std::size_t>(c)];
            for (std::size_t e = 0; e < reference.num_edges(); ++e) {
                const auto& ends = reference.edge_nodes(e);
                const auto a = static_cast<MeshIndex>(nodes[static_cast<std::size_t>(ends[0])]);
                const auto b = static_cast<MeshIndex>(nodes[static_cast<std::size_t>(ends[1])]);
                const auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto [it, inserted] = edges.emplace(key, static_cast<MeshIndex>(edges.size()));
                if (inserted) {
                    topo.edge2vertex_data.push_back(key.first);
                    topo.edge2vertex_data.push_back(key.second);
                    topo.edge_gids.push_back(static_cast<dofs::gid_t>(it->second));
                }
                topo.cell2edge_data.push_back(it->second);
            }
            topo.cell2edge_offsets.push_back(static_cast<MeshOffset>(topo.cell2edge_data.size()));
            if (mesh_->dim == 3) {
                for (std::size_t f = 0; f < reference.num_faces(); ++f) {
                    std::vector<MeshIndex> corners;
                    for (const auto node : reference.face_nodes(f)) {
                        corners.push_back(static_cast<MeshIndex>(nodes[static_cast<std::size_t>(node)]));
                    }
                    auto key = corners;
                    std::sort(key.begin(), key.end());
                    auto [it, inserted] = faces.emplace(key, static_cast<MeshIndex>(faces.size()));
                    if (inserted) {
                        topo.face2vertex_data.insert(topo.face2vertex_data.end(), corners.begin(), corners.end());
                        topo.face2vertex_offsets.push_back(static_cast<MeshOffset>(topo.face2vertex_data.size()));
                        topo.face_gids.push_back(static_cast<dofs::gid_t>(it->second));
                    }
                    topo.cell2face_data.push_back(it->second);
                }
                topo.cell2face_offsets.push_back(static_cast<MeshOffset>(topo.cell2face_data.size()));
            }
        }
        topo.n_edges = static_cast<GlobalIndex>(edges.size());
        if (mesh_->dim == 3) {
            topo.n_faces = static_cast<GlobalIndex>(faces.size());
        } else {
            topo.cell2face_offsets.clear();
            topo.face2vertex_offsets.clear();
        }
        for (int r = 0; r < size_; ++r) {
            if (r != rank_) {
                topo.neighbor_ranks.push_back(r);
            }
        }
        return topo;
    }

private:
    std::shared_ptr<const StructuredMesh> mesh_;
    int rank_{0};
    int size_{1};
    std::vector<GlobalIndex> owned_{};
};

struct Case {
    int dim{2};
    bool simplex{false};
    int order{1};
    bool curved{true};

    [[nodiscard]] std::string name() const
    {
        return std::to_string(dim) + "D " + (simplex ? "simplex" : "tensor") + " P" + std::to_string(order) +
               (curved ? " ball" : " box");
    }
};

struct BuiltSystem {
    std::unique_ptr<systems::FESystem> system{};
    const NormalComponentConstraint* constraint{nullptr};
    FieldId u{INVALID_FIELD_ID};
};

/// u . n = g on the wall; with @p side_dirichlet, u = 0 on the x = -1 side first.
BuiltSystem build(const Case& c, MPI_Comm comm, const FormExpr& g, bool side_dirichlet = false)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int n = c.dim == 2 ? 4 : 3;
    auto mesh_data = std::make_shared<const StructuredMesh>(structuredMesh(c.dim, n, c.simplex, c.curved, side_dirichlet));
    auto mesh = std::make_shared<ReplicatedMeshAccess>(mesh_data, rank, size);
    auto scalar = std::make_shared<spaces::H1Space>(mesh_data->type, c.order);
    auto vector = std::make_shared<spaces::ProductSpace>(scalar, c.dim);

    BuiltSystem out;
    out.system = std::make_unique<systems::FESystem>(mesh);
    auto& sys = *out.system;
    out.u = sys.addField(systems::FieldSpec{.name = "u", .space = vector, .components = c.dim});
    sys.addOperator("mass");
    const auto u = forms::StateField(out.u, *vector, "u");
    const auto v = forms::TestField(out.u, *vector, "v");
    (void)systems::installFormulation(sys, "mass", {out.u}, forms::inner(u, v).dx());
    if (side_dirichlet) {
        for (int d = 0; d < c.dim; ++d) {
            sys.addSystemConstraint(
                std::make_unique<StrongDirichletConstraint>(out.u, kSide, FormExpr::constant(0.0), d));
        }
    }
    auto constraint = std::make_unique<NormalComponentConstraint>(out.u, std::vector<int>{kWall}, g);
    out.constraint = constraint.get();
    sys.addSystemConstraint(std::move(constraint));

    systems::SetupOptions options;
    options.dof_options.global_numbering = dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = comm;
    systems::SetupInputs inputs;
    inputs.topology_override = mesh->topology();
    sys.setup(options, inputs);
    return out;
}

/// The lines of every rank, by slave, each as {masters..., weights..., value_factor, x...}.
std::map<GlobalIndex, std::vector<Real>> gatheredLines(const BuiltSystem& s, MPI_Comm comm)
{
    std::vector<Real> local;
    for (const auto& line : s.constraint->lines()) {
        local.push_back(static_cast<Real>(line.slave));
        local.push_back(static_cast<Real>(line.masters.size()));
        for (const auto m : line.masters) {
            local.push_back(static_cast<Real>(m));
        }
        local.insert(local.end(), line.weights.begin(), line.weights.end());
        local.push_back(line.value_factor);
        local.insert(local.end(), line.x.begin(), line.x.end());
        // The closed system constraint of the slave.
        const auto view = s.system->constraints().getConstraint(line.slave);
        local.push_back(view.has_value() ? view->inhomogeneity : std::nan(""));
    }
    int size = 1;
    MPI_Comm_size(comm, &size);
    int count = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] = displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }
    std::vector<Real> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

    std::map<GlobalIndex, std::vector<Real>> out;
    for (std::size_t k = 0; k < all.size();) {
        const auto slave = static_cast<GlobalIndex>(all[k]);
        const auto masters = static_cast<std::size_t>(all[k + 1]);
        const std::size_t length = 2 + 2 * masters + 1 + 3 + 1;
        EXPECT_TRUE(out.emplace(slave, std::vector<Real>(all.begin() + static_cast<std::ptrdiff_t>(k + 1),
                                                          all.begin() + static_cast<std::ptrdiff_t>(k + length)))
                        .second)
            << "slave " << slave << " constrained twice";
        k += length;
    }
    return out;
}

const std::vector<Case>& allCases()
{
    static const std::vector<Case> cases = [] {
        std::vector<Case> out;
        for (const int dim : {2, 3}) {
            for (const bool simplex : {true, false}) {
                for (const int order : {1, 2}) {
                    for (const bool curved : {true, false}) {
                        out.push_back({dim, simplex, order, curved});
                    }
                }
            }
        }
        return out;
    }();
    return cases;
}

[[nodiscard]] FormExpr spatialData()
{
    return FormExpr::constant(0.25) + FormExpr::constant(0.5) * forms::x().component(0);
}

} // namespace

TEST(NormalComponentConstraintMPI, LinesAreTheSameAtEveryRankCount)
{
    for (const auto& c : allCases()) {
        SCOPED_TRACE(c.name());
        const auto reference = build(c, MPI_COMM_SELF, spatialData());
        const auto distributed = build(c, MPI_COMM_WORLD, spatialData());
        const auto expected = gatheredLines(reference, MPI_COMM_SELF);
        const auto actual = gatheredLines(distributed, MPI_COMM_WORLD);
        ASSERT_FALSE(expected.empty());
        ASSERT_EQ(actual.size(), expected.size());
        for (const auto& [slave, line] : expected) {
            const auto it = actual.find(slave);
            ASSERT_NE(it, actual.end()) << "slave " << slave;
            ASSERT_EQ(it->second.size(), line.size()) << "slave " << slave;
            for (std::size_t k = 0; k < line.size(); ++k) {
                EXPECT_EQ(it->second[k], line[k]) << "slave " << slave << " entry " << k;
            }
        }
    }
}

TEST(NormalComponentConstraintMPI, CurvedWallsGetOneLineAlongTheAveragedNormal)
{
    for (const auto& c : allCases()) {
        if (!c.curved) {
            continue;
        }
        SCOPED_TRACE(c.name());
        const auto g = spatialData();
        const auto s = build(c, MPI_COMM_SELF, g);
        std::set<GlobalIndex> slaves;
        for (const auto& line : s.constraint->lines()) {
            EXPECT_TRUE(slaves.insert(line.slave).second);
            // One line per point, with the other components as masters (a
            // zero weight is left out: a normal along an axis has none), and
            // the implied normal n = n_s (e_s - sum_r w_r e_r) of unit length.
            ASSERT_LE(line.masters.size(), static_cast<std::size_t>(c.dim - 1));
            Real norm2 = 1.0;
            for (const auto w : line.weights) {
                norm2 += w * w;
                EXPECT_LE(std::abs(w), 1.0 + 1e-12);   // the slave is the largest component
            }
            const Real radius = std::sqrt(line.x[0] * line.x[0] + line.x[1] * line.x[1] + line.x[2] * line.x[2]);
            // Vertices lie on the sphere, the other nodes of P2 on its chords.
            if (c.order == 1) {
                EXPECT_NEAR(radius, 1.0, 1e-12);
            } else {
                EXPECT_LE(radius, 1.0 + 1e-12);
                EXPECT_GT(radius, 0.9);
            }
            // n . u = g with u_s from the line and the masters zero: n_s c = g.
            const Real g_at_x = 0.25 + 0.5 * line.x[0];
            const Real n_s = std::abs(1.0 / std::sqrt(norm2));
            EXPECT_NEAR(std::abs(line.value_factor * g_at_x) * n_s, std::abs(g_at_x), 1e-13);
            const auto view = s.system->constraints().getConstraint(line.slave);
            ASSERT_TRUE(view.has_value());
            EXPECT_DOUBLE_EQ(view->inhomogeneity, g_at_x * line.value_factor);
        }
        EXPECT_FALSE(slaves.empty());
    }
}

TEST(NormalComponentConstraintMPI, BoxWallsEdgesAndCornersPrescribeTheirNormalComponents)
{
    for (const auto& c : allCases()) {
        if (c.curved) {
            continue;
        }
        SCOPED_TRACE(c.name());
        const auto s = build(c, MPI_COMM_SELF, spatialData());
        // On the faceted box every normal is an axis: each line prescribes one
        // component, u_i = g n_i, without masters.
        std::map<std::array<Real, 3>, int> lines_at_point;
        for (const auto& line : s.constraint->lines()) {
            EXPECT_TRUE(line.masters.empty());
            EXPECT_EQ(std::abs(line.value_factor), 1.0);
            ++lines_at_point[line.x];
        }
        // A point holds one line per box face it lies on.
        for (const auto& [x, count] : lines_at_point) {
            int faces = 0;
            for (int d = 0; d < c.dim; ++d) {
                faces += std::abs(std::abs(x[static_cast<std::size_t>(d)]) - 1.0) < 1e-12 ? 1 : 0;
            }
            EXPECT_EQ(count, faces) << "point (" << x[0] << ", " << x[1] << ", " << x[2] << ")";
        }
        // P1: 4n + 4 lines in 2D (n = 4), 6(n-1)^2 + 24(n-1) + 24 in 3D (n = 3).
        if (c.order == 1) {
            EXPECT_EQ(s.constraint->lines().size(), c.dim == 2 ? 20u : 96u);
        }
    }
}

TEST(NormalComponentConstraintMPI, PointsAnEarlierConstraintHoldsAreLeftToIt)
{
    for (const auto& c : allCases()) {
        if (c.curved || c.order != 1) {
            continue;
        }
        SCOPED_TRACE(c.name());
        const auto reference = build(c, MPI_COMM_SELF, spatialData(), /*side_dirichlet=*/true);
        for (const auto& line : reference.constraint->lines()) {
            EXPECT_GT(line.x[0], -1.0 + 1e-12) << "a point of the x = -1 side";
            const auto view = reference.system->constraints().getConstraint(line.slave);
            ASSERT_TRUE(view.has_value());
        }
        const auto distributed = build(c, MPI_COMM_WORLD, spatialData(), /*side_dirichlet=*/true);
        EXPECT_EQ(gatheredLines(distributed, MPI_COMM_WORLD), gatheredLines(reference, MPI_COMM_SELF));
    }
}

TEST(NormalComponentConstraintMPI, TimeDependentDataUpdatesTheInhomogeneities)
{
    const Case c{.dim = 2, .simplex = true, .order = 2, .curved = true};
    const auto g = forms::t() + forms::x().component(1);
    auto s = build(c, MPI_COMM_WORLD, g);
    EXPECT_TRUE(s.system->hasTimeDependentConstraints());
    s.system->updateConstraints(/*time=*/0.75, /*dt=*/0.25);
    for (const auto& line : s.constraint->lines()) {
        EXPECT_EQ(s.system->constraints().getInhomogeneity(line.slave), (0.75 + line.x[1]) * line.value_factor);
    }
}

TEST(NormalComponentConstraintMPI, RefusesScalarFieldsAndBadArguments)
{
    EXPECT_THROW(NormalComponentConstraint(0, std::vector<int>{}), std::invalid_argument);
    EXPECT_THROW(NormalComponentConstraint(0, std::vector<int>{-1}), std::invalid_argument);
    EXPECT_THROW(NormalComponentConstraint(0, std::vector<int>{1}, FormExpr::constant(0.0),
                                           NormalComponentOptions{.feature_angle_degrees = 95.0}),
                 std::invalid_argument);

    auto mesh_data = std::make_shared<const StructuredMesh>(structuredMesh(2, 2, true, true));
    auto mesh = std::make_shared<ReplicatedMeshAccess>(mesh_data, 0, 1);
    auto scalar = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    systems::FESystem sys(mesh);
    const auto p = sys.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
    sys.addOperator("mass");
    (void)systems::installFormulation(sys, "mass", {p},
                                      (forms::StateField(p, *scalar, "p") * forms::TestField(p, *scalar, "q")).dx());
    sys.addSystemConstraint(std::make_unique<NormalComponentConstraint>(p, std::vector<int>{kWall}));
    systems::SetupOptions options;
    options.dof_options.mpi_comm = MPI_COMM_SELF;
    systems::SetupInputs inputs;
    inputs.topology_override = mesh->topology();
    EXPECT_THROW(sys.setup(options, inputs), InvalidArgumentException);
}

} // namespace svmp::FE::constraints::test_normal_component
