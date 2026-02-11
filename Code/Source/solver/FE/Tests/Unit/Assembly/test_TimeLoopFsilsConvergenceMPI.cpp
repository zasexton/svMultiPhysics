/**
 * @file test_TimeLoopFsilsConvergenceMPI.cpp
 * @brief MPI regression: TimeLoop + NewtonSolver + FSILS backend should converge on a small distributed transient problem.
 *
 * Motivation: The new OOP solver shows nonlinear convergence failures only in multi-rank runs.
 * This test exercises the end-to-end transient path under MPI:
 * - distributed DOF numbering/ownership and ghost exchange,
 * - FSILS DOF permutation + overlap vectors,
 * - TimeHistory::repack() (history vectors allocated before the Jacobian exists),
 * - Generalized-α (first-order) special handling of algebraic (non-dt) fields.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"

#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"

#include "Core/FEException.h"
#include "Core/Types.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Forms.h"
#include "Forms/Vocabulary.h"

#include "Spaces/SpaceFactory.h"

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TransientSystem.h"

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::testing {
namespace {

using svmp::FE::GlobalIndex;
using svmp::FE::Real;

int mpiRank(MPI_Comm comm)
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

int mpiSize(MPI_Comm comm)
{
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}

std::vector<int> neighborRanks(int my_rank, int world_size)
{
    std::vector<int> neighbors;
    neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size - 1)));
    for (int r = 0; r < world_size; ++r) {
        if (r != my_rank) {
            neighbors.push_back(r);
        }
    }
    return neighbors;
}

// 2D strip of Quad4 cells, with interleaved node IDs:
// x-index i has nodes {2*i (bottom), 2*i+1 (top)}.
class StripQuadMeshAccess final : public IMeshAccess {
public:
    StripQuadMeshAccess(int n_cells, int my_rank)
        : n_cells_(n_cells)
        , my_rank_(my_rank)
    {
        FE_THROW_IF(n_cells_ < 1, InvalidArgumentException, "StripQuadMeshAccess: n_cells must be >= 1");

        const int n_x = n_cells_ + 1;
        const int n_nodes = 2 * n_x;
        nodes_.resize(static_cast<std::size_t>(n_nodes));

        for (int i = 0; i < n_x; ++i) {
            const Real x = static_cast<Real>(i) / static_cast<Real>(n_cells_);
            nodes_[static_cast<std::size_t>(2 * i + 0)] = {x, 0.0, 0.0}; // bottom
            nodes_[static_cast<std::size_t>(2 * i + 1)] = {x, 1.0, 0.0}; // top
        }

        cells_.resize(static_cast<std::size_t>(n_cells_));
        for (int c = 0; c < n_cells_; ++c) {
            const GlobalIndex bl = static_cast<GlobalIndex>(2 * c + 0);
            const GlobalIndex br = static_cast<GlobalIndex>(2 * (c + 1) + 0);
            const GlobalIndex tr = static_cast<GlobalIndex>(2 * (c + 1) + 1);
            const GlobalIndex tl = static_cast<GlobalIndex>(2 * c + 1);
            cells_[static_cast<std::size_t>(c)] = {bl, br, tr, tl};
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cells_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        // We use a simple model: cell c is owned by rank c (requires n_cells == world_size in the test).
        return 1;
    }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return static_cast<int>(cell_id) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(c.begin(), c.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(c[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override { return {0, 0}; }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(static_cast<GlobalIndex>(my_rank_));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int n_cells_{0};
    int my_rank_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

[[nodiscard]] dofs::MeshTopologyInfo buildStripTopology(int n_cells, int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = static_cast<GlobalIndex>(n_cells);
    topo.n_vertices = static_cast<GlobalIndex>(2 * (n_cells + 1));

    topo.cell2vertex_offsets.resize(static_cast<std::size_t>(topo.n_cells) + 1, 0);
    topo.cell2vertex_data.resize(static_cast<std::size_t>(topo.n_cells) * 4);
    for (int c = 0; c < n_cells; ++c) {
        const std::size_t off = static_cast<std::size_t>(4 * c);
        topo.cell2vertex_offsets[static_cast<std::size_t>(c)] = static_cast<MeshOffset>(off);
        topo.cell2vertex_data[off + 0] = static_cast<MeshIndex>(2 * c + 0);
        topo.cell2vertex_data[off + 1] = static_cast<MeshIndex>(2 * (c + 1) + 0);
        topo.cell2vertex_data[off + 2] = static_cast<MeshIndex>(2 * (c + 1) + 1);
        topo.cell2vertex_data[off + 3] = static_cast<MeshIndex>(2 * c + 1);
    }
    topo.cell2vertex_offsets[static_cast<std::size_t>(topo.n_cells)] =
        static_cast<MeshOffset>(topo.cell2vertex_data.size());

    topo.vertex_gids.resize(static_cast<std::size_t>(topo.n_vertices));
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids[static_cast<std::size_t>(v)] = static_cast<dofs::gid_t>(v);
    }

    topo.cell_gids.resize(static_cast<std::size_t>(topo.n_cells));
    topo.cell_owner_ranks.resize(static_cast<std::size_t>(topo.n_cells));
    for (int c = 0; c < n_cells; ++c) {
        topo.cell_gids[static_cast<std::size_t>(c)] = static_cast<dofs::gid_t>(c);
        topo.cell_owner_ranks[static_cast<std::size_t>(c)] = c;
    }

    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

[[nodiscard]] std::shared_ptr<const backends::DofPermutation>
buildFsilsDofPermutation(const systems::FESystem& system,
                         int dof_per_node,
                         const dofs::DofDistributionOptions& dof_options)
{
    using backends::DofPermutation;

    if (dof_per_node <= 0) {
        return {};
    }

    const GlobalIndex total_dofs = system.dofHandler().getNumDofs();
    if (total_dofs <= 0) {
        return {};
    }

    // Derive node-block permutation from the field map (requires equal-order fields).
    const auto& fmap = system.fieldMap();
    const std::size_t n_fields = fmap.numFields();
    if (n_fields == 0u) {
        return {};
    }

    GlobalIndex n_nodes = -1;
    int expected_dof_per_node = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
        const auto& field = fmap.getField(f);
        expected_dof_per_node += field.n_components;
        if (field.n_components <= 0) {
            return {};
        }
        if (field.n_dofs % field.n_components != 0) {
            return {};
        }
        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_nodes < 0) {
            n_nodes = n_per_component;
        } else if (n_nodes != n_per_component) {
            return {};
        }
    }

    if (expected_dof_per_node != dof_per_node) {
        return {};
    }
    if (n_nodes <= 0) {
        return {};
    }
    if (total_dofs != static_cast<GlobalIndex>(dof_per_node) * n_nodes) {
        return {};
    }

    const bool explicit_spatial =
        dof_options.numbering == dofs::DofNumberingStrategy::Morton ||
        dof_options.numbering == dofs::DofNumberingStrategy::Hilbert;
    const bool default_spatial =
        dof_options.enable_spatial_locality_ordering &&
        dof_options.numbering == dofs::DofNumberingStrategy::Sequential;
    const bool want_spatial = explicit_spatial || default_spatial;
    const auto curve =
        explicit_spatial
            ? (dof_options.numbering == dofs::DofNumberingStrategy::Hilbert
                   ? dofs::SpatialCurveType::Hilbert
                   : dofs::SpatialCurveType::Morton)
            : dof_options.spatial_curve;

    constexpr std::uint32_t kSfcBits = 21u;
    constexpr std::uint64_t kSfcMaxCoord = (1ULL << kSfcBits) - 1ULL;

    auto morton3d = [](std::uint32_t xi, std::uint32_t yi, std::uint32_t zi) -> std::uint64_t {
        auto spread = [](std::uint64_t v) -> std::uint64_t {
            v = (v | (v << 32)) & 0x1f00000000ffffULL;
            v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
            v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
            v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
            v = (v | (v << 2)) & 0x1249249249249249ULL;
            return v;
        };
        return spread(xi) | (spread(yi) << 1) | (spread(zi) << 2);
    };

    auto hilbert_nd = [](const std::array<std::uint32_t, 3>& coords, std::uint32_t bits) -> std::uint64_t {
        std::array<std::uint32_t, 3> x = coords;
        const int n = 3;

        std::uint32_t M = 1u << (bits - 1u);
        for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
            const std::uint32_t P = Q - 1u;
            for (int i = 0; i < n; ++i) {
                if ((x[static_cast<std::size_t>(i)] & Q) != 0u) {
                    x[0] ^= P;
                } else {
                    const std::uint32_t t = (x[0] ^ x[static_cast<std::size_t>(i)]) & P;
                    x[0] ^= t;
                    x[static_cast<std::size_t>(i)] ^= t;
                }
            }
        }
        for (int i = 1; i < n; ++i) {
            x[static_cast<std::size_t>(i)] ^= x[static_cast<std::size_t>(i - 1)];
        }
        std::uint32_t t = 0u;
        for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
            if ((x[static_cast<std::size_t>(n - 1)] & Q) != 0u) {
                t ^= (Q - 1u);
            }
        }
        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] ^= t;
        }

        std::uint64_t index = 0;
        for (int b = static_cast<int>(bits) - 1; b >= 0; --b) {
            for (int i = 0; i < n; ++i) {
                index <<= 1u;
                index |= static_cast<std::uint64_t>((x[static_cast<std::size_t>(i)] >> static_cast<std::uint32_t>(b)) & 1u);
            }
        }
        return index;
    };

    auto sfc_code = [&](double x, double y, double z) -> std::uint64_t {
        auto normalize = [](double v) -> std::uint32_t {
            v = std::max(0.0, std::min(1.0, v));
            return static_cast<std::uint32_t>(v * static_cast<double>(kSfcMaxCoord));
        };
        const std::uint32_t xi = normalize(x);
        const std::uint32_t yi = normalize(y);
        const std::uint32_t zi = normalize(z);
        if (curve == dofs::SpatialCurveType::Hilbert) {
            return hilbert_nd(std::array<std::uint32_t, 3>{xi, yi, zi}, kSfcBits);
        }
        return morton3d(xi, yi, zi);
    };

    std::vector<int> node_owner(static_cast<std::size_t>(n_nodes), -1);
    std::vector<std::array<double, 3>> node_xyz(static_cast<std::size_t>(n_nodes),
                                                std::array<double, 3>{0.0, 0.0, 0.0});
    const auto* emap = system.dofHandler().getEntityDofMap();

    std::array<double, 3> min_xyz{std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_xyz{-std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()};
    const int dim = std::max(2, system.meshAccess().dimension());

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex fe0 = fmap.componentToGlobal(0, 0, node);
        const int owner0 = system.dofHandler().getDofMap().getDofOwner(fe0);
        if (owner0 < 0) {
            return {};
        }
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = fmap.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe = fmap.componentToGlobal(f, c, node);
                if (system.dofHandler().getDofMap().getDofOwner(fe) != owner0) {
                    return {};
                }
            }
        }

        node_owner[static_cast<std::size_t>(node)] = owner0;

        std::array<double, 3> xyz{0.0, 0.0, 0.0};
        bool have_xyz = false;
        if (want_spatial && emap) {
            if (const auto ent = emap->getDofEntity(fe0); ent && ent->kind == dofs::EntityKind::Vertex) {
                const auto p = system.meshAccess().getNodeCoordinates(ent->id);
                xyz = {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
                have_xyz = true;
            }
        }
        if (!have_xyz) {
            const double d = static_cast<double>(node);
            xyz = {d, d, d};
        }
        node_xyz[static_cast<std::size_t>(node)] = xyz;
        for (int a = 0; a < dim && a < 3; ++a) {
            min_xyz[static_cast<std::size_t>(a)] = std::min(min_xyz[static_cast<std::size_t>(a)], xyz[static_cast<std::size_t>(a)]);
            max_xyz[static_cast<std::size_t>(a)] = std::max(max_xyz[static_cast<std::size_t>(a)], xyz[static_cast<std::size_t>(a)]);
        }
    }

    auto norm = [&](double v, int axis) -> double {
        const auto ax = static_cast<std::size_t>(axis);
        const double lo = min_xyz[ax];
        const double hi = max_xyz[ax];
        if (!(hi > lo)) return 0.0;
        return (v - lo) / (hi - lo);
    };

    struct NodeKey {
        int owner{0};
        std::uint64_t code{0};
        GlobalIndex node{-1};
    };
    std::vector<NodeKey> ordering;
    ordering.reserve(static_cast<std::size_t>(n_nodes));
    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const auto& xyz = node_xyz[static_cast<std::size_t>(node)];
        const double x = norm(xyz[0], 0);
        const double y = (dim >= 2) ? norm(xyz[1], 1) : 0.0;
        const double z = (dim >= 3) ? norm(xyz[2], 2) : 0.0;
        const std::uint64_t code = want_spatial ? sfc_code(x, y, z) : 0u;
        ordering.push_back(NodeKey{node_owner[static_cast<std::size_t>(node)], code, node});
    }
    std::sort(ordering.begin(), ordering.end(), [&](const NodeKey& a, const NodeKey& b) {
        if (a.owner != b.owner) return a.owner < b.owner;
        if (a.code != b.code) return a.code < b.code;
        return a.node < b.node;
    });

    std::vector<GlobalIndex> node_to_backend(static_cast<std::size_t>(n_nodes), INVALID_GLOBAL_INDEX);
    for (GlobalIndex i = 0; i < n_nodes; ++i) {
        node_to_backend[static_cast<std::size_t>(ordering[static_cast<std::size_t>(i)].node)] = i;
    }

    auto perm = std::make_shared<DofPermutation>();
    perm->forward.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
    perm->inverse.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        const GlobalIndex backend_node = node_to_backend[static_cast<std::size_t>(node)];
        if (backend_node < 0) {
            return {};
        }
        int comp_offset = 0;
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = fmap.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe_dof = fmap.componentToGlobal(f, c, node);
                const GlobalIndex fs_dof =
                    backend_node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
                if (fe_dof < 0 || fe_dof >= total_dofs) {
                    return {};
                }
                perm->forward[static_cast<std::size_t>(fe_dof)] = fs_dof;
                perm->inverse[static_cast<std::size_t>(fs_dof)] = fe_dof;
                ++comp_offset;
            }
        }
        if (comp_offset != dof_per_node) {
            return {};
        }
    }

    if (std::any_of(perm->forward.begin(), perm->forward.end(),
                    [](GlobalIndex v) { return v == INVALID_GLOBAL_INDEX; })) {
        return {};
    }
    if (std::any_of(perm->inverse.begin(), perm->inverse.end(),
                    [](GlobalIndex v) { return v == INVALID_GLOBAL_INDEX; })) {
        return {};
    }

    return perm;
}

backends::SolverOptions fsilsGmresDiagOptions()
{
    backends::SolverOptions o;
    o.method = backends::SolverMethod::GMRES;
    o.preconditioner = backends::PreconditionerType::Diagonal;
    o.rel_tol = 1e-10;
    o.abs_tol = 1e-12;
    o.max_iter = 5000;
    return o;
}

} // namespace

TEST(TimeLoopFsilsConvergenceMPI, GeneralizedAlphaConvergesWithAlgebraicField)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size < 2) {
        GTEST_SKIP() << "Run with 2+ MPI ranks to enable this test";
    }

    auto run_case = [&](bool deterministic_mode) {
        SCOPED_TRACE(deterministic_mode ? "deterministic_on" : "deterministic_off");

        // One owned cell per rank; all cells present as ghosts to enable OwnedRowsOnly-style assembly.
        const int n_cells = size;
        auto mesh = std::make_shared<StripQuadMeshAccess>(n_cells, rank);

        const auto u_space = spaces::VectorSpace(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/2);
        const auto p_space = spaces::Space(spaces::SpaceType::H1, ElementType::Quad4, /*order=*/1, /*components=*/1);
        ASSERT_TRUE(u_space);
        ASSERT_TRUE(p_space);

        systems::FESystem sys(mesh);
        const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = u_space, .components = 2});
        const auto p_field = sys.addField(systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
        sys.addOperator("op");

        const auto u = forms::FormExpr::stateField(u_field, *u_space, "u");
        const auto p = forms::FormExpr::stateField(p_field, *p_space, "p");
        const auto v = forms::TestFunction(*u_space, "v");
        const auto q = forms::TestFunction(*p_space, "q");

        // Transient nonlinear reaction for velocity; algebraic pressure mass term.
        const auto one = forms::FormExpr::constant(Real(1.0));
        const auto lambda = forms::FormExpr::constant(Real(2.0));
        const auto eps = forms::FormExpr::constant(Real(0.05));
        const auto kappa = forms::FormExpr::constant(Real(1.0));

        forms::BlockLinearForm residual(/*tests=*/2);
        residual.setBlock(
            0,
            (forms::inner(u.dt(1), v) +
             lambda * forms::inner(u, v) +
             eps * (one + forms::inner(u, u)) * forms::inner(u, v))
                .dx());
        residual.setBlock(1, (kappa * p * q).dx());

        const std::array<FieldId, 2> fields = {u_field, p_field};
        (void)systems::installCoupledResidual(
            sys, "op",
            fields,
            fields,
            residual,
            systems::FormInstallOptions{.ad_mode = forms::ADMode::Forward});

        systems::SetupOptions setup_opts;
        setup_opts.assembler_name = "StandardAssembler";
        setup_opts.assembly_options.ghost_policy = GhostPolicy::ReverseScatter;
        setup_opts.assembly_options.deterministic = deterministic_mode;
        setup_opts.assembly_options.overlap_communication = false;

        // FSILS distributed layout requires each rank's owned nodes to be contiguous in node space.
        // Use dense, process-count-independent global IDs (non-owner-contiguous) to force the
        // node-interleaved distributed sparsity path, and a deterministic ownership strategy that
        // yields contiguous node blocks for this strip topology.
        setup_opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
        setup_opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
        setup_opts.dof_options.my_rank = rank;
        setup_opts.dof_options.world_size = size;
        setup_opts.dof_options.mpi_comm = comm;

        systems::SetupInputs inputs;
        inputs.topology_override = buildStripTopology(n_cells, rank, size);

        sys.setup(setup_opts, inputs);
        ASSERT_TRUE(sys.isSetup());

        const GlobalIndex n_dofs = sys.dofHandler().getNumDofs();
        ASSERT_TRUE(inputs.topology_override.has_value());
        const GlobalIndex n_nodes = inputs.topology_override->n_vertices;
        ASSERT_EQ(n_dofs, n_nodes * 3);

        constexpr int dof_per_node = 3;
        auto perm = buildFsilsDofPermutation(sys, dof_per_node, setup_opts.dof_options);
        ASSERT_TRUE(perm) << "Failed to build FSILS DOF permutation for test system";

        backends::FsilsFactory factory(dof_per_node, perm);
        auto linear = factory.createLinearSolver(fsilsGmresDiagOptions());
        ASSERT_TRUE(linear);

        // Allocate history before any matrix exists: this creates local-only vectors that must be repacked.
        auto history = timestepping::TimeHistory::allocate(factory, n_dofs, /*history_depth=*/2,
                                                           /*allocate_second_order_state=*/false);
        const double dt = 0.05;
        history.setTime(0.0);
        history.setDt(dt);
        history.setPrevDt(dt);
        history.setStepIndex(0);

        // Initialize u^{n-1}, u^{n-2} (and current u) to a nontrivial state.
        auto init = [&](backends::GenericVector& vec, double scale) {
            auto s = vec.localSpan();
            ASSERT_EQ(static_cast<GlobalIndex>(s.size()), n_dofs);
            for (GlobalIndex i = 0; i < n_dofs; ++i) {
                s[static_cast<std::size_t>(i)] = static_cast<Real>(scale) * static_cast<Real>(0.01) * static_cast<Real>(i + 1);
            }
        };
        init(history.uPrev(), /*scale=*/1.0);
        init(history.uPrev2(), /*scale=*/1.0);
        history.resetCurrentToPrevious();

        auto base_integrator = std::make_shared<systems::BackwardDifferenceIntegrator>();
        systems::TransientSystem transient(sys, std::move(base_integrator));

        timestepping::TimeLoopOptions loop_opts;
        loop_opts.t0 = 0.0;
        loop_opts.t_end = 3.0 * dt;
        loop_opts.dt = dt;
        loop_opts.max_steps = 3;
        loop_opts.scheme = timestepping::SchemeKind::GeneralizedAlpha;
        loop_opts.generalized_alpha_rho_inf = 0.5;
        loop_opts.newton.residual_op = "op";
        loop_opts.newton.jacobian_op = "op";
        loop_opts.newton.max_iterations = 12;
        loop_opts.newton.abs_tolerance = 1e-12;
        loop_opts.newton.rel_tolerance = 1e-10;

        timestepping::TimeLoop loop(loop_opts);

        int nonconverged_steps = 0;
        timestepping::TimeLoopCallbacks callbacks;
        callbacks.on_nonlinear_done = [&nonconverged_steps](const timestepping::TimeHistory&, const timestepping::NewtonReport& nr) {
            if (!nr.converged) {
                ++nonconverged_steps;
            }
        };

        timestepping::TimeLoopReport rep{};
        try {
            rep = loop.run(transient, factory, *linear, history, callbacks);
        } catch (const FEException& e) {
            ADD_FAILURE() << "Rank " << rank << ": TimeLoop threw FEException: " << e.what();
            return;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Rank " << rank << ": TimeLoop threw std::exception: " << e.what();
            return;
        }

        EXPECT_TRUE(rep.success);
        EXPECT_NEAR(rep.final_time, loop_opts.t_end, 1e-12);
        EXPECT_EQ(nonconverged_steps, 0);
    };

    run_case(/*deterministic_mode=*/true);
    run_case(/*deterministic_mode=*/false);
#endif
}

} // namespace svmp::FE::assembly::testing
