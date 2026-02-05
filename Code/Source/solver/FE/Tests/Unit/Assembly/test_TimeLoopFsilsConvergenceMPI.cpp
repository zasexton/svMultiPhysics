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
buildFsilsDofPermutation(const systems::FESystem& system, int dof_per_node)
{
    using backends::DofPermutation;

    if (dof_per_node <= 0) {
        return {};
    }

    const GlobalIndex total_dofs = system.dofHandler().getNumDofs();
    if (total_dofs <= 0) {
        return {};
    }

    // Prefer the entity map (robust against FE component-layout assumptions).
    if (const auto* emap = system.dofHandler().getEntityDofMap()) {
        const GlobalIndex n_vertices = emap->numVertices();
        if (n_vertices > 0 && total_dofs == static_cast<GlobalIndex>(dof_per_node) * n_vertices) {
            auto perm = std::make_shared<DofPermutation>();
            perm->forward.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
            perm->inverse.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

            for (GlobalIndex v = 0; v < n_vertices; ++v) {
                const auto vdofs = emap->getVertexDofs(v);
                if (vdofs.size() != static_cast<std::size_t>(dof_per_node)) {
                    return {};
                }
                for (std::size_t c = 0; c < vdofs.size(); ++c) {
                    const GlobalIndex fe_dof = vdofs[c];
                    const GlobalIndex fs_dof = v * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(c);
                    if (fe_dof < 0 || fe_dof >= total_dofs) {
                        return {};
                    }
                    perm->forward[static_cast<std::size_t>(fe_dof)] = fs_dof;
                    perm->inverse[static_cast<std::size_t>(fs_dof)] = fe_dof;
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
    }

    // Fallback: derive node-block permutation from the field map.
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

    auto perm = std::make_shared<DofPermutation>();
    perm->forward.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
    perm->inverse.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

    for (GlobalIndex node = 0; node < n_nodes; ++node) {
        int comp_offset = 0;
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = fmap.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe_dof = fmap.componentToGlobal(f, c, node);
                const GlobalIndex fs_dof =
                    node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
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
    setup_opts.assembly_options.deterministic = true;
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
    auto perm = buildFsilsDofPermutation(sys, dof_per_node);
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
#endif
}

} // namespace svmp::FE::assembly::testing
