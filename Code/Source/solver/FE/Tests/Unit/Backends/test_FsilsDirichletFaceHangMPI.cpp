/* Minimal reproduction (not committed): FSILS GMRES with Dirichlet faces hangs
 * when a rank's Dirichlet face is empty while two or more other ranks have
 * nonempty faces.  Copy into Code/Source/solver/FE/Tests/Unit/Backends/ (it is
 * picked up by the test_*MPI*.cpp glob of test_fe_backends_mpi), rebuild, then
 *   timeout -k 30 120 mpiexec -n 3 build/bin/test_fe_backends_mpi \
 *       --gtest_filter='FsilsDirichletFaceHangMPI.*'
 * Expected: the test hangs (timeout exit 124) on 3 or more ranks; it passes on 1
 * and 2 ranks.  Cause: fsils_reduce_shared_face_values_owned_row() returns early
 * on a rank with an empty face (face_nodes.size() <= 0), while the ranks with
 * nonempty faces call fsils_reverse_scatterv_contribution_buffer() and
 * fsils_syncv_owned_to_ghost(), whose neighbour exchange then waits for it
 * (FsilsLinearSolver.cpp sync_face_val_if_shared() enters it when total_has > 1).
 */
#include <gtest/gtest.h>

#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsLinearSolver.h"
#include "FsilsDirectSolveTestUtils.h"

#include <mpi.h>

namespace svmp::FE::backends {
namespace {

using namespace direct_test;

TEST(FsilsDirichletFaceHangMPI, GmresWithDirichletFaceOnTheFirstRanksOnly)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ChainOptions o;
    o.n_nodes = 26;
    // Nodes 0, 7 and 12: on 3 ranks (nodes 0-8, 9-17, 18-25) ranks 0 and 1 see
    // Dirichlet DOFs, rank 2 (nodes 17-25 with its ghost) sees none.
    o.dirichlet = {1, 7 * kChainDof + 0, 7 * kChainDof + 1, 12 * kChainDof + 2};

    SolverOptions gmres;
    gmres.method = SolverMethod::GMRES;
    gmres.preconditioner = PreconditionerType::RowColumnScaling;
    gmres.fsils_use_rcs = true;
    gmres.rel_tol = 1e-12;
    gmres.abs_tol = 1e-14;
    gmres.max_iter = 500;
    gmres.krylov_dim = 100;

    FsilsFactory factory(kChainDof);
    auto sys = buildDistributedChain(factory, o, rank, size);
    std::vector<GlobalIndex> local;
    for (const auto dof : o.dirichlet) {
        const auto node = dof / kChainDof;
        if (node >= sys.first - 1 && node <= sys.last + 1) {
            local.push_back(dof);
        }
    }
    FsilsLinearSolver solver(gmres);
    solver.setDirichletDofs(local);
    const auto rep = solver.solve(*sys.A, *sys.x, *sys.b);  // hangs on 3+ ranks
    EXPECT_TRUE(rep.converged) << rep.message;
}

} // namespace
} // namespace svmp::FE::backends
