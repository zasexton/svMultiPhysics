/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/Poisson/PoissonModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/Dofs/EntityDofMap.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

namespace svmp::Physics::test {

TEST(PoissonSquareSteady, DirichletLeftRight_NeumannTopBottom_WritesVtu)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
    try {
        factory = svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::Eigen);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Requires Eigen backend (enable FE_ENABLE_EIGEN): " << e.what();
    }
    ASSERT_TRUE(factory);

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 2);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    svmp::Physics::formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(1.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::Physics::formulations::poisson::PoissonModule module(space, std::move(opts));

    svmp::FE::systems::FESystem system(mesh);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const auto n_dofs = system.dofHandler().getNumDofs();
    ASSERT_GT(n_dofs, 0);
    ASSERT_EQ(static_cast<std::size_t>(n_dofs), mesh->n_vertices());

    svmp::FE::backends::SolverOptions lopt;
    lopt.method = svmp::FE::backends::SolverMethod::Direct;
    lopt.preconditioner = svmp::FE::backends::PreconditionerType::None;
    lopt.rel_tol = 1e-14;
    lopt.abs_tol = 1e-14;
    lopt.max_iter = 1;

    auto linear = factory->createLinearSolver(lopt);
    ASSERT_TRUE(linear);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(system, integrator);

    svmp::FE::timestepping::NewtonSolver newton;
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(system, *factory, ws);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.primeDtHistory(1.0);

    history.u().zero();
    {
        auto u = history.uSpan();
        system.constraints().distribute(reinterpret_cast<double*>(u.data()),
                                        static_cast<svmp::FE::GlobalIndex>(u.size()));
    }

    const auto report = newton.solveStep(transient,
                                         *linear,
                                         /*solve_time=*/0.0,
                                         history,
                                         ws);
    ASSERT_TRUE(report.converged) << "Newton did not converge (iters=" << report.iterations
                                  << ", |r|=" << report.residual_norm << ")";

    // Verify the expected linear solution u(x,y) = 1 - x (homogeneous Neumann on top/bottom).
    const auto& base = mesh->base();
    const auto u = history.uSpan();
    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh->n_vertices()); ++v) {
        const auto xyz = base.get_vertex_coords(v);
        const double expected = 1.0 - static_cast<double>(xyz[0]);
        max_err = std::max(max_err, std::abs(static_cast<double>(u[static_cast<std::size_t>(v)]) - expected));
    }
    EXPECT_LT(max_err, 1e-6);

    // Write the solution to a VTU file in the test working directory.
    const auto out_path = std::filesystem::path("poisson_square_solution.vtu");
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    auto& base_mut = mesh_mut->base();
    const auto h = base_mut.attach_field(svmp::EntityKind::Vertex, "u", svmp::FieldScalarType::Float64, 1);
    auto* out_u = base_mut.field_data_as<double>(h);
    ASSERT_NE(out_u, nullptr);
    for (std::size_t i = 0; i < base_mut.n_vertices(); ++i) {
        out_u[i] = static_cast<double>(u[i]);
    }

    EXPECT_NO_THROW(svmp::save_mesh(*mesh_mut, out_path.string()));
#  endif
#endif
}

TEST(PoissonSquareSteady, DirichletLeftRight_NeumannTopBottom_Quadratic_WritesVtu)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
    try {
        factory = svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::Eigen);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Requires Eigen backend (enable FE_ENABLE_EIGEN): " << e.what();
    }
    ASSERT_TRUE(factory);

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 2);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/2);
    svmp::Physics::formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(1.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::Physics::formulations::poisson::PoissonModule module(space, std::move(opts));

    svmp::FE::systems::FESystem system(mesh);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const auto n_dofs = system.dofHandler().getNumDofs();
    ASSERT_GT(n_dofs, 0);
    ASSERT_GT(static_cast<std::size_t>(n_dofs), mesh->n_vertices());

    // For Q2 on quads: 1 DOF per vertex, 1 per edge, 1 per cell.
    // In the Mesh library, 2D edges are represented as Mesh "faces" (codim-1 facets).
    const auto& base = mesh->base();
    EXPECT_EQ(static_cast<std::size_t>(n_dofs), base.n_vertices() + base.n_faces() + base.n_cells());

    svmp::FE::backends::SolverOptions lopt;
    lopt.method = svmp::FE::backends::SolverMethod::Direct;
    lopt.preconditioner = svmp::FE::backends::PreconditionerType::None;
    lopt.rel_tol = 1e-14;
    lopt.abs_tol = 1e-14;
    lopt.max_iter = 1;

    auto linear = factory->createLinearSolver(lopt);
    ASSERT_TRUE(linear);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(system, integrator);

    svmp::FE::timestepping::NewtonSolver newton;
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(system, *factory, ws);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.primeDtHistory(1.0);

    history.u().zero();
    {
        auto u = history.uSpan();
        system.constraints().distribute(reinterpret_cast<double*>(u.data()),
                                        static_cast<svmp::FE::GlobalIndex>(u.size()));
    }

    const auto report = newton.solveStep(transient,
                                         *linear,
                                         /*solve_time=*/0.0,
                                         history,
                                         ws);
    ASSERT_TRUE(report.converged) << "Newton did not converge (iters=" << report.iterations
                                  << ", |r|=" << report.residual_norm << ")";

    // Verify the expected linear solution u(x,y) = 1 - x.
    const auto* entity_map = system.dofHandler().getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    const auto u = history.uSpan();
    double max_err = 0.0;

    // Vertex DOFs
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(base.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        ASSERT_EQ(vdofs.size(), 1u);
        const auto dof = vdofs[0];
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());
        const auto xyz = base.get_vertex_coords(v);
        const double expected = 1.0 - static_cast<double>(xyz[0]);
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }

    // Edge midpoint DOFs (Mesh faces in 2D)
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base.n_faces()); ++f) {
        const auto edofs = entity_map->getEdgeDofs(static_cast<svmp::FE::GlobalIndex>(f));
        ASSERT_EQ(edofs.size(), 1u);
        const auto dof = edofs[0];
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());

        const auto fv = base.face_vertices(f);
        ASSERT_EQ(fv.size(), 2u);
        const auto p0 = base.get_vertex_coords(fv[0]);
        const auto p1 = base.get_vertex_coords(fv[1]);
        const double midx = 0.5 * (static_cast<double>(p0[0]) + static_cast<double>(p1[0]));
        const double expected = 1.0 - midx;
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }

    // Cell interior DOFs (Q2 center node)
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(base.n_cells()); ++c) {
        const auto cdofs = entity_map->getCellInteriorDofs(static_cast<svmp::FE::GlobalIndex>(c));
        ASSERT_EQ(cdofs.size(), 1u);
        const auto dof = cdofs[0];
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());

        const auto cv = base.cell_vertices(c);
        ASSERT_FALSE(cv.empty());
        double cx = 0.0;
        for (const auto v : cv) {
            const auto xyz = base.get_vertex_coords(v);
            cx += static_cast<double>(xyz[0]);
        }
        cx /= static_cast<double>(cv.size());
        const double expected = 1.0 - cx;
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }

    EXPECT_LT(max_err, 1e-6);

    // Write a vertex-sampled VTU (value at vertices only).
    const auto out_path = std::filesystem::path("poisson_square_solution_p2.vtu");
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    auto& base_mut = mesh_mut->base();
    const auto h = base_mut.attach_field(svmp::EntityKind::Vertex, "u", svmp::FieldScalarType::Float64, 1);
    auto* out_u = base_mut.field_data_as<double>(h);
    ASSERT_NE(out_u, nullptr);

    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(base_mut.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        ASSERT_EQ(vdofs.size(), 1u);
        const auto dof = vdofs[0];
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());
        out_u[static_cast<std::size_t>(v)] = static_cast<double>(u[static_cast<std::size_t>(dof)]);
    }

    EXPECT_NO_THROW(svmp::save_mesh(*mesh_mut, out_path.string()));
#  endif
#endif
}

TEST(PoissonSquareSteadyMPI, DirichletLeftRight_NeumannTopBottom_Linear_2Ranks)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    const auto world = svmp::MeshComm::world();
    if (world.size() != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
    bool have_factory = true;
    std::string factory_error;
    try {
        svmp::FE::backends::BackendFactory::CreateOptions bopt;
        bopt.dof_per_node = 1;
        factory = svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::FSILS, bopt);
    } catch (const std::exception& e) {
        have_factory = false;
        factory_error = e.what();
    }
#  if defined(MESH_HAS_MPI)
    {
        int ok = have_factory ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        if (!all_ok) {
            if (world.rank() == 0) {
                GTEST_SKIP() << "Requires FSILS backend: " << factory_error;
            }
            GTEST_SKIP();
        }
    }
#  else
    if (!have_factory) {
        GTEST_SKIP() << "Requires FSILS backend: " << factory_error;
    }
#  endif
    if (!factory) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "FSILS backend factory was not created.";
        }
        return;
    }

    std::shared_ptr<const svmp::Mesh> mesh;
    bool have_mesh = true;
    std::string mesh_error;
    try {
        mesh = loadSquareMeshWithMarkedBoundaries(world);
    } catch (const std::exception& e) {
        have_mesh = false;
        mesh_error = e.what();
    }
#  if defined(MESH_HAS_MPI)
    {
        int ok = have_mesh ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        if (!all_ok) {
            if (world.rank() == 0) {
                GTEST_SKIP() << "Failed to load distributed test mesh: " << mesh_error;
            }
            GTEST_SKIP();
        }
    }
#  else
    if (!have_mesh) {
        GTEST_SKIP() << "Failed to load test mesh: " << mesh_error;
    }
#  endif
    if (!mesh) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Distributed test mesh pointer is null.";
        }
        return;
    }
    EXPECT_EQ(mesh->dim(), 2);

    // Ensure this is a true distributed mesh (each rank owns a subset of cells).
    const bool mesh_is_distributed = (mesh->global_n_cells() > mesh->n_owned_cells());
    int mesh_dist_ok = mesh_is_distributed ? 1 : 0;
#  if defined(MESH_HAS_MPI)
    {
        int all_ok = 0;
        MPI_Allreduce(&mesh_dist_ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        mesh_dist_ok = all_ok;
    }
#  endif
    if (!mesh_dist_ok) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Mesh is not distributed (each rank appears to own the full mesh).";
        }
        return;
    }

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    svmp::Physics::formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(1.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::Physics::formulations::poisson::PoissonModule module(space, std::move(opts));

    svmp::FE::systems::FESystem system(mesh);
    module.registerOn(system);
    bool setup_ok = true;
    std::string setup_error;
    try {
        system.setup();
    } catch (const std::exception& e) {
        setup_ok = false;
        setup_error = e.what();
    }
#  if defined(MESH_HAS_MPI)
    {
        int ok = setup_ok ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        setup_ok = (all_ok != 0);
    }
#  endif
    if (!setup_ok) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "FESystem::setup failed in MPI test: " << setup_error;
        }
        return;
    }

    const auto n_dofs = system.dofHandler().getNumDofs();
    const auto& part = system.dofHandler().getPartition();
    const bool dofs_ok = (n_dofs > 0) && (part.globalSize() > part.localOwnedSize()) && (part.localOwnedSize() > 0);
    int dofs_ok_int = dofs_ok ? 1 : 0;
#  if defined(MESH_HAS_MPI)
    {
        int all_ok = 0;
        MPI_Allreduce(&dofs_ok_int, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        dofs_ok_int = all_ok;
    }
#  endif
    if (!dofs_ok_int) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Unexpected DOF partition for MPI run (n_dofs=" << n_dofs
                          << ", global=" << part.globalSize()
                          << ", local_owned=" << part.localOwnedSize() << ").";
        }
        return;
    }

    svmp::FE::backends::SolverOptions lopt;
    lopt.method = svmp::FE::backends::SolverMethod::CG;
    lopt.preconditioner = svmp::FE::backends::PreconditionerType::None;
    lopt.rel_tol = 1e-12;
    lopt.abs_tol = 1e-14;
    lopt.max_iter = 500;

    std::unique_ptr<svmp::FE::backends::LinearSolver> linear;
    bool linear_ok = true;
    std::string linear_error;
    try {
        linear = factory->createLinearSolver(lopt);
    } catch (const std::exception& e) {
        linear_ok = false;
        linear_error = e.what();
    }
#  if defined(MESH_HAS_MPI)
    {
        int ok = linear_ok ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        linear_ok = (all_ok != 0);
    }
#  endif
    if (!linear_ok || !linear) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Failed to create FSILS linear solver: " << linear_error;
        }
        return;
    }

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(system, integrator);

    svmp::FE::timestepping::NewtonSolver newton;
    svmp::FE::timestepping::NewtonWorkspace ws;
    bool ws_ok = true;
    std::string ws_error;
    try {
        newton.allocateWorkspace(system, *factory, ws);
    } catch (const std::exception& e) {
        ws_ok = false;
        ws_error = e.what();
    }
#  if defined(MESH_HAS_MPI)
    {
        int ok = ws_ok ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        ws_ok = (all_ok != 0);
    }
#  endif
    if (!ws_ok) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Newton workspace allocation failed: " << ws_error;
        }
        return;
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.primeDtHistory(1.0);

    history.u().zero();
    system.constraints().distribute(history.u());
    history.u().updateGhosts();

    svmp::FE::timestepping::NewtonReport report;
    bool solve_ok = true;
    std::string solve_error;
    try {
        report = newton.solveStep(transient,
                                  *linear,
                                  /*solve_time=*/0.0,
                                  history,
                                  ws);
    } catch (const std::exception& e) {
        solve_ok = false;
        solve_error = e.what();
    }

    int converged_ok = (solve_ok && report.converged) ? 1 : 0;
#  if defined(MESH_HAS_MPI)
    {
        int all_ok = 0;
        MPI_Allreduce(&converged_ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        converged_ok = all_ok;
    }
#  endif
    if (!converged_ok) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Newton did not converge in MPI run (iters=" << report.iterations
                          << ", |r|=" << report.residual_norm << "): " << solve_error;
        }
        return;
    }

    // Verify the expected linear solution u(x,y) = 1 - x.
    history.u().updateGhosts();
    const auto* entity_map = system.dofHandler().getEntityDofMap();
    auto u_view = history.u().createAssemblyView();

    const auto& base = mesh->base();
    double max_err_local = 0.0;
    int eval_ok = (entity_map != nullptr && static_cast<bool>(u_view)) ? 1 : 0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(base.n_vertices()); ++v) {
        if (!mesh->is_owned_vertex(v)) {
            continue;
        }
        if (entity_map == nullptr || !u_view) {
            eval_ok = 0;
            break;
        }
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        if (vdofs.size() != 1u) {
            eval_ok = 0;
            continue;
        }
        const auto dof = vdofs[0];
        const double value = static_cast<double>(u_view->getVectorEntry(dof));
        const auto xyz = base.get_vertex_coords(v);
        const double expected = 1.0 - static_cast<double>(xyz[0]);
        max_err_local = std::max(max_err_local, std::abs(value - expected));
    }

    if (!eval_ok) {
        max_err_local = std::numeric_limits<double>::infinity();
    }

    double max_err = max_err_local;
#  if defined(MESH_HAS_MPI)
    MPI_Allreduce(&max_err_local, &max_err, 1, MPI_DOUBLE, MPI_MAX, world.native());
    {
        int all_ok = 0;
        MPI_Allreduce(&eval_ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        eval_ok = all_ok;
    }
#  endif
    if (world.rank() == 0) {
        EXPECT_EQ(eval_ok, 1);
        EXPECT_LT(max_err, 1e-6);
    }

    // Write a distributed VTU output (PVTU master file + per-rank pieces).
    int can_write = (entity_map != nullptr && static_cast<bool>(u_view)) ? 1 : 0;
#  if defined(MESH_HAS_MPI)
    {
        int all_ok = 0;
        MPI_Allreduce(&can_write, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        can_write = all_ok;
    }
#  endif
    if (!can_write) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Cannot write MPI output (missing EntityDofMap or vector view).";
        }
        return;
    }

    history.u().updateGhosts();
    const auto out_path = std::filesystem::path("poisson_square_solution_mpi.pvtu");
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    auto& base_mut = mesh_mut->base();
    const auto h = base_mut.attach_field(svmp::EntityKind::Vertex, "u", svmp::FieldScalarType::Float64, 1);
    auto* out_u = base_mut.field_data_as<double>(h);
    int out_ok = (out_u != nullptr) ? 1 : 0;
#  if defined(MESH_HAS_MPI)
    {
        int all_ok = 0;
        MPI_Allreduce(&out_ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        out_ok = all_ok;
    }
#  endif
    if (!out_ok) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Failed to attach output field 'u' for MPI VTU write.";
        }
        return;
    }

    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(base_mut.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        if (vdofs.size() != 1u) {
            out_u[static_cast<std::size_t>(v)] = 0.0;
            continue;
        }
        out_u[static_cast<std::size_t>(v)] = static_cast<double>(u_view->getVectorEntry(vdofs[0]));
    }

    try {
        svmp::save_mesh(*mesh_mut, out_path.string());
    } catch (const std::exception& e) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "Failed to write MPI VTU output: " << e.what();
        }
#  if defined(MESH_HAS_MPI)
        MPI_Abort(world.native(), 1);
#  else
        std::terminate();
#  endif
    }
#  endif
#endif
}

} // namespace svmp::Physics::test
