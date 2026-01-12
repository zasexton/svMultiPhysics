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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

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

} // namespace svmp::Physics::test

