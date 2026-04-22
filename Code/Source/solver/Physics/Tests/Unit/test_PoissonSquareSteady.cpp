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
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

namespace svmp::Physics::test {

namespace {

using DarcyExactPressure = double (*)(double, double);
using DarcyExactFlux = std::array<double, 2> (*)(double, double);
using DarcyExactPressure3D = double (*)(double, double, double);
using DarcyExactFlux3D = std::array<double, 3> (*)(double, double, double);

struct SquareBounds {
    double xmin{0.0};
    double xmax{0.0};
    double ymin{0.0};
    double ymax{0.0};
};

struct DarcyMmsBackend {
    std::unique_ptr<svmp::FE::backends::BackendFactory> factory{};
    svmp::FE::backends::SolverMethod method{svmp::FE::backends::SolverMethod::Direct};
    svmp::FE::backends::PreconditionerType preconditioner{svmp::FE::backends::PreconditionerType::None};
    int max_iter{1};
    svmp::FE::Real rel_tol{1e-14};
    svmp::FE::Real abs_tol{1e-14};
    double newton_abs_tol{1e-10};
    double newton_rel_tol{1e-8};
    double pressure_tolerance{1e-8};
    double flux_tolerance{1e-10};
    double quadratic_flux_tolerance{1e-8};
};

DarcyMmsBackend tryCreateDarcyMmsBackend(std::string& error)
{
    DarcyMmsBackend backend;
    try {
        backend.factory = svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::Eigen);
        backend.method = svmp::FE::backends::SolverMethod::Direct;
        backend.preconditioner = svmp::FE::backends::PreconditionerType::None;
        backend.max_iter = 1;
        backend.rel_tol = 1e-14;
        backend.abs_tol = 1e-14;
        return backend;
    } catch (const std::exception& e) {
        error = e.what();
    }

    try {
        svmp::FE::backends::BackendFactory::CreateOptions opts;
        opts.dof_per_node = 1;
        backend.factory = svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::FSILS, opts);
        backend.method = svmp::FE::backends::SolverMethod::CG;
        backend.preconditioner = svmp::FE::backends::PreconditionerType::Diagonal;
        backend.max_iter = 2000;
        backend.rel_tol = 1e-10;
        backend.abs_tol = 1e-7;
        backend.newton_abs_tol = 1e-6;
        backend.newton_rel_tol = 1e-6;
        backend.pressure_tolerance = 2e-7;
        backend.flux_tolerance = 1e-5;
        backend.quadratic_flux_tolerance = 1e-5;
        return backend;
    } catch (const std::exception& e) {
        error += "\nFSILS fallback failed: ";
        error += e.what();
    }

    return backend;
}

svmp::FE::timestepping::TimeHistory solveSteadyDarcySystem(
    svmp::FE::systems::FESystem& system,
    DarcyMmsBackend& backend)
{
    const auto n_dofs = system.dofHandler().getNumDofs();
    EXPECT_GT(n_dofs, 0);

    svmp::FE::backends::SolverOptions lopt;
    lopt.method = backend.method;
    lopt.preconditioner = backend.preconditioner;
    lopt.rel_tol = backend.rel_tol;
    lopt.abs_tol = backend.abs_tol;
    lopt.max_iter = backend.max_iter;

    auto linear = backend.factory->createLinearSolver(lopt);
    EXPECT_TRUE(linear);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(system, integrator);

    svmp::FE::timestepping::NewtonOptions newton_options;
    newton_options.residual_op = "equations";
    newton_options.jacobian_op = "equations";
    newton_options.abs_tolerance = backend.newton_abs_tol;
    newton_options.rel_tolerance = backend.newton_rel_tol;
    svmp::FE::timestepping::NewtonSolver newton(newton_options);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(system, *backend.factory, ws);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*backend.factory, n_dofs, /*history_depth=*/2);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.primeDtHistory(1.0);
    history.u().zero();
    system.constraints().distribute(history.u());
    history.u().updateGhosts();

    const auto report = newton.solveStep(transient,
                                         *linear,
                                         /*solve_time=*/0.0,
                                         history,
                                         ws);
    EXPECT_TRUE(report.converged) << "Newton did not converge (iters=" << report.iterations
                                  << ", |r|=" << report.residual_norm << ")";
    history.u().updateGhosts();
    return history;
}

svmp::FE::systems::SystemStateView stateFromHistory(svmp::FE::timestepping::TimeHistory& history)
{
    svmp::FE::systems::SystemStateView state;
    state.dt = history.dt();
    state.dt_prev = history.dtPrev();
    state.u = history.uSpan();
    state.u_prev = history.uPrevSpan();
    state.u_prev2 = history.uPrev2Span();
    state.u_vector = &history.u();
    state.u_prev_vector = &history.uPrev();
    state.u_prev2_vector = &history.uPrev2();
    state.u_history = history.uHistorySpans();
    state.dt_history = history.dtHistory();
    return state;
}

SquareBounds squareBounds(const svmp::MeshBase& mesh)
{
    SquareBounds b;
    b.xmin = b.ymin = std::numeric_limits<double>::infinity();
    b.xmax = b.ymax = -std::numeric_limits<double>::infinity();
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto xyz = mesh.get_vertex_coords(v);
        const double x = static_cast<double>(xyz[0]);
        const double y = static_cast<double>(xyz[1]);
        b.xmin = std::min(b.xmin, x);
        b.xmax = std::max(b.xmax, x);
        b.ymin = std::min(b.ymin, y);
        b.ymax = std::max(b.ymax, y);
    }
    return b;
}

bool isSquareBoundaryVertex(const svmp::MeshBase& mesh, svmp::index_t vertex)
{
    const auto b = squareBounds(mesh);
    const auto xyz = mesh.get_vertex_coords(vertex);
    constexpr double tol = 1e-12;
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    return std::abs(x - b.xmin) <= tol ||
           std::abs(x - b.xmax) <= tol ||
           std::abs(y - b.ymin) <= tol ||
           std::abs(y - b.ymax) <= tol;
}

svmp::index_t nearestInteriorVertex(const svmp::MeshBase& mesh, double x_target, double y_target)
{
    svmp::index_t best = svmp::INVALID_INDEX;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        if (isSquareBoundaryVertex(mesh, v)) {
            continue;
        }
        const auto xyz = mesh.get_vertex_coords(v);
        const double dx = static_cast<double>(xyz[0]) - x_target;
        const double dy = static_cast<double>(xyz[1]) - y_target;
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = v;
        }
    }
    return best;
}

std::vector<formulations::poisson::PoissonOptions::NodeDirichletBC>
nodeConstraintsForVertices(const svmp::MeshBase& mesh,
                           DarcyExactPressure exact,
                           bool include_boundary,
                           std::vector<svmp::index_t> extra_vertices = {})
{
    const auto& gids = mesh.vertex_gids();
    if (gids.size() != mesh.n_vertices()) {
        throw std::runtime_error("Square Darcy MMS test requires GlobalVertexID data on the Square mesh");
    }

    std::vector<formulations::poisson::PoissonOptions::NodeDirichletBC> values;
    auto add_vertex = [&](svmp::index_t v) {
        const auto xyz = mesh.get_vertex_coords(v);
        values.push_back(formulations::poisson::PoissonOptions::NodeDirichletBC{
            static_cast<svmp::FE::GlobalIndex>(gids[static_cast<std::size_t>(v)]),
            static_cast<svmp::FE::Real>(exact(static_cast<double>(xyz[0]), static_cast<double>(xyz[1]))),
        });
    };

    if (include_boundary) {
        for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
            if (isSquareBoundaryVertex(mesh, v)) {
                add_vertex(v);
            }
        }
    }

    for (const auto v : extra_vertices) {
        if (v != svmp::INVALID_INDEX) {
            add_vertex(v);
        }
    }
    return values;
}

void appendDarcyDerivedFields(svmp::FE::systems::FESystem& system,
                              svmp::Mesh& mesh,
                              svmp::FE::timestepping::TimeHistory& history)
{
    auto state = stateFromHistory(history);
    system.appendDerivedResultFields(mesh.base(), state);
}

double maxVertexPressureError(const svmp::FE::systems::FESystem& system,
                              const svmp::MeshBase& mesh,
                              std::span<const svmp::FE::Real> u,
                              DarcyExactPressure exact)
{
    const auto* entity_map = system.fieldDofHandler(0).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    if (!entity_map) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    const auto field_offset = system.fieldDofOffset(0);
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        EXPECT_EQ(vdofs.size(), 1u);
        if (vdofs.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        const auto dof = vdofs[0] + field_offset;
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), u.size());
        const auto xyz = mesh.get_vertex_coords(v);
        const double expected = exact(static_cast<double>(xyz[0]), static_cast<double>(xyz[1]));
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }
    return max_err;
}

double maxQ2PressureDofError(const svmp::FE::systems::FESystem& system,
                             const svmp::MeshBase& mesh,
                             std::span<const svmp::FE::Real> u,
                             DarcyExactPressure exact)
{
    const auto* entity_map = system.fieldDofHandler(0).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    if (!entity_map) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = maxVertexPressureError(system, mesh, u, exact);
    const auto field_offset = system.fieldDofOffset(0);

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
        const auto edofs = entity_map->getEdgeDofs(static_cast<svmp::FE::GlobalIndex>(f));
        if (edofs.empty()) {
            continue;
        }
        EXPECT_EQ(edofs.size(), 1u);
        const auto dof = edofs[0] + field_offset;
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), u.size());

        const auto fv = mesh.face_vertices(f);
        EXPECT_EQ(fv.size(), 2u);
        if (fv.size() != 2u) {
            continue;
        }
        const auto p0 = mesh.get_vertex_coords(fv[0]);
        const auto p1 = mesh.get_vertex_coords(fv[1]);
        const double x = 0.5 * (static_cast<double>(p0[0]) + static_cast<double>(p1[0]));
        const double y = 0.5 * (static_cast<double>(p0[1]) + static_cast<double>(p1[1]));
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - exact(x, y)));
    }

    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
        const auto cdofs = entity_map->getCellInteriorDofs(static_cast<svmp::FE::GlobalIndex>(c));
        if (cdofs.empty()) {
            continue;
        }
        EXPECT_EQ(cdofs.size(), 1u);
        const auto dof = cdofs[0] + field_offset;
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), u.size());

        const auto centroid = mesh.cell_centroid(c);
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) -
                                    exact(static_cast<double>(centroid[0]), static_cast<double>(centroid[1]))));
    }
    return max_err;
}

double maxCellFluxError(const svmp::MeshBase& mesh, DarcyExactFlux exact_cell_average)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Volume, "Darcy_flux");
    EXPECT_EQ(mesh.field_components(h), 2u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
        const auto centroid = mesh.cell_centroid(c);
        const auto expected = exact_cell_average(static_cast<double>(centroid[0]), static_cast<double>(centroid[1]));
        const std::size_t offset = static_cast<std::size_t>(c) * 2u;
        max_err = std::max(max_err, std::abs(data[offset] - expected[0]));
        max_err = std::max(max_err, std::abs(data[offset + 1u] - expected[1]));
    }
    return max_err;
}

double maxVertexFluxError(const svmp::MeshBase& mesh, DarcyExactFlux exact_vertex_flux)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
    EXPECT_EQ(mesh.field_components(h), 2u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto xyz = mesh.get_vertex_coords(v);
        const auto expected = exact_vertex_flux(static_cast<double>(xyz[0]), static_cast<double>(xyz[1]));
        const std::size_t offset = static_cast<std::size_t>(v) * 2u;
        max_err = std::max(max_err, std::abs(data[offset] - expected[0]));
        max_err = std::max(max_err, std::abs(data[offset + 1u] - expected[1]));
    }
    return max_err;
}

double maxPatchAverageFluxError(const svmp::MeshBase& mesh, DarcyExactFlux exact_cell_average)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
    EXPECT_EQ(mesh.field_components(h), 2u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<std::array<double, 2>> accum(mesh.n_vertices(), {0.0, 0.0});
    std::vector<int> counts(mesh.n_vertices(), 0);
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
        const auto centroid = mesh.cell_centroid(c);
        const auto cell_flux = exact_cell_average(static_cast<double>(centroid[0]), static_cast<double>(centroid[1]));
        for (const auto v : mesh.cell_vertices(c)) {
            accum[static_cast<std::size_t>(v)][0] += cell_flux[0];
            accum[static_cast<std::size_t>(v)][1] += cell_flux[1];
            counts[static_cast<std::size_t>(v)] += 1;
        }
    }

    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const int n = counts[static_cast<std::size_t>(v)];
        EXPECT_GT(n, 0);
        if (n <= 0) {
            continue;
        }
        const std::array<double, 2> expected = {
            accum[static_cast<std::size_t>(v)][0] / static_cast<double>(n),
            accum[static_cast<std::size_t>(v)][1] / static_cast<double>(n),
        };
        const std::size_t offset = static_cast<std::size_t>(v) * 2u;
        max_err = std::max(max_err, std::abs(data[offset] - expected[0]));
        max_err = std::max(max_err, std::abs(data[offset + 1u] - expected[1]));
    }
    return max_err;
}

void attachPressureVertexField(const svmp::FE::systems::FESystem& system,
                               svmp::MeshBase& mesh,
                               std::span<const svmp::FE::Real> u)
{
    const auto h = mesh.attach_field(svmp::EntityKind::Vertex, "Pressure", svmp::FieldScalarType::Float64, 1);
    auto* out = mesh.field_data_as<double>(h);
    ASSERT_NE(out, nullptr);

    const auto* entity_map = system.fieldDofHandler(0).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto field_offset = system.fieldDofOffset(0);
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        ASSERT_EQ(vdofs.size(), 1u);
        const auto dof = vdofs[0] + field_offset;
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());
        out[static_cast<std::size_t>(v)] = static_cast<double>(u[static_cast<std::size_t>(dof)]);
    }
}

void installDarcyMmsModule(svmp::FE::systems::FESystem& system,
                           std::shared_ptr<const svmp::FE::spaces::H1Space> space,
                           formulations::poisson::PoissonOptions opts)
{
    opts.field_name = "Pressure";
    opts.diffusion = 1.0;
    opts.register_darcy_flux_output = true;
    formulations::poisson::PoissonModule module(std::move(space), std::move(opts));
    module.registerOn(system);
}

double p_linear(double x, double) { return 1.0 - x; }
std::array<double, 2> q_linear(double, double) { return {1.0, 0.0}; }

double p_bilinear(double x, double y) { return x * y; }
std::array<double, 2> q_bilinear(double x, double y) { return {-y, -x}; }

double p_quadratic(double x, double) { return x * (1.0 - x); }
std::array<double, 2> q_quadratic(double x, double) { return {2.0 * x - 1.0, 0.0}; }

double p_affine_interior(double x, double y) { return 1.0 - x + 0.25 * y; }
std::array<double, 2> q_affine_interior(double, double) { return {1.0, -0.25}; }

bool isCubeBoundaryVertex(const svmp::MeshBase& mesh, svmp::index_t vertex)
{
    const auto b = computeCubeBoundaryBounds(mesh);
    const auto xyz = mesh.get_vertex_coords(vertex);
    const double scale = std::max({1.0,
                                   std::abs(b.xmax - b.xmin),
                                   std::abs(b.ymax - b.ymin),
                                   std::abs(b.zmax - b.zmin)});
    const double tol = 1e-10 * scale;
    const double x = static_cast<double>(xyz[0]);
    const double y = static_cast<double>(xyz[1]);
    const double z = static_cast<double>(xyz[2]);
    return std::abs(x - b.xmin) <= tol ||
           std::abs(x - b.xmax) <= tol ||
           std::abs(y - b.ymin) <= tol ||
           std::abs(y - b.ymax) <= tol ||
           std::abs(z - b.zmin) <= tol ||
           std::abs(z - b.zmax) <= tol;
}

svmp::index_t nearestInteriorVertex3D(const svmp::MeshBase& mesh,
                                      double x_target,
                                      double y_target,
                                      double z_target)
{
    svmp::index_t best = svmp::INVALID_INDEX;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        if (isCubeBoundaryVertex(mesh, v)) {
            continue;
        }
        const auto xyz = mesh.get_vertex_coords(v);
        const double dx = static_cast<double>(xyz[0]) - x_target;
        const double dy = static_cast<double>(xyz[1]) - y_target;
        const double dz = static_cast<double>(xyz[2]) - z_target;
        const double dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = v;
        }
    }
    return best;
}

std::vector<formulations::poisson::PoissonOptions::NodeDirichletBC>
nodeConstraintsForVertices3D(const svmp::MeshBase& mesh,
                             DarcyExactPressure3D exact,
                             bool include_boundary,
                             std::vector<svmp::index_t> extra_vertices = {})
{
    const auto& gids = mesh.vertex_gids();
    if (gids.size() != mesh.n_vertices()) {
        throw std::runtime_error("Cube Darcy MMS test requires GlobalVertexID data on the Cube mesh");
    }

    std::vector<formulations::poisson::PoissonOptions::NodeDirichletBC> values;
    auto add_vertex = [&](svmp::index_t v) {
        const auto xyz = mesh.get_vertex_coords(v);
        values.push_back(formulations::poisson::PoissonOptions::NodeDirichletBC{
            static_cast<svmp::FE::GlobalIndex>(gids[static_cast<std::size_t>(v)]),
            static_cast<svmp::FE::Real>(exact(static_cast<double>(xyz[0]),
                                              static_cast<double>(xyz[1]),
                                              static_cast<double>(xyz[2]))),
        });
    };

    if (include_boundary) {
        for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
            if (isCubeBoundaryVertex(mesh, v)) {
                add_vertex(v);
            }
        }
    }

    for (const auto v : extra_vertices) {
        if (v != svmp::INVALID_INDEX) {
            add_vertex(v);
        }
    }
    return values;
}

double maxVertexPressureError3D(const svmp::FE::systems::FESystem& system,
                                const svmp::MeshBase& mesh,
                                std::span<const svmp::FE::Real> u,
                                DarcyExactPressure3D exact)
{
    const auto* entity_map = system.fieldDofHandler(0).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    if (!entity_map) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    const auto field_offset = system.fieldDofOffset(0);
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        EXPECT_EQ(vdofs.size(), 1u);
        if (vdofs.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        const auto dof = vdofs[0] + field_offset;
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), u.size());
        const auto xyz = mesh.get_vertex_coords(v);
        const double expected = exact(static_cast<double>(xyz[0]),
                                      static_cast<double>(xyz[1]),
                                      static_cast<double>(xyz[2]));
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }
    return max_err;
}

double maxQ2TetraPressureDofError3D(const svmp::FE::systems::FESystem& system,
                                    const svmp::MeshBase& mesh,
                                    std::span<const svmp::FE::Real> u,
                                    DarcyExactPressure3D exact)
{
    const auto* entity_map = system.fieldDofHandler(0).getEntityDofMap();
    EXPECT_NE(entity_map, nullptr);
    if (!entity_map) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = maxVertexPressureError3D(system, mesh, u, exact);
    const auto field_offset = system.fieldDofOffset(0);

    for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
        const auto edofs = entity_map->getEdgeDofs(static_cast<svmp::FE::GlobalIndex>(e));
        if (edofs.empty()) {
            continue;
        }
        EXPECT_EQ(edofs.size(), 1u);
        const auto dof = edofs[0] + field_offset;
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), u.size());

        const auto ev = mesh.edge_vertices(e);
        const auto p0 = mesh.get_vertex_coords(ev[0]);
        const auto p1 = mesh.get_vertex_coords(ev[1]);
        const double x = 0.5 * (static_cast<double>(p0[0]) + static_cast<double>(p1[0]));
        const double y = 0.5 * (static_cast<double>(p0[1]) + static_cast<double>(p1[1]));
        const double z = 0.5 * (static_cast<double>(p0[2]) + static_cast<double>(p1[2]));
        max_err = std::max(max_err,
                           std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) -
                                    exact(x, y, z)));
    }

    return max_err;
}

double maxCellFluxError3D(const svmp::MeshBase& mesh, DarcyExactFlux3D exact_cell_average)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Volume, "Darcy_flux");
    EXPECT_EQ(mesh.field_components(h), 3u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
        const auto centroid = mesh.cell_centroid(c);
        const auto expected = exact_cell_average(static_cast<double>(centroid[0]),
                                                 static_cast<double>(centroid[1]),
                                                 static_cast<double>(centroid[2]));
        const std::size_t offset = static_cast<std::size_t>(c) * 3u;
        for (std::size_t d = 0; d < 3u; ++d) {
            max_err = std::max(max_err, std::abs(data[offset + d] - expected[d]));
        }
    }
    return max_err;
}

double maxVertexFluxError3D(const svmp::MeshBase& mesh, DarcyExactFlux3D exact_vertex_flux)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
    EXPECT_EQ(mesh.field_components(h), 3u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto xyz = mesh.get_vertex_coords(v);
        const auto expected = exact_vertex_flux(static_cast<double>(xyz[0]),
                                                static_cast<double>(xyz[1]),
                                                static_cast<double>(xyz[2]));
        const std::size_t offset = static_cast<std::size_t>(v) * 3u;
        for (std::size_t d = 0; d < 3u; ++d) {
            max_err = std::max(max_err, std::abs(data[offset + d] - expected[d]));
        }
    }
    return max_err;
}

double maxPatchAverageFluxError3D(const svmp::MeshBase& mesh, DarcyExactFlux3D exact_cell_average)
{
    const auto h = mesh.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
    EXPECT_EQ(mesh.field_components(h), 3u);
    const auto* data = mesh.field_data_as<double>(h);
    EXPECT_NE(data, nullptr);
    if (!data) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<std::array<double, 3>> accum(mesh.n_vertices(), {0.0, 0.0, 0.0});
    std::vector<int> counts(mesh.n_vertices(), 0);
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(mesh.n_cells()); ++c) {
        const auto centroid = mesh.cell_centroid(c);
        const auto cell_flux = exact_cell_average(static_cast<double>(centroid[0]),
                                                  static_cast<double>(centroid[1]),
                                                  static_cast<double>(centroid[2]));
        for (const auto v : mesh.cell_vertices(c)) {
            for (std::size_t d = 0; d < 3u; ++d) {
                accum[static_cast<std::size_t>(v)][d] += cell_flux[d];
            }
            counts[static_cast<std::size_t>(v)] += 1;
        }
    }

    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const int n = counts[static_cast<std::size_t>(v)];
        EXPECT_GT(n, 0);
        if (n <= 0) {
            continue;
        }
        const std::size_t offset = static_cast<std::size_t>(v) * 3u;
        for (std::size_t d = 0; d < 3u; ++d) {
            const double expected = accum[static_cast<std::size_t>(v)][d] / static_cast<double>(n);
            max_err = std::max(max_err, std::abs(data[offset + d] - expected));
        }
    }
    return max_err;
}

double p_linear3(double x, double, double) { return 1.0 - x; }
std::array<double, 3> q_linear3(double, double, double) { return {1.0, 0.0, 0.0}; }

double p_affine_boundary3(double x, double y, double z) { return x + 2.0 * y - 0.5 * z; }
std::array<double, 3> q_affine_boundary3(double, double, double) { return {-1.0, -2.0, 0.5}; }

double p_quadratic3(double x, double, double) { return x * (1.0 - x); }
std::array<double, 3> q_quadratic3(double x, double, double) { return {2.0 * x - 1.0, 0.0, 0.0}; }

double p_affine_interior3(double x, double y, double z) { return 1.0 - x + 0.25 * y - 0.125 * z; }
std::array<double, 3> q_affine_interior3(double, double, double) { return {1.0, -0.25, 0.125}; }

} // namespace

TEST(DarcySquareMMS, LinearPressureConstantFlux)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(1.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_EQ(system.derivedResults().size(), 2u);
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError(system, mesh_mut->base(), history.uSpan(), p_linear),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError(mesh_mut->base(), q_linear), backend.flux_tolerance);
    EXPECT_LT(maxVertexFluxError(mesh_mut->base(), q_linear), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcySquareMMS, BilinearHarmonicBoundaryNodeConstraints)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    ASSERT_NO_THROW(opts.node_dirichlet.values =
                        nodeConstraintsForVertices(mesh->base(), p_bilinear, /*include_boundary=*/true));

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError(system, mesh_mut->base(), history.uSpan(), p_bilinear),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError(mesh_mut->base(), q_bilinear), backend.flux_tolerance);
    EXPECT_LT(maxPatchAverageFluxError(mesh_mut->base(), q_bilinear), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcySquareMMS, QuadraticConstantSourceQ2)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/2);
    formulations::poisson::PoissonOptions opts;
    opts.source = 2.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(0.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxQ2PressureDofError(system, mesh_mut->base(), history.uSpan(), p_quadratic),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError(mesh_mut->base(), q_quadratic), backend.quadratic_flux_tolerance);
    EXPECT_LT(maxPatchAverageFluxError(mesh_mut->base(), q_quadratic), backend.quadratic_flux_tolerance);
#  endif
#endif
}

TEST(DarcySquareMMS, AffinePressureWithInteriorNodeConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    const auto interior = nearestInteriorVertex(mesh->base(), 0.5, 0.5);
    ASSERT_NE(interior, svmp::INVALID_INDEX);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    ASSERT_NO_THROW(opts.node_dirichlet.values =
                        nodeConstraintsForVertices(mesh->base(),
                                                   p_affine_interior,
                                                   /*include_boundary=*/true,
                                                   {interior}));

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError(system, mesh_mut->base(), history.uSpan(), p_affine_interior),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError(mesh_mut->base(), q_affine_interior), backend.flux_tolerance);
    EXPECT_LT(maxVertexFluxError(mesh_mut->base(), q_affine_interior), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcySquareMMS, DerivedFieldsRoundTripVtuOutput)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kSquareBoundaryLeft), .value = svmp::FE::Real(1.0)},
        {.boundary_marker = static_cast<int>(kSquareBoundaryRight), .value = svmp::FE::Real(0.0)},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);
    attachPressureVertexField(system, mesh_mut->base(), history.uSpan());

    const auto out_path = std::filesystem::temp_directory_path() / "svmp_darcy_square_mms_roundtrip.vtu";
    std::filesystem::remove(out_path);
    ASSERT_NO_THROW(svmp::save_mesh(*mesh_mut, out_path.string()));

    svmp::MeshIOOptions opts_io;
    opts_io.format = "vtu";
    opts_io.path = out_path.string();
    const auto reloaded = svmp::MeshBase::load(opts_io);

    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Vertex, "Pressure"));
    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux_node"));
    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux"));
    EXPECT_FALSE(reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux_node"));
    EXPECT_FALSE(reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux"));

    if (reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux_node")) {
        const auto h = reloaded.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
        EXPECT_EQ(reloaded.field_components(h), 2u);
    }
    if (reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux")) {
        const auto h = reloaded.field_handle(svmp::EntityKind::Volume, "Darcy_flux");
        EXPECT_EQ(reloaded.field_components(h), 2u);
    }

    std::filesystem::remove(out_path);
#  endif
#endif
}

TEST(DarcyCubeMMS, LinearPressureConstantFlux)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadCubeMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    ASSERT_EQ(mesh->dim(), 3);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    const auto bounds = computeCubeBoundaryBounds(mesh->base());

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kCubeBoundaryLeft),
         .value = svmp::FE::Real(p_linear3(bounds.xmin, 0.0, 0.0))},
        {.boundary_marker = static_cast<int>(kCubeBoundaryRight),
         .value = svmp::FE::Real(p_linear3(bounds.xmax, 0.0, 0.0))},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_EQ(system.derivedResults().size(), 2u);
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError3D(system, mesh_mut->base(), history.uSpan(), p_linear3),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError3D(mesh_mut->base(), q_linear3), backend.flux_tolerance);
    EXPECT_LT(maxVertexFluxError3D(mesh_mut->base(), q_linear3), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcyCubeMMS, AffineHarmonicBoundaryNodeConstraints)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadCubeMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    ASSERT_EQ(mesh->dim(), 3);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    ASSERT_NO_THROW(opts.node_dirichlet.values =
                        nodeConstraintsForVertices3D(mesh->base(),
                                                     p_affine_boundary3,
                                                     /*include_boundary=*/true));

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError3D(system, mesh_mut->base(), history.uSpan(), p_affine_boundary3),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError3D(mesh_mut->base(), q_affine_boundary3), backend.flux_tolerance);
    EXPECT_LT(maxVertexFluxError3D(mesh_mut->base(), q_affine_boundary3), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcyCubeMMS, QuadraticConstantSourceQ2)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadCubeMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    ASSERT_EQ(mesh->dim(), 3);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    const auto bounds = computeCubeBoundaryBounds(mesh->base());

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/2);
    formulations::poisson::PoissonOptions opts;
    opts.source = 2.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kCubeBoundaryLeft),
         .value = svmp::FE::Real(p_quadratic3(bounds.xmin, 0.0, 0.0))},
        {.boundary_marker = static_cast<int>(kCubeBoundaryRight),
         .value = svmp::FE::Real(p_quadratic3(bounds.xmax, 0.0, 0.0))},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxQ2TetraPressureDofError3D(system, mesh_mut->base(), history.uSpan(), p_quadratic3),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError3D(mesh_mut->base(), q_quadratic3), backend.quadratic_flux_tolerance);
    EXPECT_LT(maxPatchAverageFluxError3D(mesh_mut->base(), q_quadratic3), backend.quadratic_flux_tolerance);
#  endif
#endif
}

TEST(DarcyCubeMMS, AffinePressureWithInteriorNodeConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadCubeMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    ASSERT_EQ(mesh->dim(), 3);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);

    const auto bounds = computeCubeBoundaryBounds(mesh->base());
    const auto interior = nearestInteriorVertex3D(mesh->base(),
                                                  0.5 * (bounds.xmin + bounds.xmax),
                                                  0.5 * (bounds.ymin + bounds.ymax),
                                                  0.5 * (bounds.zmin + bounds.zmax));
    ASSERT_NE(interior, svmp::INVALID_INDEX);

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    ASSERT_NO_THROW(opts.node_dirichlet.values =
                        nodeConstraintsForVertices3D(mesh->base(),
                                                     p_affine_interior3,
                                                     /*include_boundary=*/true,
                                                     {interior}));

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);

    EXPECT_LT(maxVertexPressureError3D(system, mesh_mut->base(), history.uSpan(), p_affine_interior3),
              backend.pressure_tolerance);
    EXPECT_LT(maxCellFluxError3D(mesh_mut->base(), q_affine_interior3), backend.flux_tolerance);
    EXPECT_LT(maxVertexFluxError3D(mesh_mut->base(), q_affine_interior3), backend.flux_tolerance);
#  endif
#endif
}

TEST(DarcyCubeMMS, DerivedFieldsRoundTripVtuOutput)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    std::string backend_error;
    auto backend = tryCreateDarcyMmsBackend(backend_error);
    if (!backend.factory) {
        GTEST_SKIP() << "Requires Eigen or FSILS backend: " << backend_error;
    }

    const auto mesh = loadCubeMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    ASSERT_EQ(mesh->dim(), 3);
    auto mesh_mut = std::const_pointer_cast<svmp::Mesh>(mesh);
    const auto bounds = computeCubeBoundaryBounds(mesh->base());

    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.source = 0.0;
    opts.dirichlet = {
        {.boundary_marker = static_cast<int>(kCubeBoundaryLeft),
         .value = svmp::FE::Real(p_linear3(bounds.xmin, 0.0, 0.0))},
        {.boundary_marker = static_cast<int>(kCubeBoundaryRight),
         .value = svmp::FE::Real(p_linear3(bounds.xmax, 0.0, 0.0))},
    };

    svmp::FE::systems::FESystem system(mesh);
    installDarcyMmsModule(system, space, std::move(opts));
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyDarcySystem(system, backend);
    appendDarcyDerivedFields(system, *mesh_mut, history);
    attachPressureVertexField(system, mesh_mut->base(), history.uSpan());

    const auto out_path = std::filesystem::temp_directory_path() / "svmp_darcy_cube_mms_roundtrip.vtu";
    std::filesystem::remove(out_path);
    ASSERT_NO_THROW(svmp::save_mesh(*mesh_mut, out_path.string()));

    svmp::MeshIOOptions opts_io;
    opts_io.format = "vtu";
    opts_io.path = out_path.string();
    const auto reloaded = svmp::MeshBase::load(opts_io);

    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Vertex, "Pressure"));
    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux_node"));
    EXPECT_TRUE(reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux"));
    EXPECT_FALSE(reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux_node"));
    EXPECT_FALSE(reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux"));

    if (reloaded.has_field(svmp::EntityKind::Vertex, "Darcy_flux_node")) {
        const auto h = reloaded.field_handle(svmp::EntityKind::Vertex, "Darcy_flux_node");
        EXPECT_EQ(reloaded.field_components(h), 3u);
    }
    if (reloaded.has_field(svmp::EntityKind::Volume, "Darcy_flux")) {
        const auto h = reloaded.field_handle(svmp::EntityKind::Volume, "Darcy_flux");
        EXPECT_EQ(reloaded.field_components(h), 3u);
    }

    std::filesystem::remove(out_path);
#  endif
#endif
}

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

    svmp::FE::timestepping::NewtonOptions newton_options;
    newton_options.residual_op = "equations";
    newton_options.jacobian_op = "equations";
    svmp::FE::timestepping::NewtonSolver newton(newton_options);
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
    const auto* entity_map = system.dofHandler().getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto u = history.uSpan();
    double max_err = 0.0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh->n_vertices()); ++v) {
        const auto vdofs = entity_map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(v));
        ASSERT_EQ(vdofs.size(), 1u);
        const auto dof = vdofs[0];
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), u.size());
        const auto xyz = base.get_vertex_coords(v);
        const double expected = 1.0 - static_cast<double>(xyz[0]);
        max_err = std::max(max_err, std::abs(static_cast<double>(u[static_cast<std::size_t>(dof)]) - expected));
    }
    EXPECT_LT(max_err, 1e-6);

    // Write the solution to a VTU file in the test working directory.
    const auto out_path = std::filesystem::path("poisson_square_solution.vtu");
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

    svmp::FE::timestepping::NewtonOptions newton_options;
    newton_options.residual_op = "equations";
    newton_options.jacobian_op = "equations";
    svmp::FE::timestepping::NewtonSolver newton(newton_options);
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
    svmp::FE::systems::SetupOptions setup_opts;
    setup_opts.use_backend_row_ownership_for_assembly = true;
#  if defined(MESH_HAS_MPI)
    setup_opts.dof_options.my_rank = world.rank();
    setup_opts.dof_options.world_size = world.size();
    setup_opts.dof_options.mpi_comm = world.native();
#  endif
    bool setup_ok = true;
    std::string setup_error;
    try {
        system.setup(setup_opts);
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
    lopt.rel_tol = 1e-8;
    lopt.abs_tol = 1e-10;
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

    svmp::FE::timestepping::NewtonOptions newton_options;
    newton_options.residual_op = "equations";
    newton_options.jacobian_op = "equations";
    svmp::FE::timestepping::NewtonSolver newton(newton_options);
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
    const auto& owned_dofs = system.dofHandler().getPartition().locallyOwned();
    double max_err_local = 0.0;
    int checked_local = 0;
    int eval_ok = (entity_map != nullptr && static_cast<bool>(u_view)) ? 1 : 0;
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(base.n_vertices()); ++v) {
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
        if (!owned_dofs.contains(dof)) {
            continue;
        }
        const double value = static_cast<double>(u_view->getVectorEntry(dof));
        const auto xyz = base.get_vertex_coords(v);
        const double expected = 1.0 - static_cast<double>(xyz[0]);
        max_err_local = std::max(max_err_local, std::abs(value - expected));
        ++checked_local;
    }

    if (!eval_ok) {
        max_err_local = std::numeric_limits<double>::infinity();
    }

    double max_err = max_err_local;
    int checked_global = checked_local;
#  if defined(MESH_HAS_MPI)
    MPI_Allreduce(&max_err_local, &max_err, 1, MPI_DOUBLE, MPI_MAX, world.native());
    MPI_Allreduce(&checked_local, &checked_global, 1, MPI_INT, MPI_SUM, world.native());
    {
        int all_ok = 0;
        MPI_Allreduce(&eval_ok, &all_ok, 1, MPI_INT, MPI_MIN, world.native());
        eval_ok = all_ok;
    }
#  endif
    if (world.rank() == 0) {
        EXPECT_EQ(eval_ok, 1);
        EXPECT_GT(checked_global, 0);
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
