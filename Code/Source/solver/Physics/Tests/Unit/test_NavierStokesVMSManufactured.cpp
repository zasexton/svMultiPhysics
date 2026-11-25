/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#if SVMP_FE_WITH_MESH && FE_HAS_EIGEN && SVMP_FE_ENABLE_LLVM_JIT

#include "NavierStokesManufacturedSolution.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/JIT/JITKernelWrapper.h"
#include "FE/Forms/MixedBlockKernelSet.h"
#include "FE/Forms/MonolithicCellKernel.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeSeries.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "FE/TimeStepping/TimeLoop.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"
#include "Mesh/Mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::Physics::test {
namespace {

namespace ns = formulations::navier_stokes;
namespace ts = FE::timestepping;
using ns_mms::ErrorNorms;

constexpr int boundary_marker = 1;
constexpr double final_time = 0.25;
constexpr double rho_inf = 0.5;

std::shared_ptr<Mesh> makeSquare(int n)
{
    std::vector<real_t> coordinates;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            coordinates.push_back(static_cast<double>(i) / n);
            coordinates.push_back(static_cast<double>(j) / n);
        }
    }
    std::vector<index_t> vertices;
    std::vector<offset_t> offsets{0};
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const index_t sw = j * (n + 1) + i;
            const index_t nw = sw + n + 1;
            vertices.insert(vertices.end(), {sw, sw + 1, nw + 1, sw, nw + 1, nw});
            offsets.push_back(offsets.back() + 3);
            offsets.push_back(offsets.back() + 3);
        }
    }
    auto base = std::make_shared<MeshBase>();
    const CellShape tri{.family = CellFamily::Triangle, .num_corners = 3, .order = 1};
    base->build_from_arrays(2, coordinates, offsets, vertices, std::vector<CellShape>(2 * n * n, tri));
    base->finalize();
    base->register_label("boundary", boundary_marker);
    for (const auto face : base->boundary_faces()) {
        base->set_boundary_label(face, boundary_marker);
    }
    return create_mesh(std::move(base));
}

void installFluid(FE::systems::FESystem& system)
{
    // Each parameter is updated at the assembly time. The spatial polynomial
    // is part of the compiled residual, including its occurrences in PSPG and SUPG.
    using namespace FE::forms;
    const auto coordinates = FormExpr::coordinate();
    const auto force = ns_mms::bodyAcceleration(
        component(coordinates, 0), component(coordinates, 1),
        FE::systems::timeFunction(system, "mms_amplitude", ns_mms::velocityAmplitude),
        FE::systems::timeFunction(system, "mms_amplitude_rate", ns_mms::velocityAmplitudeRate),
        FE::systems::timeFunction(system, "mms_pressure", ns_mms::pressureAmplitude));

    ns::IncompressibleNavierStokesVMSOptions options;
    options.density = ns_mms::kDensity;
    options.viscosity = ns_mms::kViscosity;
    options.enable_convection = true;
    options.enable_vms = true;
    options.jit_policy.enable = true;
    options.jit_policy.enable_was_explicitly_set = true;
    options.has_body_force_spacetime = true;
    options.body_force_spacetime = {force[0], force[1], 0.0};
    // The streamfunction and its first derivatives vanish on all four sides.
    options.velocity_dirichlet = {{.boundary_marker = boundary_marker, .value = {1.0, 0.0, 0.0}}};
    options.node_pressure_constraints.id_type = ns::IncompressibleNavierStokesVMSOptions::
        NodePressureConstraintIdType::LocalVertexId;
    options.node_pressure_constraints.values = {{.node_id = 0, .pressure = 0.0}};

    auto p1 = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
    auto velocity = std::make_shared<FE::spaces::ProductSpace>(p1, 2);
    ns::IncompressibleNavierStokesVMSModule(velocity, p1, std::move(options)).registerOn(system);
    system.setup();
}

void expectCompiled(const FE::systems::FESystem& system)
{
    using namespace FE::forms;
    const auto expect_block = [](const auto& kernel) {
        ASSERT_NE(kernel, nullptr);
        const auto* block = dynamic_cast<const jit::JITKernelWrapper*>(kernel.get());
        ASSERT_NE(block, nullptr) << kernel->name();
        EXPECT_TRUE(block->isJITReady()) << block->name();
    };
    const auto expect_blocks = [&](const auto& grouped) {
        ASSERT_GT(grouped.numBlocks(), 0u);
        for (std::size_t i = 0; i < grouped.numBlocks(); ++i) {
            expect_block(grouped.blockSpec(i).fallback_kernel);
        }
    };
    const auto& cells = system.operatorDefinition("equations").cells;
    ASSERT_FALSE(cells.empty());
    for (const auto& term : cells) {
        if (const auto* fused = dynamic_cast<const MonolithicCellKernel*>(term.kernel.get())) {
            // Residual-only assembly uses the blocks even when a fused tangent is available.
            expect_blocks(*fused);
        } else if (const auto* mixed = dynamic_cast<const MixedBlockKernelSet*>(term.kernel.get())) {
            expect_blocks(*mixed);
        } else {
            expect_block(term.kernel);
        }
    }
}

struct VertexDofs {
    FE::GlobalIndex ux, uy, p;
};

std::vector<VertexDofs> vertexDofs(const FE::systems::FESystem& system, std::size_t count)
{
    const auto u = system.findFieldByName("u");
    const auto p = system.findFieldByName("p");
    const auto* umap = system.fieldDofHandler(u).getEntityDofMap();
    const auto* pmap = system.fieldDofHandler(p).getEntityDofMap();
    if (!umap || !pmap) {
        throw std::runtime_error("Expected nodal velocity and pressure DOFs");
    }
    std::vector<VertexDofs> dofs;
    for (std::size_t v = 0; v < count; ++v) {
        const auto uv = umap->getVertexDofs(v);
        const auto pv = pmap->getVertexDofs(v);
        if (uv.size() != 2 || pv.size() != 1) {
            throw std::runtime_error("Expected P1/P1 vertex DOFs on the square");
        }
        dofs.push_back({system.fieldDofOffset(u) + uv[0], system.fieldDofOffset(u) + uv[1],
                        system.fieldDofOffset(p) + pv[0]});
    }
    return dofs;
}

// A second time difference divided by four recovers the amplitude of an
// alternating error. Remove the nodal mean on each stage to eliminate the gauge.
struct PressureHistory {
    std::vector<double> previous, older;
    double peak{}, late_peak{};

    void observe(const Mesh& mesh, std::span<const double> p, double time, double end)
    {
        std::vector<double> error(p.size());
        for (std::size_t v = 0; v < p.size(); ++v) {
            const auto x = mesh.get_vertex_coords(v);
            error[v] = p[v] - ns_mms::pressure(x[0], x[1], time);
        }
        const double mean = std::accumulate(error.begin(), error.end(), 0.0) / error.size();
        for (auto& value : error) {
            value -= mean;
        }
        if (!older.empty()) {
            double sum = 0.0;
            for (std::size_t v = 0; v < p.size(); ++v) {
                const double d = error[v] - 2 * previous[v] + older[v];
                sum += d * d;
            }
            const double amplitude = 0.25 * std::sqrt(sum / p.size());
            peak = std::max(peak, amplitude);
            if (time >= 0.5 * end) {
                late_peak = std::max(late_peak, amplitude);
            }
        }
        older = std::move(previous);
        previous = std::move(error);
    }
};

struct Result {
    ErrorNorms error;
    ns_mms::SpatialPressureCheck spatial;
    double temporal_peak, late_temporal_peak;
};

Result solve(int n, int steps, bool monitor_pressure = false)
{
    const double dt = final_time / steps;
    SCOPED_TRACE(::testing::Message() << "N=" << n << ", dt=" << dt);
    const auto mesh = makeSquare(n);
    FE::systems::FESystem system(mesh);
    installFluid(system);
    const auto dofs = vertexDofs(system, mesh->n_vertices());

    using namespace FE::backends;
    auto factory = BackendFactory::create(BackendKind::Eigen);
    SolverOptions linear_options;
    linear_options.method = SolverMethod::Direct;
    auto linear = factory->createLinearSolver(linear_options);
    auto history = ts::TimeHistory::allocate(*factory, system.dofHandler().getNumDofs());
    // The exact initial fields are constant and the exact velocity rate is zero.
    history.ensureSecondOrderState(*factory);
    history.uDot().zero();
    history.u().zero();
    for (std::size_t v = 0; v < dofs.size(); ++v) {
        const auto x = mesh->get_vertex_coords(v);
        const auto u = ns_mms::velocity(x[0], x[1], 0.0);
        const auto rate = ns_mms::velocityRate(x[0], x[1], 0.0);
        const auto& d = dofs[v];
        history.uSpan()[d.ux] = u[0];
        history.uSpan()[d.uy] = u[1];
        history.uSpan()[d.p] = ns_mms::pressure(x[0], x[1], 0.0);
        history.uDotSpan()[d.ux] = rate[0];
        history.uDotSpan()[d.uy] = rate[1];
    }
    for (int k = 1; k <= history.historyDepth(); ++k) {
        history.uPrevK(k).copyFrom(history.u());
    }
    history.primeDtHistory(dt);

    const auto ga = ts::utils::generalizedAlphaFirstOrderFromRhoInf(rho_inf);
    auto integrator = std::make_shared<ts::GeneralizedAlphaFirstOrderIntegrator>(
        ts::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m, .alpha_f = ga.alpha_f, .gamma = ga.gamma, .history_rate_order = 0});
    FE::systems::TransientSystem transient(system, integrator);
    ts::TimeLoopOptions options;
    options.scheme = ts::SchemeKind::GeneralizedAlpha;
    options.generalized_alpha_rho_inf = rho_inf;
    options.t_end = final_time;
    options.dt = dt;
    options.max_steps = steps;
    options.newton.residual_op = options.newton.jacobian_op = "equations";
    options.newton.abs_tolerance = 1e-11;
    options.newton.rel_tolerance = 1e-10;

    double stage_time = 0.0;
    std::vector<double> stage_pressure(dofs.size());
    PressureHistory pressure_history;
    ts::TimeLoopCallbacks callbacks;
    callbacks.on_candidate_stage = [&](const ts::CandidateStageObservation& stage) {
        stage_time = stage.state_time;
        const auto state = stage.state_vector->localSpan();
        for (std::size_t v = 0; v < dofs.size(); ++v) {
            stage_pressure[v] = state[dofs[v].p];
        }
    };
    if (monitor_pressure) {
        callbacks.on_step_accepted = [&](const ts::TimeHistory&) {
            pressure_history.observe(*mesh, stage_pressure, stage_time, final_time);
        };
    }
    const auto report = ts::TimeLoop(options).run(transient, *factory, *linear, history, callbacks);
    if (!report.success || report.steps_taken != steps) {
        throw std::runtime_error("Incomplete manufactured solution run: " + report.message);
    }
    EXPECT_NEAR(report.final_time, final_time, 1e-12);
    EXPECT_NEAR(stage_time, final_time - (1.0 - ga.alpha_f) * dt, 1e-12);
    expectCompiled(system);

    const auto endpoint = history.uPrevSpan();
    std::vector<ns_mms::Vector> velocity;
    for (const auto& d : dofs) {
        velocity.push_back({endpoint[d.ux], endpoint[d.uy]});
    }
    auto error = ns_mms::measureErrors(*mesh, velocity, stage_pressure, report.final_time, stage_time);
    const auto spatial = ns_mms::checkPressure(error.nodal_pressure_error, stage_pressure, n);
    return {std::move(error), spatial, pressure_history.peak, pressure_history.late_peak};
}

using FourNorms = std::array<double, 4>;

FourNorms norms(const ErrorNorms& error)
{
    return {error.velocity_l2, error.velocity_h1, error.pressure_l2, error.divergence_l2};
}

void printRow(const char* kind, int n, double dt, const FourNorms& values)
{
    std::ostringstream row;
    row << std::setprecision(10) << kind << ' ' << n << ' ' << dt;
    for (double value : values) {
        row << ' ' << value;
        EXPECT_TRUE(std::isfinite(value));
    }
    std::cout << row.str() << '\n';
}

FourNorms rates(const FourNorms& coarse, const FourNorms& fine)
{
    FourNorms result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = std::log2(coarse[i] / fine[i]);
    }
    return result;
}

TEST(NavierStokesVMSManufactured, SpatialConvergence)
{
    const auto coarse = norms(solve(12, 8).error);
    const auto fine = norms(solve(24, 8).error);
    const auto order = rates(coarse, fine);
    printRow("spatial_rate", 24, final_time / 8, order);
    EXPECT_GT(order[0], 1.4);
    EXPECT_GT(order[1], 0.7);
    EXPECT_GT(order[2], 0.6);
    EXPECT_GT(order[3], 0.5);
}

TEST(NavierStokesVMSManufactured, DISABLED_SpaceTimeGrid)
{
    std::cout << "quantity N dt velocity_L2 velocity_H1 pressure_L2 divergence_L2\n";
    std::array<FourNorms, 4> previous_mesh{};
    constexpr std::array mesh_sizes{12, 24, 48};
    for (int n : mesh_sizes) {
        FourNorms previous_dt{};
        int column = 0;
        for (int steps : {2, 4, 8, 16}) {
            const double dt = final_time / steps;
            const auto error = norms(solve(n, steps).error);
            printRow("error", n, dt, error);
            if (column != 0) {
                printRow("temporal_rate", n, dt, rates(previous_dt, error));
            }
            if (n != mesh_sizes.front()) {
                printRow("spatial_rate", n, dt, rates(previous_mesh[column], error));
            }
            previous_mesh[column++] = error;
            previous_dt = error;
        }
    }
}

TEST(NavierStokesVMSManufactured, DISABLED_SmallStepPressure)
{
    // The unit bulk velocity and side length give a unit convective time.
    constexpr int n = 12;
    std::cout << "quantity N dt velocity_L2 velocity_H1 pressure_L2 divergence_L2\n"
              << "pressure_metrics N dt spatial_ripple axial_extrema largest_reversal "
                 "max_time_ripple late_max_time_ripple\n";
    for (int steps : {25, 250, 2500}) {
        const double dt = final_time / steps;
        const auto result = solve(n, steps, true);
        printRow("small_step", n, dt, norms(result.error));
        EXPECT_TRUE(std::isfinite(result.spatial.ripple));
        EXPECT_TRUE(std::isfinite(result.temporal_peak));
        std::cout << std::setprecision(10) << "pressure_ripple " << n << ' ' << dt << ' '
                  << result.spatial.ripple << ' ' << result.spatial.extrema << ' '
                  << result.spatial.largest_reversal << ' ' << result.temporal_peak << ' '
                  << result.late_temporal_peak << '\n';
    }
}

} // namespace
} // namespace svmp::Physics::test

#else

TEST(NavierStokesVMSManufactured, SpatialConvergence)
{
    GTEST_SKIP() << "Requires native meshes, Eigen and LLVM JIT";
}

#endif
