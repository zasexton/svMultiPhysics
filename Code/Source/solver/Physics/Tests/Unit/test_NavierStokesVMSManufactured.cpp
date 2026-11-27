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

#include "FE/Assembly/FunctionalAssembler.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/FormKernels.h"
#include "FE/Forms/JIT/JITKernelWrapper.h"
#include "FE/Forms/MixedBlockKernelSet.h"
#include "FE/Forms/MonolithicCellKernel.h"
#include "FE/LevelSet/LevelSetGeometryService.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"
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
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
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

void expectSimplexGradDivWork(int dim, const std::vector<real_t>& coordinates,
                              const std::vector<std::vector<index_t>>& orderings,
                              double measure, double expected_tau_c, bool cut = false,
                              double mass_scale = 1.0, double frequency_squared_floor = 1e-12)
{
    const CellShape shape{dim == 2 ? CellFamily::Triangle : CellFamily::Tetra, dim + 1, 1};
    const auto element = dim == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
    const double viscosity = mass_scale * 0.00088733;
    for (const auto& vertices : orderings) {
        SCOPED_TRACE(::testing::PrintToString(vertices));
        auto base = std::make_shared<MeshBase>();
        base->build_from_arrays(dim, coordinates, {0, dim + 1}, vertices, {shape});
        base->finalize();
        auto mesh = create_mesh(base);
        FE::systems::FESystem system(mesh);
        auto scalar = std::make_shared<FE::spaces::H1Space>(element, 1);
        auto velocity = std::make_shared<FE::spaces::ProductSpace>(scalar, dim);
        ns::IncompressibleNavierStokesVMSOptions options;
        options.density = mass_scale * 997.0;
        options.stabilization_epsilon = frequency_squared_floor;
        options.viscosity = viscosity;
        options.enable_convection = false;
        options.enable_vms = true;
        options.jit_policy.enable = true;
        options.jit_policy.enable_was_explicitly_set = true;
        std::optional<FE::level_set::CutGeometry> geometry;
        std::vector<double> nodal_phi;
        FE::FieldId phi = FE::INVALID_FIELD_ID;
        if (cut) {
            // Retain x < h_x/2. Its fraction of the parent measure is 1 - 2^(-dim).
            for (int node = 0; node <= dim; ++node) {
                nodal_phi.push_back(coordinates[node * dim] - 0.5 * coordinates[dim]);
            }
            const auto field = base->attach_field(EntityKind::Vertex, "phi", FieldScalarType::Float64, 1);
            std::copy(nodal_phi.begin(), nodal_phi.end(), base->field_data_as<real_t>(field));
            phi = system.addField({.name = "phi", .space = scalar,
                                   .source_kind = FE::systems::FieldSourceKind::PrescribedData});
            constexpr int marker = 27190;
            geometry = FE::level_set::cutGeometry(system, *mesh, {.level_set_field_name = "phi",
                .domain_id = "half_leg", .interface_marker = marker});
            (void)geometry->side(FE::forms::CutVolumeSide::Negative);
            options.free_surface.push_back({.implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = marker, .level_set_field_name = "phi", .generated_interface_domain_id = "half_leg",
                .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative, .surface_tension = 0.0,
                .small_cut_aggregation = false});
        } else {
            options.node_pressure_constraints.id_type =
                ns::IncompressibleNavierStokesVMSOptions::NodePressureConstraintIdType::LocalVertexId;
            options.node_pressure_constraints.values = {{.node_id = 0, .pressure = 0.0}};
        }
        ns::IncompressibleNavierStokesVMSModule(velocity, scalar, std::move(options)).registerOn(system);
        system.setup();
        if (geometry) {
            std::vector<double> coefficients(system.fieldDofHandler(phi).getNumDofs());
            (void)system.projectMeshVertexValuesToFieldCoefficients(phi, nodal_phi, 1, coefficients);
            system.setPrescribedFieldCoefficients(phi, coefficients);
            (void)geometry->refresh();
        }

        // u = x/dim has unit divergence and zero strong momentum residual at constant time.
        const auto u = system.findFieldByName("u");
        const auto* map = system.fieldDofHandler(u).getEntityDofMap();
        ASSERT_NE(map, nullptr);
        const auto n = system.dofHandler().getNumDofs();
        std::vector<double> values(n, 0.0);
        for (const auto vertex : vertices) {
            const auto x = mesh->get_vertex_coords(vertex);
            const auto dofs = map->getVertexDofs(vertex);
            ASSERT_EQ(dofs.size(), dim);
            for (int i = 0; i < dim; ++i) {
                values[system.fieldDofOffset(u) + dofs[i]] = x[i] / dim;
            }
        }
        FE::systems::SystemStateView state;
        state.dt = 0.001;
        state.u = state.u_prev = values;
        const auto time = FE::systems::BackwardDifferenceIntegrator().buildContext(1, state);
        state.time_integration = &time;
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_vector = true;
        FE::assembly::DenseVectorView residual(n);
        const auto result = system.assemble(request, state, nullptr, &residual);
        ASSERT_TRUE(result.success) << result.error_message;
        if (cut) {
            EXPECT_FALSE(system.operatorDefinition("equations").cut_volumes.empty());
        } else {
            expectCompiled(system);
        }

        // Testing the momentum residual with u gives integral(2 mu / dim + tau_C).
        const double work = std::inner_product(values.begin(), values.end(), residual.data().begin(), 0.0);
        EXPECT_NEAR(work / measure - 2.0 * viscosity / dim, expected_tau_c, 1.0e-8);
    }
}

TEST(NavierStokesVMSManufactured, BulkIntegrandsInferTheCellFromACompositeTestField)
{
    using FE::forms::FormExpr;
    using FE::forms::FunctionalFormKernel;
    using IntegrandsBuilder = ns::NavierStokesVMSIntegrands (*)(
        const ns::NavierStokesTerms&, const ns::NavierStokesVMSSettings&);
    static_assert(std::is_same_v<decltype(&ns::navierStokesVMSIntegrands), IntegrandsBuilder>);
    const IntegrandsBuilder build = &ns::navierStokesVMSIntegrands;

    auto scalar = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
    FE::systems::FESystem system(makeSquare(1));
    const auto field = system.addField({.name = "p", .space = scalar, .components = 1});
    system.setup();
    FE::assembly::FunctionalAssembler assembler;
    assembler.setMesh(system.meshAccess());
    assembler.setSpace(*scalar);
    assembler.setDofMap(system.fieldDofHandler(field).getDofMap());
    assembler.setTimeStep(1.0);

    const auto zero = FormExpr::constant(0.0);
    const auto one = FormExpr::constant(1.0);
    const auto zero_vector = FormExpr::asVector({zero, zero});
    const auto x = FormExpr::coordinate().component(0);
    // A symbolic field can be constructed before the caller has chosen an element.
    const FE::forms::SpaceSignature vector_space{.space_type = FE::spaces::SpaceType::Product,
        .field_type = FE::FieldType::Vector, .value_dimension = 2};
    const auto u = FormExpr::trialFunction(vector_space, "u");
    const auto pressure_work = [&](const FormExpr& v) {
        const ns::NavierStokesTerms terms{u, x, v, x, one, one, zero_vector, 2};
        const auto r = build(terms, {.convection = false, .ct_m = 0.0, .ct_c = 1.0, .epsilon = 1.0});
        FunctionalFormKernel kernel(r.pspg_pressure_gradient, FunctionalFormKernel::Domain::Cell,
                                    FE::assembly::RequiredData::BasicGeometry);
        return assembler.assembleScalar(kernel);
    };
    // With p = q = x, the pressure work integrates 1/sqrt(1 + G:G).
    // The two unit-square triangles both have G:G = (4 + 1 + 1 + 4) / 3.
    auto triangle = vector_space;
    triangle.element_type = FE::ElementType::Triangle3;
    EXPECT_NEAR(pressure_work(FormExpr::constant(2.0) * FormExpr::testFunction(triangle)),
                std::sqrt(3.0 / 13.0), 1e-12);
    // An unspecified element uses the original chart metric, with G:G = 7 on both triangles.
    EXPECT_NEAR(pressure_work(FormExpr::testFunction(vector_space)), 1.0 / std::sqrt(8.0), 1e-12);
}

TEST(NavierStokesVMSManufactured, QuadraticConvectionIntegratesPolynomialWork)
{
    const std::vector<std::vector<index_t>> cells{{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2, 3}};
    for (const bool jit : {false, true}) {
        for (const auto& vertices : cells) {
            const bool quad = vertices.size() == 4;
            const auto type = quad ? FE::ElementType::Quad4 : FE::ElementType::Triangle3;
            SCOPED_TRACE(::testing::Message() << "jit=" << jit << " vertices="
                                              << ::testing::PrintToString(vertices));
            auto base = std::make_shared<MeshBase>();
            if (quad) {
                base->build_from_arrays(2, {0, 0, 1, 0, 1, 1, 0, 1}, {0, 4}, vertices, {{CellFamily::Quad, 4, 1}});
            } else {
                base->build_from_arrays(2, {0, 0, 1, 0, 0, 1}, {0, 3}, vertices, {{CellFamily::Triangle, 3, 1}});
            }
            base->finalize();
            auto mesh = create_mesh(base);
            FE::systems::FESystem system(mesh);
            auto p2 = std::make_shared<FE::spaces::H1Space>(type, 2);
            auto p1 = std::make_shared<FE::spaces::H1Space>(type, 1);
            auto velocity = std::make_shared<FE::spaces::ProductSpace>(p2, 2);
            ns::IncompressibleNavierStokesVMSOptions options;
            options.density = 1;
            options.viscosity = 0.1;
            options.enable_vms = false;
            options.enable_convection = true;
            options.jit_policy.enable = jit;
            options.jit_policy.enable_was_explicitly_set = true;
            ns::IncompressibleNavierStokesVMSModule(velocity, p1, std::move(options)).registerOn(system);
            system.setup();

            const auto u = system.findFieldByName("u");
            const auto cell_dofs = system.fieldDofHandler(u).getCellDofs(0);
            const auto offset = system.fieldDofOffset(u);
            const auto n = system.dofHandler().getNumDofs();
            std::vector<double> direction(n, 0), test(n, 0), local;
            const auto a = mesh->get_vertex_coords(vertices[0]);
            const auto b = mesh->get_vertex_coords(vertices[1]);
            const auto c = mesh->get_vertex_coords(vertices[2]);
            using Point = FE::spaces::FunctionSpace::Value;
            velocity->interpolate([&](const Point& xi) {
                if (quad) {
                    const double x = (1 + xi[0]) / 2, y = (1 + xi[1]) / 2;
                    return Point{x * y * y, 0, 0};
                }
                const double x = a[0] + xi[0] * (b[0] - a[0]) + xi[1] * (c[0] - a[0]);
                const double y = a[1] + xi[0] * (b[1] - a[1]) + xi[1] * (c[1] - a[1]);
                return Point{x * x, -2 * x * y, 0};
            }, local);
            ASSERT_EQ(local.size(), cell_dofs.size());
            for (std::size_t i = 0; i < local.size(); ++i) {
                direction[offset + cell_dofs[i]] = local[i];
                if (i < local.size() / 2) test[offset + cell_dofs[i]] = local[i];
            }

            FE::systems::SystemStateView field_state;
            field_state.u = direction;
            const auto point = system.evaluateFieldAtPoint(u, field_state, {0.2, 0.3, 0}, 0);
            ASSERT_TRUE(point.has_value());
            EXPECT_NEAR((*point)[0], quad ? 0.018 : 0.04, 1e-12);
            EXPECT_NEAR((*point)[1], quad ? 0.0 : -0.12, 1e-12);

            // Adding R(u) and R(-u) isolates the quadratic term. Subtracting
            // the two Jacobians also cancels the mass and viscous tangents.
            double convection = 0, tangent = 0;
            for (const double sign : {-1., 1.}) {
                auto values = direction;
                for (auto& value : values) value *= sign;
                FE::systems::SystemStateView state;
                state.dt = 1;
                state.u = state.u_prev = values;
                const auto integration = FE::systems::BackwardDifferenceIntegrator().buildContext(1, state);
                state.time_integration = &integration;
                FE::systems::AssemblyRequest request;
                request.op = "equations";
                request.want_matrix = request.want_vector = true;
                FE::assembly::DenseMatrixView matrix(n);
                FE::assembly::DenseVectorView residual(n);
                const auto result = system.assemble(request, state, &matrix, &residual);
                ASSERT_TRUE(result.success) << result.error_message;
                for (FE::GlobalIndex i = 0; i < n; ++i) {
                    convection += 0.5 * test[i] * residual.data()[i];
                    for (FE::GlobalIndex j = 0; j < n; ++j) {
                        tangent += 0.25 * sign * test[i] * matrix.getMatrixEntry(i, j) * direction[j];
                    }
                }
            }
            // The work density is 2x⁵ on the unit triangle and x²y⁶ on the unit
            // square. Both integrals equal 1/21.
            EXPECT_NEAR(convection, 1. / 21., 2e-12);
            EXPECT_NEAR(tangent, 1. / 21., 2e-12);
        }
    }
}

// Four triangles and a quadrilateral in a row, split in blocks over two ranks:
// the rank holding only triangles takes the degree 3p = 6 of the
// quadrilateral, so every cell of either rank is integrated alike.
TEST(NavierStokesVMSManufacturedMPI, ConvectionDegreeDoesNotDependOnThePartition)
{
#if !FE_HAS_MPI
    GTEST_SKIP() << "Requires MPI";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    int size = 1;
    if (initialized) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    if (size != 2) {
        GTEST_SKIP() << "Requires two MPI ranks";
    }
    // [0,3] x [0,1]: vertices 0-3 along y = 0 and 4-7 along y = 1.
    const std::vector<real_t> coordinates{0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1};
    const std::vector<index_t> vertices{0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 6};
    const std::vector<offset_t> offsets{0, 3, 6, 9, 12, 16};
    const CellShape triangle{CellFamily::Triangle, 3, 1};
    const CellShape quad{CellFamily::Quad, 4, 1};
    auto mesh = std::make_shared<Mesh>(MeshComm(MPI_COMM_WORLD));
    mesh->build_from_arrays_global_and_partition(2, coordinates, offsets, vertices,
                                                 {triangle, triangle, triangle, triangle, quad},
                                                 PartitionHint::Cells, /*ghost_layers=*/0,
                                                 {{"partition_method", "block"}});
    FE::systems::FESystem system(mesh);
    int holds_quad = 0;
    system.meshAccess().forEachCell([&](FE::GlobalIndex cell) {
        holds_quad |= system.meshAccess().getCellType(cell) == FE::ElementType::Quad4 ? 1 : 0;
    });
    int ranks_with_quad = 0;
    MPI_Allreduce(&holds_quad, &ranks_with_quad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_EQ(ranks_with_quad, 1);

    const auto velocity = [](int order) {
        return std::make_shared<FE::spaces::ProductSpace>(
            std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, order), 2);
    };
    EXPECT_EQ(ns::convectionCellQuadratureDegree(system, *velocity(2)), 6);
    EXPECT_EQ(ns::convectionCellQuadratureDegree(system, *velocity(1)), 0);
#endif
}

// On a coordinate-axis simplex with leg lengths h_i, the unit-determinant
// regular-simplex metric has G_ij = (1 + delta_ij) / ((dim+1)^(1/dim) h_i h_j).
// The two tau_C values use this closed form and the 1 ms physical step.
TEST(NavierStokesVMSManufactured, GradDivWorkScalesWithTheMassUnit)
{
    // The unit right triangle has tr(G) = 4/sqrt(3) and G:G = 10/3.
    // A mass-unit change multiplies rho, mu and pressure work by the same
    // factor while nu, the step and all inverse frequencies stay fixed.
    constexpr double nu = 0.00088733 / 997.0;
    constexpr double eps = 4e6;
    const double frequency = std::sqrt(4e6 + 36 * nu * nu * (10.0 / 3) + eps);
    for (double scale : {0.125, 1.0}) {
        const double tau_c = scale * 997 * frequency * std::sqrt(3.0) / 4;
        expectSimplexGradDivWork(2, {0, 0, 1, 0, 0, 1}, {{0, 1, 2}}, 0.5, tau_c,
                                 false, scale, eps);
    }
}

TEST(NavierStokesVMSManufactured, TriangleGradDivWorkIsInvariantToVertexOrdering)
{
    expectSimplexGradDivWork(2, {0, 0, 0.025, 0, 0, 0.0125},
                             {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}},
                             0.025 * 0.0125 / 2.0, 215.85683194317019);
}

TEST(NavierStokesVMSManufactured, TetrahedronGradDivWorkIsInvariantToVertexOrdering)
{
    expectSimplexGradDivWork(3, {0, 0, 0, 0.025, 0, 0, 0, 0.0125, 0, 0, 0, 0.05},
                             {{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 3, 0, 1}, {3, 0, 2, 1}},
                             0.025 * 0.0125 * 0.05 / 6.0, 188.40938681709790);
}

TEST(NavierStokesVMSManufactured, CutTetrahedronGradDivWorkUsesBarycentricGradientsAcrossReferenceOrigins)
{
    // For an axis-aligned tetrahedron sum_a |grad lambda_a|^2 = 2 sum_i h_i^-2.
    // The transient and viscous terms in tau_M still use the unit-determinant metric.
    constexpr double hx = 0.025, hy = 0.0125, hz = 0.05;
    const double inverse_length_squared = 2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
    const std::array<double, 3> h{hx, hy, hz};
    double metric_norm_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double gij = (i == j ? 2.0 : 1.0) / (std::cbrt(4.0) * h[i] * h[j]);
            metric_norm_squared += gij * gij;
        }
    }
    constexpr double nu = 0.00088733 / 997.0;
    const double tau_m = 1.0 / (997.0 * std::sqrt(4.0e6 + 36.0 * nu * nu * metric_norm_squared + 1.0e-12));
    const double expected_tau_c = 1.0 / (tau_m * inverse_length_squared);
    expectSimplexGradDivWork(3, {0, 0, 0, hx, 0, 0, 0, hy, 0, 0, 0, hz},
                             {{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 3, 0, 1}, {3, 0, 2, 1}},
                             (7.0 / 8.0) * hx * hy * hz / 6.0, expected_tau_c, true);
}

TEST(NavierStokesVMSManufactured, CutTractionAnchorsUniformPressureDuringUniformAcceleration)
{
    namespace fs = FE::systems;
    // The wet part of the unit reference tetrahedron lies below z = 0.4.
    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(3, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 4}, {0, 1, 2, 3},
                            {{CellFamily::Tetra, 4, 1}});
    base->finalize();
    const double diagonal = 1.0 / std::sqrt(3.0);
    const std::array<std::array<double, 3>, 4> normals{{
        {diagonal, diagonal, diagonal}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}};
    for (int v = 0; v < 4; ++v) {
        const int marker = v + 1;
        base->register_label("opposite_" + std::to_string(v), marker);
        for (auto face : base->boundary_faces()) {
            const auto vertices = base->face_vertices(face);
            if (std::find(vertices.begin(), vertices.end(), v) == vertices.end()) {
                base->set_boundary_label(face, marker);
            }
        }
    }
    const std::array<double, 4> nodal_phi{-0.4, -0.4, -0.4, 0.6};
    const auto attached = base->attach_field(EntityKind::Vertex, "phi", FieldScalarType::Float64, 1);
    std::copy(nodal_phi.begin(), nodal_phi.end(), base->field_data_as<real_t>(attached));
    auto mesh = create_mesh(base);
    fs::FESystem system(mesh);
    auto scalar = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    auto velocity = std::make_shared<FE::spaces::ProductSpace>(scalar, 3);
    const auto phi = system.addField({.name = "phi", .space = scalar, .source_kind = fs::FieldSourceKind::PrescribedData});
    constexpr int interface_marker = 27190;
    auto geometry = FE::level_set::cutGeometry(system, *mesh, {.level_set_field_name = "phi",
        .domain_id = "slab", .interface_marker = interface_marker});
    (void)geometry.side(FE::forms::CutVolumeSide::Negative);

    // sigma = -P I for a uniform velocity.  Apply its traction on each wetted
    // exterior face and the generated cut. No pressure pin or velocity constraint.
    const auto pressure = fs::timeFunction(system, "P", [](double t) { return 1.0 + 2.0 * t; });
    ns::IncompressibleNavierStokesVMSOptions options;
    options.density = 1.0;
    options.viscosity = 0.01;
    options.enable_convection = false;
    options.has_body_force_spacetime = true;
    options.body_force_spacetime = {0.1, 0.0, fs::timeFunction(system, "az", [](double) { return 0.2; })};
    options.free_surface.push_back({.implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker, .level_set_field_name = "phi", .generated_interface_domain_id = "slab",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative, .external_pressure = pressure,
        .surface_tension = 0.0, .small_cut_aggregation = false});
    for (int i = 0; i < 4; ++i) {
        const auto& normal = normals[i];
        options.traction_neumann.push_back({.boundary_marker = i + 1,
            .traction = {-normal[0] * pressure, -normal[1] * pressure, -normal[2] * pressure}});
    }
    ns::IncompressibleNavierStokesVMSModule(velocity, scalar, options).registerOn(system);
    system.setup();
    std::vector<double> coefficients(system.fieldDofHandler(phi).getNumDofs());
    (void)system.projectMeshVertexValuesToFieldCoefficients(phi, nodal_phi, 1, coefficients);
    system.setPrescribedFieldCoefficients(phi, coefficients);
    (void)geometry.refresh();
    const auto u = system.findFieldByName("u");
    const auto p = system.findFieldByName("p");
    const auto n = system.dofHandler().getNumDofs();
    const auto state_at = [&](double time, double pressure_error) {
        std::vector<double> values(n);
        const std::array<double, 3> velocity_value{0.45 + 0.1 * time, -0.2, 0.35 + 0.2 * time};
        for (int vertex = 0; vertex < 4; ++vertex) {
            const auto ud = system.fieldDofHandler(u).getEntityDofMap()->getVertexDofs(vertex);
            const auto pd = system.fieldDofHandler(p).getEntityDofMap()->getVertexDofs(vertex);
            for (int i = 0; i < 3; ++i) values[system.fieldDofOffset(u) + ud[i]] = velocity_value[i];
            values[system.fieldDofOffset(p) + pd[0]] = 1.0 + 2.0 * time + pressure_error;
        }
        return values;
    };
    constexpr double dt = 0.1;
    for (double t : {0.25, 0.75}) {
        const auto previous = state_at(t - dt, 0.0);
        for (double pressure_error : {0.0, 1.0}) {
            const auto state = state_at(t, pressure_error);
            fs::SystemStateView view{.time = t, .dt = dt, .u = state, .u_prev = previous};
            const auto integration = fs::BackwardDifferenceIntegrator().buildContext(1, view);
            view.time_integration = &integration;
            fs::AssemblyRequest request;
            request.op = "equations";
            request.want_vector = true;
            FE::assembly::DenseVectorView residual(n);
            const auto assembled = system.assemble(request, view, nullptr, &residual);
            ASSERT_TRUE(assembled.success) << assembled.error_message;
            // Test with v=(x,0,0), q=0. A unit pressure error gives -|Omega_wet|.
            const auto x_dofs = system.fieldDofHandler(u).getEntityDofMap()->getVertexDofs(1);
            const double wet_volume = (1.0 - std::pow(0.6, 3)) / 6.0;
            EXPECT_NEAR(residual.getVectorEntry(system.fieldDofOffset(u) + x_dofs[0]),
                        -pressure_error * wet_volume, 1e-12);
            if (pressure_error == 0.0) {
                for (auto r : residual.data()) EXPECT_NEAR(r, 0.0, 1e-12);
            }
        }
    }
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
