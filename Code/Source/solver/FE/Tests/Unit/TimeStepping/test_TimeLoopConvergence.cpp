/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"

#include "Core/Types.h"

#include "Constraints/DirichletBC.h"
#include "Constraints/MultiPointConstraint.h"

#include "Forms/BoundaryConditions.h"
#include "Forms/Forms.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include "Spaces/H1Space.h"

#include "Systems/FESystem.h"
#include "Systems/FormsInstallerDetail.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"
#include "TimeStepping/VSVO_BDF_Controller.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace ts_test = svmp::FE::timestepping::test;

namespace {

using ts_test::createTestFactory;
using ts_test::directSolve;
using ts_test::getVectorByDof;
using ts_test::relativeL2Error;
using ts_test::setVectorByDof;
using ts_test::singleTetraTopology;

[[nodiscard]] svmp::FE::dofs::MeshTopologyInfo twoTetraSharedFaceTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 1, 2, 3, 4};
    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

class TwoTetraCellDomainMeshAccess final
    : public svmp::FE::assembly::IMeshAccess {
public:
    TwoTetraCellDomainMeshAccess()
        : nodes_{{{0.0, 0.0, 0.0},
                  {1.0, 0.0, 0.0},
                  {0.0, 1.0, 0.0},
                  {0.0, 0.0, 1.0},
                  {1.0, 1.0, 1.0}}}
        , cells_{{{0, 1, 2, 3}, {1, 2, 3, 4}}}
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Tetra4;
    }
    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override
    {
        return static_cast<int>(cell_id);
    }

    void getCellNodes(GlobalIndex cell_id,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(
        GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        GlobalIndex cell_id,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coordinates.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coordinates[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(
        svmp::FE::assembly::CoordinateFrame frame) const override
    {
        return frame == svmp::FE::assembly::CoordinateFrame::Active ||
               frame == svmp::FE::assembly::CoordinateFrame::Reference ||
               frame == svmp::FE::assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(
        GlobalIndex cell_id,
        svmp::FE::assembly::CoordinateFrame,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        getCellCoordinates(cell_id, coordinates);
    }

    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(
        GlobalIndex face_id,
        GlobalIndex cell_id) const override
    {
        if (face_id == 0 && cell_id == 0) return 2;
        if (face_id == 0 && cell_id == 1) return 0;
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(
        GlobalIndex face_id) const override
    {
        return face_id == 0 ? std::pair<GlobalIndex, GlobalIndex>{0, 1}
                            : std::pair<GlobalIndex, GlobalIndex>{0, 0};
    }

    void forEachCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(
        int,
        std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        callback(0, 0, 1);
    }

private:
    std::array<std::array<Real, 3>, 5> nodes_{};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{};
};

struct RateInitializationSlotObservations {
    bool saw_non_dt_residual{false};
    bool saw_dt_only_jacobian{false};
    Real max_abs_slot2_value{0.0};
};

struct RateInitializationTimeObservations {
    std::vector<Real> initialization_times{};
    std::vector<Real> initialization_constraint_values{};
    std::vector<Real> injected_stage_rates{};
    const svmp::FE::constraints::AffineConstraints* constraints{nullptr};
    GlobalIndex constrained_dof{svmp::FE::INVALID_GLOBAL_INDEX};
    bool recorded_stage_rate{false};
};

class RateInitializationSlotRecordingKernel final
    : public svmp::FE::assembly::AssemblyKernel {
public:
    explicit RateInitializationSlotRecordingKernel(
        RateInitializationSlotObservations& observations)
        : observations_(observations)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
        const override
    {
        return svmp::FE::assembly::RequiredData::SolutionCoefficients;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        (void)output;

        const auto* time_integration = ctx.timeIntegrationContext();
        if (time_integration == nullptr) {
            return;
        }
        const bool is_non_dt_residual =
            time_integration->time_derivative_term_weight == Real{0.0} &&
            time_integration->non_time_derivative_term_weight == Real{1.0};
        const bool is_dt_only_jacobian =
            time_integration->time_derivative_term_weight == Real{1.0} &&
            time_integration->non_time_derivative_term_weight == Real{0.0};
        if (!is_non_dt_residual && !is_dt_only_jacobian) {
            return;
        }

        observations_.saw_non_dt_residual |= is_non_dt_residual;
        observations_.saw_dt_only_jacobian |= is_dt_only_jacobian;
        const auto slot2 = ctx.previousSolutionCoefficientsRaw(2);
        for (const auto value : slot2) {
            observations_.max_abs_slot2_value = std::max(
                observations_.max_abs_slot2_value,
                static_cast<Real>(std::abs(value)));
        }
    }

private:
    RateInitializationSlotObservations& observations_;
};

class RateInitializationTimeRecordingKernel final
    : public svmp::FE::assembly::AssemblyKernel {
public:
    explicit RateInitializationTimeRecordingKernel(
        RateInitializationTimeObservations& observations)
        : observations_(observations)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
        const override
    {
        return svmp::FE::assembly::RequiredData::SolutionCoefficients;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext& ctx,
                     svmp::FE::assembly::KernelOutput& output) override
    {
        (void)output;

        const auto* time_integration = ctx.timeIntegrationContext();
        if (time_integration == nullptr) {
            return;
        }

        const bool is_non_dt_residual =
            time_integration->time_derivative_term_weight == Real{0.0} &&
            time_integration->non_time_derivative_term_weight == Real{1.0};
        const bool is_dt_only_jacobian =
            time_integration->time_derivative_term_weight == Real{1.0} &&
            time_integration->non_time_derivative_term_weight == Real{0.0};
        if (is_non_dt_residual || is_dt_only_jacobian) {
            observations_.initialization_times.push_back(ctx.time());
            if (observations_.constraints != nullptr &&
                observations_.constrained_dof >= 0) {
                observations_.initialization_constraint_values.push_back(
                    static_cast<Real>(
                        observations_.constraints->getInhomogeneity(
                            observations_.constrained_dof)));
            }
            return;
        }

        if (!observations_.recorded_stage_rate &&
            time_integration->time_derivative_term_weight == Real{1.0} &&
            time_integration->non_time_derivative_term_weight == Real{1.0} &&
            time_integration->integrator_name ==
                "GeneralizedAlpha(1stOrder)") {
            const auto rate = ctx.previousSolutionCoefficientsRaw(2);
            observations_.injected_stage_rates.assign(rate.begin(), rate.end());
            observations_.recorded_stage_rate = true;
        }
    }

private:
    RateInitializationTimeObservations& observations_;
};

[[nodiscard]] std::pair<double, double> coupledReactionExact2x2(double t,
                                                                double u0,
                                                                double w0,
                                                                double lambda_u,
                                                                double lambda_w,
                                                                double kappa)
{
    // System: u' + lambda_u*u + kappa*w = 0,  w' + kappa*u + lambda_w*w = 0.
    // Matrix form: x' = A x,  A = [[-lambda_u, -kappa], [-kappa, -lambda_w]].
    const double a = -lambda_u;
    const double d = -lambda_w;
    const double b = -kappa;

    const double m = 0.5 * (a + d);
    const double half_diff = 0.5 * (a - d);
    const double s = std::sqrt(half_diff * half_diff + b * b);

    const double exp_mt = std::exp(m * t);
    if (s == 0.0) {
        return {exp_mt * u0, exp_mt * w0};
    }

    const double cosh_st = std::cosh(s * t);
    const double sinh_over_s = std::sinh(s * t) / s;

    const double a_m = a - m;
    const double d_m = d - m;

    const double u = exp_mt * (cosh_st * u0 + sinh_over_s * (a_m * u0 + b * w0));
    const double w = exp_mt * (cosh_st * w0 + sinh_over_s * (b * u0 + d_m * w0));
    return {u, w};
}

std::vector<Real> runReactionProblem(svmp::FE::timestepping::SchemeKind scheme,
                                     double dt,
                                     double t_end,
                                     double lambda,
                                     int history_depth = 2,
                                     std::shared_ptr<svmp::FE::timestepping::StepController> controller = {},
                                     double generalized_alpha_rho_inf = 1.0,
                                     int dg_degree = 1,
                                     int cg_degree = 2,
                                     svmp::FE::timestepping::CollocationSolveStrategy collocation_solve =
                                         svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                     int collocation_max_outer_iterations = 4,
                                     double collocation_outer_tolerance = 0.0,
                                     bool exact_initial_history = false,
                                     double theta = 0.5,
                                     int newton_max_iterations = 8,
                                     double newton_abs_tolerance = 1e-12,
                                     double newton_rel_tolerance = 0.0,
                                     std::function<void(
                                         svmp::FE::timestepping::TimeLoopCallbacks&,
                                         svmp::FE::timestepping::TimeHistory&)>
                                         configure_callbacks = {},
                                     std::function<void(
                                         const svmp::FE::timestepping::TimeHistory&,
                                         const svmp::FE::FEException&)>
                                         inspect_expected_exception = {},
                                     std::function<void(
                                         svmp::FE::timestepping::TimeLoopOptions&,
                                         svmp::FE::FieldId)>
                                         configure_options = {},
                                     std::function<void(
                                         svmp::FE::systems::FESystem&,
                                         svmp::FE::FieldId)>
                                         configure_system = {})
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    if (configure_system) {
        configure_system(sys, u_field);
    }

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = ts_test::singleTetraTopology();
    sys.setup({}, inputs);
    if (!sys.isSetup()) {
        ADD_FAILURE() << "FESystem::setup did not mark the system as setup";
        return {};
    }

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = ts_test::createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(ts_test::directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, history_depth);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    if (u0.size() != static_cast<std::size_t>(n_dofs)) {
        ADD_FAILURE() << "Unexpected DOF count: got " << n_dofs << ", expected " << u0.size();
        return {};
    }
    for (int k = 1; k <= history.historyDepth(); ++k) {
        if (exact_initial_history) {
            const double t_k = -static_cast<double>(k - 1) * dt;
            const double scale = std::exp(-lambda * t_k);
            std::vector<Real> u_k(u0.size(), 0.0);
            for (std::size_t i = 0; i < u0.size(); ++i) {
                u_k[i] = static_cast<Real>(static_cast<double>(u0[i]) * scale);
            }
            ts_test::setVectorByDof(history.uPrevK(k), u_k);
        } else {
            ts_test::setVectorByDof(history.uPrevK(k), u0);
        }
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = scheme;
    opts.theta = theta;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.dg_degree = dg_degree;
    opts.cg_degree = cg_degree;
    opts.collocation_solve = collocation_solve;
    opts.collocation_max_outer_iterations = collocation_max_outer_iterations;
    opts.collocation_outer_tolerance = collocation_outer_tolerance;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = newton_max_iterations;
    opts.newton.abs_tolerance = newton_abs_tolerance;
    opts.newton.rel_tolerance = newton_rel_tolerance;
    opts.step_controller = std::move(controller);
    if (configure_options) {
        configure_options(opts, u_field);
    }

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };
    if (configure_callbacks) {
        configure_callbacks(callbacks, history);
    }

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        if (inspect_expected_exception) {
            inspect_expected_exception(history, e);
            return {};
        }
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return ts_test::getVectorByDof(history.uPrev());
}

std::vector<Real> runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind scheme,
                                                    double dt,
                                                    double t_end,
                                                    double lambda_u,
                                                    double lambda_w,
                                                    double kappa,
                                                    double theta = 0.5,
                                                    int history_depth = 2,
                                                    bool exact_initial_history = false)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoTetraSharedFaceMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto w_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "w", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto w = svmp::FE::forms::FormExpr::trialFunction(*space, "w");
    const auto v_u = svmp::FE::forms::FormExpr::testFunction(*space, "v_u");
    const auto v_w = svmp::FE::forms::FormExpr::testFunction(*space, "v_w");

    const auto form_uu = (svmp::FE::forms::dt(u) * v_u + (u * v_u) * static_cast<Real>(lambda_u)).dx();
    const auto form_uw = ((w * v_u) * static_cast<Real>(kappa)).dx();
    const auto form_wu = ((u * v_w) * static_cast<Real>(kappa)).dx();
    const auto form_ww = (svmp::FE::forms::dt(w) * v_w + (w * v_w) * static_cast<Real>(lambda_w)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir_uu = compiler.compileResidual(form_uu);
    auto ir_uw = compiler.compileResidual(form_uw);
    auto ir_wu = compiler.compileResidual(form_wu);
    auto ir_ww = compiler.compileResidual(form_ww);

    auto kernel_uu = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_uu), svmp::FE::forms::ADMode::Forward);
    auto kernel_uw = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_uw), svmp::FE::forms::ADMode::Forward);
    auto kernel_wu = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_wu), svmp::FE::forms::ADMode::Forward);
    auto kernel_ww = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir_ww), svmp::FE::forms::ADMode::Forward);

    sys.addCellKernel("op", u_field, u_field, kernel_uu);
    sys.addCellKernel("op", u_field, w_field, kernel_uw);
    sys.addCellKernel("op", w_field, u_field, kernel_wu);
    sys.addCellKernel("op", w_field, w_field, kernel_ww);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = twoTetraSharedFaceTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, history_depth);

    const auto n_u = sys.fieldDofHandler(u_field).getNumDofs();
    const auto n_w = sys.fieldDofHandler(w_field).getNumDofs();
    if (n_u != n_w || n_u <= 0) {
        ADD_FAILURE() << "Unexpected field DOF counts";
        return {};
    }
    const auto offset_u = sys.fieldDofOffset(u_field);
    const auto offset_w = sys.fieldDofOffset(w_field);
    if (offset_u < 0 || offset_w < 0) {
        ADD_FAILURE() << "Invalid field offsets";
        return {};
    }

    // TwoTetraSharedFaceMeshAccess with H1(Tetra4,P1) yields 5 vertex DOFs per field.
    if (n_u != 5) {
        ADD_FAILURE() << "Unexpected per-field DOF count: got " << n_u << ", expected 5";
        return {};
    }
    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};

    auto exactAt = [&](double t) -> std::vector<Real> {
        std::vector<Real> out(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_u; ++i) {
            const auto [ui, wi] = coupledReactionExact2x2(t, u0[static_cast<std::size_t>(i)],
                                                          w0[static_cast<std::size_t>(i)],
                                                          lambda_u, lambda_w, kappa);
            out[static_cast<std::size_t>(offset_u + i)] = static_cast<Real>(ui);
            out[static_cast<std::size_t>(offset_w + i)] = static_cast<Real>(wi);
        }
        return out;
    };

    for (int k = 1; k <= history.historyDepth(); ++k) {
        const double t_k = exact_initial_history ? -static_cast<double>(k - 1) * dt : 0.0;
        setVectorByDof(history.uPrevK(k), exactAt(t_k));
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 10000;
    opts.scheme = scheme;
    opts.theta = theta;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&,
                                             const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::shared_ptr<svmp::FE::timestepping::VSVO_BDF_Controller>
makeFixedVsvoBdfController(int order, double dt)
{
    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    // For fixed-step/order convergence tests, keep the controller inert.
    ctrl_opts.abs_tol = 1.0;
    ctrl_opts.rel_tol = 0.0;
    ctrl_opts.min_order = order;
    ctrl_opts.max_order = order;
    ctrl_opts.initial_order = order;
    ctrl_opts.max_retries = 0;
    ctrl_opts.safety = 1.0;
    ctrl_opts.min_factor = 1.0;
    ctrl_opts.max_factor = 1.0;
    ctrl_opts.min_dt = dt;
    ctrl_opts.max_dt = dt;
    ctrl_opts.pi_alpha = 0.0;
    ctrl_opts.pi_beta = 0.0;
    ctrl_opts.increase_order_threshold = 0.0;

    return std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
}

class RecordingVsvoBdfController final : public svmp::FE::timestepping::VSVO_BDF_Controller {
public:
    explicit RecordingVsvoBdfController(svmp::FE::timestepping::VSVO_BDF_ControllerOptions options)
        : svmp::FE::timestepping::VSVO_BDF_Controller(std::move(options))
    {
    }

    std::vector<svmp::FE::timestepping::StepAttemptInfo> accepted{};

    svmp::FE::timestepping::StepDecision onAccepted(const svmp::FE::timestepping::StepAttemptInfo& info) override
    {
        accepted.push_back(info);
        return svmp::FE::timestepping::VSVO_BDF_Controller::onAccepted(info);
    }

    svmp::FE::timestepping::StepDecision onRejected(const svmp::FE::timestepping::StepAttemptInfo& info,
                                                    svmp::FE::timestepping::StepRejectReason reason) override
    {
        return svmp::FE::timestepping::VSVO_BDF_Controller::onRejected(info, reason);
    }
};

std::vector<Real> runLogisticProblem(svmp::FE::timestepping::SchemeKind scheme,
                                     double dt,
                                     double t_end,
                                     double r,
                                     double K = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const Real rr = static_cast<Real>(r);
    const Real inv_K = static_cast<Real>(1.0 / K);

    // Logistic: u' = r*u*(1 - u/K) = r*u - r/K*u^2
    // Residual: u' - r*u + r/K*u^2 = 0.
    const auto form =
        (svmp::FE::forms::dt(u) * v - (u * v) * rr + (u * u * v) * (rr * inv_K)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs);

    const std::vector<Real> u0 = {0.2, 0.4, 0.6, 0.8};
    if (u0.size() != static_cast<std::size_t>(n_dofs)) {
        ADD_FAILURE() << "Unexpected DOF count: got " << n_dofs << ", expected " << u0.size();
        return {};
    }
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.theta = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runHeatManufacturedSinForcing(double dt, double t_end)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    double current_time = 0.0;
    const auto f = svmp::FE::forms::FormExpr::coefficient(
        "f",
        [&current_time](Real, Real, Real) { return static_cast<Real>(std::sin(current_time)); });

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form = (svmp::FE::forms::dt(u) * v +
                       svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)) -
                       f * v)
                          .dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());

    const std::vector<Real> u0(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_step_start = [&current_time](const svmp::FE::timestepping::TimeHistory& h) {
        current_time = h.time() + h.dt();
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2(double dt, double t_end, double omega)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 1000;
    opts.scheme = svmp::FE::timestepping::SchemeKind::BackwardEuler;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind scheme,
                                             double dt,
                                             double t_end,
                                             double omega,
                                             double generalized_alpha_rho_inf = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

struct ScalarTrajectory {
    std::vector<double> times;
    std::vector<double> values;
};

std::optional<double> estimatePeriodFromDownwardZeroCrossings(const ScalarTrajectory& traj)
{
    if (traj.times.size() != traj.values.size()) {
        return std::nullopt;
    }
    if (traj.times.size() < 2) {
        return std::nullopt;
    }

    std::vector<double> crossing_times;
    crossing_times.reserve(traj.times.size() / 2);

    for (std::size_t i = 1; i < traj.values.size(); ++i) {
        const double y0 = traj.values[i - 1];
        const double y1 = traj.values[i];
        if (!(y0 > 0.0 && y1 <= 0.0)) {
            continue;
        }
        const double t0 = traj.times[i - 1];
        const double t1 = traj.times[i];
        const double denom = y1 - y0;
        if (denom == 0.0) {
            continue;
        }
        const double s = (0.0 - y0) / denom;
        crossing_times.push_back(t0 + s * (t1 - t0));
    }

    if (crossing_times.size() < 3) {
        return std::nullopt;
    }

    double sum = 0.0;
    int count = 0;
    for (std::size_t i = 1; i < crossing_times.size(); ++i) {
        const double dt = crossing_times[i] - crossing_times[i - 1];
        if (dt > 0.0 && std::isfinite(dt)) {
            sum += dt;
            count += 1;
        }
    }
    if (count == 0) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

ScalarTrajectory runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind scheme,
                                                      double dt,
                                                      double t_end,
                                                      double omega,
                                                      double generalized_alpha_rho_inf = 1.0)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * c);
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = static_cast<int>(std::ceil(t_end / dt)) + 10;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    ScalarTrajectory traj;
    traj.times.push_back(0.0);
    traj.values.push_back(static_cast<double>(u0[0]));

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    svmp::FE::timestepping::NewtonReport last_nr;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&,
                                             const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };
    callbacks.on_step_accepted = [&traj](const svmp::FE::timestepping::TimeHistory& h) {
        const auto up = h.uPrevSpan();
        if (up.empty()) {
            return;
        }
        traj.times.push_back(h.time());
        traj.values.push_back(static_cast<double>(up[0]));
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm
                      << " step=" << history.stepIndex()
                      << " t=" << history.time()
                      << " dt=" << history.dt()
                      << " dt_prev=" << history.dtPrev() << ")";
        return traj;
    }
    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return traj;
}

std::vector<Real> runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind scheme,
                                                   double dt,
                                                   double t_end,
                                                   double omega,
                                                   double zeta,
                                                   double generalized_alpha_rho_inf = 1.0)
{
    if (!(zeta >= 0.0 && zeta < 1.0)) {
        ADD_FAILURE() << "runDampedOscillatorDt2Structural: zeta must be in [0,1)";
        return {};
    }

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v +
         svmp::FE::forms::dt(u) * v * static_cast<Real>(2.0 * zeta * omega) +
         (u * v) * static_cast<Real>(omega * omega))
            .dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    // Prime u^{n-1} using the exact damped oscillator at t=-dt with v(0)=0.
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double c = std::cos(omega_d * dt);
    const double s = std::sin(omega_d * dt);
    const double expm = std::exp(zeta * omega * dt);
    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * expm * (c - alpha * s));
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);

    const std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 4000;
    opts.scheme = scheme;
    opts.generalized_alpha_rho_inf = generalized_alpha_rho_inf;
    opts.newmark_beta = 0.25;
    opts.newmark_gamma = 0.5;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 16;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

std::vector<Real> runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind scheme,
                                              double dt,
                                              double t_end,
                                              double omega,
                                              int dg_degree = 1,
                                              int cg_degree = 2)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    svmp::FE::forms::FormCompiler compiler;
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    if (!factory) {
        ADD_FAILURE() << "Eigen backend not available; enable FE_ENABLE_EIGEN for TimeStepping tests";
        return {};
    }
    auto linear = factory->createLinearSolver(directSolve());
    if (!linear) {
        ADD_FAILURE() << "Linear solver not created";
        return {};
    }

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 sys.dofHandler().getNumDofs(),
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    if (u0.size() != static_cast<std::size_t>(sys.dofHandler().getNumDofs())) {
        ADD_FAILURE() << "Unexpected DOF count";
        return {};
    }
    setVectorByDof(history.uPrev(), u0);

    // Choose v0 = omega*u0 so the exact solution is u(t)=u0*(cos(omega t)+sin(omega t)).
    std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        v0[i] = static_cast<Real>(omega * static_cast<double>(u0[i]));
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    const double c = std::cos(omega * dt);
    const double s = std::sin(omega * dt);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c - s));
    }
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt;
    opts.max_steps = 2000;
    opts.scheme = scheme;
    opts.dg_degree = dg_degree;
    opts.cg_degree = cg_degree;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::NewtonReport last_nr;
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    callbacks.on_nonlinear_done = [&last_nr](const svmp::FE::timestepping::TimeHistory&, const svmp::FE::timestepping::NewtonReport& nr) {
        last_nr = nr;
    };

    svmp::FE::timestepping::TimeLoopReport rep;
    try {
        rep = loop.run(transient, *factory, *linear, history, callbacks);
    } catch (const svmp::FE::FEException& e) {
        ADD_FAILURE() << e.what()
                      << " (Newton iters=" << last_nr.iterations
                      << " r0=" << last_nr.residual_norm0
                      << " r=" << last_nr.residual_norm << ")";
        return {};
    }

    EXPECT_TRUE(rep.success);
    EXPECT_NEAR(rep.final_time, t_end, 1e-12);

    return getVectorByDof(history.uPrev());
}

} // namespace

TEST(NewtonSolverSanity, LinearReactionConvergesInOneUpdate)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr double dt = 0.2;
    constexpr double lambda = 1.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    // Workspace.
    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    svmp::FE::timestepping::NewtonSolver newton(nopt);
    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    ASSERT_TRUE(ws.isAllocated());

    auto& J = *ws.jacobian;
    auto& r = *ws.residual;
    auto& du = *ws.delta;

    // Assemble at u=u_prev.
    svmp::FE::systems::SystemStateView state;
    state.time = dt;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = history.uSpan();
    state.u_prev = history.uPrevSpan();
    state.u_prev2 = history.uPrev2Span();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    auto J_view = J.createAssemblyView();
    auto r_view = r.createAssemblyView();
    ASSERT_NE(J_view.get(), nullptr);
    ASSERT_NE(r_view.get(), nullptr);
    (void)transient.assemble(req, state, J_view.get(), r_view.get());
    const double r0 = r.norm();
    ASSERT_GT(r0, 0.0);

    du.zero();
    const auto linrep = linear->solve(J, du, r);
    ASSERT_TRUE(linrep.converged);

    // u <- u - du
    {
        auto us = history.u().localSpan();
        auto dus = du.localSpan();
        ASSERT_EQ(us.size(), dus.size());
        for (std::size_t i = 0; i < us.size(); ++i) {
            us[i] -= dus[i];
        }
    }

    // Reassemble residual at updated u.
    auto r_view2 = r.createAssemblyView();
    ASSERT_NE(r_view2.get(), nullptr);
    req.want_matrix = false;
    req.want_vector = true;
    (void)transient.assemble(req, state, nullptr, r_view2.get());
    const double r1 = r.norm();
    EXPECT_LT(r1, 1e-12);
}

TEST(NewtonSolverSanity, SolveStepConvergesForLinearReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr double dt = 0.2;
    constexpr double lambda = 1.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u0);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 8;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    ASSERT_TRUE(ws.isAllocated());

    // Match TimeLoop: ensure history vectors use backend layout created after workspace.
    history.repack(*factory);

    const auto rep = newton.solveStep(transient, *linear, dt, history, ws);
    EXPECT_TRUE(rep.converged);
    EXPECT_LE(rep.iterations, 2);
}

TEST(NewtonSolverSanity, TraceInequalityBoundarySwitchesFromActiveToInactive)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif

    constexpr int marker = 2;
    constexpr double dt = 0.1;
    constexpr double target_value = -0.1;
    constexpr double penalty = 40.0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    svmp::FE::forms::FormCompiler compiler;
    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");

    const auto cell_form = ((u - svmp::FE::forms::FormExpr::constant(static_cast<Real>(target_value))) * v).dx();
    auto cell_ir = compiler.compileResidual(cell_form);
    auto cell_kernel =
        std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(cell_ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, cell_kernel);

    auto boundary_form = (svmp::FE::forms::FormExpr::constant(0.0) * u * v).dx();
    svmp::FE::forms::bc::TraceInequalityOptions opts;
    opts.trace_operator = svmp::FE::forms::bc::ScalarTraceOperator::Identity;
    opts.sense = svmp::FE::forms::bc::TraceInequalitySense::LessEqual;
    boundary_form = svmp::FE::forms::bc::applyTraceInequality(std::move(boundary_form),
                                                              u,
                                                              v,
                                                              marker,
                                                              svmp::FE::forms::FormExpr::constant(0.0),
                                                              svmp::FE::forms::FormExpr::constant(
                                                                  static_cast<Real>(penalty)),
                                                              opts);
    auto boundary_ir = compiler.compileResidual(boundary_form);
    auto boundary_kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
        std::move(boundary_ir), svmp::FE::forms::ADMode::Forward);
    sys.addBoundaryKernel("op", marker, u_field, u_field, boundary_kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, sys.dofHandler().getNumDofs());
    history.setDt(dt);
    history.setPrevDt(dt);

    const std::vector<Real> initial(sys.dofHandler().getNumDofs(), static_cast<Real>(0.3));
    setVectorByDof(history.uPrev(), initial);
    setVectorByDof(history.uPrev2(), initial);
    history.resetCurrentToPrevious();

    svmp::FE::timestepping::NewtonOptions nopt;
    nopt.residual_op = "op";
    nopt.jacobian_op = "op";
    nopt.max_iterations = 8;
    nopt.abs_tolerance = 1e-12;
    nopt.rel_tolerance = 0.0;
    svmp::FE::timestepping::NewtonSolver newton(nopt);

    svmp::FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(sys, *factory, ws);
    ASSERT_TRUE(ws.isAllocated());

    history.repack(*factory);

    const auto rep = newton.solveStep(transient, *linear, dt, history, ws);
    EXPECT_TRUE(rep.converged);
    EXPECT_GE(rep.iterations, 2);
    EXPECT_LT(rep.residual_norm, 1e-12);

    const auto solved = getVectorByDof(history.u());
    ASSERT_FALSE(solved.empty());
    for (const Real value : solved) {
        EXPECT_LT(value, static_cast<Real>(0.0));
        EXPECT_NEAR(value, static_cast<Real>(target_value), 1e-12);
    }
}

TEST(TimeLoopConvergence, BackwardEuler_IsFirstOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale1 = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale1));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopSanity, BackwardEuler_SingleStep_AdvancesSolution)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double dt = 0.2;

    const auto u_dt = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, dt, lambda);
    ASSERT_EQ(u_dt.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const double scale = 1.0 / (1.0 + lambda * dt);
    for (std::size_t i = 0; i < u_dt.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(u_dt[i]), static_cast<double>(u0[i]) * scale, 1e-12);
    }
}

TEST(TimeLoopCallbacks, BeforeStepAcceptRejectsConvergedCandidateAndRetries)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;
    using svmp::FE::timestepping::StepRejectReason;

    SimpleStepControllerOptions controller_options;
    controller_options.decrease_factor = 0.5;
    controller_options.increase_factor = 1.0;
    controller_options.max_retries = 3;
    auto controller =
        std::make_shared<SimpleStepController>(controller_options);

    int candidate_calls = 0;
    int discarded_candidate_calls = 0;
    int commit_ready_calls = 0;
    int rejected_calls = 0;
    int accepted_calls = 0;
    std::vector<double> accepted_times;
    std::vector<double> retry_dt_updates;

    const auto final_values = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::BackwardEuler,
        /*dt=*/0.2,
        /*t_end=*/0.2,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        controller,
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory&) {
            callbacks.on_before_step_accept =
                [&](svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport& nr) {
                    ++candidate_calls;
                    EXPECT_TRUE(nr.converged);
                    if (candidate_calls == 1) {
                        EXPECT_EQ(h.stepIndex(), 0);
                        EXPECT_NEAR(h.time(), 0.0, 1e-15);
                        EXPECT_NEAR(h.dt(), 0.2, 1e-15);
                        return false;
                    }
                    return true;
                };
            callbacks.on_step_candidate_discarded =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++discarded_candidate_calls;
                    EXPECT_EQ(candidate_calls, 1);
                    EXPECT_EQ(h.stepIndex(), 0);
                };
            callbacks.on_step_commit_ready =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++commit_ready_calls;
                    EXPECT_EQ(h.stepIndex(), accepted_calls);
                };
            callbacks.on_step_rejected =
                [&](const svmp::FE::timestepping::TimeHistory& h,
                    StepRejectReason reason,
                    const svmp::FE::timestepping::NewtonReport& nr) {
                    ++rejected_calls;
                    EXPECT_EQ(reason, StepRejectReason::ErrorTooLarge);
                    EXPECT_TRUE(nr.converged);
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1e-15);
                    EXPECT_NEAR(h.dt(), 0.2, 1e-15);
                };
            callbacks.on_step_accepted =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++accepted_calls;
                    accepted_times.push_back(h.time());
                };
            callbacks.on_dt_updated =
                [&](double old_dt, double new_dt, int step_index, int attempt_index) {
                    if (step_index == 0 && attempt_index == 0) {
                        retry_dt_updates.push_back(old_dt);
                        retry_dt_updates.push_back(new_dt);
                    }
                };
        });

    ASSERT_EQ(final_values.size(), 4u);
    EXPECT_EQ(candidate_calls, 3);
    EXPECT_EQ(discarded_candidate_calls, 1);
    EXPECT_EQ(commit_ready_calls, 2);
    EXPECT_EQ(rejected_calls, 1);
    EXPECT_EQ(accepted_calls, 2);
    ASSERT_EQ(accepted_times.size(), 2u);
    EXPECT_NEAR(accepted_times[0], 0.1, 1e-15);
    EXPECT_NEAR(accepted_times[1], 0.2, 1e-15);
    ASSERT_EQ(retry_dt_updates.size(), 2u);
    EXPECT_NEAR(retry_dt_updates[0], 0.2, 1e-15);
    EXPECT_NEAR(retry_dt_updates[1], 0.1, 1e-15);
}

TEST(TimeLoopCallbacks, GeneralizedAlphaRejectedCandidateRestoresExistingRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;

    SimpleStepControllerOptions controller_options;
    controller_options.decrease_factor = 0.5;
    controller_options.increase_factor = 1.0;
    controller_options.max_retries = 2;
    auto controller = std::make_shared<SimpleStepController>(controller_options);

    const std::vector<Real> initial_u_dot = {0.25, -0.5, 0.75, -1.0};
    const std::vector<Real> initial_u_ddot = {-1.25, 1.5, -1.75, 2.0};
    auto initial_rate_factory = ts_test::createTestFactory();
    ASSERT_NE(initial_rate_factory.get(), nullptr);
    int step_start_calls = 0;
    int candidate_calls = 0;
    bool rejected_candidate_changed_rate = false;
    double generated_state_value = 1.0;
    int restored_state_callbacks = 0;
    int restored_projected_state_callbacks = 0;
    int endpoint_state_callbacks = 0;
    int projected_endpoint_state_callbacks = 0;

    const auto final_values = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.2,
        /*t_end=*/0.2,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        controller,
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            history.ensureSecondOrderState(*initial_rate_factory);
            setVectorByDof(history.uDot(), initial_u_dot);
            setVectorByDof(history.uDDot(), initial_u_ddot);

            callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
                ++step_start_calls;
                if (step_start_calls == 2) {
                    EXPECT_EQ(
                        getVectorByDof(const_cast<svmp::FE::backends::GenericVector&>(h.uDot())),
                        initial_u_dot);
                    EXPECT_EQ(
                        getVectorByDof(const_cast<svmp::FE::backends::GenericVector&>(h.uDDot())),
                        initial_u_ddot);
                    EXPECT_NEAR(generated_state_value, 1.0, 1.0e-15)
                        << "Rejected stage-generated state was not rolled back "
                           "before the retry";
                }
            };
            callbacks.on_before_step_accept =
                [&](svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport&) {
                    ++candidate_calls;
                    if (candidate_calls == 1) {
                        rejected_candidate_changed_rate =
                            getVectorByDof(h.uDot()) != initial_u_dot;
                        return false;
                    }
                    return true;
                };
        },
        /*inspect_expected_exception=*/{},
        [&](svmp::FE::timestepping::TimeLoopOptions& options,
            svmp::FE::FieldId) {
            using StateSyncPoint = svmp::FE::timestepping::NewtonOptions::
                StateSynchronizationPoint;
            options.newton.synchronize_state =
                [&](const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    ASSERT_FALSE(state.u.empty());
                    generated_state_value = static_cast<double>(state.u[0]);
                    if (point == StateSyncPoint::RestoredTimeStepState) {
                        ++restored_state_callbacks;
                        EXPECT_NEAR(state.time, 0.0, 1.0e-15);
                        EXPECT_NEAR(generated_state_value, 1.0, 1.0e-15);
                    } else if (
                        point ==
                        StateSyncPoint::RestoredProjectedTimeStepState) {
                        ++restored_projected_state_callbacks;
                        EXPECT_NEAR(state.time, 0.0, 1.0e-15);
                        EXPECT_NEAR(generated_state_value, 1.0, 1.0e-15);
                    } else if (point ==
                               StateSyncPoint::EndpointCandidateState) {
                        const double expected_time =
                            0.1 * static_cast<double>(
                                      endpoint_state_callbacks + 1);
                        ++endpoint_state_callbacks;
                        EXPECT_NEAR(state.time, expected_time, 1.0e-15);
                    } else if (
                        point ==
                        StateSyncPoint::ProjectedEndpointCandidateState) {
                        const double expected_time =
                            0.1 * static_cast<double>(
                                      projected_endpoint_state_callbacks + 1);
                        ++projected_endpoint_state_callbacks;
                        EXPECT_NEAR(state.time, expected_time, 1.0e-15);
                    }
                };
        });

    ASSERT_EQ(final_values.size(), 4u);
    EXPECT_TRUE(rejected_candidate_changed_rate);
    EXPECT_GE(step_start_calls, 2);
    EXPECT_EQ(restored_state_callbacks, 1);
    EXPECT_EQ(restored_projected_state_callbacks, 1);
    EXPECT_EQ(endpoint_state_callbacks, 2);
    EXPECT_EQ(projected_endpoint_state_callbacks, 2);
}

TEST(TimeLoopCallbacks, GeneralizedAlphaRetryMatchesDirectAttemptAndRestoresMissingRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;

    const auto direct = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.2,
        /*lambda=*/1.0);

    SimpleStepControllerOptions controller_options;
    controller_options.decrease_factor = 0.5;
    controller_options.increase_factor = 1.0;
    controller_options.max_retries = 2;
    auto controller = std::make_shared<SimpleStepController>(controller_options);

    int step_start_calls = 0;
    int candidate_calls = 0;
    bool retry_started_without_rate_state = false;
    const auto retried = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.2,
        /*t_end=*/0.2,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        controller,
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory&) {
            callbacks.on_step_start = [&](const svmp::FE::timestepping::TimeHistory& h) {
                ++step_start_calls;
                if (step_start_calls == 2) {
                    retry_started_without_rate_state =
                        !h.hasUDotState() && !h.hasUDDotState();
                }
            };
            callbacks.on_before_step_accept =
                [&](svmp::FE::timestepping::TimeHistory&,
                    const svmp::FE::timestepping::NewtonReport&) {
                    ++candidate_calls;
                    return candidate_calls != 1;
                };
        });

    ASSERT_EQ(direct.size(), retried.size());
    EXPECT_TRUE(retry_started_without_rate_state);
    for (std::size_t i = 0; i < direct.size(); ++i) {
        EXPECT_NEAR(retried[i], direct[i], 1e-13) << "DOF " << i;
    }
}

TEST(TimeLoopCallbacks,
     GeneralizedAlphaCandidateCallbackExceptionRestoresGeneratedAndRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using StateSyncPoint = svmp::FE::timestepping::NewtonOptions::
        StateSynchronizationPoint;

    const std::vector<Real> initial_u_dot = {0.25, -0.5, 0.75, -1.0};
    auto rate_factory = ts_test::createTestFactory();
    ASSERT_NE(rate_factory.get(), nullptr);

    double generated_state_value = 1.0;
    int restored_callbacks = 0;
    int restored_projected_callbacks = 0;
    int endpoint_callbacks = 0;
    int projected_endpoint_callbacks = 0;
    bool inspected_exception = false;

    const auto result = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/0.5,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [rate_factory_ptr = rate_factory.get(), initial_u_dot](
            svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            ASSERT_NE(rate_factory_ptr, nullptr);
            history.ensureSecondOrderState(*rate_factory_ptr);
            setVectorByDof(history.uDot(), initial_u_dot);
            history.uDDot().zero();
            callbacks.on_before_step_accept =
                [](svmp::FE::timestepping::TimeHistory&,
                   const svmp::FE::timestepping::NewtonReport&) -> bool {
                FE_THROW(svmp::FE::FEException,
                         "expected candidate callback failure");
            };
        },
        [&](const svmp::FE::timestepping::TimeHistory& history,
            const svmp::FE::FEException& exception) {
            inspected_exception = true;
            EXPECT_NE(std::string(exception.what()).find(
                          "expected candidate callback failure"),
                      std::string::npos);
            EXPECT_NEAR(generated_state_value, 1.0, 1.0e-15);
            EXPECT_EQ(history.stepIndex(), 0);
            EXPECT_NEAR(history.time(), 0.0, 1.0e-15);
            EXPECT_EQ(getVectorByDof(const_cast<
                          svmp::FE::backends::GenericVector&>(history.uDot())),
                      initial_u_dot);
            const auto committed = getVectorByDof(const_cast<
                svmp::FE::backends::GenericVector&>(history.uPrev()));
            const std::vector<Real> initial_solution = {
                1.0, -0.5, 0.25, 2.0};
            ASSERT_EQ(committed.size(), initial_solution.size());
            for (std::size_t i = 0; i < committed.size(); ++i) {
                EXPECT_NEAR(committed[i], initial_solution[i], 1.0e-15)
                    << "DOF " << i;
            }
        },
        [&](svmp::FE::timestepping::TimeLoopOptions& options,
            svmp::FE::FieldId) {
            options.initialize_first_order_rate_from_pde = false;
            options.newton.synchronize_state =
                [&](const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    ASSERT_FALSE(state.u.empty());
                    generated_state_value = static_cast<double>(state.u[0]);
                    if (point == StateSyncPoint::RestoredTimeStepState) {
                        ++restored_callbacks;
                        EXPECT_NEAR(state.time, 0.0, 1.0e-15);
                    } else if (
                        point ==
                        StateSyncPoint::RestoredProjectedTimeStepState) {
                        ++restored_projected_callbacks;
                        EXPECT_NEAR(state.time, 0.0, 1.0e-15);
                    } else if (point ==
                               StateSyncPoint::EndpointCandidateState) {
                        ++endpoint_callbacks;
                    } else if (
                        point ==
                        StateSyncPoint::ProjectedEndpointCandidateState) {
                        ++projected_endpoint_callbacks;
                    }
                };
        });

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(inspected_exception);
    EXPECT_EQ(restored_callbacks, 1);
    EXPECT_EQ(restored_projected_callbacks, 1);
    EXPECT_EQ(endpoint_callbacks, 0);
    EXPECT_EQ(projected_endpoint_callbacks, 0);
}

TEST(TimeLoopCallbacks, AcceptedCallbackExceptionDoesNotRollbackCommittedRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const std::vector<Real> initial_u_dot = {0.25, -0.5, 0.75, -1.0};
    const std::vector<Real> initial_u_ddot = {-1.25, 1.5, -1.75, 2.0};
    auto initial_rate_factory = ts_test::createTestFactory();
    ASSERT_NE(initial_rate_factory.get(), nullptr);

    std::vector<Real> accepted_u_dot;
    bool inspected_after_unwind = false;
    const auto result = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            history.ensureSecondOrderState(*initial_rate_factory);
            setVectorByDof(history.uDot(), initial_u_dot);
            setVectorByDof(history.uDDot(), initial_u_ddot);

            callbacks.on_before_step_accept =
                [&](svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport&) {
                    accepted_u_dot = getVectorByDof(h.uDot());
                    EXPECT_NE(accepted_u_dot, initial_u_dot);
                    return true;
                };
            callbacks.on_step_accepted =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    EXPECT_EQ(h.stepIndex(), 1);
                    EXPECT_NEAR(h.time(), 0.1, 1e-15);
                    FE_THROW(svmp::FE::FEException,
                             "expected accepted-callback exception");
                };
        },
        [&](const svmp::FE::timestepping::TimeHistory& h,
            const svmp::FE::FEException& e) {
            inspected_after_unwind = true;
            EXPECT_NE(std::string(e.what()).find("expected accepted-callback exception"),
                      std::string::npos);
            EXPECT_EQ(h.stepIndex(), 1);
            EXPECT_NEAR(h.time(), 0.1, 1e-15);
            EXPECT_EQ(getVectorByDof(
                          const_cast<svmp::FE::backends::GenericVector&>(h.uDot())),
                      accepted_u_dot);
            EXPECT_NE(getVectorByDof(
                          const_cast<svmp::FE::backends::GenericVector&>(h.uDot())),
                      initial_u_dot);
        });

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(inspected_after_unwind);
}

TEST(TimeLoopCallbacks,
     CommitReadyFailureWithSuccessfulDiscardRestoresRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const std::vector<Real> initial_u_dot = {0.25, -0.5, 0.75, -1.0};
    auto rate_factory = ts_test::createTestFactory();
    ASSERT_NE(rate_factory.get(), nullptr);

    std::vector<Real> candidate_u_dot;
    int commit_ready_calls = 0;
    int discard_calls = 0;
    bool inspected_after_unwind = false;
    const auto result = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            history.ensureSecondOrderState(*rate_factory);
            setVectorByDof(history.uDot(), initial_u_dot);
            history.uDDot().zero();
            callbacks.on_before_step_accept =
                [](svmp::FE::timestepping::TimeHistory&,
                   const svmp::FE::timestepping::NewtonReport&) {
                    return true;
                };
            callbacks.on_step_commit_ready =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++commit_ready_calls;
                    candidate_u_dot = getVectorByDof(h.uDot());
                    EXPECT_NE(candidate_u_dot, initial_u_dot);
                    FE_THROW(svmp::FE::FEException,
                             "expected commit-ready failure");
                };
            callbacks.on_step_candidate_discarded =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++discard_calls;
                    EXPECT_EQ(getVectorByDof(h.uDot()), candidate_u_dot);
                };
        },
        [&](const svmp::FE::timestepping::TimeHistory& history,
            const svmp::FE::FEException& exception) {
            inspected_after_unwind = true;
            EXPECT_NE(std::string(exception.what()).find(
                          "expected commit-ready failure"),
                      std::string::npos);
            EXPECT_EQ(getVectorByDof(const_cast<
                          svmp::FE::backends::GenericVector&>(history.uDot())),
                      initial_u_dot);
        });

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(inspected_after_unwind);
    EXPECT_EQ(commit_ready_calls, 1);
    EXPECT_EQ(discard_calls, 1);
}

TEST(TimeLoopCallbacks,
     CommitReadyFailureWithFailStopDiscardRetainsCandidateRateState)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const std::vector<Real> initial_u_dot = {0.25, -0.5, 0.75, -1.0};
    auto rate_factory = ts_test::createTestFactory();
    ASSERT_NE(rate_factory.get(), nullptr);

    std::vector<Real> candidate_u_dot;
    int commit_ready_calls = 0;
    int discard_calls = 0;
    bool inspected_after_unwind = false;
    const auto result = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            history.ensureSecondOrderState(*rate_factory);
            setVectorByDof(history.uDot(), initial_u_dot);
            history.uDDot().zero();
            callbacks.on_before_step_accept =
                [](svmp::FE::timestepping::TimeHistory&,
                   const svmp::FE::timestepping::NewtonReport&) {
                    return true;
                };
            callbacks.on_step_commit_ready =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++commit_ready_calls;
                    candidate_u_dot = getVectorByDof(h.uDot());
                    EXPECT_NE(candidate_u_dot, initial_u_dot);
                    FE_THROW(svmp::FE::FEException,
                             "expected commit-ready publication failure");
                };
            callbacks.on_step_candidate_discarded =
                [&](svmp::FE::timestepping::TimeHistory& h) {
                    ++discard_calls;
                    EXPECT_EQ(getVectorByDof(h.uDot()), candidate_u_dot);
                    FE_THROW(svmp::FE::FEException,
                             "expected fail-stop discard failure");
                };
        },
        [&](const svmp::FE::timestepping::TimeHistory& history,
            const svmp::FE::FEException& exception) {
            inspected_after_unwind = true;
            EXPECT_NE(std::string(exception.what()).find(
                          "expected fail-stop discard failure"),
                      std::string::npos);
            EXPECT_EQ(getVectorByDof(const_cast<
                          svmp::FE::backends::GenericVector&>(history.uDot())),
                      candidate_u_dot);
            EXPECT_NE(candidate_u_dot, initial_u_dot);
        });

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(inspected_after_unwind);
    EXPECT_EQ(commit_ready_calls, 1);
    EXPECT_EQ(discard_calls, 1);
}

TEST(TimeLoopConvergence, Bdf2_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BDF2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::BDF2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, CoupledTwoFieldReaction_ThetaMethod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda_u = 1.0;
    const double lambda_w = 2.0;
    const double kappa = 0.5;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto x_dt1 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                                         dt1,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/false);
    const auto x_dt2 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                                         dt2,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/false);
    ASSERT_EQ(x_dt1.size(), 10u);
    ASSERT_EQ(x_dt2.size(), 10u);

    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};
    std::vector<Real> exact(10u, 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        const auto [ui, wi] = coupledReactionExact2x2(t_end, u0[i], w0[i], lambda_u, lambda_w, kappa);
        exact[i] = static_cast<Real>(ui);
        exact[5u + i] = static_cast<Real>(wi);
    }

    const double e1 = relativeL2Error(x_dt1, exact);
    const double e2 = relativeL2Error(x_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, CoupledTwoFieldReaction_Bdf2_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda_u = 1.0;
    const double lambda_w = 2.0;
    const double kappa = 0.5;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto x_dt1 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::BDF2,
                                                         dt1,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/true);
    const auto x_dt2 = runCoupledReactionTwoFieldProblem(svmp::FE::timestepping::SchemeKind::BDF2,
                                                         dt2,
                                                         t_end,
                                                         lambda_u,
                                                         lambda_w,
                                                         kappa,
                                                         /*theta=*/0.5,
                                                         /*history_depth=*/2,
                                                         /*exact_initial_history=*/true);
    ASSERT_EQ(x_dt1.size(), 10u);
    ASSERT_EQ(x_dt2.size(), 10u);

    const std::vector<double> u0 = {1.0, -0.5, 0.25, 2.0, -1.5};
    const std::vector<double> w0 = {0.5, -0.25, 0.75, -1.0, 0.1};
    std::vector<Real> exact(10u, 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        const auto [ui, wi] = coupledReactionExact2x2(t_end, u0[i], w0[i], lambda_u, lambda_w, kappa);
        exact[i] = static_cast<Real>(ui);
        exact[5u + i] = static_cast<Real>(wi);
    }

    const double e1 = relativeL2Error(x_dt1, exact);
    const double e2 = relativeL2Error(x_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder3_IsThirdOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 3;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.4);
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder4_IsFourthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 4;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, VSVO_BDF_FixedOrder5_IsFifthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const int order = 5;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt1, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt1),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF, dt2, t_end, lambda,
                                          /*history_depth=*/order + 1,
                                          /*controller=*/makeFixedVsvoBdfController(order, dt2),
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/true);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 4.0);
}

TEST(TimeLoopConvergence, ThetaMethod_CrankNicolson_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, ThetaMethod_Theta075_IsFirstOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const double theta = 0.75;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/false,
                                          /*theta=*/theta);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/2,
                                          svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                          /*collocation_max_outer_iterations=*/4,
                                          /*collocation_outer_tolerance=*/0.0,
                                          /*exact_initial_history=*/false,
                                          /*theta=*/theta);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopConvergence, ThetaMethod_CrankNicolson_IsSecondOrder_ForLogisticEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double r = 1.0;
    const double K = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;
    const double dt3 = 0.05;

    const auto u_dt1 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt1, t_end, r, K);
    const auto u_dt2 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt2, t_end, r, K);
    const auto u_dt3 = runLogisticProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt3, t_end, r, K);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);
    ASSERT_EQ(u_dt3.size(), 4u);

    auto l2Diff = [](const std::vector<Real>& a, const std::vector<Real>& b) -> double {
        EXPECT_EQ(a.size(), b.size());
        double sum = 0.0;
        const std::size_t n = std::min(a.size(), b.size());
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i] - b[i]);
            sum += d * d;
        }
        return std::sqrt(sum);
    };

    // Self-convergence against successive refinements avoids needing an analytic
    // solution for the coupled FE semi-discretization.
    const double e12 = l2Diff(u_dt1, u_dt2);
    const double e23 = l2Diff(u_dt2, u_dt3);
    const double p = std::log(e12 / e23) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e12=" << e12 << " e23=" << e23;
}

TEST(TimeLoopConvergence, Trbdf2_IsSecondOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::TRBDF2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::TRBDF2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, DG1_IsThirdOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG1, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG1, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.6);
}

TEST(TimeLoopConvergence, DG_Degree2_IsFifthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/2);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/2);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 4.0);
}

TEST(TimeLoopConvergence, DG_Degree3_IsSeventhOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 6.0) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, CG2_IsFourthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG2, dt1, t_end, lambda);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG2, dt2, t_end, lambda);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, CG_Degree3_IsSixthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.1;
    const double dt2 = 0.05;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 5.0);
}

TEST(TimeLoopConvergence, CG_Degree4_IsEighthOrder_ForReactionEquation)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.5;
    const double dt2 = 0.25;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt1, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/4);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG, dt2, t_end, lambda,
                                          /*history_depth=*/2,
                                          /*controller=*/{},
                                          /*generalized_alpha_rho_inf=*/1.0,
                                          /*dg_degree=*/1,
                                          /*cg_degree=*/4);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 7.0) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopVerification, ManufacturedHeatSinForcing_ConvergesWithBackwardEuler)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runHeatManufacturedSinForcing(dt1, t_end);
    const auto u_dt2 = runHeatManufacturedSinForcing(dt2, t_end);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const Real exact_value = static_cast<Real>(1.0 - std::cos(t_end));
    std::vector<Real> exact(u_dt1.size(), exact_value);

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopVerification, Dt2Oscillator_BackwardDifference_IsFirstOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2(dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2(dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 0.8);
    EXPECT_LT(p, 1.2);
}

TEST(TimeLoopConvergence, Dt2Oscillator_Newmark_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_Newmark_LongTimePeriod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double pi = std::acos(-1.0);
    const double omega = 2.0 * pi;
    const double exact_period = 2.0 * pi / omega;
    const int cycles = 20;
    const double t_end = static_cast<double>(cycles) * exact_period;

    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const auto traj1 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega);
    const auto traj2 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega);

    const auto T1 = estimatePeriodFromDownwardZeroCrossings(traj1);
    const auto T2 = estimatePeriodFromDownwardZeroCrossings(traj2);
    ASSERT_TRUE(T1.has_value());
    ASSERT_TRUE(T2.has_value());

    const double e1 = std::abs(*T1 - exact_period);
    const double e2 = std::abs(*T2 - exact_period);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "T1=" << *T1 << " T2=" << *T2 << " e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2DampedOscillator_Newmark_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double zeta = 0.1;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt1, t_end, omega, zeta);
    const auto u_dt2 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::Newmark, dt2, t_end, omega, zeta);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double exp_decay = std::exp(-zeta * omega * t_end);
    const double c = std::cos(omega_d * t_end);
    const double s = std::sin(omega_d * t_end);

    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * exp_decay * (c + alpha * s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, omega,
                                                  /*generalized_alpha_rho_inf=*/1.0);
    const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, omega,
                                                  /*generalized_alpha_rho_inf=*/1.0);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_LongTimePeriod_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double pi = std::acos(-1.0);
    const double omega = 2.0 * pi;
    const double exact_period = 2.0 * pi / omega;
    const int cycles = 20;
    const double t_end = static_cast<double>(cycles) * exact_period;

    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const auto traj1 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                            dt1,
                                                            t_end,
                                                            omega,
                                                            /*generalized_alpha_rho_inf=*/1.0);
    const auto traj2 = runOscillatorDt2StructuralTrajectory(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                            dt2,
                                                            t_end,
                                                            omega,
                                                            /*generalized_alpha_rho_inf=*/1.0);

    const auto T1 = estimatePeriodFromDownwardZeroCrossings(traj1);
    const auto T2 = estimatePeriodFromDownwardZeroCrossings(traj2);
    ASSERT_TRUE(T1.has_value());
    ASSERT_TRUE(T2.has_value());

    const double e1 = std::abs(*T1 - exact_period);
    const double e2 = std::abs(*T2 - exact_period);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "T1=" << *T1 << " T2=" << *T2 << " e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2DampedOscillator_GeneralizedAlphaSecondOrder_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double zeta = 0.1;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                        dt1,
                                                        t_end,
                                                        omega,
                                                        zeta,
                                                        /*generalized_alpha_rho_inf=*/1.0);
    const auto u_dt2 = runDampedOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
                                                        dt2,
                                                        t_end,
                                                        omega,
                                                        zeta,
                                                        /*generalized_alpha_rho_inf=*/1.0);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double omega_d = omega * std::sqrt(1.0 - zeta * zeta);
    const double alpha = zeta / std::sqrt(1.0 - zeta * zeta);
    const double exp_decay = std::exp(-zeta * omega * t_end);
    const double c = std::cos(omega_d * t_end);
    const double s = std::sin(omega_d * t_end);

    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * exp_decay * (c + alpha * s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6) << "e1=" << e1 << " e2=" << e2;
}

TEST(TimeLoopConvergence, Dt2Oscillator_GeneralizedAlphaSecondOrder_RhoInfSweep_IsSecondOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const std::vector<double> rho_values = {0.0, 0.2, 0.5, 0.9, 1.0};
    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const Real scale = static_cast<Real>(std::cos(omega * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    for (double rho_inf : rho_values) {
        const auto u_dt1 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, omega,
                                                      /*generalized_alpha_rho_inf=*/rho_inf);
        const auto u_dt2 = runOscillatorDt2Structural(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, omega,
                                                      /*generalized_alpha_rho_inf=*/rho_inf);
        ASSERT_EQ(u_dt1.size(), 4u);
        ASSERT_EQ(u_dt2.size(), 4u);

        const double e1 = relativeL2Error(u_dt1, exact);
        const double e2 = relativeL2Error(u_dt2, exact);
        const double p = std::log(e1 / e2) / std::log(2.0);
        EXPECT_GT(p, 1.6) << "rho_inf=" << rho_inf << " e1=" << e1 << " e2=" << e2;
    }
}

TEST(TimeLoopConvergence, Dt2Oscillator_DG1_IsThirdOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::DG1, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::DG1, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double c = std::cos(omega * t_end);
    const double s = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 2.6);
}

TEST(TimeLoopConvergence, Dt2Oscillator_CG2_IsFourthOrder)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 1.0;
    const double t_end = 1.0;
    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::CG2, dt1, t_end, omega);
    const auto u_dt2 = runOscillatorDt2Collocation(svmp::FE::timestepping::SchemeKind::CG2, dt2, t_end, omega);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    const double c = std::cos(omega * t_end);
    const double s = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 3.2);
}

TEST(TimeLoopConvergence, GeneralizedAlpha_FirstOrder_IsSecondOrderOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    const double dt1 = 0.2;
    const double dt2 = 0.1;

    const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, lambda,
                                          /*history_depth=*/3);
    const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, lambda,
                                          /*history_depth=*/3);
    ASSERT_EQ(u_dt1.size(), 4u);
    ASSERT_EQ(u_dt2.size(), 4u);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double e1 = relativeL2Error(u_dt1, exact);
    const double e2 = relativeL2Error(u_dt2, exact);
    const double p = std::log(e1 / e2) / std::log(2.0);
    EXPECT_GT(p, 1.6);
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaPdeRateInitializationRegularizesExactZeroMassRow)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<TwoTetraCellDomainMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    // Cell 0 owns the differential equation.  Cell 1 contributes only the
    // reaction term, leaving its unique vertex (DOF 4) as an exact zero row
    // in the dt-only startup operator while keeping the full stage system
    // nonsingular.  This is the algebraic shape of a field-wide dt scan over
    // a cut-domain mass operator.
    const auto differential_weight =
        svmp::FE::forms::FormExpr::constant(1.0) -
        svmp::FE::forms::FormExpr::cellDomainId();
    const auto form =
        (svmp::FE::forms::dt(u) * v * differential_weight + u * v).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
        std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);
    RateInitializationSlotObservations slot_observations;
    auto slot_recording_kernel =
        std::make_shared<RateInitializationSlotRecordingKernel>(
            slot_observations);
    sys.addCellKernel("op", u_field, u_field, slot_recording_kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = twoTetraSharedFaceTopology();
    sys.setup({}, inputs);
    ASSERT_TRUE(sys.isSetup());
    ASSERT_EQ(sys.dofHandler().getNumDofs(), 5);

    auto integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *factory, sys.dofHandler().getNumDofs(), /*history_depth=*/3);
    const std::vector<Real> initial{1.0, -0.5, 0.25, 2.0, -1.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(0.1);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.1;
    opts.dt = 0.1;
    opts.max_steps = 1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = true;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopReport report;
    std::string exception_message;
    testing::internal::CaptureStdout();
    try {
        report = loop.run(transient, *factory, *linear, history);
    } catch (const std::exception& error) {
        exception_message = error.what();
    }
    const auto output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(exception_message.empty()) << exception_message;
    EXPECT_TRUE(report.success);
    EXPECT_NE(output.find(
                  "diagnostic=timeloop_first_order_rate_initialization"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find("accepted=1"), std::string::npos) << output;
    EXPECT_NE(output.find(
                  "exact_zero_mass_owned_rows_regularized=1"),
              std::string::npos)
        << output;
    EXPECT_TRUE(slot_observations.saw_non_dt_residual);
    EXPECT_TRUE(slot_observations.saw_dt_only_jacobian);
    // history_rate_order==0 defines slot 2 as the injected uDot^n.  Startup
    // zeroes that rate before the PDE initialization; the nonzero displacement
    // history in uPrev2 must not leak into either initialization assembly.
    EXPECT_DOUBLE_EQ(slot_observations.max_abs_slot2_value, 0.0);
    ASSERT_TRUE(history.hasUDotState());
    EXPECT_TRUE(std::isfinite(static_cast<double>(history.uDot().norm())));
    EXPECT_GT(history.uDot().norm(), 0.0);
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaPdeRateInitializationPreservesZeroDiagonalCrossFieldMassRows)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "u", .space = space, .components = 1});
    const auto w_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "w", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::stateField(
        u_field, *space, "u");
    const auto w = svmp::FE::forms::FormExpr::stateField(
        w_field, *space, "w");
    const auto v = svmp::FE::forms::FormExpr::testFunction(
        u_field, *space, "v");
    const auto q = svmp::FE::forms::FormExpr::testFunction(
        w_field, *space, "q");

    // The differential operator is [0 M; M 0]. Every diagonal entry is
    // exactly zero, but every row has a valid cross-field mass coupling.
    const auto residual =
        (svmp::FE::forms::dt(w) * v + u * v).dx() +
        (svmp::FE::forms::dt(u) * q + Real{2.0} * w * q).dx();
    svmp::FE::systems::FormInstallOptions install;
    install.compiler_options.use_symbolic_tangent = true;
    const auto installed = svmp::FE::systems::installFormulation(
        sys, "op", {u_field, w_field}, residual, install);
    ASSERT_NE(installed.mixed_plan, nullptr);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    ASSERT_EQ(sys.dofHandler().getNumDofs(), 8);

    const auto dt_fields = sys.timeDerivativeFields("op");
    EXPECT_NE(std::find(dt_fields.begin(), dt_fields.end(), u_field),
              dt_fields.end());
    EXPECT_NE(std::find(dt_fields.begin(), dt_fields.end(), w_field),
              dt_fields.end());

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    auto integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *factory, sys.dofHandler().getNumDofs(), /*history_depth=*/3);

    std::vector<Real> initial(8u, Real{0.0});
    const auto u_offset = sys.fieldDofOffset(u_field);
    const auto w_offset = sys.fieldDofOffset(w_field);
    for (GlobalIndex i = 0; i < 4; ++i) {
        initial[static_cast<std::size_t>(u_offset + i)] = Real{1.0};
        initial[static_cast<std::size_t>(w_offset + i)] = Real{0.25};
    }
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(0.05);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = 0.05;
    opts.dt = 0.05;
    opts.max_steps = 1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = true;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1.0e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopReport report;
    std::string exception_message;
    testing::internal::CaptureStdout();
    try {
        report = loop.run(transient, *factory, *linear, history);
    } catch (const std::exception& error) {
        exception_message = error.what();
    }
    const auto output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(exception_message.empty()) << exception_message;
    EXPECT_TRUE(report.success);
    EXPECT_NE(output.find(
                  "diagnostic=timeloop_first_order_rate_initialization"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find(
                  "exact_zero_mass_owned_rows_regularized=0"),
              std::string::npos)
        << output;
    ASSERT_TRUE(history.hasUDotState());
    EXPECT_TRUE(std::isfinite(static_cast<double>(history.uDot().norm())));
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaPdeRateInitializationEvaluatesExplicitForcingAtAcceptedTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double t0 = 2.0;
    constexpr double dt = 0.1;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(
        u_field, *space, "v");
    const auto form =
        (svmp::FE::forms::dt(u) * v -
         svmp::FE::forms::FormExpr::time() * v)
            .dx();
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
        std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    RateInitializationTimeObservations observations;
    sys.addCellKernel(
        "op",
        u_field,
        u_field,
        std::make_shared<RateInitializationTimeRecordingKernel>(observations));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);
    auto integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *factory, sys.dofHandler().getNumDofs(), /*history_depth=*/3);
    const std::vector<Real> initial(4u, Real{0.0});
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = t0;
    opts.t_end = t0 + dt;
    opts.dt = dt;
    opts.max_steps = 1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = true;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1.0e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history);
    ASSERT_TRUE(report.success);
    ASSERT_GE(observations.initialization_times.size(), 2u);
    for (const auto observed_time : observations.initialization_times) {
        EXPECT_NEAR(static_cast<double>(observed_time), t0, 1.0e-15);
    }
    ASSERT_TRUE(observations.recorded_stage_rate);
    ASSERT_EQ(observations.injected_stage_rates.size(), 4u);
    for (const auto rate : observations.injected_stage_rates) {
        EXPECT_NEAR(static_cast<double>(rate), t0, 1.0e-12);
    }
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaPdeRateInitializationRefreshesTimeDependentConstraintAtAcceptedTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double t0 = 1.5;
    constexpr double dt = 0.3;
    constexpr GlobalIndex constrained_dof = 0;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(
        u_field, *space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + u * v).dx();
    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(
        std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    RateInitializationTimeObservations observations;
    observations.constrained_dof = constrained_dof;
    sys.addCellKernel(
        "op",
        u_field,
        u_field,
        std::make_shared<RateInitializationTimeRecordingKernel>(observations));

    std::vector<GlobalIndex> constrained_dofs{constrained_dof};
    std::vector<std::array<double, 3>> coordinates{{{0.0, 0.0, 0.0}}};
    sys.addConstraint(
        std::make_unique<svmp::FE::constraints::DirichletBC>(
            std::move(constrained_dofs),
            std::move(coordinates),
            [](double, double, double, double time) {
                return 10.0 + time;
            },
            /*initial_time=*/t0));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    observations.constraints = &sys.constraints();

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);
    auto integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);
    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *factory, sys.dofHandler().getNumDofs(), /*history_depth=*/3);
    const std::vector<Real> initial(4u, Real{0.0});
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), initial);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = t0;
    opts.t_end = t0 + dt;
    opts.dt = dt;
    opts.max_steps = 1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = true;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1.0e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto report = loop.run(transient, *factory, *linear, history);
    ASSERT_TRUE(report.success);
    ASSERT_GE(observations.initialization_times.size(), 2u);
    ASSERT_EQ(observations.initialization_constraint_values.size(),
              observations.initialization_times.size());
    for (std::size_t i = 0; i < observations.initialization_times.size(); ++i) {
        EXPECT_NEAR(
            static_cast<double>(observations.initialization_times[i]),
            t0,
            1.0e-15);
        EXPECT_NEAR(
            static_cast<double>(
                observations.initialization_constraint_values[i]),
            10.0 + t0,
            1.0e-15);
    }
}

TEST(TimeLoopConvergence, GeneralizedAlphaFusedTransientFieldCommitsEndpointNotStage)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Tetra4, /*order=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto seed_transient = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "seed_transient", .space = space, .components = 1});
    const auto phi_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "phi", .space = space, .components = 1});
    const auto extension_field = sys.addField(svmp::FE::systems::FieldSpec{
        .name = "extension", .space = space, .components = 1});
    sys.addOperator("op");

    const auto seed = svmp::FE::forms::FormExpr::stateField(
        seed_transient, *space, "seed_transient");
    const auto seed_test = svmp::FE::forms::FormExpr::testFunction(
        seed_transient, *space, "seed_test");
    svmp::FE::systems::FormInstallOptions seed_install;
    seed_install.compiler_options.jit.enable = false;
    seed_install.compiler_options.use_symbolic_tangent = true;
    (void)svmp::FE::systems::installFormulation(
        sys,
        "op",
        {seed_transient},
        (svmp::FE::forms::dt(seed) * seed_test + seed * seed_test).dx(),
        seed_install);

    const auto phi = svmp::FE::forms::FormExpr::stateField(
        phi_field, *space, "phi");
    const auto extension = svmp::FE::forms::FormExpr::stateField(
        extension_field, *space, "extension");
    const auto eta = svmp::FE::forms::FormExpr::testFunction(
        phi_field, *space, "eta");
    const auto coupled_residual =
        (svmp::FE::forms::dt(phi) * eta + extension * eta).dx();

    svmp::FE::systems::FormInstallOptions coupled_install;
    coupled_install.compiler_options.jit.enable = true;
    coupled_install.compiler_options.use_symbolic_tangent = true;
    coupled_install.extra_trial_fields.push_back(extension_field);
    const auto installed = svmp::FE::systems::installFormulation(
        sys,
        "op",
        {phi_field},
        coupled_residual,
        coupled_install);
    ASSERT_NE(installed.mixed_plan, nullptr);
    ASSERT_TRUE(installed.mixed_plan->usesMonolithicCellKernel());

    const auto dt_fields = sys.timeDerivativeFields("op");
    ASSERT_EQ(dt_fields.size(), 2u);
    EXPECT_EQ(dt_fields[0], seed_transient);
    EXPECT_EQ(dt_fields[1], phi_field);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto extension_offset = sys.fieldDofOffset(extension_field);
    const auto extension_dofs =
        static_cast<std::size_t>(sys.fieldDofHandler(extension_field).getNumDofs());
    std::vector<GlobalIndex> constrained_extension_dofs(extension_dofs);
    for (std::size_t i = 0; i < extension_dofs; ++i) {
        constrained_extension_dofs[i] =
            extension_offset + static_cast<GlobalIndex>(i);
    }
    sys.addConstraint(std::make_unique<svmp::FE::constraints::DirichletBC>(
        std::move(constrained_extension_dofs), 1.0));
    sys.setup({}, inputs);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    constexpr double dt = 0.1;
    auto history = svmp::FE::timestepping::TimeHistory::allocate(
        *factory, sys.dofHandler().getNumDofs(), /*history_depth=*/2);
    std::vector<Real> initial(
        static_cast<std::size_t>(sys.dofHandler().getNumDofs()), Real{0.0});
    for (std::size_t i = 0; i < extension_dofs; ++i) {
        initial[static_cast<std::size_t>(extension_offset) + i] = Real{1.0};
    }
    setVectorByDof(history.uPrev(), initial);
    setVectorByDof(history.uPrev2(), initial);
    history.resetCurrentToPrevious();
    history.setPrevDt(dt);

    auto integrator =
        std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt;
    opts.dt = dt;
    opts.max_steps = 1;
    opts.scheme = svmp::FE::timestepping::SchemeKind::GeneralizedAlpha;
    opts.generalized_alpha_rho_inf = 0.5;
    opts.initialize_first_order_rate_from_pde = false;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1.0e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    const auto report = loop.run(
        transient,
        *factory,
        *linear,
        history,
        callbacks);
    ASSERT_TRUE(report.success) << report.message;

    const auto accepted = getVectorByDof(history.uPrev());
    const auto phi_offset = sys.fieldDofOffset(phi_field);
    const auto phi_dofs =
        static_cast<std::size_t>(sys.fieldDofHandler(phi_field).getNumDofs());
    // rho_inf=0.5 gives alpha_m=5/6, alpha_f=gamma=2/3. With
    // phi_dot(0)=0 and phi_dot + 1 = 0, the first accepted endpoint is
    // -gamma/alpha_m*dt=-0.08; the intermediate stage is -0.053333... .
    constexpr double expected_endpoint = -0.08;
    constexpr double stage_value = -0.053333333333333333;
    for (std::size_t i = 0; i < phi_dofs; ++i) {
        const auto value = static_cast<double>(
            accepted[static_cast<std::size_t>(phi_offset) + i]);
        EXPECT_NEAR(value, expected_endpoint, 1.0e-12);
        EXPECT_GT(std::abs(value - stage_value), 1.0e-2);
    }
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaExternalStateFixedPointPreservesStageTransactionAndCommitsEndpointOnce)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using StateSyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

    static constexpr double dt = 0.1;
    static constexpr double stage_time = 1.0 / 15.0;
    static constexpr double predictor_scale = 14.0 / 15.0;
    static constexpr double stage_scale = 74.0 / 79.0;
    static constexpr double endpoint_scale = 143.0 / 158.0;
    static constexpr double endpoint_rate_scale = -73.0 / 79.0;
    const std::vector<Real> initial = {1.0, -0.5, 0.25, 2.0};
    std::vector<Real> initial_rate(initial.size(), Real{0.0});
    for (std::size_t i = 0; i < initial.size(); ++i) {
        initial_rate[i] = -initial[i];
    }

    struct Observations {
        svmp::FE::timestepping::TimeHistory* history{nullptr};
        int outer_callbacks{0};
        int projected_callbacks{0};
        int endpoint_callbacks{0};
        int projected_endpoint_callbacks{0};
        int restored_callbacks{0};
        int nonlinear_done_callbacks{0};
        int before_accept_callbacks{0};
        int accepted_callbacks{0};
        int reported_outer_iterations{-1};
        int reported_inner_iterations{-1};
        int reported_iterations{-1};
    };
    auto observations = std::make_shared<Observations>();

    auto expect_scaled = [initial](std::span<const Real> values,
                                   double scale) {
        EXPECT_EQ(values.size(), initial.size());
        if (values.size() != initial.size()) {
            return;
        }
        for (std::size_t i = 0; i < initial.size(); ++i) {
            EXPECT_NEAR(static_cast<double>(values[i]),
                        scale * static_cast<double>(initial[i]),
                        1.0e-12)
                << "DOF " << i;
        }
    };

    auto rate_factory = createTestFactory();
    ASSERT_NE(rate_factory.get(), nullptr);
    const auto accepted = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        dt,
        dt,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/0.5,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [observations, rate_factory_ptr = rate_factory.get(), initial_rate,
         expect_scaled](
            svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            observations->history = &history;
            ASSERT_NE(rate_factory_ptr, nullptr);
            history.ensureSecondOrderState(*rate_factory_ptr);
            setVectorByDof(history.uDot(), initial_rate);
            history.uDDot().zero();

            callbacks.on_nonlinear_done =
                [observations, expect_scaled](
                    const svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport& report) {
                    ++observations->nonlinear_done_callbacks;
                    observations->reported_outer_iterations =
                        report.outer_iterations;
                    observations->reported_inner_iterations =
                        report.inner_iterations_total;
                    observations->reported_iterations = report.iterations;
                    EXPECT_TRUE(report.converged);
                    EXPECT_EQ(report.outer_iterations, 2);
                    EXPECT_EQ(report.inner_iterations_total, 1);
                    EXPECT_EQ(report.iterations, 1);
                    EXPECT_NEAR(report.outer_state_change_norm, 0.0, 1.0e-12);
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1.0e-15);
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), 1.0);
                    expect_scaled(h.uPrev2Span(), 1.0);
                    expect_scaled(h.uDotSpan(), endpoint_rate_scale);
                };
            callbacks.on_before_step_accept =
                [observations, expect_scaled](
                    svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport&) {
                    ++observations->before_accept_callbacks;
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1.0e-15);
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), 1.0);
                    expect_scaled(h.uPrev2Span(), 1.0);
                    expect_scaled(h.uDotSpan(), endpoint_rate_scale);
                    return true;
                };
            callbacks.on_step_accepted =
                [observations, expect_scaled](
                    svmp::FE::timestepping::TimeHistory& h) {
                    ++observations->accepted_callbacks;
                    EXPECT_EQ(h.stepIndex(), 1);
                    EXPECT_NEAR(h.time(), dt, 1.0e-15);
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), endpoint_scale);
                    expect_scaled(h.uPrev2Span(), 1.0);
                    expect_scaled(h.uDotSpan(), endpoint_rate_scale);
                };
        },
        /*inspect_expected_exception=*/{},
        [observations, initial_rate, expect_scaled](
            svmp::FE::timestepping::TimeLoopOptions& options,
            svmp::FE::FieldId) {
            options.initialize_first_order_rate_from_pde = false;
            options.newton.external_state_fixed_point.enabled = true;
            options.newton.external_state_fixed_point.max_iterations = 4;
            options.newton.synchronize_state =
                [observations, initial_rate, expect_scaled](
                    const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    ASSERT_NE(observations->history, nullptr);
                    if (point == StateSyncPoint::EndpointCandidateState ||
                        point ==
                            StateSyncPoint::ProjectedEndpointCandidateState) {
                        EXPECT_EQ(observations->history->stepIndex(), 0);
                        EXPECT_NEAR(observations->history->time(), 0.0, 1.0e-15);
                        EXPECT_NEAR(state.time, dt, 1.0e-15);
                        EXPECT_NEAR(state.dt, dt, 1.0e-15);
                        EXPECT_NEAR(state.effective_dt, dt, 1.0e-15);
                        EXPECT_EQ(state.time_integration, nullptr);
                        expect_scaled(state.u, endpoint_scale);
                        expect_scaled(state.u_prev, 1.0);
                        expect_scaled(state.u_prev2, 1.0);
                        expect_scaled(observations->history->uDotSpan(),
                                      endpoint_rate_scale);
                        if (point == StateSyncPoint::EndpointCandidateState) {
                            ++observations->endpoint_callbacks;
                        } else {
                            ++observations->projected_endpoint_callbacks;
                        }
                        return;
                    }
                    if (point == StateSyncPoint::RestoredTimeStepState ||
                        point ==
                            StateSyncPoint::RestoredProjectedTimeStepState) {
                        ++observations->restored_callbacks;
                        return;
                    }
                    ASSERT_TRUE(
                        point == StateSyncPoint::OuterFixedPointState ||
                        point ==
                            StateSyncPoint::ProjectedOuterFixedPointState);
                    EXPECT_EQ(observations->history->stepIndex(), 0);
                    EXPECT_NEAR(observations->history->time(), 0.0, 1.0e-15);
                    EXPECT_NEAR(state.time, stage_time, 1.0e-15);
                    EXPECT_NEAR(state.dt, dt, 1.0e-15);
                    EXPECT_NEAR(state.effective_dt, stage_time, 1.0e-15);

                    ASSERT_NE(state.time_integration, nullptr);
                    EXPECT_EQ(state.time_integration->integrator_name,
                              "GeneralizedAlpha(1stOrder)");
                    ASSERT_TRUE(state.time_integration->dt1.has_value());
                    const auto& stencil = *state.time_integration->dt1;
                    ASSERT_GE(stencil.a.size(), 3u);
                    EXPECT_NEAR(static_cast<double>(stencil.a[0]),
                                18.75,
                                1.0e-13);
                    EXPECT_NEAR(static_cast<double>(stencil.a[1]),
                                -18.75,
                                1.0e-13);
                    EXPECT_NEAR(static_cast<double>(stencil.a[2]),
                                -0.25,
                                1.0e-13);

                    expect_scaled(state.u_prev, 1.0);
                    EXPECT_EQ(state.u_prev2.size(), initial_rate.size());
                    if (state.u_prev2.size() == initial_rate.size()) {
                        for (std::size_t i = 0; i < initial_rate.size(); ++i) {
                            EXPECT_NEAR(
                                static_cast<double>(state.u_prev2[i]),
                                static_cast<double>(initial_rate[i]),
                                1.0e-12)
                                << "DOF " << i;
                        }
                    }
                    const auto rate = observations->history->uDotSpan();
                    EXPECT_EQ(rate.size(), initial_rate.size());
                    if (rate.size() == initial_rate.size()) {
                        for (std::size_t i = 0; i < initial_rate.size(); ++i) {
                            EXPECT_NEAR(static_cast<double>(rate[i]),
                                        static_cast<double>(initial_rate[i]),
                                        1.0e-12)
                                << "DOF " << i;
                        }
                    }

                    if (point == StateSyncPoint::OuterFixedPointState) {
                        const double expected_scale =
                            observations->outer_callbacks == 0
                                ? predictor_scale
                                : stage_scale;
                        expect_scaled(state.u, expected_scale);
                        ++observations->outer_callbacks;
                    } else {
                        const double expected_scale =
                            observations->projected_callbacks == 0
                                ? predictor_scale
                                : stage_scale;
                        expect_scaled(state.u, expected_scale);
                        ++observations->projected_callbacks;
                    }
                };
        });

    ASSERT_EQ(accepted.size(), initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(accepted[i]),
                    endpoint_scale * static_cast<double>(initial[i]),
                    1.0e-12)
            << "DOF " << i;
    }
    EXPECT_EQ(observations->outer_callbacks, 2);
    EXPECT_EQ(observations->projected_callbacks, 2);
    EXPECT_EQ(observations->endpoint_callbacks, 1);
    EXPECT_EQ(observations->projected_endpoint_callbacks, 1);
    EXPECT_EQ(observations->restored_callbacks, 0);
    EXPECT_EQ(observations->nonlinear_done_callbacks, 1);
    EXPECT_EQ(observations->before_accept_callbacks, 1);
    EXPECT_EQ(observations->accepted_callbacks, 1);
    EXPECT_EQ(observations->reported_outer_iterations, 2);
    EXPECT_EQ(observations->reported_inner_iterations, 1);
    EXPECT_EQ(observations->reported_iterations, 1);
}

TEST(TimeLoopConvergence,
     BackwardEulerExternalStateFixedPointPreservesEndpointTransaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using StateSyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

    static constexpr double dt = 0.1;
    static constexpr double endpoint_scale = 10.0 / 11.0;
    const std::vector<Real> initial = {1.0, -0.5, 0.25, 2.0};

    struct Observations {
        int outer_callbacks{0};
        int projected_callbacks{0};
        int nonlinear_done_callbacks{0};
        int before_accept_callbacks{0};
        int commit_ready_callbacks{0};
        int accepted_callbacks{0};
    };
    auto observations = std::make_shared<Observations>();

    auto expect_scaled = [initial](std::span<const Real> values,
                                   double scale) {
        EXPECT_EQ(values.size(), initial.size());
        if (values.size() != initial.size()) {
            return;
        }
        for (std::size_t i = 0; i < initial.size(); ++i) {
            EXPECT_NEAR(static_cast<double>(values[i]),
                        scale * static_cast<double>(initial[i]),
                        1.0e-12)
                << "DOF " << i;
        }
    };

    const auto accepted = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::BackwardEuler,
        dt,
        dt,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/-7.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [observations, expect_scaled](
            svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory& history) {
            EXPECT_FALSE(history.hasUDotState());
            callbacks.on_nonlinear_done =
                [observations, expect_scaled](
                    const svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport& report) {
                    ++observations->nonlinear_done_callbacks;
                    EXPECT_TRUE(report.converged);
                    EXPECT_EQ(report.outer_iterations, 2);
                    EXPECT_EQ(report.inner_iterations_total, 1);
                    EXPECT_EQ(report.iterations, 1);
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1.0e-15);
                    EXPECT_FALSE(h.hasUDotState());
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), 1.0);
                };
            callbacks.on_before_step_accept =
                [observations, expect_scaled](
                    svmp::FE::timestepping::TimeHistory& h,
                    const svmp::FE::timestepping::NewtonReport&) {
                    ++observations->before_accept_callbacks;
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1.0e-15);
                    EXPECT_FALSE(h.hasUDotState());
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), 1.0);
                    expect_scaled(h.uPrev2Span(), 1.0);
                    return true;
                };
            callbacks.on_step_commit_ready =
                [observations, expect_scaled](
                    svmp::FE::timestepping::TimeHistory& h) {
                    ++observations->commit_ready_callbacks;
                    EXPECT_EQ(h.stepIndex(), 0);
                    EXPECT_NEAR(h.time(), 0.0, 1.0e-15);
                    EXPECT_FALSE(h.hasUDotState());
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), 1.0);
                    expect_scaled(h.uPrev2Span(), 1.0);
                };
            callbacks.on_step_accepted =
                [observations, expect_scaled](
                    svmp::FE::timestepping::TimeHistory& h) {
                    ++observations->accepted_callbacks;
                    EXPECT_EQ(h.stepIndex(), 1);
                    EXPECT_NEAR(h.time(), dt, 1.0e-15);
                    EXPECT_FALSE(h.hasUDotState());
                    expect_scaled(h.uSpan(), endpoint_scale);
                    expect_scaled(h.uPrevSpan(), endpoint_scale);
                    expect_scaled(h.uPrev2Span(), 1.0);
                };
        },
        /*inspect_expected_exception=*/{},
        [observations, expect_scaled](
            svmp::FE::timestepping::TimeLoopOptions& options,
            svmp::FE::FieldId) {
            // This deliberately remains true to prove that the option and
            // generalized-alpha spectral radius are inert for backward Euler.
            options.initialize_first_order_rate_from_pde = true;
            options.newton.external_state_fixed_point.enabled = true;
            options.newton.external_state_fixed_point.max_iterations = 4;
            options.newton.synchronize_state =
                [observations, expect_scaled](
                    const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    if (point == StateSyncPoint::OuterFixedPointState) {
                        ++observations->outer_callbacks;
                    } else if (
                        point ==
                        StateSyncPoint::ProjectedOuterFixedPointState) {
                        ++observations->projected_callbacks;
                    } else {
                        ADD_FAILURE()
                            << "Backward Euler requested an intermediate-stage "
                               "or rollback synchronization point";
                        return;
                    }
                    EXPECT_NEAR(state.time, dt, 1.0e-15);
                    EXPECT_NEAR(state.dt, dt, 1.0e-15);
                    EXPECT_NEAR(state.effective_dt, dt, 1.0e-15);
                    ASSERT_NE(state.time_integration, nullptr);
                    EXPECT_EQ(
                        state.time_integration->integrator_name,
                        "BackwardDifference");
                    ASSERT_TRUE(state.time_integration->dt1.has_value());
                    const auto& stencil =
                        *state.time_integration->dt1;
                    ASSERT_EQ(stencil.a.size(), 2u);
                    EXPECT_NEAR(
                        static_cast<double>(stencil.a[0]),
                        10.0,
                        1.0e-13);
                    EXPECT_NEAR(
                        static_cast<double>(stencil.a[1]),
                        -10.0,
                        1.0e-13);
                    expect_scaled(state.u_prev, 1.0);
                };
        });

    ASSERT_EQ(accepted.size(), initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(accepted[i]),
                    endpoint_scale * static_cast<double>(initial[i]),
                    1.0e-12)
            << "DOF " << i;
    }
    EXPECT_EQ(observations->outer_callbacks, 2);
    EXPECT_EQ(observations->projected_callbacks, 2);
    EXPECT_EQ(observations->nonlinear_done_callbacks, 1);
    EXPECT_EQ(observations->before_accept_callbacks, 1);
    EXPECT_EQ(observations->commit_ready_callbacks, 1);
    EXPECT_EQ(observations->accepted_callbacks, 1);
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaExternalStateFixedPointProjectsInjectedMpcRateHomogeneously)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using StateSyncPoint =
        svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;

    // The value constraint is u_1 = u_0 - 1.5. Its time derivative is the
    // homogeneous relation uDot_1 = uDot_0; the -1.5 offset must never be
    // injected into the generalized-alpha rate slot stored temporarily in
    // uPrev2.
    const std::vector<Real> initial_rate = {-1.0, -1.0, -0.25, -2.0};
    struct Observations {
        svmp::FE::timestepping::TimeHistory* history{nullptr};
        int outer_callbacks{0};
        int projected_callbacks{0};
        bool outer_rate_was_homogeneous{true};
        bool projected_rate_was_homogeneous{true};
        bool explicit_rate_state_was_homogeneous{true};
    };
    auto observations = std::make_shared<Observations>();
    auto rate_factory = createTestFactory();
    ASSERT_NE(rate_factory.get(), nullptr);

    const auto accepted = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/0.5,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [observations, rate_factory_ptr = rate_factory.get(), initial_rate](
            svmp::FE::timestepping::TimeLoopCallbacks&,
            svmp::FE::timestepping::TimeHistory& history) {
            observations->history = &history;
            ASSERT_NE(rate_factory_ptr, nullptr);
            history.ensureSecondOrderState(*rate_factory_ptr);
            setVectorByDof(history.uDot(), initial_rate);
            history.uDDot().zero();
        },
        /*inspect_expected_exception=*/{},
        [observations](svmp::FE::timestepping::TimeLoopOptions& options,
                       svmp::FE::FieldId) {
            options.initialize_first_order_rate_from_pde = false;
            options.newton.external_state_fixed_point.enabled = true;
            options.newton.external_state_fixed_point.max_iterations = 4;
            options.newton.synchronize_state =
                [observations](
                    const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    ASSERT_NE(observations->history, nullptr);
                    if (point == StateSyncPoint::EndpointCandidateState ||
                        point ==
                            StateSyncPoint::ProjectedEndpointCandidateState) {
                        return;
                    }
                    if (point != StateSyncPoint::OuterFixedPointState &&
                        point !=
                            StateSyncPoint::ProjectedOuterFixedPointState) {
                        ADD_FAILURE()
                            << "Unexpected external-state synchronization point";
                        return;
                    }
                    ASSERT_GE(state.u_prev2.size(), 2u);
                    const bool injected_rate_is_homogeneous =
                        std::abs(static_cast<double>(state.u_prev2[0]) + 1.0) <
                            1.0e-12 &&
                        std::abs(static_cast<double>(state.u_prev2[1]) + 1.0) <
                            1.0e-12;
                    const auto explicit_rate =
                        observations->history->uDotSpan();
                    ASSERT_GE(explicit_rate.size(), 2u);
                    const bool explicit_rate_is_homogeneous =
                        std::abs(static_cast<double>(explicit_rate[0]) + 1.0) <
                            1.0e-12 &&
                        std::abs(static_cast<double>(explicit_rate[1]) + 1.0) <
                            1.0e-12;
                    observations->explicit_rate_state_was_homogeneous =
                        observations->explicit_rate_state_was_homogeneous &&
                        explicit_rate_is_homogeneous;
                    if (point == StateSyncPoint::OuterFixedPointState) {
                        ++observations->outer_callbacks;
                        observations->outer_rate_was_homogeneous =
                            observations->outer_rate_was_homogeneous &&
                            injected_rate_is_homogeneous;
                    } else {
                        ++observations->projected_callbacks;
                        observations->projected_rate_was_homogeneous =
                            observations->projected_rate_was_homogeneous &&
                            injected_rate_is_homogeneous;
                    }
                };
        },
        [](svmp::FE::systems::FESystem& system, svmp::FE::FieldId) {
            // runReactionProblem has one scalar field, so its four global DOFs
            // are numbered 0..3 before setup.
            auto mpc = std::make_unique<
                svmp::FE::constraints::MultiPointConstraint>();
            mpc->addConstraint(
                /*slave_dof=*/1,
                /*master_dof=*/0,
                /*weight=*/1.0,
                /*inhomogeneity=*/-1.5);
            system.addConstraint(std::move(mpc));
        });

    ASSERT_EQ(accepted.size(), 4u);
    EXPECT_GE(observations->outer_callbacks, 1);
    EXPECT_GE(observations->projected_callbacks, 1);
    EXPECT_TRUE(observations->outer_rate_was_homogeneous);
    EXPECT_TRUE(observations->projected_rate_was_homogeneous)
        << "The nonzero MPC inhomogeneity was added to the generalized-alpha "
           "uDot_n slot during projected outer synchronization";
    EXPECT_TRUE(observations->explicit_rate_state_was_homogeneous);
}

TEST(TimeLoopConvergence,
     GeneralizedAlphaEndpointGeneratedStateSeesEndpointConstraintProjection)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using StateSyncPoint = svmp::FE::timestepping::NewtonOptions::
        StateSynchronizationPoint;

    struct Observations {
        int endpoint_callbacks{0};
        int projected_endpoint_callbacks{0};
        double raw_endpoint_value{0.0};
        double projected_endpoint_value{0.0};
        double projected_previous_value{0.0};
        double accepted_previous_value{0.0};
        double accepted_endpoint_rate{0.0};
    } observations;

    const auto accepted = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::GeneralizedAlpha,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/0.5,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1.0e-12,
        /*newton_rel_tolerance=*/0.0,
        [&](svmp::FE::timestepping::TimeLoopCallbacks& callbacks,
            svmp::FE::timestepping::TimeHistory&) {
            callbacks.on_step_accepted =
                [&](svmp::FE::timestepping::TimeHistory& history) {
                    observations.accepted_previous_value =
                        static_cast<double>(history.uPrev2Span()[0]);
                    observations.accepted_endpoint_rate =
                        static_cast<double>(history.uDotSpan()[0]);
                };
        },
        /*inspect_expected_exception=*/{},
        [&](svmp::FE::timestepping::TimeLoopOptions& options,
            svmp::FE::FieldId) {
            options.initialize_first_order_rate_from_pde = false;
            options.newton.synchronize_state =
                [&](const svmp::FE::systems::SystemStateView& state,
                    StateSyncPoint point) {
                    if (point == StateSyncPoint::EndpointCandidateState) {
                        ++observations.endpoint_callbacks;
                        ASSERT_FALSE(state.u.empty());
                        observations.raw_endpoint_value =
                            static_cast<double>(state.u[0]);
                    } else if (
                        point ==
                        StateSyncPoint::ProjectedEndpointCandidateState) {
                        ++observations.projected_endpoint_callbacks;
                        ASSERT_FALSE(state.u.empty());
                        ASSERT_FALSE(state.u_prev.empty());
                        observations.projected_endpoint_value =
                            static_cast<double>(state.u[0]);
                        observations.projected_previous_value =
                            static_cast<double>(state.u_prev[0]);
                    }
                };
        },
        [](svmp::FE::systems::FESystem& system, svmp::FE::FieldId) {
            std::vector<GlobalIndex> dofs{0};
            std::vector<std::array<double, 3>> coordinates{{{0.0, 0.0, 0.0}}};
            auto endpoint_bc = std::make_unique<
                svmp::FE::constraints::DirichletBC>(
                std::move(dofs),
                std::move(coordinates),
                [](double, double, double, double time) {
                    return 1.0 + time * time;
                },
                /*initial_time=*/0.0);
            system.addConstraint(std::move(endpoint_bc));
        });

    ASSERT_EQ(accepted.size(), 4u);
    EXPECT_EQ(observations.endpoint_callbacks, 1);
    EXPECT_EQ(observations.projected_endpoint_callbacks, 1);
    // At rho_inf=0.5, alpha_f=2/3.  Extrapolating g(t_stage)=
    // 1+(dt*alpha_f)^2 produces 151/150, while the actual endpoint
    // constraint is g(dt)=1.01.
    EXPECT_NEAR(observations.raw_endpoint_value, 151.0 / 150.0, 1.0e-12);
    EXPECT_NEAR(observations.projected_endpoint_value, 1.01, 1.0e-12);
    EXPECT_NEAR(observations.projected_previous_value, 1.0, 1.0e-15)
        << "Endpoint constraint projection modified u_n history";
    EXPECT_NEAR(static_cast<double>(accepted[0]), 1.01, 1.0e-12);
    EXPECT_NEAR(observations.accepted_previous_value, 1.0, 1.0e-15);
    // gamma=2/3 and uDot_n=0, hence uDot_{n+1}=
    // (1.01-1)/(gamma*dt)=0.15 after endpoint projection.
    EXPECT_NEAR(observations.accepted_endpoint_rate, 0.15, 1.0e-12);
}

TEST(TimeLoopConvergence, GeneralizedAlpha_FirstOrder_RhoInfSweep_IsSecondOrderOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;

    // Use a smaller dt pair so the strongly dissipative ρ∞→0 endpoint is in the
    // asymptotic convergence regime (avoid accidental cancellation at coarser dt).
    const double dt1 = 0.05;
    const double dt2 = 0.025;

    const std::vector<double> rho_values = {0.0, 0.2, 0.5, 0.9, 1.0};

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    for (double rho_inf : rho_values) {
        const auto u_dt1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt1, t_end, lambda,
                                              /*history_depth=*/3,
                                              /*controller=*/{},
                                              /*generalized_alpha_rho_inf=*/rho_inf,
                                              /*dg_degree=*/1,
                                              /*cg_degree=*/2,
                                              svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                              /*collocation_max_outer_iterations=*/4,
                                              /*collocation_outer_tolerance=*/0.0,
                                              /*exact_initial_history=*/true);
        const auto u_dt2 = runReactionProblem(svmp::FE::timestepping::SchemeKind::GeneralizedAlpha, dt2, t_end, lambda,
                                              /*history_depth=*/3,
                                              /*controller=*/{},
                                              /*generalized_alpha_rho_inf=*/rho_inf,
                                              /*dg_degree=*/1,
                                              /*cg_degree=*/2,
                                              svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                              /*collocation_max_outer_iterations=*/4,
                                              /*collocation_outer_tolerance=*/0.0,
                                              /*exact_initial_history=*/true);
        ASSERT_EQ(u_dt1.size(), 4u);
        ASSERT_EQ(u_dt2.size(), 4u);

        const double e1 = relativeL2Error(u_dt1, exact);
        const double e2 = relativeL2Error(u_dt2, exact);
        const double p = std::log(e1 / e2) / std::log(2.0);
        EXPECT_GT(p, 1.6) << "rho_inf=" << rho_inf << " e1=" << e1 << " e2=" << e2;
    }
}

TEST(TimeLoopEquivalences, DG0_MatchesBackwardEulerOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_be = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, t_end, lambda);
    const auto u_dg0 = runReactionProblem(svmp::FE::timestepping::SchemeKind::DG0, dt, t_end, lambda);
    ASSERT_EQ(u_be.size(), u_dg0.size());

    for (std::size_t i = 0; i < u_be.size(); ++i) {
        EXPECT_NEAR(u_be[i], u_dg0[i], 1e-12);
    }
}

TEST(TimeLoopCollocation,
     MonolithicRejectsFieldResidualCriteriaWithActionableMessage)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP()
        << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    bool saw_expected_exception = false;
    (void)runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG1,
        /*dt=*/0.1,
        /*t_end=*/0.1,
        /*lambda=*/1.0,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
        /*collocation_max_outer_iterations=*/4,
        /*collocation_outer_tolerance=*/0.0,
        /*exact_initial_history=*/false,
        /*theta=*/0.5,
        /*newton_max_iterations=*/8,
        /*newton_abs_tolerance=*/1e-12,
        /*newton_rel_tolerance=*/0.0,
        /*configure_callbacks=*/{},
        [&saw_expected_exception](
            const svmp::FE::timestepping::TimeHistory&,
            const svmp::FE::FEException& exception) {
            saw_expected_exception = true;
            const std::string message = exception.what();
            EXPECT_NE(message.find("monolithic collocation"),
                      std::string::npos);
            EXPECT_NE(message.find("StageGaussSeidel"),
                      std::string::npos);
        },
        [](svmp::FE::timestepping::TimeLoopOptions& options,
           svmp::FE::FieldId field) {
            options.newton.field_residual_criteria.push_back({
                .field = field,
                .abs_tolerance = 1e-12,
                .rel_tolerance = 0.0});
        });

    EXPECT_TRUE(saw_expected_exception);
}

TEST(TimeLoopEquivalences, CG1_MatchesThetaHalfOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_theta = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod, dt, t_end, lambda);
    const auto u_cg1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::CG1, dt, t_end, lambda);
    ASSERT_EQ(u_theta.size(), u_cg1.size());

		    for (std::size_t i = 0; i < u_theta.size(); ++i) {
		        EXPECT_NEAR(u_theta[i], u_cg1[i], 1e-12);
		    }
		}

TEST(TimeLoopEquivalences, ThetaMethod_Theta1_MatchesBackwardEulerOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.4;
    const double dt = 0.1;

    const auto u_be = runReactionProblem(svmp::FE::timestepping::SchemeKind::BackwardEuler, dt, t_end, lambda);
    const auto u_theta1 = runReactionProblem(svmp::FE::timestepping::SchemeKind::ThetaMethod,
                                             dt,
                                             t_end,
                                             lambda,
                                             /*history_depth=*/2,
                                             /*controller=*/{},
                                             /*generalized_alpha_rho_inf=*/1.0,
                                             /*dg_degree=*/1,
                                             /*cg_degree=*/2,
                                             svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                             /*collocation_max_outer_iterations=*/4,
                                             /*collocation_outer_tolerance=*/0.0,
                                             /*exact_initial_history=*/false,
                                             /*theta=*/1.0);
    ASSERT_EQ(u_be.size(), u_theta1.size());

    for (std::size_t i = 0; i < u_be.size(); ++i) {
        EXPECT_NEAR(u_be[i], u_theta1[i], 1e-12);
    }
}

		TEST(TimeLoopEquivalences, DG1_StageGaussSeidelIsCloseToMonolithicOnReaction)
		{
		#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
		    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
	#endif
	    const double lambda = 1.0;
	    const double t_end = 0.4;
	    const double dt = 0.1;

	    const auto u_monolithic = runReactionProblem(
	        svmp::FE::timestepping::SchemeKind::DG1,
	        dt,
	        t_end,
	        lambda,
	        /*history_depth=*/2,
	        /*controller=*/{},
	        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
	    const auto u_gs = runReactionProblem(
	        svmp::FE::timestepping::SchemeKind::DG1,
	        dt,
	        t_end,
	        lambda,
	        /*history_depth=*/2,
	        /*controller=*/{},
	        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
	        /*collocation_max_outer_iterations=*/20,
	        /*collocation_outer_tolerance=*/0.0);
	    ASSERT_EQ(u_monolithic.size(), u_gs.size());

		    const double e = relativeL2Error(u_gs, u_monolithic);
		    EXPECT_LT(e, 1e-5);
		}

TEST(TimeLoopEquivalences, DG1_StageGaussSeidelMatchesMonolithicOnReactionWhenConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.2;
    const double dt = 0.1;

    const auto u_monolithic = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG1,
        dt,
        t_end,
        lambda,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
    const auto u_gs = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG1,
        dt,
        t_end,
        lambda,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
	        /*collocation_max_outer_iterations=*/80,
	        /*collocation_outer_tolerance=*/1e-10);

	    ASSERT_EQ(u_monolithic.size(), u_gs.size());
	    EXPECT_LT(relativeL2Error(u_gs, u_monolithic), 1e-10);
	}

TEST(TimeLoopEquivalences, DG2_StageGaussSeidelMatchesMonolithicOnReactionWhenConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    // For 3-stage Radau IIA (dG2), the stage-wise Gauss–Seidel iteration is not
    // guaranteed to converge for weakly damped problems. Use a sufficiently
    // stiff decay (large lambda) so both strategies converge and can be compared.
    const double lambda = 20.0;
    const double t_end = 0.2;
    const double dt = 0.1;

    const auto u_monolithic = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG,
	        dt,
	        t_end,
	        lambda,
	        /*history_depth=*/3,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/2,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
		    const auto u_gs = runReactionProblem(
		        svmp::FE::timestepping::SchemeKind::DG,
		        dt,
		        t_end,
	        lambda,
	        /*history_depth=*/3,
	        /*controller=*/{},
	        /*generalized_alpha_rho_inf=*/1.0,
		        /*dg_degree=*/2,
		        /*cg_degree=*/2,
		        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
		        /*collocation_max_outer_iterations=*/200,
		        /*collocation_outer_tolerance=*/1e-10);

		    ASSERT_EQ(u_monolithic.size(), u_gs.size());
		    EXPECT_LT(relativeL2Error(u_gs, u_monolithic), 1e-9);
		}

TEST(TimeLoopEquivalences, CG2_StageGaussSeidelMatchesMonolithicOnReactionWhenConverged)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.2;
    const double dt = 0.1;

    const auto u_monolithic = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::CG2,
        dt,
        t_end,
        lambda,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/1,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
	    const auto u_gs = runReactionProblem(
	        svmp::FE::timestepping::SchemeKind::CG2,
	        dt,
	        t_end,
        lambda,
        /*history_depth=*/2,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
	        /*dg_degree=*/1,
	        /*cg_degree=*/2,
	        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
	        /*collocation_max_outer_iterations=*/80,
	        /*collocation_outer_tolerance=*/1e-10);

	    ASSERT_EQ(u_monolithic.size(), u_gs.size());
	    EXPECT_LT(relativeL2Error(u_gs, u_monolithic), 1e-10);
	}

TEST(TimeLoopEquivalences, StageGaussSeidelOuterToleranceCanExitEarlyAndDifferFromMonolithic)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 50.0;
    const double t_end = 0.1;
    const double dt = 0.1;

    const auto u_monolithic = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG,
        dt,
        t_end,
        lambda,
        /*history_depth=*/3,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/2,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::Monolithic);
    const auto u_gs_early = runReactionProblem(
        svmp::FE::timestepping::SchemeKind::DG,
        dt,
        t_end,
        lambda,
        /*history_depth=*/3,
        /*controller=*/{},
        /*generalized_alpha_rho_inf=*/1.0,
        /*dg_degree=*/2,
        /*cg_degree=*/2,
        svmp::FE::timestepping::CollocationSolveStrategy::StageGaussSeidel,
        /*collocation_max_outer_iterations=*/200,
        /*collocation_outer_tolerance=*/1e6);

    const double e = relativeL2Error(u_gs_early, u_monolithic);
    EXPECT_GT(e, 1e-10);
}

	TEST(TimeLoopVSVO_BDF, AdaptsDtOnReaction)
	{
	#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
	    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 5.0;
    const double t_end = 1.0;
    const double dt0 = 0.5;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-10;
    ctrl_opts.rel_tol = 1e-6;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 3;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 8;
    ctrl_opts.safety = 0.9;
    ctrl_opts.min_factor = 0.2;
    ctrl_opts.max_factor = 2.0;
    ctrl_opts.increase_order_threshold = 0.05;

    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/5);

    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), u0);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    std::vector<std::pair<double, double>> dt_updates;
    callbacks.on_dt_updated = [&dt_updates](double oldv, double newv, int, int) {
        dt_updates.emplace_back(oldv, newv);
    };

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
    if (!rep.success) {
        std::cerr << "TimeLoopVSVO_BDF dt2 oscillator failure message: " << rep.message << std::endl;
    }
    EXPECT_TRUE(rep.success) << rep.message;
    EXPECT_NEAR(rep.final_time, t_end, 1e-12) << rep.message;
    EXPECT_FALSE(dt_updates.empty());

    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }
	    const auto approx = getVectorByDof(history.uPrev());
	    const double err = relativeL2Error(approx, exact);
	    if (err >= 5e-4) {
	        double dt_min = dt0;
	        double dt_max = dt0;
	        for (const auto& p : dt_updates) {
	            dt_min = std::min(dt_min, std::min(p.first, p.second));
	            dt_max = std::max(dt_max, std::max(p.first, p.second));
	        }
	        std::cerr << "TimeLoopVSVO_BDF.AdaptsDtOnReaction diagnostics: "
	                  << "steps_taken=" << rep.steps_taken
	                  << " final_time=" << rep.final_time
	                  << " dt_prev=" << history.dtPrev()
	                  << " dt_updates=" << dt_updates.size()
	                  << " dt_min=" << dt_min
	                  << " dt_max=" << dt_max
	                  << std::endl;
	    }
	    EXPECT_LT(err, 5e-4);
	}

TEST(TimeLoopVSVO_BDF, RestartRequiresValidDtHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.2;
    const double dt0 = 0.1;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-6;
    ctrl_opts.rel_tol = 1e-4;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 2;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 2;
    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/3);
    for (int k = 1; k <= history.historyDepth(); ++k) {
        setVectorByDof(history.uPrevK(k), std::vector<Real>(static_cast<std::size_t>(n_dofs), 0.0));
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    // Simulate a restart with existing history but missing dtHistory() entries.
    history.setStepIndex(2);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 6;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    EXPECT_THROW((void)loop.run(transient, *factory, *linear, history), svmp::FE::InvalidArgumentException);
}

TEST(TimeLoopVSVO_BDF, RestartMatchesContinuousRunOnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 1.0;
    const double t_split = 0.5;
    const double dt = 0.05;
    const int order = 4;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);
    const std::vector<Real> u0 = {1.0, -0.5, 0.25, 2.0};

    auto makeHistory = [&](double dt0) {
        auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/order + 1);
        for (int k = 1; k <= history.historyDepth(); ++k) {
            const double t_k = -static_cast<double>(k - 1) * dt0;
            const double scale = std::exp(-lambda * t_k);
            std::vector<Real> u_k(u0.size(), 0.0);
            for (std::size_t i = 0; i < u0.size(); ++i) {
                u_k[i] = static_cast<Real>(static_cast<double>(u0[i]) * scale);
            }
            setVectorByDof(history.uPrevK(k), u_k);
        }
        history.resetCurrentToPrevious();
        history.setPrevDt(dt0);
        return history;
    };

    auto run = [&](double t0, double t1, svmp::FE::timestepping::TimeHistory& history) {
        svmp::FE::timestepping::TimeLoopOptions opts;
        opts.t0 = t0;
        opts.t_end = t1;
        opts.dt = dt;
        opts.max_steps = 2000;
        opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
        opts.step_controller = makeFixedVsvoBdfController(order, dt);
        opts.newton.residual_op = "op";
        opts.newton.jacobian_op = "op";
        opts.newton.max_iterations = 8;
        opts.newton.abs_tolerance = 1e-12;
        opts.newton.rel_tolerance = 0.0;

        svmp::FE::timestepping::TimeLoop loop(opts);
        const auto rep = loop.run(transient, *factory, *linear, history);
        EXPECT_TRUE(rep.success) << rep.message;
        EXPECT_NEAR(rep.final_time, t1, 1e-12) << rep.message;
    };

    // Continuous run.
    auto history_full = makeHistory(dt);
    run(/*t0=*/0.0, /*t1=*/t_end, history_full);
    const auto x_full = getVectorByDof(history_full.uPrev());

    // Restarted run: split at t_split and copy state into a fresh TimeHistory.
    auto history_part1 = makeHistory(dt);
    run(/*t0=*/0.0, /*t1=*/t_split, history_part1);

    auto history_restart = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/order + 1);
    for (int k = 1; k <= history_restart.historyDepth(); ++k) {
        auto dst = history_restart.uPrevK(k).localSpan();
        auto src = history_part1.uPrevK(k).localSpan();
        ASSERT_EQ(dst.size(), src.size());
        std::copy(src.begin(), src.end(), dst.begin());
    }
    history_restart.resetCurrentToPrevious();
    history_restart.setStepIndex(history_part1.stepIndex());
    history_restart.setTime(history_part1.time());
    history_restart.setPrevDt(history_part1.dtPrev());
    history_restart.setDtHistory(std::vector<double>(history_part1.dtHistory().begin(), history_part1.dtHistory().end()));

    run(/*t0=*/t_split, /*t1=*/t_end, history_restart);
    const auto x_restart = getVectorByDof(history_restart.uPrev());

    ASSERT_EQ(x_full.size(), x_restart.size());
    for (std::size_t i = 0; i < x_full.size(); ++i) {
        EXPECT_NEAR(x_full[i], x_restart[i], 1e-12);
    }

    const Real scale = static_cast<Real>(std::exp(-lambda * t_end));
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * static_cast<double>(scale));
    }

    const double err_full = relativeL2Error(x_full, exact);
    const double err_restart = relativeL2Error(x_restart, exact);
    EXPECT_LT(err_full, 1e-5);
    EXPECT_LT(err_restart, 1e-5);
}

TEST(TimeLoopVSVO_BDF, LTENormScalesLikeDtSquaredForOrder1OnReaction)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double lambda = 1.0;
    const double t_end = 0.2;

    auto runAndGetLastErrorNorm = [&](double dt) -> double {
        svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
        ctrl_opts.abs_tol = 1.0;
        ctrl_opts.rel_tol = 0.0;
        ctrl_opts.min_order = 1;
        ctrl_opts.max_order = 1;
        ctrl_opts.initial_order = 1;
        ctrl_opts.max_retries = 0;
        ctrl_opts.safety = 1.0;
        ctrl_opts.min_factor = 1.0;
        ctrl_opts.max_factor = 1.0;
        ctrl_opts.min_dt = dt;
        ctrl_opts.max_dt = dt;
        ctrl_opts.pi_alpha = 0.0;
	        ctrl_opts.pi_beta = 0.0;
	        ctrl_opts.increase_order_threshold = 0.0;

	        auto controller = std::make_shared<RecordingVsvoBdfController>(ctrl_opts);
	        if (!controller) {
	            return -1.0;
	        }

        (void)runReactionProblem(svmp::FE::timestepping::SchemeKind::VSVO_BDF,
                                 dt,
                                 t_end,
                                 lambda,
                                 /*history_depth=*/2,
                                 controller,
                                 /*generalized_alpha_rho_inf=*/1.0,
                                 /*dg_degree=*/1,
                                 /*cg_degree=*/2,
                                 svmp::FE::timestepping::CollocationSolveStrategy::Monolithic,
                                 /*collocation_max_outer_iterations=*/4,
                                 /*collocation_outer_tolerance=*/0.0,
                                 /*exact_initial_history=*/true,
                                 /*theta=*/0.5);

        double err = -1.0;
        for (const auto& info : controller->accepted) {
            const double t1 = info.time + info.dt;
            if (std::abs(t1 - t_end) <= 1e-12 && (info.error_norm > 0.0) && std::isfinite(info.error_norm)) {
                err = info.error_norm;
            }
        }
        return err;
    };

    const double err_dt1 = runAndGetLastErrorNorm(/*dt=*/0.1);
    const double err_dt2 = runAndGetLastErrorNorm(/*dt=*/0.05);
    ASSERT_GT(err_dt1, 0.0);
    ASSERT_GT(err_dt2, 0.0);

    const double ratio = err_dt1 / err_dt2;
    EXPECT_NEAR(ratio, 4.0, 1.0);
}

TEST(TimeLoopVSVO_BDF, AdaptsDtOnDt2Oscillator)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    const double omega = 2.0;
    const double t_end = 1.0;
    const double dt0 = 0.25;

    svmp::FE::timestepping::VSVO_BDF_ControllerOptions ctrl_opts;
    ctrl_opts.abs_tol = 1e-4;
    ctrl_opts.rel_tol = 1e-3;
    ctrl_opts.min_order = 1;
    ctrl_opts.max_order = 3;
    ctrl_opts.initial_order = 1;
    ctrl_opts.max_retries = 12;
    ctrl_opts.safety = 0.9;
    ctrl_opts.min_factor = 0.2;
    ctrl_opts.max_factor = 2.0;
    ctrl_opts.increase_order_threshold = 0.05;

    auto controller = std::make_shared<svmp::FE::timestepping::VSVO_BDF_Controller>(ctrl_opts);
    ASSERT_NE(controller.get(), nullptr);

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 n_dofs,
                                                                 /*history_depth=*/5,
                                                                 /*allocate_second_order_state=*/true);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    ASSERT_EQ(u0.size(), static_cast<std::size_t>(n_dofs));

    // Exact solution with v0 = omega*u0 => u(t)=u0*(cos(omega t)+sin(omega t)).
    std::vector<Real> v0(u0.size(), 0.0);
    std::vector<Real> a0(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        v0[i] = static_cast<Real>(omega * static_cast<double>(u0[i]));
        a0[i] = static_cast<Real>(-omega * omega * static_cast<double>(u0[i]));
    }
    setVectorByDof(history.uDot(), v0);
    setVectorByDof(history.uDDot(), a0);

    for (int k = 1; k <= history.historyDepth(); ++k) {
        const double t = -static_cast<double>(k - 1) * dt0;
        const double c = std::cos(omega * t);
        const double s = std::sin(omega * t);
        std::vector<Real> uk(u0.size(), 0.0);
        for (std::size_t i = 0; i < u0.size(); ++i) {
            uk[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c + s));
        }
        setVectorByDof(history.uPrevK(k), uk);
    }
    history.resetCurrentToPrevious();
    history.setPrevDt(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = t_end;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::VSVO_BDF;
    opts.step_controller = controller;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 10;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoopCallbacks callbacks;
    std::vector<std::pair<double, double>> dt_updates;
    callbacks.on_dt_updated = [&dt_updates](double oldv, double newv, int, int) {
        dt_updates.emplace_back(oldv, newv);
    };

	    svmp::FE::timestepping::TimeLoop loop(opts);
	    const auto rep = loop.run(transient, *factory, *linear, history, callbacks);
	    if (!rep.success) {
	        double dt_min = dt0;
	        double dt_max = dt0;
	        for (const auto& p : dt_updates) {
	            dt_min = std::min(dt_min, std::min(p.first, p.second));
	            dt_max = std::max(dt_max, std::max(p.first, p.second));
	        }
	        std::cerr << "TimeLoopVSVO_BDF.AdaptsDtOnDt2Oscillator diagnostics: "
	                  << "steps_taken=" << rep.steps_taken
	                  << " final_time=" << rep.final_time
	                  << " dt_prev=" << history.dtPrev()
	                  << " dt_updates=" << dt_updates.size()
	                  << " dt_min=" << dt_min
	                  << " dt_max=" << dt_max
	                  << std::endl;
	    }
	    EXPECT_TRUE(rep.success) << rep.message;
	    EXPECT_NEAR(rep.final_time, t_end, 1e-12) << rep.message;
	    EXPECT_FALSE(dt_updates.empty());

    const double c_end = std::cos(omega * t_end);
    const double s_end = std::sin(omega * t_end);
    std::vector<Real> exact(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        exact[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c_end + s_end));
    }
    const auto approx = getVectorByDof(history.uPrev());
    const double err = relativeL2Error(approx, exact);
    EXPECT_LT(err, 0.1);
}

TEST(TimeLoopTRBDF2, RestoresHistoryOnThrownExceptionInStage2)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double lambda = 2.0;
    constexpr double dt0 = 0.2;

    class FailOnNthSolveLinearSolver final : public svmp::FE::backends::LinearSolver {
    public:
        FailOnNthSolveLinearSolver(svmp::FE::backends::LinearSolver& inner, int fail_on_call)
            : inner_(inner)
            , fail_on_call_(fail_on_call)
        {
        }

        [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
        {
            return inner_.backendKind();
        }

        void setOptions(const svmp::FE::backends::SolverOptions& options) override
        {
            inner_.setOptions(options);
        }

        [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
        {
            return inner_.getOptions();
        }

        [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix& A,
                                                              svmp::FE::backends::GenericVector& x,
                                                              const svmp::FE::backends::GenericVector& b) override
        {
            calls_ += 1;
            if (calls_ == fail_on_call_) {
                svmp::FE::backends::SolverReport rep;
                rep.converged = false;
                rep.iterations = 0;
                rep.message = "forced failure for TRBDF2 restore test";
                return rep;
            }
            return inner_.solve(A, x, b);
        }

    private:
        svmp::FE::backends::LinearSolver& inner_;
        int fail_on_call_{1};
        int calls_{0};
    };

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form = (svmp::FE::forms::dt(u) * v + (u * v) * static_cast<Real>(lambda)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto inner = factory->createLinearSolver(directSolve());
    ASSERT_NE(inner.get(), nullptr);
    FailOnNthSolveLinearSolver linear(*inner, /*fail_on_call=*/2);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory, n_dofs, /*history_depth=*/2);
    const std::vector<Real> u_prev = {1.0, -2.0, 3.0, -4.0};
    const std::vector<Real> u_prev2 = {-4.0, 3.0, -2.0, 1.0};
    setVectorByDof(history.uPrev(), u_prev);
    setVectorByDof(history.uPrev2(), u_prev2);
    history.resetCurrentToPrevious();
    history.setDt(dt0);
    history.setPrevDt(dt0);
    history.primeDtHistory(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt0;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::TRBDF2;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 8;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    EXPECT_THROW((void)loop.run(transient, *factory, linear, history), svmp::FE::FEException);

    // Ensure TRBDF2 substep manipulations didn't leak into user-visible history.
    EXPECT_EQ(getVectorByDof(history.uPrev()), u_prev);
    EXPECT_EQ(getVectorByDof(history.uPrev2()), u_prev2);
    EXPECT_EQ(getVectorByDof(history.u()), u_prev);
    EXPECT_NEAR(history.dt(), dt0, 1e-15);
    EXPECT_NEAR(history.dtPrev(), dt0, 1e-15);
}

TEST(TimeLoopSecondOrderInit, NewmarkInitializesVelocityAndFallsBackForAcceleration)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double omega = 2.0;
    constexpr double dt0 = 0.1;

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto linear = factory->createLinearSolver(directSolve());
    ASSERT_NE(linear.get(), nullptr);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 n_dofs,
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/false);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    ASSERT_EQ(u0.size(), static_cast<std::size_t>(n_dofs));

    // u(t) = u0 * (cos(omega t) + sin(omega t)) => u(-dt) = u0*(cos(omega dt) - sin(omega dt)).
    const double c = std::cos(omega * dt0);
    const double s = std::sin(omega * dt0);
    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c - s));
    }

    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();
    history.setDt(dt0);
    history.setPrevDt(dt0);
    history.primeDtHistory(dt0);

    std::vector<Real> init_v;
    std::vector<Real> init_a;
    int nonlinear_calls = 0;

    svmp::FE::timestepping::TimeLoopCallbacks cb;
    cb.on_nonlinear_done = [&init_v, &init_a, &nonlinear_calls](const svmp::FE::timestepping::TimeHistory& h,
                                                                const svmp::FE::timestepping::NewtonReport&) {
        nonlinear_calls += 1;
        if (nonlinear_calls > 1) {
            return;
        }
        const auto v = h.uDotSpan();
        const auto a = h.uDDotSpan();
        init_v.assign(v.begin(), v.end());
        init_a.assign(a.begin(), a.end());
    };

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt0;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::Newmark;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    const auto rep = loop.run(transient, *factory, *linear, history, cb);
    EXPECT_TRUE(rep.success) << rep.message;
    EXPECT_GE(nonlinear_calls, 1);

    ASSERT_EQ(init_v.size(), u0.size());
    ASSERT_EQ(init_a.size(), u0.size());

    for (std::size_t i = 0; i < u0.size(); ++i) {
        const double expected_v = (static_cast<double>(u0[i]) - static_cast<double>(u_minus_dt[i])) / dt0;
        const double expected_a = -omega * omega * static_cast<double>(u0[i]);
        EXPECT_NEAR(static_cast<double>(init_v[i]), expected_v, 1e-12);
        EXPECT_NEAR(static_cast<double>(init_a[i]), expected_a, 1e-10);
    }
}

TEST(TimeLoopSecondOrderInit, NewmarkThrowsWhenAccelerationFallbackSolveFails)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeStepping tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    constexpr double omega = 2.0;
    constexpr double dt0 = 0.1;

    class AlwaysFailLinearSolver final : public svmp::FE::backends::LinearSolver {
    public:
        explicit AlwaysFailLinearSolver(svmp::FE::backends::LinearSolver& inner)
            : inner_(inner)
        {
        }

        [[nodiscard]] svmp::FE::backends::BackendKind backendKind() const noexcept override
        {
            return inner_.backendKind();
        }

        void setOptions(const svmp::FE::backends::SolverOptions& options) override
        {
            inner_.setOptions(options);
        }

        [[nodiscard]] const svmp::FE::backends::SolverOptions& getOptions() const noexcept override
        {
            return inner_.getOptions();
        }

        [[nodiscard]] svmp::FE::backends::SolverReport solve(const svmp::FE::backends::GenericMatrix&,
                                                              svmp::FE::backends::GenericVector&,
                                                              const svmp::FE::backends::GenericVector&) override
        {
            svmp::FE::backends::SolverReport rep;
            rep.converged = false;
            rep.iterations = 0;
            rep.message = "forced failure";
            return rep;
        }

    private:
        svmp::FE::backends::LinearSolver& inner_;
    };

    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    const auto form =
        (svmp::FE::forms::dt(u, 2) * v + (u * v) * static_cast<Real>(omega * omega)).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto ir = compiler.compileResidual(form);
    auto kernel = std::make_shared<svmp::FE::forms::NonlinearFormKernel>(std::move(ir), svmp::FE::forms::ADMode::Forward);
    sys.addCellKernel("op", u_field, u_field, kernel);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    svmp::FE::systems::TransientSystem transient(sys, integrator);

    auto factory = createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto inner = factory->createLinearSolver(directSolve());
    ASSERT_NE(inner.get(), nullptr);
    AlwaysFailLinearSolver linear(*inner);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 n_dofs,
                                                                 /*history_depth=*/2,
                                                                 /*allocate_second_order_state=*/false);

    const std::vector<Real> u0 = {1.0, 2.0, -0.5, 0.25};
    ASSERT_EQ(u0.size(), static_cast<std::size_t>(n_dofs));

    const double c = std::cos(omega * dt0);
    const double s = std::sin(omega * dt0);
    std::vector<Real> u_minus_dt(u0.size(), 0.0);
    for (std::size_t i = 0; i < u0.size(); ++i) {
        u_minus_dt[i] = static_cast<Real>(static_cast<double>(u0[i]) * (c - s));
    }

    setVectorByDof(history.uPrev(), u0);
    setVectorByDof(history.uPrev2(), u_minus_dt);
    history.resetCurrentToPrevious();
    history.setDt(dt0);
    history.setPrevDt(dt0);
    history.primeDtHistory(dt0);

    svmp::FE::timestepping::TimeLoopOptions opts;
    opts.t0 = 0.0;
    opts.t_end = dt0;
    opts.dt = dt0;
    opts.scheme = svmp::FE::timestepping::SchemeKind::Newmark;
    opts.newton.residual_op = "op";
    opts.newton.jacobian_op = "op";
    opts.newton.max_iterations = 12;
    opts.newton.abs_tolerance = 1e-12;
    opts.newton.rel_tolerance = 0.0;

    svmp::FE::timestepping::TimeLoop loop(opts);
    EXPECT_THROW((void)loop.run(transient, *factory, linear, history), svmp::FE::FEException);
}
