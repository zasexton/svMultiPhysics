/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_SYSTEMS_GENERALITY_BENCHMARK_H
#define SVMP_FE_TESTS_UNIT_SYSTEMS_GENERALITY_BENCHMARK_H

/**
 * @file GeneralityBenchmark.h
 * @brief Fixtures of the generality benchmark: meshes of every first-order
 * cell family, a steady Newton solve and measured errors
 *
 * The benchmark problems (test_GeneralityBenchmark.cpp and
 * test_GeneralityBenchmarkGaps.cpp) are written as an author writes a
 * formulation; these fixtures are what a test needs around them.
 */

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"
#include "Assembly/GlobalSystemView.h"
#include "Core/Types.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/FormulationBuilder.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"
#include "TimeStepping/GeneratedState.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/StepController.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::systems::generality {

/// The first-order cell families of the benchmark.
enum class Cell { Tri3, Quad4, Tet4, Hex8, Wedge6, Pyr5 };

inline const char* cellName(Cell cell)
{
    switch (cell) {
    case Cell::Tri3: return "Tri3";
    case Cell::Quad4: return "Quad4";
    case Cell::Tet4: return "Tet4";
    case Cell::Hex8: return "Hex8";
    case Cell::Wedge6: return "Wedge6";
    case Cell::Pyr5: return "Pyr5";
    }
    return "?";
}

inline ElementType elementType(Cell cell)
{
    switch (cell) {
    case Cell::Tri3: return ElementType::Triangle3;
    case Cell::Quad4: return ElementType::Quad4;
    case Cell::Tet4: return ElementType::Tetra4;
    case Cell::Hex8: return ElementType::Hex8;
    case Cell::Wedge6: return ElementType::Wedge6;
    case Cell::Pyr5: return ElementType::Pyramid5;
    }
    return ElementType::Unknown;
}

inline int cellDimension(Cell cell)
{
    return (cell == Cell::Tri3 || cell == Cell::Quad4) ? 2 : 3;
}

using Point = std::array<Real, 3>;
/// The label of a boundary face from its centre (0 or less: unlabelled).
using FaceLabel = std::function<int(const Point&)>;
/// The region (domain id) of a cell from its centre.
using CellRegion = std::function<int(const Point&)>;

/// The faces of the box [lo, hi]: x = lo 1, x = hi 2, y = lo 3, y = hi 4, z = lo 5, z = hi 6.
inline FaceLabel boxSides(const Point& lo = {0.0, 0.0, 0.0}, const Point& hi = {1.0, 1.0, 1.0})
{
    return [lo, hi](const Point& c) {
        for (int d = 0; d < 3; ++d) {
            const auto i = static_cast<std::size_t>(d);
            const Real tol = 1e-9 * (1.0 + std::abs(hi[i] - lo[i]));
            if (std::abs(c[i] - lo[i]) < tol) {
                return 2 * d + 1;
            }
            if (std::abs(c[i] - hi[i]) < tol) {
                return 2 * d + 2;
            }
        }
        return 0;
    };
}

namespace detail {

inline svmp::CellShape shapeOf(svmp::CellFamily family, int corners)
{
    svmp::CellShape shape{};
    shape.family = family;
    shape.num_corners = corners;
    shape.order = 1;
    return shape;
}

inline Real signedVolume(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
{
    const Point u{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const Point v{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    const Point w{p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
    return (u[1] * v[2] - u[2] * v[1]) * w[0] + (u[2] * v[0] - u[0] * v[2]) * w[1] +
           (u[0] * v[1] - u[1] * v[0]) * w[2];
}

} // namespace detail

/**
 * The box [lo, hi] (a rectangle for Tri3 and Quad4) in n cells per axis
 * (nz = 0: n along z as well), each square or cube split into cells of the
 * family: two Tri3 along the (i, j)-(i+1, j+1) diagonal, six Tet4 around the
 * main diagonal of the cube (Kuhn), two Wedge6 along the xy diagonal, or six
 * Pyr5 around a centre vertex.  Every split is conforming.  Boundary faces
 * take the label @p label gives their centre, cells the region the options
 * give theirs; the options can also collect interior faces in the face set
 * "gamma", turn each cell's vertex list (Tri3 and Tet4: a 3-cycle of the
 * first three vertices; Quad4: a quarter turn; Hex8, Wedge6: the bottom and
 * top faces turned together), or number the cells in reverse, which changes
 * the local orientations and the minus/plus sides of the interior faces but
 * not the discrete space.
 */
struct BoxOptions {
    CellRegion region{};             ///< the domain id of each cell (none: unset)
    Point lo{0.0, 0.0, 0.0};
    Point hi{1.0, 1.0, 1.0};
    int nz{0};                       ///< cells along z (0: n)
    /// Interior faces whose centre it accepts form the face set "gamma".
    std::function<bool(const Point&)> interface_faces{};
    bool rotate_cells{false};        ///< every cell's vertex list turned (an even permutation)
    bool reverse_cells{false};       ///< the cells numbered in reverse order
};

inline std::shared_ptr<svmp::Mesh> boxMesh(Cell cell, int n, const FaceLabel& label, const BoxOptions& options = {})
{
    const auto& region = options.region;
    const auto& lo = options.lo;
    const auto& hi = options.hi;
    const int nz = options.nz;
    const int dim = cellDimension(cell);
    const int nx = n;
    const int ny = n;
    const int nzz = dim == 3 ? (nz > 0 ? nz : n) : 0;
    std::vector<Real> x;
    const auto vid = [&](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * (ny + 1) + j) * (nx + 1) + i);
    };
    for (int k = 0; k <= nzz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                x.push_back(lo[0] + (hi[0] - lo[0]) * static_cast<Real>(i) / static_cast<Real>(nx));
                x.push_back(lo[1] + (hi[1] - lo[1]) * static_cast<Real>(j) / static_cast<Real>(ny));
                if (dim == 3) {
                    x.push_back(lo[2] + (hi[2] - lo[2]) * static_cast<Real>(k) / static_cast<Real>(nzz));
                }
            }
        }
    }
    const auto point = [&](svmp::index_t v) {
        const auto o = static_cast<std::size_t>(dim) * static_cast<std::size_t>(v);
        return Point{x[o], x[o + 1], dim == 3 ? x[o + 2] : 0.0};
    };
    std::vector<std::vector<svmp::index_t>> cells;
    std::vector<svmp::CellShape> shapes;
    const auto add = [&](std::vector<svmp::index_t> c, svmp::CellFamily family) {
        shapes.push_back(detail::shapeOf(family, static_cast<int>(c.size())));
        cells.push_back(std::move(c));
    };
    const auto addTet = [&](svmp::index_t a, svmp::index_t b, svmp::index_t c, svmp::index_t d) {
        if (detail::signedVolume(point(a), point(b), point(c), point(d)) < 0.0) {
            std::swap(b, c);
        }
        add({a, b, c, d}, svmp::CellFamily::Tetra);
    };
    if (dim == 2) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const auto a = vid(i, j, 0), b = vid(i + 1, j, 0), c = vid(i + 1, j + 1, 0), d = vid(i, j + 1, 0);
                if (cell == Cell::Quad4) {
                    add({a, b, c, d}, svmp::CellFamily::Quad);
                } else {
                    add({a, b, c}, svmp::CellFamily::Triangle);
                    add({a, c, d}, svmp::CellFamily::Triangle);
                }
            }
        }
    } else {
        for (int k = 0; k < nzz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const auto a = vid(i, j, k), b = vid(i + 1, j, k), c = vid(i + 1, j + 1, k), d = vid(i, j + 1, k);
                    const auto e = vid(i, j, k + 1), f = vid(i + 1, j, k + 1), g = vid(i + 1, j + 1, k + 1),
                               h = vid(i, j + 1, k + 1);
                    switch (cell) {
                    case Cell::Hex8:
                        add({a, b, c, d, e, f, g, h}, svmp::CellFamily::Hex);
                        break;
                    case Cell::Tet4:
                        // Kuhn: the six paths from a to g along the cube's edges.
                        addTet(a, b, c, g);
                        addTet(a, c, d, g);
                        addTet(a, d, h, g);
                        addTet(a, h, e, g);
                        addTet(a, e, f, g);
                        addTet(a, f, b, g);
                        break;
                    case Cell::Wedge6:
                        add({a, b, c, e, f, g}, svmp::CellFamily::Wedge);
                        add({a, c, d, e, g, h}, svmp::CellFamily::Wedge);
                        break;
                    case Cell::Pyr5: {
                        const auto centre = static_cast<svmp::index_t>(x.size() / 3u);
                        const Point pa = point(a), pg = point(g);
                        x.insert(x.end(), {0.5 * (pa[0] + pg[0]), 0.5 * (pa[1] + pg[1]), 0.5 * (pa[2] + pg[2])});
                        const std::array<std::array<svmp::index_t, 4>, 6> faces{{
                            {a, b, c, d}, {e, f, g, h}, {a, b, f, e}, {b, c, g, f}, {c, d, h, g}, {d, a, e, h}}};
                        for (auto face : faces) {
                            if (detail::signedVolume(point(face[0]), point(face[1]), point(face[2]), point(centre)) <
                                0.0) {
                                std::swap(face[1], face[3]);
                            }
                            add({face[0], face[1], face[2], face[3], centre}, svmp::CellFamily::Pyramid);
                        }
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
        }
    }
    if (options.rotate_cells) {
        for (auto& c : cells) {
            switch (cell) {
            case Cell::Tri3:
            case Cell::Tet4: std::rotate(c.begin(), c.begin() + 1, c.begin() + 3); break;
            case Cell::Quad4: std::rotate(c.begin(), c.begin() + 1, c.end()); break;
            case Cell::Hex8:
                std::rotate(c.begin(), c.begin() + 1, c.begin() + 4);
                std::rotate(c.begin() + 4, c.begin() + 5, c.end());
                break;
            case Cell::Wedge6:
                std::rotate(c.begin(), c.begin() + 1, c.begin() + 3);
                std::rotate(c.begin() + 3, c.begin() + 4, c.end());
                break;
            case Cell::Pyr5: std::rotate(c.begin(), c.begin() + 1, c.begin() + 4); break;
            }
        }
    }
    if (options.reverse_cells) {
        std::reverse(cells.begin(), cells.end());
        std::reverse(shapes.begin(), shapes.end());
    }
    auto base = std::make_shared<svmp::MeshBase>();
    std::vector<svmp::real_t> coordinates(x.begin(), x.end());
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> connectivity;
    for (const auto& c : cells) {
        connectivity.insert(connectivity.end(), c.begin(), c.end());
        offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
    }
    base->build_from_arrays(dim, coordinates, offsets, connectivity, shapes);
    if (region) {
        for (std::size_t c = 0; c < cells.size(); ++c) {
            Point centre{0.0, 0.0, 0.0};
            for (const auto v : cells[c]) {
                const auto p = point(v);
                for (std::size_t d = 0; d < 3; ++d) {
                    centre[d] += p[d] / static_cast<Real>(cells[c].size());
                }
            }
            base->set_region_label(static_cast<svmp::index_t>(c), region(centre));
        }
    }
    base->finalize();
    for (const auto face : base->boundary_faces()) {
        const auto c = base->face_center(face);
        const int l = label(Point{c[0], c[1], c[2]});
        if (l > 0) {
            base->set_boundary_label(face, l);
        }
    }
    if (options.interface_faces) {
        for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
            if (base->face_cells(f)[1] == svmp::INVALID_INDEX) {
                continue;
            }
            const auto c = base->face_center(f);
            if (options.interface_faces(Point{c[0], c[1], c[2]})) {
                base->add_to_set(svmp::EntityKind::Face, "gamma", f);
            }
        }
    }
    return svmp::create_mesh(std::move(base));
}

/// The state of a solved system: the DOF vector and a view of it.
struct Solution {
    std::vector<Real> u;
    bool converged{false};
    int iterations{0};

    [[nodiscard]] SystemStateView state() const
    {
        SystemStateView s;
        s.u = u;
        return s;
    }
};

/// The Eigen direct solver, every Newton step a sparse LU solve.
inline backends::SolverOptions directSolverOptions()
{
    backends::SolverOptions direct;
    direct.method = backends::SolverMethod::Direct;
    direct.preconditioner = backends::PreconditionerType::None;
    direct.rel_tol = 1e-14;
    direct.abs_tol = 1e-14;
    direct.max_iter = 1;
    return direct;
}

/**
 * A steady Newton solve of operator @p op from the constrained zero state
 * (the essential values distributed), with the Eigen direct solver: one
 * step for a linear problem.
 */
inline Solution solveSteady(FESystem& sys,
                            const OperatorTag& op,
                            int max_iterations = 8,
                            Real abs_tolerance = 1e-11,
                            Real rel_tolerance = 1e-12)
{
    Solution out;
    auto factory = backends::BackendFactory::create(backends::BackendKind::Eigen);
    EXPECT_NE(factory, nullptr);
    if (!factory) {
        return out;
    }
    auto linear = factory->createLinearSolver(directSolverOptions());
    TransientSystem transient(sys, std::make_shared<ZeroTimeDerivativeIntegrator>());
    const auto n = sys.dofHandler().getNumDofs();
    auto history = timestepping::TimeHistory::allocate(*factory, n);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.u().zero();
    sys.constraints().distribute(history.u());
    history.u().updateGhosts();
    timestepping::NewtonOptions options;
    options.residual_op = op;
    options.jacobian_op = op;
    options.max_iterations = max_iterations;
    options.abs_tolerance = abs_tolerance;
    options.rel_tolerance = rel_tolerance;
    timestepping::NewtonSolver newton(options);
    timestepping::NewtonWorkspace workspace;
    newton.allocateWorkspace(sys, *factory, workspace);
    const auto report = newton.solveStep(transient, *linear, /*solve_time=*/0.0, history, workspace);
    out.converged = report.converged;
    out.iterations = report.iterations;
    const auto span = history.uSpan();
    out.u.assign(span.begin(), span.end());
    return out;
}

/// The Jacobian and residual of operator @p op at @p u, dense (row-major).
struct DenseOperator {
    GlobalIndex n{0};
    std::vector<Real> matrix;
    std::vector<Real> vector;

    [[nodiscard]] Real operator()(GlobalIndex i, GlobalIndex j) const
    {
        return matrix[static_cast<std::size_t>(i * n + j)];
    }
};

inline DenseOperator assembleDense(FESystem& sys, const OperatorTag& op, std::span<const Real> u)
{
    DenseOperator out;
    out.n = sys.dofHandler().getNumDofs();
    assembly::DenseSystemView view(out.n);
    view.zero();
    SystemStateView state;
    state.u = u;
    AssemblyRequest request;
    request.op = op;
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;  // the residual at the state, as Newton assembles it
    EXPECT_TRUE(sys.assemble(request, state, &view, &view).success);
    out.matrix.assign(view.matrixData().begin(), view.matrixData().end());
    out.vector.assign(view.vectorData().begin(), view.vectorData().end());
    return out;
}

/// The residual of operator @p op at @p u (no matrix).
inline std::vector<Real> assembleVector(FESystem& sys, const OperatorTag& op, std::span<const Real> u)
{
    const auto n = sys.dofHandler().getNumDofs();
    assembly::DenseVectorView view(n);
    view.zero();
    SystemStateView state;
    state.u = u;
    AssemblyRequest request;
    request.op = op;
    request.want_matrix = false;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;  // the residual at the state, as Newton assembles it
    EXPECT_TRUE(sys.assemble(request, state, nullptr, &view).success);
    std::vector<Real> out(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = view.getVectorEntry(i);
    }
    return out;
}

/**
 * The integral of a scalar integrand through the rows of a scalar
 * discontinuous unknown @p cells whose basis functions sum to one on every
 * cell (DG0 or nodal DG1): the measurement operator @p op holds
 * (integrand w).dx() with w the test function of @p cells, and the sum of
 * its rows in @p cells is the integral.  The measured-form evaluator does
 * not take integrands of H(div) and H(curl) fields; assembly does.
 */
inline void declareRowIntegral(FESystem& sys, const OperatorTag& op, const forms::FormExpr& integrand, FieldId cells)
{
    FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    FormulationBuilder F(sys, op, measurement);
    (void)F.residual((integrand * forms::TestField(cells, *sys.fieldRecord(cells).space, "w_cells")).dx());
}

inline Real rowIntegral(FESystem& sys, const OperatorTag& op, FieldId cells, std::span<const Real> u)
{
    const auto rows = assembleVector(sys, op, u);
    const auto offset = sys.fieldDofOffset(cells);
    const auto count = sys.fieldDofHandler(cells).getNumDofs();
    Real sum = 0.0;
    for (GlobalIndex i = offset; i < offset + count; ++i) {
        sum += rows[static_cast<std::size_t>(i)];
    }
    return sum;
}

/// The accepted states of a fixed-step transient run.
struct TransientRun {
    std::vector<std::vector<Real>> states;  ///< after each accepted step
    std::vector<double> times;
    std::vector<int> iterations;            ///< of every converged attempt
    int rejections{0};
    bool success{false};
};

/// Retries a rejected step with the same dt and keeps dt on acceptance.
class SameStepController final : public timestepping::StepController {
public:
    [[nodiscard]] int maxRetries() const noexcept override { return 2; }
    [[nodiscard]] timestepping::StepDecision onAccepted(const timestepping::StepAttemptInfo& info) override
    {
        return {.accept = true, .retry = false, .next_dt = info.dt};
    }
    [[nodiscard]] timestepping::StepDecision onRejected(const timestepping::StepAttemptInfo& info,
                                                        timestepping::StepRejectReason) override
    {
        return {.accept = false, .retry = true, .next_dt = info.dt};
    }
};

struct TransientOptions {
    double t0{0.0};
    std::vector<int> reject_once{};  ///< accepted-step indices whose first attempt is rejected
    int max_iterations{8};
    timestepping::TimeLoopCallbacks callbacks{};
};

/**
 * Operator @p op from the initial state @p u0 (its history filled with it)
 * over @p steps steps of @p dt under @p scheme, with the Eigen direct
 * solver; the shadow fields (FieldBinding::previous, lagged) are refreshed
 * by their generated-state participant.
 */
inline TransientRun runTransient(FESystem& sys,
                                 const OperatorTag& op,
                                 std::span<const Real> u0,
                                 timestepping::SchemeKind scheme,
                                 double dt,
                                 int steps,
                                 TransientOptions run_options = {})
{
    TransientRun out;
    auto factory = backends::BackendFactory::create(backends::BackendKind::Eigen);
    EXPECT_NE(factory, nullptr);
    if (!factory) {
        return out;
    }
    auto linear = factory->createLinearSolver(directSolverOptions());
    TransientSystem transient(sys, std::make_shared<BackwardDifferenceIntegrator>());
    const auto n = sys.dofHandler().getNumDofs();
    auto history = timestepping::TimeHistory::allocate(*factory, n, 2);
    for (int k = 1; k <= history.historyDepth(); ++k) {
        auto view = history.uPrevK(k).createAssemblyView();
        view->beginAssemblyPhase();
        for (GlobalIndex i = 0; i < n; ++i) {
            view->addVectorEntry(i, u0[static_cast<std::size_t>(i)], assembly::AddMode::Insert);
        }
        view->finalizeAssembly();
    }
    history.resetCurrentToPrevious();
    history.setTime(run_options.t0);
    history.setPrevDt(dt);

    timestepping::TimeLoopOptions options;
    options.t0 = run_options.t0;
    options.t_end = run_options.t0 + dt * steps;
    options.dt = dt;
    options.max_steps = 4 * steps;
    options.scheme = scheme;
    options.newton.residual_op = op;
    options.newton.jacobian_op = op;
    options.newton.max_iterations = run_options.max_iterations;
    options.newton.abs_tolerance = 1e-11;
    options.newton.rel_tolerance = 1e-12;
    auto generated_state = std::make_shared<timestepping::GeneratedStateRegistry>(
        timestepping::GeneratedStatePolicy{.solve = timestepping::GeneratedStateSolve::Transient,
                                           .refresh_within_solve = true});
    if (!shadowFields(sys).empty()) {
        generated_state->add(timestepping::makeShadowFieldParticipant(sys));
        generated_state->bind(options.newton);
    }
    if (!run_options.reject_once.empty()) {
        options.step_controller = std::make_shared<SameStepController>();
    }
    timestepping::TimeLoop loop(options);
    auto callbacks = std::move(run_options.callbacks);
    auto user_done = callbacks.on_nonlinear_done;
    callbacks.on_nonlinear_done = [&, user_done](const timestepping::TimeHistory& h,
                                                 const timestepping::NewtonReport& report) {
        EXPECT_TRUE(report.converged);
        out.iterations.push_back(report.iterations);
        if (user_done) {
            user_done(h, report);
        }
    };
    auto user_accepted = callbacks.on_step_accepted;
    callbacks.on_step_accepted = [&, user_accepted](timestepping::TimeHistory& h) {
        const auto u = h.uSpan();
        out.states.emplace_back(u.begin(), u.end());
        out.times.push_back(h.time());
        if (user_accepted) {
            user_accepted(h);
        }
    };
    if (!run_options.reject_once.empty()) {
        const auto reject = run_options.reject_once;
        auto attempted = std::make_shared<std::vector<int>>();
        callbacks.on_before_step_accept = [&out, reject, attempted](timestepping::TimeHistory&,
                                                                    const timestepping::NewtonReport&) {
            const int step = static_cast<int>(out.states.size());
            if (std::find(reject.begin(), reject.end(), step) != reject.end() &&
                std::find(attempted->begin(), attempted->end(), step) == attempted->end()) {
                attempted->push_back(step);
                ++out.rejections;
                return false;
            }
            return true;
        };
    }
    const auto report = loop.run(transient, *factory, *linear, history, callbacks);
    out.success = report.success;
    return out;
}

/// The value of the measured 0-form @p form at the state @p u (at @p time).
inline Real integrate(const FESystem& sys, const forms::FormExpr& form, std::span<const Real> u, double time = 0.0)
{
    SystemStateView state;
    state.u = u;
    state.time = time;
    const std::array<forms::FormExpr, 1> forms_{form};
    return evaluateFunctionalForms(sys, forms_, state, CollectiveCommunicator{}).front().value;
}

/// sqrt of integrate(): the norm of a measured square.
inline Real norm(const FESystem& sys, const forms::FormExpr& square, std::span<const Real> u)
{
    return std::sqrt(std::max(Real(0.0), integrate(sys, square, u)));
}

/**
 * The largest |J - J_fd| over the free rows and columns, J the assembled
 * Jacobian of @p op at @p u and J_fd central differences of its residual
 * with step @p eps, with the scale max |J| of the same entries.
 */
inline std::pair<Real, Real> jacobianDefect(FESystem& sys, const OperatorTag& op, std::span<const Real> u,
                                            Real eps = 1e-6)
{
    const auto J = assembleDense(sys, op, u);
    std::vector<Real> plus(u.begin(), u.end());
    std::vector<Real> minus(u.begin(), u.end());
    Real defect = 0.0;
    Real scale = 0.0;
    for (GlobalIndex j = 0; j < J.n; ++j) {
        if (sys.constraints().isConstrained(j)) {
            continue;
        }
        const auto k = static_cast<std::size_t>(j);
        plus[k] = u[k] + eps;
        minus[k] = u[k] - eps;
        const auto rp = assembleDense(sys, op, plus).vector;
        const auto rm = assembleDense(sys, op, minus).vector;
        plus[k] = u[k];
        minus[k] = u[k];
        for (GlobalIndex i = 0; i < J.n; ++i) {
            if (sys.constraints().isConstrained(i)) {
                continue;
            }
            const auto fd = (rp[static_cast<std::size_t>(i)] - rm[static_cast<std::size_t>(i)]) / (2.0 * eps);
            defect = std::max(defect, std::abs(J(i, j) - fd));
            scale = std::max(scale, std::abs(J(i, j)));
        }
    }
    return {defect, scale};
}

/// log2 of the error ratio of a mesh and its refinement by two.
inline Real rate(Real coarse, Real fine)
{
    return std::log2(coarse / fine);
}

/// The observed rates of a refinement sequence by two.
inline std::vector<Real> rates(const std::vector<Real>& errors)
{
    std::vector<Real> out;
    for (std::size_t i = 1; i < errors.size(); ++i) {
        out.push_back(rate(errors[i - 1], errors[i]));
    }
    return out;
}

inline std::string join(const std::vector<Real>& values)
{
    std::ostringstream s;
    s.precision(6);
    for (std::size_t i = 0; i < values.size(); ++i) {
        s << (i == 0 ? "" : " ") << values[i];
    }
    return s.str();
}

} // namespace svmp::FE::systems::generality

#endif // SVMP_FE_TESTS_UNIT_SYSTEMS_GENERALITY_BENCHMARK_H
