/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GeneralityBenchmarkGaps.cpp
 * @brief The generality benchmark, second part: the problems the vocabulary
 * expresses only in part today
 *
 * Problems 2, 9, 12, 13, 14 and 15 of section 5 of the vocabulary review of
 * 2026-11-24 (and the remaining parts of 3, 8 and 11).  Each is written as
 * far as the public vocabulary carries it, against an independent known
 * solution; what it cannot express or solve today is a disabled test
 * (DISABLED_) whose comment names exactly what is missing, so the gap is
 * counted and not hidden.  Run them with --gtest_also_run_disabled_tests to
 * see each one fail with its reason.
 */

#include "Tests/Unit/Systems/GeneralityBenchmark.h"

#include "Forms/Authoring.h"
#include "LevelSet/LevelSetGeometryService.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
#include <Eigen/Dense>
#endif

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::systems::generality {
namespace {

using forms::FormExpr;

constexpr Real kPi = 3.14159265358979323846;

FormExpr scalarCoefficient(const std::string& name, std::function<Real(const Point&)> f)
{
    return FormExpr::coefficient(name, [f = std::move(f)](Real x, Real y, Real z) { return f(Point{x, y, z}); });
}

FormExpr timeCoefficient(const std::string& name, std::function<Real(const Point&, Real)> f)
{
    return FormExpr::coefficient(
        name, forms::TimeScalarCoefficient([f = std::move(f)](Real x, Real y, Real z, Real t) { return f(Point{x, y, z}, t); }));
}

Real maxDifference(const std::vector<Real>& a, const std::vector<Real>& b)
{
    Real m = 0.0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return a.size() == b.size() ? m : std::numeric_limits<Real>::infinity();
}

// ===========================================================================
// 2. Transient species and a global reservoir (D65 exit problem)
// ===========================================================================
//
// A species c in the unit square with the conservative flux j = beta c -
// kappa grad c, beta = (b, 0), exchanges with a well-mixed reservoir z
// through x = 1 (the exchange H (c - z) leaves the tissue), enters through
// x = 0 with a prescribed outward flux g, and the reservoir has capacity C
// and a source s:
//
//   ((dt(c) - f) v - j . grad v) dx + (g v) ds(x=0) + (H (c - z) (v - eta)) ds(x=1)
//     + (eta (C dt(z) - s)) dGlobal.
//
// With c = e^{-t} (2 - x^2) the reservoir is z = c(1, t) - j(1, t) . n / H =
// 0.65 e^{-t}; f, g and s are manufactured.  Q2 represents c(t) exactly, so
// the errors are the temporal ones.  The in-matrix lowering of the global
// field carries dt(z) (design 10 M3).
//
// The ordinary-application version (D65) needs, beyond this test: a deck
// declaration of a global reservoir unknown and its coupling (an input key,
// a user decision); the transient bordered lowering (design 10, M4) for the
// FSILS route, which setup refuses today (SystemSetup: time derivatives with
// bordered global fields); the global value in the checkpoint for restart
// from file; and the reservoir value in the scalar history (F.record of the
// global field exists, the writer is the Application's).

struct ReservoirData {
    Real b{0.5};
    Real kappa{0.1};
    Real H{2.0};
    Real C{0.5};

    [[nodiscard]] Real c(const Point& p, Real t) const { return std::exp(-t) * (2.0 - p[0] * p[0]); }
    [[nodiscard]] Real outflux() const { return b + 2.0 * kappa; }  ///< j . n at x = 1, over e^{-t}
    [[nodiscard]] Real z(Real t) const { return std::exp(-t) * (1.0 - outflux() / H); }
    [[nodiscard]] Real f(const Point& p, Real t) const
    {
        return std::exp(-t) * (-(2.0 - p[0] * p[0]) - 2.0 * b * p[0] + 2.0 * kappa);
    }
    /// s = C dz/dt - (exchange through x = 1), over e^{-t}.
    [[nodiscard]] Real sourceFactor() const { return -C * (1.0 - outflux() / H) - outflux(); }
};

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 2.
// ---------------------------------------------------------------------------
struct ReservoirInstall {
    FieldId c{INVALID_FIELD_ID};
    FieldId z{INVALID_FIELD_ID};
    Quantity tissue_mass;
};

ReservoirInstall defineSpeciesReservoir(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& V,
                                        const ReservoirData& d, bool transient)
{
    using namespace forms;
    FieldBinding fields(system, "SpeciesReservoir");
    const auto Cs = fields.unknown("c", V);
    const auto z_id = fields.global("z", "reservoir concentration");
    const auto& R1 = *system.fieldRecord(z_id).space;
    const BoundUnknown Z{z_id, StateField(z_id, R1, "z"), TestField(z_id, R1, "eta")};
    const auto& c = Cs.state;
    const auto& v = Cs.test;
    const auto& z = Z.state;
    const auto& eta = Z.test;
    const auto decay = (-FormExpr::time()).exp();
    const auto beta = as_vector({FormExpr::constant(d.b), FormExpr::constant(0.0)});
    const auto kappa = FormExpr::constant(d.kappa);
    const auto H = FormExpr::constant(d.H);
    const auto C = FormExpr::constant(d.C);
    const auto j = beta * c - kappa * grad(c);
    const auto exchange = H * (c - z);  // leaves the tissue through x = 1
    const auto f = timeCoefficient("f", [d](const Point& p, Real t) { return d.f(p, t); });
    const auto g = FormExpr::constant(-2.0 * d.b) * decay;  // j . n on x = 0
    const auto s = FormExpr::constant(d.sourceFactor()) * decay;

    const auto tissue = transient ? (dt(c) - f) * v - inner(j, grad(v)) : -f * v - inner(j, grad(v));
    const auto reservoir = transient ? eta * (C * dt(z) - s) : eta * (-s);
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    ReservoirInstall out{Cs.id, z_id, F.quantity("tissue_mass", FunctionalKind::Integral, c.dx())};
    F.record(out.tissue_mass).record(z_id);
    (void)F.residual(tissue.dx() + (g * v).ds(1) + (exchange * (v - eta)).ds(2) + reservoir.dGlobal());

    // The initial state: c(0) and z(0), projected.
    const auto x = FormExpr::coordinate().component(0);
    FormulationBuilder initial(system, "initial", {});
    (void)initial.residual(((c - (2.0 - x * x)) * v).dx() + (eta * (z - FormExpr::constant(d.z(0.0)))).dGlobal());
    return out;
}
// ---------------------------------------------------------------------------

struct ReservoirRun {
    TransientRun run;
    Real c_error{0.0};
    Real z_error{0.0};
    Real balance_defect{0.0};  ///< max over steps of the defect of int c + C z (backward Euler)
};

ReservoirRun runSpeciesReservoir(timestepping::SchemeKind scheme, double dt, int steps, TransientOptions options = {},
                                 const std::vector<Real>* start = nullptr)
{
    const ReservoirData d;
    FESystem system(boxMesh(Cell::Quad4, 2, boxSides()));
    const auto V = std::make_shared<spaces::H1Space>(ElementType::Quad4, 2);
    const auto install = defineSpeciesReservoir(system, V, d, /*transient=*/true);
    system.setup();
    const auto initial = solveSteady(system, "initial");
    const auto& u0 = start != nullptr ? *start : initial.u;
    ReservoirRun out;
    out.run = runTransient(system, "equations", u0, scheme, dt, steps, std::move(options));
    EXPECT_TRUE(out.run.success);
    if (out.run.states.empty()) {
        return out;
    }
    const auto z_dof = static_cast<std::size_t>(system.fieldDofOffset(install.z));
    const auto c_h = FormExpr::discreteField(install.c, *V, "c");
    const double T = out.run.times.back();
    const auto c_exact = scalarCoefficient("c_exact", [d, T](const Point& p) { return d.c(p, T); });
    out.c_error = norm(system, ((c_h - c_exact) * (c_h - c_exact)).dx(), out.run.states.back());
    out.z_error = std::abs(out.run.states.back()[z_dof] - d.z(T));
    if (scheme == timestepping::SchemeKind::BackwardEuler) {
        const auto f = timeCoefficient("f", [d](const Point& p, Real t) { return d.f(p, t); });
        Real stored = integrate(system, install.tissue_mass.form(), u0) + d.C * u0[z_dof];
        for (std::size_t k = 0; k < out.run.states.size(); ++k) {
            const auto& state = out.run.states[k];
            const double t = out.run.times[k];
            const Real now = integrate(system, install.tissue_mass.form(), state) + d.C * state[z_dof];
            // Inflow through x = 0 is -g, the reservoir source s, the tissue source f.
            const Real supply = integrate(system, f.dx(), state, t) + 2.0 * d.b * std::exp(-t) +
                                d.sourceFactor() * std::exp(-t);
            out.balance_defect = std::max(out.balance_defect, std::abs(now - stored - dt * supply));
            stored = now;
        }
    }
    return out;
}

void expectSpeciesReservoirOrder(timestepping::SchemeKind scheme, Real order, int max_iterations = 8)
{
    {
        std::vector<Real> c_err, z_err;
        for (const int steps : {4, 8, 16}) {
            TransientOptions options;
            options.max_iterations = max_iterations;
            const auto run = runSpeciesReservoir(scheme, 0.4 / steps, steps, options);
            c_err.push_back(run.c_error);
            z_err.push_back(run.z_error);
            if (scheme == timestepping::SchemeKind::BackwardEuler) {
                // int c + C z changes by dt times the supply at every step.
                EXPECT_LT(run.balance_defect, 1e-12) << steps;
            }
        }
        ::testing::Test::RecordProperty("p02_order" + std::to_string(static_cast<int>(order)) + "_c_rates",
                                        join(rates(c_err)));
        ::testing::Test::RecordProperty("p02_order" + std::to_string(static_cast<int>(order)) + "_z_rates",
                                        join(rates(z_err)));
        EXPECT_NEAR(rates(c_err).back(), order, 0.2) << join(c_err);
        EXPECT_NEAR(rates(z_err).back(), order, 0.2) << join(z_err);
    }
}

TEST(GeneralityBenchmarkGaps, P02_SpeciesWithGlobalReservoirConvergesUnderBackwardEuler)
{
    expectSpeciesReservoirOrder(timestepping::SchemeKind::BackwardEuler, 1.0);
}

TEST(GeneralityBenchmarkGaps, DISABLED_P02_SpeciesWithGlobalReservoirConvergesUnderBDF2)
{
    // Defect found by this benchmark (reported): with dt(z) of the global
    // field in its .dGlobal() row (in-matrix), BDF2 converges at first order
    // (c and z rates 1.01 and 1.00 for dt = 0.1, 0.05, 0.025), and its first
    // step takes 8 to 11 Newton iterations on this linear problem (the
    // tangent of that step does not match its residual); backward Euler is
    // exact to first order and takes one iteration per step.  BDF2 of
    // volume fields alone is second order (problem 10).
    expectSpeciesReservoirOrder(timestepping::SchemeKind::BDF2, 2.0, 20);
}

TEST(GeneralityBenchmarkGaps, P02_SpeciesWithGlobalReservoirRejectionAndRestartKeepTheAcceptedStates)
{
    const double dt = 0.05;
    const auto clean = runSpeciesReservoir(timestepping::SchemeKind::BackwardEuler, dt, 8);
    ASSERT_EQ(clean.run.states.size(), 8u);
    // A rejected attempt at steps 2 and 5, retried with the same dt.
    TransientOptions rejecting;
    rejecting.reject_once = {2, 5};
    const auto rejected = runSpeciesReservoir(timestepping::SchemeKind::BackwardEuler, dt, 8, rejecting);
    EXPECT_EQ(rejected.run.rejections, 2);
    ASSERT_EQ(rejected.run.states.size(), 8u);
    for (std::size_t k = 0; k < 8u; ++k) {
        EXPECT_LT(maxDifference(rejected.run.states[k], clean.run.states[k]), 1e-14) << "step " << k;
    }
    // A restart from the accepted state of step 4 continues the run.
    TransientOptions restart;
    restart.t0 = 4 * dt;
    const auto second = runSpeciesReservoir(timestepping::SchemeKind::BackwardEuler, dt, 4, restart,
                                            &clean.run.states[3]);
    ASSERT_EQ(second.run.states.size(), 4u);
    for (std::size_t k = 0; k < 4u; ++k) {
        EXPECT_LT(maxDifference(second.run.states[k], clean.run.states[4 + k]), 1e-13) << "step " << 4 + k;
    }
}

TEST(GeneralityBenchmarkGaps, P02_SpeciesWithGlobalReservoirCouplingBlocks)
{
    // The steady rows at an arbitrary state: the reservoir column of the
    // species rows and the species columns of the reservoir row are both
    // -H times the boundary integral of the species test functions on x = 1,
    // and the whole tangent equals central differences.
    const ReservoirData d;
    FESystem system(boxMesh(Cell::Quad4, 2, boxSides()));
    const auto V = std::make_shared<spaces::H1Space>(ElementType::Quad4, 2);
    const auto install = defineSpeciesReservoir(system, V, d, /*transient=*/false);
    system.setup();
    std::vector<Real> state(static_cast<std::size_t>(system.dofHandler().getNumDofs()));
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] = std::sin(0.9 * static_cast<Real>(i) + 0.2);
    }
    const auto J = assembleDense(system, "equations", state);
    const auto z = system.fieldDofOffset(install.z);
    const auto c0 = system.fieldDofOffset(install.c);
    const auto nc = system.fieldDofHandler(install.c).getNumDofs();
    Real column = 0.0;
    Real row = 0.0;
    Real asymmetry = 0.0;
    for (GlobalIndex i = c0; i < c0 + nc; ++i) {
        column += J(i, z);
        row += J(z, i);
        asymmetry = std::max(asymmetry, std::abs(J(i, z) - J(z, i)));
    }
    EXPECT_NEAR(column, -d.H, 1e-12);
    EXPECT_NEAR(row, -d.H, 1e-12);
    EXPECT_LT(asymmetry, 1e-14);
    const auto [defect, scale] = jacobianDefect(system, "equations", state);
    EXPECT_LT(defect, 1e-7 * scale);
}

} // namespace
} // namespace svmp::FE::systems::generality
