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
// a user decision); the FSILS route through the transient bordered lowering
// (design 10, M4); the global value in the checkpoint for restart from file;
// and the reservoir value in the scalar history (F.record of the global
// field exists, the writer is the Application's).

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

TEST(GeneralityBenchmarkGaps, P02_SpeciesWithGlobalReservoirConvergesUnderBDF2)
{
    expectSpeciesReservoirOrder(timestepping::SchemeKind::BDF2, 2.0);
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

// ===========================================================================
// 9. Viscoelastic material point in an elastic body
// ===========================================================================
//
// The internal variable a with dt(a) + a / tau - (strain rate) = 0, here a
// cellwise (DG0) unknown of the system, one value per cell standing for its
// material point, and E a in the stress work.  The displacement is
// prescribed with unit volumetric strain rate, so a(t) = tau (1 - e^{-t/tau}).
//
// What is missing for problem 9 as stated: a named quadrature-point state
// (an unknown or history value per quadrature point, not a cell DOF of a
// field: Forms has only byte-offset material-state terminals,
// materialStateOldRef/WorkRef, and no named spelling), the local row G(a,
// previous(a), strain, dt) = 0 solved per point and condensed out of the
// global tangent with its chain rule, the comparison of the condensed and
// uncondensed updates, and accepted history of such state on restart.

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for the cellwise proxy of problem 9.
// ---------------------------------------------------------------------------
struct ViscoelasticInstall {
    FieldId u{INVALID_FIELD_ID};
    FieldId a{INVALID_FIELD_ID};
    Quantity internal_variable;
};

ViscoelasticInstall defineViscoelasticBody(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& Vu,
                                           const std::shared_ptr<const spaces::FunctionSpace>& Va, Real modulus,
                                           Real relaxation_time, const std::vector<FormExpr>& u_wall,
                                           const std::vector<int>& walls)
{
    using namespace forms;
    FieldBinding fields(system, "Viscoelastic");
    const auto U = fields.unknown("u", Vu);
    const auto A = fields.unknown("a", Va);
    const auto E = FormExpr::constant(modulus);
    const auto tau = FormExpr::constant(relaxation_time);
    const auto& u = U.state;
    const auto& a = A.state;
    const auto strain_rate = div(dt(u));
    auto R = (2.0 * inner(symgrad(u), symgrad(U.test)) + E * a * div(U.test)).dx() +
             ((dt(a) + a / tau - strain_rate) * A.test).dx();
    BoundaryConditionManager bcs;
    for (const int wall : walls) {
        bcs.add(std::make_unique<bc::Dirichlet>(wall, u_wall));
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, u, U.test, U.id);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    ViscoelasticInstall out{U.id, A.id, F.quantity("a_mean", FunctionalKind::Integral, a.dx())};
    (void)F.residual(R);
    return out;
}
// ---------------------------------------------------------------------------

TEST(GeneralityBenchmarkGaps, P09_CellwiseViscoelasticVariableFollowsItsRelaxation)
{
    const Real tau = 0.5;
    const double T = 0.4;
    for (const auto& [scheme, order] :
         std::vector<std::pair<timestepping::SchemeKind, Real>>{{timestepping::SchemeKind::BackwardEuler, 1.0},
                                                                {timestepping::SchemeKind::BDF2, 2.0}}) {
        std::vector<Real> errors;
        for (const int steps : {4, 8, 16}) {
            FESystem system(boxMesh(Cell::Tri3, 2, boxSides()));
            const auto p1 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
            const auto Vu = std::make_shared<spaces::ProductSpace>(p1, 2);
            const auto Va = std::make_shared<spaces::L2Space>(ElementType::Triangle3, 0);
            // u = (t x, 0): div(dt u) = 1.
            const auto t = FormExpr::time();
            const std::vector<FormExpr> u_wall{t * FormExpr::coordinate().component(0), FormExpr::constant(0.0)};
            const auto install = defineViscoelasticBody(system, Vu, Va, 3.0, tau, u_wall, {1, 2, 3, 4});
            system.setup();
            const auto run = runTransient(system, "equations", std::vector<Real>(system.dofHandler().getNumDofs(), 0.0),
                                          scheme, T / steps, steps);
            ASSERT_TRUE(run.success);
            ASSERT_FALSE(run.states.empty());
            const Real a = integrate(system, install.internal_variable.form(), run.states.back());
            errors.push_back(std::abs(a - tau * (1.0 - std::exp(-T / tau))));
            RecordProperty("p09_a_" + std::to_string(steps) + "_" + std::to_string(static_cast<int>(order)), std::to_string(a));
        }
        RecordProperty("p09_order" + std::to_string(static_cast<int>(order)) + "_rates", join(rates(errors)));
        EXPECT_NEAR(rates(errors).back(), order, 0.2) << join(errors);
    }
}

TEST(GeneralityBenchmarkGaps, P09_TheDivergenceOfARateIsTheTraceOfItsGradient)
{
    // div(dt(u)) of a vector field takes the whole stencil, as
    // trace(grad(dt(u))) does: with u = (t x, 0) prescribed, the cellwise
    // b = div(dt(u)) is 1 at every backward Euler step (it read 1, 2, 3, 4
    // when the divergence of a rate kept only the stencil's leading term).
    for (const bool divergence : {true, false}) {
        FESystem system(boxMesh(Cell::Tri3, 2, boxSides()));
        FieldBinding fields(system, "Rate");
        const auto p1 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
        const auto U = fields.unknown("u", std::make_shared<spaces::ProductSpace>(p1, 2));
        const auto B = fields.unknown("b", std::make_shared<spaces::L2Space>(ElementType::Triangle3, 0));
        const auto t = FormExpr::time();
        const auto x = FormExpr::coordinate().component(0);
        const auto rate = divergence ? forms::div(forms::dt(U.state)) : forms::trace(forms::grad(forms::dt(U.state)));
        auto R = forms::inner(U.state - forms::as_vector({t * x, FormExpr::constant(0.0)}), U.test).dx() +
                 ((B.state - rate) * B.test).dx();
        BoundaryConditionManager bcs;
        for (int wall = 1; wall <= 4; ++wall) {
            bcs.add(std::make_unique<forms::bc::Dirichlet>(wall, std::vector<FormExpr>{t * x, FormExpr::constant(0.0)}));
        }
        constraints::ConstraintPlan plan(system);
        plan.essentials(bcs, R, U.state, U.test, U.id);
        plan.commit();
        FormulationBuilder F(system, "equations", {});
        (void)F.residual(R);
        system.setup();
        const auto run = runTransient(system, "equations", std::vector<Real>(system.dofHandler().getNumDofs(), 0.0),
                                      timestepping::SchemeKind::BackwardEuler, 0.1, 4);
        for (const auto& state : run.states) {
            EXPECT_NEAR(integrate(system, B.state.dx(), state), 1.0, 1e-12) << (divergence ? "div" : "trace grad");
        }
    }
}

TEST(GeneralityBenchmarkGaps, DISABLED_P09_NamedQuadratureStateWithCondensedTangent)
{
    // Missing: a named quadrature-point state of the formulation and its
    // local evolution row G(a, previous(a), strain, dt) = 0 on that state
    // (Forms has only the internal byte-offset terminals
    // materialStateOldRef/materialStateWorkRef and expert constitutive
    // callbacks); the condensed tangent of the global rows through the local
    // solve (the chain rule dG/da^{-1} dG/d strain); the uncondensed update to
    // compare it with; and the restart of accepted quadrature history.  A J2
    // loading/unloading cycle needs, in addition, the complementarity of the
    // yield condition on that state.
    FAIL() << "named quadrature-point state and its condensed tangent are not vocabulary yet";
}

// ===========================================================================
// 12. Surface diffusion under rigid rotation
// ===========================================================================
//
// inner(grad_Gamma u, grad_Gamma v) on a surface-owned field.  The first step
// is a flat patch: the same triangles in the plane z = 0 and rotated rigidly
// in three-space must give the same stiffness.

std::shared_ptr<svmp::Mesh> flatPatch(int n, const std::array<std::array<Real, 3>, 3>& rotation, int spatial_dim)
{
    std::vector<svmp::real_t> x;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            const Point p{static_cast<Real>(i) / n, static_cast<Real>(j) / n, 0.0};
            for (int d = 0; d < spatial_dim; ++d) {
                const auto& r = rotation[static_cast<std::size_t>(d)];
                x.push_back(static_cast<svmp::real_t>(r[0] * p[0] + r[1] * p[1] + r[2] * p[2]));
            }
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    const auto vid = [n](int i, int j) { return static_cast<svmp::index_t>(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            conn.insert(conn.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            conn.insert(conn.end(), {vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
        }
    }
    svmp::CellShape triangle{};
    triangle.family = svmp::CellFamily::Triangle;
    triangle.num_corners = 3;
    triangle.order = 1;
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(spatial_dim, x, offsets, conn,
                            std::vector<svmp::CellShape>(offsets.size() - 1u, triangle));
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::vector<Real> patchStiffness(const std::shared_ptr<svmp::Mesh>& mesh)
{
    FESystem system(mesh);
    FieldBinding fields(system, "SurfaceDiffusion");
    const auto U = fields.unknown("u", std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1));
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual(forms::inner(forms::grad(U.state), forms::grad(U.test)).dx());
    system.setup();
    return assembleDense(system, "equations", std::vector<Real>(system.dofHandler().getNumDofs(), 0.0)).matrix;
}

TEST(GeneralityBenchmarkGaps, DISABLED_P12_RotatedFlatPatchHasTheStiffnessOfThePlanarOne)
{
    // Missing: a surface-owned scalar field with its intrinsic gradient.  A
    // triangle mesh embedded in three-space is refused at setup
    // ("DofLayoutInfo::Lagrange: unsupported cell topology"), and its
    // geometry would need a 3 x 2 Jacobian and its pseudo-inverse;
    // interface-owned fields enter forms only linearly, one unknown per
    // product on one side, so inner(grad_Gamma u, grad_Gamma v) of an
    // interface field is refused; surfaceGradient(f, n) projects an ambient
    // gradient of a volume field.  This body states the check.
    const std::array<std::array<Real, 3>, 3> identity{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const Real c = std::cos(0.7), s = std::sin(0.7), c2 = std::cos(0.4), s2 = std::sin(0.4);
    // A rotation about z by 0.7, then about x by 0.4.
    const std::array<std::array<Real, 3>, 3> rotation{{{c, -s, 0.0}, {c2 * s, c2 * c, -s2}, {s2 * s, s2 * c, c2}}};
    const auto planar = patchStiffness(flatPatch(3, identity, 2));
    const auto rotated = patchStiffness(flatPatch(3, rotation, 3));
    ASSERT_EQ(planar.size(), rotated.size());
    EXPECT_LT(maxDifference(planar, rotated), 1e-12);
}

TEST(GeneralityBenchmarkGaps, DISABLED_P12_SurfaceLaplacianOnTheSphere)
{
    // Missing, after the flat patch: the surface field on a curved surface
    // mesh with its metric and measure, so that -lap_Gamma u = 2x on the unit
    // sphere with mean-zero u = x converges in the lifted L2/H1 norms at
    // orders two and one for P1 on adequate geometry.
    FAIL() << "surface-owned fields with an intrinsic gradient are not vocabulary yet";
}

// ===========================================================================
// 14. Membrane eigenmodes
// ===========================================================================
//
// K = inner(grad u, grad v) dx and M = u v dx with u = 0 on the boundary of
// the unit square: the first eigenvalue is 2 pi^2, the error O(h^{2p}).  The
// forms are written and assembled as authored; the pencil is solved here by
// a dense generalized eigensolver in the test, because the problem-level
// modal entry is missing.

struct MembraneOperators {
    std::vector<Real> K;
    std::vector<Real> M;
    std::vector<GlobalIndex> free;
    GlobalIndex n{0};
};

MembraneOperators assembleMembrane(Cell cell, int order, int n)
{
    using namespace forms;
    FESystem system(boxMesh(cell, n, boxSides()));
    FieldBinding fields(system, "Membrane");
    const auto U = fields.unknown("u", std::make_shared<spaces::H1Space>(elementType(cell), order));
    const auto& u = U.state;
    const auto& v = U.test;
    BoundaryConditionManager bcs;
    for (const int wall : {1, 2, 3, 4}) {
        bcs.add(std::make_unique<bc::Dirichlet>(wall, FormExpr::constant(0.0)));
    }
    constraints::ConstraintPlan plan(system);
    auto stiffness = inner(grad(u), grad(v)).dx();
    plan.essentials(bcs, stiffness, u, v, U.id);
    plan.commit();
    FormulationBuilder K(system, "stiffness", {});
    (void)K.residual(stiffness);
    FormulationBuilder M(system, "mass", {});
    (void)M.residual((u * v).dx());
    system.setup();
    const std::vector<Real> zero(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    MembraneOperators out;
    const auto k = assembleDense(system, "stiffness", zero);
    const auto m = assembleDense(system, "mass", zero);
    out.n = k.n;
    out.K = k.matrix;
    out.M = m.matrix;
    for (GlobalIndex i = 0; i < k.n; ++i) {
        if (!system.constraints().isConstrained(i)) {
            out.free.push_back(i);
        }
    }
    return out;
}

TEST(GeneralityBenchmarkGaps, P14_MembraneStiffnessAndMassGiveTheFirstEigenvalue)
{
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    const Real exact = 2.0 * kPi * kPi;
    for (const auto& c : std::vector<std::pair<Cell, int>>{{Cell::Quad4, 1}, {Cell::Tri3, 1}, {Cell::Tri3, 2}}) {
        std::vector<Real> errors;
        for (const int n : {4, 8, 16}) {
            const auto ops = assembleMembrane(c.first, c.second, n);
            const auto m = static_cast<Eigen::Index>(ops.free.size());
            Eigen::MatrixXd K(m, m), M(m, m);
            for (Eigen::Index i = 0; i < m; ++i) {
                for (Eigen::Index j = 0; j < m; ++j) {
                    const auto a = ops.free[static_cast<std::size_t>(i)];
                    const auto b = ops.free[static_cast<std::size_t>(j)];
                    K(i, j) = ops.K[static_cast<std::size_t>(a * ops.n + b)];
                    M(i, j) = ops.M[static_cast<std::size_t>(a * ops.n + b)];
                }
            }
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(K, M);
            ASSERT_EQ(solver.info(), Eigen::Success);
            const Real lambda = solver.eigenvalues()(0);
            const Eigen::VectorXd x = solver.eigenvectors().col(0);
            // ||K x - lambda M x|| and M-orthonormality of the mode.
            EXPECT_LT((K * x - lambda * (M * x)).norm(), 1e-9 * lambda);
            EXPECT_NEAR(x.dot(M * x), 1.0, 1e-10);
            errors.push_back(lambda - exact);
            EXPECT_GT(lambda, exact);  // conforming: an upper bound
        }
        const std::string label = std::string(cellName(c.first)) + "_P" + std::to_string(c.second);
        RecordProperty("p14_" + label + "_rates", join(rates(errors)));
        EXPECT_GT(rates(errors).back(), 2.0 * c.second - 0.2) << label << " " << join(errors);
    }
#else
    GTEST_SKIP() << "the test-side eigensolver needs Eigen";
#endif
}

TEST(GeneralityBenchmarkGaps, DISABLED_P14_ModalSolveFromAuthoredForms)
{
    // Missing: a problem-level eigenproblem(K, M, essential) that consumes
    // the constrained pencil of two authored bilinear forms (the essential
    // conditions removed, not left as identity rows), with normalization,
    // nullspace handling and a backend eigensolver, returning eigenpairs
    // for checks such as ||K x - lambda M x||, M-orthogonality and invariant
    // subspaces of repeated eigenvalues.  Pointwise tensor eigenvalues
    // (SpectralEigen) and dense analysis utilities answer other questions.
    FAIL() << "no modal solve consumes authored forms yet";
}

// ===========================================================================
// 15. Steady discrete adjoint with a nonsymmetric coupled Jacobian
// ===========================================================================
//
// -u'' = m, -w'' = u on the unit-width strip (0,1) x (0,1/4) with u = w = 0
// at the ends and insulated sides; Q = int w = m / 120 (per unit width) and
// J = Q^2 / 2, dJ/dm = m / 14400.  The primal is written and solved as
// authored.  The adjoint here is computed by the test from the dense
// Jacobian, because the library offers no adjoint solve.

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for the primal of problem 15.
// ---------------------------------------------------------------------------
struct CoupledPoissonInstall {
    FieldId u{INVALID_FIELD_ID};
    FieldId w{INVALID_FIELD_ID};
    Quantity Q;
};

CoupledPoissonInstall defineCoupledPoisson(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& V,
                                           Real control)
{
    using namespace forms;
    FieldBinding fields(system, "CoupledPoisson");
    const auto U = fields.unknown("u", V);
    const auto W = fields.unknown("w", V);
    const auto m = FormExpr::constant(control);
    auto R = (inner(grad(U.state), grad(U.test)) - m * U.test).dx() +
             (inner(grad(W.state), grad(W.test)) - U.state * W.test).dx();
    BoundaryConditionManager u_ends;
    BoundaryConditionManager w_ends;
    for (const int end : {1, 2}) {
        u_ends.add(std::make_unique<bc::Dirichlet>(end, FormExpr::constant(0.0)));
        w_ends.add(std::make_unique<bc::Dirichlet>(end, FormExpr::constant(0.0)));
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(u_ends, R, U.state, U.test, U.id);
    plan.essentials(w_ends, R, W.state, W.test, W.id);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    CoupledPoissonInstall out{U.id, W.id, F.quantity("Q", FunctionalKind::Integral, W.state.dx())};
    F.record(out.Q);
    (void)F.residual(R);
    // dQ/dw[v], the objective's gradient as a form (a measurement).
    FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    FormulationBuilder G(system, "objective_gradient", measurement);
    (void)G.residual(W.test.dx());
    return out;
}
// ---------------------------------------------------------------------------

struct AdjointRun {
    Real Q{0.0};
    Real J{0.0};
    Real adjoint_gradient{0.0};
};

AdjointRun runCoupledPoisson(int n, Real control)
{
    constexpr Real width = 0.25;
    BoxOptions options;
    options.hi = {1.0, width, 1.0};
    FESystem system(boxMesh(Cell::Quad4, n, boxSides(options.lo, options.hi), options));
    const auto V = std::make_shared<spaces::H1Space>(ElementType::Quad4, 2);
    const auto install = defineCoupledPoisson(system, V, control);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged);
    AdjointRun run;
    run.Q = integrate(system, install.Q.form(), solved.u) / width;
    run.J = 0.5 * run.Q * run.Q;
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    // The discrete adjoint by hand: the constrained Jacobian A (free rows and
    // columns), dJ/dU = Q dQ/dU, A^T lambda = dJ/dU, dJ/dm = -lambda . dR/dm,
    // where dR/dm = R(0)/m (R is affine in m, zero at the solution).
    const auto A = assembleDense(system, "equations", solved.u);
    const std::vector<Real> zero(solved.u.size(), 0.0);
    const auto R0 = assembleDense(system, "equations", zero).vector;
    const auto g = assembleDense(system, "objective_gradient", zero).vector;
    std::vector<GlobalIndex> free;
    for (GlobalIndex i = 0; i < A.n; ++i) {
        if (!system.constraints().isConstrained(i)) {
            free.push_back(i);
        }
    }
    const auto m = static_cast<Eigen::Index>(free.size());
    Eigen::MatrixXd At(m, m);
    Eigen::VectorXd dJ(m), dR(m);
    for (Eigen::Index i = 0; i < m; ++i) {
        const auto a = free[static_cast<std::size_t>(i)];
        dJ(i) = run.Q * g[static_cast<std::size_t>(a)] / width;
        dR(i) = R0[static_cast<std::size_t>(a)] / control;
        for (Eigen::Index j = 0; j < m; ++j) {
            At(j, i) = A(a, free[static_cast<std::size_t>(j)]);
        }
    }
    const Eigen::VectorXd lambda = At.partialPivLu().solve(dJ);
    run.adjoint_gradient = -lambda.dot(dR);
#endif
    return run;
}

TEST(GeneralityBenchmarkGaps, P15_PrimalAndTestSideAdjointGiveTheReducedGradient)
{
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    const Real m = 3.0;
    std::vector<Real> q_err, g_err;
    for (const int n : {2, 4, 8}) {
        const auto run = runCoupledPoisson(n, m);
        q_err.push_back(std::abs(run.Q - m / 120.0));
        g_err.push_back(std::abs(run.adjoint_gradient - m / 14400.0));
        // The reduced gradient of the discrete J by central differences in m
        // (J is quadratic in m, so they are exact to round-off).
        const Real delta = 1e-3;
        const Real fd = (runCoupledPoisson(n, m + delta).J - runCoupledPoisson(n, m - delta).J) / (2.0 * delta);
        EXPECT_NEAR(run.adjoint_gradient, fd, 1e-9 * std::abs(fd)) << n;
        // The first-order Taylor remainder decays at second order.
        const Real r1 = std::abs(runCoupledPoisson(n, m + 0.2).J - run.J - 0.2 * run.adjoint_gradient);
        const Real r2 = std::abs(runCoupledPoisson(n, m + 0.1).J - run.J - 0.1 * run.adjoint_gradient);
        EXPECT_NEAR(std::log2(r1 / r2), 2.0, 0.05) << n;
    }
    RecordProperty("p15_Q_errors", join(q_err));
    RecordProperty("p15_gradient_errors", join(g_err));
    // Q2 integrates w exactly up to the quartic part: the errors are small and decrease.
    EXPECT_LT(q_err.back(), 1e-6);
    EXPECT_LT(g_err.back(), 1e-7);
#else
    GTEST_SKIP() << "the test-side adjoint needs Eigen";
#endif
}

TEST(GeneralityBenchmarkGaps, DISABLED_P15_AdjointSolveOfTheConstrainedResidual)
{
    // Missing: an adjoint service of a checked problem: the transpose of the
    // constrained discrete Jacobian applied and solved through the backend
    // (GenericMatrix and LinearSolver expose forward actions only), a
    // declared objective functional with its state derivative (a Quantity is
    // measured, not differentiated into a vector), the control derivative of
    // the residual, and the reduced gradient they give; transient adjoints
    // add accepted-stage replay and the transposes of transfers and local
    // updates.
    FAIL() << "no adjoint solve of an authored residual yet";
}

// ===========================================================================
// Remaining parts of problems 3, 8, 11 and 13
// ===========================================================================

TEST(GeneralityBenchmarkGaps, DISABLED_P03_LimitedTopHatKeepsMassAndStageBounds)
{
    // Missing: a conservative stage operation for DG transport: cell means,
    // neighbor data, a limiter applied between stages that preserves mass,
    // and admissible stage bounds for a stated limiter and time step.  The
    // pointwise clamp() is not a limiter, and the level-set phase corrector
    // is specific to its transport path.
    FAIL() << "no conservative limiter stage for authored DG transport yet";
}

TEST(GeneralityBenchmarkGaps, DISABLED_P08_UnilateralContactWithARigidPlane)
{
    // Missing: g_n >= 0, lambda_n >= 0, lambda_n g_n = 0 as a declared
    // inequality with equal and opposite virtual work, its search and
    // projection, active-set changes and their rollback owned by the problem
    // declaration.  Trace inequalities and the expert surface-contact kernel
    // are separate pieces.
    FAIL() << "no searched contact inequality in the vocabulary yet";
}

TEST(GeneralityBenchmarkGaps, DISABLED_P11_NonmatchingWorkPairingAndFSIVirtualWork)
{
    // Missing (wave-7 gates): on nonmatching interface meshes, the transfer T
    // with its work-conjugate reverse action, <T v, t>_B = <v, T* t>_A in the
    // interface mass pairing; and the FSI version, matched interface
    // velocities with nonzero normal and shear traction work, solved
    // conservatively.
    FAIL() << "no verified work-conjugate nonmatching transfer yet";
}

// ===========================================================================
// 13. A cut interface with conservative transfer: the static part
// ===========================================================================
//
// The two transfer rows of problem 11 across a generated interface that cuts
// the mesh: T_A on the negative side of phi = x - s, T_B on the positive
// side, s = 0.43 between mesh lines.  The linear profiles of problem 11 are
// exact for P1 on each side, and the two transfer rows cancel.

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for the static part of problem 13.
// ---------------------------------------------------------------------------
struct CutTransferInstall {
    FieldId TA{INVALID_FIELD_ID};
    FieldId TB{INVALID_FIELD_ID};
    Quantity heat_A_to_B;
};

CutTransferInstall defineCutTransfer(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& V,
                                     const level_set::CutSide& A, const level_set::CutSide& B,
                                     const forms::InterfaceDomain& Gamma, Real kA, Real kB, Real resistance,
                                     const FormExpr& TA_wall, const FormExpr& TB_wall)
{
    using namespace forms;
    FieldBinding fields(system, "CutTransfer");
    const auto TA = fields.unknown("T_A", V);
    const auto TB = fields.unknown("T_B", V);
    const auto r = FormExpr::constant(resistance);
    const auto I = InterfaceBetween::generated(Gamma.marker());
    const auto jumpT = I.trace(TA.state, A.volume()) - I.trace(TB.state, B.volume());
    const auto jumpV = I.trace(TA.test, A.volume()) - I.trace(TB.test, B.volume());
    auto R = (FormExpr::constant(kA) * inner(grad(TA.state), grad(TA.test))).dx(A.volume()) +
             (FormExpr::constant(kB) * inner(grad(TB.state), grad(TB.test))).dx(B.volume()) +
             (jumpT * jumpV / r).dI(Gamma);
    BoundaryConditionManager wallA;
    wallA.add(std::make_unique<bc::Dirichlet>(1, TA_wall));
    BoundaryConditionManager wallB;
    wallB.add(std::make_unique<bc::Dirichlet>(2, TB_wall));
    constraints::ConstraintPlan plan(system);
    plan.essentials(wallA, R, TA.state, TA.test, TA.id);
    plan.essentials(wallB, R, TB.state, TB.test, TB.id);
    plan.activeOn({TA.id}, A, {.aggregate = false}).activeOn({TB.id}, B, {.aggregate = false});
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    CutTransferInstall out{TA.id, TB.id, F.quantity("heat_A_to_B", FunctionalKind::Integral, (jumpT / r).dI(Gamma))};
    (void)F.residual(R);
    // The transfer rows alone, a measurement.
    FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    FormulationBuilder transfer(system, "transfer", measurement);
    (void)transfer.residual((jumpT * jumpV / r).dI(Gamma));
    return out;
}
// ---------------------------------------------------------------------------

TEST(GeneralityBenchmarkGaps, P13_StaticCutInterfaceTransferIsExactAndCancels)
{
    const Real s = 0.43, L = 1.0, q = 0.6, kA = 2.0, kB = 0.5, r = 0.4;
    for (const auto cell : {Cell::Tri3, Cell::Quad4, Cell::Tet4}) {
        auto mesh = boxMesh(cell, 5, boxSides());
        FESystem system(mesh);
        const auto V = std::make_shared<spaces::H1Space>(elementType(cell), 1);
        const auto phi = system.addField(FieldSpec{.name = "phi", .space = V, .source_kind = FieldSourceKind::PrescribedData});
        auto G = level_set::cutGeometry(system, *mesh, {.level_set_field_name = "phi", .domain_id = "slab"});
        const auto A = G.side(forms::CutVolumeSide::Negative);
        const auto B = G.side(forms::CutVolumeSide::Positive);
        const auto exactA = scalarCoefficient("T_A_exact", [=](const Point& p) { return L - q * (p[0] - s) / kA; });
        const auto exactB = scalarCoefficient("T_B_exact", [=](const Point& p) { return L - r * q - q * (p[0] - s) / kB; });
        const auto install = defineCutTransfer(system, V, A, B, G.interface(), kA, kB, r, exactA, exactB);
        system.setup();
        std::vector<Real> phi_at_vertices(mesh->n_vertices());
        for (std::size_t v = 0; v < phi_at_vertices.size(); ++v) {
            phi_at_vertices[v] = static_cast<Real>(mesh->X_ref()[static_cast<std::size_t>(mesh->dim()) * v]) - s;
        }
        std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(phi).getNumDofs()), 0.0);
        (void)system.projectMeshVertexValuesToFieldCoefficients(phi, phi_at_vertices, 1u, coefficients);
        system.setPrescribedFieldCoefficients(phi, coefficients);
        (void)G.refresh();

        const auto solved = solveSteady(system, "equations");
        EXPECT_TRUE(solved.converged) << cellName(cell);
        const auto TA_h = FormExpr::discreteField(install.TA, *V, "T_A");
        const auto TB_h = FormExpr::discreteField(install.TB, *V, "T_B");
        EXPECT_LT(norm(system, ((TA_h - exactA) * (TA_h - exactA)).dx(A.volume()), solved.u), 1e-11) << cellName(cell);
        EXPECT_LT(norm(system, ((TB_h - exactB) * (TB_h - exactB)).dx(B.volume()), solved.u), 1e-11) << cellName(cell);
        // The heat crossing Gamma, through the transfer rows of T_A (the measured
        // form of a two-sided generated-interface integrand is not evaluated).
        const auto at_solution = assembleVector(system, "transfer", solved.u);
        Real heat = 0.0;
        const auto a0 = system.fieldDofOffset(install.TA);
        for (GlobalIndex i = a0; i < a0 + system.fieldDofHandler(install.TA).getNumDofs(); ++i) {
            if (!system.constraints().isConstrained(i)) {
                heat += at_solution[static_cast<std::size_t>(i)];
            }
        }
        EXPECT_NEAR(heat, q, 1e-11) << cellName(cell);
        // The transfer rows sum to zero at any state (test functions 1 on both sides).
        std::vector<Real> state(solved.u.size());
        for (std::size_t i = 0; i < state.size(); ++i) {
            state[i] = std::cos(0.8 * static_cast<Real>(i));
        }
        const auto rows = assembleVector(system, "transfer", state);
        Real sum = 0.0;
        Real scale = 0.0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (system.constraints().isConstrained(static_cast<GlobalIndex>(i))) {
                continue;  // pinned off its side or on a wall, away from the cut
            }
            sum += rows[i];
            scale = std::max(scale, std::abs(rows[i]));
        }
        EXPECT_GT(scale, 0.0);
        EXPECT_LT(std::abs(sum), 1e-13 * std::max(Real(1.0), scale)) << cellName(cell);
    }
}

TEST(GeneralityBenchmarkGaps, DISABLED_P13_MovingCutInterfaceWithConservativeTransfer)
{
    // D65 exit problem.  Missing: a cut interface moved by a prescribed s(t)
    // inside the time loop, the cut geometry refreshed per stage as generated
    // state with its rollback on a rejected step (a rejected crossing) and
    // restart; the state of DOFs that change side (no history value where a
    // field was inactive: the transfer of accepted state across the moving
    // interface); the interface velocity w_Gamma as a terminal of the
    // generated interface for the relative flux j . n - c w_Gamma . n in the
    // two transfer rows of problem 11; and measurements of each side's mass
    // and the transfer cancellation from the equation's own quantities.
    //
    // The ordinary-application version needs, beyond that: a deck route for
    // a two-sided cut transport with interface exchange (an input key, a user
    // decision), the cut-geometry service driven by a prescribed motion
    // rather than a solved level set, and the same accepted-state lifecycle
    // in the Application's step participants (checkpoint, rejection,
    // restart) that the level-set free surface uses.
    FAIL() << "a moving cut interface with conservative two-sided transfer is not expressible yet";
}

} // namespace
} // namespace svmp::FE::systems::generality
