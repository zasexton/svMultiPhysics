/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GeneralityBenchmark.cpp
 * @brief The generality benchmark: canonical PDE problems written in the
 * public vocabulary, each against an independent known solution
 *
 * Each problem is written as an author writes a formulation (FieldBinding,
 * FormExpr, measures, FormulationBuilder::residual, bc::*, quantities) and
 * solved on coarse meshes of the cell families its spaces support.  The
 * forcing is manufactured in C++ from the stated continuum solution, apart
 * from the expression under test, and the observed rates (log2 of the error
 * ratio of a mesh and its refinement by two) are checked against the
 * theoretical ones.  The problems are those of section 5 of the vocabulary
 * review of 2026-11-24 that the vocabulary expresses today: 1, 3, 4, 5, 6,
 * 7, 8, 10 and 11; test_GeneralityBenchmarkGaps.cpp holds those it expresses
 * only in part.  Where the expressed problem cannot be solved today because
 * of a defect this benchmark found (problems 5, 6 and 7, and one quantity of
 * 11), the test is disabled (DISABLED_) with a comment that names the
 * defect; --gtest_also_run_disabled_tests runs them (except the RT1/DG1
 * Darcy case, which corrupts the heap).
 */

#include "Tests/Unit/Systems/GeneralityBenchmark.h"

#include "Forms/Authoring.h"
#include "Constraints/HCurlTangentialConstraint.h"
#include "Spaces/H1Space.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/L2Space.h"
#include "Spaces/ProductSpace.h"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::systems::generality {
namespace {

using forms::FormExpr;

FormExpr scalarCoefficient(const std::string& name, std::function<Real(const Point&)> f)
{
    return FormExpr::coefficient(name, [f = std::move(f)](Real x, Real y, Real z) { return f(Point{x, y, z}); });
}

std::vector<FormExpr> componentsOf(const std::string& name, int dim, const std::function<Real(const Point&, int)>& f)
{
    std::vector<FormExpr> components;
    for (int d = 0; d < dim; ++d) {
        components.push_back(scalarCoefficient(name + "_" + std::to_string(d), [f, d](const Point& p) { return f(p, d); }));
    }
    return components;
}

FormExpr vectorOf(const std::string& name, int dim, const std::function<Real(const Point&, int)>& f)
{
    return FormExpr::asVector(componentsOf(name, dim, f));
}

// ===========================================================================
// 1. Anisotropic Poisson with a signed flux
// ===========================================================================
//
// -div(K grad u) = f in the unit square or cube, u = u_exact on the faces
// x = 0, y = 0 (z = 0), and the conormal flux K grad(u) . n = g on the faces
// x = 1, y = 1 (z = 1), with K a constant SPD tensor with nonzero
// off-diagonal entries and
//
//   u = 1 + x^2 y + x y^3 - x^4 / 2 + x z^3 + y^2 z^2,
//
// outside every P1, P2, Q1 and Q2 space.  The heat leaving through the flux
// faces is the integral of (-K grad u) . n, the opposite of the prescribed
// conormal flux g.

/// K, symmetric positive definite (its leading 2 x 2 block in 2D).
constexpr std::array<std::array<Real, 3>, 3> kConductivity{{{2.0, 0.5, 0.2}, {0.5, 1.5, 0.3}, {0.2, 0.3, 1.0}}};

Real poissonExact(const Point& p)
{
    const Real x = p[0], y = p[1], z = p[2];
    return 1.0 + x * x * y + x * y * y * y - 0.5 * x * x * x * x + x * z * z * z + y * y * z * z;
}

Point poissonGradient(const Point& p)
{
    const Real x = p[0], y = p[1], z = p[2];
    return {2.0 * x * y + y * y * y - 2.0 * x * x * x + z * z * z, x * x + 3.0 * x * y * y + 2.0 * y * z * z,
            3.0 * x * z * z + 2.0 * y * y * z};
}

std::array<std::array<Real, 3>, 3> poissonHessian(const Point& p)
{
    const Real x = p[0], y = p[1], z = p[2];
    const Real xy = 2.0 * x + 3.0 * y * y, xz = 3.0 * z * z, yz = 4.0 * y * z;
    return {{{2.0 * y - 6.0 * x * x, xy, xz}, {xy, 6.0 * x * y + 2.0 * z * z, yz}, {xz, yz, 6.0 * x * z + 2.0 * y * y}}};
}

/// K grad u, component i, in dimension dim.
Real conormalFlux(const Point& p, int dim, int i)
{
    const auto g = poissonGradient(p);
    Real s = 0.0;
    for (int j = 0; j < dim; ++j) {
        s += kConductivity[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * g[static_cast<std::size_t>(j)];
    }
    return s;
}

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 1.
// ---------------------------------------------------------------------------
struct PoissonData {
    FormExpr K;                                        ///< conductivity tensor
    FormExpr f;                                        ///< source
    FormExpr u_wall;                                   ///< u on the walls
    std::vector<int> walls;                            ///< Dirichlet faces
    std::vector<std::pair<int, FormExpr>> conormal;    ///< K grad(u) . n = g on these faces
    enum class FluxSpelling { Residual, NaturalBC, ZeroRobin } spelling{FluxSpelling::Residual};
};

struct PoissonInstall {
    FieldId u{INVALID_FIELD_ID};
    Quantity heat_leaving;
};

PoissonInstall defineAnisotropicPoisson(FESystem& system,
                                        std::shared_ptr<const spaces::FunctionSpace> V,
                                        const PoissonData& d)
{
    using namespace forms;
    FieldBinding fields(system, "AnisotropicPoisson");
    const auto U = fields.unknown("u", std::move(V));
    const auto& u = U.state;
    const auto& v = U.test;

    auto R = (inner(d.K * grad(u), grad(v)) - d.f * v).dx();
    BoundaryConditionManager bcs;
    for (const auto& [face, g] : d.conormal) {
        switch (d.spelling) {
        case PoissonData::FluxSpelling::Residual: R = R - (g * v).ds(face); break;
        case PoissonData::FluxSpelling::NaturalBC: bcs.add(std::make_unique<bc::NaturalBC>(face, g)); break;
        case PoissonData::FluxSpelling::ZeroRobin:
            bcs.add(std::make_unique<bc::RobinBC>(face, FormExpr::constant(0.0), g));
            break;
        }
    }
    for (const int wall : d.walls) {
        bcs.add(std::make_unique<bc::Dirichlet>(wall, d.u_wall));
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, u, v, U.id);
    plan.commit();

    FormulationBuilder F(system, "equations", {});

    F.symbolicTangent();
    const auto heatFlux = -(d.K * grad(u));
    FormExpr leaving;
    for (const auto& face : d.conormal) {
        leaving = sumOf(leaving, normalTrace(heatFlux).ds(face.first));
    }
    PoissonInstall out{U.id, F.quantity("heat_leaving", FunctionalKind::Integral, leaving)};
    F.record(out.heat_leaving);
    (void)F.residual(R);
    return out;
}
// ---------------------------------------------------------------------------

struct PoissonRun {
    Real l2{0.0};
    Real h1{0.0};
    Real heat_leaving{0.0};       ///< the recorded quantity at the solution
    Real prescribed_conormal{0.0}; ///< the integral of g over the flux faces
    std::vector<Real> solution;
};

PoissonRun runAnisotropicPoisson(Cell cell, int order, int n,
                                 PoissonData::FluxSpelling spelling = PoissonData::FluxSpelling::Residual)
{
    const int dim = cellDimension(cell);
    FESystem system(boxMesh(cell, n, boxSides()));
    std::vector<std::vector<FormExpr>> rows(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            rows[static_cast<std::size_t>(i)].push_back(
                FormExpr::constant(kConductivity[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));
        }
    }
    PoissonData d;
    d.K = FormExpr::asTensor(std::move(rows));
    d.f = scalarCoefficient("f", [dim](const Point& p) {
        const auto H = poissonHessian(p);
        Real s = 0.0;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                s -= kConductivity[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                     H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
        }
        return s;
    });
    d.u_wall = scalarCoefficient("u_wall", poissonExact);
    d.spelling = spelling;
    for (int axis = 0; axis < dim; ++axis) {
        d.walls.push_back(2 * axis + 1);
        d.conormal.emplace_back(2 * axis + 2, scalarCoefficient("g_" + std::to_string(axis), [dim, axis](const Point& p) {
                                    return conormalFlux(p, dim, axis);  // the outward normal is e_axis
                                }));
    }
    const auto install = defineAnisotropicPoisson(system, std::make_shared<spaces::H1Space>(elementType(cell), order), d);
    system.setup();

    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " P" << order << " n=" << n;
    const auto& record = system.fieldRecord(install.u);
    const auto u_h = FormExpr::discreteField(install.u, *record.space, "u");
    const auto e = u_h - scalarCoefficient("u_exact", poissonExact);
    const auto de = forms::grad(u_h) - vectorOf("grad_u_exact", dim, [](const Point& p, int i) {
                        return poissonGradient(p)[static_cast<std::size_t>(i)];
                    });
    PoissonRun run;
    run.l2 = norm(system, (e * e).dx(), solved.u);
    run.h1 = norm(system, forms::inner(de, de).dx(), solved.u);
    run.heat_leaving = integrate(system, install.heat_leaving.form(), solved.u);
    for (const auto& [face, g] : d.conormal) {
        run.prescribed_conormal += integrate(system, g.ds(face), solved.u);
    }
    run.solution = solved.u;
    return run;
}

struct FamilyOrder {
    Cell cell;
    int order;
    int n0;  ///< coarsest mesh; the sequence is n0, 2 n0, 4 n0
};

TEST(GeneralityBenchmark, P01_AnisotropicPoissonWithSignedFluxConvergesOnEveryFamily)
{
    const std::vector<FamilyOrder> cases{
        {Cell::Tri3, 1, 4}, {Cell::Quad4, 1, 4}, {Cell::Tet4, 1, 2}, {Cell::Hex8, 1, 2},
        {Cell::Wedge6, 1, 2}, {Cell::Pyr5, 1, 2}, {Cell::Tri3, 2, 2}, {Cell::Quad4, 2, 2},
        {Cell::Tet4, 2, 2},  {Cell::Hex8, 2, 2}};
    for (const auto& c : cases) {
        std::vector<Real> l2, h1, flux_gap;
        for (int level = 0; level < 3; ++level) {
            const auto run = runAnisotropicPoisson(c.cell, c.order, c.n0 << level);
            l2.push_back(run.l2);
            h1.push_back(run.h1);
            // The heat leaving is the opposite of the prescribed conormal flux.
            EXPECT_LT(run.heat_leaving * run.prescribed_conormal, 0.0) << cellName(c.cell);
            flux_gap.push_back(std::abs(run.heat_leaving + run.prescribed_conormal));
        }
        const auto l2_rates = rates(l2);
        const auto h1_rates = rates(h1);
        const std::string label = std::string(cellName(c.cell)) + "_P" + std::to_string(c.order);
        RecordProperty("p01_" + label + "_l2", join(l2));
        RecordProperty("p01_" + label + "_l2_rates", join(l2_rates));
        RecordProperty("p01_" + label + "_h1_rates", join(h1_rates));
        EXPECT_GT(l2_rates.back(), c.order + 1 - 0.25) << label << " L2 " << join(l2) << " rates " << join(l2_rates);
        EXPECT_GT(h1_rates.back(), c.order - 0.15) << label << " H1 " << join(h1) << " rates " << join(h1_rates);
        // The recorded flux, read from the discrete gradient, approaches the
        // prescribed one (at the rate of the gradient once asymptotic; the
        // pyramids are not yet at n = 8).
        RecordProperty("p01_" + label + "_flux_rates", join(rates(flux_gap)));
        EXPECT_LT(flux_gap[1], flux_gap[0]) << label << " flux gap " << join(flux_gap);
        EXPECT_LT(flux_gap[2], flux_gap[1]) << label << " flux gap " << join(flux_gap);
    }
}

TEST(GeneralityBenchmark, P01_NaturalBCAndZeroRobinAreTheWrittenFluxTerm)
{
    for (const auto cell : {Cell::Tri3, Cell::Tet4}) {
        const auto written = runAnisotropicPoisson(cell, 1, 3);
        const auto natural = runAnisotropicPoisson(cell, 1, 3, PoissonData::FluxSpelling::NaturalBC);
        const auto robin = runAnisotropicPoisson(cell, 1, 3, PoissonData::FluxSpelling::ZeroRobin);
        ASSERT_EQ(written.solution.size(), natural.solution.size());
        ASSERT_EQ(written.solution.size(), robin.solution.size());
        for (std::size_t i = 0; i < written.solution.size(); ++i) {
            EXPECT_NEAR(natural.solution[i], written.solution[i], 1e-12) << cellName(cell) << " dof " << i;
            EXPECT_NEAR(robin.solution[i], written.solution[i], 1e-12) << cellName(cell) << " dof " << i;
        }
    }
}

// ===========================================================================
// 4. Taylor-Hood Stokes through bound fields
// ===========================================================================
//
// -div(2 nu eps(u)) + grad p = f, div u = 0 in the unit square or cube, u = 0
// on the boundary, p of mean zero through a global multiplier.  With
// S(t) = sin^2(pi t) and phi = S(x) S(y) [S(z)] the velocity
// u = (d phi/dy, -d phi/dx, 0) is divergence free and vanishes on the
// boundary; p = cos(pi x) cos(pi y) [cos(pi z)] has mean zero.

constexpr Real kPi = 3.14159265358979323846;

struct StreamFactors {
    Real s0, s1, s2, s3;  ///< S and its first three derivatives
};

StreamFactors streamFactors(Real t)
{
    return {std::sin(kPi * t) * std::sin(kPi * t), kPi * std::sin(2.0 * kPi * t),
            2.0 * kPi * kPi * std::cos(2.0 * kPi * t), -4.0 * kPi * kPi * kPi * std::sin(2.0 * kPi * t)};
}

/// The exact velocity (component i), its gradient (i, j) and Laplacian (i).
struct StokesExact {
    int dim;

    [[nodiscard]] StreamFactors zf(Real z) const
    {
        return dim == 3 ? streamFactors(z) : StreamFactors{1.0, 0.0, 0.0, 0.0};
    }

    [[nodiscard]] Real velocity(const Point& p, int i) const
    {
        const auto X = streamFactors(p[0]), Y = streamFactors(p[1]), Z = zf(p[2]);
        if (i == 0) {
            return X.s0 * Y.s1 * Z.s0;
        }
        if (i == 1) {
            return -X.s1 * Y.s0 * Z.s0;
        }
        return 0.0;
    }

    [[nodiscard]] Real gradient(const Point& p, int i, int j) const
    {
        const auto X = streamFactors(p[0]), Y = streamFactors(p[1]), Z = zf(p[2]);
        const std::array<std::array<Real, 3>, 2> g{{{X.s1 * Y.s1 * Z.s0, X.s0 * Y.s2 * Z.s0, X.s0 * Y.s1 * Z.s1},
                                                     {-X.s2 * Y.s0 * Z.s0, -X.s1 * Y.s1 * Z.s0, -X.s1 * Y.s0 * Z.s1}}};
        return i < 2 ? g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] : 0.0;
    }

    [[nodiscard]] Real laplacian(const Point& p, int i) const
    {
        const auto X = streamFactors(p[0]), Y = streamFactors(p[1]), Z = zf(p[2]);
        if (i == 0) {
            return X.s2 * Y.s1 * Z.s0 + X.s0 * Y.s3 * Z.s0 + X.s0 * Y.s1 * Z.s2;
        }
        if (i == 1) {
            return -(X.s3 * Y.s0 * Z.s0 + X.s1 * Y.s2 * Z.s0 + X.s1 * Y.s0 * Z.s2);
        }
        return 0.0;
    }

    [[nodiscard]] Real pressure(const Point& p) const
    {
        return std::cos(kPi * p[0]) * std::cos(kPi * p[1]) * (dim == 3 ? std::cos(kPi * p[2]) : 1.0);
    }

    [[nodiscard]] Real pressureGradient(const Point& p, int i) const
    {
        std::array<Real, 3> c{std::cos(kPi * p[0]), std::cos(kPi * p[1]), dim == 3 ? std::cos(kPi * p[2]) : 1.0};
        c[static_cast<std::size_t>(i)] = -kPi * std::sin(kPi * p[static_cast<std::size_t>(i)]);
        return c[0] * c[1] * c[2];
    }
};

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 4.
// ---------------------------------------------------------------------------
struct StokesInstall {
    FieldId u{INVALID_FIELD_ID};
    FieldId p{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

StokesInstall defineStokes(FESystem& system,
                           std::shared_ptr<const spaces::FunctionSpace> velocity_space,
                           std::shared_ptr<const spaces::FunctionSpace> pressure_space,
                           Real viscosity,
                           const FormExpr& body_force,
                           const std::vector<int>& walls)
{
    using namespace forms;
    FieldBinding fields(system, "Stokes");
    const auto U = fields.unknown("u", std::move(velocity_space));
    const auto P = fields.unknown("p", std::move(pressure_space));
    const auto lambda_id = fields.global("lambda", "pressure mean multiplier");
    const auto& R1 = *system.fieldRecord(lambda_id).space;
    const BoundUnknown Lambda{lambda_id, StateField(lambda_id, R1, "lambda"), TestField(lambda_id, R1, "m")};
    const auto& u = U.state;
    const auto& v = U.test;
    const auto& p = P.state;
    const auto& q = P.test;
    const auto nu = FormExpr::constant(viscosity);

    auto R = (2.0 * nu * inner(symgrad(u), symgrad(v)) - p * div(v) + q * div(u) - inner(body_force, v)).dx() +
             (Lambda.state * q + Lambda.test * p).dx();  // the mean of p is zero
    const int dim = system.fieldRecord(U.id).components;
    BoundaryConditionManager bcs;
    for (const int wall : walls) {
        bcs.add(std::make_unique<bc::Dirichlet>(wall, std::vector<FormExpr>(static_cast<std::size_t>(dim),
                                                                            FormExpr::constant(0.0))));
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, u, v, U.id);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual(R);
    return {U.id, P.id, lambda_id};
}
// ---------------------------------------------------------------------------

struct StokesRun {
    Real velocity_l2{0.0};
    Real velocity_h1{0.0};
    Real pressure_l2{0.0};
    Real pressure_mean{0.0};
    Real multiplier{0.0};
    Real transpose_defect{-1.0};  ///< max |B_qu + B_vp^T| over free velocity DOFs (n small)
    Real coupling_scale{0.0};
};

StokesRun runStokes(Cell cell, int n, bool check_blocks = false)
{
    const int dim = cellDimension(cell);
    const StokesExact exact{dim};
    const Real nu = 0.5;
    FESystem system(boxMesh(cell, n, boxSides()));
    const auto body_force = vectorOf("f", dim, [exact, nu](const Point& p, int i) {
        return -nu * exact.laplacian(p, i) + exact.pressureGradient(p, i);
    });
    auto p2 = std::make_shared<spaces::H1Space>(elementType(cell), 2);
    auto velocity_space = std::make_shared<spaces::ProductSpace>(p2, dim);
    auto pressure_space = std::make_shared<spaces::H1Space>(elementType(cell), 1);
    std::vector<int> walls;
    for (int face = 1; face <= 2 * dim; ++face) {
        walls.push_back(face);
    }
    const auto install = defineStokes(system, velocity_space, pressure_space, nu, body_force, walls);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " n=" << n;

    const auto u_h = FormExpr::discreteField(install.u, *velocity_space, "u");
    const auto p_h = FormExpr::discreteField(install.p, *pressure_space, "p");
    const auto eu = u_h - vectorOf("u_exact", dim, [exact](const Point& p, int i) { return exact.velocity(p, i); });
    std::vector<std::vector<FormExpr>> rows(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            rows[static_cast<std::size_t>(i)].push_back(scalarCoefficient(
                "grad_u_" + std::to_string(i) + std::to_string(j),
                [exact, i, j](const Point& p) { return exact.gradient(p, i, j); }));
        }
    }
    const auto egu = forms::grad(u_h) - FormExpr::asTensor(std::move(rows));
    const auto ep = p_h - scalarCoefficient("p_exact", [exact](const Point& p) { return exact.pressure(p); });
    StokesRun run;
    run.velocity_l2 = norm(system, forms::inner(eu, eu).dx(), solved.u);
    run.velocity_h1 = norm(system, forms::inner(egu, egu).dx(), solved.u);
    run.pressure_l2 = norm(system, (ep * ep).dx(), solved.u);
    run.pressure_mean = integrate(system, p_h.dx(), solved.u);
    run.multiplier = solved.u[static_cast<std::size_t>(system.fieldDofOffset(install.lambda))];
    if (check_blocks) {
        // -p div(v) in the momentum rows and +q div(u) in the continuity rows:
        // the coupling blocks are opposite transposes.
        const auto J = assembleDense(system, "equations", solved.u);
        const auto u0 = system.fieldDofOffset(install.u);
        const auto nu_dofs = system.fieldDofHandler(install.u).getNumDofs();
        const auto p0 = system.fieldDofOffset(install.p);
        const auto np_dofs = system.fieldDofHandler(install.p).getNumDofs();
        run.transpose_defect = 0.0;
        for (GlobalIndex a = u0; a < u0 + nu_dofs; ++a) {
            if (system.constraints().isConstrained(a)) {
                continue;
            }
            for (GlobalIndex b = p0; b < p0 + np_dofs; ++b) {
                run.transpose_defect = std::max(run.transpose_defect, std::abs(J(b, a) + J(a, b)));
                run.coupling_scale = std::max(run.coupling_scale, std::abs(J(a, b)));
            }
        }
    }
    return run;
}

TEST(GeneralityBenchmark, P04_TaylorHoodStokesThroughBoundFieldsConverges)
{
    // Three meshes in 2D; two in 3D (n = 2, 4), where the velocity H1 rate is
    // still pre-asymptotic.
    const std::vector<std::pair<Cell, int>> cases{{Cell::Tri3, 4}, {Cell::Quad4, 4}, {Cell::Tet4, 2}, {Cell::Hex8, 2}};
    for (const auto& [cell, n0] : cases) {
        const auto started = std::chrono::steady_clock::now();
        const int levels = cellDimension(cell) == 2 ? 3 : 2;
        std::vector<Real> ul2, uh1, pl2;
        for (int level = 0; level < levels; ++level) {
            const auto run = runStokes(cell, n0 << level, level == 0);
            ul2.push_back(run.velocity_l2);
            uh1.push_back(run.velocity_h1);
            pl2.push_back(run.pressure_l2);
            // The gauge: the pressure has mean zero and the multiplier
            // vanishes (the divergence of a velocity zero on the boundary
            // integrates to zero).
            EXPECT_LT(std::abs(run.pressure_mean), 1e-10) << cellName(cell);
            EXPECT_LT(std::abs(run.multiplier), 1e-9) << cellName(cell);
            if (level == 0) {
                EXPECT_GT(run.coupling_scale, 0.0) << cellName(cell);
                EXPECT_LT(run.transpose_defect, 1e-13 * run.coupling_scale) << cellName(cell);
            }
        }
        const std::string label = cellName(cell);
        RecordProperty("p04_" + label + "_seconds",
                       std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
        RecordProperty("p04_" + label + "_velocity_l2_rates", join(rates(ul2)));
        RecordProperty("p04_" + label + "_velocity_h1_rates", join(rates(uh1)));
        RecordProperty("p04_" + label + "_pressure_l2_rates", join(rates(pl2)));
        EXPECT_GT(rates(ul2).back(), 2.6) << label << " velocity L2 " << join(ul2);
        EXPECT_GT(rates(uh1).back(), cellDimension(cell) == 2 ? 1.8 : 1.5) << label << " velocity H1 " << join(uh1);
        EXPECT_GT(rates(pl2).back(), 1.7) << label << " pressure L2 " << join(pl2);
    }
}

// ===========================================================================
// 10. Mixed Cahn-Hilliard
// ===========================================================================
//
// dt(c) = div(M grad mu), mu = f'(c) - kappa lap(c) in the unit square with
// no-flux walls.  With the quadratic potential f'(c) = a c, the mode
// c = cbar + eps e^{-lambda t} cos(pi x), mu = a cbar + (a + kappa pi^2) eps
// e^{-lambda t} cos(pi x) is exact for lambda = M pi^2 (a + kappa pi^2).  With
// the double well f(c) = (c^2 - 1)^2 / 4 split convexly (c^3 implicit, -c at
// the previous step), backward Euler decreases the discrete free energy at
// every step.

struct CahnHilliardData {
    Real M{1.0};
    Real a{1.0};
    Real kappa{0.01};
    bool double_well{false};
    FormExpr c0;   ///< initial concentration
    FormExpr mu0;  ///< initial chemical potential
};

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 10.
// ---------------------------------------------------------------------------
struct CahnHilliardInstall {
    FieldId c{INVALID_FIELD_ID};
    FieldId mu{INVALID_FIELD_ID};
    Quantity mass;
    Quantity free_energy;
};

CahnHilliardInstall defineCahnHilliard(FESystem& system,
                                       const std::shared_ptr<const spaces::FunctionSpace>& V,
                                       const CahnHilliardData& d)
{
    using namespace forms;
    FieldBinding fields(system, "CahnHilliard");
    const auto C = fields.unknown("c", V);
    const auto Mu = fields.unknown("mu", V);
    const auto& c = C.state;
    const auto& v = C.test;
    const auto& mu = Mu.state;
    const auto& w = Mu.test;
    const auto M = FormExpr::constant(d.M);
    const auto a = FormExpr::constant(d.a);
    const auto kappa = FormExpr::constant(d.kappa);

    // f'(c), time-discretized: a c, or c^3 at the new step and -c at the previous one.
    const auto potential = d.double_well ? c * c * c - fields.previous(C, 1) : a * c;
    const auto energy_density = d.double_well ? 0.25 * (c * c - 1.0) * (c * c - 1.0) : 0.5 * a * c * c;
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    CahnHilliardInstall out{C.id, Mu.id, F.quantity("mass", FunctionalKind::Integral, c.dx()),
                            F.quantity("free_energy", FunctionalKind::Stored,
                                       (energy_density + 0.5 * kappa * inner(grad(c), grad(c))).dx())};
    F.record(out.mass).record(out.free_energy);
    (void)F.residual((dt(c) * v + M * inner(grad(mu), grad(v))).dx() +
                     ((mu - potential) * w - kappa * inner(grad(c), grad(w))).dx());

    // The initial state: the L2 projection of (c0, mu0).
    FormulationBuilder initial(system, "initial", {});
    (void)initial.residual(((c - d.c0) * v + (mu - d.mu0) * w).dx());
    return out;
}
// ---------------------------------------------------------------------------

struct CahnHilliardModeRun {
    Real c_l2{0.0};
    Real mu_l2{0.0};
    std::vector<Real> final_state;
    Real mass_drift{0.0};       ///< max |mass(t_n) - mass(0)|
    bool energy_decreases{true};
};

constexpr Real kModeAmplitude = 0.1;
constexpr Real kModeMean = 0.3;

CahnHilliardModeRun runCahnHilliardMode(Cell cell, int order, int n, timestepping::SchemeKind scheme, double dt,
                                        int steps)
{
    CahnHilliardData d;
    const Real lambda = d.M * kPi * kPi * (d.a + d.kappa * kPi * kPi);
    const auto mode = [](const Point& p) { return std::cos(kPi * p[0]); };
    d.c0 = scalarCoefficient("c0", [mode](const Point& p) { return kModeMean + kModeAmplitude * mode(p); });
    d.mu0 = scalarCoefficient("mu0", [mode, d](const Point& p) {
        return d.a * kModeMean + (d.a + d.kappa * kPi * kPi) * kModeAmplitude * mode(p);
    });
    FESystem system(boxMesh(cell, n, boxSides()));
    const auto V = std::make_shared<spaces::H1Space>(elementType(cell), order);
    const auto install = defineCahnHilliard(system, V, d);
    system.setup();
    const auto initial = solveSteady(system, "initial");
    const auto run = runTransient(system, "equations", initial.u, scheme, dt, steps);
    EXPECT_TRUE(run.success);
    CahnHilliardModeRun out;
    if (run.states.empty()) {
        return out;
    }
    const Real mass0 = integrate(system, install.mass.form(), initial.u);
    Real energy = integrate(system, install.free_energy.form(), initial.u);
    for (const auto& state : run.states) {
        out.mass_drift = std::max(out.mass_drift, std::abs(integrate(system, install.mass.form(), state) - mass0));
        const Real e = integrate(system, install.free_energy.form(), state);
        out.energy_decreases = out.energy_decreases && e < energy;
        energy = e;
    }
    const Real decay = std::exp(-lambda * dt * steps);
    const auto c_h = FormExpr::discreteField(install.c, *V, "c");
    const auto mu_h = FormExpr::discreteField(install.mu, *V, "mu");
    const auto ec = c_h - scalarCoefficient("c_exact", [mode, decay](const Point& p) {
                        return kModeMean + kModeAmplitude * decay * mode(p);
                    });
    const auto emu = mu_h - scalarCoefficient("mu_exact", [mode, decay, d](const Point& p) {
                         return d.a * kModeMean + (d.a + d.kappa * kPi * kPi) * kModeAmplitude * decay * mode(p);
                     });
    out.final_state = run.states.back();
    out.c_l2 = norm(system, (ec * ec).dx(), out.final_state);
    out.mu_l2 = norm(system, (emu * emu).dx(), out.final_state);
    return out;
}

TEST(GeneralityBenchmark, P10_CahnHilliardModeConvergesInSpace)
{
    // Temporal error suppressed: BDF2 with lambda dt ~ 0.01.
    for (const auto cell : {Cell::Tri3, Cell::Quad4}) {
        std::vector<Real> c_err, mu_err;
        for (const int n : {8, 16, 32}) {
            const auto run = runCahnHilliardMode(cell, 1, n, timestepping::SchemeKind::BDF2, 1e-3, 20);
            c_err.push_back(run.c_l2);
            mu_err.push_back(run.mu_l2);
            EXPECT_LT(run.mass_drift, 1e-12) << cellName(cell) << " n=" << n;
            EXPECT_TRUE(run.energy_decreases) << cellName(cell) << " n=" << n;
        }
        const std::string label = cellName(cell);
        RecordProperty("p10_" + label + "_c_l2_rates", join(rates(c_err)));
        RecordProperty("p10_" + label + "_mu_l2_rates", join(rates(mu_err)));
        EXPECT_GT(rates(c_err).back(), 1.8) << label << " c " << join(c_err);
        EXPECT_GT(rates(mu_err).back(), 1.8) << label << " mu " << join(mu_err);
    }
}

TEST(GeneralityBenchmark, P10_CahnHilliardModeConvergesInTimeAtTheSchemeOrder)
{
    // The same mesh for every step: the differences of successive halvings
    // of dt carry the temporal error alone.
    for (const auto& [scheme, order] :
         std::vector<std::pair<timestepping::SchemeKind, int>>{{timestepping::SchemeKind::BackwardEuler, 1},
                                                               {timestepping::SchemeKind::BDF2, 2}}) {
        std::vector<std::vector<Real>> finals;
        for (const int steps : {4, 8, 16, 32}) {
            finals.push_back(runCahnHilliardMode(Cell::Quad4, 1, 8, scheme, 0.08 / steps, steps).final_state);
        }
        std::vector<Real> differences;
        for (std::size_t k = 1; k < finals.size(); ++k) {
            Real m = 0.0;
            for (std::size_t i = 0; i < finals[k].size(); ++i) {
                m = std::max(m, std::abs(finals[k][i] - finals[k - 1][i]));
            }
            differences.push_back(m);
        }
        RecordProperty("p10_time_order" + std::to_string(order) + "_rates", join(rates(differences)));
        EXPECT_NEAR(rates(differences).back(), order, 0.2) << join(differences);
    }
}

TEST(GeneralityBenchmark, P10_DoubleWellConvexSplittingDecreasesItsFreeEnergy)
{
    // Spinodal data: a perturbation of the unstable mixed state c = 0.
    CahnHilliardData d;
    d.double_well = true;
    d.kappa = 0.004;
    d.c0 = scalarCoefficient("c0", [](const Point& p) {
        return 0.05 * std::cos(2.0 * kPi * p[0]) * std::cos(kPi * p[1]) + 0.03 * std::cos(3.0 * kPi * p[1]);
    });
    d.mu0 = FormExpr::constant(0.0);
    FESystem system(boxMesh(Cell::Quad4, 12, boxSides()));
    const auto V = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto install = defineCahnHilliard(system, V, d);
    system.setup();
    const auto initial = solveSteady(system, "initial");
    const auto run = runTransient(system, "equations", initial.u, timestepping::SchemeKind::BackwardEuler, 0.02, 15);
    ASSERT_TRUE(run.success);
    ASSERT_EQ(run.states.size(), 15u);
    const Real mass0 = integrate(system, install.mass.form(), initial.u);
    std::vector<Real> energies{integrate(system, install.free_energy.form(), initial.u)};
    for (const auto& state : run.states) {
        EXPECT_NEAR(integrate(system, install.mass.form(), state), mass0, 1e-12);
        energies.push_back(integrate(system, install.free_energy.form(), state));
    }
    for (std::size_t k = 1; k < energies.size(); ++k) {
        EXPECT_LE(energies[k], energies[k - 1]) << "step " << k;
    }
    RecordProperty("p10_double_well_energy", join(energies));
    // Phase separation lowers the energy measurably.
    EXPECT_LT(energies.back(), energies.front() - 1e-3);
}

// ===========================================================================
// 5. Mixed Darcy
// ===========================================================================
//
// K^{-1} q + grad p = 0, div q = f in the unit square or cube with K = I and
//
//   p = prod sin(pi x_i) - (2/pi)^dim,  q = -grad p,  f = div q,
//
// either with the pressure prescribed on the boundary (natural for the mixed
// form: + (p_D r . n) ds), or with the normal moments q . n = g prescribed
// strongly and p of mean zero through a global multiplier.  The flux of each
// cell through its faces equals its source: the balance operator below sums
// q . n over every cell's faces (the interior ones once from each side) less
// the source, a measurement apart from the equations.

Real darcyPressure(const Point& p, int dim)
{
    Real v = 1.0;
    for (int i = 0; i < dim; ++i) {
        v *= std::sin(kPi * p[static_cast<std::size_t>(i)]);
    }
    return v - std::pow(2.0 / kPi, dim);
}

Real darcyFlux(const Point& p, int dim, int a)
{
    Real v = -kPi;
    for (int i = 0; i < dim; ++i) {
        const Real t = kPi * p[static_cast<std::size_t>(i)];
        v *= i == a ? std::cos(t) : std::sin(t);
    }
    return v;
}

Real darcySource(const Point& p, int dim)
{
    Real v = dim * kPi * kPi;
    for (int i = 0; i < dim; ++i) {
        v *= std::sin(kPi * p[static_cast<std::size_t>(i)]);
    }
    return v;
}

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 5.
// ---------------------------------------------------------------------------
struct DarcyData {
    FormExpr Kinv;
    FormExpr f;
    FormExpr boundary_pressure;                          ///< p on the boundary, natural; or
    std::vector<std::pair<int, FormExpr>> normal_flux;   ///< q . n = g, strong, with p of mean zero
};

struct DarcyInstall {
    FieldId q{INVALID_FIELD_ID};
    FieldId p{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

DarcyInstall defineMixedDarcy(FESystem& system,
                              std::shared_ptr<const spaces::FunctionSpace> flux_space,
                              std::shared_ptr<const spaces::FunctionSpace> pressure_space,
                              const DarcyData& d)
{
    using namespace forms;
    FieldBinding fields(system, "MixedDarcy");
    // The pressure is declared before the flux: the measured-form evaluator,
    // and the global row built on it, take their cell rules from the first
    // unknown, which may not be an H(div) field today.
    const auto P = fields.unknown("p", std::move(pressure_space));
    const auto Q = fields.unknown("q", std::move(flux_space));
    const auto& q = Q.state;
    const auto& r = Q.test;
    const auto& p = P.state;
    const auto& s = P.test;
    const auto n = FormExpr::normal();

    auto R = (inner(d.Kinv * q, r) - p * div(r) + (div(q) - d.f) * s).dx();
    DarcyInstall out{Q.id, P.id, INVALID_FIELD_ID};
    BoundaryConditionManager bcs;
    if (d.normal_flux.empty()) {
        R = R + (d.boundary_pressure * inner(r, n)).ds();
    } else {
        for (const auto& [face, g] : d.normal_flux) {
            bcs.add(std::make_unique<bc::Dirichlet>(face, bc::Trace::Normal, g));  // q . n = g
        }
        out.lambda = fields.global("lambda", "pressure mean multiplier");
        const auto& R1 = *system.fieldRecord(out.lambda).space;
        R = R + (StateField(out.lambda, R1, "lambda") * s + TestField(out.lambda, R1, "m") * p).dx();
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, q, r, Q.id);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual(R);

    // The balance of every cell: the flux leaving through its faces less its source.
    FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    FormulationBuilder balance(system, "balance", measurement);
    (void)balance.residual((inner(avg(q), n.minus()) * jump(s)).dS() + (inner(q, n) * s).ds() - (d.f * s).dx());
    return out;
}
// ---------------------------------------------------------------------------

struct DarcyRun {
    Real flux_l2{0.0};
    Real div_l2{0.0};
    Real pressure_l2{0.0};
    Real multiplier{0.0};
    Real cell_imbalance{0.0};  ///< max over cells |flux out - source + lambda |K||
    Real cell_source{0.0};     ///< max over cells |source|, the scale
};

DarcyRun runMixedDarcy(Cell cell, int order, int n, bool strong_normal_flux)
{
    const int dim = cellDimension(cell);
    FESystem system(boxMesh(cell, n, boxSides()));
    const auto flux_space = std::make_shared<spaces::HDivSpace>(elementType(cell), order);
    const auto pressure_space = std::make_shared<spaces::L2Space>(elementType(cell), order);
    DarcyData d;
    d.Kinv = FormExpr::identity(dim);
    d.f = scalarCoefficient("f", [dim](const Point& p) { return darcySource(p, dim); });
    d.boundary_pressure = scalarCoefficient("p_wall", [dim](const Point& p) { return darcyPressure(p, dim); });
    if (strong_normal_flux) {
        for (int axis = 0; axis < dim; ++axis) {
            for (const Real sign : {-1.0, 1.0}) {
                d.normal_flux.emplace_back(sign < 0.0 ? 2 * axis + 1 : 2 * axis + 2,
                                           scalarCoefficient("g_" + std::to_string(axis) + (sign < 0.0 ? "lo" : "hi"),
                                                             [dim, axis, sign](const Point& p) {
                                                                 return sign * darcyFlux(p, dim, axis);
                                                             }));
            }
        }
    }
    const auto install = defineMixedDarcy(system, flux_space, pressure_space, d);
    // The flux errors through the rows of the discontinuous pressure.
    const auto q_h = forms::StateField(install.q, *flux_space, "q");
    const auto eq = q_h - vectorOf("q_exact", dim, [dim](const Point& p, int i) { return darcyFlux(p, dim, i); });
    const auto ediv = forms::div(q_h) - d.f;
    declareRowIntegral(system, "flux_error", forms::inner(eq, eq), install.p);
    declareRowIntegral(system, "div_error", ediv * ediv, install.p);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " RT" << order << " n=" << n;

    DarcyRun run;
    const auto p_h = FormExpr::discreteField(install.p, *pressure_space, "p");
    const auto ep = p_h - scalarCoefficient("p_exact", [dim](const Point& p) { return darcyPressure(p, dim); });
    run.flux_l2 = std::sqrt(rowIntegral(system, "flux_error", install.p, solved.u));
    run.div_l2 = std::sqrt(rowIntegral(system, "div_error", install.p, solved.u));
    run.pressure_l2 = norm(system, (ep * ep).dx(), solved.u);
    if (install.lambda != INVALID_FIELD_ID) {
        run.multiplier = solved.u[static_cast<std::size_t>(system.fieldDofOffset(install.lambda))];
    }

    // The per-cell balance: the pressure test functions of a cell sum to its indicator.
    const auto balance = assembleVector(system, "balance", solved.u);
    const auto source = assembleVector(system, "balance", std::vector<Real>(solved.u.size(), 0.0));
    const auto& map = system.fieldDofHandler(install.p).getDofMap();
    const auto offset = system.fieldDofOffset(install.p);
    const auto n_cells = static_cast<GlobalIndex>(system.mesh()->local_mesh().n_cells());
    const Real cell_volume = 1.0 / static_cast<Real>(n_cells);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        Real imbalance = 0.0;
        Real cell_source = 0.0;
        for (const auto dof : map.getCellDofs(c)) {
            imbalance += balance[static_cast<std::size_t>(offset + dof)];
            cell_source += source[static_cast<std::size_t>(offset + dof)];
        }
        run.cell_imbalance = std::max(run.cell_imbalance, std::abs(imbalance + run.multiplier * cell_volume));
        run.cell_source = std::max(run.cell_source, std::abs(cell_source));
    }
    return run;
}

void expectMixedDarcyRates(const std::vector<FamilyOrder>& cases, bool strong_normal_flux)
{
    for (const auto& c : cases) {
        std::vector<Real> eq, ediv, ep;
        for (int level = 0; level < 3; ++level) {
            const auto run = runMixedDarcy(c.cell, c.order, c.n0 << level, strong_normal_flux);
            eq.push_back(run.flux_l2);
            ediv.push_back(run.div_l2);
            ep.push_back(run.pressure_l2);
            EXPECT_LT(run.cell_imbalance, 1e-11 * run.cell_source) << cellName(c.cell) << " level " << level;
            EXPECT_LT(std::abs(run.multiplier), 1e-2) << cellName(c.cell);
        }
        const std::string label = std::string(cellName(c.cell)) + "_RT" + std::to_string(c.order) +
                                  (strong_normal_flux ? "_normal_flux" : "_pressure");
        ::testing::Test::RecordProperty("p05_" + label + "_flux_rates", join(rates(eq)));
        ::testing::Test::RecordProperty("p05_" + label + "_div_rates", join(rates(ediv)));
        ::testing::Test::RecordProperty("p05_" + label + "_pressure_rates", join(rates(ep)));
        const Real expected = c.order + 1;
        EXPECT_GT(rates(eq).back(), expected - 0.2) << label << " flux " << join(eq);
        EXPECT_GT(rates(ediv).back(), expected - 0.2) << label << " div " << join(ediv);
        EXPECT_GT(rates(ep).back(), expected - 0.2) << label << " pressure " << join(ep);
    }
}

TEST(GeneralityBenchmark, DISABLED_P05_MixedDarcyRT1DG1WithBoundaryPressure)
{
    // Defect found by this benchmark (reported; not a vocabulary gap): the
    // RT1/DG1 mixed form, installed as one residual on a Tri3 mesh built from
    // arrays, corrupts the heap during its first Newton step (glibc aborts in
    // a later allocation, inside the problem analysis after assembly).
    expectMixedDarcyRates({{Cell::Tri3, 1, 4}, {Cell::Tet4, 1, 1}}, /*strong_normal_flux=*/false);
}

TEST(GeneralityBenchmark, DISABLED_P05_MixedDarcyWithStrongNormalMomentsOnEveryFamily)
{
    // Defects found by this benchmark (reported; not vocabulary gaps):
    // - strong normal moments, bc::Dirichlet(face, bc::Trace::Normal, g) on an
    //   H(div) field of a mesh built from arrays: the constrained values ignore
    //   the global orientation of the face DOFs in 2D (the boundary fluxes of
    //   q . n = 1 on the unit square sum to 0, not 4) and in 3D carry the wrong
    //   sign and the reference face measure (they sum to -24, not 6, on the
    //   2 x 2 x 2 Kuhn cube);
    // - one install holding a DG0 field and an H(div) or H(curl) field (the
    //   RT0/DG0 pair) is assembled wrongly: zero diagonal rows of the vector
    //   field on Tri3, heap corruption in the fused cell assembly on Tet4;
    // - the RT and Nedelec mass matrices of Quad4, Hex8 and Wedge6 cells are
    //   singular, and an RT0 field on Wedge6 does not reproduce a linear flux.
    expectMixedDarcyRates({{Cell::Tri3, 0, 4}, {Cell::Quad4, 0, 4}, {Cell::Tet4, 0, 2}, {Cell::Hex8, 0, 2},
                           {Cell::Tri3, 1, 4}, {Cell::Tet4, 1, 1}},
                          /*strong_normal_flux=*/true);
}

TEST(GeneralityBenchmark, DISABLED_P05_RT1AndBDM1CannotShareOneFieldName)
{
    // Defect (reported): FieldBinding::unknown reuses a registered RT1 field
    // when asked for BDM1 under the same name; spacesCompatible compares the
    // space type, order and element type but not the basis family.
    FESystem system(boxMesh(Cell::Tri3, 2, boxSides()));
    FieldBinding fields(system, "Probe");
    const auto rt1 = std::make_shared<spaces::HDivSpace>(ElementType::Triangle3, 1);
    const auto bdm1 = std::make_shared<spaces::HDivSpace>(ElementType::Triangle3, 1, BasisType::BDM);
    EXPECT_NE(rt1->dofs_per_element(), bdm1->dofs_per_element());
    (void)fields.unknown("q", rt1);
    try {
        (void)fields.unknown("q", bdm1);
        ADD_FAILURE() << "an RT1 field was reused as BDM1";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("incompatible function space"), std::string::npos) << e.what();
    }
}

// ===========================================================================
// 6. PEC Maxwell and lossy eddy currents
// ===========================================================================
//
// curl curl E + E = f in the unit cube with E x n = 0 on the boundary and
// E = (0, 0, sin(pi x) sin(pi y)): f = (0, 0, (2 pi^2 + 1) sin(pi x) sin(pi y)).
// The eddy-current variant curl curl E + (1 + i sigma) E = f with
// E = (1 + 2i) E_0, split into real and imaginary fields.

Point maxwellField(const Point& p)
{
    return {0.0, 0.0, std::sin(kPi * p[0]) * std::sin(kPi * p[1])};
}

Point maxwellCurl(const Point& p)
{
    return {kPi * std::sin(kPi * p[0]) * std::cos(kPi * p[1]), -kPi * std::cos(kPi * p[0]) * std::sin(kPi * p[1]), 0.0};
}

/// curl curl E_0 + E_0.
Point maxwellOperator(const Point& p)
{
    return {0.0, 0.0, (2.0 * kPi * kPi + 1.0) * std::sin(kPi * p[0]) * std::sin(kPi * p[1])};
}

FormExpr vector3(const std::string& name, const std::function<Point(const Point&)>& f)
{
    return vectorOf(name, 3, [f](const Point& p, int i) { return f(p)[static_cast<std::size_t>(i)]; });
}

// ---------------------------------------------------------------------------
// The formulations: everything an author writes for problem 6.
// ---------------------------------------------------------------------------
FieldId definePerfectConductor(FESystem& system, std::shared_ptr<const spaces::FunctionSpace> V, const FormExpr& f,
                               const std::vector<int>& walls)
{
    using namespace forms;
    FieldBinding fields(system, "Maxwell");
    const auto E = fields.unknown("E", std::move(V));
    const auto& e = E.state;
    const auto& v = E.test;
    constraints::ConstraintPlan plan(system);
    for (const int wall : walls) {
        plan.declare(constraints::ConstraintPhase::Essential,
                     std::make_unique<constraints::HCurlTangentialConstraint>(E.id, wall));  // E x n = 0
    }
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual((inner(curl(e), curl(v)) + inner(e, v) - inner(f, v)).dx());
    return E.id;
}

std::pair<FieldId, FieldId> defineEddyCurrent(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& V,
                                              Real conductivity, const FormExpr& f_re, const FormExpr& f_im,
                                              const std::vector<int>& walls)
{
    using namespace forms;
    FieldBinding fields(system, "EddyCurrent");
    const auto Er = fields.unknown("E_re", V);
    const auto Ei = fields.unknown("E_im", V);
    const auto sigma = FormExpr::constant(conductivity);
    constraints::ConstraintPlan plan(system);
    for (const int wall : walls) {
        plan.declare(constraints::ConstraintPhase::Essential,
                     std::make_unique<constraints::HCurlTangentialConstraint>(Er.id, wall));
        plan.declare(constraints::ConstraintPhase::Essential,
                     std::make_unique<constraints::HCurlTangentialConstraint>(Ei.id, wall));
    }
    plan.commit();
    // curl curl E + (1 + i sigma) E = f, its real and imaginary parts.
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual((inner(curl(Er.state), curl(Er.test)) + inner(Er.state, Er.test) -
                      sigma * inner(Ei.state, Er.test) - inner(f_re, Er.test)).dx() +
                     (inner(curl(Ei.state), curl(Ei.test)) + inner(Ei.state, Ei.test) +
                      sigma * inner(Er.state, Ei.test) - inner(f_im, Ei.test)).dx());
    return {Er.id, Ei.id};
}
// ---------------------------------------------------------------------------

struct MaxwellRun {
    Real l2{0.0};
    Real curl_l2{0.0};
    int dofs_per_cell{0};
};

/**
 * A DG0 unknown, declared before the H(curl) fields and held at zero by its
 * own row of "equations": the rows that measure their errors (the
 * measured-form evaluator does not take H(curl) fields).
 */
FieldId errorCells(FESystem& system, Cell cell)
{
    FieldBinding fields(system, "ErrorCells");
    const auto C = fields.unknown("cells", std::make_shared<spaces::L2Space>(elementType(cell), 0));
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual((C.state * C.test).dx());
    return C.id;
}

MaxwellRun runPerfectConductor(Cell cell, int n, bool rotate = false)
{
    BoxOptions options;
    options.rotate_cells = rotate;
    FESystem system(boxMesh(cell, n, boxSides(), options));
    const auto cells = errorCells(system, cell);
    const auto V = std::make_shared<spaces::HCurlSpace>(elementType(cell), 0);
    const auto E_id = definePerfectConductor(system, V, vector3("f", maxwellOperator), {1, 2, 3, 4, 5, 6});
    const auto E_h = forms::StateField(E_id, *V, "E");
    const auto e = E_h - vector3("E_exact", maxwellField);
    const auto ec = forms::curl(E_h) - vector3("curl_E_exact", maxwellCurl);
    declareRowIntegral(system, "l2_error", forms::inner(e, e), cells);
    declareRowIntegral(system, "curl_error", forms::inner(ec, ec), cells);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " n=" << n;
    MaxwellRun run;
    run.l2 = std::sqrt(rowIntegral(system, "l2_error", cells, solved.u));
    run.curl_l2 = std::sqrt(rowIntegral(system, "curl_error", cells, solved.u));
    run.dofs_per_cell = static_cast<int>(V->dofs_per_element());
    return run;
}

TEST(GeneralityBenchmark, DISABLED_P06_PerfectConductorMaxwellConvergesInHCurl)
{
    // Defects found by this benchmark (reported; not vocabulary gaps): on a
    // Tet4 mesh built from arrays the lowest-order Nedelec field reproduces
    // E = (-y, x, 0) exactly, but curl(E) of the field evaluates to zero in the
    // assembled kernels (the measured curl error is |curl E| = 2), so the
    // curl-curl term vanishes and the solution is the L2 projection of f; the
    // Hex8 Nedelec mass matrix is singular.
    for (const auto cell : {Cell::Tet4, Cell::Hex8}) {
        std::vector<Real> l2, curl_l2;
        for (const int n : {2, 4, 8}) {
            const auto run = runPerfectConductor(cell, n);
            EXPECT_EQ(run.dofs_per_cell, cell == Cell::Tet4 ? 6 : 12) << "lowest-order Nedelec";
            l2.push_back(run.l2);
            curl_l2.push_back(run.curl_l2);
        }
        std::vector<Real> hcurl;
        for (std::size_t i = 0; i < l2.size(); ++i) {
            hcurl.push_back(std::sqrt(l2[i] * l2[i] + curl_l2[i] * curl_l2[i]));
        }
        const std::string label = cellName(cell);
        RecordProperty("p06_" + label + "_hcurl_rates", join(rates(hcurl)));
        RecordProperty("p06_" + label + "_l2_rates", join(rates(l2)));
        EXPECT_GT(rates(hcurl).back(), 0.85) << label << " " << join(hcurl);
        EXPECT_GT(rates(l2).back(), 0.85) << label << " " << join(l2);

        // Other local edge orientations give the same discrete field.
        const auto turned = runPerfectConductor(cell, 2, /*rotate=*/true);
        EXPECT_NEAR(turned.l2, l2.front(), 1e-10 * l2.front()) << label;
        EXPECT_NEAR(turned.curl_l2, curl_l2.front(), 1e-10 * curl_l2.front()) << label;
    }
}

TEST(GeneralityBenchmark, DISABLED_P06_LossyEddyCurrentsSolveBothPartsWithAntisymmetricCoupling)
{
    // Blocked by the same curl defect as the perfect conductor above.
    const Real sigma = 3.0;
    const auto f_re = vector3("f_re", [sigma](const Point& p) {
        const auto g = maxwellOperator(p);
        const auto e = maxwellField(p);
        return Point{g[0] - 2.0 * sigma * e[0], g[1] - 2.0 * sigma * e[1], g[2] - 2.0 * sigma * e[2]};
    });
    const auto f_im = vector3("f_im", [sigma](const Point& p) {
        const auto g = maxwellOperator(p);
        const auto e = maxwellField(p);
        return Point{2.0 * g[0] + sigma * e[0], 2.0 * g[1] + sigma * e[1], 2.0 * g[2] + sigma * e[2]};
    });
    std::vector<Real> re_err, im_err;
    for (const int n : {2, 4, 8}) {
        FESystem system(boxMesh(Cell::Tet4, n, boxSides()));
        const auto cells = errorCells(system, Cell::Tet4);
        const auto V = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, 0);
        const auto [er, ei] = defineEddyCurrent(system, V, sigma, f_re, f_im, {1, 2, 3, 4, 5, 6});
        const auto er_h = forms::StateField(er, *V, "E_re");
        const auto ei_h = forms::StateField(ei, *V, "E_im");
        const auto dr = er_h - vector3("E_re_exact", maxwellField);
        const auto di = ei_h - vector3("E_im_exact", [](const Point& p) {
                            const auto e = maxwellField(p);
                            return Point{2.0 * e[0], 2.0 * e[1], 2.0 * e[2]};
                        });
        const auto cr = forms::curl(er_h) - vector3("curl_re", maxwellCurl);
        const auto ci = forms::curl(ei_h) - vector3("curl_im", [](const Point& p) {
                            const auto c = maxwellCurl(p);
                            return Point{2.0 * c[0], 2.0 * c[1], 2.0 * c[2]};
                        });
        declareRowIntegral(system, "re_error", forms::inner(dr, dr) + forms::inner(cr, cr), cells);
        declareRowIntegral(system, "im_error", forms::inner(di, di) + forms::inner(ci, ci), cells);
        system.setup();
        const auto solved = solveSteady(system, "equations");
        EXPECT_TRUE(solved.converged);
        re_err.push_back(std::sqrt(rowIntegral(system, "re_error", cells, solved.u)));
        im_err.push_back(std::sqrt(rowIntegral(system, "im_error", cells, solved.u)));
        if (n == 2) {
            // The coupling blocks: -sigma M in the real rows, +sigma M in the
            // imaginary rows, opposite transposes of each other.
            const auto J = assembleDense(system, "equations", solved.u);
            const auto r0 = system.fieldDofOffset(er);
            const auto i0 = system.fieldDofOffset(ei);
            const auto m = system.fieldDofHandler(er).getNumDofs();
            Real defect = 0.0;
            Real scale = 0.0;
            Real sign = 0.0;
            for (GlobalIndex a = 0; a < m; ++a) {
                if (system.constraints().isConstrained(r0 + a) || system.constraints().isConstrained(i0 + a)) {
                    continue;
                }
                sign += J(r0 + a, i0 + a);  // -sigma times a mass diagonal entry
                for (GlobalIndex b = 0; b < m; ++b) {
                    if (system.constraints().isConstrained(i0 + b) || system.constraints().isConstrained(r0 + b)) {
                        continue;
                    }
                    defect = std::max(defect, std::abs(J(r0 + a, i0 + b) + J(i0 + b, r0 + a)));
                    scale = std::max(scale, std::abs(J(r0 + a, i0 + b)));
                }
            }
            EXPECT_GT(scale, 0.0);
            EXPECT_LT(sign, 0.0);
            EXPECT_LT(defect, 1e-13 * scale);
        }
    }
    RecordProperty("p06_eddy_re_hcurl_rates", join(rates(re_err)));
    RecordProperty("p06_eddy_im_hcurl_rates", join(rates(im_err)));
    EXPECT_GT(rates(re_err).back(), 0.85) << join(re_err);
    EXPECT_GT(rates(im_err).back(), 0.85) << join(im_err);
}

// ===========================================================================
// 7. Clamped C0 interior-penalty plate
// ===========================================================================
//
// lap^2 u = f in the unit square, u = 0 and du/dn = 0 on the boundary, with
// u = x^2 (1-x)^2 y^2 (1-y)^2.  Continuous P2 (Q2) with the interior-penalty
// terms on the slope jumps, the clamped slope imposed by the same terms on
// the boundary, and u = 0 strongly.  The broken H^2 seminorm error is O(h).

struct Quartic {
    Real v, d1, d2, d3, d4;
};

Quartic bump(Real t)
{
    return {t * t * (1 - t) * (1 - t), 2 * t - 6 * t * t + 4 * t * t * t, 2 - 12 * t + 12 * t * t, -12 + 24 * t, 24.0};
}

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 7.
// ---------------------------------------------------------------------------
FieldId defineClampedPlate(FESystem& system, std::shared_ptr<const spaces::FunctionSpace> V, const FormExpr& f,
                           Real eta, const std::vector<int>& walls)
{
    using namespace forms;
    FieldBinding fields(system, "ClampedPlate");
    const auto U = fields.unknown("u", std::move(V));
    const auto& u = U.state;
    const auto& v = U.test;
    const auto n = FormExpr::normal();
    // Slope jumps along the minus side's normal: grad(u)(-).n(-) + grad(u)(+).n(+).
    const auto ju = inner(jump(grad(u)), n.minus());
    const auto jv = inner(jump(grad(v)), n.minus());
    const auto penalty = FormExpr::constant(eta);
    auto R = (laplacian(u) * laplacian(v) - f * v).dx() +
             (-avg(laplacian(u)) * jv - avg(laplacian(v)) * ju + penalty / avg(hNormal()) * ju * jv).dS();
    BoundaryConditionManager bcs;
    for (const int wall : walls) {
        const auto su = inner(grad(u), n);
        const auto sv = inner(grad(v), n);
        R = R + (-laplacian(u) * sv - laplacian(v) * su + penalty / hNormal() * su * sv).ds(wall);  // du/dn = 0
        bcs.add(std::make_unique<bc::Dirichlet>(wall, FormExpr::constant(0.0)));                    // u = 0
    }
    constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, u, v, U.id);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    (void)F.residual(R);
    return U.id;
}
// ---------------------------------------------------------------------------

struct PlateRun {
    Real h2{0.0};
    Real l2{0.0};
    Real asymmetry{-1.0};
    Real scale{0.0};
};

PlateRun runClampedPlate(Cell cell, int n, bool reverse = false, bool check_symmetry = false)
{
    BoxOptions options;
    options.reverse_cells = reverse;
    FESystem system(boxMesh(cell, n, boxSides(), options));
    const auto V = std::make_shared<spaces::H1Space>(elementType(cell), 2);
    const auto f = scalarCoefficient("f", [](const Point& p) {
        const auto X = bump(p[0]), Y = bump(p[1]);
        return X.d4 * Y.v + 2.0 * X.d2 * Y.d2 + X.v * Y.d4;
    });
    const auto u_id = defineClampedPlate(system, V, f, 16.0, {1, 2, 3, 4});
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " n=" << n;
    const auto u_h = FormExpr::discreteField(u_id, *V, "u");
    const auto e = u_h - scalarCoefficient("u_exact", [](const Point& p) { return bump(p[0]).v * bump(p[1]).v; });
    const auto H = forms::as_tensor(
        {{scalarCoefficient("u_xx", [](const Point& p) { return bump(p[0]).d2 * bump(p[1]).v; }),
          scalarCoefficient("u_xy", [](const Point& p) { return bump(p[0]).d1 * bump(p[1]).d1; })},
         {scalarCoefficient("u_yx", [](const Point& p) { return bump(p[0]).d1 * bump(p[1]).d1; }),
          scalarCoefficient("u_yy", [](const Point& p) { return bump(p[0]).v * bump(p[1]).d2; })}});
    const auto eh = forms::hessian(u_h) - H;
    PlateRun run;
    run.h2 = norm(system, forms::inner(eh, eh).dx(), solved.u);
    run.l2 = norm(system, (e * e).dx(), solved.u);
    if (check_symmetry) {
        const auto J = assembleDense(system, "equations", solved.u);
        run.asymmetry = 0.0;
        for (GlobalIndex a = 0; a < J.n; ++a) {
            for (GlobalIndex b = 0; b < J.n; ++b) {
                if (system.constraints().isConstrained(a) || system.constraints().isConstrained(b)) {
                    continue;
                }
                run.asymmetry = std::max(run.asymmetry, std::abs(J(a, b) - J(b, a)));
                run.scale = std::max(run.scale, std::abs(J(a, b)));
            }
        }
    }
    return run;
}

TEST(GeneralityBenchmark, DISABLED_P07_ClampedInteriorPenaltyPlateConvergesInBrokenH2)
{
    // Defect found by this benchmark (reported; not a vocabulary gap): on
    // interior facets the residual of avg(laplacian(u)) * jv evaluates to
    // zero for a P2 or Q2 state, while its (symbolic) tangent does not; the
    // consistency terms are missing from the residual, Newton does not
    // converge and the error does not decrease.  The cell Hessian, the slope
    // jumps and the boundary slope terms assemble.
    for (const auto cell : {Cell::Tri3, Cell::Quad4}) {
        std::vector<Real> h2, l2;
        for (const int n : {4, 8, 16}) {
            const auto run = runClampedPlate(cell, n, false, n == 4);
            h2.push_back(run.h2);
            l2.push_back(run.l2);
            if (n == 4) {
                EXPECT_GT(run.scale, 0.0);
                EXPECT_LT(run.asymmetry, 1e-12 * run.scale) << cellName(cell);
                // Facet reversal: the cells in reverse order swap the sides of
                // every interior facet and give the same discrete solution.
                const auto reversed = runClampedPlate(cell, n, /*reverse=*/true);
                EXPECT_NEAR(reversed.h2, run.h2, 1e-9 * run.h2) << cellName(cell);
                EXPECT_NEAR(reversed.l2, run.l2, 1e-9 * run.l2) << cellName(cell);
            }
        }
        const std::string label = cellName(cell);
        RecordProperty("p07_" + label + "_h2_rates", join(rates(h2)));
        RecordProperty("p07_" + label + "_l2_rates", join(rates(l2)));
        EXPECT_GT(rates(h2).back(), 0.85) << label << " " << join(h2);
        EXPECT_GT(rates(l2).back(), 1.7) << label << " " << join(l2);
    }
}

// ===========================================================================
// 11. Conjugate heat transfer across a matching interface
// ===========================================================================
//
// Two slabs A (x < 1/2, conductivity k_A) and B (x > 1/2, k_B) exchange heat
// through a finite resistance r on Gamma = {x = 1/2}:
//
//   (k_A grad T_A . grad v_A) dx(A) + (k_B grad T_B . grad v_B) dx(B)
//     + ((T_A - T_B) (v_A - v_B) / r) dI(Gamma),
//
// with T given at x = 0 and x = 1 and insulated elsewhere.  The exact patch
// T_A = L - q (x - s) / k_A, T_B = L - r q - q (x - s) / k_B has unequal
// conductivities, a temperature jump r q and the same heat q on both sides;
// the curved variant adds y-dependence and curvature with sources.

constexpr int kSlabA = 1;
constexpr int kSlabB = 2;
constexpr int kGamma = 31;

struct SlabData {
    Real L{1.0}, q0{0.6}, kA{2.0}, kB{0.5}, r{0.4}, s{0.5};
    bool curved{false};
    Real beta{0.3}, gamma{-0.7};

    [[nodiscard]] Real q(Real y) const { return curved ? q0 * (1.0 + 0.5 * std::cos(kPi * y)) : q0; }
    [[nodiscard]] Real w(Real y) const { return curved ? 0.2 * std::cos(kPi * y) : 0.0; }
    [[nodiscard]] Real TA(const Point& p) const
    {
        const Real d = p[0] - s;
        return L + w(p[1]) - q(p[1]) * d / kA + (curved ? beta * d * d : 0.0);
    }
    [[nodiscard]] Real TB(const Point& p) const
    {
        const Real d = p[0] - s;
        return L + w(p[1]) - r * q(p[1]) - q(p[1]) * d / kB + (curved ? gamma * d * d : 0.0);
    }
    /// -k lap(T) on each side.
    [[nodiscard]] Real fA(const Point& p) const
    {
        if (!curved) {
            return 0.0;
        }
        const Real d = p[0] - s;
        const Real w2 = -0.2 * kPi * kPi * std::cos(kPi * p[1]);
        const Real q2 = -0.5 * q0 * kPi * kPi * std::cos(kPi * p[1]);
        return -kA * (2.0 * beta + w2 - q2 * d / kA);
    }
    [[nodiscard]] Real fB(const Point& p) const
    {
        if (!curved) {
            return 0.0;
        }
        const Real d = p[0] - s;
        const Real w2 = -0.2 * kPi * kPi * std::cos(kPi * p[1]);
        const Real q2 = -0.5 * q0 * kPi * kPi * std::cos(kPi * p[1]);
        return -kB * (2.0 * gamma + w2 - r * q2 - q2 * d / kB);
    }
};

// ---------------------------------------------------------------------------
// The formulation: everything an author writes for problem 11.
// ---------------------------------------------------------------------------
struct ConjugateHeatInstall {
    FieldId TA{INVALID_FIELD_ID};
    FieldId TB{INVALID_FIELD_ID};
    Quantity heat_A_to_B;
};

ConjugateHeatInstall defineConjugateHeat(FESystem& system, const std::shared_ptr<const spaces::FunctionSpace>& V,
                                         const SlabData& d, const FormExpr& fA, const FormExpr& fB,
                                         const FormExpr& TA_wall, const FormExpr& TB_wall)
{
    using namespace forms;
    FieldBinding fields(system, "ConjugateHeat");
    const auto A = fields.unknown("T_A", V);
    const auto B = fields.unknown("T_B", V);
    const auto OA = VolumeDomain::cells(kSlabA);
    const auto OB = VolumeDomain::cells(kSlabB);
    const auto G = interfaceBetween(system, OA, OB, kGamma);
    const auto kA = FormExpr::constant(d.kA);
    const auto kB = FormExpr::constant(d.kB);
    const auto r = FormExpr::constant(d.r);
    const auto jumpT = G.trace(A.state, OA) - G.trace(B.state, OB);
    const auto jumpV = G.trace(A.test, OA) - G.trace(B.test, OB);

    auto R = (kA * inner(grad(A.state), grad(A.test)) - fA * A.test).dx(OA) +
             (kB * inner(grad(B.state), grad(B.test)) - fB * B.test).dx(OB) + (jumpT * jumpV / r).dI(G.domain());
    BoundaryConditionManager wallA;
    wallA.add(std::make_unique<bc::Dirichlet>(1, TA_wall));
    BoundaryConditionManager wallB;
    wallB.add(std::make_unique<bc::Dirichlet>(2, TB_wall));
    constraints::ConstraintPlan plan(system);
    plan.essentials(wallA, R, A.state, A.test, A.id);
    plan.essentials(wallB, R, B.state, B.test, B.id);
    plan.activeOn({A.id}, OA).activeOn({B.id}, OB);
    plan.commit();
    FormulationBuilder F(system, "equations", {});
    F.symbolicTangent();
    ConjugateHeatInstall out{A.id, B.id, F.quantity("heat_A_to_B", FunctionalKind::Integral, (jumpT / r).dI(G.domain()))};
    F.record(out.heat_A_to_B);
    (void)F.residual(R);
    // The heat crossing Gamma through the rows of T_A (a measurement): the
    // measured-form evaluator takes dI on generated interfaces only.
    FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    FormulationBuilder exchange(system, "exchange", measurement);
    (void)exchange.residual((jumpT * G.trace(A.test, OA) / r).dI(G.domain()));
    return out;
}
// ---------------------------------------------------------------------------

struct ConjugateHeatRun {
    Real errA{0.0};
    Real errB{0.0};
    Real heat{0.0};
    Real scale{0.0};
};

ConjugateHeatRun runConjugateHeat(Cell cell, int n, bool curved)
{
    SlabData d;
    d.curved = curved;
    BoxOptions options;
    options.region = [d](const Point& c) { return c[0] < d.s ? kSlabA : kSlabB; };
    options.interface_faces = [d](const Point& c) { return std::abs(c[0] - d.s) < 1e-9; };
    FESystem system(boxMesh(cell, n, boxSides(), options));
    system.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    const auto V = std::make_shared<spaces::H1Space>(elementType(cell), 1);
    const auto exactA = scalarCoefficient("T_A_exact", [d](const Point& p) { return d.TA(p); });
    const auto exactB = scalarCoefficient("T_B_exact", [d](const Point& p) { return d.TB(p); });
    const auto install = defineConjugateHeat(system, V, d, scalarCoefficient("f_A", [d](const Point& p) { return d.fA(p); }),
                                             scalarCoefficient("f_B", [d](const Point& p) { return d.fB(p); }), exactA,
                                             exactB);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_TRUE(solved.converged) << cellName(cell) << " n=" << n;
    const auto TA_h = FormExpr::discreteField(install.TA, *V, "T_A");
    const auto TB_h = FormExpr::discreteField(install.TB, *V, "T_B");
    ConjugateHeatRun run;
    run.errA = norm(system, ((TA_h - exactA) * (TA_h - exactA)).dx(forms::VolumeDomain::cells(kSlabA)), solved.u);
    run.errB = norm(system, ((TB_h - exactB) * (TB_h - exactB)).dx(forms::VolumeDomain::cells(kSlabB)), solved.u);
    const auto rows = assembleVector(system, "exchange", solved.u);
    const auto offset = system.fieldDofOffset(install.TA);
    for (GlobalIndex i = offset; i < offset + system.fieldDofHandler(install.TA).getNumDofs(); ++i) {
        if (!system.constraints().isConstrained(i)) {
            run.heat += rows[static_cast<std::size_t>(i)];
        }
    }
    run.scale = d.L;
    return run;
}

TEST(GeneralityBenchmark, P11_ConjugateHeatExactPatchOnEveryFamily)
{
    const SlabData d;
    for (const auto cell : {Cell::Tri3, Cell::Quad4, Cell::Tet4, Cell::Hex8, Cell::Wedge6, Cell::Pyr5}) {
        const auto run = runConjugateHeat(cell, 2, /*curved=*/false);
        EXPECT_LT(run.errA, 1e-12) << cellName(cell);
        EXPECT_LT(run.errB, 1e-12) << cellName(cell);
        // The same heat q crosses the unit interface.
        EXPECT_NEAR(run.heat, d.q0, 1e-12) << cellName(cell);
    }
}

TEST(GeneralityBenchmark, DISABLED_P11_TheDeclaredExchangeQuantityIsMeasured)
{
    // Missing: the measured-form evaluator takes dI only on a generated
    // (cut) interface and one side at a time; the quantity
    // F.quantity("heat_A_to_B", ..., dI(Gamma)) of a fitted interface (an
    // InterfaceMesh from a face set) is refused ("a cut measure ... needs a
    // cut integration context"), and the same two-sided integrand on a
    // generated interface throws ("requested plus-side context in non-DG
    // evaluation"), so an exchange cannot be recorded through its own
    // quantity (problem 13 measures it through the rows instead).
    const SlabData d;
    BoxOptions options;
    options.region = [d](const Point& c) { return c[0] < d.s ? kSlabA : kSlabB; };
    options.interface_faces = [d](const Point& c) { return std::abs(c[0] - d.s) < 1e-9; };
    FESystem system(boxMesh(Cell::Quad4, 2, boxSides(), options));
    system.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    const auto V = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto exactA = scalarCoefficient("T_A_exact", [d](const Point& p) { return d.TA(p); });
    const auto exactB = scalarCoefficient("T_B_exact", [d](const Point& p) { return d.TB(p); });
    const auto install = defineConjugateHeat(system, V, d, FormExpr::constant(0.0), FormExpr::constant(0.0), exactA, exactB);
    system.setup();
    const auto solved = solveSteady(system, "equations");
    EXPECT_NEAR(integrate(system, install.heat_A_to_B.form(), solved.u), d.q0, 1e-12);
}

TEST(GeneralityBenchmark, P11_ConjugateHeatCurvedProfilesConverge)
{
    for (const auto& [cell, n0] : std::vector<std::pair<Cell, int>>{{Cell::Tri3, 4}, {Cell::Quad4, 4}, {Cell::Tet4, 2}, {Cell::Hex8, 2}}) {
        std::vector<Real> ea, eb;
        for (int level = 0; level < 3; ++level) {
            const auto run = runConjugateHeat(cell, n0 << level, /*curved=*/true);
            ea.push_back(run.errA);
            eb.push_back(run.errB);
        }
        const std::string label = cellName(cell);
        RecordProperty("p11_" + label + "_A_rates", join(rates(ea)));
        RecordProperty("p11_" + label + "_B_rates", join(rates(eb)));
        EXPECT_GT(rates(ea).back(), 1.75) << label << " " << join(ea);
        EXPECT_GT(rates(eb).back(), 1.75) << label << " " << join(eb);
    }
}

} // namespace
} // namespace svmp::FE::systems::generality
