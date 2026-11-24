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

} // namespace
} // namespace svmp::FE::systems::generality
