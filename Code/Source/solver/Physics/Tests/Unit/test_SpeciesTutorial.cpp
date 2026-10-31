/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_SpeciesTutorial.cpp
 * @brief A new physics in a few lines: the species of review 07 (section 3.2)
 *
 * An advected, diffusing and reacting species c,
 *
 *   R(c; w) = ((u . grad c + k c - f) w + kappa grad c . grad w) dx,  c = 0 on the boundary,
 *
 * written with FieldBinding::unknown (the components follow the space) and
 * FormulationBuilder::residual (the fields of the install follow the
 * residual).  Checked with a prescribed velocity against a manufactured
 * solution (P1 on quadrilaterals, L2 error rate 2), and with the velocity
 * an unknown of the same system, where the residual's dependence on u must
 * give the species rows their velocity columns: the assembled Jacobian
 * equals central differences of the residual.
 */

#include <gtest/gtest.h>

#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FieldBinding.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/FunctionalFormEvaluator.h"
#include "FE/Systems/SystemCollectives.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {
namespace {

using FE::Real;
using FE::forms::FormExpr;

constexpr int kWall = 7;

/// The unit square in n x n Q1 cells, its boundary labelled kWall.
std::shared_ptr<Mesh> unitSquare(int n)
{
    auto base = std::make_shared<MeshBase>();
    std::vector<real_t> x;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            x.push_back(static_cast<real_t>(i) / static_cast<real_t>(n));
            x.push_back(static_cast<real_t>(j) / static_cast<real_t>(n));
        }
    }
    std::vector<offset_t> offsets{0};
    std::vector<index_t> cells;
    const auto vid = [n](int i, int j) { return static_cast<index_t>(j * (n + 1) + i); };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            cells.insert(cells.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
            offsets.push_back(static_cast<offset_t>(cells.size()));
        }
    }
    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(2, x, offsets, cells, std::vector<CellShape>(static_cast<std::size_t>(n * n), shape));
    base->finalize();
    base->register_label("wall", kWall);
    for (index_t f = 0; f < static_cast<index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        const auto a = base->get_vertex_coords(verts[0]);
        const auto b = base->get_vertex_coords(verts[1]);
        const auto on = [](real_t p, real_t q, real_t side) { return p == side && q == side; };
        if (on(a[0], b[0], 0) || on(a[0], b[0], 1) || on(a[1], b[1], 0) || on(a[1], b[1], 1)) {
            base->set_boundary_label(f, kWall);
            base->add_to_set(EntityKind::Face, "wall", f);
        }
    }
    return create_mesh(std::move(base));
}

struct SpeciesData {
    Real kappa{0.1};
    Real k{1.0};
    FormExpr velocity{};  ///< prescribed (a vector expression) or an unknown's state
    FormExpr source{};
};

// ---------------------------------------------------------------------------
// The formulation: everything a developer writes for the species.
// ---------------------------------------------------------------------------
FE::FieldId defineSpecies(FE::systems::FESystem& S, const SpeciesData& d)
{
    using namespace FE::forms;
    FE::systems::FieldBinding fields(S, "Species");
    const auto C = fields.unknown("Concentration", std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, 1));
    const auto& c = C.state;
    const auto& w = C.test;
    const auto kappa = FormExpr::constant(d.kappa);
    const auto k = FormExpr::constant(d.k);
    const auto R = ((inner(d.velocity, grad(c)) + k * c - d.source) * w + kappa * inner(grad(c), grad(w))).dx();
    const std::array<bc::StrongDirichlet, 1> wall{bc::strongDirichlet(C.id, kWall, FormExpr::constant(0.0), "c")};
    FE::systems::installStrongDirichlet(S, std::span<const bc::StrongDirichlet>(wall));
    FE::systems::FormulationBuilder F(S, "equations", {});
    (void)F.residual(R);
    return C.id;
}
// ---------------------------------------------------------------------------

std::vector<Real> solveDense(std::vector<Real> a, std::vector<Real> b)
{
    const auto n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i * n + k]) > std::abs(a[pivot * n + k])) {
                pivot = i;
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            std::swap(a[k * n + j], a[pivot * n + j]);
        }
        std::swap(b[k], b[pivot]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real f = a[i * n + k] / a[k * n + k];
            for (std::size_t j = k; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    std::vector<Real> x(n, 0.0);
    for (std::size_t k = n; k-- > 0;) {
        Real s = b[k];
        for (std::size_t j = k + 1; j < n; ++j) {
            s -= a[k * n + j] * x[j];
        }
        x[k] = s / a[k * n + k];
    }
    return x;
}

/// The L2 error of the species on n x n cells against c = 16 x (1-x) y (1-y).
Real speciesError(int n)
{
    auto mesh = unitSquare(n);
    FE::systems::FESystem S(mesh);
    const auto x = FormExpr::coordinate().component(0);
    const auto y = FormExpr::coordinate().component(1);
    const auto one = FormExpr::constant(1.0);
    const auto c_exact = FormExpr::constant(16.0) * x * (one - x) * y * (one - y);
    const auto cx = FormExpr::constant(16.0) * (one - FormExpr::constant(2.0) * x) * y * (one - y);
    const auto cy = FormExpr::constant(16.0) * x * (one - x) * (one - FormExpr::constant(2.0) * y);
    const auto laplacian = FormExpr::constant(-32.0) * (y * (one - y) + x * (one - x));
    SpeciesData d;
    const Real ux = 1.0;
    const Real uy = 0.5;
    d.velocity = FE::forms::as_vector({FormExpr::constant(ux), FormExpr::constant(uy)});
    d.source = FormExpr::constant(-d.kappa) * laplacian + FormExpr::constant(ux) * cx +
               FormExpr::constant(uy) * cy + FormExpr::constant(d.k) * c_exact;
    const auto c_id = defineSpecies(S, d);
    S.setup();

    // Linear: one Newton step from zero.
    const auto n_dofs = S.dofHandler().getNumDofs();
    std::vector<Real> zero(static_cast<std::size_t>(n_dofs), 0.0);
    FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    FE::systems::SystemStateView state;
    state.u = zero;
    FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_matrix = true;
    request.want_vector = true;
    EXPECT_TRUE(S.assemble(request, state, &out, &out).success);
    std::vector<Real> rhs(out.vectorData().begin(), out.vectorData().end());
    for (auto& r : rhs) {
        r = -r;
    }
    const auto solution = solveDense({out.matrixData().begin(), out.matrixData().end()}, rhs);

    const auto c_h = FormExpr::discreteField(c_id, *S.fieldRecord(c_id).space, "c");
    state.u = solution;
    const std::vector<FormExpr> error{((c_h - c_exact) * (c_h - c_exact)).dx()};
    return std::sqrt(FE::systems::evaluateFunctionalForms(S, error, state, FE::systems::CollectiveCommunicator{})
                         .front()
                         .value);
}

} // namespace

TEST(SpeciesTutorial, PrescribedVelocityConvergesAtSecondOrder)
{
    const Real e8 = speciesError(8);
    const Real e16 = speciesError(16);
    const Real e32 = speciesError(32);
    RecordProperty("species_l2_error_h8", std::to_string(e8));
    RecordProperty("species_l2_error_h16", std::to_string(e16));
    RecordProperty("species_l2_error_h32", std::to_string(e32));
    EXPECT_NEAR(std::log2(e8 / e16), 2.0, 0.15) << e8 << " " << e16;
    EXPECT_NEAR(std::log2(e16 / e32), 2.0, 0.1) << e16 << " " << e32;
    EXPECT_LT(e32, 2e-3);
}

TEST(SpeciesTutorial, AnUnknownVelocityGivesTheSpeciesRowsTheirVelocityColumns)
{
    auto mesh = unitSquare(3);
    FE::systems::FESystem S(mesh);
    FE::systems::FieldBinding fields(S, "Velocity");
    auto scalar = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, 1);
    const auto U = fields.unknown("Velocity", std::make_shared<FE::spaces::ProductSpace>(scalar, 2));
    EXPECT_EQ(S.fieldRecord(U.id).components, 2);  // the components follow the space

    SpeciesData d;
    d.velocity = U.state;
    d.source = FormExpr::constant(1.0);
    {
        // The velocity's own rows (a mass, enough for the check).
        FE::systems::FormulationBuilder F(S, "equations", {});
        (void)F.residual(FE::forms::inner(U.state, U.test).dx());
    }
    const auto c_id = defineSpecies(S, d);

    // The species install infers {c} as test field and the velocity as an
    // extra trial field.
    const auto species = FE::forms::inner(U.state, FE::forms::grad(FormExpr::stateField(
                                                        c_id, *S.fieldRecord(c_id).space, "c"))) *
                         FormExpr::testFunction(c_id, *S.fieldRecord(c_id).space, "w");
    const auto inferred = FE::systems::inferInstallFields(S, species.dx());
    EXPECT_EQ(inferred.test_fields, std::vector<FE::FieldId>{c_id});
    EXPECT_EQ(inferred.extra_trial_fields, std::vector<FE::FieldId>{U.id});

    S.setup();
    const auto n = S.dofHandler().getNumDofs();
    std::vector<Real> u(static_cast<std::size_t>(n));
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        u[static_cast<std::size_t>(i)] = 0.3 + 0.1 * std::sin(1.7 * static_cast<Real>(i));
    }
    FE::systems::SystemStateView state;
    state.u = u;
    expectOperatorJacobianMatchesCentralFD(S, state, "equations", /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-9);
}

} // namespace test
} // namespace Physics
} // namespace svmp
