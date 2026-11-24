/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Authoring.cpp
 * @brief The canonical formulation skeleton, compiled with Forms/Authoring.h alone
 *
 * The skeleton below is the one of FE/README.md (Example 1),
 * Forms/VOCABULARY.md (Canonical Workflow) and Forms/SYSTEMS_INTEGRATION.md
 * (section 2), letter for letter apart from the include path.  It comes
 * before every other include, so it compiles only with what
 * Forms/Authoring.h declares; the test then registers it on a mesh, sets the
 * system up and assembles it.
 */

#include "Forms/Authoring.h"

// ---- The canonical skeleton (keep equal to the documents above) ----
namespace fe = svmp::FE;
using namespace fe::forms;

struct HeatOptions {
    fe::Real conductivity{1.0};
    fe::Real source{0.0};
    int wall{1};                    // T = wall_temperature
    fe::Real wall_temperature{0.0};
    int inlet{2};                   // k grad(T).n = inflow
    fe::Real inflow{0.0};
};

// -div(k grad T) = f in the domain, T = T_w on the wall, k grad(T).n = q on the inlet.
void registerHeat(fe::systems::FESystem& system,
                  std::shared_ptr<const fe::spaces::FunctionSpace> V,
                  const HeatOptions& o)
{
    fe::systems::FieldBinding fields(system, "Heat::registerOn");
    const auto U = fields.unknown("T", V, "temperature");
    const auto& T = U.state;
    const auto& v = U.test;

    const auto k = FormExpr::constant(o.conductivity);
    const auto f = FormExpr::constant(o.source);
    auto R = (k * inner(grad(T), grad(v)) - f * v).dx();

    fe::systems::BoundaryConditionManager bcs;
    bcs.add(std::make_unique<bc::Dirichlet>(o.wall, FormExpr::constant(o.wall_temperature)));
    bcs.add(std::make_unique<bc::NaturalBC>(o.inlet, FormExpr::constant(o.inflow)));

    fe::constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, T, v, U.id);
    plan.commit();

    fe::systems::FormulationBuilder F(system, "equations", {});
    F.record(F.quantity("conduction_energy", fe::systems::FunctionalKind::Stored,
                        (0.5 * k * inner(grad(T), grad(T))).dx()));
    F.symbolicTangent();
    (void)F.residual(R);

    system.addDerivedResult(fe::post::DerivedResultBuilder("heat_flux")
                                .scope(fe::post::DerivedResultScope::Cell)
                                .policy(fe::post::DerivedResultPolicy::CellAverage)
                                .shape(fe::systems::FEQuantityShape::vector(V->topological_dimension()))
                                .expression(-k * grad(T))
                                .referencedField(U.id)
                                .build());
}
// ---- end of the canonical skeleton ----

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Spaces/H1Space.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Systems/TraceMultiplierTestMeshes.h"

#include <cmath>
#include <utility>
#include <vector>

namespace {

fe::dofs::MeshTopologyInfo singleTetraTopology()
{
    fe::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

/// Solves the dense system a x = b by Gaussian elimination with partial pivoting.
std::vector<fe::Real> solveDense(std::vector<fe::Real> a, std::vector<fe::Real> b)
{
    const std::size_t n = b.size();
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t r = c + 1; r < n; ++r) {
            if (std::abs(a[r * n + c]) > std::abs(a[pivot * n + c])) {
                pivot = r;
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            std::swap(a[c * n + j], a[pivot * n + j]);
        }
        std::swap(b[c], b[pivot]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const fe::Real factor = a[r * n + c] / a[c * n + c];
            for (std::size_t j = c; j < n; ++j) {
                a[r * n + j] -= factor * a[c * n + j];
            }
            b[r] -= factor * b[c];
        }
    }
    std::vector<fe::Real> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        fe::Real sum = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }
    return x;
}

} // namespace

TEST(Authoring, CanonicalSkeletonRegistersSetsUpAndAssembles)
{
    // Face {0,1,2} is the wall, face {0,1,3} the inlet; vertex 3 is free.
    // Returns the free row of the residual and of the tangent at T = 3.
    const auto freeRow = [](const HeatOptions& options) {
        auto mesh = std::make_shared<fe::forms::test::SingleTetraTwoBoundaryFaceMeshAccess>(options.wall,
                                                                                           options.inlet);
        auto V = std::make_shared<fe::spaces::H1Space>(fe::ElementType::Tetra4, 1);

        fe::systems::FESystem system(mesh);
        registerHeat(system, V, options);

        EXPECT_EQ(fe::systems::functionalForms(system).size(), 1u);
        EXPECT_EQ(system.derivedResults().size(), 1u);

        fe::systems::SetupInputs inputs;
        inputs.topology_override = singleTetraTopology();
        system.setup({}, inputs);

        const auto& constraints = system.constraints();
        for (fe::GlobalIndex dof = 0; dof < 3; ++dof) {
            EXPECT_TRUE(constraints.isConstrained(dof));
            EXPECT_EQ(constraints.getInhomogeneity(dof), options.wall_temperature);
        }
        EXPECT_FALSE(constraints.isConstrained(3));

        const auto n = system.dofHandler().getNumDofs();
        std::vector<fe::Real> U(static_cast<std::size_t>(n), options.wall_temperature);
        fe::systems::SystemStateView state;
        state.u = U;
        fe::assembly::DenseSystemView out(n);
        out.zero();
        fe::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = true;
        request.want_vector = true;
        (void)system.assemble(request, state, &out, &out);
        return std::pair{out.getVectorEntry(3), out.getMatrixEntry(3, 3)};
    };

    HeatOptions options{.conductivity = 2.0,
                        .source = 1.0,
                        .wall = 1,
                        .wall_temperature = 3.0,
                        .inlet = 2,
                        .inflow = 0.5};
    const auto [with_inflow, k33] = freeRow(options);
    options.inflow = 0.0;
    const auto [without_inflow, k33_again] = freeRow(options);

    // k grad(phi_3).grad(phi_3) |K| = 2 * 1 * 1/6, and the inlet load
    // -q |F_inlet| / 3 = -0.5 * 0.5 / 3 on the free vertex.
    EXPECT_NEAR(k33, 1.0 / 3.0, 1e-14);
    EXPECT_EQ(k33_again, k33);
    EXPECT_NEAR(with_inflow - without_inflow, -0.25 / 3.0, 1e-14);
}

TEST(Authoring, TheNaturalConditionPrescribesTheConormalFluxNotTheOutwardHeatFlux)
{
    // The skeleton on the slab [0, 1]^2: T = 3 on the wall x = 0 (label 1),
    // insulated at y = 0 and y = 1, and an outward heat flux q_out = 0.5
    // through x = 1 (label 2).  The natural condition prescribes the conormal
    // flux g = k grad(T).n, the heat entering, so the outward heat flux is
    // given as inflow = g = -q_out, and T = 3 - q_out x / k.
    constexpr fe::Real q_out = 0.5;
    HeatOptions options{.conductivity = 2.0,
                        .source = 0.0,
                        .wall = svmp::FE::test_trace_multiplier::kLeft,
                        .wall_temperature = 3.0,
                        .inlet = svmp::FE::test_trace_multiplier::kRight,
                        .inflow = -q_out};
    auto mesh = svmp::FE::test_trace_multiplier::buildQuadGrid(4, 4, 1.0, 1.0, 1);
    auto V = std::make_shared<fe::spaces::H1Space>(fe::ElementType::Quad4, 1);
    fe::systems::FESystem system(mesh);
    registerHeat(system, V, options);
    system.setup();

    // One Newton step from the wall values solves the linear problem.
    const auto n_dofs = system.dofHandler().getNumDofs();
    std::vector<fe::Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    system.constraints().distribute(U);
    fe::systems::SystemStateView state;
    state.u = U;
    fe::assembly::DenseSystemView out(n_dofs);
    out.zero();
    fe::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    ASSERT_TRUE(system.assemble(request, state, &out, &out).success);
    std::vector<fe::Real> minus_residual(out.vectorData().begin(), out.vectorData().end());
    for (auto& entry : minus_residual) {
        entry = -entry;
    }
    const auto step = solveDense(std::vector<fe::Real>(out.matrixData().begin(), out.matrixData().end()),
                                 minus_residual);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] += step[i];
    }
    system.constraints().distribute(U);
    state.u = U;

    const auto T = StateField(system.findFieldByName("T"), *V, "T");
    const auto k = FormExpr::constant(options.conductivity);
    const auto n = FormExpr::normal();
    const auto exact = FormExpr::constant(options.wall_temperature) -
                       FormExpr::constant(q_out / options.conductivity) * component(x(), 0);
    // The physical heat flux -k grad(T) (the skeleton's derived result), and
    // the heat leaving through the inlet and through the wall.
    const auto heat_flux = -k * grad(T);
    const std::vector<FormExpr> forms{((T - exact) * (T - exact)).dx(),
                                      inner(heat_flux, n).ds(options.inlet),
                                      inner(heat_flux, n).ds(options.wall),
                                      inner(heat_flux - as_vector({FormExpr::constant(q_out), FormExpr::constant(0.0)}),
                                            heat_flux - as_vector({FormExpr::constant(q_out), FormExpr::constant(0.0)}))
                                          .dx()};
    const auto values =
        fe::systems::evaluateFunctionalForms(system, forms, state, fe::systems::CollectiveCommunicator{});
    ASSERT_EQ(values.size(), forms.size());
    EXPECT_LT(values[0].value, 1e-24);              // T = 3 - q_out x / k
    EXPECT_NEAR(values[1].value, q_out, 1e-12);     // q_out leaves through the inlet
    EXPECT_NEAR(values[2].value, -q_out, 1e-12);    // and enters through the wall (its reaction)
    EXPECT_LT(values[3].value, 1e-24);              // the heat flux is (q_out, 0)
}
