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
    const auto [T_id, T, v] = fields.unknownSymbols({.name = "T", .space = V}, "temperature", "T", "v");

    const auto k = FormExpr::constant(o.conductivity);
    const auto f = FormExpr::constant(o.source);
    auto R = (k * inner(grad(T), grad(v)) - f * v).dx();

    fe::systems::BoundaryConditionManager bcs;
    bcs.add(std::make_unique<bc::EssentialBC>(o.wall, FormExpr::constant(o.wall_temperature), "T"));
    bcs.add(std::make_unique<bc::NaturalBC>(o.inlet, FormExpr::constant(o.inflow)));

    fe::constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, T, v, T_id);
    plan.commit();

    fe::systems::FormulationBuilder F(system, "equations", {});
    F.functional("conduction_energy", fe::systems::FunctionalKind::Stored,
                 (0.5 * k * inner(grad(T), grad(T))).dx());
    F.symbolicTangent();
    (void)F.install({T_id}, R);

    system.addDerivedResult(fe::post::DerivedResultBuilder("heat_flux")
                                .scope(fe::post::DerivedResultScope::Cell)
                                .policy(fe::post::DerivedResultPolicy::CellAverage)
                                .shape(fe::systems::FEQuantityShape::vector(V->topological_dimension()))
                                .expression(-k * grad(T))
                                .referencedField(T_id)
                                .build());
}
// ---- end of the canonical skeleton ----

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

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
