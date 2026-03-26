/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NavierStokesOutletFactory.cpp
 * @brief Tests for toCoupledOutflowBC factory (both legacy and migrated overloads)
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Systems/FESystem.h"
#include "FE/Systems/AuxiliaryInputRegistry.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/CoupledBoundaryManager.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/CoupledBCs.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Dofs/DofMap.h"
#include "FE/Dofs/EntityDofMap.h"

#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

using svmp::FE::Real;
using svmp::FE::FieldId;
using svmp::FE::ElementType;
using namespace svmp::FE::systems;
using namespace svmp::FE::forms;
using Opts = svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;
namespace Factories = svmp::Physics::formulations::navier_stokes::Factories;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
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

/// Set a specific component at a specific vertex using field DOF metadata.
void setFieldComponent(std::vector<Real>& sol,
                       const FESystem& sys, FieldId field,
                       svmp::FE::GlobalIndex vertex, int component, Real value)
{
    const auto& fdh = sys.fieldDofHandler(field);
    const auto offset = sys.fieldDofOffset(field);
    const auto* emap = fdh.getEntityDofMap();
    ASSERT_NE(emap, nullptr);
    auto dofs = emap->getVertexDofs(vertex);
    ASSERT_GT(static_cast<int>(dofs.size()), component);
    const auto idx = static_cast<std::size_t>(dofs[static_cast<std::size_t>(component)] + offset);
    ASSERT_LT(idx, sol.size());
    sol[idx] = value;
}

Opts::CoupledRCROutflowBC makeRCROpts(int marker, Real C = 0.001)
{
    Opts::CoupledRCROutflowBC o;
    o.boundary_marker = marker;
    o.Rp = 10.0;
    o.C = C;
    o.Rd = 100.0;
    o.Pd = 50.0;
    o.X0 = 50.0;
    o.backflow_beta = {0.0};
    return o;
}

} // namespace

// ===========================================================================
//  Legacy overload (without FESystem&): the currently working path.
//  Returns CoupledNaturalBC, evaluates through CoupledBoundaryManager.
// ===========================================================================

TEST(NavierStokesOutletFactory, LegacyOverload_RCR_ReturnsCoupledNaturalBC)
{
    const int marker = 10;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker), u_field, *u_space, "u", u, FormExpr::constant(1.0));
    ASSERT_NE(bc, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);
}

TEST(NavierStokesOutletFactory, LegacyOverload_Resistive_ReturnsCoupledNaturalBC)
{
    const int marker = 20;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker, 0.0), u_field, *u_space, "u", u, FormExpr::constant(1.0));
    ASSERT_NE(bc, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);
}

TEST(NavierStokesOutletFactory, LegacyOverload_RuntimeEvaluation)
{
    // End-to-end: legacy overload -> install BC -> setup -> evaluate
    // with a NONZERO velocity that produces a nontrivial boundary integral.
    //
    // Geometry: SingleTetraOneBoundaryFaceMeshAccess, face 0 = triangle
    // (0,0,0)-(1,0,0)-(0,1,0), area = 0.5, outward normal = (0,0,-1).
    //
    // Velocity: u = (0, 0, -1) at all nodes.
    // inner(u, n) = (-1)*(-1) = 1.  Q = integral(1) * 0.5 = 0.5.
    //
    // RCR params: Rp=10, C=0.001, Rd=100, Pd=50, X0=50, dt=0.1.
    // dX/dt = (Q - (X-Pd)/Rd) / C = (0.5 - 0) / 0.001 = 500.
    // BackwardEuler: X - 50 = 0.1*(0.5 - (X-50)/100)/0.001
    //              = 100*(0.5 - (X-50)/100) = 50 - (X-50)
    // => 2X = 150 => X = 75.

    const int marker = 30;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker), u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx();
    auto continuity = (q * div(u)).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field},
        momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);

    // Set u = (0, 0, -1) at all 4 vertices via field DOF metadata.
    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_dofs, 16u);
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v = 0; v < 4; ++v) {
        setFieldComponent(sol, sys, u_field, v, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    const auto* cbm = sys.coupledBoundaryManager();
    ASSERT_NE(cbm, nullptr);
    const_cast<CoupledBoundaryManager*>(cbm)->prepareForAssembly(state);

    // Verify Q = 0.5 from the vector-field boundary integral.
    // inner(u, n) = (0,0,-1) . (0,0,-1) = 1.  Integral = 1 * area(0.5) = 0.5.
    const auto& integrals = cbm->integrals();
    ASSERT_TRUE(integrals.has("ns_Q_30"));
    EXPECT_NEAR(integrals.get("ns_Q_30"), 0.5, 1e-10);

    // Verify auxiliary state X after BackwardEuler step.
    // dX/dt = (Q - (X-Pd)/Rd) / C = (0.5 - 0) / 0.001 = 500.
    // BackwardEuler: X - 50 = 0.1*(0.5 - (X-50)/100)/0.001
    //   => X - 50 = 50 - (X-50) => 2X = 150 => X = 75.
    const auto& aux = cbm->auxiliaryState();
    ASSERT_GT(aux.size(), 0u);
    EXPECT_NEAR(aux.values()[0], 75.0, 1e-8);
}

// ===========================================================================
//  New overload (with FESystem&): the preferred path.
//  Returns NaturalBC, uses registerBoundaryIntegralInput + AuxiliaryModelBuilder.
// ===========================================================================

TEST(NavierStokesOutletFactory, NewOverload_RCR_EndToEnd)
{
    // End-to-end: new factory overload -> setup -> finalize ->
    // evaluate Q -> advance RCR model -> verify P_out.
    //
    // The factory registers the boundary-integral input, deploys the RCR model,
    // and returns a NaturalBC whose flux references AuxiliaryOutput.
    // We install a plain residual (without the BC flux) so that setup/finalize
    // succeeds, then verify the auxiliary pipeline independently.
    // The flux expression's AuxiliaryOutput resolution is tested implicitly
    // through the real NS module where installFormulation handles it.
    const int marker = 40;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    // Call the real new factory overload.
    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::NaturalBC*>(bc.get()), nullptr);

    // Install a plain residual (without the BC flux) so setup succeeds.
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q_test = FormExpr::testFunction(*p_space, "q");
    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field},
        inner(grad(u), grad(v)).dx() + (q_test * div(u)).dx(), install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    // Set u = (0, 0, -1) at all vertices via field DOF metadata.
    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    // Evaluate boundary-integral input Q via the new path.
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    ASSERT_TRUE(reg->hasInput("ns_Q_40"));
    EXPECT_NEAR(reg->get("ns_Q_40"), 0.5, 1e-10);

    // Advance the deployed RCR model.
    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    // Verify P_out from the AuxiliaryModelBuilder RCR model.
    // With Q=0.5, X0=50, BackwardEuler dt=0.1 => X=75, P_out = X + Rp*Q = 75 + 10*0.5 = 80.
    const std::string instance_name = "ns_rcr_" + std::to_string(marker);
    const auto out_slot = sys.auxiliaryOutputSlotOf(instance_name, "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 80.0, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_EndToEnd)
{
    // End-to-end for the C==0 resistive factory path.
    // Factory registers Q input and P_out derived input, returns NaturalBC.
    // u = (0,0,-1), Q = 0.5, P_out = Pd + (Rd+Rp)*Q = 50 + 110*0.5 = 105.
    const int marker = 50;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker, /*C=*/0.0), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);

    // Verify both inputs were registered.
    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_TRUE(reg->hasInput("ns_Q_50"));
    EXPECT_TRUE(reg->hasInput("ns_P_out_50"));

    // Install plain residual (without BC flux) and run setup.
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q_test = FormExpr::testFunction(*p_space, "q");
    (void)installFormulation(sys, "ns", {u_field, p_field},
        inner(grad(u), grad(v)).dx() + (q_test * div(u)).dx());

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);

    // Set u = (0, 0, -1) at all vertices.
    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(reg->get("ns_Q_50"), 0.5, 1e-10);
    // P_out = Pd + (Rd + Rp) * Q = 50 + 110 * 0.5 = 105.
    EXPECT_NEAR(reg->get("ns_P_out_50"), 105.0, 1e-10);
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_FluxInstallation)
{
    // Verify the C==0 factory path's returned NaturalBC flux (containing
    // AuxiliaryInput("ns_P_out_<marker>")) can be installed and resolved.
    const int marker = 55;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker, /*C=*/0.0), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    // Install BC flux into the residual.
    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);

    // installFormulation must resolve AuxiliaryInput("ns_P_out_55").
    FormInstallOptions install{};
    EXPECT_NO_THROW(
        (void)installFormulation(sys, "ns", {u_field}, residual, install));
}

TEST(NavierStokesOutletFactory, NewOverload_RCR_FluxInstallation)
{
    // Verify that the returned NaturalBC's flux expression (containing
    // AuxiliaryOutput) can be installed into a residual and resolved by
    // FormsInstaller when the auxiliary model is deployed.
    //
    // Uses a single-field (scalar) system to avoid mixed-form complexity.
    // The factory is called with a 3-component velocity, but we install
    // only the boundary flux term (not the full NS residual).
    const int marker = 60;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    // Install the BC flux into the residual via BoundaryConditionManager.
    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);

    // installFormulation should resolve AuxiliaryOutput("ns_rcr_60/P_out").
    FormInstallOptions install{};
    EXPECT_NO_THROW(
        (void)installFormulation(sys, "ns", {u_field}, residual, install));
}

// ===========================================================================
//  Validation
// ===========================================================================

TEST(NavierStokesOutletFactory, RdZero_Throws)
{
    const int marker = 50;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto u = FormExpr::stateField(u_field, *u_space, "u");

    auto opts = makeRCROpts(marker);
    opts.Rd = 0.0;

    EXPECT_THROW(
        Factories::toCoupledOutflowBC(opts, u_field, *u_space, "u", u, FormExpr::constant(1.0)),
        std::invalid_argument);
}
