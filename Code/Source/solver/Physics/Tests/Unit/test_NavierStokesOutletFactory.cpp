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

#include <numeric>

#include "Assembly/GlobalSystemView.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Systems/FESystem.h"
#include "FE/Systems/AuxiliaryInputRegistry.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/CoupledBoundaryManager.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"
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

Opts::CoupledRCRCROutflowBC makeRCRCROpts(int marker)
{
    Opts::CoupledRCRCROutflowBC o;
    o.boundary_marker = marker;
    o.Rp = 10.0;
    o.C1 = 0.002;
    o.Rm = 40.0;
    o.C2 = 0.004;
    o.Rd = 100.0;
    o.Pd = 45.0;
    o.P10 = 60.0;
    o.P20 = 45.0;
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
    // End-to-end: new factory overload -> setup -> monolithic auxiliary
    // assembly -> verify the bordered data and outlet pressure.
    //
    // The factory registers the boundary-integral input, deploys the RCR model
    // monolithically,
    // and returns a NaturalBC whose flux references AuxiliaryOutput.
    // We install a plain residual (without the BC flux) so that setup/finalize
    // succeeds, then verify the monolithic auxiliary pipeline independently.
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
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

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
    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);

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

    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);
    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs.zero();
    const auto result = sys.assemble(req, state, &lhs, &rhs);
    EXPECT_TRUE(result.success);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    ASSERT_TRUE(reg->hasInput("ns_Q_40"));
    EXPECT_NEAR(reg->get("ns_Q_40"), 0.5, 1e-10);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_TRUE(bc_data.active);
    ASSERT_EQ(bc_data.n_aux, 1);
    EXPECT_EQ(bc_data.n_field_dofs, n_dofs);
    ASSERT_EQ(bc_data.dF_dxdot.size(), 1u);
    EXPECT_NEAR(bc_data.dF_dxdot[0], 1.0, 1e-12);
    ASSERT_EQ(bc_data.D.size(), 1u);
    EXPECT_NEAR(bc_data.D[0], 20.0, 1e-8);

    // Verify P_out from the monolithic RCR model before Newton updates.
    // With Q=0.5 and X0=50, P_out = X + Rp*Q = 50 + 10*0.5 = 55.
    const std::string instance_name = "ns_rcr_" + std::to_string(marker);
    const auto out_slot = sys.auxiliaryOutputSlotOf(instance_name, "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 55.0, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_EndToEnd)
{
    // End-to-end for the C==0 resistive factory path.
    // Factory registers Q, deploys an algebraic auxiliary outlet model, and
    // returns NaturalBC. u = (0,0,-1), Q = 0.5, P_out = 50 + 110*0.5 = 105.
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
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    // Verify the boundary-reduction input was registered.
    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_TRUE(reg->hasInput("ns_Q_50"));

    // Install plain residual (without BC flux) and run setup.
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q_test = FormExpr::testFunction(*p_space, "q");
    (void)installFormulation(sys, "ns", {u_field, p_field},
        inner(grad(u), grad(v)).dx() + (q_test * div(u)).dx());

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

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

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(reg->get("ns_Q_50"), 0.5, 1e-10);
    const auto out_slot = sys.auxiliaryOutputSlotOf("ns_rcr_50", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 105.0, 1e-10);
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_FluxInstallation)
{
    // Verify the C==0 factory path installs and assembles through the
    // deployed auxiliary-output runtime, without creating legacy coupled BCs.
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
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    // Install BC flux into the residual.
    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);

    // installFormulation must resolve AuxiliaryOutput("ns_rcr_55/P_out").
    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field}, residual, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_vector = true;

    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs.zero();
    const auto result = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(result.success);

    const auto out_slot = sys.auxiliaryOutputSlotOf("ns_rcr_55", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 105.0, 1e-10);
}

TEST(NavierStokesOutletFactory, NewOverload_RCR_FluxInstallation)
{
    // Verify that the returned NaturalBC's flux expression installs and
    // assembles through the deployed monolithic auxiliary runtime, without
    // using the legacy coupled-boundary manager.
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
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    // Install the BC flux into the residual via BoundaryConditionManager.
    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);

    // installFormulation should resolve AuxiliaryOutput("ns_rcr_60/P_out").
    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field}, residual, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();
    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs.zero();
    const auto result = sys.assemble(req, state, &lhs, &rhs);
    EXPECT_TRUE(result.success);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_NEAR(reg->get("ns_Q_60"), 0.5, 1e-10);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_TRUE(bc_data.active);
    ASSERT_EQ(bc_data.n_aux, 1);
    Real sum_abs_B = 0.0;
    for (const auto v_entry : bc_data.B) {
        sum_abs_B += std::abs(v_entry);
    }
    Real sum_abs_Ct = 0.0;
    for (const auto v_entry : bc_data.Ct) {
        sum_abs_Ct += std::abs(v_entry);
    }
    EXPECT_GT(sum_abs_B, 0.0);
    EXPECT_GT(sum_abs_Ct, 0.0);

    const auto out_slot = sys.auxiliaryOutputSlotOf("ns_rcr_60", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 55.0, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_RCRCR_EndToEnd)
{
    const int marker = 70;
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
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::NaturalBC*>(bc.get()), nullptr);

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx();
    auto continuity = (q * div(u)).dx();

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 0u);
    EXPECT_EQ(summary.n_partitioned, 1u);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    ASSERT_TRUE(reg->hasInput("ns_Q_70"));
    EXPECT_NEAR(reg->get("ns_Q_70"), 0.5, 1e-10);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_FALSE(bc_data.active);

    const auto out_slot = sys.auxiliaryOutputSlotOf("ns_rcrcr_70", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 71.81818181818181, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_RCRCR_FluxInstallation)
{
    const int marker = 80;
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
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx();
    auto continuity = (q * div(u)).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 0u);
    EXPECT_EQ(summary.n_partitioned, 1u);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_vector = true;
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs.zero();
    const auto result = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(result.success);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_FALSE(bc_data.active);

    const auto out_slot = sys.auxiliaryOutputSlotOf("ns_rcrcr_80", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 71.81818181818181, 1e-8);
}

TEST(NavierStokesOutletFactory, MonolithicRCRCR_MixedFieldJacobianMatchesFD)
{
    const int marker = 81;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto p = FormExpr::stateField(p_field, *p_space, "p");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx() - inner(p, div(v)).dx();
    auto continuity = (q * div(u)).dx() + (p * q).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_dofs, 0.0);
    std::vector<Real> field_jacobian(n_dofs * n_dofs, 0.0);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        base_residual[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_dofs; ++j) {
            field_jacobian[i * n_dofs + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }
    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_dofs + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;

        SystemStateView ps = state;
        ps.u = sol_pert;

        sys.restoreAuxiliaryState(packed_base);
        auto* reg = sys.auxiliaryInputRegistryIfPresent();
        ASSERT_NE(reg, nullptr);
        reg->invalidateAll();

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                          /*invalidate_auxiliary_inputs=*/false);
        const auto ar_pert = sys.assemble(req_vec, ps, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_dofs; ++row) {
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = field_jacobian[row * n_dofs + col];
            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "mixed NS outlet Jacobian mismatch at row=" << row
                << " col=" << col;
        }
    }

    sys.restoreAuxiliaryState(packed_base);
    sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
}

TEST(NavierStokesOutletFactory, MixedFieldBoundaryMeshJacobianMatchesFDWithoutOutlet)
{
    const int marker = 83;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto p = FormExpr::stateField(p_field, *p_space, "p");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx() - inner(p, div(v)).dx();
    auto continuity = (q * div(u)).dx() + (p * q).dx();

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_dofs, 0.0);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        base_residual[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
    }

    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;

        SystemStateView ps = state;
        ps.u = sol_pert;

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        rhs_pert.zero();
        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        const auto ar_pert = sys.assemble(req_vec, ps, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_dofs; ++row) {
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(row),
                                                     static_cast<svmp::FE::GlobalIndex>(col));
            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "boundary-mesh NS Jacobian mismatch at row=" << row
                << " col=" << col;
        }
    }
}

TEST(NavierStokesOutletFactory, MonolithicRCRCR_BorderedReductionMatchesDenseSolve)
{
    const int marker = 82;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto p = FormExpr::stateField(p_field, *p_space, "p");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx() - inner(p, div(v)).dx();
    auto continuity = (q * div(u)).dx() + (p * q).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_field, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    const auto& bc_data = sys.borderedCoupling();
    ASSERT_TRUE(bc_data.active);
    ASSERT_EQ(bc_data.n_aux, 2);
    ASSERT_EQ(bc_data.n_field_dofs, n_field);

    std::vector<Real> K(n_field * n_field, 0.0);
    std::vector<Real> r(n_field, 0.0);
    for (std::size_t i = 0; i < n_field; ++i) {
        r[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_field; ++j) {
            K[i * n_field + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }
    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                const auto col = static_cast<std::size_t>(col_dof);
                K[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    auto solve_dense = [](std::vector<Real> A, std::vector<Real> b) -> std::vector<Real> {
        const auto n = b.size();
        EXPECT_EQ(A.size(), n * n);
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t pivot = k;
            Real max_abs = std::abs(A[k * n + k]);
            for (std::size_t i = k + 1; i < n; ++i) {
                const Real cand = std::abs(A[i * n + k]);
                if (cand > max_abs) {
                    max_abs = cand;
                    pivot = i;
                }
            }
            EXPECT_GT(max_abs, 1e-14);
            if (pivot != k) {
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(A[k * n + j], A[pivot * n + j]);
                }
                std::swap(b[k], b[pivot]);
            }
            const Real diag = A[k * n + k];
            for (std::size_t i = k + 1; i < n; ++i) {
                const Real factor = A[i * n + k] / diag;
                if (std::abs(factor) <= 1e-30) {
                    continue;
                }
                for (std::size_t j = k; j < n; ++j) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }
        std::vector<Real> x(n, 0.0);
        for (std::size_t ii = n; ii-- > 0;) {
            Real sum = b[ii];
            for (std::size_t j = ii + 1; j < n; ++j) {
                sum -= A[ii * n + j] * x[j];
            }
            x[ii] = sum / A[ii * n + ii];
        }
        return x;
    };

    const auto u0 = solve_dense(K, r);

    const auto n_aux = static_cast<std::size_t>(bc_data.n_aux);
    std::vector<std::vector<Real>> z_columns(n_aux, std::vector<Real>(n_field, 0.0));
    for (std::size_t j = 0; j < n_aux; ++j) {
        std::vector<Real> Bj(n_field, 0.0);
        for (std::size_t i = 0; i < n_field; ++i) {
            Bj[i] = bc_data.B[i + n_field * j];
        }
        z_columns[j] = solve_dense(K, Bj);
    }

    std::vector<Real> schur = bc_data.D;
    std::vector<Real> dx_aux(n_aux, 0.0);
    for (std::size_t i = 0; i < n_aux; ++i) {
        Real ctu0 = 0.0;
        for (std::size_t k = 0; k < n_field; ++k) {
            ctu0 += bc_data.Ct[i * n_field + k] * u0[k];
        }
        dx_aux[i] = bc_data.g[i] - ctu0;

        for (std::size_t j = 0; j < n_aux; ++j) {
            Real ctz = 0.0;
            for (std::size_t k = 0; k < n_field; ++k) {
                ctz += bc_data.Ct[i * n_field + k] * z_columns[j][k];
            }
            schur[i * n_aux + j] -= ctz;
        }
    }
    dx_aux = solve_dense(schur, dx_aux);

    std::vector<Real> du_reduced(n_field, 0.0);
    for (std::size_t k = 0; k < n_field; ++k) {
        Real corrected = u0[k];
        for (std::size_t j = 0; j < n_aux; ++j) {
            corrected -= z_columns[j][k] * dx_aux[j];
        }
        du_reduced[k] = corrected;
    }

    const auto n_total = n_field + n_aux;
    std::vector<Real> full_matrix(n_total * n_total, 0.0);
    std::vector<Real> full_rhs(n_total, 0.0);
    for (std::size_t i = 0; i < n_field; ++i) {
        full_rhs[i] = r[i];
        for (std::size_t j = 0; j < n_field; ++j) {
            full_matrix[i * n_total + j] = K[i * n_field + j];
        }
        for (std::size_t j = 0; j < n_aux; ++j) {
            full_matrix[i * n_total + (n_field + j)] = bc_data.B[i + n_field * j];
        }
    }
    for (std::size_t i = 0; i < n_aux; ++i) {
        full_rhs[n_field + i] = bc_data.g[i];
        for (std::size_t j = 0; j < n_field; ++j) {
            full_matrix[(n_field + i) * n_total + j] = bc_data.Ct[i * n_field + j];
        }
        for (std::size_t j = 0; j < n_aux; ++j) {
            full_matrix[(n_field + i) * n_total + (n_field + j)] = bc_data.D[i * n_aux + j];
        }
    }

    const auto dx_dense = solve_dense(full_matrix, full_rhs);
    ASSERT_EQ(dx_dense.size(), n_total);
    for (std::size_t i = 0; i < n_field; ++i) {
        EXPECT_NEAR(du_reduced[i], dx_dense[i], std::max(1e-8, std::abs(dx_dense[i]) * 1e-7))
            << "mixed NS reduced field step mismatch at dof " << i;
    }
    for (std::size_t i = 0; i < n_aux; ++i) {
        EXPECT_NEAR(dx_aux[i], dx_dense[n_field + i],
                    std::max(1e-8, std::abs(dx_dense[n_field + i]) * 1e-7))
            << "mixed NS reduced auxiliary step mismatch at dof " << i;
    }
}

TEST(NavierStokesOutletFactory, MonolithicRCRCR_GeneralizedAlphaAuxResidualRespondsToFlux)
{
    const int marker = 84;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto p = FormExpr::stateField(p_field, *p_space, "p");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx() - inner(p, div(v)).dx();
    auto continuity = (q * div(u)).dx() + (p * q).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }
    std::vector<Real> u_dot_n(n_dofs, 0.0);

    const auto ga =
        svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator integ({
        .alpha_m = ga.alpha_m,
        .alpha_f = ga.alpha_f,
        .gamma = ga.gamma,
        .history_rate_order = 0,
    });

    SystemStateView state;
    state.time = ga.alpha_f * 0.005;
    state.dt = 0.005;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = sol;
    state.u_prev = sol;
    state.u_prev2 = u_dot_n;

    std::array<std::span<const Real>, 2> u_hist{state.u_prev, state.u_prev2};
    std::array<double, 2> dt_hist{state.dt_prev, state.dt_prev};
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    const auto ti_ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &ti_ctx;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    const auto& bc_data = sys.borderedCoupling();
    ASSERT_TRUE(bc_data.active);
    ASSERT_EQ(bc_data.n_aux, 2);

    const double aux_norm = std::sqrt(
        std::inner_product(bc_data.g.begin(), bc_data.g.end(), bc_data.g.begin(), 0.0));
    EXPECT_GT(aux_norm, 1e-6);
    EXPECT_GT(std::abs(static_cast<double>(bc_data.g[0])), 1e-6);
}

TEST(NavierStokesOutletFactory, MonolithicRCRCR_GeneralizedAlphaMixedFieldJacobianMatchesFD)
{
    const int marker = 85;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto p = FormExpr::stateField(p_field, *p_space, "p");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(
        makeRCRCROpts(marker), sys, u_field, *u_space, "u", u, rho);
    ASSERT_NE(bc, nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx() - inner(p, div(v)).dx();
    auto continuity = (q * div(u)).dx() + (p * q).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);

    FormInstallOptions install{};
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity, install);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }
    std::vector<Real> u_dot_n(n_dofs, 0.0);

    const auto ga =
        svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator integ({
        .alpha_m = ga.alpha_m,
        .alpha_f = ga.alpha_f,
        .gamma = ga.gamma,
        .history_rate_order = 0,
    });

    SystemStateView state;
    state.time = ga.alpha_f * 0.005;
    state.dt = 0.005;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = sol;
    state.u_prev = sol;
    state.u_prev2 = u_dot_n;

    std::array<std::span<const Real>, 2> u_hist{state.u_prev, state.u_prev2};
    std::array<double, 2> dt_hist{state.dt_prev, state.dt_prev};
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    const auto ti_ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &ti_ctx;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_dofs, 0.0);
    std::vector<Real> field_jacobian(n_dofs * n_dofs, 0.0);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        base_residual[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_dofs; ++j) {
            field_jacobian[i * n_dofs + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }
    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_dofs + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;

        SystemStateView ps = state;
        ps.u = sol_pert;

        sys.restoreAuxiliaryState(packed_base);
        auto* reg = sys.auxiliaryInputRegistryIfPresent();
        ASSERT_NE(reg, nullptr);
        reg->invalidateAll();

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                          /*invalidate_auxiliary_inputs=*/false);
        const auto ar_pert = sys.assemble(req_vec, ps, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_dofs; ++row) {
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = field_jacobian[row * n_dofs + col];
            EXPECT_NEAR(analytic, fd, std::max(2e-5, std::abs(fd) * 2e-4))
                << "generalized-alpha mixed NS outlet Jacobian mismatch at row=" << row
                << " col=" << col;
        }
    }

    sys.restoreAuxiliaryState(packed_base);
    sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
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
