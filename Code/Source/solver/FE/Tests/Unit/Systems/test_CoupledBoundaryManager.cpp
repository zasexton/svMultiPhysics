/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_CoupledBoundaryManager.cpp
 * @brief Unit tests for coupled boundary conditions infrastructure
 */

#include <gtest/gtest.h>

#include "Systems/CoupledBoundaryManager.h"
#include "Systems/FESystem.h"

#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace test {

TEST(CoupledBoundaryManagerTest, AuxiliaryStateResetsEachPrepareUntilCommit)
{
    auto mesh = std::make_shared<const forms::test::SingleTetraOneBoundaryFaceMeshAccess>(/*boundary_marker=*/2);
    FESystem system(mesh);

    auto space = std::make_shared<const spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    const FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");

    // FESystem DOF distribution currently requires either a full Mesh instance
    // (SVMP_FE_WITH_MESH) or a topology_override.
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};

    SetupInputs inputs;
    inputs.topology_override = topo;
    system.setup(/*opts=*/{}, inputs);

    auto& coupled = system.coupledBoundaryManager(u_id);

    forms::BoundaryFunctional Q;
    Q.integrand = forms::FormExpr::constant(1.0);
    Q.boundary_marker = 2;
    Q.name = "Q";
    Q.reduction = forms::BoundaryFunctional::Reduction::Sum;
    coupled.addBoundaryFunctional(Q);

    AuxiliaryStateRegistration reg;
    reg.spec.size = 1;
    reg.spec.name = "X";
    reg.spec.associated_markers = {2};
    reg.initial_values = {0.0};
    reg.required_integrals = {Q};
    reg.rhs = forms::FormExpr::boundaryIntegralValue("Q");
    reg.integrator = systems::ODEMethod::ForwardEuler;
    coupled.addAuxiliaryState(std::move(reg));

    const auto n_dofs = static_cast<std::size_t>(system.dofHandler().getDofMap().getNumDofs());
    std::vector<Real> u(n_dofs, 0.0);

    SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = std::span<const Real>(u.data(), u.size());

    coupled.prepareForAssembly(state);
    EXPECT_NEAR(coupled.integrals().get("Q"), 0.5, 1e-12);
    EXPECT_NEAR(coupled.auxiliaryState()["X"], 0.5, 1e-12);

    // Calling prepare again should not "drift" the work state (important for Newton iterations).
    coupled.prepareForAssembly(state);
    EXPECT_NEAR(coupled.auxiliaryState()["X"], 0.5, 1e-12);

    // After commit, the next prepare starts from the committed state.
    coupled.commitTimeStep();
    coupled.prepareForAssembly(state);
    EXPECT_NEAR(coupled.auxiliaryState()["X"], 1.0, 1e-12);
}

} // namespace test
} // namespace systems
} // namespace FE
} // namespace svmp
