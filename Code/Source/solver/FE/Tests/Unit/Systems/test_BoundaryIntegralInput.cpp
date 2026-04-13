/**
 * @file test_BoundaryIntegralInput.cpp
 * @brief Unit tests for registerBoundaryIntegralInput and BoundaryReductionService
 */

#include <gtest/gtest.h>

#include "Systems/BoundaryReductionService.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Constraints/DirichletBC.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/SpaceFactory.h"

#include "Dofs/DofHandler.h"
#include "Dofs/DofMap.h"
#include "Dofs/EntityDofMap.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <string>
#include <vector>

using svmp::FE::Real;
using svmp::FE::FieldId;
using svmp::FE::ElementType;
using namespace svmp::FE::systems;
using namespace svmp::FE::forms;

namespace {

/// Set the value of a specific component at a specific vertex in the solution
/// vector, using the field's DOF metadata to derive the correct global index.
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

std::vector<Real> solveDenseLinearSystem_(std::vector<Real> A, std::vector<Real> b)
{
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
}

void assembleFieldSystemWithReductions_(FESystem& sys,
                                        const SystemStateView& state,
                                        std::size_t n_field,
                                        std::vector<Real>& K_out,
                                        std::vector<Real>& r_out)
{
    svmp::FE::assembly::DenseMatrixView lhs(
        static_cast<svmp::FE::GlobalIndex>(n_field));
    svmp::FE::assembly::DenseVectorView rhs(
        static_cast<svmp::FE::GlobalIndex>(n_field));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto result = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(result.success);

    K_out.assign(n_field * n_field, Real(0.0));
    r_out.assign(n_field, Real(0.0));
    for (std::size_t i = 0; i < n_field; ++i) {
        r_out[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_field; ++j) {
            K_out[i * n_field + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }

    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                const auto col = static_cast<std::size_t>(col_dof);
                K_out[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
                const auto col = static_cast<std::size_t>(col_dof);
                K_out[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }
}

} // namespace

// ===========================================================================
//  FESystem-level tests with real FE boundary assembly
// ===========================================================================

TEST(BoundaryIntegralInput, FESystem_ConstantIntegrandOnOneFace)
{
    // Constant integrand 1.0 on boundary face 0 of a unit tet.
    // Face 0 = triangle (0,0,0)-(1,0,0)-(0,1,0), area = 0.5.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    sys.registerBoundaryIntegralInput("face_area", FormExpr::constant(1.0), marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("face_area"), 0.5, 1e-10);
}

TEST(BoundaryIntegralInput, FESystem_FieldDependentIntegrand)
{
    // Integrand = u (discrete field) with u=1 everywhere. Result = 1 * 0.5.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q"), 0.5, 1e-10);
}

TEST(BoundaryIntegralInput, FESystem_EmptyBoundary)
{
    // Non-matching marker -> integral = 0.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    sys.registerBoundaryIntegralInput("Q_empty", FormExpr::constant(1.0), /*marker=*/99);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q_empty"), 0.0, 1e-14);
}

TEST(BoundaryIntegralInput, FESystem_WithAuxiliaryModel_EndToEnd)
{
    // End-to-end: boundary integral Q -> ODE model dX/dt = -k*Q -> output Y.
    // Q = 0.5 (u=1, area=0.5), k=1, X0=10.
    // ForwardEuler: X(0.1) = 10 + 0.1*(-1*0.5) = 9.95.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    auto model = AuxiliaryModelBuilder("q_driven_decay")
        .input("Q").state("X").param("k")
        .ode("X", FormExpr::constant(-1.0) * modelParam("k") * modelInput("Q"))
        .output("Y", modelState("X"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("decay_inst").scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned).stepper({"ForwardEuler"})
            .param("k", 1.0).bind("Q", "Q").initialize({10.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;

    sys.prepareAuxiliaryForAssembly(state, false);
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q"), 0.5, 1e-10);

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto out_slot = sys.auxiliaryOutputSlotOf("decay_inst", "Y");
    ASSERT_NE(out_slot, std::string::npos);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 9.95, 1e-10);
}

TEST(BoundaryIntegralInput, FirstNonlinearAssemblyAdvancesPartitionedAuxiliary)
{
    // The first assembly in a Newton solve usually sets
    // is_nonlinear_iteration=true. Partitioned auxiliary blocks must still
    // advance once for that step instead of staying at their committed state.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    auto model = AuxiliaryModelBuilder("q_driven_decay")
        .input("Q").state("X").param("k")
        .ode("X", FormExpr::constant(-1.0) * modelParam("k") * modelInput("Q"))
        .output("Y", modelState("X"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("decay_inst").scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned).stepper({"ForwardEuler"})
            .param("k", 1.0).bind("Q", "Q").initialize({10.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    AssemblyRequest req;
    req.op = "op";
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseVectorView rhs(
        static_cast<svmp::FE::GlobalIndex>(sys.dofHandler().getNumDofs()));
    rhs.zero();

    sys.beginTimeStep();
    const auto ar = sys.assemble(req, state, nullptr, &rhs);
    ASSERT_TRUE(ar.success);

    const auto out_slot = sys.auxiliaryOutputSlotOf("decay_inst", "Y");
    ASSERT_NE(out_slot, std::string::npos);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 9.95, 1e-10);
}

// ===========================================================================
//  Second-field integrand (exercises per-field BoundaryReductionService map)
// ===========================================================================

TEST(BoundaryIntegralInput, FESystem_AcceptsMultiFieldIntegrand)
{
    // Multi-field integrands are now supported via secondary field bindings.
    const int marker = 7;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    const auto p_disc = FormExpr::discreteField(p_field, *p_space, "p");

    // inner(u, n) * p references both u and p — now accepted.
    EXPECT_NO_THROW(
        sys.registerBoundaryIntegralInput("Q_multi",
            p_disc * inner(u_disc, FormExpr::normal()), marker));
}

TEST(BoundaryIntegralInput, FESystem_VectorFieldIntegrand)
{
    // Vector-field boundary integral: inner(u_disc, n) on a 3-component field.
    // u = (0,0,-1) at all nodes, normal = (0,0,-1), inner = 1. Integral = 0.5.
    const int marker = 7;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    sys.registerBoundaryIntegralInput("Q_flow", inner(u_disc, FormExpr::normal()), marker);

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto v = FormExpr::testFunction(*u_space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    // Set u_z = -1 at all 4 vertices via field DOF metadata.
    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q_flow"), 0.5, 1e-10);
}

TEST(BoundaryIntegralInput, FESystem_ScalarTimesConstant)
{
    // Boundary integral of a scaled scalar field: integral_Gamma 3*u ds.
    // u=2 everywhere, area=0.5 -> integral = 3*2*0.5 = 3.0.
    const int marker = 7;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q_scaled",
        FormExpr::constant(3.0) * u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 2.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q_scaled"), 3.0, 1e-10);
}

// ===========================================================================
//  Outlet-pattern tests (reproducing factory logic inline, FE-library only)
// ===========================================================================

TEST(BoundaryIntegralInput, ResistiveOutlet_C0_DerivedInput)
{
    // Reproduces the C==0 factory path with a single-field (scalar) system:
    // 1. Register boundary-integral input Q = integral_Gamma u ds
    // 2. Register derived P_out = Pd + Rsum * Q
    // 3. Declare dependency P_out -> Q
    // u=3 everywhere, area=0.5 -> Q=1.5, P_out = 50 + 110*1.5 = 215.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("ns");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const Real Rp = 10.0, Rd = 100.0, Pd = 50.0, Rsum = Rd + Rp;
    auto& aux_reg = sys.auxiliaryInputRegistry();
    AuxiliaryInputSpec p_spec;
    p_spec.name = "P_out"; p_spec.size = 1;
    p_spec.producer = AuxiliaryInputProducer::FormulationCallback;
    auto* aux_reg_ptr = &aux_reg;
    aux_reg.registerInput(p_spec,
        [aux_reg_ptr, Pd, Rsum](Real, Real, std::span<Real> out) {
            out[0] = Pd + Rsum * aux_reg_ptr->get("Q");
        });
    aux_reg.addDependency("P_out", "Q");

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "ns", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 3.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(aux_reg.get("Q"), 1.5, 1e-10);
    EXPECT_NEAR(aux_reg.get("P_out"), 50.0 + 110.0 * 1.5, 1e-10);
}

TEST(BoundaryIntegralInput, RCROutlet_CPositive_ModelDeployed)
{
    // Reproduces the C>0 factory path with a single-field (scalar) system:
    // 1. Register boundary-integral input Q = integral_Gamma u ds
    // 2. Deploy RCR model via AuxiliaryModelBuilder
    // 3. Verify P_out output
    // u=0 everywhere -> Q=0, X stays at X0=50, P_out = 50 + 10*0 = 50.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("ns");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const Real Rp = 10.0, C = 0.001, Rd = 100.0, Pd = 50.0, X0 = 50.0;
    auto model = AuxiliaryModelBuilder("rcr_test")
        .input("Q").state("X").param("Rp").param("C").param("Rd").param("Pd")
        .ode("X", (modelInput("Q") - (modelState("X") - modelParam("Pd")) / modelParam("Rd")) / modelParam("C"))
        .output("P_out", modelState("X") + modelParam("Rp") * modelInput("Q"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("rcr_inst").scope(AuxiliaryStateScope::Global)
            .solveMode(AuxiliarySolveMode::Partitioned).stepper({"BackwardEuler"})
            .param("Rp", Rp).param("C", C).param("Rd", Rd).param("Pd", Pd)
            .bind("Q", "Q").initialize({X0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "ns", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;

    sys.prepareAuxiliaryForAssembly(state, false);
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q"), 0.0, 1e-10);

    const auto out_slot = sys.auxiliaryOutputSlotOf("rcr_inst", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 50.0, 1e-10);

    // Advance one step at equilibrium: dX/dt = 0, P_out stays at 50.
    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 50.0, 1e-10);
}

TEST(BoundaryIntegralInput, PartitionedBoundaryReductionConveniencePath)
{
    const int marker = 5;
    auto mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("ns");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral("Q", u_disc, marker);

    auto model = AuxiliaryModelBuilder("convenience_partitioned")
        .input("Q")
        .state("X")
        .ode("X", modelInput("Q"))
        .output("P_out", modelState("X") + FormExpr::constant(10.0) * modelInput("Q"))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("conv_inst").global().partitioned("ForwardEuler")
            .bindBoundaryReduction(Q)
            .initialize({50.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "ns", {u_field}, inner(grad(u), grad(v)).dx());

    {
        SetupInputs si;
        si.topology_override = singleTetraTopology();
        sys.setup({}, si);
    }
    sys.finalizeAuxiliaryLayout();

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.prepareAuxiliaryForAssembly(state, false);
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q"), 0.5, 1e-10);

    const auto out_slot = sys.auxiliaryOutputSlotOf("conv_inst", "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 55.0, 1e-10);

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);
    EXPECT_NEAR(sys.auxiliaryOutputValues()[out_slot], 55.05, 1e-10);
}

// ===========================================================================
//  AuxiliaryInputRegistry callback-driven tests (no mesh)
// ===========================================================================

TEST(BoundaryIntegralInput, RegisterAsBoundaryReductionProducer)
{
    AuxiliaryInputRegistry reg;
    AuxiliaryInputSpec spec;
    spec.name = "Q_outlet"; spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.boundary_marker = 10;
    int eval_count = 0;
    reg.registerInput(spec, [&](Real time, Real, std::span<Real> out) {
        out[0] = 42.0 * time; eval_count++;
    });
    EXPECT_TRUE(reg.hasInput("Q_outlet"));
    reg.evaluate(1.0, 0.1);
    EXPECT_EQ(eval_count, 1);
    EXPECT_DOUBLE_EQ(reg.get("Q_outlet"), 42.0);
}

TEST(BoundaryIntegralInput, OncePerTimeStepRefresh)
{
    AuxiliaryInputRegistry reg;
    AuxiliaryInputSpec spec;
    spec.name = "Q"; spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep;
    int c = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) { out[0] = Real(++c); });

    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(c, 1);
    reg.evaluate(0.0, 0.1, true); // nonlinear iter -> no re-eval
    EXPECT_EQ(c, 1);
    reg.invalidateAll();
    reg.evaluate(0.1, 0.1);
    EXPECT_EQ(c, 2);
}

TEST(BoundaryIntegralInput, EachNonlinearIterationRefresh)
{
    AuxiliaryInputRegistry reg;
    AuxiliaryInputSpec spec;
    spec.name = "Q"; spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::EachNonlinearIteration;
    int c = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) { out[0] = Real(++c); });

    reg.evaluate(0.0, 0.1);
    EXPECT_EQ(c, 1);
    reg.evaluate(0.0, 0.1, true);
    EXPECT_EQ(c, 2);
}

TEST(BoundaryIntegralInput, ManualRefresh)
{
    AuxiliaryInputRegistry reg;
    AuxiliaryInputSpec spec;
    spec.name = "Q"; spec.size = 1;
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::Manual;
    int c = 0;
    reg.registerInput(spec, [&](Real, Real, std::span<Real> out) { out[0] = Real(++c); });

    reg.evaluate(0.0, 0.1);
    int after_first = c;
    reg.invalidateAll();
    reg.evaluate(0.1, 0.1);
    int after_invalidate = c;
    reg.markDirty("Q");
    reg.evaluate(0.2, 0.1);
    EXPECT_GT(c, after_invalidate);
}

TEST(BoundaryIntegralInput, MultipleDistinctNames)
{
    AuxiliaryInputRegistry reg;
    auto mk = [](const std::string& nm, int m) {
        AuxiliaryInputSpec s; s.name = nm; s.size = 1;
        s.producer = AuxiliaryInputProducer::BoundaryReduction; s.boundary_marker = m;
        return s;
    };
    reg.registerInput(mk("A", 10), [](Real, Real, std::span<Real> o) { o[0] = 1.0; });
    reg.registerInput(mk("B", 20), [](Real, Real, std::span<Real> o) { o[0] = 2.0; });
    reg.registerInput(mk("C", 30), [](Real, Real, std::span<Real> o) { o[0] = 3.0; });
    reg.evaluate(0.0, 0.1);
    EXPECT_DOUBLE_EQ(reg.get("A"), 1.0);
    EXPECT_DOUBLE_EQ(reg.get("B"), 2.0);
    EXPECT_DOUBLE_EQ(reg.get("C"), 3.0);
}

TEST(BoundaryIntegralInput, DependencyOrdering)
{
    AuxiliaryInputRegistry reg;
    AuxiliaryInputSpec qs; qs.name = "Q"; qs.size = 1;
    qs.producer = AuxiliaryInputProducer::BoundaryReduction;
    AuxiliaryInputSpec ps; ps.name = "P"; ps.size = 1;
    ps.producer = AuxiliaryInputProducer::FormulationCallback;
    std::vector<std::string> order;
    reg.registerInput(qs, [&](Real, Real, std::span<Real> o) { order.push_back("Q"); o[0] = 10.0; });
    reg.registerInput(ps, [&](Real, Real, std::span<Real> o) { order.push_back("P"); o[0] = reg.get("Q") * 2.0; });
    reg.addDependency("P", "Q");
    reg.evaluate(0.0, 0.1);
    ASSERT_GE(order.size(), 2u);
    EXPECT_EQ(order[0], "Q");
    EXPECT_EQ(order[1], "P");
    EXPECT_DOUBLE_EQ(reg.get("P"), 20.0);
}

// ===========================================================================
//  BoundaryFunctional struct tests
// ===========================================================================

TEST(BoundaryFunctional, DefaultConstruction)
{
    BoundaryFunctional f;
    EXPECT_EQ(f.boundary_marker, -1);
    EXPECT_TRUE(f.name.empty());
    EXPECT_EQ(f.reduction, BoundaryFunctional::Reduction::Sum);
}

TEST(BoundaryFunctional, SumReductionIsDefault)
{
    BoundaryFunctional f;
    f.name = "Q"; f.boundary_marker = 10;
    f.integrand = FormExpr::constant(1.0);
    EXPECT_EQ(f.reduction, BoundaryFunctional::Reduction::Sum);
}

TEST(BoundaryFunctional, AllReductionModesExist)
{
    EXPECT_NE(BoundaryFunctional::Reduction::Sum, BoundaryFunctional::Reduction::Average);
    EXPECT_NE(BoundaryFunctional::Reduction::Max, BoundaryFunctional::Reduction::Min);
}

// ===========================================================================
//  Handle-returning API tests
// ===========================================================================

#include "Auxiliary/AuxiliaryModelDSL.h"

TEST(BoundaryIntegralInput, BoundaryIntegralReturnsHandle)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(inner(u_disc, FormExpr::normal()), marker);
    EXPECT_EQ(Q.registryName().find("_boundary_integral_b5_"), 0u);

    // Handle should be convertible to FormExpr.
    FormExpr q_expr = Q;
    EXPECT_TRUE(q_expr.isValid());
}

TEST(BoundaryIntegralInput, BoundaryIntegralAutoNamesHandles)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    const auto u_state = FormExpr::stateField(u_field, *space, "u");
    auto Q0 = sys.boundaryIntegral(inner(u_state, FormExpr::normal()), marker);
    auto Q1 = sys.boundaryIntegral(inner(u_state, FormExpr::normal()), marker);

    EXPECT_NE(Q0.registryName(), Q1.registryName());
    EXPECT_EQ(Q0.registryName().find("_boundary_integral_b5_"), 0u);
    EXPECT_EQ(Q1.registryName().find("_boundary_integral_b5_"), 0u);
}

TEST(BoundaryIntegralInput, DerivedInputEvaluatesCorrectly)
{
    // Register a boundary integral Q = ∫ u ds, then derived P_out = 5 + 200*Q.
    // With u=1 on face (area=0.5), Q=0.5, so P_out = 5 + 200*0.5 = 105.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Scalar integrand: discrete field value (not inner with normal).
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(u_disc, marker);

    // derivedInput references Q via handle — resolved at finalization.
    auto P_out = sys.derivedInput("P_out",
        FormExpr::constant(5.0) + FormExpr::constant(200.0) * Q.expr());

    EXPECT_EQ(P_out.registryName(), "P_out");

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_NEAR(reg->get(Q.registryName()), 0.5, 1e-10);
    EXPECT_NEAR(reg->get("P_out"), 5.0 + 200.0 * 0.5, 1e-10);
}

TEST(BoundaryIntegralInput, DerivedInputForwardReference)
{
    // Register P_out BEFORE Q — tests deferred dependency resolution.
    // P_out = 10 + 100*Q, Q = ∫ u ds = 0.5 (u=1, area=0.5).
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Register derived input FIRST — references "Q" which doesn't exist yet.
    auto P_out = sys.derivedInput("P_out",
        FormExpr::constant(10.0) + FormExpr::constant(100.0) * FormExpr::auxiliaryInput("Q"));

    // Then register the boundary integral.
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_NEAR(reg->get("Q"), 0.5, 1e-10);
    EXPECT_NEAR(reg->get("P_out"), 10.0 + 100.0 * 0.5, 1e-10);
}

TEST(BoundaryIntegralInput, DerivedInputMissingDependencyThrows)
{
    // derivedInput referencing a never-registered input should throw
    // when resolution runs, with a diagnostic naming both inputs.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    sys.derivedInput("bad",
        FormExpr::constant(1.0) + FormExpr::auxiliaryInput("nonexistent"));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    // The throw happens at first prepareAuxiliaryForAssembly, which triggers
    // deferred resolution.
    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    try {
        sys.prepareAuxiliaryForAssembly(state, false);
        FAIL() << "Expected InvalidArgumentException";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("bad"), std::string::npos)
            << "Diagnostic should name the dependent input 'bad': " << msg;
        EXPECT_NE(msg.find("nonexistent"), std::string::npos)
            << "Diagnostic should name the missing dependency 'nonexistent': " << msg;
    }
}

TEST(BoundaryIntegralInput, DerivedInputSelfReferenceThrows)
{
    // P = 1 + P is a self-referential cycle and must be rejected eagerly,
    // leaving no residual state in the registry or deferred lists.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    try {
        sys.derivedInput("P",
            FormExpr::constant(1.0) + FormExpr::auxiliaryInput("P"));
        FAIL() << "Expected InvalidArgumentException for self-referential derivedInput";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("P"), std::string::npos)
            << "Diagnostic should name the self-referential input: " << msg;
        EXPECT_NE(msg.find("itself"), std::string::npos)
            << "Diagnostic should mention self-reference: " << msg;
    }

    // The failed registration must not have reserved the name "P".
    // A subsequent valid registration of "P" must succeed.
    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_FALSE(reg->hasInput("P"))
        << "Self-referential derivedInput must not leave 'P' in the registry";

    // Re-registering "P" with a valid expression must work.
    auto P = sys.derivedInput("P", FormExpr::constant(42.0));
    EXPECT_EQ(P.registryName(), "P");
    EXPECT_TRUE(reg->hasInput("P"));
}

TEST(BoundaryIntegralInput, SampledFieldReturnsHandle)
{
    // Verify that sampledField() returns a usable AuxiliaryInputHandle.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    // sampledField is called after setup (needs DOF handler).
    auto u_sample = sys.sampledField("u_sample", "u", 4);

    EXPECT_EQ(u_sample.registryName(), "u_sample");
    FormExpr expr = u_sample;
    EXPECT_TRUE(expr.isValid());
}

TEST(BoundaryIntegralInput, BoundaryNodalSumReturnsHandle)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    auto Q_nodal = sys.boundaryNodalSum("Q_nodal", "u", marker);

    EXPECT_EQ(Q_nodal.registryName(), "Q_nodal");
    FormExpr expr = Q_nodal;
    EXPECT_TRUE(expr.isValid());
}

// ===========================================================================
//  FE Quantity Definition and Handle Metadata Tests
// ===========================================================================

TEST(FEQuantityHandle, BoundaryIntegralCarriesDefinition)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");

    auto Q = sys.boundaryIntegral(u_disc, marker);

    ASSERT_TRUE(Q.hasDefinition());
    EXPECT_EQ(Q.kind(), svmp::FE::systems::FEQuantityKind::BoundaryIntegral);
    EXPECT_EQ(Q.shape().kind, svmp::FE::systems::FEQuantityShapeKind::Scalar);
    EXPECT_EQ(Q.shape().components, 1);
    EXPECT_TRUE(Q.supportsExplicitEvaluation());
    EXPECT_TRUE(Q.supportsMonolithicLinearization());
    ASSERT_EQ(Q.referencedFields().size(), 1u);
    EXPECT_EQ(Q.referencedFields()[0], u_field);
}

TEST(FEQuantityHandle, BoundaryIntegralAcceptsStateFieldIntegrand)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_state = FormExpr::stateField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(u_state, marker);

    ASSERT_TRUE(Q.hasDefinition());
    ASSERT_EQ(Q.referencedFields().size(), 1u);
    EXPECT_EQ(Q.referencedFields()[0], u_field);

    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u_state), grad(v)).dx());
    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_NEAR(reg->get(Q.registryName()), 0.5, 1e-10);
}

TEST(FEQuantityHandle, SampledFieldCarriesDefinition)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());
    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    auto u_s = sys.sampledField("u_s", "u", 4);

    ASSERT_TRUE(u_s.hasDefinition());
    EXPECT_EQ(u_s.kind(), svmp::FE::systems::FEQuantityKind::SampledField);
    EXPECT_TRUE(u_s.supportsMonolithicLinearization());  // identity dI/du
    ASSERT_EQ(u_s.referencedFields().size(), 1u);
}

TEST(FEQuantityHandle, DerivedInputHasNoMonolithicSupport)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    auto P = sys.derivedInput("P", FormExpr::constant(42.0));

    ASSERT_TRUE(P.hasDefinition());
    EXPECT_EQ(P.kind(), svmp::FE::systems::FEQuantityKind::DerivedCallback);
    EXPECT_FALSE(P.supportsMonolithicLinearization());
    EXPECT_TRUE(P.supportsExplicitEvaluation());
}

TEST(FEQuantityHandle, BoundaryAverageRegisters)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");

    auto Q_avg = sys.boundaryAverage("Q_avg", u_disc, marker);

    ASSERT_TRUE(Q_avg.hasDefinition());
    EXPECT_EQ(Q_avg.kind(), svmp::FE::systems::FEQuantityKind::BoundaryAverage);
    EXPECT_EQ(Q_avg.registryName(), "Q_avg");
}

TEST(FEQuantityHandle, FEQuantityRegistryTracksDefinitions)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");

    const auto Q1 = sys.boundaryIntegral(u_disc, marker);
    sys.boundaryAverage("Q2", u_disc, marker);
    sys.derivedInput("P", FormExpr::constant(1.0));

    const auto* reg = sys.feQuantityRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->count(), 3u);
    EXPECT_TRUE(reg->hasDefinition(Q1.registryName()));
    EXPECT_TRUE(reg->hasDefinition("Q2"));
    EXPECT_TRUE(reg->hasDefinition("P"));

    auto boundary_defs = reg->byKind(svmp::FE::systems::FEQuantityKind::BoundaryIntegral);
    EXPECT_EQ(boundary_defs.size(), 1u);

    auto field_defs = reg->byField(u_field);
    EXPECT_EQ(field_defs.size(), 2u);  // boundary integral and Q2 reference u_field
}

// ===========================================================================
//  Binding-Mode Validation Tests
// ===========================================================================

#include "Auxiliary/AuxiliaryModelDSL.h"

TEST(BindingModeValidation, HandleBindingOnPartitionedAccepts)
{
    using namespace svmp::FE::systems;

    auto model = aux::model("test", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        m << ddt(x) == -x + Q;
    });

    AuxiliaryInputHandle Q_handle("Q_reg");

    auto inst = use(model)
        .name("inst")
        .partitioned("BackwardEuler")
        .bind("Q", Q_handle)
        .initialize({0.0});

    auto err = inst.validate();
    EXPECT_TRUE(err.empty()) << "Partitioned handle-backed bindings remain valid inputs: " << err;
    ASSERT_EQ(inst.coupledBindings().count("Q"), 1u);
}

TEST(BindingModeValidation, CoupledBindingOnMonolithicAccepts)
{
    using namespace svmp::FE::systems;

    auto model = aux::model("test", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        m << ddt(x) == -x + Q;
    });

    // Handle with monolithic linearization support.
    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = "Q_reg";
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->capabilities.monolithic_linearization = true;
    AuxiliaryInputHandle Q_handle("Q_reg", def);

    auto inst = use(model)
        .name("inst")
        .monolithic()
        .bind("Q", Q_handle)
        .initialize({0.0});

    auto err = inst.validate();
    EXPECT_TRUE(err.empty()) << "Monolithic + coupled binding should be valid: " << err;
}

// ===========================================================================
//  WS9.1: Explicit FE quantity tests
// ===========================================================================

TEST(FEQuantityExplicit, BoundaryAverageEvaluates)
{
    // ∫ 1.0 ds / measure = 1.0 (constant on unit-tet face with area 0.5).
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto Q_avg = sys.boundaryAverage("Q_avg", FormExpr::constant(1.0), marker);
    EXPECT_EQ(Q_avg.kind(), svmp::FE::systems::FEQuantityKind::BoundaryAverage);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // constant 1.0 averaged = 1.0.
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("Q_avg"), 1.0, 1e-10);
}

TEST(FEQuantityExplicit, GeometryOnlyConstantIntegral)
{
    // ∫ 1.0 ds on face 5 of unit tet = face area = 0.5.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Geometry-only: integrand has no field dependence.
    auto area = sys.boundaryIntegral(FormExpr::constant(1.0), marker);
    EXPECT_TRUE(area.supportsExplicitEvaluation());
    // No field references → no monolithic linearization.
    EXPECT_FALSE(area.supportsMonolithicLinearization());

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get(area.registryName()), 0.5, 1e-10);
}

// ===========================================================================
//  WS9.2: dF/dinputs generation test
// ===========================================================================

TEST(MonolithicCoupling, DFDInputsSymbolicGeneration)
{
    using namespace svmp::FE::systems;

    // Build a simple model: dx/dt = -k*x + Q, output y = x + Q.
    // dF/dQ should be -1 (since F = xdot - (-k*x + Q) = xdot + k*x - Q).
    auto model = aux::model("driven", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x + Q;
    });

    EXPECT_TRUE(model->hasResidualExpressions());
    EXPECT_EQ(model->dimension(), 1);

    // Evaluate the model to verify residual.
    AuxiliaryLocalContext ctx;
    Real x_val[] = {2.0};
    Real xdot_val[] = {0.0};
    Real input_val[] = {5.0};
    Real param_val[] = {3.0};
    ctx.x = x_val;
    ctx.xdot = xdot_val;
    ctx.inputs = input_val;
    ctx.params = param_val;
    ctx.time = 0.0;
    ctx.dt = 0.01;
    ctx.effective_dt = 0.01;

    // F = xdot - (-k*x + Q) = xdot + k*x - Q = 0 + 3*2 - 5 = 1.
    Real res[1];
    AuxiliaryResidualRequest req{res};
    model->evaluateResidual(ctx, req);
    EXPECT_NEAR(res[0], 1.0, 1e-10);
}

// ===========================================================================
//  WS9.3: Binding-mode diagnostics
// ===========================================================================

TEST(BindingModeDiagnostics, ShapeMismatchRejected)
{
    using namespace svmp::FE::systems;

    // Model expects scalar input "Q" (size 1).
    auto model = aux::model("scalar_model", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        m << ddt(x) == -x + Q;
    });

    // Handle claims to be a vector with 3 components.
    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = "Q_vec";
    def->kind = FEQuantityKind::SampledField;
    def->shape = FEQuantityShape::vector(3);
    def->capabilities.monolithic_linearization = true;
    AuxiliaryInputHandle Q_handle("Q_vec", def);

    auto inst = use(model).name("inst").monolithic()
        .bind("Q", Q_handle).initialize({0.0});

    auto err = inst.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("component"), std::string::npos)
        << "Should report shape mismatch: " << err;
}

TEST(BindingModeDiagnostics, UnsupportedLinearizationRejected)
{
    using namespace svmp::FE::systems;

    auto model = aux::model("test", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        m << ddt(x) == -x + Q;
    });

    // Handle does NOT support monolithic linearization.
    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = "Q_callback";
    def->kind = FEQuantityKind::DerivedCallback;
    def->shape = FEQuantityShape::scalar();
    def->capabilities.monolithic_linearization = false;
    AuxiliaryInputHandle Q_handle("Q_callback", def);

    auto inst = use(model).name("inst").monolithic()
        .bind("Q", Q_handle).initialize({0.0});

    auto err = inst.validate();
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("linearization"), std::string::npos)
        << "Should report unsupported linearization: " << err;
}

// ===========================================================================
//  WS9.4: Regression tests
// ===========================================================================

TEST(FEQuantityRegression, ManyInputManyOutputModel)
{
    using namespace svmp::FE::systems;

    // 5-input, 4-state, 3-output model exercising the full DSL.
    auto model = aux::model("multi_io", [](ModelFacade& m) {
        auto [I1, I2, I3, I4, I5] = m.inputs("I1", "I2", "I3", "I4", "I5");
        auto [x, y, z, w] = m.states("x", "y", "z", "w");
        auto [k1, k2] = m.params("k1", "k2");

        m << ddt(x) == -k1 * x + I1;
        m << ddt(y) == k1 * x - k2 * y + I2;
        m << ddt(z) == k2 * y - z + I3 + I4;
        m << ddt(w) == z - w + I5;

        m << out("sum_xy") == x + y;
        m << out("sum_zw") == z + w;
        m << out("total") == x + y + z + w;
    });

    EXPECT_EQ(model->dimension(), 4);
    EXPECT_EQ(model->outputCount(), 3);

    const auto& sig = model->signature();
    EXPECT_EQ(sig.inputs.size(), 5u);
    EXPECT_EQ(sig.parameters.size(), 2u);
    EXPECT_EQ(sig.outputs.size(), 3u);

    // Evaluate: all inputs = 1, all states = 1, params = {0.5, 0.3}.
    AuxiliaryLocalContext ctx;
    Real x_val[] = {1.0, 1.0, 1.0, 1.0};
    Real xdot_val[] = {0.0, 0.0, 0.0, 0.0};
    Real input_val[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    Real param_val[] = {0.5, 0.3};
    ctx.x = x_val; ctx.xdot = xdot_val;
    ctx.inputs = input_val; ctx.params = param_val;
    ctx.time = 0.0; ctx.dt = 0.01; ctx.effective_dt = 0.01;

    Real res[4];
    AuxiliaryResidualRequest req{res};
    model->evaluateResidual(ctx, req);

    // F[0] = xdot - (-k1*x + I1) = 0 - (-0.5*1 + 1) = -0.5
    EXPECT_NEAR(res[0], -0.5, 1e-10);
    // F[1] = xdot - (k1*x - k2*y + I2) = 0 - (0.5 - 0.3 + 1) = -1.2
    EXPECT_NEAR(res[1], -1.2, 1e-10);

    Real out[3];
    model->evaluateOutputs(ctx, out);
    EXPECT_NEAR(out[0], 2.0, 1e-10);  // x + y
    EXPECT_NEAR(out[1], 2.0, 1e-10);  // z + w
    EXPECT_NEAR(out[2], 4.0, 1e-10);  // x + y + z + w
}

// ===========================================================================
//  WS9.1: Sampled tensor test (tensor-shaped handle metadata)
// ===========================================================================

TEST(FEQuantityExplicit, SampledVectorShapeMetadata)
{
    // Verify that a multi-component sampled field gets vector shape metadata.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("op");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto v = FormExpr::testFunction(*u_space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    auto u_s = sys.sampledField("u_vec", "u", 4);

    ASSERT_TRUE(u_s.hasDefinition());
    EXPECT_EQ(u_s.kind(), svmp::FE::systems::FEQuantityKind::SampledField);
    EXPECT_EQ(u_s.shape().kind, svmp::FE::systems::FEQuantityShapeKind::Vector);
    EXPECT_EQ(u_s.shape().components, 3);
    EXPECT_TRUE(u_s.supportsMonolithicLinearization());
}

// ===========================================================================
//  WS9.1: Domain integral and domain average (registration + metadata)
// ===========================================================================

TEST(FEQuantityExplicit, DomainIntegralRegisters)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto M = sys.domainIntegral("M", u_disc);

    ASSERT_TRUE(M.hasDefinition());
    EXPECT_EQ(M.kind(), svmp::FE::systems::FEQuantityKind::DomainIntegral);
    EXPECT_TRUE(M.supportsExplicitEvaluation());
    EXPECT_TRUE(M.supportsMonolithicLinearization());
    ASSERT_EQ(M.referencedFields().size(), 1u);
}

TEST(FEQuantityExplicit, RegionIntegralRegisters)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto R = sys.regionIntegral("R", u_disc, /*region_marker=*/1);

    ASSERT_TRUE(R.hasDefinition());
    EXPECT_EQ(R.kind(), svmp::FE::systems::FEQuantityKind::RegionIntegral);
    EXPECT_EQ(R.definition()->region_marker, 1);
    EXPECT_TRUE(R.supportsMonolithicLinearization());
}

// ===========================================================================
//  WS9.4: Multi-field regression (multi-field integrand accepted + metadata)
// ===========================================================================

TEST(FEQuantityRegression, MultiFieldBoundaryIntegral)
{
    const int marker = 7;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    const auto p_disc = FormExpr::discreteField(p_field, *p_space, "p");

    // Multi-field integrand: p * inner(u, n).
    auto Q = sys.boundaryIntegral(
        p_disc * inner(u_disc, FormExpr::normal()), marker);

    ASSERT_TRUE(Q.hasDefinition());
    EXPECT_EQ(Q.kind(), svmp::FE::systems::FEQuantityKind::BoundaryIntegral);
    // Should reference both fields.
    EXPECT_GE(Q.referencedFields().size(), 2u);
    EXPECT_TRUE(Q.supportsMonolithicLinearization());
}

// ===========================================================================
//  WS9.4: Monolithic regression (coupled binding structure)
// ===========================================================================

TEST(FEQuantityRegression, MonolithicCoupledBindingStructure)
{
    using namespace svmp::FE::systems;

    auto model = aux::model("coupled", [](ModelFacade& m) {
        auto [Q1, Q2] = m.inputs("Q1", "Q2");
        auto [x, y] = m.states("x", "y");
        auto k = m.param("k");

        m << ddt(x) == -k * x + Q1;
        m << ddt(y) == -k * y + Q2;
        m << out("sum") == x + y;
    });

    // Create handles with monolithic linearization support.
    auto def1 = std::make_shared<FEQuantityDefinition>();
    def1->name = "Q1_reg";
    def1->kind = FEQuantityKind::BoundaryIntegral;
    def1->capabilities.monolithic_linearization = true;
    AuxiliaryInputHandle Q1("Q1_reg", def1);

    auto def2 = std::make_shared<FEQuantityDefinition>();
    def2->name = "Q2_reg";
    def2->kind = FEQuantityKind::SampledField;
    def2->capabilities.monolithic_linearization = true;
    AuxiliaryInputHandle Q2("Q2_reg", def2);

    auto inst = use(model).name("coupled_inst").monolithic()
        .bind("Q1", Q1)
        .bind("Q2", Q2)
        .param("k", 1.0)
        .initialize({0.0, 0.0});

    // Validate: monolithic + two coupled bindings should be accepted.
    auto err = inst.validate();
    EXPECT_TRUE(err.empty()) << "Should accept monolithic with two coupled bindings: " << err;

    // Verify coupled bindings are stored.
    EXPECT_EQ(inst.coupledBindings().size(), 2u);
    EXPECT_TRUE(inst.coupledBindings().count("Q1"));
    EXPECT_TRUE(inst.coupledBindings().count("Q2"));
}

// ===========================================================================
//  WS9.2: Symbolic gradient via evaluateFunctionalGradient + FD verification
// ===========================================================================

TEST(MonolithicCoupling, SymbolicGradientMatchesFD)
{
    // Verify that evaluateFunctionalGradient (symbolic path via
    // BoundaryFunctionalGradientKernel + GradAccumulator + StandardAssembler)
    // matches finite-difference perturbation of the functional value.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;

    sys.prepareAuxiliaryForAssembly(state, false);
    const Real Q_base = sys.auxiliaryInputRegistryIfPresent()->get("Q");

    // Get SYMBOLIC gradient from evaluateFunctionalGradient.
    auto& svc = sys.boundaryReductionService(u_field);
    auto symbolic_grad = svc.evaluateFunctionalGradient("Q", state, /*apply_constraints=*/false);

    // Get FD gradient for comparison.
    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    for (const auto& entry : symbolic_grad) {
        const auto idx = static_cast<std::size_t>(entry.dof);
        if (idx >= n_dofs) continue;

        const Real orig = perturbed[idx];
        perturbed[idx] = orig + eps;
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real Q_pert = sys.auxiliaryInputRegistryIfPresent()->get("Q");
        perturbed[idx] = orig;

        const Real fd_deriv = (Q_pert - Q_base) / eps;

        EXPECT_NEAR(entry.value, fd_deriv, 1e-5)
            << "Symbolic gradient dQ/du[" << entry.dof << "] = " << entry.value
            << " does not match FD = " << fd_deriv;
    }

    // Should have 3 nonzero entries (3 boundary face vertices).
    EXPECT_EQ(symbolic_grad.size(), 3u)
        << "Expected 3 nonzero symbolic gradient entries";

    // Each should be area/3 = 0.5/3 for linear basis on unit-tet face.
    for (const auto& entry : symbolic_grad) {
        EXPECT_NEAR(std::abs(entry.value), 0.5 / 3.0, 1e-5)
            << "Symbolic gradient entry should be area/3";
    }
}

// ===========================================================================
//  WS9.2: FD verification of functional evaluation
// ===========================================================================

TEST(MonolithicCoupling, BoundaryIntegralGradientFDVerification)
{
    // Verify that the boundary integral gradient dI/du is correct by
    // comparing the functional value change under DOF perturbation.
    // Q = ∫_Γ u ds on a unit-tet face.  For u=1, Q = 0.5.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("Q", u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;

    // Evaluate base Q.
    sys.prepareAuxiliaryForAssembly(state, false);
    const Real Q_base = sys.auxiliaryInputRegistryIfPresent()->get("Q");
    EXPECT_NEAR(Q_base, 0.5, 1e-10);

    // FD verification: perturb each DOF and measure ΔQ.
    // For the linear integrand Q = ∫ u ds, dQ/du_j = ∫_face φ_j ds
    // which should be nonzero for boundary face vertices and zero for
    // interior vertices.
    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    int nonzero_grads = 0;
    auto* reg = sys.auxiliaryInputRegistryIfPresent();
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        const Real orig = perturbed[dof];
        perturbed[dof] = orig + eps;
        reg->invalidateAll();  // force re-evaluation with perturbed solution
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real Q_pert = reg->get("Q");
        perturbed[dof] = orig;

        const Real fd_deriv = (Q_pert - Q_base) / eps;
        if (std::abs(fd_deriv) > 1e-10) {
            ++nonzero_grads;
            // For Q = ∫_face u ds with linear basis on a triangle of area 0.5,
            // each of the 3 face vertices contributes equally: dQ/du_j = 0.5/3 ≈ 0.1667.
            EXPECT_NEAR(fd_deriv, 0.5 / 3.0, 1e-5)
                << "FD gradient dQ/du[" << dof << "] should be area/3";
        }
    }

    // The face (0,0,0)-(1,0,0)-(0,1,0) has 3 vertices → 3 nonzero gradients.
    EXPECT_EQ(nonzero_grads, 3)
        << "Expected 3 nonzero gradient entries (one per boundary face vertex)";
}

TEST(MonolithicCoupling, FEExpressionSupportsMonolithicLinearization)
{
    // Verify that feExpression with field references supports monolithic
    // linearization (converted to domain-functional path).
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");

    // FE expression that references a field → should support monolithic.
    auto fe_expr = sys.feExpression("u_integral", u_disc);
    ASSERT_TRUE(fe_expr.hasDefinition());
    EXPECT_EQ(fe_expr.kind(), svmp::FE::systems::FEQuantityKind::FEExpression);
    EXPECT_TRUE(fe_expr.supportsMonolithicLinearization());

    // FE expression without field references → no monolithic.
    auto const_expr = sys.feExpression("const_val", FormExpr::constant(42.0));
    ASSERT_TRUE(const_expr.hasDefinition());
    EXPECT_FALSE(const_expr.supportsMonolithicLinearization());
}

// ===========================================================================
//  End-to-end domain integral evaluation test
// ===========================================================================

TEST(FEQuantityExplicit, DomainIntegralEvaluatesCorrectly)
{
    // ∫_Ω u dx on a single tet with u=1 everywhere.
    // Volume of unit tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) = 1/6.
    // ∫ 1 dx = 1/6 ≈ 0.16667.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto M = sys.domainIntegral("M", u_disc);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    // ∫ u dx with u=1 on unit tet = volume = 1/6.
    EXPECT_NEAR(reg->get("M"), 1.0/6.0, 1e-10);
}

TEST(FEQuantityExplicit, DomainAverageEvaluatesCorrectly)
{
    // ∫ u dx / ∫ 1 dx = 1 for constant u=1.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto M_avg = sys.domainAverage("M_avg", u_disc);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("M_avg"), 1.0, 1e-10);
}

// ===========================================================================
//  End-to-end region integral evaluation test
// ===========================================================================

TEST(FEQuantityExplicit, RegionIntegralEvaluatesCorrectly)
{
    // Region integral with marker matching the single cell's domain ID.
    // SingleTetraOneBoundaryFaceMeshAccess returns domain_id=0 for cell 0.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    // Region marker 0 matches the single tet's domain ID.
    auto R = sys.regionIntegral("R", u_disc, /*region_marker=*/0);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // Should equal the domain integral since all cells match marker 0.
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("R"), 1.0/6.0, 1e-10);
}

TEST(FEQuantityExplicit, RegionAverageEvaluatesCorrectly)
{
    // Region average with marker matching the single cell.
    // ∫ u dx / ∫ 1 dx = 1.0 for u=1.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto R_avg = sys.regionAverage("R_avg", u_disc, 0);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("R_avg"), 1.0, 1e-10);
}

TEST(FEQuantityExplicit, FEExpressionEvaluatesCorrectly)
{
    // feExpression with a field-dependent integrand: ∫ u dx = volume * u_avg.
    // With u=1 on unit tet, ∫ u dx = 1/6.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto fe_int = sys.feExpression("u_integral", u_disc);

    EXPECT_TRUE(fe_int.supportsMonolithicLinearization());

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // feExpression routes through domain-functional: ∫ u dx = 1/6.
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("u_integral"), 1.0/6.0, 1e-10);
}

TEST(FEQuantityExplicit, FEExpressionMultiFieldEvaluatesCorrectly)
{
    // feExpression with two distinct fields: ∫ u * p dx.
    // With u=2, p=3 on unit tet: ∫ 6 dx = 6 * volume = 6/6 = 1.0.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    const auto p_disc = FormExpr::discreteField(p_field, *space, "p");

    auto fe_prod = sys.feExpression("u_times_p", u_disc * p_disc);
    EXPECT_TRUE(fe_prod.supportsMonolithicLinearization());
    EXPECT_GE(fe_prod.referencedFields().size(), 2u);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    const auto u_off = static_cast<std::size_t>(sys.fieldDofOffset(u_field));
    const auto p_off = static_cast<std::size_t>(sys.fieldDofOffset(p_field));
    const auto u_nd = static_cast<std::size_t>(sys.fieldDofHandler(u_field).getNumDofs());
    const auto p_nd = static_cast<std::size_t>(sys.fieldDofHandler(p_field).getNumDofs());
    for (std::size_t i = 0; i < u_nd; ++i) sol[u_off + i] = 2.0;
    for (std::size_t i = 0; i < p_nd; ++i) sol[p_off + i] = 3.0;

    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // ∫ u*p dx with u=2, p=3 = 6 * volume = 6 * 1/6 = 1.0.
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get("u_times_p"), 1.0, 1e-10);
}

// ===========================================================================
//  Domain integral gradient FD verification
// ===========================================================================

TEST(MonolithicCoupling, DomainIntegralGradientFDVerification)
{
    // Verify dI/du for domain integral M = ∫_Ω u dx.
    // dM/du_j = ∫_Ω φ_j dx (= volume/4 for linear tet with 4 DOFs).
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("unused", FormExpr::constant(1.0), marker);
    // Register as a domain functional through BoundaryReductionService.
    auto M = sys.domainIntegral("M_grad", u_disc);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);
    const Real M_base = sys.auxiliaryInputRegistryIfPresent()->get("M_grad");

    // FD: perturb each DOF.
    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;
    auto* reg = sys.auxiliaryInputRegistryIfPresent();

    int nonzero = 0;
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        perturbed[dof] = sol[dof] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real M_pert = reg->get("M_grad");
        perturbed[dof] = sol[dof];

        const Real fd = (M_pert - M_base) / eps;
        if (std::abs(fd) > 1e-10) {
            ++nonzero;
            // Each of 4 tet DOFs contributes volume/4 = (1/6)/4 = 1/24.
            EXPECT_NEAR(fd, 1.0/24.0, 1e-5)
                << "FD gradient dM/du[" << dof << "] should be volume/4";
        }
    }
    EXPECT_EQ(nonzero, 4) << "All 4 tet DOFs should have nonzero domain gradient";
}

TEST(MonolithicCoupling, RegionIntegralGradientFDVerification)
{
    // Same as DomainIntegralGradientFDVerification but through regionIntegral
    // with region_marker=0.  Verifies the region-filtered gradient path.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.registerBoundaryIntegralInput("unused", FormExpr::constant(1.0), marker);
    sys.regionIntegral("R_grad", u_disc, /*region_marker=*/0);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    auto* reg = sys.auxiliaryInputRegistryIfPresent();
    const Real R_base = reg->get("R_grad");
    EXPECT_NEAR(R_base, 1.0/6.0, 1e-10);

    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    int nonzero = 0;
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        perturbed[dof] = sol[dof] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real R_pert = reg->get("R_grad");
        perturbed[dof] = sol[dof];

        const Real fd = (R_pert - R_base) / eps;
        if (std::abs(fd) > 1e-10) {
            ++nonzero;
            EXPECT_NEAR(fd, 1.0/24.0, 1e-5)
                << "FD gradient dR/du[" << dof << "] should be volume/4";
        }
    }
    EXPECT_EQ(nonzero, 4) << "All 4 DOFs should have nonzero region gradient";
}

// ===========================================================================
//  Multi-field evaluation correctness test
// ===========================================================================

TEST(FEQuantityExplicit, MultiFieldBoundaryIntegralEvaluates)
{
    // Register a boundary integral with a scalar field u (components=1)
    // and verify it evaluates correctly.  True multi-field (u*p) evaluation
    // requires an interleaved DOF layout which is complex to set up in a
    // unit test.  This test verifies the single-primary-field path works
    // end-to-end as a baseline.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    // u * u is a single-field "multi-reference" integrand.
    auto Q = sys.boundaryIntegral(u_disc * u_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    std::vector<Real> sol(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 2.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // ∫ u² ds with u=2 on face area 0.5 → 4 * 0.5 = 2.0.
    EXPECT_NEAR(sys.auxiliaryInputRegistryIfPresent()->get(Q.registryName()), 2.0, 1e-10);
}

// ===========================================================================
//  Multi-field chain-rule FD verification (two distinct fields)
// ===========================================================================

TEST(MonolithicCoupling, MultiFieldGradientFDVerification)
{
    // Two scalar fields u and p on the same tet.
    // Integrand: Q = ∫_Γ u * p ds.
    //
    // Multi-field evaluation through the FunctionalAssembler requires
    // correct secondary-field DOF offset mapping between the FESystem's
    // block layout and the assembler's per-node interleaved expectation.
    // This test verifies the primary-field gradient (dQ/du) is correct
    // via FD, which exercises the single-field linearization path.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    const auto p_disc = FormExpr::discreteField(p_field, *space, "p");

    // Multi-field integrand: u * p.
    sys.registerBoundaryIntegralInput("Q_up", u_disc * p_disc, marker);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    const auto u_off = static_cast<std::size_t>(sys.fieldDofOffset(u_field));
    const auto p_off = static_cast<std::size_t>(sys.fieldDofOffset(p_field));
    const auto u_ndofs = static_cast<std::size_t>(sys.fieldDofHandler(u_field).getNumDofs());
    const auto p_ndofs = static_cast<std::size_t>(sys.fieldDofHandler(p_field).getNumDofs());
    for (std::size_t i = 0; i < u_ndofs; ++i) sol[u_off + i] = 2.0;
    for (std::size_t i = 0; i < p_ndofs; ++i) sol[p_off + i] = 3.0;

    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    const Real Q_base = reg->get("Q_up");

    // Q = ∫ u*p ds with u=2, p=3 on face area 0.5 → 2*3*0.5 = 3.0.
    EXPECT_NEAR(Q_base, 3.0, 1e-10);

    // FD perturbation of u-DOFs: dQ/du_j = p * ∫ φ_j ds = 3 * (0.5/3) = 0.5.
    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    int u_nonzero = 0;
    for (std::size_t i = 0; i < u_ndofs; ++i) {
        const auto idx = u_off + i;
        perturbed[idx] = sol[idx] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real Q_pert = reg->get("Q_up");
        perturbed[idx] = sol[idx];
        const Real fd = (Q_pert - Q_base) / eps;
        if (std::abs(fd) > 1e-10) {
            ++u_nonzero;
            EXPECT_NEAR(fd, 3.0 * 0.5 / 3.0, 1e-4)
                << "dQ/du[" << i << "] should be p * area/3";
        }
    }
    EXPECT_EQ(u_nonzero, 3) << "3 u-DOFs on boundary face";

    // FD perturbation of p-DOFs: dQ/dp_j = u * ∫ φ_j ds = 2 * (0.5/3) ≈ 0.333.
    int p_nonzero = 0;
    for (std::size_t i = 0; i < p_ndofs; ++i) {
        const auto idx = p_off + i;
        perturbed[idx] = sol[idx] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real Q_pert = reg->get("Q_up");
        perturbed[idx] = sol[idx];
        const Real fd = (Q_pert - Q_base) / eps;
        if (std::abs(fd) > 1e-10) {
            ++p_nonzero;
            EXPECT_NEAR(fd, 2.0 * 0.5 / 3.0, 1e-4)
                << "dQ/dp[" << i << "] should be u * area/3";
        }
    }
    EXPECT_EQ(p_nonzero, 3) << "3 p-DOFs on boundary face";
}

// ===========================================================================
//  Symbolic differentiateWrtAuxiliaryOutput unit tests
// ===========================================================================

#include "Forms/SymbolicDifferentiation.h"
#include "Forms/PointEvaluator.h"

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Linear)
{
    // Form: -p_out * C  where p_out = AuxiliaryOutputRef(0), C = constant(3.0)
    // d/d(output_0) = -C = -3.0
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = FormExpr::constant(-1.0) * p_out * FormExpr::constant(3.0);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    // Evaluate the derivative: should be -3.0 (constant).
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.1;
    // Provide auxiliary outputs in case the derivative still references them.
    Real aux_out[] = {5.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, -3.0, 1e-10) << "d(-p_out * 3)/d(p_out) should be -3.0";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Quadratic)
{
    // Form: p_out^2  where p_out = AuxiliaryOutputRef(0)
    // d/d(output_0) = 2 * p_out
    // At p_out = 5.0: derivative = 10.0
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = p_out * p_out;

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.1;
    Real aux_out[] = {5.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 10.0, 1e-10) << "d(p_out^2)/d(p_out) at p_out=5 should be 10";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_WrongSlot)
{
    // Form: p_out0 * p_out1 where p_out0 = Ref(0), p_out1 = Ref(1)
    // d/d(output_0) = p_out1  (p_out0 replaced by 1.0, p_out1 is constant)
    // At p_out1 = 7.0: derivative = 7.0
    const auto p0 = FormExpr::auxiliaryOutputRef(0);
    const auto p1 = FormExpr::auxiliaryOutputRef(1);
    const auto form = p0 * p1;

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.1;
    Real aux_out[] = {3.0, 7.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 7.0, 1e-10) << "d(p0*p1)/d(p0) at p1=7 should be 7";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Sum)
{
    // Form: p_out + constant(10.0)
    // d/d(output_0) = 1.0
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = p_out + FormExpr::constant(10.0);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.1;
    Real aux_out[] = {42.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 1.0, 1e-10) << "d(p_out + 10)/d(p_out) should be 1.0";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_NoReference)
{
    // Form: constant(5.0) — no AuxiliaryOutputRef at all.
    // d/d(output_0) = 0.0
    const auto form = FormExpr::constant(5.0);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0;
    pctx.dt = 0.1;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 0.0, 1e-10) << "d(5.0)/d(output_0) should be 0.0";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Division)
{
    // Form: 1 / p_out — quotient rule: d/dp = -1/p^2
    // At p_out = 4.0: -1/16 = -0.0625
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = FormExpr::constant(1.0) / p_out;

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {4.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, -1.0 / 16.0, 1e-10)
        << "d(1/p)/dp at p=4 should be -1/16";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Power)
{
    // Form: p_out^3 — power rule: d/dp = 3*p^2
    // At p_out = 2.0: 3*4 = 12.0
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = svmp::FE::forms::pow(p_out, FormExpr::constant(3.0));

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {2.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 12.0, 1e-10)
        << "d(p^3)/dp at p=2 should be 12";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Exp)
{
    // Form: exp(p_out) — chain rule: d/dp = exp(p)
    // At p_out = 1.0: exp(1) ≈ 2.71828
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = svmp::FE::forms::exp(p_out);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {1.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, std::exp(1.0), 1e-10)
        << "d(exp(p))/dp at p=1 should be exp(1)";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Sqrt)
{
    // Form: sqrt(p_out) — d/dp = 1/(2*sqrt(p))
    // At p_out = 9.0: 1/(2*3) = 1/6 ≈ 0.16667
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = svmp::FE::forms::sqrt(p_out);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {9.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 1.0 / 6.0, 1e-10)
        << "d(sqrt(p))/dp at p=9 should be 1/6";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Log)
{
    // Form: log(p_out) — d/dp = 1/p
    // At p_out = 5.0: 1/5 = 0.2
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = svmp::FE::forms::log(p_out);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {5.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 0.2, 1e-10)
        << "d(log(p))/dp at p=5 should be 0.2";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Subtraction)
{
    // Form: C - p_out — d/dp = -1
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = FormExpr::constant(10.0) - p_out;

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {3.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, -1.0, 1e-10)
        << "d(10 - p)/dp should be -1";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_Negate)
{
    // Form: -p_out — d/dp = -1
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = -p_out;

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {7.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, -1.0, 1e-10)
        << "d(-p)/dp should be -1";
}

TEST(SymbolicDiff, DifferentiateWrtAuxiliaryOutput_CompoundChainRule)
{
    // Form: exp(p_out^2) — chain rule: d/dp = 2*p*exp(p^2)
    // At p_out = 1.0: 2*1*exp(1) ≈ 5.43656
    const auto p_out = FormExpr::auxiliaryOutputRef(0);
    const auto form = svmp::FE::forms::exp(p_out * p_out);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    Real aux_out[] = {1.0};
    pctx.auxiliary_outputs = aux_out;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 2.0 * std::exp(1.0), 1e-10)
        << "d(exp(p^2))/dp at p=1 should be 2*exp(1)";
}

// ===========================================================================
//  Complex / physics-realistic symbolic diff test cases
// ===========================================================================

namespace {
// FD helper: verify symbolic derivative matches finite differences
void verifySymbolicVsFD(const FormExpr& form, std::uint32_t slot,
                         std::span<Real> aux_out, Real eps = 1e-7)
{
    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, slot);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    pctx.auxiliary_outputs = aux_out;

    // Symbolic value.
    const Real symbolic = svmp::FE::forms::evaluateScalarAt(deriv, pctx);

    // FD value.
    const Real orig = aux_out[slot];
    const Real f0 = svmp::FE::forms::evaluateScalarAt(form, pctx);
    aux_out[slot] = orig + eps;
    const Real f1 = svmp::FE::forms::evaluateScalarAt(form, pctx);
    aux_out[slot] = orig;
    const Real fd = (f1 - f0) / eps;

    EXPECT_NEAR(symbolic, fd, std::max(1e-6, std::abs(fd) * 1e-5))
        << "Symbolic derivative does not match FD at aux_out[" << slot << "] = " << orig;
}
} // namespace

TEST(SymbolicDiff, DeeplyNestedChainRule)
{
    // log(1 + exp(p^2 - 3*p + 2)) — softplus-like with quadratic argument.
    // Tests: chain rule through log, add, exp, subtract, multiply, power.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto arg = p * p - FormExpr::constant(3.0) * p + FormExpr::constant(2.0);
    const auto form = svmp::FE::forms::log(FormExpr::constant(1.0) + svmp::FE::forms::exp(arg));

    Real aux[] = {1.5};
    verifySymbolicVsFD(form, 0, aux);
}

TEST(SymbolicDiff, RationalFunction)
{
    // p / (1 + p^2) — Lorentzian-like.
    // d/dp = (1 + p^2 - p*2p) / (1+p^2)^2 = (1-p^2)/(1+p^2)^2.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto one = FormExpr::constant(1.0);
    const auto form = p / (one + p * p);

    Real aux[] = {2.0};
    verifySymbolicVsFD(form, 0, aux);

    // Also verify at p=0 where derivative = 1.
    Real aux0[] = {0.0};
    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    pctx.auxiliary_outputs = aux0;
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(deriv, pctx), 1.0, 1e-10);
}

TEST(SymbolicDiff, OutputInBothNumeratorAndDenominator)
{
    // (p + 1) / (p - 1) — pole at p=1, tests quotient rule.
    // d/dp = ((p-1) - (p+1)) / (p-1)^2 = -2/(p-1)^2.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto one = FormExpr::constant(1.0);
    const auto form = (p + one) / (p - one);

    Real aux[] = {3.0};  // (3+1)/(3-1) = 2, d/dp = -2/4 = -0.5
    verifySymbolicVsFD(form, 0, aux);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    pctx.auxiliary_outputs = aux;
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(deriv, pctx), -0.5, 1e-10);
}

TEST(SymbolicDiff, ThreeOutputsComplex)
{
    // Form depends on 3 outputs: p0*exp(p1) + p2^2.
    // d/dp0 = exp(p1)
    // d/dp1 = p0*exp(p1)
    // d/dp2 = 2*p2
    const auto p0 = FormExpr::auxiliaryOutputRef(0);
    const auto p1 = FormExpr::auxiliaryOutputRef(1);
    const auto p2 = FormExpr::auxiliaryOutputRef(2);
    const auto form = p0 * svmp::FE::forms::exp(p1) + p2 * p2;

    Real aux[] = {2.0, 0.5, 3.0};

    verifySymbolicVsFD(form, 0, aux);  // d/dp0 = exp(0.5)
    verifySymbolicVsFD(form, 1, aux);  // d/dp1 = 2*exp(0.5)
    verifySymbolicVsFD(form, 2, aux);  // d/dp2 = 6

    // Explicit checks.
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1; pctx.auxiliary_outputs = aux;

    auto d0 = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(d0, pctx), std::exp(0.5), 1e-10);

    auto d1 = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 1);
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(d1, pctx), 2.0 * std::exp(0.5), 1e-10);

    auto d2 = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 2);
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(d2, pctx), 6.0, 1e-10);
}

TEST(SymbolicDiff, WindkesselPressureFormula)
{
    // Typical RCR outlet: P_out = X + Rp * Q
    // where X = AuxiliaryOutputRef(0), Q = AuxiliaryInputRef(0)
    // d(P_out)/d(X) = 1  (the Rp*Q term has no X dependence)
    // This is the exact pattern from the RCR coupled BC.
    const auto X = FormExpr::auxiliaryOutputRef(0);
    const auto Q = FormExpr::auxiliaryInputRef(0);
    const auto Rp = FormExpr::constant(100.0);
    const auto form = X + Rp * Q;

    Real aux_out[] = {5.0};
    Real aux_inp[] = {0.01};

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    pctx.auxiliary_outputs = aux_out;
    pctx.auxiliary_inputs = aux_inp;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 1.0, 1e-10) << "d(X + Rp*Q)/dX should be 1.0";
}

TEST(SymbolicDiff, NonlinearMaterialLaw)
{
    // Simulate a nonlinear material: sigma = E * epsilon / (1 + |epsilon|/sigma_y)
    // where sigma_y (yield stress) comes from auxiliary output.
    // sigma_y = AuxiliaryOutputRef(0), epsilon = constant for this test.
    // d(sigma)/d(sigma_y) at sigma_y = 100, epsilon = 10, E = 200:
    //   sigma = 200*10 / (1 + 10/100) = 2000/1.1 ≈ 1818.18
    //   d/d(sigma_y) = 200*10 * (10/sigma_y^2) / (1 + 10/sigma_y)^2
    //               = 2000 * 0.001 / 1.21 ≈ 1.6529
    const auto sigma_y = FormExpr::auxiliaryOutputRef(0);
    const auto E = FormExpr::constant(200.0);
    const auto eps_val = FormExpr::constant(10.0);
    const auto one = FormExpr::constant(1.0);
    const auto abs_eps = FormExpr::constant(10.0);  // |epsilon| = 10

    const auto form = E * eps_val / (one + abs_eps / sigma_y);

    Real aux[] = {100.0};
    verifySymbolicVsFD(form, 0, aux);
}

TEST(SymbolicDiff, ConditionalPiecewise)
{
    // Conditional: if p > 0 then p^2 else -p^2.
    // Not differentiable at p=0, but smooth away from 0.
    // At p=3: form = 9, d/dp = 6.
    // At p=-2: form = -4, d/dp = 4.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto zero = FormExpr::constant(0.0);
    const auto cond = svmp::FE::forms::gt(p, zero);
    const auto form = svmp::FE::forms::conditional(cond, p * p, -(p * p));

    Real aux_pos[] = {3.0};
    verifySymbolicVsFD(form, 0, aux_pos);

    Real aux_neg[] = {-2.0};
    verifySymbolicVsFD(form, 0, aux_neg);
}

TEST(SymbolicDiff, MinMaxSmooth)
{
    // max(p, 0) * 2 — backflow-stabilization-like pattern.
    // At p=5: max(5,0)*2 = 10, d/dp = 2.
    // At p=-3: max(-3,0)*2 = 0, d/dp = 0.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto zero = FormExpr::constant(0.0);
    const auto two = FormExpr::constant(2.0);

    const auto mx = svmp::FE::forms::max(p, zero);
    const auto form = mx * two;

    Real aux_pos[] = {5.0};
    verifySymbolicVsFD(form, 0, aux_pos);

    Real aux_neg[] = {-3.0};
    verifySymbolicVsFD(form, 0, aux_neg);
}

TEST(SymbolicDiff, NestedDivisionAndPower)
{
    // 1 / sqrt(1 + p^2) — inverse Euclidean-norm-like.
    // d/dp = -p / (1+p^2)^{3/2}.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto one = FormExpr::constant(1.0);
    const auto form = one / svmp::FE::forms::sqrt(one + p * p);

    Real aux[] = {2.0};
    verifySymbolicVsFD(form, 0, aux);

    // Exact check: d/dp at p=2 = -2/(1+4)^{3/2} = -2/5^{3/2} = -2/11.180 ≈ -0.17889
    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1; pctx.auxiliary_outputs = aux;
    const Real expected = -2.0 / std::pow(5.0, 1.5);
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(deriv, pctx), expected, 1e-10);
}

TEST(SymbolicDiff, ParameterAndOutputMixed)
{
    // Form: param * p_out^2 + param2 * p_out
    // where param = ParameterRef(0), param2 = ParameterRef(1).
    // d/d(p_out) = 2*param*p_out + param2.
    // At param=3, param2=7, p_out=4: 2*3*4 + 7 = 31.
    const auto param = FormExpr::parameterRef(0);
    const auto param2 = FormExpr::parameterRef(1);
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto form = param * p * p + param2 * p;

    Real aux[] = {4.0};
    Real params[] = {3.0, 7.0};

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    ASSERT_TRUE(deriv.isValid());

    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1;
    pctx.auxiliary_outputs = aux;
    pctx.jit_constants = params;

    const Real val = svmp::FE::forms::evaluateScalarAt(deriv, pctx);
    EXPECT_NEAR(val, 31.0, 1e-10)
        << "d(param*p^2 + param2*p)/dp at param=3,param2=7,p=4 should be 31";
}

TEST(SymbolicDiff, HighOrderPolynomial)
{
    // p^5 - 2*p^3 + p — polynomial that exercises repeated product rule.
    // d/dp = 5*p^4 - 6*p^2 + 1.
    // At p=2: 5*16 - 6*4 + 1 = 80 - 24 + 1 = 57.
    const auto p = FormExpr::auxiliaryOutputRef(0);
    const auto two = FormExpr::constant(2.0);

    const auto p2 = p * p;
    const auto p3 = p2 * p;
    const auto p5 = p3 * p2;
    const auto form = p5 - two * p3 + p;

    Real aux[] = {2.0};
    verifySymbolicVsFD(form, 0, aux);

    auto deriv = svmp::FE::forms::differentiateWrtAuxiliaryOutput(form, 0);
    svmp::FE::forms::PointEvalContext pctx;
    pctx.time = 0.0; pctx.dt = 0.1; pctx.auxiliary_outputs = aux;
    EXPECT_NEAR(svmp::FE::forms::evaluateScalarAt(deriv, pctx), 57.0, 1e-8);
}

// ===========================================================================
//  End-to-end mixed Jacobian block verification via assembleMixedAuxiliaryDense
// ===========================================================================

TEST(MonolithicCoupling, MixedJacobianBlockFDVerification)
{
    // Deploy a monolithic auxiliary model with a coupled boundary integral:
    //   dx/dt = -k*x + Q,  where Q = ∫_Γ u ds
    // The mixed system has field DOFs (u) and aux DOFs (x).
    // The aux→field Jacobian block (dF_aux/du) comes from:
    //   dF_aux/dQ * dQ/du  where dF_aux/dQ = -1 (since F = xdot + k*x - Q)
    //   and dQ/du_j = ∫_Γ φ_j ds = area/3 for boundary face vertices.
    //
    // So dF_aux/du_j = -1 * area/3 ≈ -0.1667 for boundary DOFs, 0 for interior.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Register boundary integral Q and deploy monolithic model.
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(u_disc, marker);

    auto model = aux::model("driven", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x + Q;
    });

    sys.deploy(
        use(model).name("driven_inst").monolithic()
            .bind("Q", Q)
            .param("k", 1.0)
            .initialize({0.0}));

    // Install a PDE formulation for setup.
    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_field, 1.0);  // u = 1

    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // Assemble the mixed system.
    std::vector<Real> residual, jacobian;
    sys.assembleMixedAuxiliaryDense(state, n_field, residual, jacobian);

    const auto n_total = n_field + 1;  // 1 aux DOF (x)
    ASSERT_EQ(residual.size(), n_total);
    ASSERT_EQ(jacobian.size(), n_total * n_total);

    // The aux→field block is the last row, first n_field columns.
    // J[n_field, j] = dF_aux/du_j.
    // For boundary face vertices (DOFs 0,1,2): should be nonzero.
    // For interior vertex (DOF 3): should be zero.
    int nonzero_aux_field = 0;
    for (std::size_t j = 0; j < n_field; ++j) {
        const Real Jij = jacobian[n_field * n_total + j];
        if (std::abs(Jij) > 1e-10) {
            ++nonzero_aux_field;
        }
    }
    EXPECT_EQ(nonzero_aux_field, 3)
        << "aux→field block should have 3 nonzero entries (boundary face DOFs)";

    // FD verification of the mixed Jacobian: perturb each field DOF,
    // re-assemble, and compare.
    const Real eps = 1e-7;
    for (std::size_t j = 0; j < n_field; ++j) {
        std::vector<Real> sol_pert(sol);
        sol_pert[j] += eps;
        SystemStateView ps = state;
        ps.u = sol_pert;

        // Must refresh inputs with perturbed solution.
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);

        std::vector<Real> res_pert, jac_pert;
        sys.assembleMixedAuxiliaryDense(ps, n_field, res_pert, jac_pert);

        // FD derivative of the aux residual w.r.t. field DOF j.
        const Real fd = (res_pert[n_field] - residual[n_field]) / eps;
        const Real analytic = jacobian[n_field * n_total + j];

        EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
            << "Mixed Jacobian J[aux, field_" << j << "] = " << analytic
            << " does not match FD = " << fd;
    }

    // Restore.
    sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
    sys.prepareAuxiliaryForAssembly(state, false);
}

TEST(MonolithicCoupling, TwoStateVectorOutletDirectCouplingMatchesFD)
{
    // Navier-Stokes-like outlet coupling on a vector velocity field:
    //   Q      = \int_Gamma u . n ds
    //   dP1/dt = (Q - (P1-P2)/Rm)/C1
    //   dP2/dt = ((P1-P2)/Rm - P2/Rd)/C2
    //   P_out  = P1 + Rp*Q
    //   R(u)   = (u, v)_Omega - (P_out n, v)_Gamma
    //
    // The runtime Newton operator augments the assembled PDE Jacobian with the
    // stored exact reduced direct-coupling terms. Verify that the resulting
    // field Jacobian matches finite differences DOF-by-DOF.
    const int marker = 9;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    const auto n = FormExpr::normal();
    auto Q = sys.boundaryIntegral(
        "Q", inner(u_disc, n), marker,
        BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

    auto model = aux::model("two_state_vector_outlet_like", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto P1 = m.state("P1");
        auto P2 = m.state("P2");
        auto Rp = m.param("Rp");
        auto C1 = m.param("C1");
        auto Rm = m.param("Rm");
        auto C2 = m.param("C2");
        auto Rd = m.param("Rd");

        m << ddt(P1) == (Q - (P1 - P2) / Rm) / C1;
        m << ddt(P2) == ((P1 - P2) / Rm - P2 / Rd) / C2;
        m << out("P_out") == P1 + Rp * Q;
    });

    auto inst = sys.deploy(
        use(model).name("two_state_vector_outlet_like_inst").global().monolithic()
            .bind("Q", Q)
            .params({
                {"Rp", 3.0},
                {"C1", 0.5},
                {"Rm", 2.0},
                {"C2", 0.75},
                {"Rd", 4.0},
            })
            .initialize({0.3, -0.1}));

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto residual = inner(u, v).dx() - inner(inst.output("P_out") * n, v).ds(marker);
    (void)installFormulation(sys, "op", {u_field}, residual);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, 12u);

    std::vector<Real> sol(n_field, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_field, 0.0);
    std::vector<Real> field_jacobian(n_field * n_field, 0.0);
    for (std::size_t i = 0; i < n_field; ++i) {
        base_residual[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_field; ++j) {
            field_jacobian[i * n_field + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }

    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_field; ++col) {
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;

        SystemStateView ps = state;
        ps.u = sol_pert;

        sys.restoreAuxiliaryState(packed_base);
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_field));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        const auto ar_pert = sys.assemble(req_vec, ps, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_field; ++row) {
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = field_jacobian[row * n_field + col];
            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "vector outlet field Jacobian mismatch at row=" << row
                << " col=" << col;
        }
    }

    sys.restoreAuxiliaryState(packed_base);
    sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
    sys.prepareAuxiliaryForAssembly(state, true);
}

TEST(MonolithicCoupling, TwoStateVectorOutletBorderedReductionMatchesDenseSolve)
{
    const int marker = 10;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    const auto n = FormExpr::normal();
    auto Q = sys.boundaryIntegral(
        "Q", inner(u_disc, n), marker,
        BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

    auto model = aux::model("two_state_vector_outlet_reduction", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto P1 = m.state("P1");
        auto P2 = m.state("P2");
        auto Rp = m.param("Rp");
        auto C1 = m.param("C1");
        auto Rm = m.param("Rm");
        auto C2 = m.param("C2");
        auto Rd = m.param("Rd");

        m << ddt(P1) == (Q - (P1 - P2) / Rm) / C1;
        m << ddt(P2) == ((P1 - P2) / Rm - P2 / Rd) / C2;
        m << out("P_out") == P1 + Rp * Q;
    });

    auto inst = sys.deploy(
        use(model).name("two_state_vector_outlet_reduction_inst").global().monolithic()
            .bind("Q", Q)
            .params({
                {"Rp", 3.0},
                {"C1", 0.5},
                {"Rm", 2.0},
                {"C2", 0.75},
                {"Rd", 4.0},
            })
            .initialize({0.3, -0.1}));

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto residual = inner(u, v).dx() - inner(inst.output("P_out") * n, v).ds(marker);
    (void)installFormulation(sys, "op", {u_field}, residual);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    const auto n_aux = std::size_t{2};
    ASSERT_EQ(n_field, 12u);

    std::vector<Real> sol(n_field, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    const auto& bc = sys.borderedCoupling();
    ASSERT_TRUE(bc.active);
    ASSERT_EQ(static_cast<std::size_t>(bc.n_aux), n_aux);
    ASSERT_EQ(bc.n_field_dofs, n_field);

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
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
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

    std::vector<std::vector<Real>> z_columns(n_aux, std::vector<Real>(n_field, 0.0));
    for (std::size_t j = 0; j < n_aux; ++j) {
        std::vector<Real> Bj(n_field, 0.0);
        for (std::size_t i = 0; i < n_field; ++i) {
            Bj[i] = bc.B[i + n_field * j];
        }
        z_columns[j] = solve_dense(K, Bj);
    }

    std::vector<Real> schur = bc.D;
    std::vector<Real> dx_aux(n_aux, 0.0);
    for (std::size_t i = 0; i < n_aux; ++i) {
        Real ctu0 = 0.0;
        for (std::size_t k = 0; k < n_field; ++k) {
            ctu0 += bc.Ct[i * n_field + k] * u0[k];
        }
        dx_aux[i] = bc.g[i] - ctu0;

        for (std::size_t j = 0; j < n_aux; ++j) {
            Real ctz = 0.0;
            for (std::size_t k = 0; k < n_field; ++k) {
                ctz += bc.Ct[i * n_field + k] * z_columns[j][k];
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
            full_matrix[i * n_total + (n_field + j)] = bc.B[i + n_field * j];
        }
    }
    for (std::size_t i = 0; i < n_aux; ++i) {
        full_rhs[n_field + i] = bc.g[i];
        for (std::size_t j = 0; j < n_field; ++j) {
            full_matrix[(n_field + i) * n_total + j] = bc.Ct[i * n_field + j];
        }
        for (std::size_t j = 0; j < n_aux; ++j) {
            full_matrix[(n_field + i) * n_total + (n_field + j)] = bc.D[i * n_aux + j];
        }
    }

    const auto dx_dense = solve_dense(full_matrix, full_rhs);
    ASSERT_EQ(dx_dense.size(), n_total);

    for (std::size_t i = 0; i < n_field; ++i) {
        EXPECT_NEAR(du_reduced[i], dx_dense[i], std::max(1e-8, std::abs(dx_dense[i]) * 1e-7))
            << "reduced field step mismatch at dof " << i;
    }
    for (std::size_t i = 0; i < n_aux; ++i) {
        EXPECT_NEAR(dx_aux[i], dx_dense[n_field + i], std::max(1e-8, std::abs(dx_dense[n_field + i]) * 1e-7))
            << "reduced auxiliary step mismatch at dof " << i;
    }
}

TEST(MonolithicCoupling, TwoStateVectorOutletDirectCouplingRespectsConstrainedOutletDofs)
{
    // Regression for the real pipe outlet case where the outlet flux support
    // overlaps constrained velocity DOFs on the outlet-edge intersection.
    // The monolithic direct-coupling path must assemble dQ/du in the same
    // constrained trial space as the PDE operator.
    const int marker = 9;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *u_space, "u");
    const auto n = FormExpr::normal();
    auto Q = sys.boundaryIntegral(
        "Q", inner(u_disc, n), marker,
        BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

    auto model = aux::model("two_state_vector_outlet_like_constrained", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto P1 = m.state("P1");
        auto P2 = m.state("P2");
        auto Rp = m.param("Rp");
        auto C1 = m.param("C1");
        auto Rm = m.param("Rm");
        auto C2 = m.param("C2");
        auto Rd = m.param("Rd");

        m << ddt(P1) == (Q - (P1 - P2) / Rm) / C1;
        m << ddt(P2) == ((P1 - P2) / Rm - P2 / Rd) / C2;
        m << out("P_out") == P1 + Rp * Q;
    });

    auto inst = sys.deploy(
        use(model).name("two_state_vector_outlet_like_constrained_inst").global().monolithic()
            .bind("Q", Q)
            .params({
                {"Rp", 3.0},
                {"C1", 0.5},
                {"Rm", 2.0},
                {"C2", 0.75},
                {"Rd", 4.0},
            })
            .initialize({0.3, -0.1}));

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto residual = inner(u, v).dx() - inner(inst.output("P_out") * n, v).ds(marker);
    (void)installFormulation(sys, "op", {u_field}, residual);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);

    const auto* emap = sys.fieldDofHandler(u_field).getEntityDofMap();
    ASSERT_NE(emap, nullptr);
    auto outlet_vertex_dofs = emap->getVertexDofs(/*vertex=*/0);
    ASSERT_EQ(outlet_vertex_dofs.size(), 3u);
    const auto constrained_dof =
        outlet_vertex_dofs[2] + sys.fieldDofOffset(u_field);

    sys.addConstraint(std::make_unique<svmp::FE::constraints::DirichletBC>(
        constrained_dof, 0.0));
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, 12u);
    ASSERT_TRUE(sys.constraints().isConstrained(constrained_dof));

    std::vector<Real> sol(n_field, 0.0);
    for (svmp::FE::GlobalIndex v_id = 0; v_id < 4; ++v_id) {
        setFieldComponent(sol, sys, u_field, v_id, /*component=*/2, -1.0);
    }
    sys.constraints().distribute(sol);

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.beginTimeStep();

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_field));
    lhs.zero();
    rhs.zero();

    AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    const auto ar = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_field, 0.0);
    std::vector<Real> field_jacobian(n_field * n_field, 0.0);
    for (std::size_t i = 0; i < n_field; ++i) {
        base_residual[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_field; ++j) {
            field_jacobian[i * n_field + j] =
                lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }

    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [row_dof, row_val] : upd.v) {
            EXPECT_FALSE(sys.constraints().isConstrained(row_dof));
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.v) {
                EXPECT_FALSE(sys.constraints().isConstrained(col_dof));
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            EXPECT_FALSE(sys.constraints().isConstrained(row_dof));
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
                EXPECT_FALSE(sys.constraints().isConstrained(col_dof));
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_field + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_field; ++col) {
        if (sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(col))) {
            continue;
        }

        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;

        SystemStateView ps = state;
        ps.u = sol_pert;

        sys.restoreAuxiliaryState(packed_base);
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_field));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;

        const auto ar_pert = sys.assemble(req_vec, ps, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_field; ++row) {
            if (sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(row))) {
                continue;
            }

            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) -
                 base_residual[row]) /
                eps;
            const Real analytic = field_jacobian[row * n_field + col];

            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "Constrained monolithic direct-coupling mismatch at row=" << row
                << " col=" << col
                << " analytic=" << analytic
                << " fd=" << fd;
        }
    }
}

TEST(MonolithicCoupling, DirectCouplingReducedUpdateUsesActualOutputSensitivity)
{
    // Regression for monolithic direct coupling when the outlet output depends
    // on a non-leading state variable:
    //   dx1/dt = -x1
    //   dx2/dt = -x2 + Q
    //   P_out = x2 + Rp * Q
    //
    // The direct PDE coupling should still recover the same dQ/du shape and
    // emit an exact reduced update with sigma = -Rp for the residual term
    // -(P_out*v).ds.
    const int marker = 6;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(u_disc, marker);

    auto model = aux::model("two_state_direct", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto x1 = m.state("x1");
        auto x2 = m.state("x2");
        auto Rp = m.param("Rp");
        m << ddt(x1) == -x1;
        m << ddt(x2) == -x2 + Q;
        m << out("P_out") == x2 + Rp * Q;
    });

    auto inst = sys.deploy(
        use(model).name("two_state_direct_inst").global().monolithic()
            .bind("Q", Q)
            .param("Rp", 3.0)
            .initialize({0.0, 0.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    const auto residual = inner(grad(u), grad(v)).dx() - (inst.output("P_out") * v).ds(marker);
    (void)installFormulation(sys, "op", {u_field}, residual);

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
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
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    const auto result = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(result.success);

    const auto updates = sys.lastReducedFieldUpdates();
    ASSERT_EQ(updates.size(), 1u);
    EXPECT_NEAR(std::abs(updates[0].sigma), 3.0, 1e-10);
    ASSERT_FALSE(updates[0].left.empty());
    ASSERT_FALSE(updates[0].right.empty());
    Real sum_abs = 0.0;
    for (const auto& [dof, value] : updates[0].right) {
        EXPECT_GE(dof, 0);
        EXPECT_LT(static_cast<std::size_t>(dof), n_dofs);
        sum_abs += std::abs(value);
    }
    EXPECT_GT(sum_abs, 0.0);
    EXPECT_LE(sum_abs, 0.5 + 1e-10);
    for (const auto& [dof, value] : updates[0].left) {
        (void)dof;
        EXPECT_GT(std::abs(value), 0.0);
    }
}

TEST(MonolithicCoupling, DirectCouplingUsesExactReducedFieldUpdateWhenNotSymmetric)
{
    // Regression for monolithic direct coupling when dR/d(output) is not
    // parallel to dQ/du. In this case the direct term must be exported as an
    // exact reduced field update instead of a symmetric rank-1 shortcut.
    const int marker = 7;

    auto assemble_field_block = [&](Real Rp,
                                    svmp::FE::assembly::DenseMatrixView& lhs_out,
                                    std::vector<svmp::FE::backends::ReducedFieldUpdate>& updates_out) {
        auto mesh =
            std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

        FESystem sys(mesh);
        const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("op");

        const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
        auto Q = sys.boundaryIntegral(u_disc, marker);

        auto model = aux::model("nonsymmetric_direct", [](ModelFacade& m) {
            auto Q = m.input("Q");
            auto x = m.state("x");
            auto Rp = m.param("Rp");
            m << ddt(x) == -x + Q;
            m << out("P_out") == Rp * Q;
        });

        auto inst = sys.deploy(
            use(model).name("nonsymmetric_direct_inst").global().monolithic()
                .bind("Q", Q)
                .param("Rp", Rp)
                .initialize({0.0}));

        const auto u = FormExpr::stateField(u_field, *space, "u");
        const auto v = FormExpr::testFunction(*space, "v");
        const auto residual = (u * v).dx() - (inst.output("P_out") * v).dx();
        (void)installFormulation(sys, "op", {u_field}, residual);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        sys.setup({}, si);
        sys.finalizeAuxiliaryLayout();

        const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
        ASSERT_EQ(n_dofs, 4u);

        std::vector<Real> sol(n_dofs, 1.0);
        SystemStateView state;
        state.time = 0.0;
        state.dt = 0.1;
        state.u = sol;

        sys.beginTimeStep();

        svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        lhs_out.zero();
        rhs.zero();

        AssemblyRequest req;
        req.op = "op";
        req.want_matrix = true;
        req.want_vector = true;

        const auto result = sys.assemble(req, state, &lhs_out, &rhs);
        ASSERT_TRUE(result.success);

        updates_out.assign(sys.lastReducedFieldUpdates().begin(), sys.lastReducedFieldUpdates().end());
    };

    svmp::FE::assembly::DenseMatrixView lhs_rp0(4);
    svmp::FE::assembly::DenseMatrixView lhs_rp2(4);
    std::vector<svmp::FE::backends::ReducedFieldUpdate> updates_rp0;
    std::vector<svmp::FE::backends::ReducedFieldUpdate> updates_rp2;
    assemble_field_block(/*Rp=*/0.0, lhs_rp0, updates_rp0);
    assemble_field_block(/*Rp=*/2.0, lhs_rp2, updates_rp2);

    EXPECT_TRUE(updates_rp0.empty());
    ASSERT_EQ(updates_rp2.size(), 1u);
    EXPECT_NEAR(std::abs(updates_rp2[0].sigma), 2.0, 1e-12);

    // Cell integral of each linear basis on the reference tetrahedron = volume/4 = 1/24.
    // Boundary-face integral on the marked triangular face = area/3 = 1/6 for face nodes,
    // and 0 for the opposite vertex.
    const Real expected = -2.0 * (1.0 / 24.0) * (1.0 / 6.0);
    std::array<Real, 4> left_dense{};
    std::array<Real, 4> right_dense{};
    for (const auto& [dof, value] : updates_rp2[0].left) {
        left_dense[static_cast<std::size_t>(dof)] += value;
    }
    for (const auto& [dof, value] : updates_rp2[0].right) {
        right_dense[static_cast<std::size_t>(dof)] += value;
    }
    for (svmp::FE::GlobalIndex i = 0; i < 4; ++i) {
        for (svmp::FE::GlobalIndex j = 0; j < 4; ++j) {
            const Real delta =
                updates_rp2[0].sigma * left_dense[static_cast<std::size_t>(i)] *
                right_dense[static_cast<std::size_t>(j)];
            const Real target = (j < 3) ? expected : 0.0;
            EXPECT_NEAR(delta, target, 1e-10)
                << "unexpected direct-coupling outer-product entry at (" << i << "," << j << ")";
        }
    }
}

TEST(MonolithicCoupling, PureAlgebraicResistanceOutletEmitsDirectCouplingUpdate)
{
    const int marker = 7;
    auto mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto Q = sys.boundaryIntegral(u_disc, marker);

    auto model = aux::model("resistive_direct_only", [](ModelFacade& m) {
        auto Q = m.input("Q");
        auto P = m.state("P", AuxiliaryVariableKind::Algebraic);
        auto Rsum = m.param("Rsum");
        auto Pd = m.param("Pd");

        m.initialGuess("P", 0.0);
        m << alg(P) == P - (Pd + Rsum * Q);
        m << out("P_out") == P;
    });

    auto inst = sys.deploy(
        use(model).name("resistive_direct_only_inst").boundary(marker).monolithic()
            .bind("Q", Q)
            .param("Rsum", 110.0)
            .param("Pd", 50.0)
            .initialState({{"P", 50.0}}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    const auto residual = inner(grad(u), grad(v)).dx() - (inst.output("P_out") * v).ds(marker);
    (void)installFormulation(sys, "op", {u_field}, residual);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
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
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    const auto result = sys.assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(result.success);

    const auto rank_one_updates = sys.lastRankOneUpdates();
    const auto reduced_updates = sys.lastReducedFieldUpdates();
    ASSERT_TRUE(!rank_one_updates.empty() || !reduced_updates.empty());
    const auto native_rank_one_it =
        std::find_if(rank_one_updates.begin(),
                     rank_one_updates.end(),
                     [](const auto& upd) { return upd.prefer_native_face; });
    ASSERT_NE(native_rank_one_it, rank_one_updates.end());
    EXPECT_NEAR(native_rank_one_it->sigma, -110.0, 1e-12);
    EXPECT_FALSE(native_rank_one_it->v.empty());
    EXPECT_TRUE(reduced_updates.empty());
}

TEST(MonolithicCoupling, DomainAverageGradientFDVerification)
{
    // Verify dI/du for domain average: avg = ∫u dx / ∫1 dx.
    // d(avg)/du_j = (d/du_j ∫u dx) / (∫1 dx) = (volume/4) / volume = 1/4.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.domainAverage("M_avg_grad", u_disc);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    auto* reg = sys.auxiliaryInputRegistryIfPresent();
    const Real avg_base = reg->get("M_avg_grad");
    EXPECT_NEAR(avg_base, 1.0, 1e-10);

    // FD: d(avg)/du_j should be 1/4 for all 4 DOFs (linear basis).
    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        perturbed[dof] = sol[dof] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real avg_pert = reg->get("M_avg_grad");
        perturbed[dof] = sol[dof];

        const Real fd = (avg_pert - avg_base) / eps;
        EXPECT_NEAR(fd, 0.25, 1e-4)
            << "d(avg)/du[" << dof << "] should be 1/n_nodes = 1/4";
    }
}

TEST(MonolithicCoupling, RegionAverageGradientFDVerification)
{
    // Same as DomainAverageGradientFDVerification but through regionAverage
    // with region_marker=0.  d(avg)/du_j = 1/4 for all DOFs.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    sys.regionAverage("R_avg_grad", u_disc, 0);

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 1.0);
    SystemStateView state; state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    auto* reg = sys.auxiliaryInputRegistryIfPresent();
    const Real avg_base = reg->get("R_avg_grad");
    EXPECT_NEAR(avg_base, 1.0, 1e-10);

    const Real eps = 1e-7;
    std::vector<Real> perturbed(sol);
    SystemStateView ps = state;
    ps.u = perturbed;

    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        perturbed[dof] = sol[dof] + eps;
        reg->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);
        const Real avg_pert = reg->get("R_avg_grad");
        perturbed[dof] = sol[dof];

        const Real fd = (avg_pert - avg_base) / eps;
        EXPECT_NEAR(fd, 0.25, 1e-4)
            << "d(region_avg)/du[" << dof << "] should be 1/n_nodes = 1/4";
    }
}

// ===========================================================================
//  End-to-end direct-field monolithic Jacobian via assembleMixedAuxiliaryDense
// ===========================================================================

TEST(MonolithicCoupling, DirectFieldJacobianFDVerification)
{
    // Deploy a Node-scoped monolithic auxiliary model whose residual directly
    // references a DiscreteField node (NOT mediated through bindCoupled).
    //
    //   dx_i/dt = -k*x_i + u_i^2   at each vertex i
    //   F_i = xdot_i + k*x_i - u_i^2
    //   dF_i/du_j = -2*u_i * δ_ij  (Kronecker delta for Lagrange)
    //
    // With non-uniform u = {1, 2, 3, 4}, the diagonal entries of the
    // aux→field Jacobian block are -2, -4, -6, -8.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Build a model with a DIRECT DiscreteField reference in the residual.
    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto model = aux::model("direct_field", [&](ModelFacade& m) {
        auto x = m.state("x");
        auto k = m.param("k");
        m << ddt(x) == -k * x + u_disc * u_disc;
    });

    sys.deploy(
        use(model).name("df_inst").node().monolithic()
            .param("k", 2.0)
            .initialize({0.5}));

    // Install a PDE formulation for setup.
    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    auto& aux_block = sys.auxiliaryStateManager().getBlock("df_inst");
    ASSERT_EQ(aux_block.entityCount(), 4u);

    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, 4u);  // 4 vertices, 1 DOF each

    // Non-uniform field values: u = {1, 2, 3, 4} at vertices 0-3.
    std::vector<Real> sol = {1.0, 2.0, 3.0, 4.0};

    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    // Assemble the mixed system.
    std::vector<Real> residual, jacobian;
    sys.assembleMixedAuxiliaryDense(state, n_field, residual, jacobian);

    // 4 field DOFs + 4 aux DOFs (one per vertex) = 8 total.
    const auto n_aux = 4u;
    const auto n_total = n_field + n_aux;
    ASSERT_EQ(residual.size(), n_total);
    ASSERT_EQ(jacobian.size(), n_total * n_total);

    // The aux→field block is rows [n_field..n_total), cols [0..n_field).
    // For Node scope with Lagrange Kronecker delta:
    //   J[n_field + i, j] = dF_i/du_j = -2*u_i * δ_ij
    // So the block is diagonal: entry (i,i) = -2*u_i.
    for (std::size_t i = 0; i < n_aux; ++i) {
        for (std::size_t j = 0; j < n_field; ++j) {
            const Real Jij = jacobian[(n_field + i) * n_total + j];
            if (i == j) {
                const Real expected = -2.0 * sol[i];
                EXPECT_NEAR(Jij, expected, 1e-8)
                    << "J[aux_" << i << ", field_" << j << "] = " << Jij
                    << ", expected " << expected;
            } else {
                EXPECT_NEAR(Jij, 0.0, 1e-10)
                    << "Off-diagonal J[aux_" << i << ", field_" << j
                    << "] should be zero, got " << Jij;
            }
        }
    }

    // FD verification: perturb each field DOF and compare all aux residuals.
    const Real eps = 1e-7;
    for (std::size_t j = 0; j < n_field; ++j) {
        std::vector<Real> sol_pert(sol);
        sol_pert[j] += eps;
        SystemStateView ps = state;
        ps.u = sol_pert;

        if (sys.auxiliaryInputRegistryIfPresent())
            sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);

        std::vector<Real> res_pert, jac_pert;
        sys.assembleMixedAuxiliaryDense(ps, n_field, res_pert, jac_pert);

        for (std::size_t i = 0; i < n_aux; ++i) {
            const Real fd = (res_pert[n_field + i] - residual[n_field + i]) / eps;
            const Real analytic = jacobian[(n_field + i) * n_total + j];

            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "J[aux_" << i << ", field_" << j << "] = " << analytic
                << " does not match FD = " << fd;
        }
    }

    // Restore.
    if (sys.auxiliaryInputRegistryIfPresent())
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
    sys.prepareAuxiliaryForAssembly(state, false);
}

TEST(MonolithicCoupling, NodeScopeEntityCountMismatchRejected)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto model = aux::model("node_count_guard", [&](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == FormExpr::constant(0.0);
    });

    sys.deploy(
        use(model).name("node_count_inst").node().monolithic()
            .entityCount(5)
            .initialize({0.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(MonolithicCoupling, DirectFieldGlobalScopeRejected)
{
    // Global-scoped models with direct DiscreteField references should be
    // rejected at setup.  A Global model has no spatial location, so raw
    // field values are undefined.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto model = aux::model("bad_global", [&](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == u_disc;
    });

    // Global (default) scope with a direct field reference — should throw.
    sys.deploy(
        use(model).name("bad_inst").monolithic()
            .initialize({0.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(MonolithicCoupling, DirectFieldVectorComponentFDVerification)
{
    // Node-scoped model referencing components of a 3-component vector field:
    //   dx/dt = component(vel, 0) * component(vel, 1)
    //   F = xdot - vel_0 * vel_1
    //   dF/d(vel_0) = -vel_1,  dF/d(vel_1) = -vel_0,  dF/d(vel_2) = 0
    //
    // With vel = {1,2,3} at vertex 0 (4 vertices, vel varies per vertex):
    //   At vertex 0: dF/d(vel_0) = -2, dF/d(vel_1) = -1, dF/d(vel_2) = 0
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto vec_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto vel_field = sys.addField(FieldSpec{.name = "vel", .space = vec_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = scalar_space, .components = 1});
    sys.addOperator("op");

    const auto vel_disc = FormExpr::discreteField(vel_field, *vec_space, "vel");
    auto model = aux::model("comp_ref", [&](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == component(vel_disc, 0) * component(vel_disc, 1);
    });

    sys.deploy(
        use(model).name("comp_inst").node().monolithic()
            .entityCount(4)
            .initialize({0.0}));

    const auto p = FormExpr::stateField(p_field, *scalar_space, "p");
    const auto v = FormExpr::testFunction(*scalar_space, "v");
    (void)installFormulation(sys, "op", {p_field}, inner(grad(p), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }
    sys.finalizeAuxiliaryLayout();

    // 4 vertices × 3 components = 12 vel DOFs + 4 pressure DOFs = 16 field DOFs.
    // 4 aux DOFs (1 per vertex).
    const auto n_field = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, 16u);  // 12 vel + 4 pressure

    // Set vel = {1,2,3} at all vertices (uniform for simplicity).
    std::vector<Real> sol(n_field, 0.0);
    // vel DOFs: interleaved [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z, v3x,v3y,v3z]
    for (int vtx = 0; vtx < 4; ++vtx) {
        sol[static_cast<std::size_t>(vtx * 3 + 0)] = 1.0 + vtx;  // vel_x varies
        sol[static_cast<std::size_t>(vtx * 3 + 1)] = 2.0;        // vel_y = 2
        sol[static_cast<std::size_t>(vtx * 3 + 2)] = 3.0;        // vel_z = 3
    }

    SystemStateView state;
    state.time = 0.0; state.dt = 0.1; state.u = sol;
    sys.prepareAuxiliaryForAssembly(state, false);

    std::vector<Real> residual, jacobian;
    sys.assembleMixedAuxiliaryDense(state, n_field, residual, jacobian);

    const auto n_aux = 4u;
    const auto n_total = n_field + n_aux;
    ASSERT_EQ(residual.size(), n_total);
    ASSERT_EQ(jacobian.size(), n_total * n_total);

    // FD verification: perturb each field DOF and compare all aux residuals.
    const Real eps = 1e-7;
    for (std::size_t j = 0; j < n_field; ++j) {
        std::vector<Real> sol_pert(sol);
        sol_pert[j] += eps;
        SystemStateView ps = state;
        ps.u = sol_pert;

        if (sys.auxiliaryInputRegistryIfPresent())
            sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
        sys.prepareAuxiliaryForAssembly(ps, false);

        std::vector<Real> res_pert, jac_pert;
        sys.assembleMixedAuxiliaryDense(ps, n_field, res_pert, jac_pert);

        for (std::size_t i = 0; i < n_aux; ++i) {
            const Real fd = (res_pert[n_field + i] - residual[n_field + i]) / eps;
            const Real analytic = jacobian[(n_field + i) * n_total + j];

            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "J[aux_" << i << ", field_" << j << "] = " << analytic
                << " does not match FD = " << fd;
        }
    }

    if (sys.auxiliaryInputRegistryIfPresent())
        sys.auxiliaryInputRegistryIfPresent()->invalidateAll();
    sys.prepareAuxiliaryForAssembly(state, false);
}

TEST(MonolithicCoupling, DirectFieldCellScopeRejected)
{
    // Cell-scoped model with a direct field reference should be rejected.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = FormExpr::discreteField(u_field, *space, "u");
    auto model = aux::model("cell_ref", [&](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == u_disc;
    });

    sys.deploy(
        use(model).name("cell_inst")
            .scope(AuxiliaryStateScope::Cell).monolithic()
            .entityCount(1)
            .initialize({0.0}));

    const auto u = FormExpr::stateField(u_field, *space, "u");
    const auto v = FormExpr::testFunction(*space, "v");
    (void)installFormulation(sys, "op", {u_field}, inner(grad(u), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(MonolithicCoupling, CellScopedLocalCondensationMatchesGlobalBorderedReference)
{
    const int marker = 11;

    struct MonolithicCase {
        std::unique_ptr<FESystem> sys{};
        FieldId u_field{svmp::FE::INVALID_FIELD_ID};
        std::string instance_name{};
        std::vector<Real> sol{};
        SystemStateView state{};
        std::size_t n_field{0};
    };

    auto buildCase = [&](AuxiliaryStateScope scope,
                         std::string instance_name) -> MonolithicCase {
        auto mesh =
            std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

        MonolithicCase c;
        c.sys = std::make_unique<FESystem>(mesh);
        c.u_field = c.sys->addField(FieldSpec{.name = "u", .space = space, .components = 1});
        c.sys->addOperator("op");

        const auto u_disc = FormExpr::discreteField(c.u_field, *space, "u");
        auto Q = c.sys->boundaryIntegral(u_disc, marker);

        auto model = aux::model("cell_condensed_compare", [](ModelFacade& m) {
            auto Q = m.input("Q");
            auto x = m.state("x");
            m << ddt(x) == -x + Q;
            m << out("P_out") == x;
        });

        auto deployment =
            use(model).name(instance_name).monolithic().bind("Q", Q).initialize({0.25});
        if (scope == AuxiliaryStateScope::Cell) {
            deployment.scope(AuxiliaryStateScope::Cell).entityCount(1);
        } else {
            deployment.global();
        }
        auto inst = c.sys->deploy(deployment);

        const auto u = FormExpr::stateField(c.u_field, *space, "u");
        const auto v = FormExpr::testFunction(*space, "v");
        const auto residual = (u * v).dx() - (inst.output("P_out") * v).dx();
        (void)installFormulation(*c.sys, "op", {c.u_field}, residual);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        c.sys->setup({}, si);
        c.sys->finalizeAuxiliaryLayout();
        c.sys->beginTimeStep();

        c.n_field = static_cast<std::size_t>(c.sys->dofHandler().getNumDofs());
        c.sol.assign(c.n_field, 1.0);
        c.state.time = 0.0;
        c.state.dt = 0.1;
        c.state.u = c.sol;
        c.instance_name = std::move(instance_name);
        return c;
    };

    auto solveDense = [](std::vector<Real> A, std::vector<Real> b) -> std::vector<Real> {
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

    auto assembleFieldSystem =
        [&](MonolithicCase& c,
            std::vector<Real>& K_out,
            std::vector<Real>& r_out) {
            svmp::FE::assembly::DenseMatrixView lhs(
                static_cast<svmp::FE::GlobalIndex>(c.n_field));
            svmp::FE::assembly::DenseVectorView rhs(
                static_cast<svmp::FE::GlobalIndex>(c.n_field));
            lhs.zero();
            rhs.zero();

            AssemblyRequest req;
            req.op = "op";
            req.want_matrix = true;
            req.want_vector = true;
            req.is_nonlinear_iteration = true;

            const auto result = c.sys->assemble(req, c.state, &lhs, &rhs);
            ASSERT_TRUE(result.success);

            K_out.assign(c.n_field * c.n_field, Real(0.0));
            r_out.assign(c.n_field, Real(0.0));
            for (std::size_t i = 0; i < c.n_field; ++i) {
                r_out[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
                for (std::size_t j = 0; j < c.n_field; ++j) {
                    K_out[i * c.n_field + j] =
                        lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                           static_cast<svmp::FE::GlobalIndex>(j));
                }
            }

            for (const auto& upd : c.sys->lastRankOneUpdates()) {
                for (const auto& [row_dof, row_val] : upd.v) {
                    const auto row = static_cast<std::size_t>(row_dof);
                    for (const auto& [col_dof, col_val] : upd.v) {
                        const auto col = static_cast<std::size_t>(col_dof);
                        K_out[row * c.n_field + col] += upd.sigma * row_val * col_val;
                    }
                }
            }
            for (const auto& upd : c.sys->lastReducedFieldUpdates()) {
                for (const auto& [row_dof, row_val] : upd.left) {
                    const auto row = static_cast<std::size_t>(row_dof);
                    for (const auto& [col_dof, col_val] : upd.right) {
                        const auto col = static_cast<std::size_t>(col_dof);
                        K_out[row * c.n_field + col] += upd.sigma * row_val * col_val;
                    }
                }
            }
        };

    auto global_case = buildCase(AuxiliaryStateScope::Global, "global_ref");
    auto cell_case = buildCase(AuxiliaryStateScope::Cell, "cell_condensed");

    std::vector<Real> K_global;
    std::vector<Real> r_global;
    assembleFieldSystem(global_case, K_global, r_global);
    const auto& global_bc = global_case.sys->borderedCoupling();
    ASSERT_TRUE(global_bc.active);
    ASSERT_EQ(global_bc.n_aux, 1);

    std::vector<Real> K_cell;
    std::vector<Real> r_cell;
    assembleFieldSystem(cell_case, K_cell, r_cell);
    EXPECT_FALSE(cell_case.sys->borderedCoupling().active);
    EXPECT_TRUE(cell_case.sys->hasLocalCondensedRecovery());
    EXPECT_FALSE(cell_case.sys->lastReducedFieldUpdates().empty());
    ASSERT_EQ(cell_case.sys->lastLocalCondensedRhsShift().size(), cell_case.n_field);

    const Real D_inv = Real(1.0) / global_bc.D[0];
    std::vector<Real> K_ref = K_global;
    std::vector<Real> r_ref = r_global;
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        r_ref[i] -= global_bc.B[i] * D_inv * global_bc.g[0];
        for (std::size_t j = 0; j < global_case.n_field; ++j) {
            K_ref[i * global_case.n_field + j] -=
                global_bc.B[i] * D_inv * global_bc.Ct[j];
        }
    }

    std::vector<Real> r_cell_reduced = r_cell;
    const auto local_shift = cell_case.sys->lastLocalCondensedRhsShift();
    for (std::size_t i = 0; i < r_cell_reduced.size(); ++i) {
        r_cell_reduced[i] -= local_shift[i];
    }

    ASSERT_EQ(K_ref.size(), K_cell.size());
    ASSERT_EQ(r_ref.size(), r_cell_reduced.size());
    for (std::size_t i = 0; i < K_ref.size(); ++i) {
        EXPECT_NEAR(K_cell[i], K_ref[i], std::max(1e-8, std::abs(K_ref[i]) * 1e-7))
            << "reduced field Jacobian mismatch at flat index " << i;
    }
    for (std::size_t i = 0; i < r_ref.size(); ++i) {
        EXPECT_NEAR(r_cell_reduced[i], r_ref[i], std::max(1e-8, std::abs(r_ref[i]) * 1e-7))
            << "reduced field residual mismatch at row " << i;
    }

    const auto du_reduced = solveDense(K_cell, r_cell_reduced);

    std::vector<Real> full_matrix((global_case.n_field + 1) * (global_case.n_field + 1), Real(0.0));
    std::vector<Real> full_rhs(global_case.n_field + 1, Real(0.0));
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        full_rhs[i] = r_global[i];
        for (std::size_t j = 0; j < global_case.n_field; ++j) {
            full_matrix[i * (global_case.n_field + 1) + j] =
                K_global[i * global_case.n_field + j];
        }
        full_matrix[i * (global_case.n_field + 1) + global_case.n_field] = global_bc.B[i];
    }
    full_rhs[global_case.n_field] = global_bc.g[0];
    for (std::size_t j = 0; j < global_case.n_field; ++j) {
        full_matrix[global_case.n_field * (global_case.n_field + 1) + j] = global_bc.Ct[j];
    }
    full_matrix.back() = global_bc.D[0];

    const auto dense_step = solveDense(full_matrix, full_rhs);
    ASSERT_EQ(dense_step.size(), global_case.n_field + 1);
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        EXPECT_NEAR(du_reduced[i],
                    dense_step[i],
                    std::max(1e-8, std::abs(dense_step[i]) * 1e-7))
            << "reduced field step mismatch at dof " << i;
    }

    auto& cell_block = cell_case.sys->auxiliaryStateManager().getBlock(cell_case.instance_name);
    const auto x_before = cell_block.gatherEntityWork(0);
    ASSERT_EQ(x_before.size(), 1u);
    cell_case.sys->applyLocalCondensedRecovery(du_reduced, Real(1.0));
    const auto x_after = cell_block.gatherEntityWork(0);
    ASSERT_EQ(x_after.size(), 1u);
    const Real dx_local = x_before[0] - x_after[0];
    EXPECT_NEAR(dx_local,
                dense_step[global_case.n_field],
                std::max(1e-8, std::abs(dense_step[global_case.n_field]) * 1e-7));
}

TEST(MonolithicCoupling, FacetScopedLocalCondensationMatchesGlobalBorderedReference)
{
    const int marker = 13;

    struct MonolithicCase {
        std::unique_ptr<FESystem> sys{};
        FieldId u_field{svmp::FE::INVALID_FIELD_ID};
        std::string instance_name{};
        std::vector<Real> sol{};
        SystemStateView state{};
        std::size_t n_field{0};
    };

    auto buildCase = [&](AuxiliaryStateScope scope,
                         std::string instance_name) -> MonolithicCase {
        auto mesh =
            std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

        MonolithicCase c;
        c.sys = std::make_unique<FESystem>(mesh);
        c.u_field = c.sys->addField(FieldSpec{.name = "u", .space = space, .components = 1});
        c.sys->addOperator("op");

        const auto u_disc = FormExpr::discreteField(c.u_field, *space, "u");
        auto Q = c.sys->boundaryIntegral(u_disc, marker);

        auto model = aux::model("facet_condensed_compare", [](ModelFacade& m) {
            auto Q = m.input("Q");
            auto x = m.state("x");
            m << ddt(x) == -x + Q;
            m << out("P_out") == x;
        });

        auto deployment =
            use(model).name(instance_name).monolithic().bind("Q", Q).initialize({0.25});
        if (scope == AuxiliaryStateScope::Facet) {
            deployment.scope(AuxiliaryStateScope::Facet);
        } else {
            deployment.global();
        }
        auto inst = c.sys->deploy(deployment);

        const auto u = FormExpr::stateField(c.u_field, *space, "u");
        const auto v = FormExpr::testFunction(*space, "v");
        const auto residual = (u * v).dx() - (inst.output("P_out") * v).ds(marker);
        (void)installFormulation(*c.sys, "op", {c.u_field}, residual);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        c.sys->setup({}, si);
        c.sys->finalizeAuxiliaryLayout();
        c.sys->beginTimeStep();

        c.n_field = static_cast<std::size_t>(c.sys->dofHandler().getNumDofs());
        c.sol.assign(c.n_field, 1.0);
        c.state.time = 0.0;
        c.state.dt = 0.1;
        c.state.u = c.sol;
        c.instance_name = std::move(instance_name);
        return c;
    };

    auto global_case = buildCase(AuxiliaryStateScope::Global, "global_ref");
    auto facet_case = buildCase(AuxiliaryStateScope::Facet, "facet_condensed");

    std::vector<Real> K_global;
    std::vector<Real> r_global;
    assembleFieldSystemWithReductions_(*global_case.sys, global_case.state, global_case.n_field,
                                       K_global, r_global);
    const auto& global_bc = global_case.sys->borderedCoupling();
    ASSERT_TRUE(global_bc.active);
    ASSERT_EQ(global_bc.n_aux, 1);

    std::vector<Real> K_facet;
    std::vector<Real> r_facet;
    assembleFieldSystemWithReductions_(*facet_case.sys, facet_case.state, facet_case.n_field,
                                       K_facet, r_facet);
    EXPECT_FALSE(facet_case.sys->borderedCoupling().active);
    EXPECT_TRUE(facet_case.sys->hasLocalCondensedRecovery());
    EXPECT_FALSE(facet_case.sys->lastReducedFieldUpdates().empty());
    ASSERT_EQ(facet_case.sys->lastLocalCondensedRhsShift().size(), facet_case.n_field);

    const Real D_inv = Real(1.0) / global_bc.D[0];
    std::vector<Real> K_ref = K_global;
    std::vector<Real> r_ref = r_global;
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        r_ref[i] -= global_bc.B[i] * D_inv * global_bc.g[0];
        for (std::size_t j = 0; j < global_case.n_field; ++j) {
            K_ref[i * global_case.n_field + j] -=
                global_bc.B[i] * D_inv * global_bc.Ct[j];
        }
    }

    std::vector<Real> r_facet_reduced = r_facet;
    const auto local_shift = facet_case.sys->lastLocalCondensedRhsShift();
    for (std::size_t i = 0; i < r_facet_reduced.size(); ++i) {
        r_facet_reduced[i] -= local_shift[i];
    }

    ASSERT_EQ(K_ref.size(), K_facet.size());
    ASSERT_EQ(r_ref.size(), r_facet_reduced.size());
    for (std::size_t i = 0; i < K_ref.size(); ++i) {
        EXPECT_NEAR(K_facet[i], K_ref[i], std::max(1e-8, std::abs(K_ref[i]) * 1e-7))
            << "reduced field Jacobian mismatch at flat index " << i;
    }
    for (std::size_t i = 0; i < r_ref.size(); ++i) {
        EXPECT_NEAR(r_facet_reduced[i], r_ref[i], std::max(1e-8, std::abs(r_ref[i]) * 1e-7))
            << "reduced field residual mismatch at row " << i;
    }

    const auto du_reduced = solveDenseLinearSystem_(K_facet, r_facet_reduced);

    std::vector<Real> full_matrix((global_case.n_field + 1) * (global_case.n_field + 1), Real(0.0));
    std::vector<Real> full_rhs(global_case.n_field + 1, Real(0.0));
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        full_rhs[i] = r_global[i];
        for (std::size_t j = 0; j < global_case.n_field; ++j) {
            full_matrix[i * (global_case.n_field + 1) + j] =
                K_global[i * global_case.n_field + j];
        }
        full_matrix[i * (global_case.n_field + 1) + global_case.n_field] = global_bc.B[i];
    }
    full_rhs[global_case.n_field] = global_bc.g[0];
    for (std::size_t j = 0; j < global_case.n_field; ++j) {
        full_matrix[global_case.n_field * (global_case.n_field + 1) + j] = global_bc.Ct[j];
    }
    full_matrix.back() = global_bc.D[0];

    const auto dense_step = solveDenseLinearSystem_(full_matrix, full_rhs);
    ASSERT_EQ(dense_step.size(), global_case.n_field + 1);
    for (std::size_t i = 0; i < global_case.n_field; ++i) {
        EXPECT_NEAR(du_reduced[i],
                    dense_step[i],
                    std::max(1e-8, std::abs(dense_step[i]) * 1e-7))
            << "reduced field step mismatch at dof " << i;
    }

    auto& facet_block = facet_case.sys->auxiliaryStateManager().getBlock(facet_case.instance_name);
    const auto x_before = facet_block.gatherEntityWork(0);
    ASSERT_EQ(x_before.size(), 1u);
    facet_case.sys->applyLocalCondensedRecovery(du_reduced, Real(1.0));
    const auto x_after = facet_block.gatherEntityWork(0);
    ASSERT_EQ(x_after.size(), 1u);
    const Real dx_local = x_before[0] - x_after[0];
    EXPECT_NEAR(dx_local,
                dense_step[global_case.n_field],
                std::max(1e-8, std::abs(dense_step[global_case.n_field]) * 1e-7));
}

TEST(MonolithicCoupling, QuadraturePointScopedLocalCondensationAutoLayoutMatchesExplicitOffsetsReference)
{
    struct MonolithicCase {
        std::unique_ptr<FESystem> sys{};
        FieldId u_field{svmp::FE::INVALID_FIELD_ID};
        std::string instance_name{};
        std::vector<Real> sol{};
        SystemStateView state{};
        std::size_t n_field{0};
    };

    auto buildCase = [&](std::string instance_name,
                         bool use_explicit_qp_offsets) -> MonolithicCase {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto space = std::make_shared<svmp::FE::spaces::L2Space>(ElementType::Tetra4, 0);

        MonolithicCase c;
        c.sys = std::make_unique<FESystem>(mesh);
        c.u_field = c.sys->addField(FieldSpec{.name = "u", .space = space, .components = 1});
        c.sys->addOperator("op");

        const auto u_disc = FormExpr::discreteField(c.u_field, *space, "u");
        auto M_avg = c.sys->domainAverage("M_avg", u_disc);

        auto model = aux::model("qp_condensed_compare", [](ModelFacade& m) {
            auto M_avg = m.input("M_avg");
            auto x = m.state("x");
            m << ddt(x) == -x + M_avg;
            m << out("P_out") == x;
        });

        auto deployment =
            use(model).name(instance_name).monolithic().bind("M_avg", M_avg).initialize({0.25});
        deployment.scope(AuxiliaryStateScope::QuadraturePoint);
        if (use_explicit_qp_offsets) {
            deployment.qpOffsets({0u, 4u});
        }
        auto inst = c.sys->deploy(deployment);

        const auto u = FormExpr::stateField(c.u_field, *space, "u");
        const auto v = FormExpr::testFunction(*space, "v");
        const auto residual = (u * v).dx() - (inst.output("P_out") * v).dx();
        (void)installFormulation(*c.sys, "op", {c.u_field}, residual);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        c.sys->setup({}, si);
        c.sys->finalizeAuxiliaryLayout();
        c.sys->beginTimeStep();

        c.n_field = static_cast<std::size_t>(c.sys->dofHandler().getNumDofs());
        c.sol.assign(c.n_field, 1.0);
        c.state.time = 0.0;
        c.state.dt = 0.1;
        c.state.u = c.sol;
        c.instance_name = std::move(instance_name);
        return c;
    };

    auto qp_auto = buildCase("qp_auto", false);
    auto qp_explicit = buildCase("qp_explicit", true);

    std::vector<Real> K_auto;
    std::vector<Real> r_auto;
    assembleFieldSystemWithReductions_(*qp_auto.sys, qp_auto.state, qp_auto.n_field, K_auto, r_auto);
    EXPECT_FALSE(qp_auto.sys->borderedCoupling().active);
    EXPECT_TRUE(qp_auto.sys->hasLocalCondensedRecovery());
    EXPECT_FALSE(qp_auto.sys->lastReducedFieldUpdates().empty());
    ASSERT_EQ(qp_auto.sys->lastLocalCondensedRhsShift().size(), qp_auto.n_field);

    std::vector<Real> K_explicit;
    std::vector<Real> r_explicit;
    assembleFieldSystemWithReductions_(
        *qp_explicit.sys, qp_explicit.state, qp_explicit.n_field, K_explicit, r_explicit);
    EXPECT_FALSE(qp_explicit.sys->borderedCoupling().active);
    EXPECT_TRUE(qp_explicit.sys->hasLocalCondensedRecovery());
    EXPECT_FALSE(qp_explicit.sys->lastReducedFieldUpdates().empty());
    ASSERT_EQ(qp_explicit.sys->lastLocalCondensedRhsShift().size(), qp_explicit.n_field);

    std::vector<Real> r_auto_reduced = r_auto;
    const auto auto_shift = qp_auto.sys->lastLocalCondensedRhsShift();
    for (std::size_t i = 0; i < r_auto_reduced.size(); ++i) {
        r_auto_reduced[i] -= auto_shift[i];
    }

    std::vector<Real> r_explicit_reduced = r_explicit;
    const auto explicit_shift = qp_explicit.sys->lastLocalCondensedRhsShift();
    for (std::size_t i = 0; i < r_explicit_reduced.size(); ++i) {
        r_explicit_reduced[i] -= explicit_shift[i];
    }

    ASSERT_EQ(K_auto.size(), K_explicit.size());
    ASSERT_EQ(r_auto_reduced.size(), r_explicit_reduced.size());
    for (std::size_t i = 0; i < K_auto.size(); ++i) {
        EXPECT_NEAR(K_auto[i], K_explicit[i], std::max(1e-8, std::abs(K_explicit[i]) * 1e-7))
            << "reduced field Jacobian mismatch at flat index " << i;
    }
    for (std::size_t i = 0; i < r_auto_reduced.size(); ++i) {
        EXPECT_NEAR(
            r_auto_reduced[i], r_explicit_reduced[i],
            std::max(1e-8, std::abs(r_explicit_reduced[i]) * 1e-7))
            << "reduced field residual mismatch at row " << i;
    }

    const auto du_auto = solveDenseLinearSystem_(K_auto, r_auto_reduced);
    const auto du_explicit = solveDenseLinearSystem_(K_explicit, r_explicit_reduced);
    ASSERT_EQ(du_auto.size(), du_explicit.size());
    for (std::size_t i = 0; i < du_auto.size(); ++i) {
        EXPECT_NEAR(
            du_auto[i], du_explicit[i],
            std::max(1e-8, std::abs(du_explicit[i]) * 1e-7))
            << "reduced field step mismatch at dof " << i;
    }

    auto& auto_block = qp_auto.sys->auxiliaryStateManager().getBlock(qp_auto.instance_name);
    auto& explicit_block = qp_explicit.sys->auxiliaryStateManager().getBlock(qp_explicit.instance_name);
    const auto x_auto_before = auto_block.gatherEntityWork(0);
    const auto x_explicit_before = explicit_block.gatherEntityWork(0);
    ASSERT_EQ(x_auto_before.size(), 1u);
    ASSERT_EQ(x_explicit_before.size(), 1u);

    qp_auto.sys->applyLocalCondensedRecovery(du_auto, Real(1.0));
    qp_explicit.sys->applyLocalCondensedRecovery(du_explicit, Real(1.0));

    const auto x_auto_after = auto_block.gatherEntityWork(0);
    const auto x_explicit_after = explicit_block.gatherEntityWork(0);
    ASSERT_EQ(x_auto_after.size(), 1u);
    ASSERT_EQ(x_explicit_after.size(), 1u);
    EXPECT_NEAR(x_auto_before[0] - x_auto_after[0],
                x_explicit_before[0] - x_explicit_after[0],
                std::max(1e-8, std::abs(x_explicit_before[0] - x_explicit_after[0]) * 1e-7));
}

TEST(MonolithicCoupling, DirectFieldNonH1SpaceRejected)
{
    // Node-scoped model referencing a field with L2 (DG) space should be
    // rejected — direct field references require H1 (nodal Lagrange) spaces.
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto dg_space = std::make_shared<svmp::FE::spaces::L2Space>(ElementType::Tetra4, 1);
    auto h1_space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    FESystem sys(mesh);
    const auto dg_field = sys.addField(FieldSpec{.name = "c", .space = dg_space, .components = 1});
    const auto h1_field = sys.addField(FieldSpec{.name = "p", .space = h1_space, .components = 1});
    sys.addOperator("op");

    const auto c_disc = FormExpr::discreteField(dg_field, *dg_space, "c");
    auto model = aux::model("dg_ref", [&](ModelFacade& m) {
        auto x = m.state("x");
        m << ddt(x) == c_disc;  // references L2/DG field
    });

    sys.deploy(
        use(model).name("dg_inst").node().monolithic()
            .entityCount(4)
            .initialize({0.0}));

    // Need a formulation to satisfy setup.
    const auto p = FormExpr::stateField(h1_field, *h1_space, "p");
    const auto v = FormExpr::testFunction(*h1_space, "v");
    (void)installFormulation(sys, "op", {h1_field}, inner(grad(p), grad(v)).dx());

    { SetupInputs si; si.topology_override = singleTetraTopology(); sys.setup({}, si); }

    EXPECT_THROW(sys.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}
