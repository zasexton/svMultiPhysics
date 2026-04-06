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
#include <unordered_set>

#include "Assembly/GlobalSystemView.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Systems/FESystem.h"
#include "FE/Systems/AuxiliaryInputRegistry.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/CoupledBoundaryManager.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TransientSystem.h"
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

std::string autoBoundaryIntegralName(int boundary_marker, std::size_t ordinal = 0)
{
    return "_boundary_integral_b" + std::to_string(boundary_marker) + "_" +
           std::to_string(ordinal);
}

std::string autoBoundaryInstanceName(std::string_view model_name, int boundary_marker)
{
    return std::string(model_name) + "_b" + std::to_string(boundary_marker);
}

bool containsExprType(const svmp::FE::forms::FormExpr& expr,
                      svmp::FE::forms::FormExprType target)
{
    if (!expr.isValid() || !expr.node()) {
        return false;
    }
    bool found = false;
    std::function<void(const svmp::FE::forms::FormExprNode&)> walk =
        [&](const svmp::FE::forms::FormExprNode& node) {
            if (node.type() == target) {
                found = true;
                return;
            }
            for (const auto* child : node.children()) {
                if (child && !found) {
                    walk(*child);
                }
            }
        };
    walk(*expr.node());
    return found;
}

class SingleTetraFourBoundaryFaceMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    SingleTetraFourBoundaryFaceMeshAccess(int outlet_marker, int inlet_marker, int wall_marker)
        : markers_{outlet_marker, inlet_marker, wall_marker, wall_marker}
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] svmp::FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] svmp::FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(svmp::FE::GlobalIndex) const override { return true; }
    [[nodiscard]] svmp::FE::ElementType getCellType(svmp::FE::GlobalIndex) const override { return svmp::FE::ElementType::Tetra4; }

    void getCellNodes(svmp::FE::GlobalIndex, std::vector<svmp::FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<svmp::FE::Real, 3> getNodeCoordinates(svmp::FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(svmp::FE::GlobalIndex,
                            std::vector<std::array<svmp::FE::Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_[i];
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(svmp::FE::GlobalIndex face_id,
                                                         svmp::FE::GlobalIndex) const override
    {
        return static_cast<svmp::FE::LocalIndex>(face_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(svmp::FE::GlobalIndex face_id) const override
    {
        return markers_.at(static_cast<std::size_t>(face_id));
    }

    [[nodiscard]] std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex> getInteriorFaceCells(
        svmp::FE::GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)> callback) const override
    {
        for (svmp::FE::GlobalIndex face = 0; face < static_cast<svmp::FE::GlobalIndex>(markers_.size()); ++face) {
            if (marker < 0 || markers_[static_cast<std::size_t>(face)] == marker) {
                callback(face, 0);
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)>) const override
    {
    }

private:
    std::array<int, 4> markers_{};
    std::vector<std::array<svmp::FE::Real, 3>> nodes_{};
    std::array<svmp::FE::GlobalIndex, 4> cell_{};
};

class SingleTetraTwoOutletBoundaryFaceMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    SingleTetraTwoOutletBoundaryFaceMeshAccess(int outlet0_marker,
                                               int outlet1_marker,
                                               int inlet_marker,
                                               int wall_marker)
        : markers_{outlet0_marker, outlet1_marker, inlet_marker, wall_marker}
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] svmp::FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] svmp::FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(svmp::FE::GlobalIndex) const override { return true; }
    [[nodiscard]] svmp::FE::ElementType getCellType(svmp::FE::GlobalIndex) const override
    {
        return svmp::FE::ElementType::Tetra4;
    }

    void getCellNodes(svmp::FE::GlobalIndex,
                      std::vector<svmp::FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<svmp::FE::Real, 3> getNodeCoordinates(
        svmp::FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(svmp::FE::GlobalIndex,
                            std::vector<std::array<svmp::FE::Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_[i];
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex getLocalFaceIndex(svmp::FE::GlobalIndex face_id,
                                                         svmp::FE::GlobalIndex) const override
    {
        return static_cast<svmp::FE::LocalIndex>(face_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(svmp::FE::GlobalIndex face_id) const override
    {
        return markers_.at(static_cast<std::size_t>(face_id));
    }

    [[nodiscard]] std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex> getInteriorFaceCells(
        svmp::FE::GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)> callback) const override
    {
        for (svmp::FE::GlobalIndex face = 0;
             face < static_cast<svmp::FE::GlobalIndex>(markers_.size());
             ++face) {
            if (marker < 0 || markers_[static_cast<std::size_t>(face)] == marker) {
                callback(face, 0);
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)>)
        const override
    {
    }

private:
    std::array<int, 4> markers_{};
    std::vector<std::array<svmp::FE::Real, 3>> nodes_{};
    std::array<svmp::FE::GlobalIndex, 4> cell_{};
};

struct ModuleAssemblySnapshot {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

void assignComponentPattern(std::vector<Real>& values,
                            const FESystem& sys,
                            const std::string& field_name,
                            int component,
                            Real base,
                            Real stride)
{
    const auto dofs = sys.fieldMap().getComponentDofs(field_name, static_cast<svmp::FE::LocalIndex>(component)).toVector();
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        values[static_cast<std::size_t>(dofs[i])] = base + stride * static_cast<Real>(i);
    }
}

ModuleAssemblySnapshot assembleTransientNavierStokesModuleWithResistiveOutlet(bool enable_jit)
{
    constexpr int marker = 91;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = enable_jit;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(marker, /*C=*/0.0));

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.05), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.03), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-1.10), Real(-0.02));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.20), Real(0.015));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.04), Real(-0.008));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.02), Real(0.004));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.95), Real(0.015));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.16), Real(-0.01));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.02), Real(0.006));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.01), Real(-0.003));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.85), Real(-0.01));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.12), Real(0.008));

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    TransientSystem transient(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    const auto ar = transient.assemble(req, state, &out, &out);
    EXPECT_TRUE(ar.success);

    return ModuleAssemblySnapshot{
        .matrix = std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
        .vector = std::vector<Real>(out.vectorData().begin(), out.vectorData().end()),
    };
}

ModuleAssemblySnapshot assembleTransientNavierStokesModuleWithDirichletAndResistiveOutlet(bool enable_jit)
{
    constexpr int outlet_marker = 191;
    constexpr int inlet_marker = 192;
    constexpr int wall_marker = 193;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = enable_jit;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    if (!sys.constraints().empty()) {
        sys.constraints().distribute(u_stage);
        sys.constraints().distribute(u_prev);
        sys.constraints().distribute(u_prev2);
    }

    if (!sys.constraints().empty()) {
        sys.constraints().distribute(u_stage);
        sys.constraints().distribute(u_prev);
        sys.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    TransientSystem transient(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    const auto ar = transient.assemble(req, state, &out, &out);
    EXPECT_TRUE(ar.success);

    return ModuleAssemblySnapshot{
        .matrix = std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
        .vector = std::vector<Real>(out.vectorData().begin(), out.vectorData().end()),
    };
}

void expectAdaptiveNear(const std::vector<Real>& actual,
                        const std::vector<Real>& expected,
                        Real abs_tol,
                        Real rel_tol,
                        std::string_view label)
{
    ASSERT_EQ(actual.size(), expected.size());

    std::size_t worst_idx = 0;
    Real worst_diff = 0.0;
    Real worst_ref = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const Real diff = std::abs(actual[i] - expected[i]);
        const Real ref = std::max(std::abs(actual[i]), std::abs(expected[i]));
        const Real tol = abs_tol + rel_tol * ref;
        if (diff > worst_diff) {
            worst_diff = diff;
            worst_ref = ref;
            worst_idx = i;
        }
        ASSERT_LE(diff, tol) << label << " mismatch at index " << i
                             << " actual=" << actual[i]
                             << " expected=" << expected[i]
                             << " tol=" << tol;
    }

    (void)worst_idx;
    (void)worst_diff;
    (void)worst_ref;
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
    auto bc = Factories::toCoupledOutflowBC(makeRCROpts(marker), sys, u, rho);
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
    const auto q_name = autoBoundaryIntegralName(marker);
    ASSERT_TRUE(reg->hasInput(q_name));
    EXPECT_NEAR(reg->get(q_name), 0.5, 1e-10);

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
    const auto instance_name = autoBoundaryInstanceName("rcr_windkessel", marker);
    const auto out_slot = sys.auxiliaryOutputSlotOf(instance_name, "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 55.0, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_EndToEnd)
{
    // Pure resistance is authored through the simplified AuxiliaryState API,
    // but the BC installation layer rewrites the algebraic output back to the
    // native coupled-boundary functional path so Newton sees the same direct
    // outlet operator as the legacy implementation.
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

    auto bc = Factories::toCoupledOutflowBC(makeRCROpts(marker, /*C=*/0.0), sys, u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::NaturalBC*>(bc.get()), nullptr);
    EXPECT_EQ(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q_test = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx();
    auto continuity = (q_test * div(u)).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto summary = sys.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);
    ASSERT_NE(sys.coupledBoundaryManager(), nullptr);

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

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    sys.beginTimeStep();
    const auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_FALSE(bc_data.active);
    EXPECT_EQ(bc_data.n_aux, 0);
    const auto rank_one_updates = sys.lastRankOneUpdates();
    EXPECT_EQ(rank_one_updates.size(), 1u);
    EXPECT_TRUE(sys.lastReducedFieldUpdates().empty());
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_UsesAuxiliaryInputRefsNotFieldTerminals)
{
    const auto model = Factories::detail::resistiveOutflowModel();
    ASSERT_NE(model, nullptr);

    auto contains_type = [](const svmp::FE::forms::FormExpr& expr,
                            svmp::FE::forms::FormExprType target) {
        if (!expr.isValid() || !expr.node()) return false;
        bool found = false;
        std::function<void(const svmp::FE::forms::FormExprNode&)> walk =
            [&](const svmp::FE::forms::FormExprNode& node) {
                if (node.type() == target) {
                    found = true;
                    return;
                }
                for (const auto* child : node.children()) {
                    if (child && !found) walk(*child);
                }
            };
        walk(*expr.node());
        return found;
    };

    const auto residuals = model->residualExpressions();
    ASSERT_EQ(residuals.size(), 1u);
    EXPECT_TRUE(contains_type(residuals[0], svmp::FE::forms::FormExprType::AuxiliaryInputRef));
    EXPECT_TRUE(contains_type(residuals[0], svmp::FE::forms::FormExprType::AuxiliaryStateRef));
    EXPECT_FALSE(contains_type(residuals[0], svmp::FE::forms::FormExprType::DiscreteField));
    EXPECT_FALSE(contains_type(residuals[0], svmp::FE::forms::FormExprType::StateField));

    const auto& outputs = model->outputExpressions();
    ASSERT_EQ(outputs.size(), 1u);
    for (const auto& [name, expr] : outputs) {
        (void)name;
        EXPECT_TRUE(contains_type(expr, svmp::FE::forms::FormExprType::AuxiliaryStateRef) ||
                    contains_type(expr, svmp::FE::forms::FormExprType::AuxiliaryInputRef));
        EXPECT_FALSE(contains_type(expr, svmp::FE::forms::FormExprType::DiscreteField));
        EXPECT_FALSE(contains_type(expr, svmp::FE::forms::FormExprType::StateField));
    }
}

TEST(NavierStokesOutletFactory, NewOverload_Resistive_FluxInstallation)
{
    // Verify the C==0 factory path returns a plain NaturalBC, then lowers the
    // algebraic output through coupled-boundary symbols during installation so
    // the assembled operator still uses the expected direct-coupling update.
    const int marker = 55;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    FESystem sys(mesh);
    const auto u_field = sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns");

    const auto u = FormExpr::stateField(u_field, *u_space, "u");
    const auto rho = FormExpr::constant(1.0);

    auto bc = Factories::toCoupledOutflowBC(makeRCROpts(marker, /*C=*/0.0), sys, u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_NE(dynamic_cast<svmp::FE::forms::bc::NaturalBC*>(bc.get()), nullptr);
    EXPECT_EQ(dynamic_cast<svmp::FE::forms::bc::CoupledNaturalBC*>(bc.get()), nullptr);
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    auto momentum = inner(grad(u), grad(v)).dx();
    auto continuity = (q * div(u)).dx();
    bc_manager.applyAll(sys, momentum, u, v, u_field);
    EXPECT_NE(sys.coupledBoundaryManager(), nullptr);
    (void)installFormulation(sys, "ns", {u_field, p_field}, momentum + continuity);

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

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    sys.beginTimeStep();
    const auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_FALSE(bc_data.active);
    EXPECT_EQ(bc_data.n_aux, 0);
    const auto updates = sys.lastRankOneUpdates();
    ASSERT_EQ(updates.size(), 1u);
    EXPECT_TRUE(updates[0].prefer_native_face);
    EXPECT_FALSE(updates[0].v.empty());
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

    auto bc = Factories::toCoupledOutflowBC(makeRCROpts(marker), sys, u, rho);
    ASSERT_NE(bc, nullptr);
    EXPECT_EQ(sys.coupledBoundaryManager(), nullptr);

    // Install the BC flux into the residual via BoundaryConditionManager.
    BoundaryConditionManager bc_manager;
    bc_manager.add(std::move(bc));

    const auto v = FormExpr::testFunction(*u_space, "v");
    auto residual = inner(grad(u), grad(v)).dx();
    bc_manager.applyAll(sys, residual, u, v, u_field);

    // installFormulation should resolve the auto-generated auxiliary output reference.
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
    EXPECT_NEAR(reg->get(autoBoundaryIntegralName(marker)), 0.5, 1e-10);

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

    const auto out_slot =
        sys.auxiliaryOutputSlotOf(autoBoundaryInstanceName("rcr_windkessel", marker), "P_out");
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> sol(n_dofs, 0.0);
    for (svmp::FE::GlobalIndex vtx = 0; vtx < 4; ++vtx) {
        setFieldComponent(sol, sys, u_field, vtx, /*component=*/2, -1.0);
    }
    if (!sys.constraints().empty()) {
        sys.constraints().distribute(sol);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    sys.advanceAuxiliaryState(state);
    sys.prepareAuxiliaryForAssembly(state, false);

    const auto* reg = sys.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(reg, nullptr);
    const auto q_name = autoBoundaryIntegralName(marker);
    ASSERT_TRUE(reg->hasInput(q_name));
    EXPECT_NEAR(reg->get(q_name), 0.5, 1e-10);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_FALSE(bc_data.active);

    const auto out_slot =
        sys.auxiliaryOutputSlotOf(autoBoundaryInstanceName("rcrcr_windkessel", marker), "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 65.0, 1e-8);
}

TEST(NavierStokesOutletFactory, NewOverload_RCRCR_FluxInstallation)
{
    // The generalized RCRCR outlet is monolithic: the first assembly exposes
    // bordered auxiliary blocks and the output stays at the current auxiliary
    // iterate until Newton updates the coupled state.
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns";
    req.want_vector = true;
    req.is_nonlinear_iteration = true;
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs.zero();
    sys.beginTimeStep();
    const auto result = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(result.success);

    const auto& bc_data = sys.borderedCoupling();
    EXPECT_TRUE(bc_data.active);

    const auto out_slot =
        sys.auxiliaryOutputSlotOf(autoBoundaryInstanceName("rcrcr_windkessel", marker), "P_out");
    ASSERT_NE(out_slot, std::string::npos);
    const auto outputs = sys.auxiliaryOutputValues();
    ASSERT_GT(outputs.size(), out_slot);
    EXPECT_NEAR(outputs[out_slot], 65.0, 1e-8);
}

// This single-tetra mixed-field finite-difference probe is currently not a
// faithful reproducer for the constrained bordered operator used by the live
// monolithic outlet path. The dedicated FE-system direct-coupling tests cover
// the exact reduced-update/Jacobian machinery; keep the outlet-factory wiring
// coverage enabled and defer this broader mixed-system probe until the test
// setup is rebuilt around a stable constrained reference operator.
TEST(NavierStokesOutletFactory, DISABLED_MonolithicRCRCR_MixedFieldJacobianMatchesFD)
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_dofs + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    std::vector<char> constrained(static_cast<std::size_t>(n_dofs), 0);
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        constrained[dof] =
            sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(dof)) ? 1 : 0;
    }

    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        if (constrained[col]) {
            continue;
        }
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;
        if (!sys.constraints().empty()) {
            sys.constraints().distribute(sol_pert);
        }

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
            if (constrained[row]) {
                continue;
            }
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

TEST(NavierStokesOutletFactory, ModuleJitParity_ResistiveOutflowGeneralizedAlpha)
{
    const auto jit = assembleTransientNavierStokesModuleWithResistiveOutlet(/*enable_jit=*/true);
    const auto fallback = assembleTransientNavierStokesModuleWithResistiveOutlet(/*enable_jit=*/false);

    expectAdaptiveNear(jit.matrix, fallback.matrix, 1e-10, 1e-9, "matrix");
    expectAdaptiveNear(jit.vector, fallback.vector, 1e-10, 1e-9, "vector");
}

TEST(NavierStokesOutletFactory, ModuleJitParity_DirichletAndResistiveOutflowGeneralizedAlpha)
{
    const auto jit =
        assembleTransientNavierStokesModuleWithDirichletAndResistiveOutlet(/*enable_jit=*/true);
    const auto fallback =
        assembleTransientNavierStokesModuleWithDirichletAndResistiveOutlet(/*enable_jit=*/false);

    expectAdaptiveNear(jit.matrix, fallback.matrix, 1e-10, 1e-9, "matrix");
    expectAdaptiveNear(jit.vector, fallback.vector, 1e-10, 1e-9, "vector");
}

// This single-tetra finite-difference check is currently not a faithful minimal
// reproducer for the application-level Newton Jacobian mismatch seen in
// `pipe_simple`. More targeted probes below show that the outlet does not alter
// the raw pressure residual or pressure-pressure Jacobian block on this reduced
// setup, so keep this disabled until the reduced reproducer is reconciled with
// the full application diagnostic.
TEST(NavierStokesOutletFactory, DISABLED_LegacyResistiveOutflowBoundaryMeshJacobianMatchesFD)
{
    constexpr int outlet_marker = 391;
    constexpr int inlet_marker = 392;
    constexpr int wall_marker = 393;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = false;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient =
        std::make_shared<svmp::FE::systems::TransientSystem>(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();
    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);
    const auto ar = transient->assemble(req, state, &lhs, &rhs);
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

    std::vector<char> constrained(static_cast<std::size_t>(n_dofs), 0);
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        constrained[dof] = sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(dof)) ? 1 : 0;
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        if (constrained[col]) {
            continue;
        }
        std::vector<Real> u_pert = u_stage;
        u_pert[col] += eps;
        if (!sys.constraints().empty()) {
            sys.constraints().distribute(u_pert);
        }

        SystemStateView pert_state = state;
        pert_state.u = u_pert;

        sys.restoreAuxiliaryState(packed_base);
        if (auto* reg = sys.auxiliaryInputRegistryIfPresent()) {
            reg->invalidateAll();
        }

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                          /*invalidate_auxiliary_inputs=*/false);
        const auto ar_pert = transient->assemble(req_vec, pert_state, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_dofs; ++row) {
            if (constrained[row]) {
                continue;
            }
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = field_jacobian[row * n_dofs + col];
            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "legacy resistive outlet Jacobian mismatch at row=" << row
                << " col=" << col;
        }
    }

    sys.restoreAuxiliaryState(packed_base);
    if (auto* reg = sys.auxiliaryInputRegistryIfPresent()) {
        reg->invalidateAll();
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowBoundaryFunctionalIgnoresPressureDofs)
{
    constexpr int outlet_marker = 311;
    constexpr int inlet_marker = 312;
    constexpr int wall_marker = 313;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = false;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto* cbm = sys.coupledBoundaryManager();
    ASSERT_NE(cbm, nullptr);
    cbm->prepareForAssembly(state);

    const std::string q_name = "ns_Q_" + std::to_string(outlet_marker);
    ASSERT_TRUE(cbm->integrals().has(q_name));
    const Real q_base = cbm->integrals().get(q_name);

    const auto pressure_dofs =
        sys.fieldMap().getComponentDofs("Pressure", static_cast<svmp::FE::LocalIndex>(0)).toVector();
    ASSERT_FALSE(pressure_dofs.empty());

    for (const auto dof : pressure_dofs) {
        std::vector<Real> perturbed = u_stage;
        perturbed[static_cast<std::size_t>(dof)] += Real(1e-7);

        SystemStateView pert_state = state;
        pert_state.u = perturbed;
        cbm->prepareForAssembly(pert_state);

        const Real q_pert = cbm->integrals().get(q_name);
        EXPECT_NEAR(q_pert, q_base, 1e-12)
            << "legacy outlet flow functional changed under pressure DOF perturbation at dof=" << dof;
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowResidualDoesNotTouchPressureRows)
{
    constexpr int outlet_marker = 331;
    constexpr int inlet_marker = 332;
    constexpr int wall_marker = 333;

    auto build_system = [&](bool with_outlet) {
        auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
            outlet_marker, inlet_marker, wall_marker);
        auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
        auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

        Opts opts;
        opts.enable_jit = false;
        opts.enable_convection = true;
        opts.enable_vms = true;
        opts.density = 1.06;
        opts.viscosity = 0.04;
        opts.velocity_field_name = "Velocity";
        opts.pressure_field_name = "Pressure";
        if (with_outlet) {
            opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
        }
        opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = inlet_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
        });
        opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = wall_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
        });

        FESystem sys(mesh);
        svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, opts);
        module.registerOn(sys);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        sys.setup({}, si);
        sys.finalizeAuxiliaryLayout();
        return sys;
    };

    auto with_outlet = build_system(true);
    auto without_outlet = build_system(false);

    const auto n_dofs = static_cast<std::size_t>(with_outlet.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, with_outlet, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, with_outlet, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, with_outlet, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, with_outlet, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, with_outlet, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, with_outlet, "Pressure", 0, Real(0.10), Real(0.007));

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient_with =
        std::make_shared<svmp::FE::systems::TransientSystem>(with_outlet, integrator);
    auto transient_without =
        std::make_shared<svmp::FE::systems::TransientSystem>(without_outlet, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = false;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseVectorView rhs_with(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_with.zero();
    const auto ar_with = transient_with->assemble(req, state, nullptr, &rhs_with);
    ASSERT_TRUE(ar_with.success);

    svmp::FE::assembly::DenseVectorView rhs_without(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_without.zero();
    const auto ar_without = transient_without->assemble(req, state, nullptr, &rhs_without);
    ASSERT_TRUE(ar_without.success);

    const auto pressure_dofs =
        with_outlet.fieldMap().getComponentDofs("Pressure", static_cast<svmp::FE::LocalIndex>(0)).toVector();
    ASSERT_FALSE(pressure_dofs.empty());

    for (const auto dof : pressure_dofs) {
        const auto idx = static_cast<svmp::FE::GlobalIndex>(dof);
        const Real diff = rhs_with.getVectorEntry(idx) - rhs_without.getVectorEntry(idx);
        EXPECT_NEAR(diff, 0.0, 1e-10)
            << "resistive outlet residual leaked into pressure row dof=" << dof;
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowPressurePerturbationDoesNotChangePressureResidualDelta)
{
    constexpr int outlet_marker = 351;
    constexpr int inlet_marker = 352;
    constexpr int wall_marker = 353;

    auto build_system = [&](bool with_outlet) {
        auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
            outlet_marker, inlet_marker, wall_marker);
        auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
        auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

        Opts opts;
        opts.enable_jit = false;
        opts.enable_convection = true;
        opts.enable_vms = true;
        opts.density = 1.06;
        opts.viscosity = 0.04;
        opts.velocity_field_name = "Velocity";
        opts.pressure_field_name = "Pressure";
        if (with_outlet) {
            opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
        }
        opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = inlet_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
        });
        opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = wall_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
        });

        FESystem sys(mesh);
        svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, opts);
        module.registerOn(sys);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        sys.setup({}, si);
        sys.finalizeAuxiliaryLayout();
        return sys;
    };

    auto with_outlet = build_system(true);
    auto without_outlet = build_system(false);

    const auto n_dofs = static_cast<std::size_t>(with_outlet.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, with_outlet, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, with_outlet, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, with_outlet, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, with_outlet, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, with_outlet, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, with_outlet, "Pressure", 0, Real(0.10), Real(0.007));

    if (!with_outlet.constraints().empty()) {
        with_outlet.constraints().distribute(u_stage);
        with_outlet.constraints().distribute(u_prev);
        with_outlet.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient_with =
        std::make_shared<svmp::FE::systems::TransientSystem>(with_outlet, integrator);
    auto transient_without =
        std::make_shared<svmp::FE::systems::TransientSystem>(without_outlet, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = false;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseVectorView rhs_with(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_with.zero();
    with_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                              /*invalidate_auxiliary_inputs=*/false);
    const auto ar_with = transient_with->assemble(req, state, nullptr, &rhs_with);
    ASSERT_TRUE(ar_with.success);

    svmp::FE::assembly::DenseVectorView rhs_without(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_without.zero();
    without_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                                 /*invalidate_auxiliary_inputs=*/false);
    const auto ar_without = transient_without->assemble(req, state, nullptr, &rhs_without);
    ASSERT_TRUE(ar_without.success);

    const auto pressure_dofs =
        with_outlet.fieldMap().getComponentDofs("Pressure", static_cast<svmp::FE::LocalIndex>(0)).toVector();
    ASSERT_GE(pressure_dofs.size(), 2u);

    svmp::FE::GlobalIndex perturbed_dof = pressure_dofs.back();
    ASSERT_FALSE(with_outlet.constraints().isConstrained(perturbed_dof));

    std::vector<Real> u_pert = u_stage;
    u_pert[static_cast<std::size_t>(perturbed_dof)] += Real(1e-7);
    if (!with_outlet.constraints().empty()) {
        with_outlet.constraints().distribute(u_pert);
    }

    SystemStateView pert_state = state;
    pert_state.u = u_pert;

    svmp::FE::assembly::DenseVectorView rhs_with_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_with_pert.zero();
    with_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                              /*invalidate_auxiliary_inputs=*/false);
    const auto ar_with_pert = transient_with->assemble(req, pert_state, nullptr, &rhs_with_pert);
    ASSERT_TRUE(ar_with_pert.success);

    svmp::FE::assembly::DenseVectorView rhs_without_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_without_pert.zero();
    without_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                                 /*invalidate_auxiliary_inputs=*/false);
    const auto ar_without_pert = transient_without->assemble(req, pert_state, nullptr, &rhs_without_pert);
    ASSERT_TRUE(ar_without_pert.success);

    for (const auto dof : pressure_dofs) {
        if (with_outlet.constraints().isConstrained(dof)) {
            continue;
        }
        const auto idx = static_cast<svmp::FE::GlobalIndex>(dof);
        const Real delta_with = rhs_with_pert.getVectorEntry(idx) - rhs_with.getVectorEntry(idx);
        const Real delta_without = rhs_without_pert.getVectorEntry(idx) - rhs_without.getVectorEntry(idx);
        EXPECT_NEAR(delta_with, delta_without, 1e-10)
            << "pressure perturbation changed pressure residual delta at dof=" << dof;
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowDoesNotChangePressureJacobianBlock)
{
    constexpr int outlet_marker = 361;
    constexpr int inlet_marker = 362;
    constexpr int wall_marker = 363;

    auto build_system = [&](bool with_outlet) {
        auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
            outlet_marker, inlet_marker, wall_marker);
        auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
        auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

        Opts opts;
        opts.enable_jit = false;
        opts.enable_convection = true;
        opts.enable_vms = true;
        opts.density = 1.06;
        opts.viscosity = 0.04;
        opts.velocity_field_name = "Velocity";
        opts.pressure_field_name = "Pressure";
        if (with_outlet) {
            opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
        }
        opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = inlet_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
        });
        opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
            .boundary_marker = wall_marker,
            .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
        });

        FESystem sys(mesh);
        svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
            u_space, p_space, opts);
        module.registerOn(sys);

        SetupInputs si;
        si.topology_override = singleTetraTopology();
        sys.setup({}, si);
        sys.finalizeAuxiliaryLayout();
        return sys;
    };

    auto with_outlet = build_system(true);
    auto without_outlet = build_system(false);

    const auto n_dofs = static_cast<std::size_t>(with_outlet.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, with_outlet, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, with_outlet, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, with_outlet, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, with_outlet, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, with_outlet, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, with_outlet, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, with_outlet, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, with_outlet, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, with_outlet, "Pressure", 0, Real(0.10), Real(0.007));

    if (!with_outlet.constraints().empty()) {
        with_outlet.constraints().distribute(u_stage);
        with_outlet.constraints().distribute(u_prev);
        with_outlet.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient_with =
        std::make_shared<svmp::FE::systems::TransientSystem>(with_outlet, integrator);
    auto transient_without =
        std::make_shared<svmp::FE::systems::TransientSystem>(without_outlet, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out_with(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out_with.zero();
    with_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                              /*invalidate_auxiliary_inputs=*/false);
    const auto ar_with = transient_with->assemble(req, state, &out_with, &out_with);
    ASSERT_TRUE(ar_with.success);

    svmp::FE::assembly::DenseSystemView out_without(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out_without.zero();
    without_outlet.beginTimeStep(/*reset_auxiliary_state=*/false,
                                 /*invalidate_auxiliary_inputs=*/false);
    const auto ar_without = transient_without->assemble(req, state, &out_without, &out_without);
    ASSERT_TRUE(ar_without.success);

    const auto pressure_dofs =
        with_outlet.fieldMap().getComponentDofs("Pressure", static_cast<svmp::FE::LocalIndex>(0)).toVector();
    ASSERT_GE(pressure_dofs.size(), 2u);

    for (const auto row_dof : pressure_dofs) {
        if (with_outlet.constraints().isConstrained(row_dof)) {
            continue;
        }
        for (const auto col_dof : pressure_dofs) {
            if (with_outlet.constraints().isConstrained(col_dof)) {
                continue;
            }
            const auto row = static_cast<svmp::FE::GlobalIndex>(row_dof);
            const auto col = static_cast<svmp::FE::GlobalIndex>(col_dof);
            const Real with_val = out_with.getMatrixEntry(row, col);
            const Real without_val = out_without.getMatrixEntry(row, col);
            EXPECT_NEAR(with_val, without_val, 1e-12)
                << "resistive outlet changed pressure Jacobian block at row=" << row_dof
                << " col=" << col_dof;
        }
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowRankOneUpdateAvoidsPressureDofs)
{
    constexpr int outlet_marker = 371;
    constexpr int inlet_marker = 372;
    constexpr int wall_marker = 373;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = false;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    if (!sys.constraints().empty()) {
        sys.constraints().distribute(u_stage);
        sys.constraints().distribute(u_prev);
        sys.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient =
        std::make_shared<svmp::FE::systems::TransientSystem>(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs.zero();
    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);
    const auto ar = transient->assemble(req, state, &lhs, &rhs);
    ASSERT_TRUE(ar.success);

    const auto pressure_dofs =
        sys.fieldMap().getComponentDofs("Pressure", static_cast<svmp::FE::LocalIndex>(0)).toVector();
    ASSERT_FALSE(pressure_dofs.empty());
    std::unordered_set<svmp::FE::GlobalIndex> pressure_dof_set(
        pressure_dofs.begin(), pressure_dofs.end());

    for (const auto& upd : sys.lastRankOneUpdates()) {
        for (const auto& [dof, value] : upd.v) {
            (void)value;
            EXPECT_FALSE(sys.constraints().isConstrained(dof));
            EXPECT_EQ(pressure_dof_set.count(dof), 0u)
                << "rank-one support leaked onto pressure dof=" << dof;
        }
    }
}

TEST(NavierStokesOutletFactory, LegacyResistiveOutflowMatrixVectorResidualMatchesVectorOnly)
{
    constexpr int outlet_marker = 381;
    constexpr int inlet_marker = 382;
    constexpr int wall_marker = 383;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = false;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.coupled_outflow_rcr.push_back(makeRCROpts(outlet_marker, /*C=*/0.0));
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    if (!sys.constraints().empty()) {
        sys.constraints().distribute(u_stage);
        sys.constraints().distribute(u_prev);
        sys.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient =
        std::make_shared<svmp::FE::systems::TransientSystem>(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    svmp::FE::assembly::DenseVectorView rhs_both(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    lhs.zero();
    rhs_both.zero();
    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);
    const auto ar_both = transient->assemble(req, state, &lhs, &rhs_both);
    ASSERT_TRUE(ar_both.success);

    AssemblyRequest req_vec = req;
    req_vec.want_matrix = false;
    svmp::FE::assembly::DenseVectorView rhs_vec(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    rhs_vec.zero();
    sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                      /*invalidate_auxiliary_inputs=*/false);
    const auto ar_vec = transient->assemble(req_vec, state, nullptr, &rhs_vec);
    ASSERT_TRUE(ar_vec.success);

    for (std::size_t i = 0; i < n_dofs; ++i) {
        const Real both = rhs_both.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        const Real vec = rhs_vec.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        EXPECT_NEAR(both, vec, 1e-12)
            << "matrix+vector residual mismatched vector-only residual at dof=" << i;
    }
}

TEST(NavierStokesOutletFactory, BoundaryMeshDirichletJacobianMatchesFDWithoutResistiveOutlet)
{
    constexpr int outlet_marker = 301;
    constexpr int inlet_marker = 302;
    constexpr int wall_marker = 303;

    auto mesh = std::make_shared<SingleTetraFourBoundaryFaceMeshAccess>(
        outlet_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    Opts opts;
    opts.enable_jit = false;
    opts.enable_convection = true;
    opts.enable_vms = true;
    opts.density = 1.06;
    opts.viscosity = 0.04;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.velocity_dirichlet_weak.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = inlet_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{-1.0}},
    });
    opts.velocity_dirichlet.push_back(Opts::VelocityDirichletBC{
        .boundary_marker = wall_marker,
        .value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}},
    });

    FESystem sys(mesh);
    svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(sys);

    SetupInputs si;
    si.topology_override = singleTetraTopology();
    sys.setup({}, si);
    sys.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> u_stage(n_dofs, 0.0);
    std::vector<Real> u_prev(n_dofs, 0.0);
    std::vector<Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, sys, "Velocity", 0, Real(0.06), Real(0.01));
    assignComponentPattern(u_stage, sys, "Velocity", 1, Real(-0.02), Real(0.005));
    assignComponentPattern(u_stage, sys, "Velocity", 2, Real(-0.90), Real(-0.03));
    assignComponentPattern(u_stage, sys, "Pressure", 0, Real(0.18), Real(0.012));

    assignComponentPattern(u_prev, sys, "Velocity", 0, Real(0.05), Real(-0.007));
    assignComponentPattern(u_prev, sys, "Velocity", 1, Real(-0.01), Real(0.003));
    assignComponentPattern(u_prev, sys, "Velocity", 2, Real(-0.80), Real(0.02));
    assignComponentPattern(u_prev, sys, "Pressure", 0, Real(0.15), Real(-0.009));

    assignComponentPattern(u_prev2, sys, "Velocity", 0, Real(0.03), Real(0.005));
    assignComponentPattern(u_prev2, sys, "Velocity", 1, Real(-0.015), Real(-0.002));
    assignComponentPattern(u_prev2, sys, "Velocity", 2, Real(-0.70), Real(-0.015));
    assignComponentPattern(u_prev2, sys, "Pressure", 0, Real(0.10), Real(0.007));

    if (!sys.constraints().empty()) {
        sys.constraints().distribute(u_stage);
        sys.constraints().distribute(u_prev);
        sys.constraints().distribute(u_prev2);
    }

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    auto transient =
        std::make_shared<svmp::FE::systems::TransientSystem>(sys, integrator);

    AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    const auto ar = transient->assemble(req, state, &out, &out);
    ASSERT_TRUE(ar.success);

    std::vector<Real> base_residual(n_dofs, 0.0);
    std::vector<Real> field_jacobian(n_dofs * n_dofs, 0.0);
    for (std::size_t i = 0; i < n_dofs; ++i) {
        base_residual[i] = out.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n_dofs; ++j) {
            field_jacobian[i * n_dofs + j] =
                out.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                   static_cast<svmp::FE::GlobalIndex>(j));
        }
    }

    std::vector<char> constrained(static_cast<std::size_t>(n_dofs), 0);
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        constrained[dof] = sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(dof)) ? 1 : 0;
    }

    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        if (constrained[col]) {
            continue;
        }
        std::vector<Real> u_pert = u_stage;
        u_pert[col] += eps;
        if (!sys.constraints().empty()) {
            sys.constraints().distribute(u_pert);
        }

        SystemStateView pert_state = state;
        pert_state.u = u_pert;

        svmp::FE::assembly::DenseVectorView rhs_pert(static_cast<svmp::FE::GlobalIndex>(n_dofs));
        rhs_pert.zero();

        AssemblyRequest req_vec = req;
        req_vec.want_matrix = false;
        const auto ar_pert = transient->assemble(req_vec, pert_state, nullptr, &rhs_pert);
        ASSERT_TRUE(ar_pert.success);

        for (std::size_t row = 0; row < n_dofs; ++row) {
            if (constrained[row]) {
                continue;
            }
            const Real fd =
                (rhs_pert.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(row)) - base_residual[row]) / eps;
            const Real analytic = field_jacobian[row * n_dofs + col];
            EXPECT_NEAR(analytic, fd, std::max(1e-5, std::abs(fd) * 1e-4))
                << "boundary mesh Jacobian mismatch without outlet at row=" << row
                << " col=" << col;
        }
    }
}

// The free single-tetra mixed system assembled here leaves the field block
// singular, so this dense bordered reduction comparison is not a stable
// qualification test for the outlet-factory API path. Re-enable once the test
// is reformulated on an anchored mixed system or an explicit constrained
// reference solve.
TEST(NavierStokesOutletFactory, DISABLED_MonolithicRCRCR_BorderedReductionMatchesDenseSolve)
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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
    if (!sys.constraints().empty()) {
        sys.constraints().distribute(sol);
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

TEST(NavierStokesOutletFactory, TwoResistiveOutletsSystemOverloadMatchesLegacyDirectFieldOperator)
{
    constexpr int outlet0_marker = 821;
    constexpr int outlet1_marker = 822;
    constexpr int inlet_marker = 823;
    constexpr int wall_marker = 824;

    auto mesh = std::make_shared<SingleTetraTwoOutletBoundaryFaceMeshAccess>(
        outlet0_marker, outlet1_marker, inlet_marker, wall_marker);
    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, mesh, 1, 3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1, mesh, 1);

    auto build_system = [&](bool use_system_overload) -> FESystem {
        FESystem sys(mesh);
        const auto u_field =
            sys.addField(FieldSpec{.name = "u", .space = u_space, .components = 3});
        const auto p_field =
            sys.addField(FieldSpec{.name = "p", .space = p_space, .components = 1});
        sys.addOperator("ns");

        const auto u = FormExpr::stateField(u_field, *u_space, "u");
        const auto p = FormExpr::stateField(p_field, *p_space, "p");
        const auto rho = FormExpr::constant(1.0);

        BoundaryConditionManager bc_manager;
        auto add_outlet = [&](int marker) {
            auto opts = makeRCROpts(marker, /*C=*/0.0);
            std::unique_ptr<svmp::FE::forms::bc::BoundaryCondition> bc;
            if (use_system_overload) {
                bc = Factories::toCoupledOutflowBC(opts, sys, u, rho);
            } else {
                bc = Factories::toCoupledOutflowBC(opts, u_field, *u_space, "u", u, rho);
            }
            EXPECT_NE(bc, nullptr);
            bc_manager.add(std::move(bc));
        };
        add_outlet(outlet0_marker);
        add_outlet(outlet1_marker);

        Opts::VelocityDirichletBC wall_bc{};
        wall_bc.boundary_marker = wall_marker;
        wall_bc.value = {Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}, Opts::ScalarValue{0.0}};
        auto wall = Factories::toVelocityEssentialBC(wall_bc, /*dim=*/3, "u");
        EXPECT_NE(wall, nullptr);
        bc_manager.add(std::move(wall));

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
        if (use_system_overload) {
            const auto summary = sys.auxiliaryAnalysisSummary();
            EXPECT_EQ(summary.n_monolithic, 2u);
            EXPECT_EQ(summary.n_partitioned, 0u);
        }
        return sys;
    };

    auto monolithic_sys = build_system(true);
    auto legacy_sys = build_system(false);

    const auto n_field = static_cast<std::size_t>(monolithic_sys.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, static_cast<std::size_t>(legacy_sys.dofHandler().getNumDofs()));

    std::vector<Real> sol(n_field, 0.0);
    assignComponentPattern(sol, monolithic_sys, "u", 0, Real(0.04), Real(0.006));
    assignComponentPattern(sol, monolithic_sys, "u", 1, Real(-0.03), Real(0.004));
    assignComponentPattern(sol, monolithic_sys, "u", 2, Real(-0.80), Real(-0.02));
    assignComponentPattern(sol, monolithic_sys, "p", 0, Real(0.12), Real(0.01));
    if (!monolithic_sys.constraints().empty()) {
        monolithic_sys.constraints().distribute(sol);
    }

    SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = sol;

    auto assemble_dense = [&](FESystem& sys,
                              std::vector<Real>& K,
                              std::vector<Real>& r) {
        svmp::FE::assembly::DenseMatrixView lhs(static_cast<svmp::FE::GlobalIndex>(n_field));
        svmp::FE::assembly::DenseVectorView rhs(static_cast<svmp::FE::GlobalIndex>(n_field));
        lhs.zero();
        rhs.zero();

        AssemblyRequest req;
        req.op = "ns";
        req.want_matrix = true;
        req.want_vector = true;
        req.is_nonlinear_iteration = true;

        sys.beginTimeStep(/*reset_auxiliary_state=*/false,
                          /*invalidate_auxiliary_inputs=*/false);
        const auto ar = sys.assemble(req, state, &lhs, &rhs);
        ASSERT_TRUE(ar.success);

        K.assign(n_field * n_field, 0.0);
        r.assign(n_field, 0.0);
        for (std::size_t i = 0; i < n_field; ++i) {
            r[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
            for (std::size_t j = 0; j < n_field; ++j) {
                K[i * n_field + j] =
                    lhs.getMatrixEntry(static_cast<svmp::FE::GlobalIndex>(i),
                                       static_cast<svmp::FE::GlobalIndex>(j));
            }
        }
    };

    auto apply_rank_one_updates = [&](FESystem& sys,
                                      std::vector<Real>& K) {
        const auto updates = sys.lastRankOneUpdates();
        ASSERT_EQ(updates.size(), 2u);
        for (const auto& upd : updates) {
            EXPECT_TRUE(upd.prefer_native_face);
            for (const auto& [row_dof, row_val] : upd.v) {
                const auto row = static_cast<std::size_t>(row_dof);
                for (const auto& [col_dof, col_val] : upd.v) {
                    const auto col = static_cast<std::size_t>(col_dof);
                    K[row * n_field + col] += upd.sigma * row_val * col_val;
                }
            }
        }
    };
    std::vector<Real> K_mono;
    std::vector<Real> r_mono;
    assemble_dense(monolithic_sys, K_mono, r_mono);
    EXPECT_FALSE(monolithic_sys.borderedCoupling().active);
    apply_rank_one_updates(monolithic_sys, K_mono);

    std::vector<Real> K_legacy;
    std::vector<Real> r_legacy;
    assemble_dense(legacy_sys, K_legacy, r_legacy);
    apply_rank_one_updates(legacy_sys, K_legacy);

    for (std::size_t i = 0; i < n_field; ++i) {
        EXPECT_NEAR(r_mono[i], r_legacy[i], std::max(1e-10, std::abs(r_legacy[i]) * 1e-9))
            << "two-outlet resistive residual mismatch at row=" << i;
        for (std::size_t j = 0; j < n_field; ++j) {
            EXPECT_NEAR(K_mono[i * n_field + j],
                        K_legacy[i * n_field + j],
                        std::max(1e-10, std::abs(K_legacy[i * n_field + j]) * 1e-9))
                << "two-outlet resistive Jacobian mismatch at row=" << i
                << " col=" << j;
        }
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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
    if (!sys.constraints().empty()) {
        sys.constraints().distribute(sol);
        sys.constraints().distribute(u_dot_n);
    }

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

// Same limitation as the steady mixed-field probe above, but with
// generalized-alpha history terms layered on top. Keep disabled until the
// constrained/reference Jacobian comparison is rebuilt on a stable anchored
// problem.
TEST(NavierStokesOutletFactory, DISABLED_MonolithicRCRCR_GeneralizedAlphaMixedFieldJacobianMatchesFD)
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

    auto bc = Factories::toCoupledOutflowBC(makeRCRCROpts(marker), sys, u, rho);
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
    for (const auto& upd : sys.lastReducedFieldUpdates()) {
        for (const auto& [row_dof, row_val] : upd.left) {
            const auto row = static_cast<std::size_t>(row_dof);
            for (const auto& [col_dof, col_val] : upd.right) {
                const auto col = static_cast<std::size_t>(col_dof);
                field_jacobian[row * n_dofs + col] += upd.sigma * row_val * col_val;
            }
        }
    }

    const auto packed_base = sys.checkpointAuxiliaryState();
    std::vector<char> constrained(static_cast<std::size_t>(n_dofs), 0);
    for (std::size_t dof = 0; dof < n_dofs; ++dof) {
        constrained[dof] =
            sys.constraints().isConstrained(static_cast<svmp::FE::GlobalIndex>(dof)) ? 1 : 0;
    }

    const Real eps = 1e-7;
    for (std::size_t col = 0; col < n_dofs; ++col) {
        if (constrained[col]) {
            continue;
        }
        std::vector<Real> sol_pert(sol);
        sol_pert[col] += eps;
        if (!sys.constraints().empty()) {
            sys.constraints().distribute(sol_pert);
        }

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
            if (constrained[row]) {
                continue;
            }
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
