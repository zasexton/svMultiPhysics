/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include "Assembly/GlobalSystemView.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#endif

namespace svmp::Physics::formulations::navier_stokes {
void forceLink_NavierStokesRegister();
}

namespace svmp::Physics::test {

namespace {

svmp::Physics::ParameterValue defined(std::string v)
{
    return svmp::Physics::ParameterValue{true, std::move(v)};
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

[[nodiscard]] std::filesystem::path repoRoot()
{
    auto p = std::filesystem::path(__FILE__).parent_path();
    for (int i = 0; i < 6; ++i) {
        p = p.parent_path();
    }
    return p;
}

[[nodiscard]] std::filesystem::path beamMeshPath()
{
    return repoRoot() / "tests" / "cases" / "linear-elasticity" / "beam" / "mesh" / "mesh-complete.mesh.vtu";
}

struct InletOutletMarkers {
    svmp::label_t inlet{101};
    svmp::label_t outlet{102};
    int axis{0};
};

InletOutletMarkers labelBeamInletOutlet(svmp::Mesh& mesh_mut)
{
    const auto bbox = mesh_mut.global_bounding_box();
    const double len[3] = {
        static_cast<double>(bbox.max[0] - bbox.min[0]),
        static_cast<double>(bbox.max[1] - bbox.min[1]),
        static_cast<double>(bbox.max[2] - bbox.min[2]),
    };

    int axis = 0;
    if (len[1] > len[axis]) axis = 1;
    if (len[2] > len[axis]) axis = 2;

    const double minv = static_cast<double>(bbox.min[axis]);
    const double maxv = static_cast<double>(bbox.max[axis]);
    const double tol = 1e-10 * std::max(1.0, std::abs(maxv - minv));

    auto& base = mesh_mut.base();
    const auto& f2c = base.face2cell();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(f2c.size()); ++f) {
        const auto& fc = f2c[static_cast<std::size_t>(f)];
        const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
        const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
        if (c0_valid == c1_valid) {
            continue;
        }

        const auto c = base.face_center(f);
        const double v = static_cast<double>(c[axis]);
        if (std::abs(v - minv) <= tol) {
            base.set_boundary_label(f, static_cast<svmp::label_t>(101));
        } else if (std::abs(v - maxv) <= tol) {
            base.set_boundary_label(f, static_cast<svmp::label_t>(102));
        }
    }

    InletOutletMarkers out{};
    out.inlet = 101;
    out.outlet = 102;
    out.axis = axis;
    return out;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh> loadBeamMesh()
{
    const auto path = beamMeshPath();
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Missing beam test mesh file: " + path.string());
    }

    svmp::MeshIOOptions opts;
    opts.format = "vtu";
    opts.path = path.string();

    return svmp::load_mesh(opts, svmp::MeshComm::world());
}

#endif

void expectParabolicInflowVaries(const svmp::FE::systems::FESystem& system, int component)
{
    const auto& constraints = system.constraints();
    const auto comp_dofs = system.fieldMap().getComponentDofs("Velocity", static_cast<svmp::FE::LocalIndex>(component)).toVector();
    std::vector<double> values;
    values.reserve(comp_dofs.size());

    for (const auto dof : comp_dofs) {
        if (!constraints.isConstrained(dof)) {
            continue;
        }
        const auto c = constraints.getConstraint(dof);
        ASSERT_TRUE(c.has_value());
        if (!c.has_value()) {
            continue;
        }
        ASSERT_TRUE(c->isDirichlet());
        values.push_back(c->inhomogeneity);
    }

    ASSERT_FALSE(values.empty());
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const double minv = *min_it;
    const double maxv = *max_it;
    EXPECT_GT(maxv, 0.0);
    EXPECT_LT(minv, 0.75 * maxv);
}

struct LegacyAssemblySnapshot {
    std::vector<double> matrix;
    std::vector<double> vector;
};

void assignComponentPattern(std::vector<svmp::FE::Real>& values,
                            const svmp::FE::systems::FESystem& system,
                            const std::string& field_name,
                            int component,
                            svmp::FE::Real base,
                            svmp::FE::Real stride)
{
    const auto dofs =
        system.fieldMap().getComponentDofs(field_name, static_cast<svmp::FE::LocalIndex>(component)).toVector();
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        values[static_cast<std::size_t>(dofs[i])] = base + stride * static_cast<svmp::FE::Real>(i);
    }
}

LegacyAssemblySnapshot assembleBeamFluidCase(bool enable_jit)
{
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    if (!mesh) {
        throw std::runtime_error("assembleBeamFluidCase: failed to load beam mesh");
    }
    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();
    input.module_options = enable_jit ? "jit = true" : "jit = false";

    input.default_domain.params["Density"] = defined("1.06");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.04");

    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("Resistance");
        bc.params["Value"] = defined("16000");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    if (!module) {
        throw std::runtime_error("assembleBeamFluidCase: failed to create fluid module");
    }
    system.setup();
    system.finalizeAuxiliaryLayout();

    const auto n_dofs = static_cast<std::size_t>(system.dofHandler().getNumDofs());
    if (n_dofs > 5000u) {
        throw std::runtime_error("assembleBeamFluidCase: beam mesh DOF count unexpectedly large for dense parity test: " +
                                 std::to_string(n_dofs));
    }

    std::vector<svmp::FE::Real> u_stage(n_dofs, 0.0);
    std::vector<svmp::FE::Real> u_prev(n_dofs, 0.0);
    std::vector<svmp::FE::Real> u_prev2(n_dofs, 0.0);

    assignComponentPattern(u_stage, system, "Velocity", 0, 0.05, 0.002);
    assignComponentPattern(u_stage, system, "Velocity", 1, -0.02, 0.001);
    assignComponentPattern(u_stage, system, "Velocity", 2, -0.90, -0.003);
    assignComponentPattern(u_stage, system, "Pressure", 0, 0.20, 0.0025);

    assignComponentPattern(u_prev, system, "Velocity", 0, 0.04, -0.0015);
    assignComponentPattern(u_prev, system, "Velocity", 1, -0.01, 0.0007);
    assignComponentPattern(u_prev, system, "Velocity", 2, -0.80, 0.002);
    assignComponentPattern(u_prev, system, "Pressure", 0, 0.15, -0.0015);

    assignComponentPattern(u_prev2, system, "Velocity", 0, 0.03, 0.0012);
    assignComponentPattern(u_prev2, system, "Velocity", 1, -0.015, -0.0005);
    assignComponentPattern(u_prev2, system, "Velocity", 2, -0.70, -0.0018);
    assignComponentPattern(u_prev2, system, "Pressure", 0, 0.10, 0.001);

    const auto ga = svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    auto integrator = std::make_shared<svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator>(
        svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = ga.alpha_m,
            .alpha_f = ga.alpha_f,
            .gamma = ga.gamma,
            .history_rate_order = 2,
        });

    std::array<std::span<const svmp::FE::Real>, 2> u_hist{u_prev, u_prev2};
    std::array<double, 2> dt_hist{0.05, 0.05};

    svmp::FE::systems::SystemStateView state;
    state.time = ga.alpha_f * 0.05;
    state.dt = 0.05;
    state.effective_dt = ga.alpha_f * state.dt;
    state.dt_prev = state.dt;
    state.u = u_stage;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    svmp::FE::systems::TransientSystem transient(system, integrator);

    svmp::FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    svmp::FE::assembly::DenseSystemView out(static_cast<svmp::FE::GlobalIndex>(n_dofs));
    out.zero();
    const auto ar = transient.assemble(req, state, &out, &out);
    EXPECT_TRUE(ar.success);

    return LegacyAssemblySnapshot{
        .matrix = std::vector<double>(out.matrixData().begin(), out.matrixData().end()),
        .vector = std::vector<double>(out.vectorData().begin(), out.vectorData().end()),
    };
}

void expectAdaptiveNear(const std::vector<double>& actual,
                        const std::vector<double>& expected,
                        double abs_tol,
                        double rel_tol,
                        std::string_view label)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(actual[i] - expected[i]);
        const double ref = std::max(std::abs(actual[i]), std::abs(expected[i]));
        const double tol = abs_tol + rel_tol * ref;
        ASSERT_LE(diff, tol) << label << " mismatch at index " << i
                             << " actual=" << actual[i]
                             << " expected=" << expected[i]
                             << " tol=" << tol;
    }
}

} // namespace

TEST(NavierStokesLegacyBCs, ParabolicFluxInflow_ResistanceOutflow_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    // Ensure the registrar TU is linked so the registry contains the "fluid" factory.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 3);

    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    // Inflow: Dirichlet, parabolic, impose flux.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    // Outflow: Neumann, resistance.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("Resistance");
        bc.params["Value"] = defined("16000");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());
    system.finalizeAuxiliaryLayout();

    const auto* aux_inputs = system.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(aux_inputs, nullptr);
    EXPECT_FALSE(aux_inputs->inputNames().empty());
    EXPECT_EQ(system.auxiliaryAnalysisSummary().n_monolithic, 1u);

    const auto out_slot = system.auxiliaryOutputSlotOf("resistive_outflow_b102", "P_out");
    EXPECT_NE(out_slot, std::string::npos);

    // Parabolic inflow should produce a non-uniform velocity component along the beam axis.
    expectParabolicInflowVaries(system, markers.axis);
#  endif
#endif
}

TEST(NavierStokesLegacyBCs, ParabolicFluxInflow_RCROutflow_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 3);

    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    // Inflow: Dirichlet, parabolic, impose flux.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    // Outflow: Neumann, RCR.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("RCR");
        bc.params["RCR.Capacitance"] = defined("1.5e-5");
        bc.params["RCR.Distal_resistance"] = defined("1212");
        bc.params["RCR.Proximal_resistance"] = defined("121");
        bc.params["RCR.Distal_pressure"] = defined("0");
        bc.params["RCR.Initial_pressure"] = defined("0");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());
    system.finalizeAuxiliaryLayout();

    const auto* aux_inputs = system.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(aux_inputs, nullptr);
    EXPECT_FALSE(aux_inputs->inputNames().empty());
    EXPECT_EQ(system.auxiliaryAnalysisSummary().n_monolithic, 1u);

    const auto out_slot = system.auxiliaryOutputSlotOf("rcr_windkessel_b102", "P_out");
    EXPECT_NE(out_slot, std::string::npos);

    expectParabolicInflowVaries(system, markers.axis);
#  endif
#endif
}

TEST(NavierStokesLegacyBCs, ParabolicFluxInflow_RCRCROutflow_UsesAuxiliaryStatePath)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 3);

    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("RCRCR");
        bc.params["RCRCR.Proximal_resistance"] = defined("121");
        bc.params["RCRCR.Proximal_capacitance"] = defined("5.0e-6");
        bc.params["RCRCR.Intermediate_resistance"] = defined("300");
        bc.params["RCRCR.Distal_capacitance"] = defined("1.0e-5");
        bc.params["RCRCR.Distal_resistance"] = defined("912");
        bc.params["RCRCR.Distal_pressure"] = defined("0");
        bc.params["RCRCR.Initial_pressure_1"] = defined("0");
        bc.params["RCRCR.Initial_pressure_2"] = defined("0");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());
    system.finalizeAuxiliaryLayout();

    const auto* aux_inputs = system.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(aux_inputs, nullptr);
    EXPECT_TRUE(aux_inputs->hasInput("ns_Q_102"));

    const auto out_slot = system.auxiliaryOutputSlotOf("ns_rcrcr_102", "P_out");
    EXPECT_NE(out_slot, std::string::npos);
#  endif
#endif
}

TEST(NavierStokesLegacyBCs, BeamMesh_JitParity_ParabolicInflowResistanceOutflow)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    const auto jit = assembleBeamFluidCase(/*enable_jit=*/true);
    const auto fallback = assembleBeamFluidCase(/*enable_jit=*/false);

    expectAdaptiveNear(jit.matrix, fallback.matrix, 1e-10, 1e-9, "matrix");
    expectAdaptiveNear(jit.vector, fallback.vector, 1e-10, 1e-9, "vector");
#  endif
#endif
}

} // namespace svmp::Physics::test
