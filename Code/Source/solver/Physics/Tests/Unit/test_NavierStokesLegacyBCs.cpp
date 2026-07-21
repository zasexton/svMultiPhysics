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
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Forms/Vocabulary.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
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

[[nodiscard]] std::shared_ptr<svmp::Mesh> buildSingleTetraBoundaryMesh(
    int marker,
    bool label_all_faces = true)
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/3, x_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    base->register_label("free_surface", marker);
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        if (!label_all_faces && f != 0) {
            continue;
        }
        base->set_boundary_label(f, static_cast<svmp::label_t>(marker));
        base->add_to_set(svmp::EntityKind::Face, "free_surface", f);
    }

    return svmp::create_mesh(std::move(base));
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
    std::vector<double> magnitudes;
    magnitudes.reserve(values.size());
    for (const auto value : values) {
        magnitudes.push_back(std::abs(value));
    }
    const auto [min_it, max_it] =
        std::minmax_element(magnitudes.begin(), magnitudes.end());
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
    req.want_matrix = false;
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

bool containsExprType(const svmp::FE::forms::FormExprNode* node,
                      svmp::FE::forms::FormExprType target)
{
    if (!node) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool containsExprType(const svmp::FE::forms::FormExpr& expr,
                      svmp::FE::forms::FormExprType target)
{
    return expr.isValid() && containsExprType(expr.node(), target);
}

bool containsCutVolumeSide(const svmp::FE::forms::FormExprNode* node,
                           svmp::FE::forms::CutVolumeSide target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == svmp::FE::forms::FormExprType::CutVolumeIntegral &&
        node->cutVolumeSide().value_or(svmp::FE::forms::CutVolumeSide::Negative) == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsCutVolumeSide(child, target)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContainCutVolumeSide(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::forms::CutVolumeSide target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsCutVolumeSide(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsCutVolumeSide(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContain(const svmp::FE::systems::FESystem& system,
                               svmp::FE::forms::FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

bool containsInterfaceMarker(const svmp::FE::forms::FormExprNode* node, int marker)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == svmp::FE::forms::FormExprType::InterfaceIntegral) {
        const auto found = node->interfaceMarker();
        if (found.has_value() && *found == marker) {
            return true;
        }
    }
    for (const auto* child : node->children()) {
        if (containsInterfaceMarker(child, marker)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContainInterfaceMarker(
    const svmp::FE::systems::FESystem& system,
    int marker)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsInterfaceMarker(record.residual_expr.get(), marker)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsInterfaceMarker(expr.get(), marker)) {
                return true;
            }
        }
    }
    return false;
}

int stableGeneratedContactLineMarker(svmp::FE::FieldId phi_field,
                                     int interface_marker,
                                     int wall_boundary_marker)
{
    svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi_field);
    key.domain_id = "free_surface";
    key.interface_marker = interface_marker;
    key.boundary_marker = wall_boundary_marker;
    return svmp::FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key);
}

std::size_t interiorFaceKernelCountForBlock(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId test_field,
    svmp::FE::FieldId trial_field)
{
    if (!system.hasOperator("equations")) {
        return 0u;
    }
    std::size_t count = 0u;
    const auto& equations = system.operatorDefinition("equations");
    for (const auto& term : equations.interior) {
        if (term.test_field == test_field && term.trial_field == trial_field) {
            ++count;
        }
    }
    return count;
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

TEST(NavierStokesLegacyBCs, FittedFreeSurfaceBCTranslation_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 77;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::BoundaryIntegral));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::Normal));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(NavierStokesLegacyBCs, FittedFreeSurfaceKinematicBCTranslation_UsesCurrentGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 78;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.equation_params["Enable_ALE"] = defined("true");
    input.equation_params["Mesh_velocity_source"] = defined("prescribed_data");
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.25");
    bc.params["Use_current_geometry_curvature"] = defined("true");
    bc.params["Kinematic_enforcement"] = defined("Nitsche");
    bc.params["Normal_kinematic_policy"] = defined("MatchFluidNormalVelocity");
    bc.params["Tangential_mesh_policy"] = defined("SmoothingOnly");
    bc.params["Kinematic_nitsche_gamma"] = defined("18.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::BoundaryIntegral));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::MeshVelocity));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::CurrentNormal));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::CurrentMeasure));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::CurrentMeanCurvature));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::FacetArea));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(NavierStokesLegacyBCs,
     FittedFreeSurfacePrescribedTangentialMeshPolicyTranslation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 79;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Free_surface_configuration_schema_version"] =
        defined("2");
    input.equation_params["Enable_ALE"] = defined("true");
    input.equation_params["Mesh_velocity_source"] =
        defined("coupled_displacement");
    input.equation_params["Auto_register_mesh_displacement_field"] =
        defined("true");
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["Tangential_mesh_policy"] = defined("Prescribed");
    bc.params["Prescribed_tangential_mesh_velocity"] =
        defined("0.2, -0.1, 0.0");
    bc.params["Tangential_mesh_penalty"] = defined("6.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create(
        "fluid", input, system);
    ASSERT_TRUE(module);
    const auto policies = system.meshTangentialBoundaryPolicies();
    ASSERT_EQ(policies.size(), 1u);
    EXPECT_EQ(
        policies.front().policy,
        svmp::FE::systems::MeshTangentialBoundaryPolicy::Prescribed);
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(
        artifact->json.find(
            "\"prescribed_tangential_mesh_velocity\":["
            "0.20000000000000001,-0.10000000000000001,0]"),
        std::string::npos);
    EXPECT_NE(
        artifact->json.find("\"tangential_mesh_penalty\":6"),
        std::string::npos);
    EXPECT_NE(
        artifact->json.find(
            "\"tangential_mesh_owner\":\"FreeSurfaceBoundary\""),
        std::string::npos);
    ASSERT_NO_THROW(system.setup());

    auto conflicting_input = input;
    conflicting_input.boundary_conditions.front()
        .params["Tangential_mesh_policy"] = defined("Free");
    svmp::FE::systems::FESystem rejected_system(mesh);
    EXPECT_THROW(
        (void)svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", conflicting_input, rejected_system),
        std::runtime_error);
    EXPECT_EQ(rejected_system.findFieldByName("u"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(rejected_system.meshTangentialBoundaryPolicies().empty());
#endif
}

TEST(NavierStokesLegacyBCs, FittedFreeSurfacePrescribedAngleTranslationFailsClosed)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 79;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.equation_params["Enable_ALE"] = defined("true");
    input.equation_params["Mesh_velocity_source"] = defined("coupled_displacement");
    input.equation_params["Auto_register_mesh_displacement_field"] = defined("true");
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["External_pressure"] = defined("0.0");
    bc.params["Contact_line_model"] = defined("PrescribedContactAngle");
    bc.params["Contact_line_wall_marker"] = defined("88");
    bc.params["Contact_line_marker"] = defined("89");
    bc.params["Contact_angle_degrees"] = defined("60.0");
    bc.params["Contact_line_wall_normal"] = defined("1.0, 0.0, 0.0");
    bc.params["Contact_angle_penalty"] = defined("7.5");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_THROW(
        {
            auto module = svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
            (void)module;
        },
        std::invalid_argument);
    EXPECT_FALSE(system.hasOperator("mesh_motion"));
#endif
}

TEST(NavierStokesLegacyBCs, FittedFreeSurfaceContactLineBCTranslation_RejectsBadModel)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["Contact_line_model"] = defined("Rolling");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::runtime_error);
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceContactAliasesFailClosedWhenIncomplete)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    const std::vector<std::pair<std::string, std::string>> incomplete = {
        {"Contact_line_model", "PrescribedContactAngle"},
        {"ContactLineModel", "PrescribedContactAngle"},
        {"Free_surface_contact_line_model", "PrescribedContactAngle"},
        {"FreeSurfaceContactLineModel", "PrescribedContactAngle"},
        {"Contact_angle_radians", "1.0"},
        {"ContactAngleRadians", "1.0"},
        {"Prescribed_contact_angle_radians", "1.0"},
        {"PrescribedContactAngleRadians", "1.0"},
        {"Contact_angle_degrees", "60"},
        {"ContactAngleDegrees", "60"},
        {"Prescribed_contact_angle_degrees", "60"},
        {"PrescribedContactAngleDegrees", "60"},
        {"Contact_line_wall_markers", "80;81"},
        {"ContactLineWallMarkers", "80;81"},
        {"Wall_boundary_markers", "80;81"},
        {"WallBoundaryMarkers", "80;81"},
        {"Contact_line_wall_marker", "80"},
        {"ContactLineWallMarker", "80"},
        {"Wall_boundary_marker", "80"},
        {"WallBoundaryMarker", "80"},
        {"Contact_line_wall_faces", "left;right"},
        {"ContactLineWallFaces", "left;right"},
        {"Wall_boundary_faces", "left;right"},
        {"WallBoundaryFaces", "left;right"},
        {"Contact_line_wall_face", "left"},
        {"ContactLineWallFace", "left"},
        {"Wall_boundary_face", "left"},
        {"WallBoundaryFace", "left"},
        {"Contact_line_marker", "82"},
        {"ContactLineMarker", "82"},
        {"Contact_line_wall_normals", "1 0 0;0 1 0"},
        {"ContactLineWallNormals", "1 0 0;0 1 0"},
        {"Contact_angle_wall_normals", "1 0 0;0 1 0"},
        {"ContactAngleWallNormals", "1 0 0;0 1 0"},
        {"Wall_normals", "1 0 0;0 1 0"},
        {"WallNormals", "1 0 0;0 1 0"},
        {"Contact_line_wall_normal", "1 0 0"},
        {"ContactLineWallNormal", "1 0 0"},
        {"Contact_angle_wall_normal", "1 0 0"},
        {"ContactAngleWallNormal", "1 0 0"},
        {"Wall_normal", "1 0 0"},
        {"WallNormal", "1 0 0"},
        {"Contact_angle_penalty", "2"},
        {"ContactAnglePenalty", "2"},
        {"Contact_line_angle_penalty", "2"},
        {"ContactLineAnglePenalty", "2"},
        {"Contact_line_mobility", "0.5"},
        {"ContactLineMobility", "0.5"},
        {"Mobility", "0.5"},
        {"Wall_slip_model", "Navier"},
        {"WallSlipModel", "Navier"},
        {"Contact_line_wall_slip_model", "Navier"},
        {"ContactLineWallSlipModel", "Navier"},
        {"Wall_slip_length", "0.1"},
        {"WallSlipLength", "0.1"},
        {"Slip_length", "0.1"},
        {"SlipLength", "0.1"},
    };

    for (const auto& [key, value] : incomplete) {
        SCOPED_TRACE(key);
        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.default_domain.params["Density"] = defined("1.0");
        input.default_domain.params["Viscosity.model"] = defined("Constant");
        input.default_domain.params["Viscosity.Value"] = defined("0.01");

        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "free_surface";
        bc.boundary_marker = marker;
        bc.params["Type"] = defined("Free_surface");
        bc.params["Implementation"] = defined("FittedALE");
        bc.params[key] = defined(value);
        input.boundary_conditions.push_back(std::move(bc));

        svmp::FE::systems::FESystem system(mesh);
        EXPECT_THROW(
            {
                auto module =
                    svmp::Physics::EquationModuleRegistry::instance().create(
                        "fluid", input, system);
                (void)module;
            },
            std::runtime_error);
    }
#endif
}

TEST(NavierStokesLegacyBCs,
     FreeSurfaceContactConfigurationRejectsAmbiguityAndCrossModelFields)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    struct FailureCase {
        std::string name;
        std::vector<std::pair<std::string, std::string>> params;
        std::string expected_message;
    };
    const std::vector<FailureCase> failure_cases = {
        {
            "duplicate model aliases",
            {
                {"Contact_line_model", "None"},
                {"ContactLineModel", "None"},
            },
            "multiple aliases for contact-line model",
        },
        {
            "mixed angle units",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_marker", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_radians", "1.0"},
                {"Contact_angle_degrees", "60"},
            },
            "either radians or degrees, not both",
        },
        {
            "none with wall state",
            {
                {"Contact_line_model", "None"},
                {"Contact_line_wall_marker", "80"},
            },
            "Contact_line_model=None does not accept",
        },
        {
            "pinned with angle state",
            {
                {"Contact_line_model", "Pinned"},
                {"Contact_line_marker", "81"},
                {"Contact_angle_radians", "1.0"},
            },
            "Contact_line_model=Pinned accepts only",
        },
        {
            "prescribed with dynamic state",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_marker", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_radians", "1.0"},
                {"Contact_line_mobility", "0.5"},
            },
            "PrescribedAngle does not accept mobility",
        },
        {
            "dynamic with penalty state",
            {
                {"Contact_line_model", "DynamicRenE"},
                {"Contact_line_wall_marker", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_radians", "1.0"},
                {"Contact_line_mobility", "0.5"},
                {"Wall_slip_model", "Navier"},
                {"Wall_slip_length", "0.1"},
                {"Contact_angle_penalty", "2.0"},
            },
            "DynamicRenE does not use Contact_angle_penalty",
        },
        {
            "dynamic without navier slip",
            {
                {"Contact_line_model", "DynamicRenE"},
                {"Contact_line_wall_marker", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_radians", "1.0"},
                {"Contact_line_mobility", "0.5"},
                {"Wall_slip_model", "None"},
                {"Wall_slip_length", "0.1"},
            },
            "DynamicRenE requires Wall_slip_model=Navier",
        },
        {
            "one explicit marker for multiple walls",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80;81"},
                {"Contact_line_marker", "82"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_radians", "1.0"},
            },
            "cannot be shared by multiple contact walls",
        },
        {
            "normal count mismatch",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80;81"},
                {"Contact_line_wall_normals", "1 0 0;0 1 0;0 0 1"},
                {"Contact_angle_radians", "1.0"},
            },
            "one vector or one vector per wall marker",
        },
        {
            "contact key typo",
            {
                {"Contact_line_model", "DynamicRenE"},
                {"Contact_line_moblity", "0.5"},
            },
            "Unknown free-surface contact key",
        },
    };

    for (const auto& failure : failure_cases) {
        SCOPED_TRACE(failure.name);
        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.default_domain.params["Density"] = defined("1.0");
        input.default_domain.params["Viscosity.model"] = defined("Constant");
        input.default_domain.params["Viscosity.Value"] = defined("0.01");

        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "free_surface";
        bc.boundary_marker = marker;
        bc.params["Type"] = defined("Free_surface");
        bc.params["Implementation"] = defined("FittedALE");
        for (const auto& [key, value] : failure.params) {
            bc.params[key] = defined(value);
        }
        input.boundary_conditions.push_back(std::move(bc));

        svmp::FE::systems::FESystem system(mesh);
        try {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
            ADD_FAILURE() << "Expected contact configuration rejection";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find(failure.expected_message),
                      std::string::npos)
                << error.what();
        }
    }
#endif
}

TEST(NavierStokesLegacyBCs,
     FreeSurfaceConfigurationSchemaAliasesAreExplicitAndUnambiguous)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    const auto make_input = [&]() {
        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.default_domain.params["Density"] = defined("1.0");
        input.default_domain.params["Viscosity.model"] = defined("Constant");
        input.default_domain.params["Viscosity.Value"] = defined("0.01");
        return input;
    };

    for (const std::string key : {
             "Free_surface_configuration_schema_version",
             "FreeSurfaceConfigurationSchemaVersion",
             "Free_surface_schema_version"}) {
        SCOPED_TRACE(key);
        auto input = make_input();
        input.equation_params[key] = defined("2");
        svmp::FE::systems::FESystem system(mesh);
        auto module = svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", input, system);
        ASSERT_TRUE(module);
        const auto artifact = module->effectiveConfigurationArtifact();
        ASSERT_TRUE(artifact.has_value());
        EXPECT_NE(artifact->json.find("\"input_version\":2"),
                  std::string::npos);
        EXPECT_NE(artifact->json.find("\"migration_mode\":\"current\""),
                  std::string::npos);
    }

    for (const std::string key : {
             "Enable_explicit_legacy_free_surface_configuration",
             "EnableExplicitLegacyFreeSurfaceConfiguration",
             "Free_surface_legacy_behavior"}) {
        SCOPED_TRACE(key);
        auto input = make_input();
        input.equation_params["Free_surface_configuration_schema_version"] =
            defined("1");
        input.equation_params[key] = defined("true");
        svmp::FE::systems::FESystem system(mesh);
        auto module = svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", input, system);
        ASSERT_TRUE(module);
        const auto artifact = module->effectiveConfigurationArtifact();
        ASSERT_TRUE(artifact.has_value());
        EXPECT_NE(artifact->json.find("\"migration_mode\":\"explicit_legacy\""),
                  std::string::npos);
        EXPECT_NE(artifact->json.find("\"capability_label\":\"legacy_diagnostic\""),
                  std::string::npos);
    }

    {
        auto input = make_input();
        input.equation_params["Free_surface_configuration_schema_version"] =
            defined("2");
        input.equation_params["FreeSurfaceConfigurationSchemaVersion"] =
            defined("2");
        svmp::FE::systems::FESystem system(mesh);
        EXPECT_THROW(
            {
                auto module =
                    svmp::Physics::EquationModuleRegistry::instance().create(
                        "fluid", input, system);
                (void)module;
            },
            std::runtime_error);
        EXPECT_EQ(system.findFieldByName("Velocity"),
                  svmp::FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.findFieldByName("Pressure"),
                  svmp::FE::INVALID_FIELD_ID);
    }
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceContactNoneIsExplicitAndComplete)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["Contact_line_model"] = defined("None");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_NO_THROW({
        auto module =
            svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
        EXPECT_TRUE(module);
    });
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfacePinnedIsExplicitAndComplete)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 80;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Enable_ALE"] = defined("true");
    input.equation_params["Mesh_velocity_source"] =
        defined("coupled_displacement");
    input.equation_params["Auto_register_mesh_displacement_field"] =
        defined("true");
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["Contact_line_model"] = defined("Pinned");
    bc.params["Contact_line_marker"] = defined(std::to_string(marker));
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_NO_THROW({
        auto module =
            svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
        EXPECT_TRUE(module);
    });
#endif
}

TEST(NavierStokesLegacyBCs,
     UnfittedFreeSurfaceContactLinePluralMarkersTranslation_AddsGeneratedMarkers)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int mesh_marker = 81;
    constexpr int interface_marker = 203;
    constexpr int first_wall_marker = 88;
    constexpr int second_wall_marker = 89;
    auto mesh = buildSingleTetraBoundaryMesh(mesh_marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Interface_marker"] = defined(std::to_string(interface_marker));
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Contact_line_model"] = defined("PrescribedContactAngle");
    bc.params["Contact_line_wall_markers"] =
        defined(std::to_string(first_wall_marker) + "; " +
                std::to_string(second_wall_marker));
    bc.params["Contact_line_wall_normals"] =
        defined("1.0 0.0 0.0; 0.0 1.0 0.0");
    bc.params["Contact_angle_degrees"] = defined("90.0");
    bc.params["Contact_angle_penalty"] = defined("4.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    const auto phi = system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.addOperator("level_set_owner");
    const auto phi_state =
        svmp::FE::forms::StateField(phi, *scalar_space, "phi_contact_owner");
    const auto eta =
        svmp::FE::forms::TestField(phi, *scalar_space, "eta_contact_owner");
    (void)svmp::FE::systems::installFormulation(
        system,
        "level_set_owner",
        {phi},
        (phi_state * eta).dx());

    const int first_contact_marker =
        stableGeneratedContactLineMarker(phi, interface_marker, first_wall_marker);
    const int second_contact_marker =
        stableGeneratedContactLineMarker(phi, interface_marker, second_wall_marker);

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, first_contact_marker));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, second_contact_marker));
#endif
}

TEST(NavierStokesLegacyBCs,
     UnfittedDynamicContactAngleTranslationRoutesLineAndSharpWallGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int wall_marker = 82;
    constexpr int interface_marker = 204;
    auto mesh = buildSingleTetraBoundaryMesh(
        wall_marker, /*label_all_faces=*/false);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Operator_tag"] = defined("equations");
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput free_surface{};
    free_surface.name = "free_surface";
    free_surface.boundary_marker = svmp::INVALID_LABEL;
    free_surface.params["Type"] = defined("Free_surface");
    free_surface.params["Implementation"] = defined("UnfittedLevelSet");
    free_surface.params["Interface_marker"] =
        defined(std::to_string(interface_marker));
    free_surface.params["Level_set_field_name"] = defined("phi_dynamic");
    free_surface.params["Active_domain"] = defined("LevelSetNegative");
    free_surface.params["Active_domain_method"] = defined("CutVolume");
    free_surface.params["Generated_interface_geometry"] =
        defined("LinearCorner");
    free_surface.params["Small_cut_aggregation"] = defined("false");
    free_surface.params["Surface_tension"] = defined("0.8");
    free_surface.params["Curvature"] = defined("0.0");
    free_surface.params["Use_level_set_curvature"] = defined("false");
    free_surface.params["Contact_line_model"] =
        defined("DynamicContactAngle");
    free_surface.params["Contact_line_wall_marker"] =
        defined(std::to_string(wall_marker));
    free_surface.params["Contact_line_wall_normal"] =
        defined("0.0 0.0 -1.0");
    free_surface.params["Contact_angle_degrees"] = defined("60.0");
    free_surface.params["Contact_line_mobility"] = defined("0.5");
    free_surface.params["Wall_slip_model"] = defined("Navier");
    free_surface.params["Wall_slip_length"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(free_surface));

    svmp::Physics::BoundaryConditionInput wall{};
    wall.name = "dynamic_contact_wall";
    wall.boundary_marker = wall_marker;
    wall.params["Type"] = defined("Dirichlet");
    wall.params["Value"] = defined("0.0");
    wall.params["Effective_direction"] = defined("0 0 1");
    input.boundary_conditions.push_back(std::move(wall));

    svmp::FE::systems::FESystem system(mesh);
    auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
        svmp::FE::ElementType::Tetra4, 1);
    const auto phi = system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi_dynamic",
        .space = scalar_space,
        .components = 1,
    });
    const int contact_marker = stableGeneratedContactLineMarker(
        phi, interface_marker, wall_marker);
    svmp::FE::interfaces::GeneratedActiveBoundaryMarkerKey active_key{};
    active_key.source =
        svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    active_key.domain_id = "free_surface";
    active_key.interface_marker = interface_marker;
    active_key.boundary_marker = wall_marker;
    active_key.side = svmp::FE::geometry::CutIntegrationSide::Negative;
    const int active_wall_marker =
        svmp::FE::interfaces::stableGeneratedActiveBoundaryMarker(active_key);

    auto module = svmp::Physics::EquationModuleRegistry::instance().create(
        "fluid", input, system);
    ASSERT_TRUE(module);
    const auto velocity = system.findFieldByName("Velocity");
    ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(
        system, contact_marker));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(
        system, active_wall_marker));

    bool found_velocity_phi_coupling = false;
    for (const auto& record : system.formulationRecords()) {
        if (record.operator_tag != "equations") {
            continue;
        }
        found_velocity_phi_coupling =
            found_velocity_phi_coupling ||
            std::find(record.block_couplings.begin(),
                      record.block_couplings.end(),
                      std::pair<svmp::FE::FieldId, svmp::FE::FieldId>{
                          velocity, phi}) != record.block_couplings.end();
    }
    // Refreshed-frozen sharp geometry is tied to the level-set revision by the
    // cut context.  It deliberately does not advertise a direct field tangent
    // in the inner Newton operator; WP-8 must supply that tangent or a
    // converged common-stage outer iteration.
    EXPECT_FALSE(found_velocity_phi_coupling);
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceCutCellStabilizationTranslation_AddsFacetTerms)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 81;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("true");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InterfaceIntegral));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InteriorFaceIntegral));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::Jump));
    ASSERT_FALSE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::ParameterRef));
#endif
}

TEST(NavierStokesLegacyBCs,
     UnfittedFreeSurfaceCutCellPressurePolicyTranslation_DisablesPressureFacetTerms)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 81;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Generated_interface_geometry"] = defined("HighOrderImplicit");
    bc.params["Geometry_tangent_policy"] = defined("RefreshedFrozenQuadrature");
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("true");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    bc.params["Cut_cell_pressure_stabilization_policy"] =
        defined("DisabledForRefreshedFrozenHighOrder");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    auto u_id = system.findFieldByName("u");
    if (u_id == svmp::FE::INVALID_FIELD_ID) {
        u_id = system.findFieldByName("Velocity");
    }
    auto p_id = system.findFieldByName("p");
    if (p_id == svmp::FE::INVALID_FIELD_ID) {
        p_id = system.findFieldByName("Pressure");
    }
    ASSERT_NE(u_id, svmp::FE::INVALID_FIELD_ID);
    ASSERT_NE(p_id, svmp::FE::INVALID_FIELD_ID);
    // Velocity ghost penalty retired: the disabled pressure policy leaves
    // no cut-facet terms in either block.
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, u_id, u_id), 0u);
    EXPECT_EQ(interiorFaceKernelCountForBlock(system, p_id, p_id), 0u);
#endif
}

TEST(NavierStokesLegacyBCs,
     UnfittedFreeSurfaceCutMetadataScaleCapTranslation_AddsBoundedScale)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 81;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("true");
    bc.params["Use_cut_metadata_scale"] = defined("true");
    bc.params["Cut_cell_metadata_scale_cap"] = defined("3.5");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    bc.params["Small_cut_aggregation_maximum_root_path_length"] =
        defined("5");
    bc.params[
        "Small_cut_aggregation_maximum_reference_extrapolation_distance"] =
        defined("2.5");
    bc.params["Small_cut_aggregation_maximum_absolute_coefficient"] =
        defined("7");
    bc.params["Small_cut_aggregation_maximum_row_l1_norm"] = defined("9");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::ParameterRef));
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::Minimum));
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(
        artifact->json.find(
            "\"aggregation_guards\":{\"maximum_root_path_length\":5,"
            "\"maximum_reference_extrapolation_distance\":2.5,"
            "\"maximum_absolute_coefficient\":7,"
            "\"maximum_row_l1_norm\":9}"),
        std::string::npos);

    auto disabled_input = input;
    disabled_input.boundary_conditions.front()
        .params["Small_cut_aggregation"] = defined("false");
    svmp::FE::systems::FESystem rejected_system(mesh);
    rejected_system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind =
            svmp::FE::systems::FieldSourceKind::PrescribedData,
    });
    EXPECT_THROW(
        (void)svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", disabled_input, rejected_system),
        std::runtime_error);
    EXPECT_NE(rejected_system.findFieldByName("phi"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(rejected_system.findFieldByName("u"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(rejected_system.findFieldByName("p"),
              svmp::FE::INVALID_FIELD_ID);
#endif
}

TEST(NavierStokesLegacyBCs,
     RetiredVelocityGhostPenaltySettingsFailClosedInCurrentSchema)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 81;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Element_order"] = defined("2");

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Generated_interface_geometry"] = defined("HighOrderImplicit");
    bc.params["Geometry_tangent_policy"] = defined("RefreshedFrozenQuadrature");
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("true");
    bc.params["Cut_cell_velocity_gradient_penalty"] = defined("1.5");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    bc.params["Cut_cell_pressure_stabilization_policy"] =
        defined("DisabledForRefreshedFrozenHighOrder");
    bc.params["Cut_cell_velocity_max_derivative_order"] = defined("1");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 2);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
        },
        std::runtime_error);
    EXPECT_EQ(system.findFieldByName("Velocity"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("Pressure"),
              svmp::FE::INVALID_FIELD_ID);
#endif
}

TEST(NavierStokesLegacyBCs,
     DisabledCutCellStabilizationRejectsUnusedSuboptions)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 81;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("false");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
        },
        std::runtime_error);
    EXPECT_EQ(system.findFieldByName("Velocity"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("Pressure"),
              svmp::FE::INVALID_FIELD_ID);
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 82;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContainCutVolumeSide(
        system,
        svmp::FE::forms::CutVolumeSide::Negative));
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceTranslation_AcceptsProjectedCurvatureField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 82;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Surface_tension"] = defined("0.0728");
    bc.params["Curvature_field"] = defined("kappa_projected");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "kappa_projected",
        .space = scalar_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContainCutVolumeSide(
        system,
        svmp::FE::forms::CutVolumeSide::Negative));
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_RejectsMissingActiveDomain)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 87;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::invalid_argument);
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_RejectsNoneWithoutOptIn)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 88;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("None");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::invalid_argument);
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_AllowsExplicitFullDomainOptIn)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 89;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Allow_full_domain_unfitted_free_surface"] = defined("true");
    bc.params["External_pressure"] = defined("1.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::InterfaceIntegral));
    EXPECT_FALSE(formulationRecordsContainCutVolumeSide(
        system,
        svmp::FE::forms::CutVolumeSide::Negative));
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceLegacyVelocityExtensionFailsClosed)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 86;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Free_surface_velocity_extension"] = defined("true");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
        },
        std::invalid_argument);
#endif
}

TEST(NavierStokesLegacyBCs,
     DisabledVelocityExtensionRejectsUnusedDiffusivity)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 90;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CutVolume");
    bc.params["Enable_velocity_extension"] = defined("false");
    bc.params["Velocity_extension_diffusivity"] = defined("2.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
        },
        std::runtime_error);
    EXPECT_EQ(system.findFieldByName("Velocity"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("Pressure"),
              svmp::FE::INVALID_FIELD_ID);
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceSmoothedIndicatorTranslation_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 85;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Free_surface_configuration_schema_version"] =
        defined("1");
    input.equation_params[
        "Enable_explicit_legacy_free_surface_configuration"] =
        defined("true");

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("SmoothedIndicator");
    bc.params["Active_domain_smoothing_width"] = defined("0.05");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceActiveDomainTranslation_RejectsBadValue)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 83;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("WaterSide");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::runtime_error);
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceActiveDomainMethodTranslation_RejectsBadValue)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 84;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = svmp::INVALID_LABEL;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("UnfittedLevelSet");
    bc.params["Level_set_field_name"] = defined("phi");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    bc.params["Active_domain_method"] = defined("CellAverage");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto phi_space =
        svmp::FE::spaces::SpaceFactory::create_h1(svmp::FE::ElementType::Tetra4, 1);
    system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
        .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData,
    });

    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::runtime_error);
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceActiveDomainTranslation_RejectsFittedSurface)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 85;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput bc{};
    bc.name = "free_surface";
    bc.boundary_marker = marker;
    bc.params["Type"] = defined("Free_surface");
    bc.params["Implementation"] = defined("FittedALE");
    bc.params["Active_domain"] = defined("LevelSetNegative");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_THROW(
        {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
            (void)module;
        },
        std::invalid_argument);
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
    EXPECT_FALSE(aux_inputs->inputNames().empty());

    const auto out_slot =
        system.auxiliaryOutputSlotOf("rcrcr_windkessel_b102", "P_out");
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
    if (!svmp::FE::forms::jit::llvmJITEnabled()) {
        GTEST_SKIP() << "JIT parity requires FE_ENABLE_LLVM_JIT=ON.";
    }
    const auto jit = assembleBeamFluidCase(/*enable_jit=*/true);
    const auto fallback = assembleBeamFluidCase(/*enable_jit=*/false);

    expectAdaptiveNear(jit.vector, fallback.vector, 1e-10, 1e-9, "vector");
#  endif
#endif
}

} // namespace svmp::Physics::test
