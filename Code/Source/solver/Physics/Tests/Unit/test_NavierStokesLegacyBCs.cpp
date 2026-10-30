/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Options/NavierStokesRegistration.h"
#include "Physics/Options/NavierStokesOptionRules.h"

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
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#if defined(_WIN32)
#  include <process.h>
#else
#  include <unistd.h>
#endif
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Systems/FunctionalRecords.h"

namespace {
// Every small-cut aggregation prolongation report requested in this test
// binary is also built at its closure and compared bit for bit with the
// report built on request.
[[maybe_unused]] const bool small_cut_aggregation_lazy_report_verification =
    [] {
        svmp::FE::constraints::
            setSmallCutAggregationLazyReportVerificationForTesting(true);
        return true;
    }();
} // namespace


#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#if defined(FE_HAS_MPI) && FE_HAS_MPI && defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

namespace svmp::Physics::test {

namespace {

class ScopedPathRemoval {
public:
    explicit ScopedPathRemoval(std::filesystem::path path)
        : path_(std::move(path))
    {
    }

    ~ScopedPathRemoval()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    ScopedPathRemoval(const ScopedPathRemoval&) = delete;
    ScopedPathRemoval& operator=(const ScopedPathRemoval&) = delete;

private:
    std::filesystem::path path_;
};

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
    bool label_all_faces = true,
    svmp::MeshComm comm = svmp::MeshComm::world())
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

    return svmp::create_mesh(std::move(base), comm);
}

// Single tetra whose x = 0 and y = 0 faces carry the given wall markers
// (outward normals -x and -y); the remaining faces carry free_marker.
[[nodiscard]] std::shared_ptr<svmp::Mesh> buildSingleTetraTwoWallMesh(
    int free_marker,
    int x_wall_marker,
    int y_wall_marker)
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
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        bool on_x_wall = true;
        bool on_y_wall = true;
        for (const auto vertex : base->face_vertices(f)) {
            on_x_wall = on_x_wall && x_ref[3u * static_cast<std::size_t>(vertex)] == 0.0;
            on_y_wall = on_y_wall && x_ref[3u * static_cast<std::size_t>(vertex) + 1u] == 0.0;
        }
        const int marker = on_x_wall ? x_wall_marker
                           : on_y_wall ? y_wall_marker
                                       : free_marker;
        base->set_boundary_label(f, static_cast<svmp::label_t>(marker));
    }
    return svmp::create_mesh(std::move(base));
}

svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions
resolveVelocityInlet(const std::shared_ptr<svmp::Mesh>& mesh,
                     const svmp::Physics::BoundaryConditionInput& boundary)
{
    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra_inlet";
    input.mesh = mesh->local_mesh_ptr();
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");
    input.boundary_conditions.push_back(boundary);
    svmp::FE::systems::FESystem system(mesh);
    auto resolved = svmp::Physics::options::resolveNavierStokesInput(input, system);
    if (!resolved.one_phase.has_value()) {
        throw std::runtime_error("Expected one-phase inlet options");
    }
    return std::move(resolved.one_phase->options);
}

svmp::Physics::BoundaryConditionInput velocityInlet(int marker)
{
    svmp::Physics::BoundaryConditionInput boundary{};
    boundary.name = "inlet";
    boundary.boundary_marker = marker;
    boundary.params["Type"] = defined("Dir");
    boundary.params["Time_dependence"] = defined("Steady");
    boundary.params["Value"] = defined("3.1");
    boundary.params["Impose_flux"] = defined("true");
    boundary.params["Profile"] = defined("Flat");
    return boundary;
}

std::filesystem::path inletFlowPath()
{
    static std::atomic<unsigned long long> sequence{0u};
#if defined(_WIN32)
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    int rank = 0;
#if defined(FE_HAS_MPI) && FE_HAS_MPI && defined(MESH_HAS_MPI)
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
#endif
    return std::filesystem::temp_directory_path() /
           ("svmp_velocity_inlet_" + std::to_string(pid) + "_rank_" +
            std::to_string(rank) + "_" + std::to_string(sequence.fetch_add(1u)) + ".flow");
}

void writeTemporalSpatialValuesFile(
    const std::filesystem::path& path,
    const std::vector<long long>& node_ids,
    std::string_view time_value = "0",
    std::string_view trailing_data = {})
{
    std::ofstream values(path);
    ASSERT_TRUE(values.is_open());
    values << "3 1 " << node_ids.size() << "\n"
           << time_value << '\n';
    for (std::size_t i = 0; i < node_ids.size(); ++i) {
        const auto base = 10 * static_cast<int>(i + 1);
        values << node_ids[i] << ' '
               << base << ' ' << base + 1 << ' ' << base + 2 << '\n';
    }
    if (!trailing_data.empty()) {
        values << trailing_data << '\n';
    }
    ASSERT_TRUE(values.good());
}

[[nodiscard]] svmp::Physics::EquationModuleInput
makeGeneralVelocityInput(
    const std::shared_ptr<svmp::Mesh>& mesh,
    int marker,
    const std::filesystem::path& values_path)
{
    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::Physics::BoundaryConditionInput boundary{};
    boundary.name = "wall";
    boundary.boundary_marker = marker;
    boundary.params["Type"] = defined("Dir");
    boundary.params["Time_dependence"] = defined("General");
    boundary.params["Weakly_applied"] = defined("false");
    boundary.params["Temporal_and_spatial_values_file_path"] =
        defined(values_path.string());
    input.boundary_conditions.push_back(std::move(boundary));
    return input;
}

void expectGeneralVelocityFileConstraints(
    std::vector<svmp::gid_t> vertex_gids,
    std::string_view path_suffix,
    svmp::MeshComm comm = svmp::MeshComm::world())
{
    constexpr int marker = 76;
    auto mesh = buildSingleTetraBoundaryMesh(marker, true, comm);
    ASSERT_TRUE(mesh);
    mesh->base().set_vertex_gids(std::move(vertex_gids));

    const auto values_path =
        std::filesystem::temp_directory_path() /
        ("svmp_general_velocity_" + std::string(path_suffix) + ".dat");
    const ScopedPathRemoval cleanup(values_path);
    writeTemporalSpatialValuesFile(values_path, {1, 2, 3, 4});
    auto input = makeGeneralVelocityInput(mesh, marker, values_path);

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create(
        "fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    const std::array<std::array<double, 4>, 3> expected{{
        {{10.0, 20.0, 30.0, 40.0}},
        {{11.0, 21.0, 31.0, 41.0}},
        {{12.0, 22.0, 32.0, 42.0}},
    }};
    for (int component = 0; component < 3; ++component) {
        const auto dofs = system.fieldMap()
                              .getComponentDofs(
                                  "Velocity",
                                  static_cast<svmp::FE::LocalIndex>(component))
                              .toVector();
        std::vector<double> actual;
        for (const auto dof : dofs) {
            const auto constraint = system.constraints().getConstraint(dof);
            ASSERT_TRUE(constraint.has_value());
            ASSERT_TRUE(constraint->isDirichlet());
            actual.push_back(constraint->inhomogeneity);
        }
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual,
                  std::vector<double>(expected[static_cast<std::size_t>(
                      component)].begin(),
                                      expected[static_cast<std::size_t>(
                                          component)].end()));
    }
}

void expectGeneralVelocityFileRejected(
    std::vector<long long> file_node_ids,
    std::string_view expected_message,
    std::string_view path_suffix,
    std::string_view time_value = "0",
    std::string_view trailing_data = {})
{
    constexpr int marker = 76;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);
    mesh->base().set_vertex_gids({1, 2, 3, 4});

    const auto values_path =
        std::filesystem::temp_directory_path() /
        ("svmp_general_velocity_" + std::string(path_suffix) + ".dat");
    const ScopedPathRemoval cleanup(values_path);
    writeTemporalSpatialValuesFile(
        values_path, file_node_ids, time_value, trailing_data);
    auto input = makeGeneralVelocityInput(mesh, marker, values_path);

    svmp::FE::systems::FESystem system(mesh);
    try {
        auto module = svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", input, system);
        (void)module;
        FAIL() << "Expected temporal/spatial node-ID mapping to fail closed.";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected_message),
                  std::string::npos)
            << error.what();
    }
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

} // namespace

TEST(NavierStokesLegacyBCs, ScalarLiteralValidationPreservesDiagnosticsAndVariantPolicy)
{
    namespace ns = svmp::Physics::formulations::navier_stokes;
    using Real = svmp::FE::Real;
    using ScalarValue = ns::IncompressibleNavierStokesVMSOptions::ScalarValue;
    const std::array validators{
        ns::validateFiniteConstantScalar,
        ns::validateNonnegativeConstantScalar,
        ns::validatePositiveConstantScalar};
    struct Case {
        Real value;
        std::array<const char*, 3> diagnostic;
    };
    const auto infinity = std::numeric_limits<Real>::infinity();
    const std::array cases{
        Case{-infinity, {"finite", "finite", "finite"}},
        Case{infinity, {"finite", "finite", "finite"}},
        Case{std::numeric_limits<Real>::quiet_NaN(), {"finite", "finite", "finite"}},
        Case{-1.0, {nullptr, "nonnegative", "positive"}},
        Case{-0.0, {nullptr, nullptr, "positive"}},
        Case{0.0, {nullptr, nullptr, "positive"}},
        Case{1.0, {nullptr, nullptr, nullptr}}};
    for (const auto& test : cases) {
        for (std::size_t i = 0; i < validators.size(); ++i) {
            SCOPED_TRACE(test.value);
            SCOPED_TRACE(i);
            if (test.diagnostic[i] == nullptr) {
                EXPECT_NO_THROW(validators[i](ScalarValue{test.value}, "test parameter"));
                continue;
            }
            try {
                validators[i](ScalarValue{test.value}, "test parameter");
                FAIL() << "Expected the scalar literal to be rejected";
            } catch (const std::invalid_argument& error) {
                EXPECT_EQ(std::string(error.what()),
                          "IncompressibleNavierStokesVMSModule: test parameter must be " +
                              std::string(test.diagnostic[i]));
            }
        }
    }
    int callback_calls = 0;
    const std::array<ScalarValue, 6> deferred{
        svmp::FE::forms::ScalarCoefficient{[&](Real, Real, Real) {
            ++callback_calls;
            return infinity;
        }},
        svmp::FE::forms::TimeScalarCoefficient{[&](Real, Real, Real, Real) {
            ++callback_calls;
            return infinity;
        }},
        svmp::FE::forms::ScalarCoefficient{},
        svmp::FE::forms::TimeScalarCoefficient{},
        svmp::FE::forms::FormExpr::constant(infinity),
        svmp::FE::forms::FormExpr{}};
    for (const auto& value : deferred) {
        for (const auto validate : validators) {
            EXPECT_NO_THROW(validate(value, "deferred parameter"));
        }
    }
    EXPECT_EQ(callback_calls, 0);
}

TEST(NavierStokesLegacyBCs,
     FreeSurfacePhysicalModelDirectInputIsExplicitAndFailClosed)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::
        forceLink_NavierStokesRegister();

    constexpr int marker = 76;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    const auto make_input = [&]() {
        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.default_domain.params["Density"] = defined("1.0");
        input.default_domain.params["Viscosity.model"] =
            defined("Constant");
        input.default_domain.params["Viscosity.Value"] = defined("0.01");
        return input;
    };

    const auto expect_supported = [&](std::string model_key) {
        auto input = make_input();
        input.mesh_name = "gas_two_phase_mesh";
        if (!model_key.empty()) {
            input.equation_params[model_key] =
                defined("OnePhaseLiquidPrescribedExteriorPressure");
        }
        svmp::FE::systems::FESystem system(mesh);
        auto module =
            svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
        ASSERT_TRUE(module) << model_key;
        const auto artifact = module->effectiveConfigurationArtifact();
        ASSERT_TRUE(artifact.has_value()) << model_key;
        EXPECT_NE(
            artifact->json.find(
                "\"operator\":\"equations\""),
            std::string::npos)
            << model_key;
        EXPECT_NE(
            artifact->json.find(
                "\"name\":"
                "\"one_phase_liquid_prescribed_exterior_pressure\""),
            std::string::npos)
            << model_key;
        EXPECT_NE(
            artifact->json.find(
                "\"exterior_pressure_mode\":"
                "\"prescribed_scalar_traction_reference\""),
            std::string::npos)
            << model_key;
        EXPECT_NE(
            artifact->json.find(
                "\"exterior_momentum_solved\":false"),
            std::string::npos)
            << model_key;
        EXPECT_NE(
            artifact->json.find(
                "\"exterior_pressure_field_solved\":false"),
            std::string::npos)
            << model_key;
    };

    expect_supported("");
    expect_supported("Free_surface_physical_model");

    struct Rejection {
        const char* label;
        std::function<void(svmp::Physics::EquationModuleInput&)> mutate;
        const char* diagnostic;
    };
    const std::vector<Rejection> rejections = {
        {
            "undefined gas key",
            [](auto& input) {
                input.equation_params["Gas_density"] =
                    svmp::Physics::ParameterValue{false, ""};
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "generic selector value",
            [](auto& input) {
                input.equation_params["Model"] =
                    defined("two_fluid");
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "unmarked generic selector value",
            [](auto& input) {
                input.equation_params["Model"] =
                    defined("AirLiquid");
            },
            "unsupported_free_surface_physical_model",
        },
        {
            "default-domain key",
            [](auto& input) {
                input.default_domain.params["Pressure_enrichment"] =
                    defined("xfem");
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "default-domain unmarked selector",
            [](auto& input) {
                input.default_domain.params["Model"] =
                    defined("AirLiquid");
            },
            "unsupported_free_surface_physical_model",
        },
        {
            "explicit domain key",
            [](auto& input) {
                svmp::Physics::DomainInput domain{};
                domain.id = "liquid";
                domain.params["Gas_viscosity"] = defined("1.8e-5");
                input.domains.push_back(std::move(domain));
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "boundary selector value",
            [](auto& input) {
                svmp::Physics::BoundaryConditionInput boundary{};
                boundary.name = "free_surface";
                boundary.params["Interface_model"] =
                    defined("pressure_jump");
                input.boundary_conditions.push_back(std::move(boundary));
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "implementation on non-free-surface boundary",
            [](auto& input) {
                svmp::Physics::BoundaryConditionInput boundary{};
                boundary.name = "wall";
                boundary.params["Type"] = defined("Dir");
                boundary.params["FreeSurfaceImplementation"] =
                    defined("Hysing");
                input.boundary_conditions.push_back(
                    std::move(boundary));
            },
            "unsupported_free_surface_physical_model",
        },
        {
            "noncanonical free-surface implementation key",
            [](auto& input) {
                svmp::Physics::BoundaryConditionInput boundary{};
                boundary.name = "free_surface";
                boundary.params["Type"] = defined("Free_surface");
                boundary.params["implementation"] =
                    defined("Hysing");
                input.boundary_conditions.push_back(
                    std::move(boundary));
            },
            "unsupported_free_surface_physical_model",
        },
        {
            "module-options selector value",
            [](auto& input) {
                input.module_options = "model = two_phase";
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "module-options physical selector",
            [](auto& input) {
                input.module_options =
                    "Free_surface_physical_model:"
                    "OnePhaseLiquidPrescribedExteriorPressure";
            },
            "misplaced_free_surface_physical_model",
        },
        {
            "misplaced physical selector",
            [](auto& input) {
                input.default_domain
                    .params["Free_surface_physical_model"] =
                    defined(
                        "OnePhaseLiquidPrescribedExteriorPressure");
            },
            "misplaced_free_surface_physical_model",
        },
        {
            "noncanonical physical selector spelling",
            [](auto& input) {
                input.equation_params[
                    "Free-surface physical model"] =
                    defined(
                        "OnePhaseLiquidPrescribedExteriorPressure");
            },
            "unsupported_free_surface_physical_model",
        },
        {
            "unknown physical model",
            [](auto& input) {
                input.equation_params[
                    "Free_surface_physical_model"] =
                    defined("one_phase_liquid_sharp_interface");
            },
            "unsupported_free_surface_physical_model",
        },
    };

    for (const auto& rejection : rejections) {
        SCOPED_TRACE(rejection.label);
        auto input = make_input();
        rejection.mutate(input);
        svmp::FE::systems::FESystem system(mesh);
        try {
            (void)svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
            FAIL() << "unsupported physical input must fail closed";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(
                std::string(error.what()).find(rejection.diagnostic),
                std::string::npos);
        }
        EXPECT_EQ(system.findFieldByName("Velocity"),
                  svmp::FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.findFieldByName("Pressure"),
                  svmp::FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.fieldMap().numFields(), 0u);
        EXPECT_TRUE(system.formulationRecords().empty());
        EXPECT_FALSE(system.hasOperator("equations"));
    }

    {
        auto input = make_input();
        svmp::Physics::BoundaryConditionInput boundary{};
        boundary.name = "free_surface";
        boundary.params["Type"] = defined("Free_surface");
        boundary.params["Implementation"] =
            defined("UnfittedLevelSet");
        input.boundary_conditions.push_back(std::move(boundary));
        svmp::FE::systems::FESystem system(mesh);
        EXPECT_NE(
            svmp::Physics::formulations::navier_stokes::
                preRegisterPrimaryVelocityField(input, system),
            svmp::FE::INVALID_FIELD_ID);
        EXPECT_NE(system.findFieldByName("Velocity"),
                  svmp::FE::INVALID_FIELD_ID);
    }

    const std::vector<Rejection> preregistration_rejections = {
        {
            "pre-registration scope marker",
            [](auto& input) {
                input.equation_params["Formulation"] =
                    defined("incompressible_two_fluid");
            },
            "unsupported_two_phase_or_jump_free_surface_scope",
        },
        {
            "pre-registration boundary context",
            [](auto& input) {
                svmp::Physics::BoundaryConditionInput boundary{};
                boundary.name = "wall";
                boundary.params["Type"] = defined("Dir");
                boundary.params["Free_surface_implementation"] =
                    defined("Hysing");
                input.boundary_conditions.push_back(
                    std::move(boundary));
            },
            "unsupported_free_surface_physical_model",
        },
    };
    for (const auto& rejection : preregistration_rejections) {
        SCOPED_TRACE(rejection.label);
        auto input = make_input();
        rejection.mutate(input);
        svmp::FE::systems::FESystem system(mesh);
        try {
            (void)svmp::Physics::formulations::navier_stokes::
                preRegisterPrimaryVelocityField(input, system);
            FAIL()
                << "pre-registration must reject unsupported physical input";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(
                std::string(error.what()).find(rejection.diagnostic),
                std::string::npos);
        }
        EXPECT_EQ(system.findFieldByName("Velocity"),
                  svmp::FE::INVALID_FIELD_ID);
        EXPECT_EQ(system.fieldMap().numFields(), 0u);
        EXPECT_TRUE(system.formulationRecords().empty());
    }
#endif
}

TEST(NavierStokesLegacyBCs, VelocityInletZeroSkipsMissingMarkerGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildSingleTetraTwoWallMesh(76, 77, 78);
    auto boundary = velocityInlet(999); // Valid label value, with no faces.
    boundary.params["Value"] = defined("0");
    boundary.params["Profile"] = defined("Parabolic");
    boundary.params["Effective_direction"] = defined("0 1 0");
    for (const bool weak : {false, true}) {
        boundary.params["Weakly_applied"] = defined(weak ? "true" : "false");
        const auto options = resolveVelocityInlet(mesh, boundary);
        const auto& values = weak ? options.velocity_dirichlet_weak : options.velocity_dirichlet;
        ASSERT_EQ(values.size(), 1u);
        EXPECT_EQ(values.front().active_components, (std::array<bool, 3>{false, true, false}));
        for (const auto& value : values.front().value) {
            ASSERT_TRUE(std::holds_alternative<svmp::FE::Real>(value));
            EXPECT_EQ(std::get<svmp::FE::Real>(value), 0.0);
        }
    }
#endif
}

TEST(NavierStokesLegacyBCs, VelocityInletProfileErrorsPrecedeZeroShortcutAndFollowFlowFile)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildSingleTetraTwoWallMesh(76, 77, 78);
    auto boundary = velocityInlet(999);
    boundary.params["Value"] = defined("0");
    boundary.params["Profile"] = defined("Unknown");
    const auto expect_message = [&](std::string_view expected) {
        try {
            static_cast<void>(resolveVelocityInlet(mesh, boundary));
            FAIL() << "Expected inlet input rejection";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()), expected);
        }
    };
    expect_message(
        "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='Unknown' for the new solver "
        "Navier-Stokes module. Supported profiles: Flat, Parabolic.");
    boundary.params["Profile"] = defined("User_defined");
    expect_message(
        "[svMultiPhysics::Physics] Dirichlet BC Profile='User_defined' is not supported for the new solver "
        "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    boundary.params["Time_dependence"] = defined("Unsteady");
    expect_message(
        "[svMultiPhysics::Physics] Unsteady Dirichlet BC 'inlet' is missing <Temporal_values_file_path>.");

    const auto path = inletFlowPath();
    const ScopedPathRemoval cleanup(path);
    {
        std::ofstream flow(path);
        ASSERT_TRUE(flow.is_open());
        flow << "2 0\n2 3.1\n6 3.1\n";
    }
    boundary.params["Temporal_values_file_path"] = defined(path.string());
    expect_message(
        "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='User_defined' for Unsteady BC 'inlet'. "
        "Supported profiles: Flat, Parabolic.");
#endif
}

TEST(NavierStokesLegacyBCs, VelocityInletFlatNormalizationAndDirectionMasks)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Marker 77 is x=0, with area 1/2 and outward normal (-1,0,0).
    auto mesh = buildSingleTetraTwoWallMesh(76, 77, 78);
    auto boundary = velocityInlet(77);
    const auto path = inletFlowPath();
    const ScopedPathRemoval cleanup(path);
    {
        std::ofstream flow(path);
        ASSERT_TRUE(flow.is_open());
        flow << "2 0\n2 3.1\n6 3.1\n";
    }
    for (const bool weak : {false, true}) {
        boundary.params["Weakly_applied"] = defined(weak ? "true" : "false");
        for (const bool flux : {false, true}) {
            boundary.params["Impose_flux"] = defined(flux ? "true" : "false");
            for (const auto* direction : {"", "0 -2 0", "1 1 1", "0 0 0"}) {
                boundary.params["Effective_direction"] = defined(direction);
                for (const bool unsteady : {false, true}) {
                    auto timed_boundary = boundary;
                    if (unsteady) {
                        timed_boundary.params["Time_dependence"] = defined("Unsteady");
                        timed_boundary.params["Temporal_values_file_path"] = defined(path.string());
                        timed_boundary.params["Value"] = defined("0"); // The file supplies the amplitude.
                    }
                    const auto options = resolveVelocityInlet(mesh, timed_boundary);
                    const auto& values = weak ? options.velocity_dirichlet_weak : options.velocity_dirichlet;
                    ASSERT_EQ(values.size(), 1u);
                    const auto& inlet = values.front();
                    const bool masked = std::string_view(direction) == "0 -2 0";
                    EXPECT_EQ(inlet.active_components,
                              (masked ? std::array<bool, 3>{false, true, false}
                                      : std::array<bool, 3>{true, true, true}));
                    const double amplitude = flux ? 6.2 : 3.1;
                    const std::array<double, 3> expected =
                        masked ? std::array<double, 3>{0.0, amplitude, 0.0}
                               : std::array<double, 3>{-amplitude, 0.0, 0.0};
                    for (std::size_t component = 0; component < 3; ++component) {
                        const auto& value = std::get<svmp::FE::forms::TimeScalarCoefficient>(inlet.value[component]);
                        EXPECT_EQ(value(0.0, 0.2, 0.3, 7.0), expected[component]);
                    }
                }
            }
        }
    }
#endif
}

TEST(NavierStokesLegacyBCs, VelocityInletParabolicScalingAndPeriodicTimeOrigin)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // At the three degree-2 triangle points the radial weight is 3/4,
    // giving normalization (1/2)*(3/4)=3/8.  The centre has weight one.
    auto mesh = buildSingleTetraTwoWallMesh(76, 77, 78);
    auto boundary = velocityInlet(77);
    boundary.params["Profile"] = defined("Parabolic");
    const auto steady = resolveVelocityInlet(mesh, boundary);
    ASSERT_EQ(steady.velocity_dirichlet.size(), 1u);
    const auto& steady_value = std::get<svmp::FE::forms::TimeScalarCoefficient>(
        steady.velocity_dirichlet.front().value[0]);
    // Preserve division in the steady path: 3.1/(3/8).
    EXPECT_EQ(steady_value(0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0), -0x1.0888888888889p+3);
    EXPECT_EQ(steady_value(0.0, 0.0, 0.0, 0.0), 0.0);

    const auto path = inletFlowPath();
    const ScopedPathRemoval cleanup(path);
    {
        std::ofstream flow(path);
        ASSERT_TRUE(flow.is_open());
        flow << "3 0\n2 3.1\n4 7.1\n6 3.1\n";
    }
    boundary.params["Time_dependence"] = defined("Unsteady");
    boundary.params["Temporal_values_file_path"] = defined(path.string());
    const auto unsteady = resolveVelocityInlet(mesh, boundary);
    ASSERT_EQ(unsteady.velocity_dirichlet.size(), 1u);
    const auto& unsteady_value = std::get<svmp::FE::forms::TimeScalarCoefficient>(
        unsteady.velocity_dirichlet.front().value[0]);
    // The temporal amplitude multiplies the already normalized profile;
    // for 3.1 this differs by one bit from the steady division above.
    for (const auto [time, expected] : std::array<std::pair<double, double>, 7>{{
             {2.0, -0x1.0888888888888p+3},
             {3.0, -0x1.b333333333332p+3},
             {4.0, -0x1.2eeeeeeeeeeeep+4},
             {6.0, -0x1.0888888888888p+3},
             {7.0, -0x1.b333333333332p+3},
             {0.0, -0x1.2eeeeeeeeeeeep+4},
             {-2.0, -0x1.0888888888888p+3}}}) {
        EXPECT_EQ(unsteady_value(0.0, 1.0 / 3.0, 1.0 / 3.0, time), expected) << time;
    }
    EXPECT_EQ(unsteady_value(0.0, 0.0, 0.0, 3.0), 0.0);
#endif
}

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

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileUsesImportedOneBasedGlobalNodeIds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileConstraints(
        {1, 2, 3, 4}, "imported_one_based_global_node_ids");
#endif
}

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileSupportsLegacyOneBasedOrdinalNodeIds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileConstraints(
        {0, 1, 2, 3}, "legacy_one_based_ordinal_node_ids");
#endif
}

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileRejectsMixedNodeIdConventions)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileRejected(
        {1, 2, 3, 5},
        "do not exactly cover boundary marker",
        "mixed_node_id_conventions");
#endif
}

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileRejectsDuplicateNodeIds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileRejected(
        {1, 2, 2, 4},
        "contains duplicate node id 2",
        "duplicate_node_ids");
#endif
}

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileRejectsNonfiniteTimeData)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileRejected(
        {1, 2, 3, 4},
        "contains a missing or nonfinite time value",
        "nonfinite_time_data",
        "nan");
#endif
}

TEST(NavierStokesLegacyBCs,
     GeneralVelocityFileRejectsTrailingData)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    expectGeneralVelocityFileRejected(
        {1, 2, 3, 4},
        "contains trailing data",
        "trailing_data",
        "0",
        "unexpected_token");
#endif
}

TEST(NavierStokesLegacyBCs,
     SpacetimeMomentumFileUsesAllMeshVertices)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 76;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);
    mesh->base().set_vertex_gids({1, 2, 3, 4});

    const auto values_path =
        std::filesystem::temp_directory_path() /
        "svmp_spacetime_momentum_all_mesh_vertices.dat";
    const ScopedPathRemoval cleanup(values_path);
    writeTemporalSpatialValuesFile(values_path, {1, 2, 3, 4});

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "tetra";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params[
        "Momentum_source_temporal_and_spatial_values_file_path"] =
        defined(values_path.string());
    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    svmp::FE::systems::FESystem system(mesh);
    ASSERT_NO_THROW({
        auto module =
            svmp::Physics::EquationModuleRegistry::instance().create(
                "fluid", input, system);
        ASSERT_TRUE(module);
    });
#endif
}

TEST(NavierStokesLegacyBCsMPI,
     GeneralVelocityFileMappingIsolatedToActiveCommunicator)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && FE_HAS_MPI && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Requires MPI-enabled FE and Mesh integration.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    const auto path_suffix =
        "split_communicator_rank_" + std::to_string(world_rank);
    expectGeneralVelocityFileConstraints(
        world_rank == 0 ? std::vector<svmp::gid_t>{1, 2, 3, 4}
                        : std::vector<svmp::gid_t>{0, 1, 2, 3},
        path_suffix,
        svmp::MeshComm(MPI_COMM_SELF));
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
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.0");
    bc.params["Tangential_mesh_policy"] = defined("Prescribed");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::BoundaryIntegral));
    ASSERT_TRUE(formulationRecordsContain(
        system, svmp::FE::forms::FormExprType::CurrentNormal));
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
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.25");
    bc.params["Tangential_mesh_policy"] = defined("Prescribed");
    bc.params["Kinematic_nitsche_gamma"] = defined("18.0");
    input.boundary_conditions.push_back(std::move(bc));

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::BoundaryIntegral));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::MeshVelocity));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::CurrentNormal));
    // Fitted boundary terms integrate over the assembly frame; the current
    // surface measure is not applied a second time.
    ASSERT_FALSE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::CurrentMeasure));
    ASSERT_TRUE(formulationRecordsContain(system, svmp::FE::forms::FormExprType::FacetArea));
    // SurfaceStress is the only capillary form (decision D47 Q2(b)); the
    // nonzero surface tension needs no opt-in.
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(artifact->json.find("\"surface_tension_form_effective\":\"SurfaceStress\""),
              std::string::npos)
        << artifact->json;
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(NavierStokesLegacyBCs, FittedFreeSurfaceKinematicEnforcementDefaultsToMeshNitsche)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Decision D35: a fitted ALE free surface of the qualified contract
    // always uses the MeshNitsche normal enforcement; Kinematic_nitsche_gamma
    // sets its penalty.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 79;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    const auto make_input =
        [&](const std::vector<std::pair<std::string, std::string>>& extra) {
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
            input.default_domain.params["Viscosity.model"] =
                defined("Constant");
            input.default_domain.params["Viscosity.Value"] = defined("0.01");

            svmp::Physics::BoundaryConditionInput bc{};
            bc.name = "free_surface";
            bc.boundary_marker = marker;
            bc.params["Type"] = defined("Free_surface");
            bc.params["Implementation"] = defined("FittedALE");
            bc.params["External_pressure"] = defined("12.5");
            bc.params["Surface_tension"] = defined("0.0");
            bc.params["Tangential_mesh_policy"] = defined("Free");
            for (const auto& [key, value] : extra) {
                bc.params[key] = defined(value);
            }
            input.boundary_conditions.push_back(std::move(bc));
            return input;
        };
    struct Declared {
        std::size_t count{0};
        bool mesh_flux_consistency{false};
        double gamma{0.0};
    };
    const auto declared =
        [&](const std::vector<std::pair<std::string, std::string>>& extra) {
            const auto input = make_input(extra);
            svmp::FE::systems::FESystem system(mesh);
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            EXPECT_TRUE(module);
            const auto declarations = system.meshNormalBoundaryConstraints();
            Declared result{};
            result.count = declarations.size();
            if (!declarations.empty()) {
                result.mesh_flux_consistency =
                    declarations.front().requires_mesh_flux_consistency;
                result.gamma = static_cast<double>(
                    declarations.front().mesh_flux_consistency_nitsche_gamma);
            }
            return result;
        };

    const auto by_default = declared({});
    EXPECT_EQ(by_default.count, 1u);
    EXPECT_TRUE(by_default.mesh_flux_consistency);
    EXPECT_DOUBLE_EQ(by_default.gamma, 10.0);

    const auto default_with_gamma =
        declared({{"Kinematic_nitsche_gamma", "24.0"}});
    EXPECT_EQ(default_with_gamma.count, 1u);
    EXPECT_TRUE(default_with_gamma.mesh_flux_consistency);
    EXPECT_DOUBLE_EQ(default_with_gamma.gamma, 24.0);
#endif
}

TEST(NavierStokesLegacyBCs, FreeSurfaceBoundaryKeysResolveThroughTheSchema)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Decision D47 (P2): the keys of a Free_surface boundary condition and of
    // its contact lines resolve through the option schemas.  Values that do
    // not parse report the schema texts (D52); the relations between keys
    // keep the register's texts.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 79;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    const auto make_input =
        [&](const std::vector<std::pair<std::string, std::string>>& extra) {
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
            input.default_domain.params["Viscosity.model"] =
                defined("Constant");
            input.default_domain.params["Viscosity.Value"] = defined("0.01");

            svmp::Physics::BoundaryConditionInput bc{};
            bc.name = "free_surface";
            bc.boundary_marker = marker;
            bc.params["Type"] = defined("Free_surface");
            bc.params["Implementation"] = defined("FittedALE");
            bc.params["External_pressure"] = defined("12.5");
            bc.params["Surface_tension"] = defined("0.0");
            bc.params["Tangential_mesh_policy"] = defined("Free");
            for (const auto& [key, value] : extra) {
                bc.params[key] = defined(value);
            }
            input.boundary_conditions.push_back(std::move(bc));
            return input;
        };

    struct FailureCase {
        std::vector<std::pair<std::string, std::string>> params;
        std::string expected_message;
    };
    const std::vector<FailureCase> failures = {
        // The scope check of the physical model reads Implementation first
        // (through the schema) and keeps its token.
        {{{"Implementation", "Sideways"}},
         "[svMultiPhysics::Physics] unsupported_free_surface_physical_model"},
        // Value and format errors: the schema texts.
        {{{"Surface_tension", "abc"}},
         "[Free_surface] Surface_tension: expected a number, got 'abc'"},
        {{{"Contact_line_model", "Sideways"}},
         "[Free_surface contact line] Contact_line_model: expected one of "
         "None | Pinned | PrescribedAngle | DynamicRenE, got 'Sideways'"},
        // Relations keep the register's texts.
        {{{"Tangential_mesh_penalty", "2.0"}},
         "[svMultiPhysics::Physics] Tangential_mesh_penalty requires "
         "Tangential_mesh_policy=Prescribed or SmoothingOnly; Free has no "
         "tangential boundary row."},
        {{{"Prescribed_tangential_mesh_velocity", "1 0 0"}},
         "[svMultiPhysics::Physics] Prescribed tangential mesh velocity "
         "conflicts with an explicit non-Prescribed tangential policy."},
    };
    for (const auto& failure : failures) {
        SCOPED_TRACE(failure.params.front().first + "=" +
                     failure.params.front().second);
        const auto input = make_input(failure.params);
        svmp::FE::systems::FESystem system(mesh);
        try {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
            ADD_FAILURE() << "the input must fail";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()), failure.expected_message);
        }
    }

    // Decision D47 item 3: a Free_surface boundary condition states its
    // Implementation.
    {
        auto input = make_input({});
        input.boundary_conditions.front().params.erase("Implementation");
        svmp::FE::systems::FESystem system(mesh);
        try {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
            ADD_FAILURE() << "a Free_surface BC without Implementation must fail";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()),
                      "[Free_surface] Implementation: required: FittedALE or "
                      "UnfittedLevelSet (decision D47)");
        }
    }

    // Accepted: the level-set keys of an unfitted interface on a fitted
    // boundary, which are not read.
    const std::vector<std::vector<std::pair<std::string, std::string>>>
        accepted = {
            {{"Interface_marker", "not-a-marker"}},
            {{"Level_set_isovalue", "not-a-number"}},
        };
    for (const auto& extra : accepted) {
        SCOPED_TRACE(extra.front().first + "=" + extra.front().second);
        const auto input = make_input(extra);
        svmp::FE::systems::FESystem system(mesh);
        EXPECT_NO_THROW({
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            EXPECT_TRUE(module);
        });
    }
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
            "\"tangential_mesh_owner\":"
            "\"IncompressibleNavierStokesVMSModule."
            "FreeSurfaceBoundary\""),
        std::string::npos);
    EXPECT_NE(
        artifact->json.find("\"policy_consumed\":true"),
        std::string::npos);
    EXPECT_NE(
        artifact->json.find("\"operator_tag\":\"equations\""),
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
    bc.params["Tangential_mesh_policy"] = defined("Prescribed");
    bc.params["Contact_line_model"] = defined("PrescribedContactAngle");
    bc.params["Contact_line_wall_markers"] = defined("88");
    bc.params["Contact_line_marker"] = defined("89");
    bc.params["Contact_angle_degrees"] = defined("60.0");
    bc.params["Contact_line_wall_normal"] = defined("1.0, 0.0, 0.0");
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

    // Each key alone must fail closed.  The canonical keys fail as incomplete
    // contact configurations; the other spellings (the removed aliases,
    // Contact_angle_radians and the singular Contact_line_wall_marker) fail
    // as unknown contact keys.
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
            // ContactLineModel was removed (decision D47): a second
            // spelling of the model is now an unknown contact key.
            "removed model alias",
            {
                {"Contact_line_model", "None"},
                {"ContactLineModel", "None"},
            },
            "Unknown free-surface contact key 'ContactLineModel'",
        },
        {
            "none with wall state",
            {
                {"Contact_line_model", "None"},
                {"Contact_line_wall_markers", "80"},
            },
            "Contact_line_model=None does not accept",
        },
        {
            "pinned with angle state",
            {
                {"Contact_line_model", "Pinned"},
                {"Contact_line_marker", "81"},
                {"Contact_angle_degrees", "57.29577951308232"},
            },
            "Contact_line_model=Pinned accepts only",
        },
        {
            "prescribed with dynamic state",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
                {"Contact_line_mobility", "0.5"},
            },
            "PrescribedAngle does not accept contact-line mobility",
        },
        {
            "prescribed with incomplete slip model",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
                {"Wall_slip_model", "Navier"},
            },
            "requires both Wall_slip_model and Wall_slip_length",
        },
        {
            "prescribed with incomplete slip length",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
                {"Wall_slip_length", "0.1"},
            },
            "requires both Wall_slip_model and Wall_slip_length",
        },
        {
            "prescribed with unsupported slip model",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
                {"Wall_slip_model", "None"},
                {"Wall_slip_length", "0.1"},
            },
            "requires Wall_slip_model=Navier",
        },
        {
            "removed prescribed-angle coefficient",
            {
                {"Contact_line_model", "DynamicRenE"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
                {"Contact_line_mobility", "0.5"},
                {"Wall_slip_model", "Navier"},
                {"Wall_slip_length", "0.1"},
                {"Contact_angle_penalty", "2.0"},
            },
            "Unknown free-surface contact key",
        },
        {
            "dynamic without navier slip",
            {
                {"Contact_line_model", "DynamicRenE"},
                {"Contact_line_wall_markers", "80"},
                {"Contact_line_wall_normal", "1 0 0"},
                {"Contact_angle_degrees", "57.29577951308232"},
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
                {"Contact_angle_degrees", "57.29577951308232"},
            },
            "cannot be shared by multiple contact walls",
        },
        {
            "normal count mismatch",
            {
                {"Contact_line_model", "PrescribedAngle"},
                {"Contact_line_wall_markers", "80;81"},
                {"Contact_line_wall_normal", "1 0 0;0 1 0;0 0 1"},
                {"Contact_angle_degrees", "57.29577951308232"},
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
    // The qualified fitted contract.
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
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.0");
    bc.params["Tangential_mesh_policy"] = defined("Free");
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
    // The qualified fitted contract.
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
    bc.params["External_pressure"] = defined("12.5");
    bc.params["Surface_tension"] = defined("0.0");
    bc.params["Tangential_mesh_policy"] = defined("Free");
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
    // Decision D4: each prescribed contact wall is a planar wall with Navier
    // slip and a strong normal-only velocity condition.
    auto mesh = buildSingleTetraTwoWallMesh(
        mesh_marker, first_wall_marker, second_wall_marker);
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
    bc.params["Contact_line_model"] = defined("PrescribedContactAngle");
    bc.params["Contact_line_wall_markers"] =
        defined(std::to_string(first_wall_marker) + "; " +
                std::to_string(second_wall_marker));
    bc.params["Contact_line_wall_normal"] =
        defined("-1.0 0.0 0.0; 0.0 -1.0 0.0");
    bc.params["Contact_angle_degrees"] = defined("90.0");
    bc.params["Wall_slip_model"] = defined("Navier");
    bc.params["Wall_slip_length"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(bc));
    for (const auto& [wall_marker, direction] :
         std::array<std::pair<int, const char*>, 2>{{
             {first_wall_marker, "1 0 0"},
             {second_wall_marker, "0 1 0"}}}) {
        svmp::Physics::BoundaryConditionInput wall{};
        wall.name = "contact_wall_" + std::to_string(wall_marker);
        wall.boundary_marker = wall_marker;
        wall.params["Type"] = defined("Dirichlet");
        wall.params["Value"] = defined("0.0");
        wall.params["Effective_direction"] = defined(direction);
        input.boundary_conditions.push_back(std::move(wall));
    }

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
    EXPECT_TRUE(system.isGeneratedEmbeddedInterfaceMarkerRegistered(
        first_contact_marker));
    EXPECT_TRUE(system.isGeneratedEmbeddedInterfaceMarkerRegistered(
        second_contact_marker));
    EXPECT_FALSE(formulationRecordsContainInterfaceMarker(
        system, first_contact_marker));
    EXPECT_FALSE(formulationRecordsContainInterfaceMarker(
        system, second_contact_marker));
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
    free_surface.params["Generated_interface_geometry"] =
        defined("LinearCorner");
    free_surface.params["Small_cut_aggregation"] = defined("false");
    free_surface.params["Surface_tension"] = defined("0.8");
    free_surface.params["Contact_line_model"] =
        defined("DynamicContactAngle");
    free_surface.params["Contact_line_wall_markers"] =
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

TEST(NavierStokesLegacyBCs,
     UnfittedPrescribedAngleNavierSlipUsesSharpWallWithoutLineFriction)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int wall_marker = 82;
    constexpr int interface_marker = 205;
    auto mesh = buildSingleTetraBoundaryMesh(
        wall_marker, /*label_all_faces=*/false);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
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
    free_surface.params["Level_set_field_name"] =
        defined("phi_prescribed_slip");
    free_surface.params["Active_domain"] = defined("LevelSetNegative");
    free_surface.params["Generated_interface_geometry"] =
        defined("LinearCorner");
    free_surface.params["Small_cut_aggregation"] = defined("false");
    free_surface.params["Surface_tension"] = defined("0.8");
    free_surface.params["Contact_line_model"] =
        defined("PrescribedAngle");
    free_surface.params["Contact_line_wall_markers"] =
        defined(std::to_string(wall_marker));
    free_surface.params["Contact_line_wall_normal"] =
        defined("0.0 0.0 -1.0");
    free_surface.params["Contact_angle_degrees"] = defined("60.0");
    free_surface.params["Wall_slip_model"] = defined("Navier");
    free_surface.params["Wall_slip_length"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(free_surface));

    svmp::Physics::BoundaryConditionInput wall{};
    wall.name = "prescribed_contact_wall";
    wall.boundary_marker = wall_marker;
    wall.params["Type"] = defined("Dirichlet");
    wall.params["Value"] = defined("0.0");
    wall.params["Effective_direction"] = defined("0 0 1");
    input.boundary_conditions.push_back(std::move(wall));

    svmp::FE::systems::FESystem system(mesh);
    auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
        svmp::FE::ElementType::Tetra4, 1);
    const auto phi = system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi_prescribed_slip",
        .space = scalar_space,
        .components = 1,
    });
    system.addOperator("level_set_owner");
    const auto phi_state = svmp::FE::forms::StateField(
        phi, *scalar_space, "phi_prescribed_slip_owner");
    const auto eta = svmp::FE::forms::TestField(
        phi, *scalar_space, "eta_prescribed_slip_owner");
    (void)svmp::FE::systems::installFormulation(
        system,
        "level_set_owner",
        {phi},
        (phi_state * eta).dx());
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
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(
        system, contact_marker));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(
        system, active_wall_marker));

    const auto declarations =
        svmp::FE::systems::functionalRecords(system).functionalDeclarations();
    ASSERT_EQ(declarations.size(), 1u);
    ASSERT_EQ(
        declarations.front().parameters.dynamic_contact_coefficients.size(),
        1u);
    const auto& coefficient =
        declarations.front().parameters.dynamic_contact_coefficients.front();
    EXPECT_EQ(coefficient.law,
              svmp::FE::interfaces::FreeSurfaceContactLaw::PrescribedAngle);
    EXPECT_DOUBLE_EQ(coefficient.mobility, 0.0);
    EXPECT_DOUBLE_EQ(coefficient.slip_length, 0.2);
    EXPECT_DOUBLE_EQ(coefficient.dynamic_viscosity, 0.01);

    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(artifact->json.find("\"wall_slip_model\":\"Navier\""),
              std::string::npos);
    EXPECT_NE(artifact->json.find("\"line_friction\":\"absent\""),
              std::string::npos);
#endif
}

TEST(NavierStokesLegacyBCs,
     DerivedRequestInstallsConservativeBalanceOperatorsWithoutTheEnvironmentSwitch)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // The static-capillary certificate reads the conservative-balance
    // operators.  The Application requests them through the module input
    // when a level-set equation enables the static-capillary initialization
    // (decision D47, P0.4); the environment switch cannot withdraw them.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();
    using Operators = svmp::Physics::formulations::navier_stokes::
        FreeSurfaceConservativeBalanceDiagnosticOperators;

    constexpr const char* kSwitch =
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC";
    std::optional<std::string> saved_switch;
    if (const char* existing = std::getenv(kSwitch); existing != nullptr) {
        saved_switch = std::string(existing);
    }
    ASSERT_EQ(::setenv(kSwitch, "0", 1), 0);

    constexpr int wall_marker = 82;
    constexpr int interface_marker = 205;
    const auto operators_installed = [&](bool requested) {
        auto mesh = buildSingleTetraBoundaryMesh(
            wall_marker, /*label_all_faces=*/false);
        EXPECT_TRUE(mesh);

        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.derived_requests.conservative_balance_diagnostics = requested;
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
        free_surface.params["Level_set_field_name"] =
            defined("phi_derived_request");
        free_surface.params["Active_domain"] = defined("LevelSetNegative");
        free_surface.params["Generated_interface_geometry"] =
            defined("LinearCorner");
        free_surface.params["Small_cut_aggregation"] = defined("false");
        free_surface.params["Surface_tension"] = defined("0.8");
        free_surface.params["Contact_line_model"] =
            defined("PrescribedAngle");
        free_surface.params["Contact_line_wall_markers"] =
            defined(std::to_string(wall_marker));
        free_surface.params["Contact_line_wall_normal"] =
            defined("0.0 0.0 -1.0");
        free_surface.params["Contact_angle_degrees"] = defined("60.0");
        free_surface.params["Wall_slip_model"] = defined("Navier");
        free_surface.params["Wall_slip_length"] = defined("0.2");
        input.boundary_conditions.push_back(std::move(free_surface));

        svmp::Physics::BoundaryConditionInput wall{};
        wall.name = "prescribed_contact_wall";
        wall.boundary_marker = wall_marker;
        wall.params["Type"] = defined("Dirichlet");
        wall.params["Value"] = defined("0.0");
        wall.params["Effective_direction"] = defined("0 0 1");
        input.boundary_conditions.push_back(std::move(wall));

        svmp::FE::systems::FESystem system(mesh);
        auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Tetra4, 1);
        const auto phi = system.addField(svmp::FE::systems::FieldSpec{
            .name = "phi_derived_request",
            .space = scalar_space,
            .components = 1,
        });
        system.addOperator("level_set_owner");
        const auto phi_state = svmp::FE::forms::StateField(
            phi, *scalar_space, "phi_derived_request_owner");
        const auto eta = svmp::FE::forms::TestField(
            phi, *scalar_space, "eta_derived_request_owner");
        (void)svmp::FE::systems::installFormulation(
            system,
            "level_set_owner",
            {phi},
            (phi_state * eta).dx());

        auto module = svmp::Physics::EquationModuleRegistry::instance().create(
            "fluid", input, system);
        EXPECT_TRUE(module);
        std::size_t installed = 0u;
        for (const std::string_view tag : {
                 Operators::pressure_virtual_work,
                 Operators::surface_energy_virtual_work,
                 Operators::gravitational_potential_virtual_work,
                 Operators::physical_potential_virtual_work,
                 Operators::pressure_representability_load_virtual_work,
                 Operators::conservative_balance,
                 Operators::pressure_representability_pair,
             }) {
            if (system.hasOperator(std::string(tag))) {
                ++installed;
            }
        }
        return installed;
    };

    const auto without_request = operators_installed(false);
    const auto with_request = operators_installed(true);
    if (saved_switch.has_value()) {
        (void)::setenv(kSwitch, saved_switch->c_str(), 1);
    } else {
        (void)::unsetenv(kSwitch);
    }
    EXPECT_EQ(without_request, 0u);
    EXPECT_EQ(with_request, 7u);
#endif
}

TEST(NavierStokesLegacyBCs, ResolvedArtifactSectionRecordsTheSourceOfEveryFreeSurfaceKey)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Decision D47 (P1.2a): the Navier-Stokes artifact ends with a
    // "resolved" section with every key of each free-surface boundary, its
    // resolved value and its source.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int wall_marker = 82;
    constexpr int interface_marker = 205;
    auto mesh = buildSingleTetraBoundaryMesh(
        wall_marker, /*label_all_faces=*/false);
    ASSERT_TRUE(mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "single_tetra";
    input.mesh = mesh->local_mesh_ptr();
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
    free_surface.params["Level_set_field_name"] = defined("phi_resolved");
    // A key both elements read: the domain value replaces the equation's.
    input.equation_params["Mesh_velocity_field"] = defined("equation_mesh_velocity");
    input.default_domain.params["Mesh_velocity_field"] = defined("domain_mesh_velocity");
    free_surface.params["Active_domain"] = defined("negative");
    free_surface.params["Generated_interface_geometry"] =
        defined("LinearCorner");
    free_surface.params["Small_cut_aggregation"] = defined("false");
    free_surface.params["Surface_tension"] = defined("0.8");
    free_surface.params["Contact_line_model"] =
        defined("PrescribedAngle");
    free_surface.params["Contact_line_wall_markers"] =
        defined(std::to_string(wall_marker));
    free_surface.params["Contact_line_wall_normal"] =
        defined("0.0 0.0 -1.0");
    free_surface.params["Contact_angle_degrees"] = defined("60.0");
    free_surface.params["Wall_slip_model"] = defined("Navier");
    free_surface.params["Wall_slip_length"] = defined("0.2");
    input.boundary_conditions.push_back(std::move(free_surface));

    svmp::Physics::BoundaryConditionInput wall{};
    wall.name = "prescribed_contact_wall";
    wall.boundary_marker = wall_marker;
    wall.params["Type"] = defined("Dirichlet");
    wall.params["Value"] = defined("0.0");
    wall.params["Effective_direction"] = defined("0 0 1");
    input.boundary_conditions.push_back(std::move(wall));

    svmp::FE::systems::FESystem system(mesh);
    auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
        svmp::FE::ElementType::Tetra4, 1);
    const auto phi = system.addField(svmp::FE::systems::FieldSpec{
        .name = "phi_resolved",
        .space = scalar_space,
        .components = 1,
    });
    system.addOperator("level_set_owner");
    const auto phi_state = svmp::FE::forms::StateField(
        phi, *scalar_space, "phi_resolved_owner");
    const auto eta = svmp::FE::forms::TestField(
        phi, *scalar_space, "eta_resolved_owner");
    (void)svmp::FE::systems::installFormulation(
        system,
        "level_set_owner",
        {phi},
        (phi_state * eta).dx());

    auto module = svmp::Physics::EquationModuleRegistry::instance().create(
        "fluid", input, system);
    ASSERT_TRUE(module);
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    const auto& json = artifact->json;
    const auto resolved_at = json.rfind(",\"resolved\":{\"version\":1,\"free_surfaces\":[{\"boundary\":{");
    ASSERT_NE(resolved_at, std::string::npos) << json;
    EXPECT_EQ(json.back(), '}');
    const auto resolved = json.substr(resolved_at);
    const auto expect_entry = [&](std::string_view entry) {
        EXPECT_NE(resolved.find(entry), std::string::npos) << entry << "\n" << resolved;
    };
    // Given by the input; the canonical value text is kept, the input text
    // only when it differs.
    expect_entry("\"Implementation\":{\"value\":\"UnfittedLevelSet\",\"source\":\"explicit\","
                 "\"spelling\":\"Implementation\"}");
    expect_entry("\"Active_domain\":{\"value\":\"LevelSetNegative\",\"source\":\"explicit\","
                 "\"spelling\":\"Active_domain\",\"input\":\"negative\"}");
    expect_entry("\"Interface_marker\":{\"value\":\"205\",\"source\":\"explicit\","
                 "\"spelling\":\"Interface_marker\"}");
    expect_entry("\"Surface_tension\":{\"value\":\"0.8\",\"source\":\"explicit\","
                 "\"spelling\":\"Surface_tension\"}");
    expect_entry("\"Small_cut_aggregation\":{\"value\":\"false\",\"source\":\"explicit\","
                 "\"spelling\":\"Small_cut_aggregation\"}");
    // Left to the defaults.
    expect_entry("\"Level_set_isovalue\":{\"value\":\"0\",\"source\":\"default\"}");
    // The contact-line keys.
    expect_entry(",\"contact_line\":{\"Contact_line_model\":{\"value\":\"PrescribedAngle\","
                 "\"source\":\"explicit\",\"spelling\":\"Contact_line_model\"}");
    expect_entry("\"source\":\"explicit\",\"spelling\":\"Contact_angle_degrees\",\"input\":\"60.0\"}");
    // The wall markers, given by their only input spelling.
    expect_entry("{\"value\":\"82\",\"source\":\"explicit\","
                 "\"spelling\":\"Contact_line_wall_markers\"}");
    expect_entry("\"Wall_slip_model\":{\"value\":\"Navier\",\"source\":\"explicit\","
                 "\"spelling\":\"Wall_slip_model\"}");
    expect_entry("\"Contact_line_mobility\":{\"value\":\"0\",\"source\":\"default\"}");
    // Keys outside the schema (Type) are not listed.
    EXPECT_EQ(resolved.find("\"Type\""), std::string::npos);

    // The equation and domain keys (decision D47, P2): the value the module
    // uses, and its source in the element's own input ("input" keeps the
    // element's text when the other element set the value).
    const auto equation_at = resolved.find(",\"equation\":{");
    const auto domain_at = resolved.find(",\"domain\":{");
    ASSERT_NE(equation_at, std::string::npos) << resolved;
    ASSERT_NE(domain_at, std::string::npos) << resolved;
    ASSERT_LT(equation_at, domain_at);
    const auto equation_section = resolved.substr(equation_at, domain_at - equation_at);
    const auto domain_section = resolved.substr(domain_at);
    const auto expect_in = [](const std::string& section, std::string_view entry) {
        EXPECT_NE(section.find(entry), std::string::npos) << entry << "\n" << section;
    };
    expect_in(equation_section, "\"Enable_ALE\":{\"value\":\"false\",\"source\":\"default\"}");
    expect_in(equation_section,
              "\"Mesh_velocity_field\":{\"value\":\"domain_mesh_velocity\",\"source\":\"explicit\","
              "\"spelling\":\"Mesh_velocity_field\",\"input\":\"equation_mesh_velocity\"}");
    expect_in(domain_section,
              "\"Mesh_velocity_field\":{\"value\":\"domain_mesh_velocity\",\"source\":\"explicit\","
              "\"spelling\":\"Mesh_velocity_field\"}");
    expect_in(domain_section,
              "\"Density\":{\"value\":\"1\",\"source\":\"explicit\",\"spelling\":\"Density\","
              "\"input\":\"1.0\"}");
    expect_in(domain_section,
              "\"Viscosity.model\":{\"value\":\"Constant\",\"source\":\"explicit\","
              "\"spelling\":\"Viscosity.model\"}");
    expect_in(domain_section,
              "\"Viscosity.Value\":{\"value\":\"0.01\",\"source\":\"explicit\","
              "\"spelling\":\"Viscosity.Value\"}");
    expect_in(domain_section,
              "\"Backflow_stabilization_coefficient\":{\"value\":\"0\",\"source\":\"default\"}");
#endif
}

TEST(NavierStokesLegacyBCs, NavierStokesEquationAndDomainKeysResolveThroughTheSchema)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Decision D47 (P2): the equation and domain keys resolve through the
    // option schemas.  Values that do not parse report the schema texts
    // (D52); the register's own checks keep their texts.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    constexpr int marker = 83;
    auto mesh = buildSingleTetraBoundaryMesh(marker);
    ASSERT_TRUE(mesh);

    using Keys = std::vector<std::pair<std::string, std::string>>;
    struct FailureCase {
        Keys equation;
        Keys domain;
        std::string expected_message;
    };
    const std::vector<FailureCase> failures = {
        {{}, {{"Density", "abc"}},
         "[Navier-Stokes domain] Density: expected a number, got 'abc'"},
        {{{"Mesh_velocity_source", "sideways"}}, {},
         "[Navier-Stokes equation] Mesh_velocity_source: expected one of "
         "prescribed_data | coupled_displacement, got 'sideways'"},
        {{}, {{"Viscosity.Value", "0"}},
         "[svMultiPhysics::Physics] Constant viscosity must be > 0 for the new "
         "solver Navier-Stokes module. Set <Use_new_OOP_solver>false"
         "</Use_new_OOP_solver> to use the legacy solver."},
    };
    for (const auto& failure : failures) {
        SCOPED_TRACE(failure.expected_message);
        svmp::Physics::EquationModuleInput input{};
        input.equation_type = "fluid";
        input.mesh_name = "single_tetra";
        input.mesh = mesh->local_mesh_ptr();
        input.default_domain.params["Density"] = defined("1.0");
        input.default_domain.params["Viscosity.model"] = defined("Constant");
        input.default_domain.params["Viscosity.Value"] = defined("0.01");
        for (const auto& [key, value] : failure.equation) {
            input.equation_params[key] = defined(value);
        }
        for (const auto& [key, value] : failure.domain) {
            input.default_domain.params[key] = defined(value);
        }
        svmp::FE::systems::FESystem system(mesh);
        try {
            auto module =
                svmp::Physics::EquationModuleRegistry::instance().create(
                    "fluid", input, system);
            (void)module;
            ADD_FAILURE() << "the input must fail";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()), failure.expected_message);
        }
    }
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
     UnfittedFreeSurfaceSmallCutAggregationGuardsTranslation_RecordsGuards)
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
    bc.params["External_pressure"] = defined("1.0");
    bc.params["Enable_cut_cell_stabilization"] = defined("true");
    bc.params["Cut_cell_pressure_gradient_penalty"] = defined("0.2");
    bc.params["Small_cut_aggregation_maximum_root_path_length"] =
        defined("5");
    bc.params[
        "Small_cut_aggregation_maximum_reference_extrapolation_distance"] =
        defined("2.5");
    bc.params["Small_cut_aggregation_maximum_absolute_coefficient"] =
        defined("7");
    bc.params["Small_cut_aggregation_maximum_row_l1_norm"] = defined("9");
    bc.params["Small_cut_aggregation_rootless_fallback"] = defined("true");
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
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(
        artifact->json.find(
            "\"aggregation_guards\":{\"maximum_root_path_length\":5,"
            "\"maximum_reference_extrapolation_distance\":2.5,"
            "\"maximum_absolute_coefficient\":7,"
            "\"maximum_row_l1_norm\":9,"
            "\"rootless_fallback\":true}"),
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

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_DefaultsToLevelSetNegative)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    // Decision D47 item 3: an unfitted surface that does not state
    // Active_domain is liquid on the negative side (it failed before).
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

    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    EXPECT_TRUE(formulationRecordsContainCutVolumeSide(
        system,
        svmp::FE::forms::CutVolumeSide::Negative));
#endif
}

TEST(NavierStokesLegacyBCs, UnfittedFreeSurfaceActiveDomainTranslation_RejectsNone)
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
