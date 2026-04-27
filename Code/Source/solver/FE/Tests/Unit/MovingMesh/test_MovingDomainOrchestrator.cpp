#include <gtest/gtest.h>

#include "MovingMesh/MovingDomainOrchestrator.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "TimeStepping/TimeHistory.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Configuration;
using svmp::INVALID_INDEX;
using svmp::INVALID_LABEL;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::offset_t;
using svmp::real_t;
using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::MassKernel;
using svmp::FE::moving_mesh::BoundaryMotionConfig;
using svmp::FE::moving_mesh::BoundaryMotionValueMode;
using svmp::FE::moving_mesh::MotionBackendModel;
using svmp::FE::moving_mesh::MovingDomainAdvancePoint;
using svmp::FE::moving_mesh::MovingDomainConfig;
using svmp::FE::moving_mesh::MovingDomainOrchestrator;
using svmp::FE::moving_mesh::MovingMeshMode;
using svmp::FE::moving_mesh::moving_domain_config_from_kv;
using svmp::FE::spaces::H1Space;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

std::shared_ptr<Mesh> make_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3};

    CellShape quad{};
    quad.family = CellFamily::Quad;
    quad.num_corners = 4;
    quad.order = 1;
    base->build_from_arrays(2, x_ref, offsets, conn, {quad});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> make_square_fan_mesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> coords = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.5,
    };
    const std::vector<offset_t> offsets = {0, 3, 6, 9, 12};
    const std::vector<svmp::index_t> conn = {
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4,
    };
    CellShape tri{};
    tri.family = CellFamily::Triangle;
    tri.num_corners = 3;
    tri.order = 1;
    base->build_from_arrays(2, coords, offsets, conn, {tri, tri, tri, tri});
    base->finalize();

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto cells = base->face_cells(f);
        if (cells[1] != INVALID_INDEX) {
            continue;
        }
        const auto verts = base->face_vertices(f);
        bool all_left = true;
        bool all_right = true;
        for (const auto v : verts) {
            const auto xyz = base->get_vertex_coords(v);
            all_left = all_left && std::abs(xyz[0]) < 1.0e-12;
            all_right = all_right && std::abs(xyz[0] - 1.0) < 1.0e-12;
        }
        if (all_left) {
            base->set_boundary_label(f, 10);
        } else if (all_right) {
            base->set_boundary_label(f, 20);
        } else {
            base->set_boundary_label(f, 30);
        }
    }

    return svmp::create_mesh(std::move(base));
}

FESystem make_mass_system(std::shared_ptr<Mesh> mesh, Configuration cfg)
{
    auto space = std::make_shared<H1Space>(ElementType::Quad4, 1);
    FESystem sys(std::move(mesh), cfg);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();
    return sys;
}

DenseMatrixView assemble_mass(FESystem& sys)
{
    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    EXPECT_TRUE(sys.assembleMass(state, mass).success);
    return mass;
}

void expect_scaled_matrix(const DenseMatrixView& actual,
                          const DenseMatrixView& expected,
                          Real scale)
{
    ASSERT_EQ(actual.numRows(), expected.numRows());
    ASSERT_EQ(actual.numCols(), expected.numCols());
    for (GlobalIndex i = 0; i < actual.numRows(); ++i) {
        for (GlobalIndex j = 0; j < actual.numCols(); ++j) {
            EXPECT_NEAR(actual.getMatrixEntry(i, j),
                        scale * expected.getMatrixEntry(i, j),
                        1.0e-12);
        }
    }
}

} // namespace

TEST(MovingDomainOrchestrator, FlatInputConfigurationCoversModeBackendOptionsAndBoundaries)
{
    const auto config = moving_domain_config_from_kv({
        {"mesh_motion.mode", "prescribed"},
        {"mesh_motion.coordinate_configuration", "current"},
        {"mesh_motion.backend", "prescribed"},
        {"mesh_motion.max_step_scale", "0.5"},
        {"mesh_motion.max_substeps", "4"},
        {"mesh_motion.enable_quality_guard", "false"},
        {"mesh_motion.geometry_regularization.model", "artificial_pseudo_elastic"},
        {"mesh_motion.geometry_regularization.weight_mode", "element_size"},
        {"mesh_motion.geometry_regularization.max_linear_iterations", "17"},
        {"mesh_motion.boundary.count", "1"},
        {"mesh_motion.boundary.0.label", "42"},
        {"mesh_motion.boundary.0.components", "x,z"},
        {"mesh_motion.boundary.0.mode", "constant_velocity"},
        {"mesh_motion.boundary.0.value", "1.0,2.0,3.0"},
    });

    EXPECT_EQ(config.mode, MovingMeshMode::PrescribedMotion);
    EXPECT_EQ(config.fe_coordinate_configuration, Configuration::Current);
    EXPECT_EQ(config.backend_model, MotionBackendModel::Prescribed);
    EXPECT_DOUBLE_EQ(config.motion.max_step_scale, 0.5);
    EXPECT_EQ(config.motion.max_substeps, 4);
    EXPECT_FALSE(config.motion.enable_quality_guard);
    EXPECT_EQ(config.geometry_regularization.max_linear_iterations, 17);
    ASSERT_EQ(config.boundary_motion.size(), 1u);
    EXPECT_EQ(config.boundary_motion[0].boundary_label, 42);
    EXPECT_TRUE(config.boundary_motion[0].component_mask[0]);
    EXPECT_FALSE(config.boundary_motion[0].component_mask[1]);
    EXPECT_TRUE(config.boundary_motion[0].component_mask[2]);
    EXPECT_EQ(config.boundary_motion[0].value_mode, BoundaryMotionValueMode::ConstantVelocity);
    EXPECT_DOUBLE_EQ(config.boundary_motion[0].value[0], 1.0);
    EXPECT_DOUBLE_EQ(config.boundary_motion[0].value[1], 2.0);
    EXPECT_DOUBLE_EQ(config.boundary_motion[0].value[2], 3.0);
}

TEST(MovingDomainOrchestrator, DisabledDefaultPreservesReferenceStaticBehavior)
{
    auto mesh = make_single_quad_mesh();
    MovingDomainConfig config;
    MovingDomainOrchestrator orchestrator(mesh, config);

    const auto before_revision = mesh->local_mesh().geometry_revision();
    const auto diagnostics = orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                                  1.0,
                                                  0.25);

    EXPECT_TRUE(diagnostics.success);
    EXPECT_FALSE(diagnostics.advanced_geometry);
    EXPECT_FALSE(mesh->local_mesh().has_current_coords());
    EXPECT_EQ(mesh->local_mesh().active_configuration(), Configuration::Reference);
    EXPECT_EQ(mesh->local_mesh().geometry_revision(), before_revision);
}

TEST(MovingDomainOrchestrator, CoupledMonolithicModeReportsUnsupportedWithoutAdvancing)
{
    auto mesh = make_single_quad_mesh();

    MovingDomainConfig config;
    config.mode = MovingMeshMode::CoupledMonolithic;
    config.fe_coordinate_configuration = Configuration::Current;
    MovingDomainOrchestrator orchestrator(mesh, config);

    const auto before_revision = mesh->local_mesh().geometry_revision();
    const auto diagnostics = orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                                  1.0,
                                                  0.25);

    EXPECT_FALSE(diagnostics.success);
    EXPECT_FALSE(diagnostics.advanced_geometry);
    EXPECT_NE(diagnostics.message.find("not supported"), std::string::npos);
    EXPECT_FALSE(mesh->local_mesh().has_current_coords());
    EXPECT_EQ(mesh->local_mesh().active_configuration(), Configuration::Reference);
    EXPECT_EQ(mesh->local_mesh().geometry_revision(), before_revision);
}

TEST(MovingDomainOrchestrator, CoordinateConfigurationMismatchFailsBeforeMotion)
{
    auto mesh = make_single_quad_mesh();
    auto sys = make_mass_system(mesh, Configuration::Reference);

    BoundaryMotionConfig boundary;
    boundary.boundary_label = INVALID_LABEL;
    boundary.value = {{0.1, 0.0, 0.0}};

    MovingDomainConfig config;
    config.mode = MovingMeshMode::PrescribedMotion;
    config.fe_coordinate_configuration = Configuration::Current;
    config.boundary_motion = {boundary};
    MovingDomainOrchestrator orchestrator(mesh, config);

    std::array<FESystem*, 1> systems{{&sys}};
    std::span<FESystem* const> system_span(systems.data(), systems.size());

    EXPECT_THROW((void)orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                            1.0,
                                            0.25,
                                            system_span),
                 std::exception);
    EXPECT_FALSE(mesh->local_mesh().has_current_coords());
    EXPECT_EQ(mesh->local_mesh().active_configuration(), Configuration::Reference);
}

TEST(MovingDomainOrchestrator, InvalidPrescribedBoundaryRollsBackAndReportsDiagnostic)
{
    auto mesh = make_single_quad_mesh();

    BoundaryMotionConfig boundary;
    boundary.boundary_label = 999;
    boundary.value = {{0.1, 0.0, 0.0}};

    MovingDomainConfig config;
    config.mode = MovingMeshMode::PrescribedMotion;
    config.fe_coordinate_configuration = Configuration::Current;
    config.boundary_motion = {boundary};
    MovingDomainOrchestrator orchestrator(mesh, config);

    const auto before_revision = mesh->local_mesh().geometry_revision();
    const auto diagnostics = orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                                  1.0,
                                                  0.25);

    EXPECT_FALSE(diagnostics.success);
    EXPECT_TRUE(diagnostics.rolled_back);
    EXPECT_FALSE(diagnostics.advanced_geometry);
    EXPECT_NE(diagnostics.message.find("rolled back"), std::string::npos);
    EXPECT_FALSE(mesh->local_mesh().has_current_coords());
    EXPECT_EQ(mesh->local_mesh().active_configuration(), Configuration::Reference);
    EXPECT_GE(mesh->local_mesh().geometry_revision(), before_revision);
}

TEST(MovingDomainOrchestrator, ExternalBackendRequiresInjection)
{
    auto mesh = make_single_quad_mesh();

    BoundaryMotionConfig boundary;
    boundary.boundary_label = INVALID_LABEL;
    boundary.value = {{0.1, 0.0, 0.0}};

    MovingDomainConfig config;
    config.mode = MovingMeshMode::PrescribedMotion;
    config.fe_coordinate_configuration = Configuration::Current;
    config.backend_model = MotionBackendModel::External;
    config.boundary_motion = {boundary};
    MovingDomainOrchestrator orchestrator(mesh, config);

    EXPECT_THROW((void)orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                            1.0,
                                            0.25),
                 std::exception);
}

TEST(MovingDomainOrchestrator, PrescribedMotionNotifiesFESystemBeforeCurrentAssembly)
{
    auto mesh = make_single_quad_mesh();
    auto sys = make_mass_system(mesh, Configuration::Current);
    const auto reference_mass = assemble_mass(sys);

    BoundaryMotionConfig boundary;
    boundary.boundary_label = INVALID_LABEL;
    boundary.value_mode = BoundaryMotionValueMode::Function;
    boundary.function = [](const std::array<real_t, 3>& x, double, double step_scale) {
        return std::array<real_t, 3>{{x[0] * step_scale, 0.0, 0.0}};
    };

    MovingDomainConfig config;
    config.mode = MovingMeshMode::PrescribedMotion;
    config.fe_coordinate_configuration = Configuration::Current;
    config.backend_model = MotionBackendModel::Prescribed;
    config.boundary_motion = {boundary};
    MovingDomainOrchestrator orchestrator(mesh, config);

    std::array<FESystem*, 1> systems{{&sys}};
    std::span<FESystem* const> system_span(systems.data(), systems.size());
    const auto diagnostics = orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                                  1.0,
                                                  1.0,
                                                  system_span);

    EXPECT_TRUE(diagnostics.success) << diagnostics.message;
    EXPECT_TRUE(diagnostics.advanced_geometry);
    EXPECT_EQ(diagnostics.notified_fe_systems, 1u);
    EXPECT_EQ(mesh->local_mesh().active_configuration(), Configuration::Current);
    ASSERT_TRUE(mesh->local_mesh().has_current_coords());

    const auto moved_mass = assemble_mass(sys);
    expect_scaled_matrix(moved_mass, reference_mass, 2.0);
}

TEST(MovingDomainOrchestrator, BeforePhysicsSolveCallbackAdvancesMovingDomainAtDocumentedPoint)
{
    auto mesh = make_single_quad_mesh();

    BoundaryMotionConfig boundary;
    boundary.boundary_label = INVALID_LABEL;
    boundary.value_mode = BoundaryMotionValueMode::ConstantDisplacement;
    boundary.value = {{0.1, 0.0, 0.0}};

    MovingDomainConfig config;
    config.mode = MovingMeshMode::PrescribedMotion;
    config.fe_coordinate_configuration = Configuration::Current;
    config.boundary_motion = {boundary};

    MovingDomainOrchestrator orchestrator(mesh, config);
    auto callback = orchestrator.makeBeforePhysicsSolveCallback();
    svmp::FE::timestepping::TimeHistory history;
    ASSERT_TRUE(callback(history, 2.0, 0.5));

    const auto& diagnostics = orchestrator.lastDiagnostics();
    EXPECT_EQ(diagnostics.advance_point, std::string("before_physics_solve"));
    EXPECT_DOUBLE_EQ(diagnostics.time, 2.0);
    EXPECT_DOUBLE_EQ(diagnostics.dt, 0.5);
    EXPECT_TRUE(diagnostics.advanced_geometry);
}

TEST(MovingDomainOrchestrator, FEBackedSmoothingCanBeEnabledWithoutPhysicsCode)
{
    auto mesh = make_square_fan_mesh();

    BoundaryMotionConfig left_x;
    left_x.boundary_label = 10;
    left_x.component_mask = {{true, false, false}};
    left_x.value = {{0.0, 0.0, 0.0}};

    BoundaryMotionConfig right_x;
    right_x.boundary_label = 20;
    right_x.component_mask = {{true, false, false}};
    right_x.value = {{0.2, 0.0, 0.0}};

    BoundaryMotionConfig all_y;
    all_y.boundary_label = INVALID_LABEL;
    all_y.component_mask = {{false, true, false}};
    all_y.value = {{0.0, 0.0, 0.0}};

    MovingDomainConfig config;
    config.mode = MovingMeshMode::FEBackedSmoothing;
    config.fe_coordinate_configuration = Configuration::Current;
    config.boundary_motion = {left_x, right_x, all_y};

    MovingDomainOrchestrator orchestrator(mesh, config);
    const auto diagnostics = orchestrator.advance(MovingDomainAdvancePoint::BeforePhysicsSolve,
                                                  1.0,
                                                  0.5);

    EXPECT_TRUE(diagnostics.success) << diagnostics.message;
    EXPECT_TRUE(diagnostics.advanced_geometry);
    EXPECT_EQ(diagnostics.backend_model, std::string("fe_geometry_regularization"));
    EXPECT_NE(diagnostics.backend_name.find("GeometryRegularization"), std::string::npos);
    EXPECT_FALSE(diagnostics.has_inverted_cells);

    const auto& x_cur = mesh->local_mesh().X_cur();
    ASSERT_EQ(x_cur.size(), 10u);
    EXPECT_GT(x_cur[8], 0.5);
    EXPECT_LT(x_cur[8], 0.7);
    EXPECT_NEAR(x_cur[9], 0.5, 1.0e-12);
}
