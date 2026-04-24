#include <gtest/gtest.h>

#include "MovingMesh/GeometryRegularizationBackend.h"

#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Motion/MeshMotion.h"
#include "Mesh/Motion/MotionFields.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Configuration;
using svmp::EntityKind;
using svmp::INVALID_INDEX;
using svmp::INVALID_LABEL;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::offset_t;
using svmp::real_t;
using svmp::FE::moving_mesh::GeometryRegularizationModel;
using svmp::FE::moving_mesh::GeometryRegularizationMotionBackend;
using svmp::FE::moving_mesh::GeometryRegularizationOptions;
using svmp::FE::moving_mesh::GeometryRegularizationScopeContract;
using svmp::FE::moving_mesh::GeometryRegularizationWeightMode;

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

std::shared_ptr<Mesh> make_unit_tet_mesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const std::vector<offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3};
    CellShape tet{};
    tet.family = CellFamily::Tetra;
    tet.num_corners = 4;
    tet.order = 1;
    base->build_from_arrays(3, coords, offsets, conn, {tet});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> make_unit_cube_hex_mesh()
{
    auto base = std::make_shared<MeshBase>();
    for (svmp::index_t v = 0; v < 8; ++v) {
        const real_t x = static_cast<real_t>((v & 1) != 0);
        const real_t y = static_cast<real_t>((v & 2) != 0);
        const real_t z = static_cast<real_t>((v & 4) != 0);
        base->add_vertex(v, {{x, y, z}});
    }
    base->add_cell(0, CellShape::Hexahedron, {0, 1, 3, 2, 4, 5, 7, 6});
    base->add_boundary_face(0, {0, 2, 3, 1});
    base->add_boundary_face(1, {4, 5, 7, 6});
    base->add_boundary_face(2, {0, 1, 5, 4});
    base->add_boundary_face(3, {2, 6, 7, 3});
    base->add_boundary_face(4, {0, 4, 6, 2});
    base->add_boundary_face(5, {1, 3, 7, 5});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains_forbidden_physics_term(const std::string& value)
{
    const auto lower = lower_copy(value);
    const std::vector<std::string> forbidden = {
        "stress",
        "traction",
        "material",
        "constitutive",
        "history",
        "fluid",
        "solid",
        "fsi",
        "wall_law",
        "turbulence",
    };
    return std::any_of(forbidden.begin(), forbidden.end(), [&](const std::string& term) {
        return lower.find(term) != std::string::npos;
    });
}

} // namespace

TEST(FEMeshMotionBackend, ConstructionOptionsAndDiagnosticsArePhysicsNeutral)
{
    static_assert(!GeometryRegularizationScopeContract::depends_on_physics_modules);
    static_assert(!GeometryRegularizationScopeContract::implements_physical_coupling_laws);
    static_assert(!GeometryRegularizationScopeContract::returns_physics_outputs);

    GeometryRegularizationMotionBackend backend;
    EXPECT_STREQ(backend.name(), "FE::MovingMesh::GeometryRegularizationMotionBackend");

    GeometryRegularizationOptions bad = backend.options();
    bad.max_linear_iterations = 0;
    EXPECT_THROW((void)GeometryRegularizationMotionBackend{bad}, std::invalid_argument);

    bad = backend.options();
    bad.weight_mode = GeometryRegularizationWeightMode::VertexField;
    bad.vertex_weight_field.clear();
    EXPECT_THROW((void)GeometryRegularizationMotionBackend{bad}, std::invalid_argument);

    for (const auto& name : svmp::FE::moving_mesh::geometry_regularization_option_names()) {
        EXPECT_FALSE(contains_forbidden_physics_term(name)) << name;
    }
    for (const auto& name : svmp::FE::moving_mesh::geometry_regularization_diagnostic_field_names()) {
        EXPECT_FALSE(contains_forbidden_physics_term(name)) << name;
    }
}

TEST(FEMeshMotionBackend, SolvesTwoDimensionalSquareHarmonicMotionDeterministically)
{
    auto mesh = make_square_fan_mesh();

    auto backend = std::make_shared<GeometryRegularizationMotionBackend>();
    svmp::motion::MeshMotion motion(*mesh);
    motion.set_backend(backend);
    motion.set_dirichlet_bcs({
        svmp::motion::MotionDirichletBC{
            INVALID_LABEL,
            [](const std::array<real_t, 3>& x, double, double) {
                return std::array<real_t, 3>{{0.2 * x[0], 0.0, 0.0}};
            },
            {{true, true, true}}},
    });

    ASSERT_TRUE(motion.advance(0.5));

    const auto& x_cur = mesh->local_mesh().X_cur();
    ASSERT_EQ(x_cur.size(), 10u);
    EXPECT_NEAR(x_cur[0], 0.0, 1.0e-12);
    EXPECT_NEAR(x_cur[2], 1.2, 1.0e-12);
    EXPECT_NEAR(x_cur[4], 1.2, 1.0e-12);
    EXPECT_NEAR(x_cur[6], 0.0, 1.0e-12);
    EXPECT_NEAR(x_cur[8], 0.6, 1.0e-12);
    EXPECT_NEAR(x_cur[9], 0.5, 1.0e-12);

    const auto handles = svmp::motion::attach_motion_fields(mesh->local_mesh(), 2);
    const auto* disp = svmp::MeshFields::field_data_as<real_t>(mesh->local_mesh(), handles.displacement);
    const auto* vel = svmp::MeshFields::field_data_as<real_t>(mesh->local_mesh(), handles.velocity);
    ASSERT_NE(disp, nullptr);
    ASSERT_NE(vel, nullptr);
    EXPECT_NEAR(disp[8], 0.1, 1.0e-12);
    EXPECT_NEAR(vel[8], 0.2, 1.0e-12);

    const auto& diagnostics = backend->last_diagnostics();
    EXPECT_TRUE(diagnostics.success);
    EXPECT_EQ(diagnostics.nonlinear_iterations, 0);
    EXPECT_LE(diagnostics.max_constraint_violation, backend->options().constraint_tolerance);
}

TEST(FEMeshMotionBackend, SupportsComponentWiseConstraintsAndRejectsConflicts)
{
    auto mesh = make_square_fan_mesh();
    GeometryRegularizationMotionBackend backend;
    svmp::motion::MotionConfig config;
    std::vector<real_t> displacement(mesh->n_vertices() * 2u, 0.0);
    std::vector<real_t> velocity(mesh->n_vertices() * 2u, 0.0);

    std::vector<svmp::motion::MotionDirichletBC> bcs = {
        {10,
         [](const std::array<real_t, 3>&, double, double) {
             return std::array<real_t, 3>{{0.0, 0.0, 0.0}};
         },
         {{true, false, false}}},
        {20,
         [](const std::array<real_t, 3>&, double, double) {
             return std::array<real_t, 3>{{0.2, 0.0, 0.0}};
         },
         {{true, false, false}}},
        {INVALID_LABEL,
         [](const std::array<real_t, 3>&, double, double) {
             return std::array<real_t, 3>{{0.0, 0.0, 0.0}};
         },
         {{false, true, false}}},
    };

    svmp::motion::MotionSolveRequest request{
        *mesh,
        config,
        1.0,
        1.0,
        Configuration::Reference,
        {displacement.data(), mesh->n_vertices(), 2u},
        {velocity.data(), mesh->n_vertices(), 2u},
        &bcs};

    const auto result = backend.solve(request);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_TRUE(result.wrote_velocity);
    EXPECT_GT(displacement[8], 0.0);
    EXPECT_LT(displacement[8], 0.2);
    EXPECT_NEAR(displacement[9], 0.0, 1.0e-12);

    bcs.push_back({10,
                   [](const std::array<real_t, 3>&, double, double) {
                       return std::array<real_t, 3>{{1.0, 0.0, 0.0}};
                   },
                   {{true, false, false}}});
    std::fill(displacement.begin(), displacement.end(), 0.0);
    const auto conflict = backend.solve(request);
    EXPECT_FALSE(conflict.success);
    EXPECT_NE(conflict.message.find("conflicting"), std::string::npos);
}

TEST(FEMeshMotionBackend, SolvesThreeDimensionalCubeBoundaryMotion)
{
    auto mesh = make_unit_cube_hex_mesh();
    auto backend = std::make_shared<GeometryRegularizationMotionBackend>();
    svmp::motion::MeshMotion motion(*mesh);
    motion.set_backend(backend);
    motion.set_dirichlet_bcs({
        {INVALID_LABEL,
         [](const std::array<real_t, 3>& x, double, double) {
             return std::array<real_t, 3>{{0.05 * x[0], 0.02 * x[1], 0.01 * x[2]}};
         },
         {{true, true, true}}},
    });

    ASSERT_TRUE(motion.advance(0.25));
    const auto& x_cur = mesh->local_mesh().X_cur();
    ASSERT_EQ(x_cur.size(), 24u);
    EXPECT_NEAR(x_cur[3 * 7 + 0], 1.05, 1.0e-12);
    EXPECT_NEAR(x_cur[3 * 7 + 1], 1.02, 1.0e-12);
    EXPECT_NEAR(x_cur[3 * 7 + 2], 1.01, 1.0e-12);
    EXPECT_TRUE(backend->last_diagnostics().success);
}

TEST(FEMeshMotionBackend, SpatialWeightModesAndPseudoElasticRegularizationReportDiagnostics)
{
    auto mesh = make_square_fan_mesh();
    auto& base = mesh->local_mesh();
    const auto weight = base.attach_field(EntityKind::Vertex, "mesh_quality_weight", svmp::FieldScalarType::Float64, 1);
    auto* weight_data = base.field_data_as<real_t>(weight);
    ASSERT_NE(weight_data, nullptr);
    for (std::size_t v = 0; v < base.n_vertices(); ++v) {
        weight_data[v] = 1.0 + static_cast<real_t>(v);
    }

    GeometryRegularizationOptions options;
    options.model = GeometryRegularizationModel::ArtificialPseudoElastic;
    options.weight_mode = GeometryRegularizationWeightMode::VertexField;
    options.vertex_weight_field = "mesh_quality_weight";
    GeometryRegularizationMotionBackend backend(options);

    svmp::motion::MotionConfig config;
    std::vector<real_t> displacement(mesh->n_vertices() * 2u, 0.0);
    std::vector<real_t> velocity(mesh->n_vertices() * 2u, 0.0);
    std::vector<svmp::motion::MotionDirichletBC> bcs = {
        {INVALID_LABEL,
         [](const std::array<real_t, 3>& x, double, double) {
             return std::array<real_t, 3>{{0.1 * x[0], 0.0, 0.0}};
         },
         {{true, true, true}}},
    };
    svmp::motion::MotionSolveRequest request{
        *mesh,
        config,
        1.0,
        1.0,
        Configuration::Reference,
        {displacement.data(), mesh->n_vertices(), 2u},
        {velocity.data(), mesh->n_vertices(), 2u},
        &bcs};

    const auto result = backend.solve(request);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_TRUE(backend.last_diagnostics().success);
    EXPECT_EQ(backend.last_diagnostics().model,
              "artificial_pseudo_elastic_geometry_regularization");
    EXPECT_EQ(backend.last_diagnostics().weight_mode,
              "vertex_field_artificial_weight");
}

TEST(FEMeshMotionBackend, RejectsUnderconstrainedMotionBeforeSolving)
{
    auto mesh = make_square_fan_mesh();
    GeometryRegularizationMotionBackend backend;
    svmp::motion::MotionConfig config;
    std::vector<real_t> displacement(mesh->n_vertices() * 2u, 0.0);
    std::vector<real_t> velocity(mesh->n_vertices() * 2u, 0.0);
    const std::vector<svmp::motion::MotionDirichletBC> bcs;

    svmp::motion::MotionSolveRequest request{
        *mesh,
        config,
        1.0,
        1.0,
        Configuration::Reference,
        {displacement.data(), mesh->n_vertices(), 2u},
        {velocity.data(), mesh->n_vertices(), 2u},
        &bcs};

    const auto result = backend.solve(request);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.message.find("underconstrained"), std::string::npos);
}

TEST(FEMeshMotionBackend, MeshMotionQualityGateRejectsInvertingUpdateAndRollsBack)
{
    auto mesh = make_unit_tet_mesh();
    auto backend = std::make_shared<GeometryRegularizationMotionBackend>();
    svmp::motion::MeshMotion motion(*mesh);
    svmp::motion::MotionConfig cfg;
    cfg.enable_quality_guard = true;
    cfg.enforce_quality_thresholds = true;
    cfg.max_substeps = 1;
    cfg.quality_min_jacobian = 0.0;
    motion.set_config(cfg);
    motion.set_backend(backend);
    motion.set_dirichlet_bcs({
        {INVALID_LABEL,
         [](const std::array<real_t, 3>& x, double, double step_scale) {
             const real_t dz = (x[2] > 0.5) ? static_cast<real_t>(-2.0 * step_scale) : 0.0;
             return std::array<real_t, 3>{{0.0, 0.0, dz}};
         },
         {{true, true, true}}},
    });

    EXPECT_FALSE(motion.advance(1.0));
    EXPECT_FALSE(mesh->local_mesh().has_current_coords());
    const auto& x_ref = mesh->local_mesh().X_ref();
    ASSERT_EQ(x_ref.size(), 12u);
    EXPECT_NEAR(x_ref[11], 1.0, 1.0e-12);
}
