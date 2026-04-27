#include "Core/DistributedMesh.h"
#include "Core/MeshBase.h"
#include "IO/MovingMeshRestart.h"
#include "Motion/IMotionBackend.h"
#include "Motion/MeshMotion.h"
#include "Topology/CellShape.h"
#include "Validation/MovingGeometryValidity.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace svmp;

namespace {

constexpr label_t kA = 11;
constexpr label_t kB = 12;
constexpr label_t kC = 13;

CellShape quad_shape(int order = 1, int corners = 4)
{
  return {CellFamily::Quad, corners, order};
}

CellShape line_shape()
{
  return {CellFamily::Line, 2, 1};
}

MeshBase make_two_crossing_quads()
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 1.0, 0.0,

      0.5, -0.25, -0.5,
      0.5,  1.25, -0.5,
      0.5,  1.25,  0.5,
      0.5, -0.25,  0.5,

      5.0, 0.0, 0.0,
      6.0, 0.0, 0.0,
      6.0, 1.0, 0.0,
      5.0, 1.0, 0.0};
  mesh.build_from_arrays(3, x, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape(), quad_shape(), quad_shape()},
                             {0, 4, 8, 12},
                             {0, 1, 2, 3,
                              4, 5, 6, 7,
                              8, 9, 10, 11},
                             {{{INVALID_INDEX, INVALID_INDEX}},
                              {{INVALID_INDEX, INVALID_INDEX}},
                              {{INVALID_INDEX, INVALID_INDEX}}});
  mesh.set_face_gids({100, 101, 102});
  mesh.set_boundary_label(0, kA);
  mesh.set_boundary_label(1, kA);
  mesh.set_boundary_label(2, kC);
  return mesh;
}

MeshBase make_parallel_quads(real_t gap)
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 1.0, 1.0,
      0.0, 0.0, 1.0,

      gap, 0.0, 0.0,
      gap, 1.0, 0.0,
      gap, 1.0, 1.0,
      gap, 0.0, 1.0};
  mesh.build_from_arrays(3, x, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape(), quad_shape()}, {0, 4, 8},
                             {0, 1, 2, 3,
                              4, 7, 6, 5},
                             {{{INVALID_INDEX, INVALID_INDEX}},
                              {{INVALID_INDEX, INVALID_INDEX}}});
  mesh.set_face_gids({200, 201});
  mesh.set_boundary_label(0, kA);
  mesh.set_boundary_label(1, kB);
  return mesh;
}

MeshBase make_flipped_quad_current()
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 1.0, 0.0};
  mesh.build_from_arrays(3, x, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape()}, {0, 4}, {0, 1, 2, 3},
                             {{{INVALID_INDEX, INVALID_INDEX}}});
  mesh.set_face_gids({300});
  mesh.set_boundary_label(0, kA);
  mesh.set_current_coords({
      0.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      1.0, 1.0, 0.0,
      1.0, 0.0, 0.0});
  return mesh;
}

MeshBase make_2d_crossing_segments()
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      1.0, 0.0};
  mesh.build_from_arrays(2, x, {0}, {}, {});
  mesh.set_edges_from_arrays({{{0, 1}}, {{2, 3}}});
  mesh.set_edge_gids({400, 401});
  mesh.set_edge_label(0, kA);
  mesh.set_edge_label(1, kA);
  return mesh;
}

MeshBase make_high_order_face()
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 1.0, 0.0,
      0.5, 0.5, 0.25};
  mesh.build_from_arrays(3, x, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape(2, 4)}, {0, 5}, {0, 1, 2, 3, 4},
                             {{{INVALID_INDEX, INVALID_INDEX}}});
  mesh.set_face_gids({500});
  mesh.set_boundary_label(0, kA);
  return mesh;
}

MeshBase make_swept_vertex_face()
{
  MeshBase mesh;
  const std::vector<real_t> x = {
      0.0, 0.5, 0.5,
      0.5, 0.0, 0.0,
      0.5, 1.0, 0.0,
      0.5, 1.0, 1.0,
      0.5, 0.0, 1.0};
  mesh.build_from_arrays(3, x, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape()}, {0, 4}, {1, 2, 3, 4},
                             {{{INVALID_INDEX, INVALID_INDEX}}});
  mesh.set_face_gids({600});
  mesh.set_boundary_label(0, kA);
  mesh.set_current_coords({
      1.0, 0.5, 0.5,
      0.5, 0.0, 0.0,
      0.5, 1.0, 0.0,
      0.5, 1.0, 1.0,
      0.5, 0.0, 1.0});
  return mesh;
}

class UniformXBackend final : public motion::IMotionBackend {
public:
  const char* name() const noexcept override { return "UniformXBackend"; }

  motion::MotionSolveResult solve(const motion::MotionSolveRequest& request) override
  {
    motion::MotionSolveResult result;
    if (!request.displacement.valid()) {
      result.success = false;
      return result;
    }
    for (std::size_t i = 0; i < request.displacement.n_entities; ++i) {
      request.displacement.data[i * request.displacement.components] = 1.0;
      if (request.displacement.components > 1) {
        request.displacement.data[i * request.displacement.components + 1] = 0.0;
      }
      if (request.displacement.components > 2) {
        request.displacement.data[i * request.displacement.components + 2] = 0.0;
      }
    }
    result.success = true;
    return result;
  }
};

} // namespace

TEST(MovingGeometryValidityTest, StructuredPolicyReportsAndRestartMetadataAreAuditable)
{
  auto mesh = make_parallel_quads(0.2);
  auto xcur = mesh.X_ref();
  xcur[0] = std::numeric_limits<real_t>::quiet_NaN();
  mesh.set_current_coords(xcur);

  auto policy = validation::MovingGeometryValidity::ale_basic_policy();
  policy.configuration = Configuration::Current;
  policy.time_level = 2.5;
  auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);

  ASSERT_FALSE(report.passed);
  ASSERT_FALSE(report.failures.empty());
  const auto& failure = report.failures.front();
  EXPECT_EQ(failure.check_name, "DegenerateBoundary");
  EXPECT_EQ(failure.recommended_action, validation::ValidityAction::Reject);
  EXPECT_EQ(failure.configuration, Configuration::Current);
  EXPECT_EQ(failure.time_level, 2.5);
  EXPECT_GE(failure.revision_state.geometry, 1u);
  EXPECT_FALSE(failure.global_ids.empty());
  EXPECT_FALSE(failure.labels.empty());

  const auto policy_meta = policy.restart_metadata();
  EXPECT_EQ(policy_meta.at("policy_group"), "ALEBasic");
  const auto report_meta = report.restart_metadata();
  EXPECT_EQ(report_meta.at("passed"), "false");
  EXPECT_EQ(report_meta.at("failure.0.check"), "DegenerateBoundary");
}

TEST(MovingGeometryValidityTest, SelfIntersectionAndLabelScopedBroadPhaseAreDetected)
{
  auto mesh = make_two_crossing_quads();
  auto policy = validation::MovingGeometryValidity::preset(validation::ValidityPolicyGroup::Contact);
  policy.checks.clear();
  auto spec = validation::MovingGeometryCheckSpec{};
  spec.check = validation::MovingGeometryCheck::BoundarySelfIntersection;
  spec.name = "label-scoped-self-intersection";
  spec.labels.insert(kA);
  spec.action = validation::ValidityAction::Reject;
  policy.checks.push_back(spec);

  const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
  EXPECT_FALSE(report.passed);
  EXPECT_TRUE(report.requires_rejection());
  ASSERT_FALSE(report.failures.empty());
  EXPECT_EQ(report.failures.front().global_ids.size(), 2u);
  EXPECT_LT(report.exact_candidate_pairs, 3u);
}

TEST(MovingGeometryValidityTest, MinimumContactShellAndBoundaryLayerSeparationUseLabelPairs)
{
  auto mesh = make_parallel_quads(0.05);
  auto policy = validation::MovingGeometryValidity::preset(validation::ValidityPolicyGroup::Shell);
  policy.checks.clear();
  for (auto check : {validation::MovingGeometryCheck::MinimumSeparation,
                     validation::MovingGeometryCheck::ContactSeparation,
                     validation::MovingGeometryCheck::ShellThicknessSeparation,
                     validation::MovingGeometryCheck::BoundaryLayer}) {
    auto spec = validation::MovingGeometryCheckSpec{};
    spec.check = check;
    spec.name = validation::MovingGeometryValidity::check_name(check);
    spec.threshold = 0.10;
    spec.action = validation::ValidityAction::Reject;
    spec.label_pairs.push_back({kA, kB, true});
    policy.checks.push_back(spec);
  }

  const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
  EXPECT_FALSE(report.passed);
  EXPECT_GE(report.failures.size(), 3u);
  for (const auto& failure : report.failures) {
    EXPECT_EQ(failure.labels.size(), 2u);
    EXPECT_NEAR(failure.threshold, 0.10, 1.0e-12);
  }
}

TEST(MovingGeometryValidityTest, OrientationHighOrderSweptAndTwoDimensionalChecksAreCovered)
{
  {
    auto mesh = make_flipped_quad_current();
    auto policy = validation::MovingGeometryValidity::ale_basic_policy();
    policy.configuration = Configuration::Current;
    const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
    EXPECT_FALSE(report.passed);
    EXPECT_TRUE(std::any_of(report.failures.begin(), report.failures.end(), [](const auto& f) {
      return f.check_name == "SurfaceFolding" || f.check_name == "NormalOrientation";
    }));
  }

  {
    auto mesh = make_high_order_face();
    auto policy = validation::MovingGeometryValidity::contact_policy();
    policy.checks.clear();
    auto spec = validation::MovingGeometryCheckSpec{};
    spec.check = validation::MovingGeometryCheck::CurvedBoundarySampling;
    spec.name = "curved-contact-sampling";
    spec.threshold = 0.1;
    spec.action = validation::ValidityAction::Reject;
    policy.checks.push_back(spec);
    const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
    EXPECT_FALSE(report.passed);
    ASSERT_FALSE(report.failures.empty());
    EXPECT_EQ(report.failures.front().check_name, "curved-contact-sampling");
  }

  {
    auto mesh = make_swept_vertex_face();
    auto policy = validation::MovingGeometryValidity::large_step_policy();
    policy.configuration = Configuration::Current;
    policy.checks.clear();
    policy.checks.push_back({validation::MovingGeometryCheck::SweptVolume,
                             true,
                             "swept-volume",
                             validation::ValiditySeverity::Error,
                             validation::ValidityAction::Backtrack});
    const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
    EXPECT_FALSE(report.passed);
    EXPECT_TRUE(report.recommends_backtrack());
  }

  {
    auto mesh = make_2d_crossing_segments();
    auto policy = validation::MovingGeometryValidity::ale_basic_policy();
    policy.checks.clear();
    policy.checks.push_back({validation::MovingGeometryCheck::TwoDBoundary,
                             true,
                             "2d-boundary",
                             validation::ValiditySeverity::Error,
                             validation::ValidityAction::Reject});
    const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
    EXPECT_FALSE(report.passed);
    EXPECT_EQ(report.failures.front().entity_kind, EntityKind::Edge);
  }
}

TEST(MovingGeometryValidityTest, GenericMotionConstraintsRemainPhysicsNeutral)
{
  MeshBase mesh;
  mesh.build_from_arrays(3, {0.0, 0.0, 0.0}, {0}, {}, {});
  mesh.set_vertex_gids({700});
  mesh.set_vertex_label(0, kA);
  mesh.set_current_coords({0.25, 0.0, 0.2});

  validation::MovingGeometryValidityPolicy policy;
  policy.group_name = "constraints";
  policy.configuration = Configuration::Current;
  policy.constraints.push_back({validation::MotionConstraintKind::ManifoldPlane,
                                "plane-following",
                                kA,
                                {},
                                {{0.0, 0.0, 0.0}},
                                {{0.0, 0.0, 1.0}},
                                1.0e-6,
                                validation::ValidityAction::Constrain,
                                true});
  policy.constraints.push_back({validation::MotionConstraintKind::TangentialOnly,
                                "tangential-only",
                                kA,
                                {},
                                {{0.0, 0.0, 0.0}},
                                {{0.0, 0.0, 1.0}},
                                1.0e-6,
                                validation::ValidityAction::Backtrack,
                                true});
  policy.constraints.push_back({validation::MotionConstraintKind::MaximumDisplacement,
                                "bounded-displacement",
                                kA,
                                {},
                                {{0.0, 0.0, 0.0}},
                                {{1.0, 0.0, 0.0}},
                                0.1,
                                validation::ValidityAction::Backtrack,
                                true});

  const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
  EXPECT_FALSE(report.passed);
  EXPECT_TRUE(report.provides_constraints());
  EXPECT_TRUE(report.recommends_backtrack());
}

TEST(MovingGeometryValidityTest, MeshMotionRollsBackValidityRejectedSubsteps)
{
  Mesh mesh;
  mesh.build_from_arrays(3, {0.0, 0.0, 0.0}, {0}, {}, {});
  mesh.set_vertex_gids({800});

  motion::MeshMotion motion(mesh);
  motion.set_backend(std::make_shared<UniformXBackend>());
  auto cfg = motion.config();
  cfg.enable_quality_guard = false;
  cfg.enable_validity_guard = true;
  cfg.max_substeps = 1;
  cfg.validity_policy = {};
  cfg.validity_policy.group_name = "bounded-motion";
  cfg.validity_policy.configuration = Configuration::Current;
  cfg.validity_policy.constraints.push_back({validation::MotionConstraintKind::MaximumDisplacement,
                                             "bounded-displacement",
                                             INVALID_LABEL,
                                             {},
                                             {{0.0, 0.0, 0.0}},
                                             {{1.0, 0.0, 0.0}},
                                             0.25,
                                             validation::ValidityAction::Reject,
                                             true});
  motion.set_config(cfg);

  EXPECT_FALSE(motion.advance(1.0));
  EXPECT_FALSE(mesh.has_current_coords());
  EXPECT_NE(motion.last_error().find("bounded-displacement"), std::string::npos);
}

TEST(MovingGeometryValidityTest, RestartMetadataPersistsValidityDiagnostics)
{
  auto mesh = make_parallel_quads(0.05);
  auto policy = validation::MovingGeometryValidity::contact_policy();
  policy.checks.clear();
  auto spec = validation::MovingGeometryCheckSpec{};
  spec.check = validation::MovingGeometryCheck::MinimumSeparation;
  spec.name = "restart-min-separation";
  spec.threshold = 0.10;
  spec.label_pairs.push_back({kA, kB, true});
  policy.checks.push_back(spec);
  const auto report = validation::MovingGeometryValidity::evaluate(mesh, policy);
  ASSERT_FALSE(report.passed);

  moving_mesh_restart::WriteOptions options;
  options.include_fields = false;
  options.moving_geometry_validity_state = report.restart_metadata();
  const std::string path = "moving_geometry_validity_restart_test.svmmr";
  moving_mesh_restart::write(mesh, path, options);

  const auto metadata = moving_mesh_restart::inspect(path);
  EXPECT_EQ(metadata.moving_geometry_validity_state.at("passed"), "false");
  EXPECT_EQ(metadata.moving_geometry_validity_state.at("failure.0.check"), "restart-min-separation");
  std::remove(path.c_str());
}

