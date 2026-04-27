#include <gtest/gtest.h>

#include "Constraints/MovingConstraintMetadata.h"
#include "Constraints/HangingVertexConstraints.h"
#include "Core/MeshBase.h"
#include "Mesh.h"
#include "Motion/MeshMotion.h"

namespace {

svmp::CellShape line_shape()
{
  svmp::CellShape s{};
  s.family = svmp::CellFamily::Line;
  s.num_corners = 2;
  s.order = 1;
  return s;
}

svmp::CellShape quad_shape()
{
  svmp::CellShape s{};
  s.family = svmp::CellFamily::Quad;
  s.num_corners = 4;
  s.order = 1;
  return s;
}

std::shared_ptr<svmp::Mesh> build_labeled_quad()
{
  auto base = std::make_shared<svmp::MeshBase>();
  base->build_from_arrays(
      2,
      std::vector<svmp::real_t>{0.0, 0.0,
                                1.0, 0.0,
                                1.0, 1.0,
                                0.0, 1.0},
      std::vector<svmp::offset_t>{0, 4},
      std::vector<svmp::index_t>{0, 1, 2, 3},
      std::vector<svmp::CellShape>{quad_shape()});
  base->set_faces_from_arrays(
      std::vector<svmp::CellShape>{line_shape(), line_shape(), line_shape(), line_shape()},
      std::vector<svmp::offset_t>{0, 2, 4, 6, 8},
      std::vector<svmp::index_t>{0, 1, 1, 2, 2, 3, 3, 0},
      std::vector<std::array<svmp::index_t, 2>>{
          {0, svmp::INVALID_INDEX},
          {0, svmp::INVALID_INDEX},
          {0, svmp::INVALID_INDEX},
          {0, svmp::INVALID_INDEX}});
  base->set_boundary_label(1, 20); // right
  base->set_boundary_label(3, 10); // left
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(MovingConstraintMetadata, RevisionDependenciesAreAuditable)
{
  auto mesh = build_labeled_quad();
  auto before = svmp::constraints::MeshConstraintRevisionSnapshot::capture(mesh->local_mesh());

  auto deps = svmp::constraints::MeshConstraintRevisionDependencies::moving_boundary_relation();
  EXPECT_FALSE(svmp::constraints::dependency_changed(deps, before, before));

  auto cur = mesh->local_mesh().X_ref();
  cur[0] += 0.1;
  mesh->local_mesh().set_current_coords(cur);

  auto after = svmp::constraints::MeshConstraintRevisionSnapshot::capture(mesh->local_mesh());
  EXPECT_TRUE(svmp::constraints::dependency_changed(deps, before, after));
}

TEST(MovingConstraintMetadata, PrescribedMotionValidationRejectsIncompatiblePeriodicPair)
{
  auto mesh = build_labeled_quad();

  svmp::constraints::MovingMeshConstraintRegistry registry;
  registry.add(svmp::constraints::make_periodic_boundary_metadata(
      "left-right", 10, 20, {{1.0, 0.0, 0.0}}));

  std::vector<svmp::motion::MotionDirichletBC> bcs;
  svmp::motion::MotionDirichletBC left;
  left.boundary_label = 10;
  left.value = [](const std::array<svmp::real_t, 3>&, double, double) {
    return std::array<svmp::real_t, 3>{{0.1, 0.0, 0.0}};
  };
  svmp::motion::MotionDirichletBC right = left;
  right.boundary_label = 20;
  right.value = [](const std::array<svmp::real_t, 3>&, double, double) {
    return std::array<svmp::real_t, 3>{{0.2, 0.0, 0.0}};
  };
  bcs = {left, right};

  const auto result =
      registry.validate_prescribed_motion(mesh->local_mesh(), bcs, 1.0, 1.0, 1e-12);
  EXPECT_FALSE(result.ok);
  EXPECT_NE(result.message.find("incompatible prescribed motion"), std::string::npos);

  svmp::motion::MeshMotion motion(*mesh);
  motion.set_constraint_metadata(registry);
  motion.set_dirichlet_bcs(std::move(bcs));
  EXPECT_FALSE(motion.advance(1.0));
  EXPECT_NE(motion.last_error().find("incompatible prescribed motion"), std::string::npos);
}

TEST(MovingConstraintMetadata, HangingContinuityChecksMovedConfiguration)
{
  svmp::MeshBase mesh;
  mesh.build_from_arrays(
      1,
      std::vector<svmp::real_t>{0.0, 0.5, 1.0},
      std::vector<svmp::offset_t>{0, 2, 4},
      std::vector<svmp::index_t>{0, 1, 1, 2},
      std::vector<svmp::CellShape>{line_shape(), line_shape()});
  mesh.finalize();

  svmp::HangingVertexConstraint c;
  c.constrained_vertex = 1;
  c.parent_type = svmp::ConstraintParentType::Edge;
  c.parent_vertices = {0, 2};
  c.weights = {0.5, 0.5};
  c.refinement_level = 1;

  svmp::HangingVertexConstraints constraints;
  ASSERT_TRUE(constraints.add_constraint(c));

  std::string message;
  EXPECT_TRUE(constraints.validate_geometric_continuity(
      mesh, svmp::Configuration::Reference, 1e-12, &message));

  mesh.set_current_coords(std::vector<svmp::real_t>{0.0, 0.6, 1.0});
  EXPECT_FALSE(constraints.validate_geometric_continuity(
      mesh, svmp::Configuration::Current, 1e-12, &message));
  EXPECT_NE(message.find("geometric continuity violation"), std::string::npos);
}
