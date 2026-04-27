#include "Core/MeshBase.h"
#include "Search/CutCell.h"

#include <gtest/gtest.h>

#include <vector>

using namespace svmp;
using namespace svmp::search;

namespace {

MeshBase make_tetra_mesh()
{
  MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0},
      std::vector<offset_t>{0, 4},
      std::vector<index_t>{0, 1, 2, 3},
      std::vector<CellShape>{CellShape{CellFamily::Tetra, 4, 1}});
  mesh.finalize();
  return mesh;
}

EmbeddedGeometryDescriptor plane(real_t x, std::uint64_t epoch = 1)
{
  EmbeddedGeometryDescriptor embedded;
  embedded.kind = EmbeddedGeometryKind::Plane;
  embedded.origin = {{x, 0.0, 0.0}};
  embedded.normal = {{1.0, 0.0, 0.0}};
  embedded.geometry_epoch = epoch;
  embedded.provenance.persistent_id = "embedded-plane";
  embedded.provenance.name = "plane";
  embedded.provenance.provenance_epoch = epoch;
  return embedded;
}

} // namespace

TEST(CutCell, ClassifiesStaticPlaneAndSphereCuts)
{
  auto mesh = make_tetra_mesh();

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = true;

  const auto plane_map = classify_embedded_geometry(mesh, plane(0.5), options);
  ASSERT_EQ(plane_map.cells.size(), 1u);
  EXPECT_EQ(plane_map.cells[0].classification, CutClassification::Cut);
  EXPECT_FALSE(plane_map.cells[0].intersections.empty());
  EXPECT_TRUE(plane_map.valid_for(mesh));

  EmbeddedGeometryDescriptor sphere;
  sphere.kind = EmbeddedGeometryKind::Sphere;
  sphere.origin = {{0.0, 0.0, 0.0}};
  sphere.radius = 0.5;
  sphere.geometry_epoch = 2;
  sphere.provenance.persistent_id = "embedded-sphere";
  const auto sphere_map = classify_embedded_geometry(mesh, sphere, options);
  ASSERT_EQ(sphere_map.cells.size(), 1u);
  EXPECT_EQ(sphere_map.cells[0].classification, CutClassification::Cut);
}

TEST(CutCell, MovingEmbeddedGeometryTransactionCanRollbackAndAccept)
{
  auto mesh = make_tetra_mesh();
  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  auto map = classify_embedded_geometry(mesh, plane(0.5, 1), options);
  map.accept_trial();
  ASSERT_EQ(map.state, CutClassificationState::Committed);
  ASSERT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutClassificationTransaction tx(map);
  tx.stage(classify_embedded_geometry(mesh, plane(2.0, 2), options));
  ASSERT_EQ(map.cells[0].classification, CutClassification::Negative);
  tx.rollback();
  EXPECT_EQ(tx.state(), CutClassificationState::RolledBack);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Cut);

  CutClassificationTransaction tx2(map);
  tx2.stage(classify_embedded_geometry(mesh, plane(-1.0, 3), options));
  tx2.accept();
  EXPECT_EQ(map.state, CutClassificationState::Committed);
  EXPECT_EQ(map.cells[0].classification, CutClassification::Positive);
}

TEST(CutCell, KinematicConstraintProvenanceAndRestartMetadataArePreserved)
{
  auto mesh = make_tetra_mesh();
  auto embedded = plane(0.25, 7);

  EmbeddedKinematicConstraint constraint;
  constraint.kind = EmbeddedKinematicConstraintKind::RelationMap;
  constraint.id = "plane-relation";
  constraint.source_geometry_id = "embedded-plane";
  constraint.relation_map_id = "plane-map";
  constraint.constraint_epoch = 9;
  constraint.provenance.persistent_id = "embedded-plane";
  constraint.source_revision = EmbeddedRevisionSnapshot::capture(
      mesh, Configuration::Reference, embedded.geometry_epoch, constraint.constraint_epoch, 3);

  CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;
  options.fe_layout_revision = 3;
  options.kinematic_constraints.push_back(constraint);

  auto map = classify_embedded_geometry(mesh, embedded, options);
  ASSERT_EQ(map.kinematic_constraints.size(), 1u);
  EXPECT_EQ(map.kinematic_constraints[0].relation_map_id, "plane-map");
  EXPECT_TRUE(map.valid_for(mesh));

  const auto restart = make_cut_classification_restart_record(map);
  EXPECT_EQ(restart.provenance.persistent_id, "embedded-plane");
  EXPECT_EQ(restart.embedded_geometry_epoch, 7u);
  EXPECT_EQ(restart.embedded_constraint_epoch, 9u);
  EXPECT_EQ(restart.fe_layout_revision, 3u);
  EXPECT_EQ(restart.cut_cell_count, 1u);
}
