/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Core/DistributedMesh.h"
#include "Observer/ObserverRegistry.h"
#include "Topology/CellShape.h"

namespace svmp {
namespace test {

TEST(DistributedMeshStubEventsTest, EmitsPartitionAndFieldEvents) {
#if defined(MESH_HAS_MPI)
  GTEST_SKIP() << "DistributedMesh stub events are only applicable when MPI is disabled";
#else
  DistributedMesh dm;

  // Build a minimal mesh (one tet).
  std::vector<real_t> X_ref = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::vector<offset_t> offs = {0, 4};
  std::vector<index_t> conn = {0, 1, 2, 3};
  std::vector<CellShape> shapes(1);
  shapes[0].family = CellFamily::Tetra;
  shapes[0].order = 1;
  shapes[0].num_corners = 4;
  dm.build_from_arrays(3, X_ref, offs, conn, shapes);

  auto counter = ObserverRegistry::attach_event_counter(dm.local_mesh());
  counter->reset();

  dm.set_ownership(0, EntityKind::Volume, Ownership::Owned, 0);
  EXPECT_EQ(counter->count(MeshEvent::PartitionChanged), 1u);

  counter->reset();
  dm.build_ghost_layer(1);
  EXPECT_EQ(counter->count(MeshEvent::PartitionChanged), 1u);

  counter->reset();
  dm.clear_ghosts();
  EXPECT_EQ(counter->count(MeshEvent::PartitionChanged), 1u);

  counter->reset();
  dm.update_ghosts({});
  EXPECT_EQ(counter->count(MeshEvent::FieldsChanged), 1u);
#endif
}

} // namespace test
} // namespace svmp

