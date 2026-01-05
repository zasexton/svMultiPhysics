/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_MeshPublicApi.cpp
 * @brief Compile/link smoke test for the public Mesh entry point.
 *
 * This is intentionally lightweight: it should build and run in both serial
 * and MPI configurations without needing to launch under mpiexec.
 */

#include "Mesh.h"

#include <cassert>
#include <vector>

int main() {
  using namespace svmp;

  // Construction and factories must compile in all builds.
  Mesh mesh;
  auto m0 = create_mesh();
  auto m1 = create_mesh(MeshComm{});
  assert(m0 && m1);

  // Minimal local build.
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
  shapes[0].num_corners = 4;
  shapes[0].order = 1;

  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  mesh.finalize();

  assert(mesh.dim() == 3);
  assert(mesh.n_vertices() == 4);
  assert(mesh.n_cells() == 1);

  // Partition lifecycle entry point must remain available in all builds.
  mesh.build_from_arrays_global_and_partition(3, X_ref, offs, conn, shapes);
  mesh.finalize();

  // MeshBase-like surface: labels/sets and fields.
  mesh.set_region_label(0, 1);
  assert(mesh.region_label(0) == 1);
  mesh.add_to_set(EntityKind::Volume, "cellset", 0);
  assert(mesh.has_set(EntityKind::Volume, "cellset"));

  const auto f = mesh.attach_field(EntityKind::Vertex, "v", FieldScalarType::Float64, 3);
  assert(mesh.field_components(f) == 3);
  assert(mesh.field_data_as<real_t>(f) != nullptr);

  (void)mesh.event_bus();

  return 0;
}

