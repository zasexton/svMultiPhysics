/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Mesh.h"

#include <cassert>

int main() {
  using namespace svmp;

  // Ensure the public entry point compiles and the common MeshBase-like surface
  // is available on svmp::Mesh in both serial and MPI builds.
  Mesh mesh;

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

  mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
  mesh.finalize();

  assert(mesh.dim() == 3);
  assert(mesh.n_vertices() == 4);
  assert(mesh.n_cells() == 1);

  // Labels/sets
  mesh.set_region_label(0, 7);
  assert(mesh.region_label(0) == 7);
  mesh.add_to_set(EntityKind::Volume, "region7_cells", 0);
  assert(mesh.has_set(EntityKind::Volume, "region7_cells"));

  // Fields
  const auto vh = mesh.attach_field(EntityKind::Vertex, "velocity", FieldScalarType::Float64, 3);
  assert(mesh.field_components(vh) == 3);
  auto* vdata = mesh.field_data_as<real_t>(vh);
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    vdata[static_cast<size_t>(v) * 3 + 0] = 1.0;
    vdata[static_cast<size_t>(v) * 3 + 1] = 2.0;
    vdata[static_cast<size_t>(v) * 3 + 2] = 3.0;
  }

  // Event bus forwarding exists (no behavioral assertion beyond compile/link).
  (void)mesh.event_bus();

  // Search API: local + distributed entry points compile in all builds.
  const std::array<real_t, 3> p = {0.1, 0.1, 0.1};
  const auto loc_local = MeshSearch::locate_point(mesh.local_mesh(), p);
  assert(loc_local.found);
  (void)MeshSearch::contains_point_global(mesh, p);
  (void)MeshSearch::locate_point_global(mesh, p);

  // Factories compile in all builds.
  auto m0 = create_mesh();
  auto m1 = create_mesh(MeshComm{});
  auto m2 = create_mesh(std::make_shared<MeshBase>());
  auto m3 = create_mesh(std::make_shared<MeshBase>(), MeshComm{});
  assert(m0 && m1 && m2 && m3);

  return 0;
}
