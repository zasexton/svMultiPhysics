/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"

#include "Mesh.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace svmp::test {

static std::string unique_vtu_name() {
  const auto stamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return "svmp_mesh_roundtrip_" + std::to_string(static_cast<long long>(stamp)) + ".vtu";
}

TEST(MeshLoadSaveRoundtrip, VtuRoundtrip) {
  // Unit/IO tests are only enabled when VTK is available.
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

  const std::string filename = unique_vtu_name();
  save_mesh(mesh, filename);

  auto loaded = load_mesh(filename);
  ASSERT_TRUE(loaded != nullptr);

  EXPECT_EQ(loaded->n_cells(), mesh.n_cells());
  EXPECT_EQ(loaded->n_vertices(), mesh.n_vertices());
  EXPECT_EQ(loaded->dim(), mesh.dim());

  std::error_code ec;
  std::filesystem::remove(filename, ec);
}

} // namespace svmp::test

