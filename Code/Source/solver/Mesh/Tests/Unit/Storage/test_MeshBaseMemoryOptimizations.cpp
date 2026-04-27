#include "../../../Core/MeshBase.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace {

svmp::CellShape tetra_shape() {
  svmp::CellShape shape;
  shape.family = svmp::CellFamily::Tetra;
  shape.num_corners = 4;
  shape.order = 1;
  shape.is_mixed_order = false;
  return shape;
}

std::vector<svmp::index_t> sorted_face_vertices(const svmp::MeshBase& mesh, svmp::index_t face) {
  auto verts = mesh.face_vertices(face);
  std::sort(verts.begin(), verts.end());
  return verts;
}

} // namespace

TEST(MeshBaseMemoryOptimizations, BuildFromArraysMoveTransfersStorage) {
  svmp::MeshBase mesh;

  std::vector<svmp::real_t> coords = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::vector<svmp::offset_t> offsets = {0, 4};
  std::vector<svmp::index_t> connectivity = {0, 1, 2, 3};
  std::vector<svmp::CellShape> shapes = {tetra_shape()};

  const auto* coords_data = coords.data();
  const auto* offsets_data = offsets.data();
  const auto* connectivity_data = connectivity.data();
  const auto* shapes_data = shapes.data();

  mesh.build_from_arrays(3,
                         std::move(coords),
                         std::move(offsets),
                         std::move(connectivity),
                         std::move(shapes));

  EXPECT_EQ(mesh.X_ref().data(), coords_data);
  EXPECT_EQ(mesh.cell2vertex_offsets().data(), offsets_data);
  EXPECT_EQ(mesh.cell2vertex().data(), connectivity_data);
  EXPECT_EQ(mesh.cell_shapes().data(), shapes_data);
  EXPECT_EQ(mesh.n_vertices(), 4u);
  EXPECT_EQ(mesh.n_cells(), 1u);
}

TEST(MeshBaseMemoryOptimizations, SetFacesFromArraysMoveTransfersStorage) {
  svmp::MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<svmp::real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
      },
      std::vector<svmp::offset_t>{0, 4},
      std::vector<svmp::index_t>{0, 1, 2, 3},
      std::vector<svmp::CellShape>{tetra_shape()});

  svmp::CellShape tri;
  tri.family = svmp::CellFamily::Triangle;
  tri.num_corners = 3;
  tri.order = 1;
  tri.is_mixed_order = false;

  std::vector<svmp::CellShape> face_shapes(4, tri);
  std::vector<svmp::offset_t> face_offsets = {0, 3, 6, 9, 12};
  std::vector<svmp::index_t> face_connectivity = {
      1, 2, 3,
      0, 3, 2,
      0, 1, 3,
      0, 2, 1,
  };
  std::vector<std::array<svmp::index_t, 2>> face2cell = {
      std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX},
      std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX},
      std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX},
      std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX},
  };

  const auto* face_shapes_data = face_shapes.data();
  const auto* face_offsets_data = face_offsets.data();
  const auto* face_connectivity_data = face_connectivity.data();
  const auto* face2cell_data = face2cell.data();

  mesh.set_faces_from_arrays(std::move(face_shapes),
                             std::move(face_offsets),
                             std::move(face_connectivity),
                             std::move(face2cell));

  EXPECT_EQ(mesh.face_shapes().data(), face_shapes_data);
  EXPECT_EQ(mesh.face2vertex_offsets().data(), face_offsets_data);
  EXPECT_EQ(mesh.face2vertex().data(), face_connectivity_data);
  EXPECT_EQ(mesh.face2cell().data(), face2cell_data);
  EXPECT_EQ(mesh.n_faces(), 4u);
  EXPECT_EQ(mesh.cell_faces(0).size(), 4u);
}

TEST(MeshBaseMemoryOptimizations, CompactLinearTetraFinalizeBuildsSharedFaces) {
  svmp::MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<svmp::real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 1.0, 1.0,
      },
      std::vector<svmp::offset_t>{0, 4, 8},
      std::vector<svmp::index_t>{0, 1, 2, 3, 1, 2, 3, 4},
      std::vector<svmp::CellShape>{tetra_shape(), tetra_shape()});

  mesh.finalize();

  EXPECT_EQ(mesh.n_faces(), 7u);
  EXPECT_EQ(mesh.cell_faces(0).size(), 4u);
  EXPECT_EQ(mesh.cell_faces(1).size(), 4u);

  int shared_face_count = 0;
  bool found_expected_shared_face = false;
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
    const auto& cells = mesh.face2cell()[static_cast<size_t>(f)];
    if (cells[0] != svmp::INVALID_INDEX && cells[1] != svmp::INVALID_INDEX) {
      ++shared_face_count;
      found_expected_shared_face =
          found_expected_shared_face ||
          (sorted_face_vertices(mesh, f) == std::vector<svmp::index_t>{1, 2, 3});
    }
  }

  EXPECT_EQ(shared_face_count, 1);
  EXPECT_TRUE(found_expected_shared_face);
}

TEST(MeshBaseMemoryOptimizations, FinalizeCanSkipCodim1AndEdgeStorage) {
  svmp::MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<svmp::real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 1.0, 1.0,
      },
      std::vector<svmp::offset_t>{0, 4, 8},
      std::vector<svmp::index_t>{0, 1, 2, 3, 1, 2, 3, 4},
      std::vector<svmp::CellShape>{tetra_shape(), tetra_shape()});

  svmp::MeshFinalizeOptions options;
  options.codim1_storage = svmp::MeshCodim1StorageMode::None;
  options.edge_storage = false;
  mesh.finalize(options);

  EXPECT_EQ(mesh.n_cells(), 2u);
  EXPECT_EQ(mesh.n_faces(), 0u);
  EXPECT_EQ(mesh.n_edges(), 0u);
  EXPECT_EQ(mesh.codim1_storage_mode(), svmp::MeshCodim1StorageMode::None);
}

TEST(MeshBaseMemoryOptimizations, FinalizeBoundaryOnlySkipsInteriorFaces) {
  svmp::MeshBase mesh;
  mesh.build_from_arrays(
      3,
      std::vector<svmp::real_t>{
          0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 1.0, 1.0,
      },
      std::vector<svmp::offset_t>{0, 4, 8},
      std::vector<svmp::index_t>{0, 1, 2, 3, 1, 2, 3, 4},
      std::vector<svmp::CellShape>{tetra_shape(), tetra_shape()});

  svmp::MeshFinalizeOptions options;
  options.codim1_storage = svmp::MeshCodim1StorageMode::BoundaryOnly;
  options.edge_storage = false;
  mesh.finalize(options);

  EXPECT_EQ(mesh.n_faces(), 6u);
  EXPECT_EQ(mesh.codim1_storage_mode(), svmp::MeshCodim1StorageMode::BoundaryOnly);

  bool found_interior_face = false;
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
    const auto& cells = mesh.face2cell()[static_cast<size_t>(f)];
    EXPECT_TRUE(cells[0] == svmp::INVALID_INDEX || cells[1] == svmp::INVALID_INDEX);
    found_interior_face =
        found_interior_face ||
        (sorted_face_vertices(mesh, f) == std::vector<svmp::index_t>{1, 2, 3});
  }
  EXPECT_FALSE(found_interior_face);
}
