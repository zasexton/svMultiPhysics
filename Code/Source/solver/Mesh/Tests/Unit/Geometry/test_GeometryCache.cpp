/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "gtest/gtest.h"
#include "Geometry/GeometryCache.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

#include <cmath>

namespace svmp {
namespace test {

class GeometryCacheTest : public ::testing::Test {
protected:
  static constexpr real_t tol = 1e-12;

  static bool approx(real_t a, real_t b, real_t t = tol) { return std::abs(a - b) < t; }
  static bool approx3(const std::array<real_t, 3>& a, const std::array<real_t, 3>& b, real_t t = tol) {
    return approx(a[0], b[0], t) && approx(a[1], b[1], t) && approx(a[2], b[2], t);
  }

  static MeshBase create_unit_tet_mesh() {
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // Vertex 0
        1.0, 0.0, 0.0,  // Vertex 1
        0.0, 1.0, 0.0,  // Vertex 2
        0.0, 0.0, 1.0   // Vertex 3
    };

    std::vector<offset_t> offs = {0, 4};
    std::vector<index_t> conn = {0, 1, 2, 3};
    std::vector<CellShape> shapes(1);
    shapes[0].family = CellFamily::Tetra;
    shapes[0].order = 1;
    shapes[0].num_corners = 4;

    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
    return mesh;
  }

  static MeshBase create_unit_hex_mesh(bool finalize_topology = true) {
    std::vector<real_t> X_ref = {
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        1.0, 1.0, 0.0,  // 2
        0.0, 1.0, 0.0,  // 3
        0.0, 0.0, 1.0,  // 4
        1.0, 0.0, 1.0,  // 5
        1.0, 1.0, 1.0,  // 6
        0.0, 1.0, 1.0   // 7
    };

    std::vector<offset_t> offs = {0, 8};
    std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<CellShape> shapes(1);
    shapes[0].family = CellFamily::Hex;
    shapes[0].order = 1;
    shapes[0].num_corners = 8;

    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
    if (finalize_topology) {
      mesh.finalize();
    }
    return mesh;
  }

  static MeshBase create_two_tet_mesh() {
    // Two tetrahedra separated in +x.
    std::vector<real_t> X_ref = {
        // Cell 0
        0.0, 0.0, 0.0,  // 0
        1.0, 0.0, 0.0,  // 1
        0.0, 1.0, 0.0,  // 2
        0.0, 0.0, 1.0,  // 3
        // Cell 1 (shifted)
        10.0, 0.0, 0.0,  // 4
        11.0, 0.0, 0.0,  // 5
        10.0, 1.0, 0.0,  // 6
        10.0, 0.0, 1.0   // 7
    };

    std::vector<offset_t> offs = {0, 4, 8};
    std::vector<index_t> conn = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<CellShape> shapes(2);
    shapes[0].family = CellFamily::Tetra;
    shapes[0].order = 1;
    shapes[0].num_corners = 4;
    shapes[1].family = CellFamily::Tetra;
    shapes[1].order = 1;
    shapes[1].num_corners = 4;

    MeshBase mesh;
    mesh.build_from_arrays(3, X_ref, offs, conn, shapes);
    return mesh;
  }
};

TEST_F(GeometryCacheTest, CellCenterCachingReference) {
  MeshBase mesh = create_unit_tet_mesh();

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_centers = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  auto c0 = cache.cell_center(0, Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.cell_center_misses, 1u);
  EXPECT_EQ(stats1.cell_center_hits, 0u);

  auto c1 = cache.cell_center(0, Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.cell_center_misses, 1u);
  EXPECT_EQ(stats2.cell_center_hits, 1u);
  EXPECT_TRUE(approx3(c0, c1));
}

TEST_F(GeometryCacheTest, CellMeasureCachingReference) {
  MeshBase mesh = create_unit_tet_mesh();

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_measures = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  auto m0 = cache.cell_measure(0, Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.cell_measure_misses, 1u);
  EXPECT_EQ(stats1.cell_measure_hits, 0u);

  auto m1 = cache.cell_measure(0, Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.cell_measure_misses, 1u);
  EXPECT_EQ(stats2.cell_measure_hits, 1u);
  EXPECT_TRUE(approx(m0, m1));
}

TEST_F(GeometryCacheTest, CurrentCacheInvalidatesOnGeometryChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  // Ensure current coordinates exist.
  mesh.set_current_coords(mesh.X_ref());

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_centers = true;
  cfg.cache_reference = false;
  cfg.cache_current = true;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  auto c0 = cache.cell_center(0, Configuration::Current);
  EXPECT_EQ(cache.get_stats().cell_center_misses, 1u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 0u);

  auto c1 = cache.cell_center(0, Configuration::Current);
  EXPECT_EQ(cache.get_stats().cell_center_misses, 1u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 1u);
  EXPECT_TRUE(approx3(c0, c1));

  // Mutate current coordinates (triggers MeshEvent::GeometryChanged).
  mesh.set_vertex_coords(0, {2.0, 2.0, 2.0});

  auto c2 = cache.cell_center(0, Configuration::Current);
  auto stats = cache.get_stats();
  EXPECT_EQ(stats.cell_center_misses, 2u);
  EXPECT_EQ(stats.cell_center_hits, 1u);
  EXPECT_FALSE(approx3(c0, c2));
}

TEST_F(GeometryCacheTest, FaceCenterCachingReferenceHasSeparateStats) {
  MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/true);
  ASSERT_GT(mesh.n_faces(), 0u);

  GeometryCache::CacheConfig cfg;
  cfg.enable_face_centers = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  const auto f0 = cache.face_center(0, Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.face_center_misses, 1u);
  EXPECT_EQ(stats1.face_center_hits, 0u);
  EXPECT_EQ(stats1.cell_center_hits, 0u);
  EXPECT_EQ(stats1.cell_center_misses, 0u);

  const auto f1 = cache.face_center(0, Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.face_center_misses, 1u);
  EXPECT_EQ(stats2.face_center_hits, 1u);
  EXPECT_TRUE(approx3(f0, f1));
}

TEST_F(GeometryCacheTest, CellBoundingBoxCachingReference) {
  MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/false);

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_bboxes = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  const auto b0 = cache.cell_bounding_box(0, Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.cell_bbox_misses, 1u);
  EXPECT_EQ(stats1.cell_bbox_hits, 0u);

  const auto b1 = cache.cell_bounding_box(0, Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.cell_bbox_misses, 1u);
  EXPECT_EQ(stats2.cell_bbox_hits, 1u);

  EXPECT_TRUE(approx3(b0.min, {0.0, 0.0, 0.0}));
  EXPECT_TRUE(approx3(b0.max, {1.0, 1.0, 1.0}));
  EXPECT_TRUE(approx3(b0.min, b1.min));
  EXPECT_TRUE(approx3(b0.max, b1.max));
}

TEST_F(GeometryCacheTest, MeshBoundingBoxCachingReference) {
  MeshBase mesh = create_two_tet_mesh();

  GeometryCache::CacheConfig cfg;
  cfg.enable_mesh_bbox = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  const auto b0 = cache.mesh_bounding_box(Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.mesh_bbox_misses, 1u);
  EXPECT_EQ(stats1.mesh_bbox_hits, 0u);

  const auto b1 = cache.mesh_bounding_box(Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.mesh_bbox_misses, 1u);
  EXPECT_EQ(stats2.mesh_bbox_hits, 1u);

  EXPECT_TRUE(approx3(b0.min, {0.0, 0.0, 0.0}));
  EXPECT_TRUE(approx3(b0.max, {11.0, 1.0, 1.0}));
  EXPECT_TRUE(approx3(b0.min, b1.min));
  EXPECT_TRUE(approx3(b0.max, b1.max));
}

TEST_F(GeometryCacheTest, WarmCachePopulatesAllEnabledCaches) {
  MeshBase mesh = create_unit_hex_mesh(/*finalize_topology=*/true);
  ASSERT_EQ(mesh.n_cells(), 1u);
  ASSERT_GT(mesh.n_faces(), 0u);
  ASSERT_GT(mesh.n_edges(), 0u);

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_centers = true;
  cfg.enable_cell_measures = true;
  cfg.enable_cell_bboxes = true;
  cfg.enable_face_centers = true;
  cfg.enable_face_normals = true;
  cfg.enable_face_areas = true;
  cfg.enable_edge_centers = true;
  cfg.enable_mesh_bbox = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  cache.warm_cache(Configuration::Reference);
  auto stats1 = cache.get_stats();
  EXPECT_EQ(stats1.cell_center_misses, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats1.cell_measure_misses, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats1.cell_bbox_misses, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats1.face_center_misses, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats1.face_normal_misses, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats1.face_area_misses, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats1.edge_center_misses, static_cast<size_t>(mesh.n_edges()));
  EXPECT_EQ(stats1.mesh_bbox_misses, 1u);

  cache.warm_cache(Configuration::Reference);
  auto stats2 = cache.get_stats();
  EXPECT_EQ(stats2.cell_center_hits, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats2.cell_measure_hits, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats2.cell_bbox_hits, static_cast<size_t>(mesh.n_cells()));
  EXPECT_EQ(stats2.face_center_hits, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats2.face_normal_hits, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats2.face_area_hits, static_cast<size_t>(mesh.n_faces()));
  EXPECT_EQ(stats2.edge_center_hits, static_cast<size_t>(mesh.n_edges()));
  EXPECT_EQ(stats2.mesh_bbox_hits, 1u);
}

TEST_F(GeometryCacheTest, CacheInvalidatesOnTopologyChanged) {
  MeshBase mesh = create_unit_tet_mesh();

  GeometryCache::CacheConfig cfg;
  cfg.enable_cell_centers = true;
  cfg.cache_reference = true;
  cfg.cache_current = false;

  GeometryCache cache(mesh, cfg);
  cache.reset_stats();

  const auto c0 = cache.cell_center(0, Configuration::Reference);
  (void)c0;
  EXPECT_EQ(cache.get_stats().cell_center_misses, 1u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 0u);

  const auto c1 = cache.cell_center(0, Configuration::Reference);
  (void)c1;
  EXPECT_EQ(cache.get_stats().cell_center_misses, 1u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 1u);

  // Rebuild the mesh to trigger TopologyChanged + GeometryChanged.
  const MeshBase new_mesh = create_two_tet_mesh();
  mesh.build_from_arrays(new_mesh.dim(),
                         new_mesh.X_ref(),
                         new_mesh.cell2vertex_offsets(),
                         new_mesh.cell2vertex(),
                         new_mesh.cell_shapes());

  const auto c2 = cache.cell_center(0, Configuration::Reference);
  (void)c2;
  EXPECT_EQ(cache.get_stats().cell_center_misses, 2u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 1u);

  const auto c3 = cache.cell_center(1, Configuration::Reference);
  (void)c3;
  EXPECT_EQ(cache.get_stats().cell_center_misses, 3u);
  EXPECT_EQ(cache.get_stats().cell_center_hits, 1u);
}

} // namespace test
} // namespace svmp
