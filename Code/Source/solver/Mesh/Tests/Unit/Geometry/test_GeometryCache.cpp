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

} // namespace test
} // namespace svmp
