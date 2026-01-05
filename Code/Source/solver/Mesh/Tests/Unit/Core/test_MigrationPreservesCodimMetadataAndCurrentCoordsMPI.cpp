/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file test_MigrationPreservesCodimMetadataAndCurrentCoordsMPI.cpp
 * @brief MPI regression test: DistributedMesh::migrate preserves codim metadata + X_cur.
 *
 * Production requirement: migrating a partition must not corrupt:
 * - face boundary labels, edge labels
 * - face/edge sets and face/edge fields
 * - per-vertex X_cur and active configuration
 *
 * This test builds a small distributed tetra "star", constructs a ghost layer, then migrates
 * all locally-owned cells to rank 0 while ensuring ghost cells are ignored (no duplicates).
 * Rank 0 validates that labels/sets/fields roundtrip and current coordinates are preserved.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NEAR(a, b, tol) ASSERT(std::fabs((a) - (b)) <= (tol))

static std::vector<svmp::gid_t> face_key(const svmp::MeshBase& mesh, svmp::index_t f) {
  auto [verts, n] = mesh.face_vertices_span(f);
  std::vector<svmp::gid_t> key;
  key.reserve(n);
  const auto& vg = mesh.vertex_gids();
  for (size_t i = 0; i < n; ++i) {
    const auto v = verts[i];
    ASSERT(v >= 0);
    ASSERT(static_cast<size_t>(v) < vg.size());
    key.push_back(vg[static_cast<size_t>(v)]);
  }
  std::sort(key.begin(), key.end());
  key.erase(std::unique(key.begin(), key.end()), key.end());
  return key;
}

static std::pair<svmp::gid_t, svmp::gid_t> edge_key(const svmp::MeshBase& mesh, svmp::index_t e) {
  const auto ev = mesh.edge_vertices(e);
  const auto& vg = mesh.vertex_gids();
  ASSERT(ev[0] >= 0);
  ASSERT(ev[1] >= 0);
  ASSERT(static_cast<size_t>(ev[0]) < vg.size());
  ASSERT(static_cast<size_t>(ev[1]) < vg.size());
  svmp::gid_t a = vg[static_cast<size_t>(ev[0])];
  svmp::gid_t b = vg[static_cast<size_t>(ev[1])];
  if (a > b) std::swap(a, b);
  return {a, b};
}

static svmp::real_t expected_face_value(const std::vector<svmp::gid_t>& key) {
  ASSERT_EQ(key.size(), 3u);
  const long long code = static_cast<long long>(key[0]) * 10000LL +
                         static_cast<long long>(key[1]) * 100LL +
                         static_cast<long long>(key[2]);
  return static_cast<svmp::real_t>(code);
}

static svmp::real_t expected_edge_value(const std::pair<svmp::gid_t, svmp::gid_t>& key) {
  const long long code =
      static_cast<long long>(key.first) * 100000LL + static_cast<long long>(key.second);
  return static_cast<svmp::real_t>(code);
}

static std::shared_ptr<svmp::MeshBase> make_star_tet_mesh(int rank, int world) {
  (void)world;
  auto mesh = std::make_shared<svmp::MeshBase>();
  const int dim = 3;

  std::vector<svmp::gid_t> vertex_gids;
  std::vector<svmp::real_t> coords;

  if (rank == 0) {
    vertex_gids = {0, 1, 2, 3};
    coords = {
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 2
        0.0, 0.0, 1.0, // 3
    };
  } else if (rank == 1) {
    vertex_gids = {0, 1, 2, 101};
    coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, -1.0,
    };
  } else if (rank == 2) {
    vertex_gids = {0, 1, 3, 102};
    coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
    };
  } else if (rank == 3) {
    vertex_gids = {0, 2, 3, 103};
    coords = {
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };
  } else {
    vertex_gids = {0, 1, 2, static_cast<svmp::gid_t>(100 + rank)};
    coords = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
  }

  std::vector<svmp::offset_t> offsets = {0, 4};
  std::vector<svmp::index_t> conn = {0, 1, 2, 3};
  std::vector<svmp::CellShape> shapes = {{svmp::CellFamily::Tetra, 4, 1}};
  std::vector<svmp::gid_t> cell_gids = {static_cast<svmp::gid_t>(1000 + rank)};

  mesh->build_from_arrays(dim, coords, offsets, conn, shapes);
  mesh->set_vertex_gids(std::move(vertex_gids));
  mesh->set_cell_gids(std::move(cell_gids));
  mesh->finalize();
  return mesh;
}

static void register_common_labels(svmp::MeshBase& mesh, int max_rank) {
  mesh.register_label("shared_vertex", 10);
  for (int r = 0; r <= max_rank; ++r) {
    mesh.register_label("special_vertex_" + std::to_string(r), static_cast<svmp::label_t>(20 + r));
    mesh.register_label("region_" + std::to_string(r), static_cast<svmp::label_t>(300 + r));
    mesh.register_label("marked_face_" + std::to_string(r), static_cast<svmp::label_t>(100 + r));
    mesh.register_label("marked_edge_" + std::to_string(r), static_cast<svmp::label_t>(200 + r));
  }
}

static svmp::gid_t special_vertex_gid_for_rank(int rank) {
  if (rank == 0) return 3;
  if (rank == 1) return 101;
  if (rank == 2) return 102;
  if (rank == 3) return 103;
  return static_cast<svmp::gid_t>(100 + rank);
}

static std::vector<svmp::gid_t> marked_face_key_for_rank(int rank) {
  if (rank == 0) return {1, 2, 3};
  if (rank == 1) return {0, 1, 101};
  if (rank == 2) return {0, 1, 102};
  if (rank == 3) return {0, 2, 103};
  return {};
}

static std::pair<svmp::gid_t, svmp::gid_t> marked_edge_key_for_rank(int rank) {
  if (rank == 0) return {1, 2};
  if (rank == 1) return {0, 101};
  if (rank == 2) return {0, 102};
  if (rank == 3) return {0, 103};
  return {svmp::INVALID_GID, svmp::INVALID_GID};
}

static svmp::index_t find_face_by_key(const svmp::MeshBase& mesh, const std::vector<svmp::gid_t>& key) {
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
    if (face_key(mesh, f) == key) return f;
  }
  return svmp::INVALID_INDEX;
}

static svmp::index_t find_edge_by_key(const svmp::MeshBase& mesh, const std::pair<svmp::gid_t, svmp::gid_t>& key) {
  for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
    if (edge_key(mesh, e) == key) return e;
  }
  return svmp::INVALID_INDEX;
}

static void attach_metadata_and_fields(svmp::DistributedMesh& dmesh, int rank, int world) {
  (void)world;
  auto& mesh = dmesh.local_mesh();

  register_common_labels(mesh, 3);

  // Cell metadata.
  mesh.set_region_label(0, static_cast<svmp::label_t>(300 + rank));
  mesh.set_refinement_level(0, static_cast<size_t>(rank + 1));

  // Vertex labels.
  const auto v0 = mesh.global_to_local_vertex(0);
  ASSERT(v0 != svmp::INVALID_INDEX);
  mesh.set_vertex_label(v0, 10);

  const auto sp_gid = special_vertex_gid_for_rank(rank);
  const auto vs = mesh.global_to_local_vertex(sp_gid);
  ASSERT(vs != svmp::INVALID_INDEX);
  mesh.set_vertex_label(vs, static_cast<svmp::label_t>(20 + rank));

  // Marked face/edge labels.
  const auto fkey = marked_face_key_for_rank(rank);
  const auto ekey = marked_edge_key_for_rank(rank);
  ASSERT(!fkey.empty());
  ASSERT(ekey.first != svmp::INVALID_GID);

  const auto f = find_face_by_key(mesh, fkey);
  ASSERT(f != svmp::INVALID_INDEX);
  mesh.set_boundary_label(f, static_cast<svmp::label_t>(100 + rank));

  const auto e = find_edge_by_key(mesh, ekey);
  ASSERT(e != svmp::INVALID_INDEX);
  mesh.set_edge_label(e, static_cast<svmp::label_t>(200 + rank));

  // Sets (same names across ranks; migration should union them by GID).
  mesh.add_to_set(svmp::EntityKind::Vertex, "vset", v0);
  mesh.add_to_set(svmp::EntityKind::Vertex, "vset", vs);
  mesh.add_to_set(svmp::EntityKind::Volume, "cset", 0);
  mesh.add_to_set(svmp::EntityKind::Face, "fset", f);
  mesh.add_to_set(svmp::EntityKind::Edge, "eset", e);

  // Fields
  const auto vh = mesh.attach_field(svmp::EntityKind::Vertex, "v_field", svmp::FieldScalarType::Float64, 1);
  auto* v_data = mesh.field_data_as<svmp::real_t>(vh);
  ASSERT(v_data);
  for (svmp::index_t i = 0; i < static_cast<svmp::index_t>(mesh.n_vertices()); ++i) {
    v_data[i] = static_cast<svmp::real_t>(mesh.vertex_gids()[static_cast<size_t>(i)]);
  }

  const auto ch = mesh.attach_field(svmp::EntityKind::Volume, "c_field", svmp::FieldScalarType::Float64, 1);
  auto* c_data = mesh.field_data_as<svmp::real_t>(ch);
  ASSERT(c_data);
  c_data[0] = static_cast<svmp::real_t>(mesh.cell_gids()[0]) + static_cast<svmp::real_t>(0.25);

  const auto fh = mesh.attach_field(svmp::EntityKind::Face, "f_field", svmp::FieldScalarType::Float64, 1);
  auto* f_data = mesh.field_data_as<svmp::real_t>(fh);
  ASSERT(f_data);
  for (svmp::index_t i = 0; i < static_cast<svmp::index_t>(mesh.n_faces()); ++i) {
    f_data[i] = expected_face_value(face_key(mesh, i));
  }

  const auto eh = mesh.attach_field(svmp::EntityKind::Edge, "e_field", svmp::FieldScalarType::Float64, 1);
  auto* e_data = mesh.field_data_as<svmp::real_t>(eh);
  ASSERT(e_data);
  for (svmp::index_t i = 0; i < static_cast<svmp::index_t>(mesh.n_edges()); ++i) {
    e_data[i] = expected_edge_value(edge_key(mesh, i));
  }

  // Current coordinates: X_cur = X_ref + f(gid) offset, active configuration Current.
  {
    ASSERT_EQ(mesh.dim(), 3);
    const auto& vg = mesh.vertex_gids();
    std::vector<svmp::real_t> xcur = mesh.X_ref();
    for (size_t i = 0; i < vg.size(); ++i) {
      const auto gid = vg[i];
      xcur[i * 3 + 0] += static_cast<svmp::real_t>(1e-3) * static_cast<svmp::real_t>(gid);
      xcur[i * 3 + 1] += static_cast<svmp::real_t>(2e-3) * static_cast<svmp::real_t>(gid);
      xcur[i * 3 + 2] += static_cast<svmp::real_t>(3e-3) * static_cast<svmp::real_t>(gid);
    }
    mesh.set_current_coords(xcur);
    mesh.use_current_configuration();
  }
}

static svmp::label_t expected_face_boundary_label(const std::vector<svmp::gid_t>& key, int world) {
  if (key == std::vector<svmp::gid_t>({1, 2, 3})) return 100;
  if (world >= 2 && key == std::vector<svmp::gid_t>({0, 1, 101})) return 101;
  if (world >= 4 && key == std::vector<svmp::gid_t>({0, 1, 102})) return 102;
  if (world >= 4 && key == std::vector<svmp::gid_t>({0, 2, 103})) return 103;
  return svmp::INVALID_LABEL;
}

static svmp::label_t expected_edge_label(const std::pair<svmp::gid_t, svmp::gid_t>& key, int world) {
  if (key == std::make_pair<svmp::gid_t, svmp::gid_t>(1, 2)) return 200;
  if (world >= 2 && key == std::make_pair<svmp::gid_t, svmp::gid_t>(0, 101)) return 201;
  if (world >= 4 && key == std::make_pair<svmp::gid_t, svmp::gid_t>(0, 102)) return 202;
  if (world >= 4 && key == std::make_pair<svmp::gid_t, svmp::gid_t>(0, 103)) return 203;
  return svmp::INVALID_LABEL;
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world != 2 && world != 4) {
    if (rank == 0) {
      std::cout << "Skipping migrate metadata+coords test (designed for 2 or 4 ranks, got " << world << ")\n";
    }
    MPI_Finalize();
    return 0;
  }

  auto local_mesh = svmp::test::make_star_tet_mesh(rank, world);
  svmp::DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  svmp::test::attach_metadata_and_fields(dmesh, rank, world);

  // Build ghost layer so migrate() must ignore ghost cells.
  dmesh.build_ghost_layer(1);

  std::vector<svmp::rank_t> new_owner(dmesh.local_mesh().n_cells(), 0);
  dmesh.migrate(new_owner);

  if (rank == 0) {
    const auto& mesh = dmesh.local_mesh();

    // Ghost cells must not have been migrated: expect exactly one owned cell per rank.
    ASSERT_EQ(mesh.n_cells(), static_cast<size_t>(world));
    ASSERT_EQ(mesh.n_vertices(), static_cast<size_t>(world + 3)); // {0,1,2,3}+{101..}

    ASSERT(mesh.has_current_coords());
    ASSERT_EQ(mesh.active_configuration(), svmp::Configuration::Current);

    // Cell region labels + refinement levels preserved.
    const auto& cg = mesh.cell_gids();
    for (size_t i = 0; i < cg.size(); ++i) {
      const auto gid = cg[i];
      const int owner_rank = static_cast<int>(gid - 1000);
      const auto c = mesh.global_to_local_cell(gid);
      ASSERT(c != svmp::INVALID_INDEX);
      ASSERT_EQ(mesh.region_label(c), static_cast<svmp::label_t>(300 + owner_rank));
      ASSERT_EQ(mesh.refinement_level(c), static_cast<size_t>(owner_rank + 1));
    }

    // Current coordinates preserved: X_cur == X_ref + offset(gid).
    {
      const auto& vg = mesh.vertex_gids();
      const auto& Xref = mesh.X_ref();
      const auto& Xcur = mesh.X_cur();
      ASSERT_EQ(Xref.size(), Xcur.size());
      ASSERT_EQ(mesh.dim(), 3);
      for (size_t i = 0; i < vg.size(); ++i) {
        const auto gid = vg[i];
        const svmp::real_t dx = static_cast<svmp::real_t>(1e-3) * static_cast<svmp::real_t>(gid);
        const svmp::real_t dy = static_cast<svmp::real_t>(2e-3) * static_cast<svmp::real_t>(gid);
        const svmp::real_t dz = static_cast<svmp::real_t>(3e-3) * static_cast<svmp::real_t>(gid);
        ASSERT_NEAR(Xcur[i * 3 + 0], Xref[i * 3 + 0] + dx, 1e-12);
        ASSERT_NEAR(Xcur[i * 3 + 1], Xref[i * 3 + 1] + dy, 1e-12);
        ASSERT_NEAR(Xcur[i * 3 + 2], Xref[i * 3 + 2] + dz, 1e-12);
      }
    }

    // Face boundary labels preserved (by face key).
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
      const auto key = svmp::test::face_key(mesh, f);
      const auto expected = svmp::test::expected_face_boundary_label(key, world);
      ASSERT_EQ(mesh.boundary_label(f), expected);
    }

    // Edge labels preserved (by edge key).
    for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
      const auto key = svmp::test::edge_key(mesh, e);
      const auto expected = svmp::test::expected_edge_label(key, world);
      ASSERT_EQ(mesh.edge_label(e), expected);
    }

    // Sets preserved and unioned across ranks.
    ASSERT(mesh.has_set(svmp::EntityKind::Vertex, "vset"));
    ASSERT(mesh.has_set(svmp::EntityKind::Volume, "cset"));
    ASSERT(mesh.has_set(svmp::EntityKind::Face, "fset"));
    ASSERT(mesh.has_set(svmp::EntityKind::Edge, "eset"));

    {
      std::set<svmp::gid_t> got;
      for (const auto id : mesh.get_set(svmp::EntityKind::Vertex, "vset")) {
        ASSERT(id >= 0);
        ASSERT(static_cast<size_t>(id) < mesh.vertex_gids().size());
        got.insert(mesh.vertex_gids()[static_cast<size_t>(id)]);
      }
      std::set<svmp::gid_t> expected = {0, 3};
      expected.insert(101);
      if (world >= 4) {
        expected.insert(102);
        expected.insert(103);
      }
      ASSERT(got == expected);
    }

    {
      std::set<svmp::gid_t> got;
      for (const auto id : mesh.get_set(svmp::EntityKind::Volume, "cset")) {
        ASSERT(id >= 0);
        ASSERT(static_cast<size_t>(id) < mesh.cell_gids().size());
        got.insert(mesh.cell_gids()[static_cast<size_t>(id)]);
      }
      std::set<svmp::gid_t> expected;
      for (int r = 0; r < world; ++r) expected.insert(static_cast<svmp::gid_t>(1000 + r));
      ASSERT(got == expected);
    }

    {
      std::set<std::vector<svmp::gid_t>> got;
      for (const auto id : mesh.get_set(svmp::EntityKind::Face, "fset")) {
        ASSERT(id >= 0);
        ASSERT(static_cast<size_t>(id) < mesh.n_faces());
        got.insert(svmp::test::face_key(mesh, id));
      }
      std::set<std::vector<svmp::gid_t>> expected;
      expected.insert({1, 2, 3});
      expected.insert({0, 1, 101});
      if (world >= 4) {
        expected.insert({0, 1, 102});
        expected.insert({0, 2, 103});
      }
      ASSERT(got == expected);
    }

    {
      std::set<std::pair<svmp::gid_t, svmp::gid_t>> got;
      for (const auto id : mesh.get_set(svmp::EntityKind::Edge, "eset")) {
        ASSERT(id >= 0);
        ASSERT(static_cast<size_t>(id) < mesh.n_edges());
        got.insert(svmp::test::edge_key(mesh, id));
      }
      std::set<std::pair<svmp::gid_t, svmp::gid_t>> expected;
      expected.insert({1, 2});
      expected.insert({0, 101});
      if (world >= 4) {
        expected.insert({0, 102});
        expected.insert({0, 103});
      }
      ASSERT(got == expected);
    }

    // Fields preserved (and de-duplicated by GID correctly).
    ASSERT(mesh.has_field(svmp::EntityKind::Vertex, "v_field"));
    ASSERT(mesh.has_field(svmp::EntityKind::Volume, "c_field"));
    ASSERT(mesh.has_field(svmp::EntityKind::Face, "f_field"));
    ASSERT(mesh.has_field(svmp::EntityKind::Edge, "e_field"));

    {
      const auto vh = mesh.field_handle(svmp::EntityKind::Vertex, "v_field");
      const auto ch = mesh.field_handle(svmp::EntityKind::Volume, "c_field");
      const auto fh = mesh.field_handle(svmp::EntityKind::Face, "f_field");
      const auto eh = mesh.field_handle(svmp::EntityKind::Edge, "e_field");
      const auto* v_data = mesh.field_data_as<const svmp::real_t>(vh);
      const auto* c_data = mesh.field_data_as<const svmp::real_t>(ch);
      const auto* f_data = mesh.field_data_as<const svmp::real_t>(fh);
      const auto* e_data = mesh.field_data_as<const svmp::real_t>(eh);
      ASSERT(v_data);
      ASSERT(c_data);
      ASSERT(f_data);
      ASSERT(e_data);

      for (size_t i = 0; i < mesh.vertex_gids().size(); ++i) {
        ASSERT_NEAR(v_data[i], static_cast<svmp::real_t>(mesh.vertex_gids()[i]), 1e-12);
      }
      for (size_t i = 0; i < mesh.cell_gids().size(); ++i) {
        ASSERT_NEAR(c_data[i], static_cast<svmp::real_t>(mesh.cell_gids()[i]) + 0.25, 1e-12);
      }
      for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
        ASSERT_NEAR(f_data[f], svmp::test::expected_face_value(svmp::test::face_key(mesh, f)), 1e-12);
      }
      for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
        ASSERT_NEAR(e_data[e], svmp::test::expected_edge_value(svmp::test::edge_key(mesh, e)), 1e-12);
      }
    }

    std::cout << "Migration codim metadata + current coords test PASSED (" << world << " ranks)\n";
  } else {
    ASSERT_EQ(dmesh.local_mesh().n_cells(), 0u);
  }

  MPI_Finalize();
  return 0;
}

