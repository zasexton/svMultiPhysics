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
 * @file test_GhostLayerPreservesMetadataMPI.cpp
 * @brief MPI regression test: build_ghost_layer()/clear_ghosts() preserve metadata and X_cur.
 *
 * Production requirement: ghost-layer rebuilds must not drop or corrupt:
 * - label registry (name<->id)
 * - vertex labels, face boundary labels, edge labels
 * - sets across all entity kinds
 * - per-cell refinement levels
 * - current coordinates (X_cur) and active configuration
 *
 * This test constructs a small distributed "star" tetra mesh, annotates each rank's owned
 * mesh with metadata + fields, then runs build_ghost_layer(1) and clear_ghosts() and
 * verifies the annotations are preserved for the base (owned) entities.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
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
    // Central tetra (v0,v1,v2,v3).
    vertex_gids = {0, 1, 2, 3};
    coords = {
        0.0, 0.0, 0.0, // gid 0
        1.0, 0.0, 0.0, // gid 1
        0.0, 1.0, 0.0, // gid 2
        0.0, 0.0, 1.0, // gid 3
    };
  } else if (rank == 1) {
    // Shares face {0,1,2} with rank 0.
    vertex_gids = {0, 1, 2, 101};
    coords = {
        0.0, 0.0, 0.0, // gid 0
        1.0, 0.0, 0.0, // gid 1
        0.0, 1.0, 0.0, // gid 2
        0.0, 0.0, -1.0 // gid 101
    };
  } else if (rank == 2) {
    // Shares face {0,1,3} with rank 0.
    vertex_gids = {0, 1, 3, 102};
    coords = {
        0.0, 0.0, 0.0, // gid 0
        1.0, 0.0, 0.0, // gid 1
        0.0, 0.0, 1.0, // gid 3
        0.0, 1.0, 1.0  // gid 102
    };
  } else if (rank == 3) {
    // Shares face {0,2,3} with rank 0.
    vertex_gids = {0, 2, 3, 103};
    coords = {
        0.0, 0.0, 0.0, // gid 0
        0.0, 1.0, 0.0, // gid 2
        0.0, 0.0, 1.0, // gid 3
        1.0, 1.0, 1.0  // gid 103
    };
  } else {
    // This test suite is registered with 2 and 4 ranks; guard against unexpected sizes.
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

struct Expected {
  svmp::gid_t cell_gid = svmp::INVALID_GID;
  svmp::label_t region_label = svmp::INVALID_LABEL;
  size_t refinement_level = 0;

  std::map<std::string, svmp::label_t> registered;

  std::set<svmp::gid_t> vset_gids;
  std::set<svmp::gid_t> cset_gids;
  std::set<std::vector<svmp::gid_t>> fset_keys;
  std::set<std::pair<svmp::gid_t, svmp::gid_t>> eset_keys;

  std::map<svmp::gid_t, svmp::label_t> vertex_labels;

  std::map<svmp::gid_t, std::array<svmp::real_t, 3>> xcur_by_gid;

  std::vector<svmp::gid_t> base_vertex_gids;
  size_t base_n_cells = 0;
  size_t base_n_vertices = 0;

  // One marked face/edge (by canonical GID).
  svmp::gid_t marked_face_gid = svmp::INVALID_GID;
  svmp::label_t marked_face_label = svmp::INVALID_LABEL;
  svmp::gid_t marked_edge_gid = svmp::INVALID_GID;
  svmp::label_t marked_edge_label = svmp::INVALID_LABEL;

  // Field names are fixed.
  svmp::real_t marked_face_field_value = 0;
  svmp::real_t marked_edge_field_value = 0;
};

static Expected capture_expected(const svmp::DistributedMesh& dmesh,
                                 const std::vector<svmp::gid_t>& marked_face_key,
                                 const std::pair<svmp::gid_t, svmp::gid_t>& marked_edge_key,
                                 svmp::label_t marked_face_label,
                                 svmp::label_t marked_edge_label) {
  const auto& mesh = dmesh.local_mesh();

  Expected exp;
  exp.base_n_cells = mesh.n_cells();
  exp.base_n_vertices = mesh.n_vertices();
  exp.base_vertex_gids = mesh.vertex_gids();
  exp.cell_gid = mesh.cell_gids().at(0);
  exp.region_label = mesh.region_label(0);
  exp.refinement_level = mesh.refinement_level(0);

  // Registry: capture a small set of name->label expectations.
  for (const auto& [label, name] : mesh.list_label_names()) {
    if (!name.empty()) exp.registered[name] = label;
  }

  // Vertex labels + current coordinates for base vertices.
  ASSERT(mesh.has_current_coords());
  const int dim = mesh.dim();
  ASSERT_EQ(dim, 3);
  for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
    const auto gid = mesh.vertex_gids()[static_cast<size_t>(v)];
    exp.vertex_labels[gid] = mesh.vertex_label(v);
    const size_t off = static_cast<size_t>(v) * static_cast<size_t>(dim);
    exp.xcur_by_gid[gid] = {mesh.X_cur()[off + 0], mesh.X_cur()[off + 1], mesh.X_cur()[off + 2]};
  }

  // Sets: capture membership by GID/key.
  ASSERT(mesh.has_set(svmp::EntityKind::Vertex, "vset"));
  for (const auto id : mesh.get_set(svmp::EntityKind::Vertex, "vset")) {
    ASSERT(id >= 0);
    ASSERT(static_cast<size_t>(id) < mesh.vertex_gids().size());
    exp.vset_gids.insert(mesh.vertex_gids()[static_cast<size_t>(id)]);
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Volume, "cset"));
  for (const auto id : mesh.get_set(svmp::EntityKind::Volume, "cset")) {
    ASSERT(id >= 0);
    ASSERT(static_cast<size_t>(id) < mesh.cell_gids().size());
    exp.cset_gids.insert(mesh.cell_gids()[static_cast<size_t>(id)]);
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Face, "fset"));
  for (const auto id : mesh.get_set(svmp::EntityKind::Face, "fset")) {
    ASSERT(id >= 0);
    ASSERT(static_cast<size_t>(id) < mesh.n_faces());
    exp.fset_keys.insert(face_key(mesh, id));
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Edge, "eset"));
  for (const auto id : mesh.get_set(svmp::EntityKind::Edge, "eset")) {
    ASSERT(id >= 0);
    ASSERT(static_cast<size_t>(id) < mesh.n_edges());
    exp.eset_keys.insert(edge_key(mesh, id));
  }

  // Marked face/edge canonical GIDs (requires build_exchange_patterns() to have run).
  const auto f = find_face_by_key(mesh, marked_face_key);
  ASSERT(f != svmp::INVALID_INDEX);
  exp.marked_face_gid = mesh.face_gids()[static_cast<size_t>(f)];
  exp.marked_face_label = marked_face_label;

  const auto e = find_edge_by_key(mesh, marked_edge_key);
  ASSERT(e != svmp::INVALID_INDEX);
  exp.marked_edge_gid = mesh.edge_gids()[static_cast<size_t>(e)];
  exp.marked_edge_label = marked_edge_label;

  // Field values (Face + Edge) for the marked entities.
  ASSERT(mesh.has_field(svmp::EntityKind::Face, "f_field"));
  ASSERT(mesh.has_field(svmp::EntityKind::Edge, "e_field"));
  const auto fh = mesh.field_handle(svmp::EntityKind::Face, "f_field");
  const auto eh = mesh.field_handle(svmp::EntityKind::Edge, "e_field");
  const auto* f_data = mesh.field_data_as<const svmp::real_t>(fh);
  const auto* e_data = mesh.field_data_as<const svmp::real_t>(eh);
  ASSERT(f_data);
  ASSERT(e_data);
  exp.marked_face_field_value = f_data[f];
  exp.marked_edge_field_value = e_data[e];

  return exp;
}

static void verify_expected(const svmp::DistributedMesh& dmesh, const Expected& exp) {
  const auto& mesh = dmesh.local_mesh();
  ASSERT(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), svmp::Configuration::Current);

  // Registry roundtrip for captured entries.
  for (const auto& [name, label] : exp.registered) {
    ASSERT_EQ(mesh.label_from_name(name), label);
    ASSERT_EQ(mesh.label_name(label), name);
  }

  // Cell metadata (the owned base cell must exist after ghost ops and after clear_ghosts()).
  const auto c0 = mesh.global_to_local_cell(exp.cell_gid);
  ASSERT(c0 != svmp::INVALID_INDEX);
  ASSERT_EQ(mesh.region_label(c0), exp.region_label);
  ASSERT_EQ(mesh.refinement_level(c0), exp.refinement_level);

  // Marked face/edge labels via canonical GID lookup.
  const auto f = mesh.global_to_local_face(exp.marked_face_gid);
  ASSERT(f != svmp::INVALID_INDEX);
  ASSERT_EQ(mesh.boundary_label(f), exp.marked_face_label);

  const auto e = mesh.global_to_local_edge(exp.marked_edge_gid);
  ASSERT(e != svmp::INVALID_INDEX);
  ASSERT_EQ(mesh.edge_label(e), exp.marked_edge_label);

  // Sets (compare by key/GID rather than local indices).
  ASSERT(mesh.has_set(svmp::EntityKind::Vertex, "vset"));
  {
    std::set<svmp::gid_t> got;
    for (const auto id : mesh.get_set(svmp::EntityKind::Vertex, "vset")) {
      ASSERT(id >= 0);
      ASSERT(static_cast<size_t>(id) < mesh.vertex_gids().size());
      got.insert(mesh.vertex_gids()[static_cast<size_t>(id)]);
    }
    ASSERT(got == exp.vset_gids);
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Volume, "cset"));
  {
    std::set<svmp::gid_t> got;
    for (const auto id : mesh.get_set(svmp::EntityKind::Volume, "cset")) {
      ASSERT(id >= 0);
      ASSERT(static_cast<size_t>(id) < mesh.cell_gids().size());
      got.insert(mesh.cell_gids()[static_cast<size_t>(id)]);
    }
    ASSERT(got == exp.cset_gids);
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Face, "fset"));
  {
    std::set<std::vector<svmp::gid_t>> got;
    for (const auto id : mesh.get_set(svmp::EntityKind::Face, "fset")) {
      ASSERT(id >= 0);
      ASSERT(static_cast<size_t>(id) < mesh.n_faces());
      got.insert(face_key(mesh, id));
    }
    ASSERT(got == exp.fset_keys);
  }

  ASSERT(mesh.has_set(svmp::EntityKind::Edge, "eset"));
  {
    std::set<std::pair<svmp::gid_t, svmp::gid_t>> got;
    for (const auto id : mesh.get_set(svmp::EntityKind::Edge, "eset")) {
      ASSERT(id >= 0);
      ASSERT(static_cast<size_t>(id) < mesh.n_edges());
      got.insert(edge_key(mesh, id));
    }
    ASSERT(got == exp.eset_keys);
  }

  // Fields still exist and remain correct on marked entities.
  ASSERT(mesh.has_field(svmp::EntityKind::Face, "f_field"));
  ASSERT(mesh.has_field(svmp::EntityKind::Edge, "e_field"));
  {
    const auto fh = mesh.field_handle(svmp::EntityKind::Face, "f_field");
    const auto eh = mesh.field_handle(svmp::EntityKind::Edge, "e_field");
    const auto* f_data = mesh.field_data_as<const svmp::real_t>(fh);
    const auto* e_data = mesh.field_data_as<const svmp::real_t>(eh);
    ASSERT(f_data);
    ASSERT(e_data);

    const auto f = mesh.global_to_local_face(exp.marked_face_gid);
    ASSERT(f != svmp::INVALID_INDEX);
    ASSERT_NEAR(f_data[f], exp.marked_face_field_value, 1e-12);

    const auto e = mesh.global_to_local_edge(exp.marked_edge_gid);
    ASSERT(e != svmp::INVALID_INDEX);
    ASSERT_NEAR(e_data[e], exp.marked_edge_field_value, 1e-12);
  }

  // Current coordinates for base vertices are preserved by GID mapping.
  const int dim = mesh.dim();
  ASSERT_EQ(dim, 3);
  for (const auto gid : exp.base_vertex_gids) {
    const auto v = mesh.global_to_local_vertex(gid);
    ASSERT(v != svmp::INVALID_INDEX);
    const size_t off = static_cast<size_t>(v) * static_cast<size_t>(dim);
    const auto it = exp.xcur_by_gid.find(gid);
    ASSERT(it != exp.xcur_by_gid.end());
    ASSERT_NEAR(mesh.X_cur()[off + 0], it->second[0], 1e-12);
    ASSERT_NEAR(mesh.X_cur()[off + 1], it->second[1], 1e-12);
    ASSERT_NEAR(mesh.X_cur()[off + 2], it->second[2], 1e-12);
  }

  // Vertex label preservation for base vertices.
  for (const auto& [gid, label] : exp.vertex_labels) {
    const auto v = mesh.global_to_local_vertex(gid);
    if (v == svmp::INVALID_INDEX) continue;
    ASSERT_EQ(mesh.vertex_label(v), label);
  }
}

static void attach_metadata_and_fields(svmp::DistributedMesh& dmesh, int rank, int world,
                                       std::vector<svmp::gid_t>& marked_face_key_out,
                                       std::pair<svmp::gid_t, svmp::gid_t>& marked_edge_key_out,
                                       svmp::label_t& marked_face_label_out,
                                       svmp::label_t& marked_edge_label_out) {
  (void)world;
  auto& mesh = dmesh.local_mesh();

  // Region label + refinement metadata.
  const svmp::label_t region_label = static_cast<svmp::label_t>(300 + rank);
  mesh.register_label("region_" + std::to_string(rank), region_label);
  mesh.set_region_label(0, region_label);
  mesh.set_refinement_level(0, static_cast<size_t>(rank + 1));

  // Vertex labels: mark shared vertex gid 0 and one "special" vertex per rank.
  const svmp::label_t shared_v_label = 10;
  mesh.register_label("shared_vertex", shared_v_label);
  const auto v0 = mesh.global_to_local_vertex(0);
  ASSERT(v0 != svmp::INVALID_INDEX);
  mesh.set_vertex_label(v0, shared_v_label);

  svmp::gid_t special_gid = 3;
  if (rank == 1) special_gid = 101;
  if (rank == 2) special_gid = 102;
  if (rank == 3) special_gid = 103;
  if (rank >= 4) special_gid = mesh.vertex_gids().back();

  const svmp::label_t special_v_label = static_cast<svmp::label_t>(20 + rank);
  mesh.register_label("special_vertex_" + std::to_string(rank), special_v_label);
  const auto vs = mesh.global_to_local_vertex(special_gid);
  ASSERT(vs != svmp::INVALID_INDEX);
  mesh.set_vertex_label(vs, special_v_label);

  // Marked face/edge choices (physical boundary entities in the global mesh).
  if (rank == 0) {
    marked_face_key_out = {1, 2, 3};
    marked_edge_key_out = {1, 2};
  } else if (rank == 1) {
    marked_face_key_out = {0, 1, 101};
    marked_edge_key_out = {0, 101};
  } else if (rank == 2) {
    marked_face_key_out = {0, 1, 102};
    marked_edge_key_out = {0, 102};
  } else if (rank == 3) {
    marked_face_key_out = {0, 2, 103};
    marked_edge_key_out = {0, 103};
  } else {
    // Fallback for unexpected ranks.
    const auto& vg = mesh.vertex_gids();
    ASSERT(vg.size() >= 2);
    marked_face_key_out = face_key(mesh, 0);
    marked_edge_key_out = edge_key(mesh, 0);
  }

  marked_face_label_out = static_cast<svmp::label_t>(100 + rank);
  marked_edge_label_out = static_cast<svmp::label_t>(200 + rank);
  mesh.register_label("marked_face_" + std::to_string(rank), marked_face_label_out);
  mesh.register_label("marked_edge_" + std::to_string(rank), marked_edge_label_out);

  const auto mf = find_face_by_key(mesh, marked_face_key_out);
  ASSERT(mf != svmp::INVALID_INDEX);
  mesh.set_boundary_label(mf, marked_face_label_out);

  const auto me = find_edge_by_key(mesh, marked_edge_key_out);
  ASSERT(me != svmp::INVALID_INDEX);
  mesh.set_edge_label(me, marked_edge_label_out);

  // Sets
  mesh.add_to_set(svmp::EntityKind::Vertex, "vset", v0);
  mesh.add_to_set(svmp::EntityKind::Vertex, "vset", vs);
  mesh.add_to_set(svmp::EntityKind::Volume, "cset", 0);
  mesh.add_to_set(svmp::EntityKind::Face, "fset", mf);
  mesh.add_to_set(svmp::EntityKind::Edge, "eset", me);

  // Fields (all kinds) - use values derived from GIDs/keys.
  {
    const auto vh = mesh.attach_field(svmp::EntityKind::Vertex, "v_field", svmp::FieldScalarType::Float64, 1);
    auto* v_data = mesh.field_data_as<svmp::real_t>(vh);
    ASSERT(v_data);
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
      const auto gid = mesh.vertex_gids()[static_cast<size_t>(v)];
      v_data[v] = static_cast<svmp::real_t>(gid);
    }

    const auto ch = mesh.attach_field(svmp::EntityKind::Volume, "c_field", svmp::FieldScalarType::Float64, 1);
    auto* c_data = mesh.field_data_as<svmp::real_t>(ch);
    ASSERT(c_data);
    c_data[0] = static_cast<svmp::real_t>(mesh.cell_gids().at(0)) + static_cast<svmp::real_t>(0.25);

    const auto fh = mesh.attach_field(svmp::EntityKind::Face, "f_field", svmp::FieldScalarType::Float64, 1);
    auto* f_data = mesh.field_data_as<svmp::real_t>(fh);
    ASSERT(f_data);
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
      f_data[f] = expected_face_value(face_key(mesh, f));
    }

    const auto eh = mesh.attach_field(svmp::EntityKind::Edge, "e_field", svmp::FieldScalarType::Float64, 1);
    auto* e_data = mesh.field_data_as<svmp::real_t>(eh);
    ASSERT(e_data);
    for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
      e_data[e] = expected_edge_value(edge_key(mesh, e));
    }
  }

  // Current coordinates: X_cur = X_ref + f(gid) offset, and mark Current as active.
  {
    const int dim = mesh.dim();
    ASSERT_EQ(dim, 3);
    const auto& vg = mesh.vertex_gids();
    std::vector<svmp::real_t> xcur = mesh.X_ref();
    ASSERT_EQ(xcur.size(), mesh.X_ref().size());
    for (size_t i = 0; i < vg.size(); ++i) {
      const auto gid = vg[i];
      const svmp::real_t dx = static_cast<svmp::real_t>(1e-3) * static_cast<svmp::real_t>(gid);
      const svmp::real_t dy = static_cast<svmp::real_t>(2e-3) * static_cast<svmp::real_t>(gid);
      const svmp::real_t dz = static_cast<svmp::real_t>(3e-3) * static_cast<svmp::real_t>(gid);
      xcur[i * 3 + 0] += dx;
      xcur[i * 3 + 1] += dy;
      xcur[i * 3 + 2] += dz;
    }
    mesh.set_current_coords(xcur);
    mesh.use_current_configuration();
  }
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
      std::cout << "Skipping ghost-layer metadata test (designed for 2 or 4 ranks, got " << world << ")\n";
    }
    MPI_Finalize();
    return 0;
  }

  auto local_mesh = svmp::test::make_star_tet_mesh(rank, world);
  svmp::DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);

  std::vector<svmp::gid_t> marked_face_key;
  std::pair<svmp::gid_t, svmp::gid_t> marked_edge_key;
  svmp::label_t marked_face_label = svmp::INVALID_LABEL;
  svmp::label_t marked_edge_label = svmp::INVALID_LABEL;

  svmp::test::attach_metadata_and_fields(dmesh, rank, world,
                                         marked_face_key, marked_edge_key,
                                         marked_face_label, marked_edge_label);

  // Canonicalize face/edge GIDs via exchange pattern build; this is required for
  // face/edge metadata roundtrips (labels, sets, fields) across ghost rebuilds.
  dmesh.build_exchange_patterns();

  const auto expected = svmp::test::capture_expected(dmesh,
                                                     marked_face_key, marked_edge_key,
                                                     marked_face_label, marked_edge_label);

  // Sanity: base mesh is one local cell before ghosting.
  ASSERT_EQ(dmesh.local_mesh().n_cells(), 1u);

  // Build ghost layer and validate metadata preservation.
  dmesh.build_ghost_layer(1);

  if (rank == 0) {
    ASSERT_EQ(dmesh.local_mesh().n_cells(), static_cast<size_t>(world));
  } else {
    ASSERT_EQ(dmesh.local_mesh().n_cells(), 2u);
  }

  svmp::test::verify_expected(dmesh, expected);

  // Clear ghosts and validate metadata preservation.
  dmesh.clear_ghosts();
  ASSERT_EQ(dmesh.local_mesh().n_cells(), 1u);
  ASSERT_EQ(dmesh.local_mesh().n_vertices(), expected.base_n_vertices);

  svmp::test::verify_expected(dmesh, expected);

  if (rank == 0) {
    std::cout << "Ghost layer metadata preservation test PASSED (" << world << " ranks)\n";
  }

  MPI_Finalize();
  return 0;
}

