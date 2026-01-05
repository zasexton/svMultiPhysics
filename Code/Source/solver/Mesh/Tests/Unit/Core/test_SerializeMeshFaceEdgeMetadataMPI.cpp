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
 * @file test_SerializeMeshFaceEdgeMetadataMPI.cpp
 * @brief MPI test ensuring mesh serialization preserves face/edge metadata.
 *
 * This exercises the internal serialize/deserialize path used by several
 * DistributedMesh algorithms (root distribution, ghost rebuild, etc.).
 *
 * Production requirement: face boundary labels, edge labels, face/edge sets,
 * and face/edge fields (including FieldDescriptor) must roundtrip without
 * index-order contamination.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <algorithm>
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

static bool contains_gid(const std::vector<svmp::gid_t>& key, svmp::gid_t gid) {
  return std::find(key.begin(), key.end(), gid) != key.end();
}

static svmp::label_t expected_face_boundary_label(const std::vector<svmp::gid_t>& key, bool is_boundary) {
  if (!is_boundary) {
    return svmp::INVALID_LABEL;
  }

  // Label only a sparse subset of boundary faces.
  if (key.size() == 3 && key[0] == 10 && key[1] == 20 && key[2] == 40) {
    return 101;
  }
  if (key.size() == 3 && key[0] == 20 && key[1] == 30 && key[2] == 50) {
    return 102;
  }
  return svmp::INVALID_LABEL;
}

static svmp::label_t expected_edge_label(const std::pair<svmp::gid_t, svmp::gid_t>& key) {
  if (key.first == 10 && key.second == 40) {
    return 201;
  }
  if (key.first == 30 && key.second == 50) {
    return 202;
  }
  return svmp::INVALID_LABEL;
}

static svmp::real_t expected_face_value(const std::vector<svmp::gid_t>& key) {
  ASSERT_EQ(key.size(), 3u);
  // Encode the sorted GIDs into an exact integer (fits comfortably in double here).
  const long long code = static_cast<long long>(key[0]) * 10000LL +
                         static_cast<long long>(key[1]) * 100LL +
                         static_cast<long long>(key[2]);
  return static_cast<svmp::real_t>(code);
}

static svmp::real_t expected_edge_value(const std::pair<svmp::gid_t, svmp::gid_t>& key) {
  const long long code = static_cast<long long>(key.first) * 100LL + static_cast<long long>(key.second);
  return static_cast<svmp::real_t>(code);
}

static void reorder_faces_and_edges(svmp::MeshBase& mesh) {
  // Reorder faces and edges on the sender so receiver finalize() likely produces
  // a different local ordering; correctness requires GID-based remapping.
  const size_t n_faces = mesh.n_faces();
  std::vector<svmp::CellShape> face_shapes;
  std::vector<svmp::offset_t> face2vertex_offsets;
  std::vector<svmp::index_t> face2vertex;
  std::vector<std::array<svmp::index_t, 2>> face2cell;
  face_shapes.reserve(n_faces);
  face2cell.reserve(n_faces);
  face2vertex_offsets.reserve(n_faces + 1);
  face2vertex_offsets.push_back(0);

  for (size_t i = 0; i < n_faces; ++i) {
    const svmp::index_t f_old = static_cast<svmp::index_t>(n_faces - 1u - i);
    face_shapes.push_back(mesh.face_shapes()[static_cast<size_t>(f_old)]);

    auto [verts, n] = mesh.face_vertices_span(f_old);
    for (size_t j = 0; j < n; ++j) {
      face2vertex.push_back(verts[j]);
    }
    face2vertex_offsets.push_back(static_cast<svmp::offset_t>(face2vertex.size()));
    face2cell.push_back(mesh.face_cells(f_old));
  }

  mesh.set_faces_from_arrays(face_shapes, face2vertex_offsets, face2vertex, face2cell);

  const size_t n_edges = mesh.n_edges();
  std::vector<std::array<svmp::index_t, 2>> edge2vertex;
  edge2vertex.reserve(n_edges);
  for (size_t i = 0; i < n_edges; ++i) {
    edge2vertex.push_back(mesh.edge2vertex()[n_edges - 1u - i]);
  }
  mesh.set_edges_from_arrays(edge2vertex);
}

static void attach_face_edge_metadata(svmp::MeshBase& mesh) {
  // Sets
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
    const auto key = face_key(mesh, f);
    const auto fc = mesh.face_cells(f);
    const bool is_boundary = (fc[0] == svmp::INVALID_INDEX) || (fc[1] == svmp::INVALID_INDEX);

    if (contains_gid(key, 10)) {
      mesh.add_to_set(svmp::EntityKind::Face, "faces_with_10", f);
    }
    if (is_boundary) {
      mesh.add_to_set(svmp::EntityKind::Face, "boundary_faces", f);
    }

    const auto label = expected_face_boundary_label(key, is_boundary);
    if (label != svmp::INVALID_LABEL) {
      mesh.set_boundary_label(f, label);
    }
  }

  for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
    const auto key = edge_key(mesh, e);

    if (key.first == 10 || key.second == 10) {
      mesh.add_to_set(svmp::EntityKind::Edge, "edges_with_10", e);
    }
    if (key.first < 40 && key.second < 40) {
      mesh.add_to_set(svmp::EntityKind::Edge, "edges_lt_40", e);
    }

    const auto label = expected_edge_label(key);
    if (label != svmp::INVALID_LABEL) {
      mesh.set_edge_label(e, label);
    }
  }

  // Fields (Face + Edge)
  const auto f_h =
      mesh.attach_field(svmp::EntityKind::Face, "fval", svmp::FieldScalarType::Float64, 1);
  auto* f_data = mesh.field_data_as<svmp::real_t>(f_h);
  ASSERT(f_data);
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh.n_faces()); ++f) {
    f_data[f] = expected_face_value(face_key(mesh, f));
  }

  svmp::FieldDescriptor f_desc;
  f_desc.location = svmp::EntityKind::Face;
  f_desc.components = 1;
  f_desc.component_names = {"p"};
  f_desc.units = "Pa";
  f_desc.unit_scale = 1e-3;
  f_desc.time_dependent = true;
  f_desc.intent = svmp::FieldIntent::Temporary;
  f_desc.ghost_policy = svmp::FieldGhostPolicy::Exchange;
  f_desc.description = "test face scalar";
  mesh.set_field_descriptor(f_h, f_desc);

  const auto e_h =
      mesh.attach_field(svmp::EntityKind::Edge, "eval", svmp::FieldScalarType::Float64, 1);
  auto* e_data = mesh.field_data_as<svmp::real_t>(e_h);
  ASSERT(e_data);
  for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh.n_edges()); ++e) {
    e_data[e] = expected_edge_value(edge_key(mesh, e));
  }

  svmp::FieldDescriptor e_desc;
  e_desc.location = svmp::EntityKind::Edge;
  e_desc.components = 1;
  e_desc.component_names = {"e"};
  e_desc.units = "m";
  e_desc.unit_scale = 2.0;
  e_desc.time_dependent = false;
  e_desc.intent = svmp::FieldIntent::ReadOnly;
  e_desc.ghost_policy = svmp::FieldGhostPolicy::Accumulate;
  e_desc.description = "test edge scalar";
  mesh.set_field_descriptor(e_h, e_desc);

  // Add a vertex/cell field with descriptors to ensure those descriptors roundtrip too.
  const auto v_h =
      mesh.attach_field(svmp::EntityKind::Vertex, "vval", svmp::FieldScalarType::Float64, 1);
  auto* v_data = mesh.field_data_as<svmp::real_t>(v_h);
  ASSERT(v_data);
  const auto& vg = mesh.vertex_gids();
  for (size_t i = 0; i < vg.size(); ++i) {
    v_data[i] = 0.5 * static_cast<svmp::real_t>(vg[i]);
  }

  svmp::FieldDescriptor v_desc = svmp::FieldDescriptor::scalar(svmp::EntityKind::Vertex, "mm", false);
  v_desc.unit_scale = 1e-3;
  v_desc.intent = svmp::FieldIntent::ReadWrite;
  v_desc.ghost_policy = svmp::FieldGhostPolicy::Exchange;
  v_desc.description = "test vertex scalar";
  mesh.set_field_descriptor(v_h, v_desc);

  const auto c_h =
      mesh.attach_field(svmp::EntityKind::Volume, "cval", svmp::FieldScalarType::Float64, 1);
  auto* c_data = mesh.field_data_as<svmp::real_t>(c_h);
  ASSERT(c_data);
  const auto& cg = mesh.cell_gids();
  for (size_t i = 0; i < cg.size(); ++i) {
    c_data[i] = static_cast<svmp::real_t>(cg[i]) + 0.25;
  }

  svmp::FieldDescriptor c_desc = svmp::FieldDescriptor::scalar(svmp::EntityKind::Volume, "", false);
  c_desc.intent = svmp::FieldIntent::ReadWrite;
  c_desc.ghost_policy = svmp::FieldGhostPolicy::None;
  c_desc.description = "test cell scalar";
  mesh.set_field_descriptor(c_h, c_desc);
}

static svmp::MeshBase make_mesh() {
  const int dim = 3;

  // Two tetrahedra sharing the face (v0,v1,v2).
  std::vector<svmp::real_t> coords = {
      0.0, 0.0, 0.0, // v0
      1.0, 0.0, 0.0, // v1
      0.0, 1.0, 0.0, // v2
      0.0, 0.0, 1.0, // v3
      0.0, 0.0, -1.0 // v4
  };

  std::vector<svmp::offset_t> offsets = {0, 4, 8};
  std::vector<svmp::index_t> conn = {
      0, 1, 2, 3, // tet 0
      0, 1, 2, 4  // tet 1
  };
  std::vector<svmp::CellShape> shapes = {
      {svmp::CellFamily::Tetra, 4, 1},
      {svmp::CellFamily::Tetra, 4, 1},
  };

  svmp::MeshBase mesh;
  mesh.build_from_arrays(dim, coords, offsets, conn, shapes);
  mesh.set_vertex_gids({10, 20, 30, 40, 50});
  mesh.set_cell_gids({100, 200});
  mesh.finalize();

  reorder_faces_and_edges(mesh);
  attach_face_edge_metadata(mesh);

  return mesh;
}

static void verify_descriptor(const svmp::MeshBase& mesh,
                              svmp::EntityKind kind,
                              const std::string& name,
                              const svmp::FieldDescriptor& expected) {
  ASSERT(mesh.has_field(kind, name));
  const auto h = mesh.field_handle(kind, name);
  const auto* desc = mesh.field_descriptor(h);
  ASSERT(desc);
  ASSERT_EQ(desc->location, expected.location);
  ASSERT_EQ(desc->components, expected.components);
  ASSERT_NEAR(desc->unit_scale, expected.unit_scale, 1e-15);
  ASSERT_EQ(desc->time_dependent, expected.time_dependent);
  ASSERT_EQ(desc->intent, expected.intent);
  ASSERT_EQ(desc->ghost_policy, expected.ghost_policy);
  ASSERT_EQ(desc->units, expected.units);
  ASSERT_EQ(desc->description, expected.description);
  ASSERT_EQ(desc->component_names, expected.component_names);
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

#if !defined(MESH_BUILD_TESTS)
  if (rank == 0) {
    std::cout << "Skipping serialize/deserialize metadata test (MESH_BUILD_TESTS disabled)\n";
  }
  MPI_Finalize();
  return 0;
#else
  std::vector<char> buffer;

  if (rank == 0) {
    auto mesh = svmp::test::make_mesh();
    svmp::test::internal::serialize_mesh_for_test(mesh, buffer);
  }

  int buffer_size = static_cast<int>(buffer.size());
  MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  ASSERT(buffer_size >= 0);
  buffer.resize(static_cast<size_t>(buffer_size));
  if (buffer_size > 0) {
    MPI_Bcast(buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  svmp::MeshBase mesh2;
  svmp::test::internal::deserialize_mesh_for_test(buffer, mesh2);

  ASSERT_EQ(mesh2.n_vertices(), 5u);
  ASSERT_EQ(mesh2.n_cells(), 2u);
  ASSERT_EQ(mesh2.n_faces(), 7u);
  ASSERT_EQ(mesh2.n_edges(), 9u);

  // Face labels + face field values
  ASSERT(mesh2.has_field(svmp::EntityKind::Face, "fval"));
  const auto f_h = mesh2.field_handle(svmp::EntityKind::Face, "fval");
  const auto* f_data = mesh2.field_data_as<const svmp::real_t>(f_h);
  ASSERT(f_data);

  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh2.n_faces()); ++f) {
    const auto key = svmp::test::face_key(mesh2, f);
    const auto fc = mesh2.face_cells(f);
    const bool is_boundary = (fc[0] == svmp::INVALID_INDEX) || (fc[1] == svmp::INVALID_INDEX);
    const auto expected_label = svmp::test::expected_face_boundary_label(key, is_boundary);
    ASSERT_EQ(mesh2.boundary_label(f), expected_label);
    ASSERT_NEAR(f_data[f], svmp::test::expected_face_value(key), 1e-12);
  }

  // Edge labels + edge field values
  ASSERT(mesh2.has_field(svmp::EntityKind::Edge, "eval"));
  const auto e_h = mesh2.field_handle(svmp::EntityKind::Edge, "eval");
  const auto* e_data = mesh2.field_data_as<const svmp::real_t>(e_h);
  ASSERT(e_data);

  for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh2.n_edges()); ++e) {
    const auto key = svmp::test::edge_key(mesh2, e);
    ASSERT_EQ(mesh2.edge_label(e), svmp::test::expected_edge_label(key));
    ASSERT_NEAR(e_data[e], svmp::test::expected_edge_value(key), 1e-12);
  }

  // Sets
  auto face_set_keys = [&](const std::string& name) -> std::set<std::vector<svmp::gid_t>> {
    ASSERT(mesh2.has_set(svmp::EntityKind::Face, name));
    std::set<std::vector<svmp::gid_t>> out;
    for (const auto f : mesh2.get_set(svmp::EntityKind::Face, name)) {
      ASSERT(f >= 0);
      ASSERT(static_cast<size_t>(f) < mesh2.n_faces());
      out.insert(svmp::test::face_key(mesh2, f));
    }
    return out;
  };
  auto edge_set_keys = [&](const std::string& name) -> std::set<std::pair<svmp::gid_t, svmp::gid_t>> {
    ASSERT(mesh2.has_set(svmp::EntityKind::Edge, name));
    std::set<std::pair<svmp::gid_t, svmp::gid_t>> out;
    for (const auto e : mesh2.get_set(svmp::EntityKind::Edge, name)) {
      ASSERT(e >= 0);
      ASSERT(static_cast<size_t>(e) < mesh2.n_edges());
      out.insert(svmp::test::edge_key(mesh2, e));
    }
    return out;
  };

  std::set<std::vector<svmp::gid_t>> expected_faces_with_10;
  std::set<std::vector<svmp::gid_t>> expected_boundary_faces;
  for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh2.n_faces()); ++f) {
    const auto key = svmp::test::face_key(mesh2, f);
    const auto fc = mesh2.face_cells(f);
    const bool is_boundary = (fc[0] == svmp::INVALID_INDEX) || (fc[1] == svmp::INVALID_INDEX);
    if (svmp::test::contains_gid(key, 10)) expected_faces_with_10.insert(key);
    if (is_boundary) expected_boundary_faces.insert(key);
  }

  ASSERT(face_set_keys("faces_with_10") == expected_faces_with_10);
  ASSERT(face_set_keys("boundary_faces") == expected_boundary_faces);

  std::set<std::pair<svmp::gid_t, svmp::gid_t>> expected_edges_with_10;
  std::set<std::pair<svmp::gid_t, svmp::gid_t>> expected_edges_lt_40;
  for (svmp::index_t e = 0; e < static_cast<svmp::index_t>(mesh2.n_edges()); ++e) {
    const auto key = svmp::test::edge_key(mesh2, e);
    if (key.first == 10 || key.second == 10) expected_edges_with_10.insert(key);
    if (key.first < 40 && key.second < 40) expected_edges_lt_40.insert(key);
  }

  ASSERT(edge_set_keys("edges_with_10") == expected_edges_with_10);
  ASSERT(edge_set_keys("edges_lt_40") == expected_edges_lt_40);

  // Field descriptors (Face + Edge + Vertex + Volume)
  {
    svmp::FieldDescriptor f_desc;
    f_desc.location = svmp::EntityKind::Face;
    f_desc.components = 1;
    f_desc.component_names = {"p"};
    f_desc.units = "Pa";
    f_desc.unit_scale = 1e-3;
    f_desc.time_dependent = true;
    f_desc.intent = svmp::FieldIntent::Temporary;
    f_desc.ghost_policy = svmp::FieldGhostPolicy::Exchange;
    f_desc.description = "test face scalar";
    svmp::test::verify_descriptor(mesh2, svmp::EntityKind::Face, "fval", f_desc);

    svmp::FieldDescriptor e_desc;
    e_desc.location = svmp::EntityKind::Edge;
    e_desc.components = 1;
    e_desc.component_names = {"e"};
    e_desc.units = "m";
    e_desc.unit_scale = 2.0;
    e_desc.time_dependent = false;
    e_desc.intent = svmp::FieldIntent::ReadOnly;
    e_desc.ghost_policy = svmp::FieldGhostPolicy::Accumulate;
    e_desc.description = "test edge scalar";
    svmp::test::verify_descriptor(mesh2, svmp::EntityKind::Edge, "eval", e_desc);

    svmp::FieldDescriptor v_desc = svmp::FieldDescriptor::scalar(svmp::EntityKind::Vertex, "mm", false);
    v_desc.unit_scale = 1e-3;
    v_desc.intent = svmp::FieldIntent::ReadWrite;
    v_desc.ghost_policy = svmp::FieldGhostPolicy::Exchange;
    v_desc.description = "test vertex scalar";
    svmp::test::verify_descriptor(mesh2, svmp::EntityKind::Vertex, "vval", v_desc);

    svmp::FieldDescriptor c_desc = svmp::FieldDescriptor::scalar(svmp::EntityKind::Volume, "", false);
    c_desc.intent = svmp::FieldIntent::ReadWrite;
    c_desc.ghost_policy = svmp::FieldGhostPolicy::None;
    c_desc.description = "test cell scalar";
    svmp::test::verify_descriptor(mesh2, svmp::EntityKind::Volume, "cval", c_desc);
  }

  // Spot-check vertex/cell field values.
  {
    ASSERT(mesh2.has_field(svmp::EntityKind::Vertex, "vval"));
    const auto vh = mesh2.field_handle(svmp::EntityKind::Vertex, "vval");
    const auto* v_data = mesh2.field_data_as<const svmp::real_t>(vh);
    ASSERT(v_data);
    const auto& vg = mesh2.vertex_gids();
    ASSERT_EQ(vg.size(), mesh2.n_vertices());
    for (size_t i = 0; i < vg.size(); ++i) {
      ASSERT_NEAR(v_data[i], 0.5 * static_cast<svmp::real_t>(vg[i]), 1e-12);
    }

    ASSERT(mesh2.has_field(svmp::EntityKind::Volume, "cval"));
    const auto ch = mesh2.field_handle(svmp::EntityKind::Volume, "cval");
    const auto* c_data = mesh2.field_data_as<const svmp::real_t>(ch);
    ASSERT(c_data);
    const auto& cg = mesh2.cell_gids();
    ASSERT_EQ(cg.size(), mesh2.n_cells());
    for (size_t i = 0; i < cg.size(); ++i) {
      ASSERT_NEAR(c_data[i], static_cast<svmp::real_t>(cg[i]) + 0.25, 1e-12);
    }
  }

  if (rank == 0) {
    std::cout << "Serialize/deserialize face+edge metadata test PASSED (" << world << " ranks)\n";
  }

  MPI_Finalize();
  return 0;
#endif
}
