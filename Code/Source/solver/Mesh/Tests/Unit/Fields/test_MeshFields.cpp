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

#include <gtest/gtest.h>

#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFieldDescriptor.h"
#include "../../../Fields/MeshFields.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace svmp;

namespace {

MeshBase make_two_triangle_mesh_2d() {
  MeshBase mesh;
  const std::vector<real_t> coords = {
      0.0, 0.0,  // v0
      1.0, 0.0,  // v1
      1.0, 1.0,  // v2
      0.0, 1.0   // v3
  };

  const std::vector<offset_t> offsets = {0, 3, 6};
  const std::vector<index_t> conn = {0, 1, 2, 0, 2, 3};
  const std::vector<CellShape> shapes = {{CellFamily::Triangle, 3, 1}, {CellFamily::Triangle, 3, 1}};

  mesh.build_from_arrays(2, coords, offsets, conn, shapes);
  return mesh;
}

MeshBase make_1d_line_mesh(const std::vector<real_t>& x) {
  if (x.size() < 2) {
    throw std::invalid_argument("make_1d_line_mesh: need at least 2 vertices");
  }

  MeshBase mesh;
  std::vector<real_t> coords;
  coords.reserve(x.size());
  for (real_t xi : x) {
    coords.push_back(xi);
  }

  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  offsets.reserve(x.size());
  offsets.push_back(0);
  for (size_t i = 0; i + 1 < x.size(); ++i) {
    conn.push_back(static_cast<index_t>(i));
    conn.push_back(static_cast<index_t>(i + 1));
    offsets.push_back(static_cast<offset_t>(conn.size()));
    shapes.push_back({CellFamily::Line, 2, 1});
  }

  mesh.build_from_arrays(1, coords, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

} // namespace

TEST(MeshFieldsTest, AttachListAndLookupHandles) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto v_h = MeshFields::attach_field(mesh, EntityKind::Vertex, "v", FieldScalarType::Float64, 2);
  const auto c_h = MeshFields::attach_field(mesh, EntityKind::Volume, "c", FieldScalarType::Int32, 1);

  ASSERT_NE(v_h.id, 0u);
  ASSERT_NE(c_h.id, 0u);

  auto v_names = MeshFields::list_fields(mesh, EntityKind::Vertex);
  auto c_names = MeshFields::list_fields(mesh, EntityKind::Volume);

  ASSERT_EQ(v_names.size(), 1u);
  ASSERT_EQ(c_names.size(), 1u);
  EXPECT_EQ(v_names[0], "v");
  EXPECT_EQ(c_names[0], "c");

  const auto v2 = MeshFields::get_field_handle(mesh, EntityKind::Vertex, "v");
  const auto c2 = MeshFields::get_field_handle(mesh, EntityKind::Volume, "c");

  EXPECT_EQ(v2.id, v_h.id);
  EXPECT_EQ(v2.kind, EntityKind::Vertex);
  EXPECT_EQ(v2.name, "v");

  EXPECT_EQ(c2.id, c_h.id);
  EXPECT_EQ(c2.kind, EntityKind::Volume);
  EXPECT_EQ(c2.name, "c");

  EXPECT_EQ(MeshFields::get_field_handle(mesh, EntityKind::Vertex, "does_not_exist").id, 0u);
}

TEST(MeshFieldsTest, DescriptorAccess) {
  auto mesh = make_two_triangle_mesh_2d();

  // Field without descriptor -> nullptr
  const auto plain = MeshFields::attach_field(mesh, EntityKind::Vertex, "plain", FieldScalarType::Float64, 1);
  EXPECT_EQ(MeshFields::field_descriptor(mesh, plain), nullptr);

  FieldDescriptor desc = FieldDescriptor::vector(EntityKind::Vertex, 2, "m/s", true);
  desc.intent = FieldIntent::ReadOnly;
  desc.ghost_policy = FieldGhostPolicy::Exchange;
  desc.description = "Test field";

  const auto with_desc = MeshFields::attach_field_with_descriptor(
      mesh, EntityKind::Vertex, "vel", FieldScalarType::Float64, desc);

  const auto* got = MeshFields::field_descriptor(mesh, with_desc);
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got->location, EntityKind::Vertex);
  EXPECT_EQ(got->components, 2u);
  EXPECT_EQ(got->units, "m/s");
  EXPECT_TRUE(got->time_dependent);
  EXPECT_EQ(got->intent, FieldIntent::ReadOnly);
  EXPECT_EQ(got->ghost_policy, FieldGhostPolicy::Exchange);
  EXPECT_EQ(got->description, "Test field");
}

TEST(MeshFieldsTest, FieldManagerRegistry) {
  auto mesh = make_two_triangle_mesh_2d();
  FieldManager mgr(mesh);

  FieldDescriptor desc = FieldDescriptor::scalar(EntityKind::Vertex, "K", true);
  desc.intent = FieldIntent::ReadOnly;

  const auto h = mgr.attach("temperature", FieldScalarType::Float64, desc);
  ASSERT_NE(h.id, 0u);
  EXPECT_TRUE(mesh.has_field(EntityKind::Vertex, "temperature"));
  EXPECT_TRUE(mgr.has_descriptor(h));
  EXPECT_EQ(mgr.descriptor(h).units, "K");

  const auto ro = mgr.fields_with_intent(FieldIntent::ReadOnly);
  ASSERT_EQ(ro.size(), 1u);
  EXPECT_EQ(ro[0].id, h.id);
  EXPECT_EQ(ro[0].kind, EntityKind::Vertex);
  EXPECT_EQ(ro[0].name, "temperature");

  const auto td = mgr.time_dependent_fields();
  ASSERT_EQ(td.size(), 1u);
  EXPECT_EQ(td[0].id, h.id);
}

TEST(MeshFieldsTest, MemoryUsage) {
  auto mesh = make_two_triangle_mesh_2d();

  // Vertex: 4 vertices * 2 comps * 8 bytes = 64
  MeshFields::attach_field(mesh, EntityKind::Vertex, "v", FieldScalarType::Float64, 2);
  // Cell: 2 cells * 1 comp * 4 bytes = 8
  MeshFields::attach_field(mesh, EntityKind::Volume, "c", FieldScalarType::Int32, 1);

  EXPECT_EQ(MeshFields::field_memory_usage(mesh), 72u);
  EXPECT_EQ(MeshFields::total_field_count(mesh), 2u);
}

TEST(MeshFieldsTest, CopyField) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto src_h = MeshFields::attach_field(mesh, EntityKind::Vertex, "src", FieldScalarType::Float64, 1);
  const auto dst_h = MeshFields::attach_field(mesh, EntityKind::Vertex, "dst", FieldScalarType::Float64, 1);

  auto* src = MeshFields::field_data_as<real_t>(mesh, src_h);
  auto* dst = MeshFields::field_data_as<real_t>(mesh, dst_h);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);

  for (size_t i = 0; i < mesh.n_vertices(); ++i) {
    src[i] = static_cast<real_t>(i + 1);
    dst[i] = 0.0;
  }

  MeshFields::copy_field(mesh, src_h, dst_h);
  for (size_t i = 0; i < mesh.n_vertices(); ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }

  const auto bad_dst = MeshFields::attach_field(mesh, EntityKind::Vertex, "bad", FieldScalarType::Float64, 2);
  EXPECT_THROW(MeshFields::copy_field(mesh, src_h, bad_dst), std::runtime_error);
}

TEST(MeshFieldsTest, FillField) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto h = MeshFields::attach_field(mesh, EntityKind::Vertex, "fill", FieldScalarType::Float64, 1);
  MeshFields::fill_field_checked<real_t>(mesh, h, 3.14);

  const auto* data = MeshFields::field_data_as<real_t>(mesh, h);
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < mesh.n_vertices(); ++i) {
    EXPECT_DOUBLE_EQ(data[i], 3.14);
  }
}

TEST(MeshFieldsTest, CheckedTypedAccessors) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto f64 = MeshFields::attach_field(mesh, EntityKind::Vertex, "f64", FieldScalarType::Float64, 1);
  EXPECT_NO_THROW((void)MeshFields::field_data_as_checked<double>(mesh, f64));

  const auto i32 = MeshFields::attach_field(mesh, EntityKind::Vertex, "i32", FieldScalarType::Int32, 1);
  EXPECT_THROW((void)MeshFields::field_data_as_checked<double>(mesh, i32), std::invalid_argument);

  EXPECT_THROW(MeshFields::fill_field_checked<double>(mesh, i32, 1.0), std::invalid_argument);
}

TEST(EntityTransferMapTest, ValidateRejectsInvalidShapes) {
  EntityTransferMap map;
  map.kind = EntityKind::Vertex;
  map.src_count = 2;
  map.dst_count = 2;
  map.dst_offsets = {0, 1};  // wrong size
  map.src_indices = {0, 1};
  map.weights = {1.0, 1.0};
  EXPECT_THROW(map.validate(true), std::invalid_argument);

  map.dst_count = 1;
  map.dst_offsets = {1, 1};  // must start at 0
  map.src_indices = {0};
  map.weights = {1.0};
  EXPECT_THROW(map.validate(true), std::invalid_argument);

  map.dst_count = 2;
  map.dst_offsets = {0, 2, 1};  // not non-decreasing
  map.src_indices = {0, 1};
  map.weights = {1.0, 1.0};
  EXPECT_THROW(map.validate(true), std::invalid_argument);

  map.dst_offsets = {0, 1, 1};  // back() must equal nnz (=2)
  EXPECT_THROW(map.validate(true), std::invalid_argument);

  map.dst_offsets = {0, 1, 2};
  map.weights = {1.0};  // weights size mismatch
  EXPECT_THROW(map.validate(true), std::invalid_argument);

  map.weights = {1.0, 1.0};
  map.src_count = 1;
  map.src_indices = {0, 1};  // out of range
  EXPECT_THROW(map.validate(true), std::invalid_argument);
}

TEST(EntityTransferMapTest, NormalizeWeightsRejectsZeroSum) {
  EntityTransferMap map;
  map.kind = EntityKind::Vertex;
  map.src_count = 2;
  map.dst_count = 1;
  map.dst_offsets = {0, 2};
  map.src_indices = {0, 1};
  map.weights = {1.0, -1.0};
  EXPECT_THROW(map.normalize_weights(1e-15), std::invalid_argument);
}

TEST(MeshFieldsTest, StatsAndNorms) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto h = MeshFields::attach_field(mesh, EntityKind::Vertex, "x", FieldScalarType::Float64, 1);
  auto* data = MeshFields::field_data_as<real_t>(mesh, h);
  ASSERT_NE(data, nullptr);

  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  data[3] = 4.0;

  const auto stats = MeshFields::compute_stats(mesh, h);
  EXPECT_DOUBLE_EQ(stats.min, 1.0);
  EXPECT_DOUBLE_EQ(stats.max, 4.0);
  EXPECT_DOUBLE_EQ(stats.sum, 10.0);
  EXPECT_DOUBLE_EQ(stats.mean, 2.5);
  EXPECT_NEAR(stats.std_dev, std::sqrt(1.25), 1e-12);

  EXPECT_NEAR(MeshFields::compute_l2_norm(mesh, h), std::sqrt(30.0), 1e-12);
  EXPECT_DOUBLE_EQ(MeshFields::compute_inf_norm(mesh, h), 4.0);
}

TEST(MeshFieldsTest, InterpolationCellToVertexAndVertexToCell) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto cell_h = MeshFields::attach_field(mesh, EntityKind::Volume, "cell", FieldScalarType::Float64, 1);
  const auto vtx_h = MeshFields::attach_field(mesh, EntityKind::Vertex, "vtx", FieldScalarType::Float64, 1);

  auto* cell = MeshFields::field_data_as<real_t>(mesh, cell_h);
  auto* vtx = MeshFields::field_data_as<real_t>(mesh, vtx_h);
  ASSERT_NE(cell, nullptr);
  ASSERT_NE(vtx, nullptr);

  cell[0] = 2.0;
  cell[1] = 4.0;

  MeshFields::interpolate_cell_to_vertex(mesh, cell_h, vtx_h);

  EXPECT_DOUBLE_EQ(vtx[0], 3.0);
  EXPECT_DOUBLE_EQ(vtx[1], 2.0);
  EXPECT_DOUBLE_EQ(vtx[2], 3.0);
  EXPECT_DOUBLE_EQ(vtx[3], 4.0);

  // Now set a vertex field and interpolate to cells.
  vtx[0] = 0.0;
  vtx[1] = 1.0;
  vtx[2] = 2.0;
  vtx[3] = 3.0;

  MeshFields::interpolate_vertex_to_cell(mesh, vtx_h, cell_h);
  EXPECT_NEAR(cell[0], 1.0, 1e-12);          // (0+1+2)/3
  EXPECT_NEAR(cell[1], 5.0 / 3.0, 1e-12);    // (0+2+3)/3
}

TEST(MeshFieldsTest, ProlongateAndRestrictWithMap) {
  auto coarse = make_1d_line_mesh({0.0, 1.0});
  auto fine = make_1d_line_mesh({0.0, 0.5, 1.0});

  const auto u_coarse = MeshFields::attach_field(coarse, EntityKind::Vertex, "u", FieldScalarType::Float64, 1);
  const auto u_fine = MeshFields::attach_field(fine, EntityKind::Vertex, "u", FieldScalarType::Float64, 1);

  auto* uc = MeshFields::field_data_as_checked<real_t>(coarse, u_coarse);
  auto* uf = MeshFields::field_data_as_checked<real_t>(fine, u_fine);
  ASSERT_NE(uc, nullptr);
  ASSERT_NE(uf, nullptr);

  uc[0] = 0.0;
  uc[1] = 2.0;

  const std::vector<std::vector<real_t>> prolong_weights = {{1.0}, {0.5, 0.5}, {1.0}};
  EntityTransferMap prolong = EntityTransferMap::from_lists(
      EntityKind::Vertex,
      coarse.n_vertices(),
      {{0}, {0, 1}, {1}},
      &prolong_weights);

  MeshFields::prolongate_field(coarse, fine, u_coarse, u_fine, prolong);
  EXPECT_DOUBLE_EQ(uf[0], 0.0);
  EXPECT_DOUBLE_EQ(uf[1], 1.0);
  EXPECT_DOUBLE_EQ(uf[2], 2.0);

  // Restrict back using injection (take endpoints).
  EntityTransferMap restrict = EntityTransferMap::injection(
      EntityKind::Vertex,
      fine.n_vertices(),
      {0, 2});

  uc[0] = -1.0;
  uc[1] = -1.0;
  MeshFields::restrict_field(fine, coarse, u_fine, u_coarse, restrict);
  EXPECT_DOUBLE_EQ(uc[0], 0.0);
  EXPECT_DOUBLE_EQ(uc[1], 2.0);

  // Non-injection map on integer fields should throw.
  const auto i_coarse = MeshFields::attach_field(coarse, EntityKind::Vertex, "i", FieldScalarType::Int32, 1);
  const auto i_fine = MeshFields::attach_field(fine, EntityKind::Vertex, "i", FieldScalarType::Int32, 1);
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, i_coarse, i_fine, prolong), std::invalid_argument);
}

TEST(MeshFieldsTest, ProlongateMultiComponentFieldWithMap) {
  auto coarse = make_1d_line_mesh({0.0, 1.0});
  auto fine = make_1d_line_mesh({0.0, 0.5, 1.0});

  const auto u_coarse = MeshFields::attach_field(coarse, EntityKind::Vertex, "u", FieldScalarType::Float64, 3);
  const auto u_fine = MeshFields::attach_field(fine, EntityKind::Vertex, "u", FieldScalarType::Float64, 3);

  auto* uc = MeshFields::field_data_as_checked<real_t>(coarse, u_coarse);
  auto* uf = MeshFields::field_data_as_checked<real_t>(fine, u_fine);
  ASSERT_NE(uc, nullptr);
  ASSERT_NE(uf, nullptr);

  // coarse v0: (0,10,100), v1: (2,20,200)
  uc[0] = 0.0;
  uc[1] = 10.0;
  uc[2] = 100.0;
  uc[3] = 2.0;
  uc[4] = 20.0;
  uc[5] = 200.0;

  const std::vector<std::vector<real_t>> prolong_weights = {{1.0}, {0.5, 0.5}, {1.0}};
  EntityTransferMap prolong = EntityTransferMap::from_lists(
      EntityKind::Vertex,
      coarse.n_vertices(),
      {{0}, {0, 1}, {1}},
      &prolong_weights);

  MeshFields::prolongate_field(coarse, fine, u_coarse, u_fine, prolong);

  EXPECT_DOUBLE_EQ(uf[0], 0.0);
  EXPECT_DOUBLE_EQ(uf[1], 10.0);
  EXPECT_DOUBLE_EQ(uf[2], 100.0);

  EXPECT_DOUBLE_EQ(uf[3], 1.0);
  EXPECT_DOUBLE_EQ(uf[4], 15.0);
  EXPECT_DOUBLE_EQ(uf[5], 150.0);

  EXPECT_DOUBLE_EQ(uf[6], 2.0);
  EXPECT_DOUBLE_EQ(uf[7], 20.0);
  EXPECT_DOUBLE_EQ(uf[8], 200.0);

  // Float32 path too.
  const auto f_coarse = MeshFields::attach_field(coarse, EntityKind::Vertex, "f", FieldScalarType::Float32, 2);
  const auto f_fine = MeshFields::attach_field(fine, EntityKind::Vertex, "f", FieldScalarType::Float32, 2);

  auto* fc = MeshFields::field_data_as_checked<float>(coarse, f_coarse);
  auto* ff = MeshFields::field_data_as_checked<float>(fine, f_fine);
  ASSERT_NE(fc, nullptr);
  ASSERT_NE(ff, nullptr);

  fc[0] = 0.0f;
  fc[1] = 2.0f;
  fc[2] = 1.0f;
  fc[3] = 4.0f;

  MeshFields::prolongate_field(coarse, fine, f_coarse, f_fine, prolong);
  EXPECT_NEAR(ff[0], 0.0f, 1e-6f);
  EXPECT_NEAR(ff[1], 2.0f, 1e-6f);
  EXPECT_NEAR(ff[2], 0.5f, 1e-6f);
  EXPECT_NEAR(ff[3], 3.0f, 1e-6f);
  EXPECT_NEAR(ff[4], 1.0f, 1e-6f);
  EXPECT_NEAR(ff[5], 4.0f, 1e-6f);
}

TEST(MeshFieldsTest, TransferWithMapRejectsMismatchedMetadata) {
  auto coarse = make_1d_line_mesh({0.0, 1.0});
  auto fine = make_1d_line_mesh({0.0, 0.5, 1.0});

  const auto src = MeshFields::attach_field(coarse, EntityKind::Vertex, "src", FieldScalarType::Float64, 1);
  const auto dst = MeshFields::attach_field(fine, EntityKind::Vertex, "dst", FieldScalarType::Float64, 1);

  const std::vector<std::vector<real_t>> w = {{1.0}, {0.5, 0.5}, {1.0}};
  EntityTransferMap map = EntityTransferMap::from_lists(EntityKind::Vertex, coarse.n_vertices(), {{0}, {0, 1}, {1}}, &w);

  EntityTransferMap wrong_kind = map;
  wrong_kind.kind = EntityKind::Volume;
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, src, dst, wrong_kind), std::invalid_argument);

  EntityTransferMap wrong_src_count = map;
  wrong_src_count.src_count = coarse.n_vertices() + 1;
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, src, dst, wrong_src_count), std::invalid_argument);

  const auto dst_vec = MeshFields::attach_field(fine, EntityKind::Vertex, "dst_vec", FieldScalarType::Float64, 2);
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, src, dst_vec, map), std::runtime_error);

  const auto dst_int = MeshFields::attach_field(fine, EntityKind::Vertex, "dst_int", FieldScalarType::Int32, 1);
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, src, dst_int, map), std::runtime_error);
}

TEST(MeshFieldsTest, VolumeWeightedCellRestrictionMap) {
  auto fine = make_1d_line_mesh({0.0, 1.0, 4.0});   // two cells, lengths 1 and 3
  auto coarse = make_1d_line_mesh({0.0, 4.0});      // one cell

  const auto pf = MeshFields::attach_field(fine, EntityKind::Volume, "p", FieldScalarType::Float64, 1);
  const auto pc = MeshFields::attach_field(coarse, EntityKind::Volume, "p", FieldScalarType::Float64, 1);

  auto* fine_p = MeshFields::field_data_as_checked<real_t>(fine, pf);
  auto* coarse_p = MeshFields::field_data_as_checked<real_t>(coarse, pc);
  ASSERT_NE(fine_p, nullptr);
  ASSERT_NE(coarse_p, nullptr);
  ASSERT_EQ(fine.n_cells(), 2u);
  ASSERT_EQ(coarse.n_cells(), 1u);

  fine_p[0] = 10.0;
  fine_p[1] = 20.0;
  coarse_p[0] = 0.0;

  const EntityTransferMap map = MeshFields::make_volume_weighted_cell_restriction_map(
      fine, coarse, {{0, 1}}, Configuration::Reference);
  MeshFields::restrict_field(fine, coarse, pf, pc, map);

  EXPECT_NEAR(coarse_p[0], 17.5, 1e-12);
}

TEST(MeshFieldsTest, FieldManagerLifecycleDoesNotReturnStaleFields) {
  auto mesh = make_two_triangle_mesh_2d();
  FieldManager mgr(mesh);

  FieldDescriptor a_desc = FieldDescriptor::scalar(EntityKind::Vertex, "K", true);
  a_desc.intent = FieldIntent::ReadOnly;
  const auto a = mgr.attach("a", FieldScalarType::Float64, a_desc);
  ASSERT_NE(a.id, 0u);

  FieldDescriptor b_desc = FieldDescriptor::scalar(EntityKind::Vertex, "Pa", false);
  b_desc.intent = FieldIntent::ReadWrite;
  const auto b = mgr.attach("b", FieldScalarType::Float64, b_desc);
  ASSERT_NE(b.id, 0u);

  const auto ro_before = mgr.fields_with_intent(FieldIntent::ReadOnly);
  ASSERT_EQ(ro_before.size(), 1u);
  EXPECT_EQ(ro_before[0].name, "a");
  EXPECT_TRUE(mgr.has_descriptor(a));

  mesh.remove_field(a);
  EXPECT_FALSE(mgr.has_descriptor(a));
  EXPECT_THROW((mgr.descriptor(a)), std::runtime_error);
  EXPECT_TRUE(mgr.fields_with_intent(FieldIntent::ReadOnly).empty());

  // After mesh reset (IDs may be reused), the manager should reflect the current mesh state.
  mesh.clear();

  FieldDescriptor c_desc = FieldDescriptor::scalar(EntityKind::Vertex, "m", false);
  c_desc.intent = FieldIntent::ReadOnly;
  const auto c = mgr.attach("c", FieldScalarType::Float64, c_desc);
  ASSERT_NE(c.id, 0u);

  const auto ro_after = mgr.fields_with_intent(FieldIntent::ReadOnly);
  ASSERT_EQ(ro_after.size(), 1u);
  EXPECT_EQ(ro_after[0].name, "c");
  EXPECT_FALSE(mgr.has_descriptor(b));
}

TEST(MeshFieldsTest, FieldManagerQueryHelpers) {
  auto mesh = make_two_triangle_mesh_2d();
  FieldManager mgr(mesh);

  FieldDescriptor a_desc = FieldDescriptor::scalar(EntityKind::Vertex, "K", true);
  a_desc.intent = FieldIntent::ReadOnly;
  a_desc.ghost_policy = FieldGhostPolicy::None;
  mgr.attach("a", FieldScalarType::Float64, a_desc);

  FieldDescriptor b_desc = FieldDescriptor::scalar(EntityKind::Vertex, "Pa", false);
  b_desc.intent = FieldIntent::ReadWrite;
  b_desc.ghost_policy = FieldGhostPolicy::Exchange;
  mgr.attach("b", FieldScalarType::Float64, b_desc);

  const auto ro = mgr.fields_with_intent(FieldIntent::ReadOnly);
  ASSERT_EQ(ro.size(), 1u);
  EXPECT_EQ(ro[0].name, "a");

  const auto td = mgr.time_dependent_fields();
  ASSERT_EQ(td.size(), 1u);
  EXPECT_EQ(td[0].name, "a");

  const auto exch = mgr.fields_requiring_exchange();
  ASSERT_EQ(exch.size(), 1u);
  EXPECT_EQ(exch[0].name, "b");
}

TEST(MeshFieldsTest, FieldsWithGhostPolicyQuery) {
  auto mesh = make_two_triangle_mesh_2d();

  FieldDescriptor a_desc = FieldDescriptor::scalar(EntityKind::Vertex);
  a_desc.ghost_policy = FieldGhostPolicy::None;
  MeshFields::attach_field_with_descriptor(mesh, EntityKind::Vertex, "a", FieldScalarType::Float64, a_desc);

  FieldDescriptor b_desc = FieldDescriptor::scalar(EntityKind::Vertex);
  b_desc.ghost_policy = FieldGhostPolicy::Exchange;
  const auto b = MeshFields::attach_field_with_descriptor(mesh, EntityKind::Vertex, "b", FieldScalarType::Float64, b_desc);

  const auto exch = MeshFields::fields_with_ghost_policy(mesh, FieldGhostPolicy::Exchange);
  ASSERT_EQ(exch.size(), 1u);
  EXPECT_EQ(exch[0].id, b.id);
  EXPECT_EQ(exch[0].name, "b");

  const auto exch2 = MeshFields::fields_requiring_exchange(mesh);
  ASSERT_EQ(exch2.size(), 1u);
  EXPECT_EQ(exch2[0].id, b.id);
}

TEST(MeshFieldsTest, TransferCustomWithInjectionMap) {
  auto coarse = make_1d_line_mesh({0.0, 1.0});
  auto fine = make_1d_line_mesh({0.0, 0.5, 1.0});

  const auto src = MeshFields::attach_field(coarse, EntityKind::Vertex, "src", FieldScalarType::Custom, 2, 4);
  const auto dst = MeshFields::attach_field(fine, EntityKind::Vertex, "dst", FieldScalarType::Custom, 2, 4);

  auto* src_bytes = static_cast<uint8_t*>(MeshFields::field_data(coarse, src));
  auto* dst_bytes = static_cast<uint8_t*>(MeshFields::field_data(fine, dst));
  ASSERT_NE(src_bytes, nullptr);
  ASSERT_NE(dst_bytes, nullptr);

  const size_t bpe = MeshFields::field_bytes_per_entity(coarse, src);
  ASSERT_EQ(bpe, 8u);

  // Fill coarse with distinct patterns per vertex.
  for (size_t v = 0; v < coarse.n_vertices(); ++v) {
    for (size_t k = 0; k < bpe; ++k) {
      src_bytes[v * bpe + k] = static_cast<uint8_t>(10 * v + k);
    }
  }
  std::fill(dst_bytes, dst_bytes + fine.n_vertices() * bpe, 0);

  // Injection: fine[0] <- coarse[0], fine[1] <- coarse[0], fine[2] <- coarse[1]
  const EntityTransferMap inj = EntityTransferMap::injection(EntityKind::Vertex, coarse.n_vertices(), {0, 0, 1});
  MeshFields::prolongate_field(coarse, fine, src, dst, inj);

  for (size_t k = 0; k < bpe; ++k) {
    EXPECT_EQ(dst_bytes[0 * bpe + k], src_bytes[0 * bpe + k]);
    EXPECT_EQ(dst_bytes[1 * bpe + k], src_bytes[0 * bpe + k]);
    EXPECT_EQ(dst_bytes[2 * bpe + k], src_bytes[1 * bpe + k]);
  }

  // Non-injection should be rejected for Custom fields.
  const std::vector<std::vector<real_t>> w = {{1.0}, {0.5, 0.5}, {1.0}};
  const EntityTransferMap weighted = EntityTransferMap::from_lists(EntityKind::Vertex, coarse.n_vertices(), {{0}, {0, 1}, {1}}, &w);
  EXPECT_THROW(MeshFields::prolongate_field(coarse, fine, src, dst, weighted), std::invalid_argument);
}

TEST(MeshFieldsTest, TotalFieldCountAndMemoryUsage) {
  auto mesh = make_two_triangle_mesh_2d();

  MeshFields::attach_field(mesh, EntityKind::Vertex, "v", FieldScalarType::Float64, 2);
  MeshFields::attach_field(mesh, EntityKind::Volume, "c", FieldScalarType::Int32, 1);

  EXPECT_EQ(MeshFields::total_field_count(mesh), 2u);
  EXPECT_EQ(MeshFields::field_memory_usage(mesh), 4u * 2u * 8u + 2u * 1u * 4u);
}

TEST(MeshFieldsTest, RemoveFieldAndFillUnchecked) {
  auto mesh = make_two_triangle_mesh_2d();

  const auto h = MeshFields::attach_field(mesh, EntityKind::Vertex, "i", FieldScalarType::Int32, 1);
  EXPECT_TRUE(MeshFields::has_field(mesh, EntityKind::Vertex, "i"));

  MeshFields::fill_field<int32_t>(mesh, h, 7);
  const auto* data = MeshFields::field_data_as<int32_t>(mesh, h);
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < mesh.n_vertices(); ++i) {
    EXPECT_EQ(data[i], 7);
  }

  MeshFields::remove_field(mesh, h);
  EXPECT_FALSE(MeshFields::has_field(mesh, EntityKind::Vertex, "i"));
  EXPECT_EQ(MeshFields::get_field_handle(mesh, EntityKind::Vertex, "i").id, 0u);
}

TEST(MeshFieldsTest, RestrictAndProlongateWithoutMap) {
  auto a = make_1d_line_mesh({0.0, 1.0});
  auto b = make_1d_line_mesh({0.0, 1.0});

  const auto fa = MeshFields::attach_field(a, EntityKind::Vertex, "u", FieldScalarType::Float64, 1);
  const auto fb = MeshFields::attach_field(b, EntityKind::Vertex, "u", FieldScalarType::Float64, 1);

  auto* ua = MeshFields::field_data_as_checked<real_t>(a, fa);
  auto* ub = MeshFields::field_data_as_checked<real_t>(b, fb);
  ASSERT_NE(ua, nullptr);
  ASSERT_NE(ub, nullptr);

  ua[0] = 1.0;
  ua[1] = 2.0;
  ub[0] = 0.0;
  ub[1] = 0.0;

  MeshFields::prolongate_field(a, b, fa, fb);
  EXPECT_DOUBLE_EQ(ub[0], 1.0);
  EXPECT_DOUBLE_EQ(ub[1], 2.0);

  // Same-count restrict also uses identity map.
  ua[0] = 0.0;
  ua[1] = 0.0;
  MeshFields::restrict_field(b, a, fb, fa);
  EXPECT_DOUBLE_EQ(ua[0], 1.0);
  EXPECT_DOUBLE_EQ(ua[1], 2.0);

  // Mismatched counts require explicit maps.
  auto fine = make_1d_line_mesh({0.0, 0.5, 1.0});
  const auto ff = MeshFields::attach_field(fine, EntityKind::Vertex, "u", FieldScalarType::Float64, 1);
  EXPECT_THROW(MeshFields::prolongate_field(a, fine, fa, ff), std::runtime_error);
  EXPECT_THROW(MeshFields::restrict_field(fine, a, ff, fa), std::runtime_error);
}
