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

#include "MeshFieldsFixture.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace svmp {
namespace test {

// ========================================
// Helper: Create uniform triangle mesh
// ========================================
// Reuses the pattern from test_ErrorEstimator.cpp (TriangleMeshFixture)
// to maintain consistency with existing tests.

static MeshBase create_uniform_triangle_mesh(size_t nx, size_t ny, double dx = 1.0, double dy = 1.0) {
  MeshBase mesh(2);  // 2D mesh

  // Create vertices on (nx+1) x (ny+1) grid
  for (size_t j = 0; j <= ny; ++j) {
    for (size_t i = 0; i <= nx; ++i) {
      index_t vid = static_cast<index_t>(j * (nx + 1) + i);
      std::array<real_t, 3> pos = {
        static_cast<real_t>(i * dx),
        static_cast<real_t>(j * dy),
        0.0
      };
      mesh.add_vertex(vid, pos);
    }
  }

  // Create triangles (2 per quad)
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      index_t v0 = static_cast<index_t>(j * (nx + 1) + i);
      index_t v1 = static_cast<index_t>(j * (nx + 1) + i + 1);
      index_t v2 = static_cast<index_t>((j + 1) * (nx + 1) + i);
      index_t v3 = static_cast<index_t>((j + 1) * (nx + 1) + i + 1);

      // Lower triangle
      index_t tri1_id = static_cast<index_t>(2 * (j * nx + i));
      mesh.add_cell(tri1_id, CellFamily::Triangle, {v0, v1, v2});

      // Upper triangle
      index_t tri2_id = static_cast<index_t>(2 * (j * nx + i) + 1);
      mesh.add_cell(tri2_id, CellFamily::Triangle, {v1, v3, v2});
    }
  }

  mesh.finalize();
  return mesh;
}

// ========================================
// Factory Method Implementations
// ========================================

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_uniform_with_linear_field(
    size_t nx, size_t ny,
    double slope_x, double slope_y, double offset,
    const std::string& field_name) {

  MeshWithFieldsFixture fixture;
  fixture.mesh = create_uniform_triangle_mesh(nx, ny);

  // Attach field (using MeshBase API directly, as per project convention)
  auto handle = fixture.mesh.attach_field(
      EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

  // Fill field with linear function: f(x,y) = slope_x * x + slope_y * y + offset
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  size_t n_cells = fixture.mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    // Get cell centroid
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0, cy = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
      cy += pos[1];
    }
    cx /= verts.size();
    cy /= verts.size();

    // Linear field: f(x,y) = slope_x * x + slope_y * y + offset
    data[c] = slope_x * cx + slope_y * cy + offset;
  }

  fixture.register_field(field_name, handle);
  return fixture;
}

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(
    size_t nx, size_t ny,
    const std::string& field_name) {

  MeshWithFieldsFixture fixture;
  fixture.mesh = create_uniform_triangle_mesh(nx, ny);

  // Attach field
  auto handle = fixture.mesh.attach_field(
      EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

  // Fill field with quadratic function: f(x,y) = x^2 + y^2
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  size_t n_cells = fixture.mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0, cy = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
      cy += pos[1];
    }
    cx /= verts.size();
    cy /= verts.size();

    // Quadratic field: f(x,y) = x^2 + y^2
    data[c] = cx * cx + cy * cy;
  }

  fixture.register_field(field_name, handle);
  return fixture;
}

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(
    size_t nx, size_t ny,
    const std::string& field_name) {

  MeshWithFieldsFixture fixture;
  fixture.mesh = create_uniform_triangle_mesh(nx, ny);

  // Attach field
  auto handle = fixture.mesh.attach_field(
      EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

  // Fill field with step function: jump at x = 0.5 * nx
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  size_t n_cells = fixture.mesh.n_cells();
  double jump_x = static_cast<double>(nx) * 0.5;

  for (size_t c = 0; c < n_cells; ++c) {
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
    }
    cx /= verts.size();

    // Step function: jump at x = jump_x
    data[c] = (cx < jump_x) ? 0.0 : 1.0;
  }

  fixture.register_field(field_name, handle);
  return fixture;
}

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_uniform_with_constant_field(
    size_t nx, size_t ny,
    double value,
    const std::string& field_name) {

  MeshWithFieldsFixture fixture;
  fixture.mesh = create_uniform_triangle_mesh(nx, ny);

  // Attach field
  auto handle = fixture.mesh.attach_field(
      EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

  // Fill with constant value
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  size_t n_cells = fixture.mesh.n_cells();
  std::fill(data, data + n_cells, value);

  fixture.register_field(field_name, handle);
  return fixture;
}

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_uniform_with_multiple_fields(
    size_t nx, size_t ny,
    const std::vector<std::string>& field_names) {

  MeshWithFieldsFixture fixture;
  fixture.mesh = create_uniform_triangle_mesh(nx, ny);

  // Attach all requested fields
  for (const auto& field_name : field_names) {
    auto handle = fixture.mesh.attach_field(
        EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

    // Initialize to zeros
    double* data = static_cast<double*>(fixture.mesh.field_data(handle));
    size_t n_cells = fixture.mesh.n_cells();
    std::fill(data, data + n_cells, 0.0);

    fixture.register_field(field_name, handle);
  }

  return fixture;
}

MeshWithFieldsFixture MeshWithFieldsFixture::create_2d_anisotropic_with_linear_field(
    size_t nx, size_t ny,
    double slope_x, double slope_y, double offset,
    const std::string& field_name) {

  MeshWithFieldsFixture fixture;

  // Create anisotropic mesh with aspect ratio ~10:1 (stretched in y)
  fixture.mesh = create_uniform_triangle_mesh(nx, ny, 1.0, 0.1);

  // Attach field
  auto handle = fixture.mesh.attach_field(
      EntityKind::Volume, field_name, FieldScalarType::Float64, 1);

  // Fill field with linear function
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  size_t n_cells = fixture.mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    // Get cell centroid
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0, cy = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
      cy += pos[1];
    }
    cx /= verts.size();
    cy /= verts.size();

    // Linear field
    data[c] = slope_x * cx + slope_y * cy + offset;
  }

  fixture.register_field(field_name, handle);
  return fixture;
}

// ========================================
// Field Access Methods
// ========================================

FieldHandle MeshWithFieldsFixture::get_field(const std::string& name) const {
  auto it = field_map_.find(name);
  if (it == field_map_.end()) {
    throw std::runtime_error("Field not found: " + name);
  }
  return it->second;
}

}} // namespace svmp::test
