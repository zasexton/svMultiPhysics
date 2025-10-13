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

#include "MeshBuilders.h"
#include "../Core/MeshBase.h"
#include "../Topology/CellShape.h"
#include <stdexcept>
#include <numeric>
#include <cmath>

namespace svmp {

// ---- Cartesian mesh builders ----

MeshBase MeshBuilders::build_cartesian_2d(int nx, int ny,
                                         const std::array<real_t,2>& origin,
                                         const std::array<real_t,2>& spacing) {
  if (nx <= 0 || ny <= 0) {
    throw std::invalid_argument("Grid dimensions must be positive");
  }

  MeshBase mesh(2);

  // Generate nodes
  size_t n_nodes = (nx + 1) * (ny + 1);
  std::vector<real_t> coords;
  coords.reserve(n_nodes * 2);

  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      coords.push_back(origin[0] + i * spacing[0]);
      coords.push_back(origin[1] + j * spacing[1]);
    }
  }

  // Generate cells (quads)
  size_t n_cells = nx * ny;
  std::vector<offset_t> offsets;
  std::vector<index_t> connectivity;
  std::vector<CellShape> shapes;

  offsets.reserve(n_cells + 1);
  connectivity.reserve(n_cells * 4);
  shapes.reserve(n_cells);

  offsets.push_back(0);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      // Add quad nodes in counter-clockwise order
      index_t n0 = j * (nx + 1) + i;
      index_t n1 = n0 + 1;
      index_t n2 = n1 + (nx + 1);
      index_t n3 = n0 + (nx + 1);

      connectivity.push_back(n0);
      connectivity.push_back(n1);
      connectivity.push_back(n2);
      connectivity.push_back(n3);

      offsets.push_back(offsets.back() + 4);

      CellShape shape;
      shape.family = CellFamily::Quadrilateral;
      shape.order = 1;
      shapes.push_back(shape);
    }
  }

  mesh.build_from_arrays(2, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_cartesian_3d(int nx, int ny, int nz,
                                         const std::array<real_t,3>& origin,
                                         const std::array<real_t,3>& spacing) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument("Grid dimensions must be positive");
  }

  MeshBase mesh(3);

  // Generate nodes
  size_t n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
  std::vector<real_t> coords;
  coords.reserve(n_nodes * 3);

  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        coords.push_back(origin[0] + i * spacing[0]);
        coords.push_back(origin[1] + j * spacing[1]);
        coords.push_back(origin[2] + k * spacing[2]);
      }
    }
  }

  // Generate cells (hexes)
  size_t n_cells = nx * ny * nz;
  std::vector<offset_t> offsets;
  std::vector<index_t> connectivity;
  std::vector<CellShape> shapes;

  offsets.reserve(n_cells + 1);
  connectivity.reserve(n_cells * 8);
  shapes.reserve(n_cells);

  offsets.push_back(0);

  int nx1 = nx + 1;
  int ny1 = ny + 1;

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        // Add hex nodes in VTK order
        index_t n0 = k * ny1 * nx1 + j * nx1 + i;
        index_t n1 = n0 + 1;
        index_t n2 = n0 + nx1 + 1;
        index_t n3 = n0 + nx1;
        index_t n4 = n0 + ny1 * nx1;
        index_t n5 = n4 + 1;
        index_t n6 = n4 + nx1 + 1;
        index_t n7 = n4 + nx1;

        connectivity.push_back(n0);
        connectivity.push_back(n1);
        connectivity.push_back(n2);
        connectivity.push_back(n3);
        connectivity.push_back(n4);
        connectivity.push_back(n5);
        connectivity.push_back(n6);
        connectivity.push_back(n7);

        offsets.push_back(offsets.back() + 8);

        CellShape shape;
        shape.family = CellFamily::Hexahedron;
        shape.order = 1;
        shapes.push_back(shape);
      }
    }
  }

  mesh.build_from_arrays(3, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_cartesian_box(const BoundingBox& box,
                                          const std::array<int,3>& divisions) {
  std::array<real_t,3> origin = box.min;
  std::array<real_t,3> spacing;

  for (int i = 0; i < 3; ++i) {
    if (divisions[i] <= 0) {
      throw std::invalid_argument("Divisions must be positive");
    }
    spacing[i] = (box.max[i] - box.min[i]) / divisions[i];
  }

  return build_cartesian_3d(divisions[0], divisions[1], divisions[2], origin, spacing);
}

// ---- Simple shape builders ----

MeshBase MeshBuilders::build_triangle(const std::array<std::array<real_t,2>,3>& vertices) {
  MeshBase mesh(2);

  // Create nodes
  std::vector<real_t> coords;
  coords.reserve(6);
  for (const auto& v : vertices) {
    coords.push_back(v[0]);
    coords.push_back(v[1]);
  }

  // Create triangle cell
  std::vector<offset_t> offsets = {0, 3};
  std::vector<index_t> connectivity = {0, 1, 2};

  CellShape shape;
  shape.family = CellFamily::Triangle;
  shape.order = 1;
  std::vector<CellShape> shapes = {shape};

  mesh.build_from_arrays(2, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_quadrilateral(const std::array<std::array<real_t,2>,4>& vertices) {
  MeshBase mesh(2);

  // Create nodes
  std::vector<real_t> coords;
  coords.reserve(8);
  for (const auto& v : vertices) {
    coords.push_back(v[0]);
    coords.push_back(v[1]);
  }

  // Create quad cell
  std::vector<offset_t> offsets = {0, 4};
  std::vector<index_t> connectivity = {0, 1, 2, 3};

  CellShape shape;
  shape.family = CellFamily::Quadrilateral;
  shape.order = 1;
  std::vector<CellShape> shapes = {shape};

  mesh.build_from_arrays(2, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_tetrahedron(const std::array<std::array<real_t,3>,4>& vertices) {
  MeshBase mesh(3);

  // Create nodes
  std::vector<real_t> coords;
  coords.reserve(12);
  for (const auto& v : vertices) {
    coords.push_back(v[0]);
    coords.push_back(v[1]);
    coords.push_back(v[2]);
  }

  // Create tet cell
  std::vector<offset_t> offsets = {0, 4};
  std::vector<index_t> connectivity = {0, 1, 2, 3};

  CellShape shape;
  shape.family = CellFamily::Tetrahedron;
  shape.order = 1;
  std::vector<CellShape> shapes = {shape};

  mesh.build_from_arrays(3, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_hexahedron(const std::array<std::array<real_t,3>,8>& vertices) {
  MeshBase mesh(3);

  // Create nodes
  std::vector<real_t> coords;
  coords.reserve(24);
  for (const auto& v : vertices) {
    coords.push_back(v[0]);
    coords.push_back(v[1]);
    coords.push_back(v[2]);
  }

  // Create hex cell (VTK ordering)
  std::vector<offset_t> offsets = {0, 8};
  std::vector<index_t> connectivity = {0, 1, 2, 3, 4, 5, 6, 7};

  CellShape shape;
  shape.family = CellFamily::Hexahedron;
  shape.order = 1;
  std::vector<CellShape> shapes = {shape};

  mesh.build_from_arrays(3, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

// ---- Extrusion builders ----

MeshBase MeshBuilders::extrude_2d_to_3d(const MeshBase& mesh_2d,
                                       real_t height,
                                       int n_layers) {
  if (mesh_2d.dim() != 2) {
    throw std::invalid_argument("Input mesh must be 2D");
  }
  if (n_layers <= 0) {
    throw std::invalid_argument("Number of layers must be positive");
  }

  MeshBase mesh_3d(3);

  // Generate 3D nodes
  size_t n_nodes_2d = mesh_2d.n_nodes();
  size_t n_nodes_3d = n_nodes_2d * (n_layers + 1);
  std::vector<real_t> coords_3d;
  coords_3d.reserve(n_nodes_3d * 3);

  const auto& coords_2d = mesh_2d.X_ref();
  real_t dz = height / n_layers;

  for (int layer = 0; layer <= n_layers; ++layer) {
    real_t z = layer * dz;
    for (size_t n = 0; n < n_nodes_2d; ++n) {
      coords_3d.push_back(coords_2d[n * 2]);
      coords_3d.push_back(coords_2d[n * 2 + 1]);
      coords_3d.push_back(z);
    }
  }

  // Generate 3D cells
  size_t n_cells_2d = mesh_2d.n_cells();
  size_t n_cells_3d = n_cells_2d * n_layers;
  std::vector<offset_t> offsets_3d;
  std::vector<index_t> connectivity_3d;
  std::vector<CellShape> shapes_3d;

  offsets_3d.reserve(n_cells_3d + 1);
  shapes_3d.reserve(n_cells_3d);
  offsets_3d.push_back(0);

  const auto& offsets_2d = mesh_2d.cell2node_offsets();
  const auto& connectivity_2d = mesh_2d.cell2node();
  const auto& shapes_2d = mesh_2d.cell_shapes();

  for (int layer = 0; layer < n_layers; ++layer) {
    for (size_t c = 0; c < n_cells_2d; ++c) {
      size_t n_cell_nodes = offsets_2d[c+1] - offsets_2d[c];
      CellShape shape_2d = shapes_2d[c];
      CellShape shape_3d;

      if (shape_2d.family == CellFamily::Triangle && n_cell_nodes == 3) {
        // Extrude triangle to wedge (prism)
        shape_3d.family = CellFamily::Wedge;
        shape_3d.order = 1;

        // Bottom triangle
        for (size_t i = 0; i < 3; ++i) {
          index_t node_2d = connectivity_2d[offsets_2d[c] + i];
          connectivity_3d.push_back(node_2d + layer * n_nodes_2d);
        }

        // Top triangle
        for (size_t i = 0; i < 3; ++i) {
          index_t node_2d = connectivity_2d[offsets_2d[c] + i];
          connectivity_3d.push_back(node_2d + (layer + 1) * n_nodes_2d);
        }

        offsets_3d.push_back(offsets_3d.back() + 6);

      } else if (shape_2d.family == CellFamily::Quadrilateral && n_cell_nodes == 4) {
        // Extrude quad to hex
        shape_3d.family = CellFamily::Hexahedron;
        shape_3d.order = 1;

        // Bottom quad
        for (size_t i = 0; i < 4; ++i) {
          index_t node_2d = connectivity_2d[offsets_2d[c] + i];
          connectivity_3d.push_back(node_2d + layer * n_nodes_2d);
        }

        // Top quad
        for (size_t i = 0; i < 4; ++i) {
          index_t node_2d = connectivity_2d[offsets_2d[c] + i];
          connectivity_3d.push_back(node_2d + (layer + 1) * n_nodes_2d);
        }

        offsets_3d.push_back(offsets_3d.back() + 8);

      } else {
        throw std::runtime_error("Unsupported 2D cell type for extrusion");
      }

      shapes_3d.push_back(shape_3d);
    }
  }

  mesh_3d.build_from_arrays(3, coords_3d, offsets_3d, connectivity_3d, shapes_3d);
  mesh_3d.finalize();

  return mesh_3d;
}

MeshBase MeshBuilders::revolve_2d_around_axis(const MeshBase& mesh_2d,
                                             const std::array<real_t,3>& axis,
                                             real_t angle,
                                             int n_sectors) {
  if (mesh_2d.dim() != 2) {
    throw std::invalid_argument("Input mesh must be 2D");
  }
  if (n_sectors <= 0) {
    throw std::invalid_argument("Number of sectors must be positive");
  }

  // For simplicity, assume rotation around z-axis
  // Full implementation would handle arbitrary axis
  return extrude_2d_to_3d(mesh_2d, 1.0, n_sectors);  // Placeholder
}

// ---- Refinement builders ----

MeshBase MeshBuilders::refine_uniformly(const MeshBase& mesh) {
  // Uniform refinement by edge bisection
  // This is a complex operation that would need full implementation
  throw std::runtime_error("Uniform refinement not yet implemented");
}

MeshBase MeshBuilders::refine_by_marking(const MeshBase& mesh,
                                        const std::vector<index_t>& marked_cells) {
  // Adaptive refinement of marked cells
  throw std::runtime_error("Adaptive refinement not yet implemented");
}

MeshBase MeshBuilders::coarsen_uniformly(const MeshBase& mesh) {
  // Uniform coarsening
  throw std::runtime_error("Uniform coarsening not yet implemented");
}

// ---- Mesh combination ----

MeshBase MeshBuilders::merge_meshes(const std::vector<MeshBase>& meshes) {
  if (meshes.empty()) {
    throw std::invalid_argument("No meshes to merge");
  }

  // Check that all meshes have same dimension
  int dim = meshes[0].dim();
  for (size_t i = 1; i < meshes.size(); ++i) {
    if (meshes[i].dim() != dim) {
      throw std::invalid_argument("All meshes must have same dimension");
    }
  }

  // Count total entities
  size_t total_nodes = 0;
  size_t total_cells = 0;
  size_t total_connectivity = 0;

  for (const auto& m : meshes) {
    total_nodes += m.n_nodes();
    total_cells += m.n_cells();
    const auto& offsets = m.cell2node_offsets();
    if (!offsets.empty()) {
      total_connectivity += offsets.back();
    }
  }

  // Merge coordinates
  std::vector<real_t> merged_coords;
  merged_coords.reserve(total_nodes * dim);

  for (const auto& m : meshes) {
    const auto& coords = m.X_ref();
    merged_coords.insert(merged_coords.end(), coords.begin(), coords.end());
  }

  // Merge connectivity with offset adjustment
  std::vector<offset_t> merged_offsets;
  std::vector<index_t> merged_connectivity;
  std::vector<CellShape> merged_shapes;

  merged_offsets.reserve(total_cells + 1);
  merged_connectivity.reserve(total_connectivity);
  merged_shapes.reserve(total_cells);

  merged_offsets.push_back(0);
  index_t node_offset = 0;

  for (const auto& m : meshes) {
    const auto& offsets = m.cell2node_offsets();
    const auto& connectivity = m.cell2node();
    const auto& shapes = m.cell_shapes();

    for (size_t c = 0; c < m.n_cells(); ++c) {
      // Copy connectivity with offset
      for (offset_t i = offsets[c]; i < offsets[c+1]; ++i) {
        merged_connectivity.push_back(connectivity[i] + node_offset);
      }

      merged_offsets.push_back(merged_offsets.back() + (offsets[c+1] - offsets[c]));
      merged_shapes.push_back(shapes[c]);
    }

    node_offset += m.n_nodes();
  }

  MeshBase merged(dim);
  merged.build_from_arrays(dim, merged_coords, merged_offsets, merged_connectivity, merged_shapes);
  merged.finalize();

  return merged;
}

MeshBase MeshBuilders::concatenate_meshes(const MeshBase& mesh1,
                                         const MeshBase& mesh2) {
  std::vector<MeshBase> meshes = {mesh1, mesh2};
  return merge_meshes(meshes);
}

// ---- Mesh from primitives ----

MeshBase MeshBuilders::build_circle(real_t radius, int n_segments) {
  if (n_segments < 3) {
    throw std::invalid_argument("Circle must have at least 3 segments");
  }

  MeshBase mesh(2);

  // Generate nodes on circle
  std::vector<real_t> coords;
  coords.reserve((n_segments + 1) * 2);

  // Center node
  coords.push_back(0.0);
  coords.push_back(0.0);

  // Boundary nodes
  real_t dtheta = 2 * M_PI / n_segments;
  for (int i = 0; i < n_segments; ++i) {
    real_t theta = i * dtheta;
    coords.push_back(radius * std::cos(theta));
    coords.push_back(radius * std::sin(theta));
  }

  // Generate triangle cells
  std::vector<offset_t> offsets;
  std::vector<index_t> connectivity;
  std::vector<CellShape> shapes;

  offsets.reserve(n_segments + 1);
  connectivity.reserve(n_segments * 3);
  shapes.reserve(n_segments);

  offsets.push_back(0);
  CellShape shape;
  shape.family = CellFamily::Triangle;
  shape.order = 1;

  for (int i = 0; i < n_segments; ++i) {
    connectivity.push_back(0);  // Center
    connectivity.push_back(i + 1);  // Current boundary node
    connectivity.push_back((i + 1) % n_segments + 1);  // Next boundary node

    offsets.push_back(offsets.back() + 3);
    shapes.push_back(shape);
  }

  mesh.build_from_arrays(2, coords, offsets, connectivity, shapes);
  mesh.finalize();

  return mesh;
}

MeshBase MeshBuilders::build_sphere(real_t radius, int refinement_level) {
  // Build icosphere by subdivision
  // Start with icosahedron and refine
  throw std::runtime_error("Sphere builder not yet implemented");
}

MeshBase MeshBuilders::build_cylinder(real_t radius, real_t height,
                                     int n_radial, int n_axial) {
  // Build 2D circle and extrude
  auto circle = build_circle(radius, n_radial);
  return extrude_2d_to_3d(circle, height, n_axial);
}

} // namespace svmp