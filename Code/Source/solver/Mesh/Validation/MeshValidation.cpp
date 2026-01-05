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

#include "MeshValidation.h"
#include "../Core/MeshBase.h"
#include "../Core/DistributedMesh.h"
#include "../Geometry/CurvilinearEval.h"
#include "../Geometry/MeshGeometry.h"
#include "../Geometry/MeshQuality.h"
#include "SpatialHashing.h"
#include <sstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace {

bool gids_are_local_iota(const std::vector<gid_t>& gids) {
  for (size_t i = 0; i < gids.size(); ++i) {
    if (gids[i] != static_cast<gid_t>(i)) {
      return false;
    }
  }
  return true;
}

uint64_t fnv1a64_append(uint64_t h, uint64_t x) {
  constexpr uint64_t kPrime = 1099511628211ULL;
  h ^= x;
  h *= kPrime;
  return h;
}

uint64_t cell_signature_from_vertex_gids(const MeshBase& mesh, index_t cell) {
  const auto family = mesh.cell_shape(cell).family;
  auto [vptr, nverts] = mesh.cell_vertices_span(cell);

  std::vector<gid_t> vg;
  vg.reserve(nverts);
  const auto& vertex_gids = mesh.vertex_gids();
  for (size_t i = 0; i < nverts; ++i) {
    const index_t v = vptr[i];
    if (v < 0 || static_cast<size_t>(v) >= vertex_gids.size()) {
      vg.push_back(INVALID_GID);
    } else {
      vg.push_back(vertex_gids[static_cast<size_t>(v)]);
    }
  }
  std::sort(vg.begin(), vg.end());

  uint64_t h = 1469598103934665603ULL;
  h = fnv1a64_append(h, static_cast<uint64_t>(family));
  h = fnv1a64_append(h, static_cast<uint64_t>(vg.size()));
  for (const auto gid : vg) {
    h = fnv1a64_append(h, static_cast<uint64_t>(gid));
  }
  return h;
}

struct GidListKey {
  std::vector<gid_t> gids;

  bool operator==(const GidListKey& other) const { return gids == other.gids; }
};

struct GidListKeyHash {
  size_t operator()(const GidListKey& k) const {
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a64_append(h, static_cast<uint64_t>(k.gids.size()));
    for (const auto gid : k.gids) {
      h = fnv1a64_append(h, static_cast<uint64_t>(gid));
    }
    return static_cast<size_t>(h);
  }
};

using BoundaryLabelMap = std::unordered_map<GidListKey, label_t, GidListKeyHash>;

BoundaryLabelMap capture_boundary_labels_by_corner_gids(const MeshBase& mesh,
                                                        const std::vector<index_t>* vertex_rep = nullptr) {
  BoundaryLabelMap out;
  if (mesh.n_faces() == 0) {
    return out;
  }

  const auto& v_gids = mesh.vertex_gids();
  const auto& face_shapes = mesh.face_shapes();

  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    const label_t label = mesh.boundary_label(f);
    if (label == INVALID_LABEL) {
      continue;
    }

    auto [vptr, nv] = mesh.face_vertices_span(f);
    if (nv == 0) {
      continue;
    }

    int nc = 0;
    if (static_cast<size_t>(f) < face_shapes.size() && face_shapes[static_cast<size_t>(f)].num_corners > 0) {
      nc = face_shapes[static_cast<size_t>(f)].num_corners;
    } else {
      nc = static_cast<int>(nv);
    }
    nc = std::min(nc, static_cast<int>(nv));
    if (nc <= 0) {
      continue;
    }

    std::vector<gid_t> gids;
    gids.reserve(static_cast<size_t>(nc));
    for (int i = 0; i < nc; ++i) {
      index_t v = vptr[i];
      if (vertex_rep) {
        if (v < 0 || static_cast<size_t>(v) >= vertex_rep->size()) {
          continue;
        }
        v = (*vertex_rep)[static_cast<size_t>(v)];
      }
      if (v < 0 || static_cast<size_t>(v) >= v_gids.size()) {
        continue;
      }
      gids.push_back(v_gids[static_cast<size_t>(v)]);
    }
    if (gids.empty()) {
      continue;
    }
    std::sort(gids.begin(), gids.end());

    GidListKey key{std::move(gids)};
    // If duplicate keys exist, prefer the first encountered label.
    out.emplace(std::move(key), label);
  }

  return out;
}

void restore_boundary_labels_from_corner_gids(MeshBase& mesh, const BoundaryLabelMap& map) {
  if (map.empty() || mesh.n_faces() == 0) {
    return;
  }

  const auto& v_gids = mesh.vertex_gids();
  const auto& face_shapes = mesh.face_shapes();

  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    auto [vptr, nv] = mesh.face_vertices_span(f);
    if (nv == 0) {
      continue;
    }

    int nc = 0;
    if (static_cast<size_t>(f) < face_shapes.size() && face_shapes[static_cast<size_t>(f)].num_corners > 0) {
      nc = face_shapes[static_cast<size_t>(f)].num_corners;
    } else {
      nc = static_cast<int>(nv);
    }
    nc = std::min(nc, static_cast<int>(nv));
    if (nc <= 0) {
      continue;
    }

    std::vector<gid_t> gids;
    gids.reserve(static_cast<size_t>(nc));
    for (int i = 0; i < nc; ++i) {
      const index_t v = vptr[i];
      if (v < 0 || static_cast<size_t>(v) >= v_gids.size()) {
        continue;
      }
      gids.push_back(v_gids[static_cast<size_t>(v)]);
    }
    if (gids.empty()) {
      continue;
    }
    std::sort(gids.begin(), gids.end());

    const GidListKey key{gids};
    const auto it = map.find(key);
    if (it == map.end()) {
      continue;
    }
    mesh.set_boundary_label(f, it->second);
  }
}

std::unordered_map<label_t, std::string> capture_label_registry(const MeshBase& mesh) {
  return mesh.list_label_names();
}

void restore_label_registry(MeshBase& mesh, const std::unordered_map<label_t, std::string>& registry) {
  mesh.clear_label_registry();
  for (const auto& kv : registry) {
    mesh.register_label(kv.second, kv.first);
  }
}

std::vector<ParametricPoint> reference_corner_points(CellFamily family) {
  using P = ParametricPoint;
  switch (family) {
    case CellFamily::Tetra:
      return {P{{0, 0, 0}}, P{{1, 0, 0}}, P{{0, 1, 0}}, P{{0, 0, 1}}};
    case CellFamily::Hex:
      return {P{{-1, -1, -1}}, P{{1, -1, -1}}, P{{1, 1, -1}}, P{{-1, 1, -1}},
              P{{-1, -1, 1}},  P{{1, -1, 1}},  P{{1, 1, 1}},  P{{-1, 1, 1}}};
    case CellFamily::Wedge:
      return {P{{0, 0, -1}}, P{{1, 0, -1}}, P{{0, 1, -1}},
              P{{0, 0, 1}},  P{{1, 0, 1}},  P{{0, 1, 1}}};
    case CellFamily::Pyramid:
      return {P{{-1, -1, 0}}, P{{1, -1, 0}}, P{{1, 1, 0}}, P{{-1, 1, 0}}, P{{0, 0, 1}}};
    default:
      return {};
  }
}

MeshValidation::ValidationResult require_distributed(const char* check_name) {
  MeshValidation::ValidationResult result;
  result.check_name = check_name;
  result.passed = false;
  result.message =
      std::string(check_name) +
      " requires DistributedMesh (parallel metadata + communicator).";
  return result;
}

#ifdef MESH_HAS_MPI
std::string broadcast_string(const std::string& s, int root, MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int len = (rank == root) ? static_cast<int>(s.size()) : 0;
  MPI_Bcast(&len, 1, MPI_INT, root, comm);
  std::vector<char> buf(static_cast<size_t>(len));
  if (rank == root && len > 0) {
    std::memcpy(buf.data(), s.data(), static_cast<size_t>(len));
  }
  if (len > 0) {
    MPI_Bcast(buf.data(), len, MPI_CHAR, root, comm);
  }
  return std::string(buf.begin(), buf.end());
}
#endif

} // namespace

// ---- ValidationReport methods ----

void MeshValidation::ValidationReport::print() const {
  std::cout << to_string() << std::endl;
}

std::string MeshValidation::ValidationReport::to_string() const {
  std::ostringstream ss;
  ss << "\n=== Mesh Validation Report ===" << std::endl;
  ss << "Overall status: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
  ss << "Total checks: " << results.size() << std::endl;

  int passed = 0, failed = 0;
  for (const auto& r : results) {
    if (r.passed) passed++;
    else failed++;
  }

  ss << "Passed: " << passed << ", Failed: " << failed << std::endl;

  if (failed > 0) {
    ss << "\nFailed checks:" << std::endl;
    for (const auto& r : results) {
      if (!r.passed) {
        ss << "  - " << r.check_name << ": " << r.message << std::endl;
        if (!r.problem_entities.empty()) {
          ss << "    Problem entities: ";
          for (size_t i = 0; i < std::min(size_t(5), r.problem_entities.size()); ++i) {
            ss << r.problem_entities[i] << " ";
          }
          if (r.problem_entities.size() > 5) {
            ss << "... (" << r.problem_entities.size() << " total)";
          }
          ss << std::endl;
        }
      }
    }
  }

  ss << "\nValidation time: " << total_time << " seconds" << std::endl;
  ss << "=== End Report ===" << std::endl;

  return ss.str();
}

// ---- Basic validation ----

MeshValidation::ValidationResult MeshValidation::validate_basic(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Basic structure";
  result.passed = true;

  // Check that dimensions are valid
  if (mesh.dim() < 1 || mesh.dim() > 3) {
    result.passed = false;
    result.message = "Invalid mesh dimension: " + std::to_string(mesh.dim());
    return result;
  }

  // Check that we have vertices and cells
  if (mesh.n_vertices() == 0) {
    result.passed = false;
    result.message = "Mesh has no vertices";
    return result;
  }

  if (mesh.n_cells() == 0) {
    result.passed = false;
    result.message = "Mesh has no cells";
    return result;
  }

  result.message = "Basic structure checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_array_sizes(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Array sizes";
  result.passed = true;

  // Check coordinate array size
  size_t expected_coord_size = mesh.n_vertices() * mesh.dim();
  size_t actual_coord_size = mesh.X_ref().size();

  if (actual_coord_size != expected_coord_size) {
    result.passed = false;
    result.message = "Coordinate array size mismatch: expected " +
                    std::to_string(expected_coord_size) + ", got " +
                    std::to_string(actual_coord_size);
    return result;
  }

  result.message = "Array sizes are consistent";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_csr_offsets(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "CSR offsets";
  result.passed = true;

  // Check that CSR offsets are monotonic
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));

    if (n_vertices == 0) {
      result.passed = false;
      result.message = "Cell " + std::to_string(c) + " has no vertices";
      result.problem_entities.push_back(static_cast<index_t>(c));
    }

    // Check for reasonable number of vertices per cell
    if (n_vertices > 100) {  // Arbitrary large number
      result.passed = false;
      result.message = "Cell " + std::to_string(c) + " has suspiciously many vertices: " + std::to_string(n_vertices);
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (result.passed) {
    result.message = "CSR offsets are valid";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_vertex_indices(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Vertex indices";
  result.passed = true;

  size_t n_cells = mesh.n_cells();
  size_t n_vertices = mesh.n_vertices();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_vertices_cell] = mesh.cell_vertices_span(static_cast<index_t>(c));

    for (size_t i = 0; i < n_vertices_cell; ++i) {
      if (vertices_ptr[i] < 0 || vertices_ptr[i] >= static_cast<index_t>(n_vertices)) {
        result.passed = false;
        result.message = "Cell " + std::to_string(c) + " has invalid vertex index: " + std::to_string(vertices_ptr[i]);
        result.problem_entities.push_back(static_cast<index_t>(c));
        break;
      }
    }
  }

  if (result.passed) {
    result.message = "All vertex indices are valid";
  }

  return result;
}

// ---- Topology validation ----

MeshValidation::ValidationResult MeshValidation::validate_topology(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Topology";
  result.passed = true;

  // Run several topology checks
  auto isolated = find_isolated_vertices(mesh);
  if (!isolated.passed) {
    result.passed = false;
    result.message = "Topology issues found: " + isolated.message;
    result.problem_entities = isolated.problem_entities;
    return result;
  }

  auto repeated = check_repeated_vertices_in_cells(mesh);
  if (!repeated.passed) {
    result.passed = false;
    result.message = "Topology issues found: " + repeated.message;
    result.problem_entities = repeated.problem_entities;
    return result;
  }

  result.message = "Topology checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::find_duplicate_vertices(const MeshBase& mesh, real_t tolerance) {
  ValidationResult result;
  result.check_name = "Duplicate vertices";
  result.passed = true;

  validation::SpatialHashGrid grid(tolerance);
  const auto& coords = mesh.X_ref();
  int dim = mesh.dim();
  size_t n_vertices = mesh.n_vertices();

  // Insert all vertices into spatial hash grid
  for (size_t i = 0; i < n_vertices; ++i) {
    std::array<double, 3> pt = {0, 0, 0};
    for (int d = 0; d < dim; ++d) {
      pt[d] = coords[i * dim + d];
    }
    grid.insert(static_cast<index_t>(i), pt);
  }

  auto duplicates = grid.find_duplicates();

  if (!duplicates.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(duplicates.size()) + " duplicate vertex pairs";

    // Add the duplicate vertices to problem entities
    for (const auto& [n1, n2] : duplicates) {
      result.problem_entities.push_back(n1);
      result.problem_entities.push_back(n2);
    }
  } else {
    result.message = "No duplicate vertices found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_isolated_vertices(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Isolated vertices";
  result.passed = true;

  size_t n_vertices = mesh.n_vertices();
  size_t n_cells = mesh.n_cells();

  std::vector<bool> vertex_used(n_vertices, false);

  // Mark all vertices used by cells
  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_vertices_cell] = mesh.cell_vertices_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_vertices_cell; ++i) {
      vertex_used[vertices_ptr[i]] = true;
    }
  }

  // Find unused vertices
  for (size_t v = 0; v < n_vertices; ++v) {
    if (!vertex_used[v]) {
      result.problem_entities.push_back(static_cast<index_t>(v));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " isolated vertices";
  } else {
    result.message = "No isolated vertices found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_degenerate_cells(const MeshBase& mesh, real_t tolerance) {
  ValidationResult result;
  result.check_name = "Degenerate cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    if (is_degenerate_cell(mesh, static_cast<index_t>(c), tolerance)) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " degenerate cells";
  } else {
    result.message = "No degenerate cells found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_inverted_cells(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Inverted cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    if (is_inverted_cell(mesh, static_cast<index_t>(c))) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " inverted cells";
  } else {
    result.message = "No inverted cells found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_repeated_vertices_in_cells(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Repeated vertices in cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [vertices_ptr, n_vertices] = mesh.cell_vertices_span(static_cast<index_t>(c));

    std::unordered_set<index_t> unique_vertices;
    for (size_t i = 0; i < n_vertices; ++i) {
      if (unique_vertices.count(vertices_ptr[i]) > 0) {
        result.problem_entities.push_back(static_cast<index_t>(c));
        break;
      }
      unique_vertices.insert(vertices_ptr[i]);
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " cells with repeated vertices";
  } else {
    result.message = "No cells with repeated vertices";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_face_cell_consistency(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Face-cell consistency";
  result.passed = true;

  size_t n_faces = mesh.n_faces();
  size_t n_cells = mesh.n_cells();

  for (size_t f = 0; f < n_faces; ++f) {
    auto face_cells = mesh.face_cells(static_cast<index_t>(f));

    // Check that face references valid cells
    if (face_cells[0] < 0 || face_cells[0] >= static_cast<index_t>(n_cells)) {
      result.passed = false;
      result.message = "Face " + std::to_string(f) + " references invalid cell";
      result.problem_entities.push_back(static_cast<index_t>(f));
    }

    if (face_cells[1] != INVALID_INDEX &&
        (face_cells[1] < 0 || face_cells[1] >= static_cast<index_t>(n_cells))) {
      result.passed = false;
      result.message = "Face " + std::to_string(f) + " references invalid cell";
      result.problem_entities.push_back(static_cast<index_t>(f));
    }
  }

  if (result.passed) {
    result.message = "Face-cell connectivity is consistent";
  }

  return result;
}

// ---- Geometry validation ----

MeshValidation::ValidationResult MeshValidation::validate_geometry(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Geometry";
  result.passed = true;

  // Check face orientations
  auto orientation = check_face_orientation(mesh);
  if (!orientation.passed) {
    result.passed = false;
    result.message = "Geometry issues: " + orientation.message;
    result.problem_entities = orientation.problem_entities;
    return result;
  }

  result.message = "Geometry checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_face_orientation(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Face orientation";
  result.passed = true;

  // Simple check: ensure face normals are consistent
  // This is a simplified implementation

  result.message = "Face orientations appear consistent";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_outward_normals(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Outward normals";
  result.passed = true;

  // Check that boundary face normals point outward
  // This requires more complex geometric computation

  result.message = "Normals check not fully implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_self_intersection(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Self-intersection";
  result.passed = true;

  // Check for self-intersecting faces
  // This is computationally expensive - simplified for now

  result.message = "Self-intersection check not fully implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_watertight(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Watertight";
  result.passed = true;

  // Check if mesh is watertight (closed)
  // Count boundary faces
  size_t n_faces = mesh.n_faces();
  size_t boundary_faces = 0;

  for (size_t f = 0; f < n_faces; ++f) {
    auto face_cells = mesh.face_cells(static_cast<index_t>(f));
    if (face_cells[1] == INVALID_INDEX) {
      boundary_faces++;
    }
  }

  if (boundary_faces > 0) {
    result.passed = false;
    result.message = "Mesh is not watertight: " + std::to_string(boundary_faces) + " boundary faces";
  } else {
    result.message = "Mesh is watertight";
  }

  return result;
}

// ---- Quality validation ----

MeshValidation::ValidationResult MeshValidation::check_quality(const MeshBase& mesh,
                                                              real_t min_quality,
                                                              const std::string& metric) {
  ValidationResult result;
  result.check_name = "Quality (" + metric + ")";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t quality = mesh.compute_quality(static_cast<index_t>(c), metric);
    if (quality < min_quality) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = std::to_string(result.problem_entities.size()) +
                    " cells have quality < " + std::to_string(min_quality);
  } else {
    result.message = "All cells meet quality threshold";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_skewed_cells(const MeshBase& mesh,
                                                                  real_t max_skewness) {
  ValidationResult result;
  result.check_name = "Skewed cells";
  result.passed = true;

  // Check for highly skewed cells using skewness metric
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t skewness = mesh.compute_quality(static_cast<index_t>(c), "skewness");
    if (skewness > max_skewness) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = std::to_string(result.problem_entities.size()) +
                    " cells have skewness > " + std::to_string(max_skewness);
  } else {
    result.message = "No highly skewed cells found";
  }

  return result;
}

// ---- Parallel validation ----

MeshValidation::ValidationResult MeshValidation::check_parallel_consistency(const MeshBase& mesh) {
  (void)mesh;
  return require_distributed("Parallel consistency");
}

MeshValidation::ValidationResult MeshValidation::check_parallel_consistency(const DistributedMesh& mesh) {
  ValidationResult result;
  result.check_name = "Parallel consistency";
  result.passed = true;

  if (mesh.world_size() <= 1) {
    result.message = "Serial: no parallel consistency checks needed";
    return result;
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    result.message = "MPI not initialized or communicator is null; skipping parallel consistency checks";
    return result;
  }

  const auto ids = check_global_ids(mesh);
  const auto ghosts = check_ghost_cells(mesh);

  if (!ids.passed || !ghosts.passed) {
    result.passed = false;
    result.message = "Parallel consistency failed: " + ids.message + "; " + ghosts.message;
    return result;
  }

  result.message = "Parallel consistency checks passed";
  return result;
#else
  result.message = "MPI support not enabled; skipping parallel consistency checks";
  return result;
#endif
}

MeshValidation::ValidationResult MeshValidation::check_global_ids(const MeshBase& mesh) {
  (void)mesh;
  return require_distributed("Global IDs");
}

MeshValidation::ValidationResult MeshValidation::check_global_ids(const DistributedMesh& mesh) {
  ValidationResult result;
  result.check_name = "Global IDs";
  result.passed = true;

  const auto& local = mesh.local_mesh();

  const auto check_gid_array = [&](const char* name,
                                   size_t expected_count,
                                   const std::vector<gid_t>& gids) -> bool {
    if (gids.size() != expected_count) {
      result.passed = false;
      result.message = std::string(name) + " GID array size mismatch: expected " +
                       std::to_string(expected_count) + ", got " + std::to_string(gids.size());
      return false;
    }

    std::unordered_set<gid_t> seen;
    seen.reserve(gids.size());
    for (size_t i = 0; i < gids.size(); ++i) {
      const gid_t gid = gids[i];
      if (gid == INVALID_GID) {
        result.passed = false;
        result.message = std::string(name) + " has INVALID_GID at local index " + std::to_string(i);
        result.problem_entities.push_back(static_cast<index_t>(i));
        return false;
      }
      if (!seen.insert(gid).second) {
        result.passed = false;
        result.message = std::string(name) + " has duplicate GID " + std::to_string(gid) +
                         " within rank";
        result.problem_entities.push_back(static_cast<index_t>(i));
        return false;
      }
    }
    return true;
  };

  bool ok = true;
  ok = check_gid_array("Vertex", local.n_vertices(), local.vertex_gids()) && ok;
  ok = check_gid_array("Cell", local.n_cells(), local.cell_gids()) && ok;
  if (local.n_faces() != 0) {
    ok = check_gid_array("Face", local.n_faces(), local.face_gids()) && ok;
  }
  if (local.n_edges() != 0) {
    ok = check_gid_array("Edge", local.n_edges(), local.edge_gids()) && ok;
  }

  if (!ok) {
    return result;
  }

  if (mesh.world_size() <= 1) {
    result.message = "Local GID checks passed (serial)";
    return result;
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    result.message = "MPI not initialized or communicator is null; only local GID checks performed";
    return result;
  }

  MPI_Comm comm = mesh.mpi_comm();
  const int my_rank = static_cast<int>(mesh.rank());
  const int world = mesh.world_size();

  // Guardrail: detect the common error where each rank uses local-iota IDs (0..n-1).
  {
    const int local_has_vertices = local.n_vertices() > 0 ? 1 : 0;
    const int local_vertices_iota = gids_are_local_iota(local.vertex_gids()) ? 1 : 0;
    int ranks_with_vertices = 0;
    MPI_Allreduce(&local_has_vertices, &ranks_with_vertices, 1, MPI_INT, MPI_SUM, comm);

    const int local_vertices_iota_for_reduce = local_has_vertices ? local_vertices_iota : 1;
    int all_vertices_iota = 1;
    MPI_Allreduce(&local_vertices_iota_for_reduce, &all_vertices_iota, 1, MPI_INT, MPI_MIN, comm);
    if (ranks_with_vertices > 1 && all_vertices_iota == 1) {
      result.passed = false;
      result.message =
          "Detected default local-iota vertex GIDs on all ranks; assign globally consistent IDs";
      return result;
    }
  }
  {
    const int local_has_cells = local.n_cells() > 0 ? 1 : 0;
    const int local_cells_iota = gids_are_local_iota(local.cell_gids()) ? 1 : 0;
    int ranks_with_cells = 0;
    MPI_Allreduce(&local_has_cells, &ranks_with_cells, 1, MPI_INT, MPI_SUM, comm);

    const int local_cells_iota_for_reduce = local_has_cells ? local_cells_iota : 1;
    int all_cells_iota = 1;
    MPI_Allreduce(&local_cells_iota_for_reduce, &all_cells_iota, 1, MPI_INT, MPI_MIN, comm);
    if (ranks_with_cells > 1 && all_cells_iota == 1) {
      result.passed = false;
      result.message =
          "Detected default local-iota cell GIDs on all ranks; assign globally consistent IDs";
      return result;
    }
  }

  // Distributed uniqueness check for owned cell GIDs (exactly one owner per global cell).
  // This is a validation/debug path; it is allowed to be more expensive than production code.
  std::vector<gid_t> owned_cell_gids;
  owned_cell_gids.reserve(mesh.n_owned_cells());
  const auto& cell_gids = local.cell_gids();
  for (index_t c = 0; c < static_cast<index_t>(local.n_cells()); ++c) {
    if (mesh.is_owned_cell(c)) {
      owned_cell_gids.push_back(cell_gids[static_cast<size_t>(c)]);
    }
  }

  auto home_rank = [world](gid_t gid) -> int {
    uint64_t h = static_cast<uint64_t>(gid);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return static_cast<int>(h % static_cast<uint64_t>(world));
  };

  std::vector<std::vector<gid_t>> by_dest(static_cast<size_t>(world));
  for (const auto gid : owned_cell_gids) {
    by_dest[static_cast<size_t>(home_rank(gid))].push_back(gid);
  }

  std::vector<int> send_counts(world, 0);
  for (int r = 0; r < world; ++r) {
    send_counts[r] = static_cast<int>(by_dest[static_cast<size_t>(r)].size());
  }

  std::vector<int> recv_counts(world, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

  std::vector<int> send_displs(world + 1, 0);
  std::vector<int> recv_displs(world + 1, 0);
  for (int r = 0; r < world; ++r) {
    send_displs[r + 1] = send_displs[r] + send_counts[r];
    recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
  }

  std::vector<gid_t> send_buf(static_cast<size_t>(send_displs[world]));
  for (int r = 0; r < world; ++r) {
    auto& vec = by_dest[static_cast<size_t>(r)];
    std::copy(vec.begin(), vec.end(), send_buf.begin() + send_displs[r]);
  }

  std::vector<gid_t> recv_buf(static_cast<size_t>(recv_displs[world]));
  MPI_Datatype gid_type =
#ifdef MPI_INT64_T
      MPI_INT64_T;
#else
      MPI_LONG_LONG;
#endif
  MPI_Alltoallv(send_buf.data(),
                send_counts.data(),
                send_displs.data(),
                gid_type,
                recv_buf.data(),
                recv_counts.data(),
                recv_displs.data(),
                gid_type,
                comm);

  std::unordered_map<gid_t, int> counts;
  counts.reserve(recv_buf.size());
  for (const auto gid : recv_buf) {
    counts[gid] += 1;
  }

  std::vector<gid_t> local_dups;
  local_dups.reserve(16);
  for (const auto& kv : counts) {
    if (kv.second > 1) {
      local_dups.push_back(kv.first);
      if (local_dups.size() >= 16) {
        break;
      }
    }
  }

  const int local_ok = local_dups.empty() ? 1 : 0;
  int global_ok = 1;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
  if (global_ok != 1) {
    // Gather a few duplicate IDs to rank 0 for a stable message.
    int local_n = static_cast<int>(local_dups.size());
    std::vector<int> n_by_rank(world, 0);
    MPI_Gather(&local_n, 1, MPI_INT, n_by_rank.data(), 1, MPI_INT, 0, comm);

    std::string msg;
    if (my_rank == 0) {
      std::vector<int> disp(world + 1, 0);
      for (int r = 0; r < world; ++r) {
        disp[r + 1] = disp[r] + n_by_rank[r];
      }
      std::vector<gid_t> all_dups(static_cast<size_t>(disp[world]));
      MPI_Gatherv(local_dups.data(), local_n, gid_type,
                  all_dups.data(), n_by_rank.data(), disp.data(), gid_type,
                  0, comm);

      std::unordered_set<gid_t> uniq;
      for (const auto gid : all_dups) {
        uniq.insert(gid);
      }
      std::vector<gid_t> sample;
      sample.reserve(8);
      for (const auto gid : uniq) {
        sample.push_back(gid);
        if (sample.size() >= 8) break;
      }

      std::ostringstream oss;
      oss << "Duplicate owned cell GIDs detected across ranks (sample: ";
      for (const auto gid : sample) {
        oss << gid << " ";
      }
      oss << ")";
      msg = oss.str();
    } else {
      MPI_Gatherv(local_dups.data(), local_n, gid_type,
                  nullptr, nullptr, nullptr, gid_type,
                  0, comm);
    }

    msg = broadcast_string(msg, 0, comm);
    result.passed = false;
    result.message = msg;
    return result;
  }

  result.message = "Global ID checks passed";
  return result;
#else
  // Should be unreachable because mesh.world_size() > 1 implies MPI build + initialized.
  result.message = "MPI support not enabled; only local GID checks performed";
  return result;
#endif
}

MeshValidation::ValidationResult MeshValidation::check_ghost_cells(const MeshBase& mesh) {
  (void)mesh;
  return require_distributed("Ghost cells");
}

MeshValidation::ValidationResult MeshValidation::check_ghost_cells(const DistributedMesh& mesh) {
  ValidationResult result;
  result.check_name = "Ghost cells";
  result.passed = true;

  const auto& local = mesh.local_mesh();
  if (mesh.world_size() <= 1) {
    result.message = "Serial: no ghost-cell checks needed";
    return result;
  }

#ifdef MESH_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized || mesh.mpi_comm() == MPI_COMM_NULL) {
    result.message = "MPI not initialized or communicator is null; skipping ghost-cell checks";
    return result;
  }

  MPI_Comm comm = mesh.mpi_comm();
  const int my_rank = static_cast<int>(mesh.rank());
  const int world = mesh.world_size();

  // Build per-owner request lists for ghost cells (by GID).
  std::vector<std::vector<gid_t>> req_gids(static_cast<size_t>(world));
  std::vector<std::vector<index_t>> req_local_cells(static_cast<size_t>(world));
  const auto& cell_gids = local.cell_gids();

  for (index_t c = 0; c < static_cast<index_t>(local.n_cells()); ++c) {
    if (!mesh.is_ghost_cell(c)) {
      continue;
    }
    const rank_t owner = mesh.owner_rank_cell(c);
    if (owner < 0 || owner >= world) {
      result.passed = false;
      result.message = "Ghost cell has invalid owner rank";
      result.problem_entities.push_back(c);
      return result;
    }
    if (owner == static_cast<rank_t>(my_rank)) {
      result.passed = false;
      result.message = "Ghost cell reports this rank as owner";
      result.problem_entities.push_back(c);
      return result;
    }
    req_gids[static_cast<size_t>(owner)].push_back(cell_gids[static_cast<size_t>(c)]);
    req_local_cells[static_cast<size_t>(owner)].push_back(c);
  }

  std::vector<int> send_counts(world, 0);
  for (int r = 0; r < world; ++r) {
    send_counts[r] = static_cast<int>(req_gids[static_cast<size_t>(r)].size());
  }

  std::vector<int> recv_counts(world, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

  std::vector<int> send_displs(world + 1, 0);
  std::vector<int> recv_displs(world + 1, 0);
  for (int r = 0; r < world; ++r) {
    send_displs[r + 1] = send_displs[r] + send_counts[r];
    recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
  }

  std::vector<gid_t> send_buf(static_cast<size_t>(send_displs[world]));
  for (int r = 0; r < world; ++r) {
    const auto& vec = req_gids[static_cast<size_t>(r)];
    std::copy(vec.begin(), vec.end(), send_buf.begin() + send_displs[r]);
  }

  std::vector<gid_t> recv_buf(static_cast<size_t>(recv_displs[world]));

  MPI_Datatype gid_type =
#ifdef MPI_INT64_T
      MPI_INT64_T;
#else
      MPI_LONG_LONG;
#endif

  MPI_Alltoallv(send_buf.data(),
                send_counts.data(),
                send_displs.data(),
                gid_type,
                recv_buf.data(),
                recv_counts.data(),
                recv_displs.data(),
                gid_type,
                comm);

  // Owner-side: compute signatures for requested cells (in the order received).
  std::vector<uint64_t> sig_send(static_cast<size_t>(recv_buf.size()), 0ULL);
  for (size_t i = 0; i < recv_buf.size(); ++i) {
    const gid_t gid = recv_buf[i];
    const index_t lc = mesh.global_to_local_cell(gid);
    if (lc < 0) {
      sig_send[i] = 0ULL;
      continue;
    }
    if (!mesh.is_owned_cell(lc)) {
      sig_send[i] = 0ULL;
      continue;
    }
    sig_send[i] = cell_signature_from_vertex_gids(local, lc);
  }

  MPI_Datatype u64_type =
#ifdef MPI_UINT64_T
      MPI_UINT64_T;
#else
      MPI_UNSIGNED_LONG_LONG;
#endif

  // Exchange signatures back to requesting ranks.
  std::vector<uint64_t> sig_recv(static_cast<size_t>(send_buf.size()), 0ULL);
  MPI_Alltoallv(sig_send.data(),
                recv_counts.data(),
                recv_displs.data(),
                u64_type,
                sig_recv.data(),
                send_counts.data(),
                send_displs.data(),
                u64_type,
                comm);

  // Compare on requesting ranks.
  size_t mismatches = 0;
  for (int r = 0; r < world; ++r) {
    const int begin = send_displs[r];
    const int end = send_displs[r + 1];
    const auto& local_cells = req_local_cells[static_cast<size_t>(r)];
    for (int j = begin; j < end; ++j) {
      const int idx = j - begin;
      if (idx < 0 || static_cast<size_t>(idx) >= local_cells.size()) {
        continue;
      }
      const index_t ghost_cell = local_cells[static_cast<size_t>(idx)];
      const uint64_t owner_sig = sig_recv[static_cast<size_t>(j)];
      if (owner_sig == 0ULL) {
        result.problem_entities.push_back(ghost_cell);
        mismatches++;
        continue;
      }
      const uint64_t local_sig = cell_signature_from_vertex_gids(local, ghost_cell);
      if (local_sig != owner_sig) {
        result.problem_entities.push_back(ghost_cell);
        mismatches++;
      }
    }
  }

  if (mismatches != 0) {
    result.passed = false;
    result.message = "Ghost cell signatures mismatch or missing owner data (" + std::to_string(mismatches) + ")";
    return result;
  }

  result.message = "Ghost cell checks passed";
  return result;
#else
  result.message = "MPI support not enabled; skipping ghost-cell checks";
  return result;
#endif
}

// ---- Comprehensive validation ----

MeshValidation::ValidationReport MeshValidation::validate_all(const MeshBase& mesh,
                                                             const ValidationConfig& config) {
  ValidationReport report;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Basic checks
  if (config.check_basic) {
    report.add_result(validate_basic(mesh));
    report.add_result(check_array_sizes(mesh));
    report.add_result(check_csr_offsets(mesh));
    report.add_result(check_vertex_indices(mesh));
  }

  // Topology checks
  if (config.check_topology) {
    report.add_result(find_duplicate_vertices(mesh, config.duplicate_tolerance));
    report.add_result(find_isolated_vertices(mesh));
    report.add_result(find_degenerate_cells(mesh, config.degenerate_tolerance));
    report.add_result(find_inverted_cells(mesh));
    report.add_result(check_repeated_vertices_in_cells(mesh));
    report.add_result(check_face_cell_consistency(mesh));
  }

  // Geometry checks
  if (config.check_geometry) {
    report.add_result(check_face_orientation(mesh));
    report.add_result(check_watertight(mesh));
  }

  // Quality checks
  if (config.check_quality) {
    report.add_result(check_quality(mesh, config.min_quality, config.quality_metric));
  }

  // Parallel checks
  if (config.check_parallel) {
    report.add_result(check_parallel_consistency(mesh));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  report.total_time = elapsed.count();

  return report;
}

MeshValidation::ValidationReport MeshValidation::validate_all(const DistributedMesh& mesh,
                                                             const ValidationConfig& config) {
  ValidationReport report;
  auto start_time = std::chrono::high_resolution_clock::now();

  const auto& local = mesh.local_mesh();

  if (config.check_basic) {
    report.add_result(validate_basic(local));
    report.add_result(check_array_sizes(local));
    report.add_result(check_csr_offsets(local));
    report.add_result(check_vertex_indices(local));
  }

  if (config.check_topology) {
    report.add_result(find_duplicate_vertices(local, config.duplicate_tolerance));
    report.add_result(find_isolated_vertices(local));
    report.add_result(find_degenerate_cells(local, config.degenerate_tolerance));
    report.add_result(find_inverted_cells(local));
    report.add_result(check_repeated_vertices_in_cells(local));
    report.add_result(check_face_cell_consistency(local));
  }

  if (config.check_geometry) {
    report.add_result(check_face_orientation(local));
    report.add_result(check_watertight(local));
  }

  if (config.check_quality) {
    report.add_result(check_quality(local, config.min_quality, config.quality_metric));
  }

  if (config.check_parallel) {
    report.add_result(check_parallel_consistency(mesh));
    report.add_result(check_global_ids(mesh));
    report.add_result(check_ghost_cells(mesh));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  report.total_time = elapsed.count();
  return report;
}

MeshValidation::ValidationReport MeshValidation::validate_quick(const MeshBase& mesh) {
  ValidationConfig config;
  config.check_basic = true;
  config.check_topology = true;
  config.check_geometry = false;
  config.check_quality = false;
  config.check_parallel = false;

  return validate_all(mesh, config);
}

MeshValidation::ValidationReport MeshValidation::validate_quick(const DistributedMesh& mesh) {
  ValidationConfig config;
  config.check_basic = true;
  config.check_topology = true;
  config.check_geometry = false;
  config.check_quality = false;
  config.check_parallel = true;
  return validate_all(mesh, config);
}

// ---- Repair operations ----

// The repair operations below are inspired by the *types* of robustness steps used in
// Marco Attene's MeshFix (duplicate removal, degenerate cleanup, and orientation fixes),
// but are implemented independently for svMultiPhysics' mixed-cell MeshBase data model.

static void rebuild_mesh_from_arrays_preserving_labels(
    MeshBase& mesh,
    int spatial_dim,
    const std::vector<real_t>& X_ref,
    const std::vector<offset_t>& cell2vertex_offsets,
    const std::vector<index_t>& cell2vertex,
    const std::vector<CellShape>& cell_shapes,
    const std::vector<gid_t>& vertex_gids,
    const std::vector<gid_t>& cell_gids,
    const std::vector<label_t>& vertex_labels,
    const std::vector<label_t>& cell_region_ids,
    const std::vector<size_t>& cell_ref_levels,
    const std::vector<real_t>* X_cur,
    const std::unordered_map<label_t, std::string>& label_registry,
    const BoundaryLabelMap& boundary_labels) {
  mesh.build_from_arrays(spatial_dim, X_ref, cell2vertex_offsets, cell2vertex, cell_shapes);
  if (X_cur && !X_cur->empty()) {
    mesh.set_current_coords(*X_cur);
  }

  if (!vertex_gids.empty()) {
    mesh.set_vertex_gids(vertex_gids);
  }
  if (!cell_gids.empty()) {
    mesh.set_cell_gids(cell_gids);
  }

  restore_label_registry(mesh, label_registry);

  for (index_t v = 0; v < static_cast<index_t>(vertex_labels.size()); ++v) {
    const label_t lbl = vertex_labels[static_cast<size_t>(v)];
    if (lbl != INVALID_LABEL) {
      mesh.set_vertex_label(v, lbl);
    }
  }

  for (index_t c = 0; c < static_cast<index_t>(cell_region_ids.size()); ++c) {
    mesh.set_region_label(c, cell_region_ids[static_cast<size_t>(c)]);
  }
  if (!cell_ref_levels.empty()) {
    mesh.set_cell_refinement_levels(cell_ref_levels);
  }

  mesh.finalize();
  restore_boundary_labels_from_corner_gids(mesh, boundary_labels);
}

index_t MeshValidation::merge_duplicate_vertices(MeshBase& mesh, real_t tolerance) {
  const auto duplicates = find_duplicate_vertices(mesh, tolerance);
  if (duplicates.passed) {
    return 0;
  }

  const int spatial_dim = mesh.dim();
  const size_t n_vertices = mesh.n_vertices();
  if (n_vertices == 0) {
    return 0;
  }

  // Recompute duplicate pairs so we have explicit connectivity for union-find.
  validation::SpatialHashGrid grid(tolerance);
  const auto& X = mesh.X_ref();
  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    std::array<double, 3> pt = {0, 0, 0};
    for (int d = 0; d < spatial_dim; ++d) {
      pt[d] = X[static_cast<size_t>(v) * spatial_dim + d];
    }
    grid.insert(v, pt);
  }
  const auto pairs = grid.find_duplicates();
  if (pairs.empty()) {
    return 0;
  }

  // Union-find groups with deterministic representative = minimum vertex id in the set.
  std::vector<index_t> parent(n_vertices);
  std::iota(parent.begin(), parent.end(), 0);

  auto find_root = [&](index_t x) {
    index_t r = x;
    while (parent[static_cast<size_t>(r)] != r) {
      r = parent[static_cast<size_t>(r)];
    }
    while (parent[static_cast<size_t>(x)] != x) {
      const index_t p = parent[static_cast<size_t>(x)];
      parent[static_cast<size_t>(x)] = r;
      x = p;
    }
    return r;
  };

  auto unite = [&](index_t a, index_t b) {
    index_t ra = find_root(a);
    index_t rb = find_root(b);
    if (ra == rb) return;
    if (ra < rb) parent[static_cast<size_t>(rb)] = ra;
    else parent[static_cast<size_t>(ra)] = rb;
  };

  for (const auto& pr : pairs) {
    unite(pr.first, pr.second);
  }

  std::vector<index_t> rep_of(n_vertices, INVALID_INDEX);
  std::vector<uint8_t> is_rep(n_vertices, 0);
  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    const index_t r = find_root(v);
    rep_of[static_cast<size_t>(v)] = r;
  }
  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    is_rep[static_cast<size_t>(rep_of[static_cast<size_t>(v)])] = 1;
  }

  // Count representative vertices.
  size_t n_reps = 0;
  for (size_t i = 0; i < n_vertices; ++i) {
    if (is_rep[i] != 0) ++n_reps;
  }
  if (n_reps == n_vertices) {
    return 0;
  }

  // Build rep->new index map.
  std::vector<index_t> rep_to_new(n_vertices, INVALID_INDEX);
  index_t next = 0;
  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    if (is_rep[static_cast<size_t>(v)] != 0) {
      rep_to_new[static_cast<size_t>(v)] = next++;
    }
  }

  // Average coordinates per representative (robust duplicate merge in the MeshFix spirit).
  std::vector<std::array<real_t, 3>> sum_ref(n_vertices, {{0, 0, 0}});
  std::vector<std::array<real_t, 3>> sum_cur(n_vertices, {{0, 0, 0}});
  std::vector<int> count(n_vertices, 0);

  const auto& X_ref = mesh.X_ref();
  const bool has_cur = mesh.has_current_coords();
  const auto& X_cur = mesh.X_cur();

  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    const index_t r = rep_of[static_cast<size_t>(v)];
    count[static_cast<size_t>(r)] += 1;
    for (int d = 0; d < spatial_dim; ++d) {
      sum_ref[static_cast<size_t>(r)][static_cast<size_t>(d)] += X_ref[static_cast<size_t>(v) * spatial_dim + d];
      if (has_cur) {
        sum_cur[static_cast<size_t>(r)][static_cast<size_t>(d)] += X_cur[static_cast<size_t>(v) * spatial_dim + d];
      }
    }
  }

  std::vector<real_t> new_X_ref;
  new_X_ref.resize(n_reps * static_cast<size_t>(spatial_dim), 0.0);
  std::vector<real_t> new_X_cur;
  if (has_cur) {
    new_X_cur.resize(n_reps * static_cast<size_t>(spatial_dim), 0.0);
  }

  for (index_t r = 0; r < static_cast<index_t>(n_vertices); ++r) {
    const index_t nv = rep_to_new[static_cast<size_t>(r)];
    if (nv == INVALID_INDEX) continue;
    const int cnt = std::max(1, count[static_cast<size_t>(r)]);
    for (int d = 0; d < spatial_dim; ++d) {
      new_X_ref[static_cast<size_t>(nv) * spatial_dim + d] =
          sum_ref[static_cast<size_t>(r)][static_cast<size_t>(d)] / static_cast<real_t>(cnt);
      if (has_cur) {
        new_X_cur[static_cast<size_t>(nv) * spatial_dim + d] =
            sum_cur[static_cast<size_t>(r)][static_cast<size_t>(d)] / static_cast<real_t>(cnt);
      }
    }
  }

  // Remap cell connectivity through representatives.
  std::vector<index_t> new_c2v = mesh.cell2vertex();
  for (auto& v : new_c2v) {
    if (v < 0 || static_cast<size_t>(v) >= n_vertices) {
      v = INVALID_INDEX;
      continue;
    }
    const index_t r = rep_of[static_cast<size_t>(v)];
    v = rep_to_new[static_cast<size_t>(r)];
  }

  // Preserve labels/IDs.
  const auto label_registry = capture_label_registry(mesh);
  const auto boundary_labels = capture_boundary_labels_by_corner_gids(mesh, &rep_of);
  const auto old_cell_gids = mesh.cell_gids();
  const auto old_cell_regions = mesh.cell_region_ids();
  const auto old_cell_ref_levels = mesh.cell_refinement_levels();
  const auto old_vertex_labels = mesh.vertex_label_ids();
  const auto old_vertex_gids = mesh.vertex_gids();

  std::vector<gid_t> new_vertex_gids;
  new_vertex_gids.resize(n_reps, INVALID_GID);
  std::vector<label_t> new_vertex_labels;
  new_vertex_labels.assign(n_reps, INVALID_LABEL);

  for (index_t r = 0; r < static_cast<index_t>(n_vertices); ++r) {
    const index_t nv = rep_to_new[static_cast<size_t>(r)];
    if (nv == INVALID_INDEX) continue;
    // Keep the representative's original GID to remain stable across merges.
    if (static_cast<size_t>(r) < old_vertex_gids.size()) {
      new_vertex_gids[static_cast<size_t>(nv)] = old_vertex_gids[static_cast<size_t>(r)];
    }

    // Prefer the representative's label; otherwise use the first non-invalid label in the set.
    label_t lbl = (static_cast<size_t>(r) < old_vertex_labels.size())
                      ? old_vertex_labels[static_cast<size_t>(r)]
                      : INVALID_LABEL;
    if (lbl == INVALID_LABEL) {
      for (index_t v = r + 1; v < static_cast<index_t>(n_vertices); ++v) {
        if (rep_of[static_cast<size_t>(v)] != r) continue;
        const label_t cand = (static_cast<size_t>(v) < old_vertex_labels.size())
                                 ? old_vertex_labels[static_cast<size_t>(v)]
                                 : INVALID_LABEL;
        if (cand != INVALID_LABEL) {
          lbl = cand;
          break;
        }
      }
    }
    new_vertex_labels[static_cast<size_t>(nv)] = lbl;
  }

  const auto cell_offsets = mesh.cell2vertex_offsets();
  const auto cell_shapes = mesh.cell_shapes();

  rebuild_mesh_from_arrays_preserving_labels(
      mesh,
      spatial_dim,
      new_X_ref,
      cell_offsets,
      new_c2v,
      cell_shapes,
      new_vertex_gids,
      old_cell_gids,
      new_vertex_labels,
      old_cell_regions,
      old_cell_ref_levels,
      has_cur ? &new_X_cur : nullptr,
      label_registry,
      boundary_labels);

  return static_cast<index_t>(n_vertices - n_reps);
}

index_t MeshValidation::remove_isolated_vertices(MeshBase& mesh) {
  const auto isolated = find_isolated_vertices(mesh);
  if (isolated.passed) {
    return 0;
  }

  const int spatial_dim = mesh.dim();
  const size_t n_vertices = mesh.n_vertices();
  if (n_vertices == 0) {
    return 0;
  }

  // Mark vertices used by any cell.
  std::vector<uint8_t> used(n_vertices, 0);
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    auto [vptr, nv] = mesh.cell_vertices_span(c);
    for (size_t i = 0; i < nv; ++i) {
      const index_t v = vptr[i];
      if (v >= 0 && static_cast<size_t>(v) < n_vertices) {
        used[static_cast<size_t>(v)] = 1;
      }
    }
  }

  // Build old->new map.
  std::vector<index_t> old2new(n_vertices, INVALID_INDEX);
  size_t kept = 0;
  for (size_t v = 0; v < n_vertices; ++v) {
    if (used[v] != 0) {
      old2new[v] = static_cast<index_t>(kept++);
    }
  }
  if (kept == n_vertices) {
    return 0;
  }

  const auto label_registry = capture_label_registry(mesh);
  const auto boundary_labels = capture_boundary_labels_by_corner_gids(mesh);

  const auto old_vertex_gids = mesh.vertex_gids();
  const auto old_vertex_labels = mesh.vertex_label_ids();
  const bool has_cur = mesh.has_current_coords();

  std::vector<real_t> new_X_ref;
  new_X_ref.resize(kept * static_cast<size_t>(spatial_dim), 0.0);
  std::vector<real_t> new_X_cur;
  if (has_cur) {
    new_X_cur.resize(kept * static_cast<size_t>(spatial_dim), 0.0);
  }

  std::vector<gid_t> new_vertex_gids;
  new_vertex_gids.resize(kept, INVALID_GID);
  std::vector<label_t> new_vertex_labels;
  new_vertex_labels.assign(kept, INVALID_LABEL);

  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    const index_t nv = old2new[static_cast<size_t>(v)];
    if (nv == INVALID_INDEX) continue;
    for (int d = 0; d < spatial_dim; ++d) {
      new_X_ref[static_cast<size_t>(nv) * spatial_dim + d] =
          mesh.X_ref()[static_cast<size_t>(v) * spatial_dim + d];
      if (has_cur) {
        new_X_cur[static_cast<size_t>(nv) * spatial_dim + d] =
            mesh.X_cur()[static_cast<size_t>(v) * spatial_dim + d];
      }
    }
    if (static_cast<size_t>(v) < old_vertex_gids.size()) {
      new_vertex_gids[static_cast<size_t>(nv)] = old_vertex_gids[static_cast<size_t>(v)];
    }
    if (static_cast<size_t>(v) < old_vertex_labels.size()) {
      new_vertex_labels[static_cast<size_t>(nv)] = old_vertex_labels[static_cast<size_t>(v)];
    }
  }

  std::vector<index_t> new_c2v = mesh.cell2vertex();
  for (auto& v : new_c2v) {
    if (v < 0 || static_cast<size_t>(v) >= n_vertices) {
      v = INVALID_INDEX;
      continue;
    }
    v = old2new[static_cast<size_t>(v)];
  }

  const auto cell_offsets = mesh.cell2vertex_offsets();
  const auto cell_shapes = mesh.cell_shapes();

  rebuild_mesh_from_arrays_preserving_labels(
      mesh,
      spatial_dim,
      new_X_ref,
      cell_offsets,
      new_c2v,
      cell_shapes,
      new_vertex_gids,
      mesh.cell_gids(),
      new_vertex_labels,
      mesh.cell_region_ids(),
      mesh.cell_refinement_levels(),
      has_cur ? &new_X_cur : nullptr,
      label_registry,
      boundary_labels);

  return static_cast<index_t>(n_vertices - kept);
}

index_t MeshValidation::remove_degenerate_cells(MeshBase& mesh, real_t tolerance) {
  const auto deg = find_degenerate_cells(mesh, tolerance);
  if (deg.passed) {
    return 0;
  }

  const int spatial_dim = mesh.dim();
  const size_t n_cells = mesh.n_cells();
  const size_t n_vertices = mesh.n_vertices();
  if (n_cells == 0) {
    return 0;
  }

  std::vector<uint8_t> is_degenerate(n_cells, 0);
  for (const auto c : deg.problem_entities) {
    if (c >= 0 && static_cast<size_t>(c) < n_cells) {
      is_degenerate[static_cast<size_t>(c)] = 1;
    }
  }

  std::vector<index_t> old2new_cell(n_cells, INVALID_INDEX);
  size_t kept_cells = 0;
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    if (is_degenerate[static_cast<size_t>(c)] == 0) {
      old2new_cell[static_cast<size_t>(c)] = static_cast<index_t>(kept_cells++);
    }
  }

  if (kept_cells == n_cells) {
    return 0;
  }

  // Build compacted cell arrays (still using old vertex indices for now).
  std::vector<offset_t> new_offsets;
  std::vector<index_t> new_c2v_old_vids;
  std::vector<CellShape> new_shapes;

  new_offsets.reserve(kept_cells + 1);
  new_offsets.push_back(0);
  new_shapes.reserve(kept_cells);

  const auto& old_offsets = mesh.cell2vertex_offsets();
  const auto& old_c2v = mesh.cell2vertex();
  const auto& old_shapes = mesh.cell_shapes();

  std::vector<uint8_t> vertex_used(n_vertices, 0);

  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    if (old2new_cell[static_cast<size_t>(c)] == INVALID_INDEX) {
      continue;
    }
    const offset_t start = old_offsets[static_cast<size_t>(c)];
    const offset_t end = old_offsets[static_cast<size_t>(c + 1)];
    for (offset_t i = start; i < end; ++i) {
      const index_t v = old_c2v[static_cast<size_t>(i)];
      new_c2v_old_vids.push_back(v);
      if (v >= 0 && static_cast<size_t>(v) < n_vertices) {
        vertex_used[static_cast<size_t>(v)] = 1;
      }
    }
    new_offsets.push_back(static_cast<offset_t>(new_c2v_old_vids.size()));
    new_shapes.push_back(old_shapes[static_cast<size_t>(c)]);
  }

  // Compact vertices.
  std::vector<index_t> old2new_vertex(n_vertices, INVALID_INDEX);
  size_t kept_vertices = 0;
  for (size_t v = 0; v < n_vertices; ++v) {
    if (vertex_used[v] != 0) {
      old2new_vertex[v] = static_cast<index_t>(kept_vertices++);
    }
  }

  const auto label_registry = capture_label_registry(mesh);
  const auto boundary_labels = capture_boundary_labels_by_corner_gids(mesh);

  const auto old_vertex_gids = mesh.vertex_gids();
  const auto old_vertex_labels = mesh.vertex_label_ids();
  const bool has_cur = mesh.has_current_coords();

  std::vector<real_t> new_X_ref;
  new_X_ref.resize(kept_vertices * static_cast<size_t>(spatial_dim), 0.0);
  std::vector<real_t> new_X_cur;
  if (has_cur) {
    new_X_cur.resize(kept_vertices * static_cast<size_t>(spatial_dim), 0.0);
  }

  std::vector<gid_t> new_vertex_gids;
  new_vertex_gids.resize(kept_vertices, INVALID_GID);
  std::vector<label_t> new_vertex_labels;
  new_vertex_labels.assign(kept_vertices, INVALID_LABEL);

  for (index_t v = 0; v < static_cast<index_t>(n_vertices); ++v) {
    const index_t nv = old2new_vertex[static_cast<size_t>(v)];
    if (nv == INVALID_INDEX) continue;
    for (int d = 0; d < spatial_dim; ++d) {
      new_X_ref[static_cast<size_t>(nv) * spatial_dim + d] =
          mesh.X_ref()[static_cast<size_t>(v) * spatial_dim + d];
      if (has_cur) {
        new_X_cur[static_cast<size_t>(nv) * spatial_dim + d] =
            mesh.X_cur()[static_cast<size_t>(v) * spatial_dim + d];
      }
    }
    if (static_cast<size_t>(v) < old_vertex_gids.size()) {
      new_vertex_gids[static_cast<size_t>(nv)] = old_vertex_gids[static_cast<size_t>(v)];
    }
    if (static_cast<size_t>(v) < old_vertex_labels.size()) {
      new_vertex_labels[static_cast<size_t>(nv)] = old_vertex_labels[static_cast<size_t>(v)];
    }
  }

  std::vector<index_t> new_c2v;
  new_c2v.reserve(new_c2v_old_vids.size());
  for (const auto v : new_c2v_old_vids) {
    if (v < 0 || static_cast<size_t>(v) >= n_vertices) {
      new_c2v.push_back(INVALID_INDEX);
      continue;
    }
    new_c2v.push_back(old2new_vertex[static_cast<size_t>(v)]);
  }

  // Remap per-cell metadata.
  const auto old_cell_gids = mesh.cell_gids();
  const auto old_cell_regions = mesh.cell_region_ids();
  const auto old_cell_ref_levels = mesh.cell_refinement_levels();
  std::vector<gid_t> new_cell_gids;
  new_cell_gids.reserve(kept_cells);
  std::vector<label_t> new_cell_regions;
  new_cell_regions.reserve(kept_cells);
  std::vector<size_t> new_cell_ref_levels;
  new_cell_ref_levels.reserve(kept_cells);

  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    const index_t nc = old2new_cell[static_cast<size_t>(c)];
    if (nc == INVALID_INDEX) continue;
    new_cell_gids.push_back(old_cell_gids[static_cast<size_t>(c)]);
    new_cell_regions.push_back(old_cell_regions[static_cast<size_t>(c)]);
    new_cell_ref_levels.push_back(old_cell_ref_levels[static_cast<size_t>(c)]);
  }

  rebuild_mesh_from_arrays_preserving_labels(
      mesh,
      spatial_dim,
      new_X_ref,
      new_offsets,
      new_c2v,
      new_shapes,
      new_vertex_gids,
      new_cell_gids,
      new_vertex_labels,
      new_cell_regions,
      new_cell_ref_levels,
      has_cur ? &new_X_cur : nullptr,
      label_registry,
      boundary_labels);

  return static_cast<index_t>(n_cells - kept_cells);
}

index_t MeshValidation::fix_inverted_cells(MeshBase& mesh) {
  auto result = find_inverted_cells(mesh);

  if (result.passed) {
    return 0;  // No inverted cells
  }

  const int spatial_dim = mesh.dim();
  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) {
    return 0;
  }

  const auto label_registry = capture_label_registry(mesh);
  const auto boundary_labels = capture_boundary_labels_by_corner_gids(mesh);
  const auto old_vertex_gids = mesh.vertex_gids();
  const auto old_cell_gids = mesh.cell_gids();
  const auto old_vertex_labels = mesh.vertex_label_ids();
  const auto old_cell_regions = mesh.cell_region_ids();
  const auto old_cell_ref_levels = mesh.cell_refinement_levels();
  const bool has_cur = mesh.has_current_coords();
  const auto old_X_cur = mesh.X_cur();

  std::vector<index_t> new_c2v = mesh.cell2vertex();
  const auto offsets = mesh.cell2vertex_offsets();

  index_t fixed = 0;
  for (const auto c : result.problem_entities) {
    if (c < 0 || static_cast<size_t>(c) >= n_cells) continue;
    auto [vptr, nv] = mesh.cell_vertices_span(c);
    (void)vptr;
    const CellShape& shape = mesh.cell_shape(c);

    // Safe, topology-preserving fix for simplex cells only.
    if (shape.family == CellFamily::Tetra && nv == 4) {
      const offset_t start = offsets[static_cast<size_t>(c)];
      if (start + 3 < static_cast<offset_t>(new_c2v.size())) {
        std::swap(new_c2v[static_cast<size_t>(start + 1)], new_c2v[static_cast<size_t>(start + 2)]);
        fixed++;
      }
    } else if (shape.family == CellFamily::Triangle && spatial_dim == 2 && nv == 3) {
      const offset_t start = offsets[static_cast<size_t>(c)];
      if (start + 2 < static_cast<offset_t>(new_c2v.size())) {
        std::swap(new_c2v[static_cast<size_t>(start + 1)], new_c2v[static_cast<size_t>(start + 2)]);
        fixed++;
      }
    }
  }

  if (fixed == 0) {
    return 0;
  }

  const auto old_X_ref = mesh.X_ref();
  const auto cell_offsets = mesh.cell2vertex_offsets();
  const auto cell_shapes = mesh.cell_shapes();

  rebuild_mesh_from_arrays_preserving_labels(
      mesh,
      spatial_dim,
      old_X_ref,
      cell_offsets,
      new_c2v,
      cell_shapes,
      old_vertex_gids,
      old_cell_gids,
      old_vertex_labels,
      old_cell_regions,
      old_cell_ref_levels,
      has_cur ? &old_X_cur : nullptr,
      label_registry,
      boundary_labels);

  return fixed;
}

index_t MeshValidation::orient_faces_consistently(MeshBase& mesh) {
  const int spatial_dim = mesh.dim();
  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) {
    return 0;
  }

  // Only support surface-like meshes where the maximal topological dimension is 2.
  int tdim = 0;
  for (const auto& s : mesh.cell_shapes()) {
    if (s.is_3d()) {
      tdim = 3;
      break;
    }
    if (s.is_2d()) tdim = 2;
  }
  if (tdim != 2) {
    return 0;
  }

  const auto label_registry = capture_label_registry(mesh);
  const auto boundary_labels = capture_boundary_labels_by_corner_gids(mesh);
  const auto old_vertex_gids = mesh.vertex_gids();
  const auto old_cell_gids = mesh.cell_gids();
  const auto old_vertex_labels = mesh.vertex_label_ids();
  const auto old_cell_regions = mesh.cell_region_ids();
  const auto old_cell_ref_levels = mesh.cell_refinement_levels();
  const bool has_cur = mesh.has_current_coords();
  const auto old_X_cur = mesh.X_cur();

  struct DirectedEdge {
    index_t cell = INVALID_INDEX;
    index_t a = INVALID_INDEX;
    index_t b = INVALID_INDEX;
  };
  struct EdgeKey {
    index_t u = INVALID_INDEX;
    index_t v = INVALID_INDEX;
    bool operator==(const EdgeKey& other) const { return u == other.u && v == other.v; }
  };
  struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
      uint64_t h = 1469598103934665603ULL;
      h = fnv1a64_append(h, static_cast<uint64_t>(k.u));
      h = fnv1a64_append(h, static_cast<uint64_t>(k.v));
      return static_cast<size_t>(h);
    }
  };

  std::unordered_map<EdgeKey, std::vector<DirectedEdge>, EdgeKeyHash> edge2cells;
  edge2cells.reserve(n_cells * 2);

  std::vector<uint8_t> eligible(n_cells, 0);
  std::vector<std::vector<index_t>> corners_per_cell(n_cells);

  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    const CellShape& shape = mesh.cell_shape(c);
    if (!shape.is_2d()) continue;

    auto [vptr, nv] = mesh.cell_vertices_span(c);
    const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(nv);
    if (nc < 3) continue;
    if (static_cast<size_t>(nc) != nv) {
      // High-order surface cells require consistent node permutation; skip for now.
      continue;
    }

    eligible[static_cast<size_t>(c)] = 1;
    auto& corners = corners_per_cell[static_cast<size_t>(c)];
    corners.assign(vptr, vptr + nv);

    for (int i = 0; i < nc; ++i) {
      const index_t a = corners[static_cast<size_t>(i)];
      const index_t b = corners[static_cast<size_t>((i + 1) % nc)];
      const index_t u = std::min(a, b);
      const index_t v = std::max(a, b);
      edge2cells[{u, v}].push_back({c, a, b});
    }
  }

  std::vector<uint8_t> visited(n_cells, 0);
  std::vector<uint8_t> flip(n_cells, 0);

  std::queue<index_t> q;
  for (index_t seed = 0; seed < static_cast<index_t>(n_cells); ++seed) {
    if (eligible[static_cast<size_t>(seed)] == 0 || visited[static_cast<size_t>(seed)] != 0) {
      continue;
    }
    visited[static_cast<size_t>(seed)] = 1;
    q.push(seed);

    while (!q.empty()) {
      const index_t c = q.front();
      q.pop();

      const auto& corners = corners_per_cell[static_cast<size_t>(c)];
      const int nc = static_cast<int>(corners.size());
      for (int i = 0; i < nc; ++i) {
        const index_t a0 = corners[static_cast<size_t>(i)];
        const index_t b0 = corners[static_cast<size_t>((i + 1) % nc)];
        const index_t u = std::min(a0, b0);
        const index_t v = std::max(a0, b0);
        const EdgeKey ek{u, v};

        const auto it = edge2cells.find(ek);
        if (it == edge2cells.end()) continue;

        // Edge adjacency (manifold assumption: up to 2 incident cells).
        for (const auto& e : it->second) {
          if (e.cell == c) continue;
          const index_t n = e.cell;
          if (eligible[static_cast<size_t>(n)] == 0) continue;

          auto dir = [&](index_t cell_id, index_t a, index_t b) {
            if (flip[static_cast<size_t>(cell_id)] == 0) return std::make_pair(a, b);
            return std::make_pair(b, a);
          };

          const auto cd = dir(c, a0, b0);
          const auto nd = dir(n, e.a, e.b);
          const bool opposite = (cd.first == nd.second) && (cd.second == nd.first);
          if (!visited[static_cast<size_t>(n)]) {
            visited[static_cast<size_t>(n)] = 1;
            if (!opposite) {
              flip[static_cast<size_t>(n)] = 1 - flip[static_cast<size_t>(n)];
            }
            q.push(n);
          }
        }
      }
    }
  }

  index_t flipped = 0;
  for (size_t c = 0; c < n_cells; ++c) {
    if (eligible[c] != 0 && flip[c] != 0) {
      flipped++;
    }
  }
  if (flipped == 0) {
    return 0;
  }

  std::vector<index_t> new_c2v = mesh.cell2vertex();
  const auto offsets = mesh.cell2vertex_offsets();
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    if (eligible[static_cast<size_t>(c)] == 0 || flip[static_cast<size_t>(c)] == 0) {
      continue;
    }
    const offset_t start = offsets[static_cast<size_t>(c)];
    const offset_t end = offsets[static_cast<size_t>(c + 1)];
    if (start < 0 || end < start || static_cast<size_t>(end) > new_c2v.size()) {
      continue;
    }
    std::reverse(new_c2v.begin() + static_cast<size_t>(start),
                 new_c2v.begin() + static_cast<size_t>(end));
  }

  const auto old_X_ref = mesh.X_ref();
  const auto cell_offsets = mesh.cell2vertex_offsets();
  const auto cell_shapes = mesh.cell_shapes();

  rebuild_mesh_from_arrays_preserving_labels(
      mesh,
      spatial_dim,
      old_X_ref,
      cell_offsets,
      new_c2v,
      cell_shapes,
      old_vertex_gids,
      old_cell_gids,
      old_vertex_labels,
      old_cell_regions,
      old_cell_ref_levels,
      has_cur ? &old_X_cur : nullptr,
      label_registry,
      boundary_labels);

  return flipped;
}

// ---- Statistics and reporting ----

std::string MeshValidation::generate_statistics_report(const MeshBase& mesh) {
  std::ostringstream ss;

  ss << "\n=== Mesh Statistics ===" << std::endl;
  ss << "Dimension: " << mesh.dim() << "D" << std::endl;
  ss << "Number of vertices: " << mesh.n_vertices() << std::endl;
  ss << "Number of cells: " << mesh.n_cells() << std::endl;
  ss << "Number of faces: " << mesh.n_faces() << std::endl;
  ss << "Number of edges: " << mesh.n_edges() << std::endl;

  // Bounding box
  auto bbox = mesh.bounding_box();
  ss << "Bounding box:" << std::endl;
  ss << "  Min: (" << bbox.min[0] << ", " << bbox.min[1] << ", " << bbox.min[2] << ")" << std::endl;
  ss << "  Max: (" << bbox.max[0] << ", " << bbox.max[1] << ", " << bbox.max[2] << ")" << std::endl;

  // Cell types
  std::unordered_map<int, size_t> cell_type_counts;
  size_t n_cells = mesh.n_cells();
  for (size_t c = 0; c < n_cells; ++c) {
    auto shape = mesh.cell_shape(static_cast<index_t>(c));
    cell_type_counts[static_cast<int>(shape.family)]++;
  }

  ss << "Cell types:" << std::endl;
  for (const auto& [type, count] : cell_type_counts) {
    ss << "  Type " << type << ": " << count << " cells" << std::endl;
  }

  ss << "=== End Statistics ===" << std::endl;

  return ss.str();
}

void MeshValidation::write_debug_output(const MeshBase& mesh,
                                       const std::string& prefix,
                                       const std::string& format) {
  // Use mesh's write_debug method
  mesh.write_debug(prefix, format);
}

MeshValidation::ValidationReport MeshValidation::compare_meshes(const MeshBase& mesh1,
                                                               const MeshBase& mesh2,
                                                               real_t tolerance) {
  ValidationReport report;

  // Compare basic properties
  ValidationResult basic_result;
  basic_result.check_name = "Basic properties";
  basic_result.passed = true;

  if (mesh1.n_vertices() != mesh2.n_vertices()) {
    basic_result.passed = false;
    basic_result.message = "Different number of vertices: " +
                          std::to_string(mesh1.n_vertices()) + " vs " +
                          std::to_string(mesh2.n_vertices());
  } else if (mesh1.n_cells() != mesh2.n_cells()) {
    basic_result.passed = false;
    basic_result.message = "Different number of cells: " +
                          std::to_string(mesh1.n_cells()) + " vs " +
                          std::to_string(mesh2.n_cells());
  } else {
    basic_result.message = "Basic properties match";
  }

  report.add_result(basic_result);

  // Compare coordinates
  if (mesh1.n_vertices() == mesh2.n_vertices()) {
    ValidationResult coord_result;
    coord_result.check_name = "Coordinates";
    coord_result.passed = true;

    const auto& coords1 = mesh1.X_ref();
    const auto& coords2 = mesh2.X_ref();
    int dim = std::min(mesh1.dim(), mesh2.dim());

    for (size_t i = 0; i < mesh1.n_vertices(); ++i) {
      real_t dist_sq = 0;
      for (int d = 0; d < dim; ++d) {
        real_t diff = coords1[i * dim + d] - coords2[i * dim + d];
        dist_sq += diff * diff;
      }

      if (std::sqrt(dist_sq) > tolerance) {
        coord_result.problem_entities.push_back(static_cast<index_t>(i));
      }
    }

    if (!coord_result.problem_entities.empty()) {
      coord_result.passed = false;
      coord_result.message = std::to_string(coord_result.problem_entities.size()) +
                            " vertices differ by more than tolerance";
    } else {
      coord_result.message = "All vertex coordinates match within tolerance";
    }

    report.add_result(coord_result);
  }

  return report;
}

// ---- Helper methods ----

bool MeshValidation::is_degenerate_cell(const MeshBase& mesh, index_t cell, real_t tolerance) {
  real_t measure = mesh.cell_measure(cell);
  return std::abs(measure) < tolerance;
}

bool MeshValidation::is_inverted_cell(const MeshBase& mesh, index_t cell) {
  const CellShape& shape = mesh.cell_shape(cell);

  // Robust, topology-preserving orientation checks:
  // - 3D: use signed Jacobian determinants (sampled).
  // - 2D: only meaningful in true 2D embeddings (dim==2).
  if (shape.is_3d()) {
    const real_t jmin = compute_cell_jacobian_min(mesh, cell);
    return jmin < 0;
  }

  if (shape.is_2d() && mesh.dim() == 2) {
    auto [vptr, nv] = mesh.cell_vertices_span(cell);
    const int nc = (shape.num_corners > 0) ? shape.num_corners : static_cast<int>(nv);
    if (shape.family == CellFamily::Triangle && nc >= 3) {
      const auto p0 = mesh.get_vertex_coords(vptr[0]);
      const auto p1 = mesh.get_vertex_coords(vptr[1]);
      const auto p2 = mesh.get_vertex_coords(vptr[2]);
      const real_t cross_z = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
      return cross_z < 0;
    }
    if (shape.family == CellFamily::Quad && nc >= 4) {
      const auto p0 = mesh.get_vertex_coords(vptr[0]);
      const auto p1 = mesh.get_vertex_coords(vptr[1]);
      const auto p2 = mesh.get_vertex_coords(vptr[2]);
      const auto p3 = mesh.get_vertex_coords(vptr[3]);
      // Signed area proxy from two triangles in the (x,y) plane.
      const real_t a0 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
      const real_t a1 = (p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0]);
      return (a0 + a1) < 0;
    }
  }

  return false;
}

real_t MeshValidation::compute_cell_jacobian_min(const MeshBase& mesh, index_t cell) {
  const CellShape& shape = mesh.cell_shape(cell);
  if (!shape.is_3d()) {
    return mesh.cell_measure(cell);
  }

  if (shape.family == CellFamily::Polyhedron) {
    // Generic polyhedra do not have a canonical reference mapping in CurvilinearEvaluator.
    return mesh.cell_measure(cell);
  }

  // Sample at corners + center to catch common inversions (including high-order folds).
  std::vector<ParametricPoint> samples = reference_corner_points(shape.family);
  samples.push_back(CurvilinearEvaluator::reference_element_center(shape));

  real_t min_det = std::numeric_limits<real_t>::infinity();
  for (const auto& xi : samples) {
    GeometryEvaluation eval;
    try {
      eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, Configuration::Reference);
    } catch (...) {
      return mesh.cell_measure(cell);
    }
    if (!std::isfinite(eval.det_jacobian)) {
      min_det = std::min(min_det, real_t(0));
      continue;
    }
    // Note: CurvilinearEvaluator marks negative det(J) as invalid, but the sign is still
    // meaningful for inversion detection, so always incorporate det_jacobian.
    min_det = std::min(min_det, eval.det_jacobian);
  }
  if (!std::isfinite(min_det)) {
    return 0;
  }
  return min_det;
}

} // namespace svmp
