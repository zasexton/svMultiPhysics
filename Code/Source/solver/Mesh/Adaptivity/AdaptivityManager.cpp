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

#include "AdaptivityManager.h"
#include "Conformity.h"
#include "FEInterface.h"
#include "FieldTransfer.h"
#include "QualityGuards.h"
#include "RefinementDelta.h"
#include "../Core/MeshBase.h"
#include "../Geometry/CurvilinearEval.h"
#include "../Topology/CellTopology.h"
#include "../Fields/MeshFields.h"
#include "../Labels/MeshLabels.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace svmp {

namespace {

gid_t max_gid_or(const std::vector<gid_t>& gids, gid_t fallback) {
  gid_t max_g = fallback;
  for (gid_t g : gids) {
    if (g != INVALID_GID) max_g = std::max(max_g, g);
  }
  return max_g;
}

struct WeightedGidKey {
  struct Term {
    gid_t gid = INVALID_GID;
    long long w = 0; // scaled
  };
  std::vector<Term> terms;

  bool operator==(const WeightedGidKey& o) const noexcept {
    if (terms.size() != o.terms.size()) return false;
    for (size_t i = 0; i < terms.size(); ++i) {
      if (terms[i].gid != o.terms[i].gid) return false;
      if (terms[i].w != o.terms[i].w) return false;
    }
    return true;
  }
};

struct WeightedGidKeyHash {
  size_t operator()(const WeightedGidKey& k) const noexcept {
    size_t h = 0;
    auto mix = [&](size_t x) {
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<size_t>{}(k.terms.size()));
    for (const auto& t : k.terms) {
      mix(std::hash<gid_t>{}(t.gid));
      mix(std::hash<long long>{}(t.w));
    }
    return h;
  }
};

WeightedGidKey make_key_from_gid_weights(const std::vector<std::pair<gid_t, double>>& weights) {
  static constexpr long double kScale = 1.0e12L;
  WeightedGidKey k;
  k.terms.reserve(weights.size());
  for (const auto& [g, w] : weights) {
    if (g == INVALID_GID) continue;
    const long double ws = static_cast<long double>(w) * kScale;
    const long long wi = static_cast<long long>(std::llround(ws));
    if (wi == 0) continue;
    k.terms.push_back({g, wi});
  }
  std::sort(k.terms.begin(), k.terms.end(), [](const auto& a, const auto& b) {
    if (a.gid != b.gid) return a.gid < b.gid;
    return a.w < b.w;
  });
  // Combine duplicate gids (can happen in recursive assembly).
  std::vector<WeightedGidKey::Term> merged;
  for (const auto& t : k.terms) {
    if (!merged.empty() && merged.back().gid == t.gid) {
      merged.back().w += t.w;
    } else {
      merged.push_back(t);
    }
  }
  k.terms.swap(merged);
  return k;
}

struct GidVecHash {
  size_t operator()(const std::vector<gid_t>& v) const noexcept {
    size_t h = 0;
    auto mix = [&](size_t x) {
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<size_t>{}(v.size()));
    for (gid_t g : v) mix(std::hash<gid_t>{}(g));
    return h;
  }
};

std::vector<gid_t> sorted_unique(std::vector<gid_t> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

int default_num_corners(CellFamily family) {
  switch (family) {
    case CellFamily::Point:   return 1;
    case CellFamily::Line:    return 2;
    case CellFamily::Triangle:return 3;
    case CellFamily::Quad:    return 4;
    case CellFamily::Tetra:   return 4;
    case CellFamily::Hex:     return 8;
    case CellFamily::Wedge:   return 6;
    case CellFamily::Pyramid: return 5;
    case CellFamily::Polygon:
    case CellFamily::Polyhedron:
      return 0;
  }
  return 0;
}

std::vector<std::array<real_t, 3>> reference_corners(CellFamily family) {
  switch (family) {
    case CellFamily::Line:
      return {{-1.0, 0.0, 0.0}, {+1.0, 0.0, 0.0}};
    case CellFamily::Triangle:
      return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    case CellFamily::Quad:
      return {{-1.0, -1.0, 0.0}, {+1.0, -1.0, 0.0}, {+1.0, +1.0, 0.0}, {-1.0, +1.0, 0.0}};
    case CellFamily::Tetra:
      return {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    case CellFamily::Hex:
      return {{-1.0, -1.0, -1.0}, {+1.0, -1.0, -1.0}, {+1.0, +1.0, -1.0}, {-1.0, +1.0, -1.0},
              {-1.0, -1.0, +1.0}, {+1.0, -1.0, +1.0}, {+1.0, +1.0, +1.0}, {-1.0, +1.0, +1.0}};
    case CellFamily::Wedge:
      return {{0.0, 0.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
              {0.0, 0.0, +1.0}, {1.0, 0.0, +1.0}, {0.0, 1.0, +1.0}};
    case CellFamily::Pyramid:
      return {{-1.0, -1.0, 0.0}, {+1.0, -1.0, 0.0}, {+1.0, +1.0, 0.0}, {-1.0, +1.0, 0.0}, {0.0, 0.0, 1.0}};
    default:
      break;
  }
  throw std::runtime_error("reference_corners(): unsupported cell family for high-order refinement");
}

std::vector<ParametricPoint> quadratic_reference_nodes(CellFamily family, size_t n_nodes) {
  switch (family) {
    case CellFamily::Line: {
      if (n_nodes != 3u) throw std::runtime_error("quadratic_reference_nodes(Line): expected 3 nodes");
      return {ParametricPoint{-1.0, 0.0, 0.0}, ParametricPoint{+1.0, 0.0, 0.0}, ParametricPoint{0.0, 0.0, 0.0}};
    }
    case CellFamily::Triangle: {
      if (n_nodes != 6u) throw std::runtime_error("quadratic_reference_nodes(Triangle): expected 6 nodes");
      return {ParametricPoint{0.0, 0.0, 0.0},   ParametricPoint{1.0, 0.0, 0.0},   ParametricPoint{0.0, 1.0, 0.0},
              ParametricPoint{0.5, 0.0, 0.0},   ParametricPoint{0.5, 0.5, 0.0},   ParametricPoint{0.0, 0.5, 0.0}};
    }
    case CellFamily::Quad: {
      if (n_nodes != 8u && n_nodes != 9u) throw std::runtime_error("quadratic_reference_nodes(Quad): expected 8 or 9 nodes");
      std::vector<ParametricPoint> xi;
      xi.reserve(n_nodes);
      xi.push_back({-1.0, -1.0, 0.0});
      xi.push_back({+1.0, -1.0, 0.0});
      xi.push_back({+1.0, +1.0, 0.0});
      xi.push_back({-1.0, +1.0, 0.0});
      xi.push_back({0.0, -1.0, 0.0});  // (0-1)
      xi.push_back({+1.0, 0.0, 0.0});  // (1-2)
      xi.push_back({0.0, +1.0, 0.0});  // (2-3)
      xi.push_back({-1.0, 0.0, 0.0});  // (3-0)
      if (n_nodes == 9u) {
        xi.push_back({0.0, 0.0, 0.0});
      }
      return xi;
    }
    case CellFamily::Tetra: {
      if (n_nodes != 10u) throw std::runtime_error("quadratic_reference_nodes(Tetra): expected 10 nodes");
      return {ParametricPoint{0.0, 0.0, 0.0},   ParametricPoint{1.0, 0.0, 0.0},   ParametricPoint{0.0, 1.0, 0.0},   ParametricPoint{0.0, 0.0, 1.0},
              ParametricPoint{0.5, 0.0, 0.0},   ParametricPoint{0.0, 0.5, 0.0},   ParametricPoint{0.0, 0.0, 0.5},   ParametricPoint{0.5, 0.5, 0.0},
              ParametricPoint{0.5, 0.0, 0.5},   ParametricPoint{0.0, 0.5, 0.5}};
    }
    case CellFamily::Hex: {
      if (n_nodes != 20u && n_nodes != 27u) throw std::runtime_error("quadratic_reference_nodes(Hex): expected 20 or 27 nodes");
      std::vector<ParametricPoint> xi;
      xi.reserve(n_nodes);
      // Corners (VTK)
      xi.push_back({-1.0, -1.0, -1.0}); // 0
      xi.push_back({+1.0, -1.0, -1.0}); // 1
      xi.push_back({+1.0, +1.0, -1.0}); // 2
      xi.push_back({-1.0, +1.0, -1.0}); // 3
      xi.push_back({-1.0, -1.0, +1.0}); // 4
      xi.push_back({+1.0, -1.0, +1.0}); // 5
      xi.push_back({+1.0, +1.0, +1.0}); // 6
      xi.push_back({-1.0, +1.0, +1.0}); // 7
      // Edge mids (VTK edge order)
      xi.push_back({0.0, -1.0, -1.0});  // (0-1)
      xi.push_back({+1.0, 0.0, -1.0});  // (1-2)
      xi.push_back({0.0, +1.0, -1.0});  // (2-3)
      xi.push_back({-1.0, 0.0, -1.0});  // (3-0)
      xi.push_back({0.0, -1.0, +1.0});  // (4-5)
      xi.push_back({+1.0, 0.0, +1.0});  // (5-6)
      xi.push_back({0.0, +1.0, +1.0});  // (6-7)
      xi.push_back({-1.0, 0.0, +1.0});  // (7-4)
      xi.push_back({-1.0, -1.0, 0.0});  // (0-4)
      xi.push_back({+1.0, -1.0, 0.0});  // (1-5)
      xi.push_back({+1.0, +1.0, 0.0});  // (2-6)
      xi.push_back({-1.0, +1.0, 0.0});  // (3-7)
      if (n_nodes == 27u) {
        // Face centers in VTK face order: bottom, top, y=-1, x=+1, y=+1, x=-1.
        xi.push_back({0.0, 0.0, -1.0});
        xi.push_back({0.0, 0.0, +1.0});
        xi.push_back({0.0, -1.0, 0.0});
        xi.push_back({+1.0, 0.0, 0.0});
        xi.push_back({0.0, +1.0, 0.0});
        xi.push_back({-1.0, 0.0, 0.0});
        // Center
        xi.push_back({0.0, 0.0, 0.0});
      }
      return xi;
    }
    case CellFamily::Wedge: {
      if (n_nodes != 15u && n_nodes != 18u) throw std::runtime_error("quadratic_reference_nodes(Wedge): expected 15 or 18 nodes");
      std::vector<ParametricPoint> xi;
      xi.reserve(n_nodes);

      auto corner = [](int c) -> ParametricPoint {
        switch (c) {
          case 0: return {0.0, 0.0, -1.0};
          case 1: return {1.0, 0.0, -1.0};
          case 2: return {0.0, 1.0, -1.0};
          case 3: return {0.0, 0.0, +1.0};
          case 4: return {1.0, 0.0, +1.0};
          case 5: return {0.0, 1.0, +1.0};
          default: return {0.0, 0.0, 0.0};
        }
      };

      // Corners (VTK)
      for (int c = 0; c < 6; ++c) xi.push_back(corner(c));

      // Edge mids (VTK edge order via CellTopology view)
      const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
      for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = corner(a);
        const auto B = corner(b);
        xi.push_back({0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1]), 0.5 * (A[2] + B[2])});
      }

      if (n_nodes == 18u) {
        // Quad-face centers in VTK oriented-face order. Only the 3 quad faces contribute.
        const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
        for (int fi = 0; fi < fview.face_count; ++fi) {
          const int b = fview.offsets[fi];
          const int e = fview.offsets[fi + 1];
          const int fv = e - b;
          if (fv != 4) continue;
          ParametricPoint c{0.0, 0.0, 0.0};
          for (int k = 0; k < 4; ++k) {
            const auto P = corner(fview.indices[b + k]);
            c[0] += P[0];
            c[1] += P[1];
            c[2] += P[2];
          }
          xi.push_back({0.25 * c[0], 0.25 * c[1], 0.25 * c[2]});
        }
      }

      if (xi.size() != n_nodes) {
        throw std::runtime_error("quadratic_reference_nodes(Wedge): node count mismatch");
      }
      return xi;
    }
    case CellFamily::Pyramid: {
      if (n_nodes != 13u && n_nodes != 14u) throw std::runtime_error("quadratic_reference_nodes(Pyramid): expected 13 or 14 nodes");
      std::vector<ParametricPoint> xi;
      xi.reserve(n_nodes);

      auto corner = [](int c) -> ParametricPoint {
        switch (c) {
          case 0: return {-1.0, -1.0, 0.0};
          case 1: return {+1.0, -1.0, 0.0};
          case 2: return {+1.0, +1.0, 0.0};
          case 3: return {-1.0, +1.0, 0.0};
          case 4: return {0.0, 0.0, 1.0};
          default: return {0.0, 0.0, 0.0};
        }
      };

      // Corners (VTK)
      for (int c = 0; c < 5; ++c) xi.push_back(corner(c));

      // Edge mids (VTK edge order)
      const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
      for (int ei = 0; ei < eview.edge_count; ++ei) {
        const int a = eview.pairs_flat[2 * ei + 0];
        const int b = eview.pairs_flat[2 * ei + 1];
        const auto A = corner(a);
        const auto B = corner(b);
        xi.push_back({0.5 * (A[0] + B[0]), 0.5 * (A[1] + B[1]), 0.5 * (A[2] + B[2])});
      }

      if (n_nodes == 14u) {
        // Base face center only (quad face in oriented-face order).
        const auto fview = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
        for (int fi = 0; fi < fview.face_count; ++fi) {
          const int b = fview.offsets[fi];
          const int e = fview.offsets[fi + 1];
          const int fv = e - b;
          if (fv != 4) continue;
          ParametricPoint c{0.0, 0.0, 0.0};
          for (int k = 0; k < 4; ++k) {
            const auto P = corner(fview.indices[b + k]);
            c[0] += P[0];
            c[1] += P[1];
            c[2] += P[2];
          }
          xi.push_back({0.25 * c[0], 0.25 * c[1], 0.25 * c[2]});
        }
      }

      if (xi.size() != n_nodes) {
        throw std::runtime_error("quadratic_reference_nodes(Pyramid): node count mismatch");
      }
      return xi;
    }
    default:
      break;
  }
  throw std::runtime_error("quadratic_reference_nodes(): unsupported cell family for high-order refinement");
}

bool is_single_injection(const std::vector<std::pair<size_t, double>>& weights,
                         size_t& parent_local,
                         double tol = 1e-12) {
  if (weights.size() != 1u) return false;
  if (std::abs(weights[0].second - 1.0) > tol) return false;
  parent_local = weights[0].first;
  return true;
}

double min_child_quality_from_refined_element(int spatial_dim,
                                              const std::vector<std::array<real_t, 3>>& parent_corners,
                                              const RefinedElement& refined) {
  std::vector<std::array<real_t, 3>> pts = parent_corners;
  pts.insert(pts.end(), refined.new_vertices.begin(), refined.new_vertices.end());

  MeshBase tmp(spatial_dim);
  for (size_t i = 0; i < pts.size(); ++i) {
    tmp.add_vertex(static_cast<index_t>(i), pts[i]);
  }
  for (size_t c = 0; c < refined.child_connectivity.size(); ++c) {
    std::vector<index_t> conn;
    conn.reserve(refined.child_connectivity[c].size());
    for (size_t li : refined.child_connectivity[c]) {
      conn.push_back(static_cast<index_t>(li));
    }
    tmp.add_cell(static_cast<index_t>(c), refined.child_families[c], conn);
  }
  tmp.finalize();

  GeometricQualityChecker checker;
  double min_q = std::numeric_limits<double>::infinity();
  for (size_t c = 0; c < refined.child_connectivity.size(); ++c) {
    const auto q = checker.compute_element_quality(tmp, c);
    min_q = std::min(min_q, q.overall_quality());
  }
  if (!std::isfinite(min_q)) return 0.0;
  return min_q;
}

} // namespace

std::string AdaptivityResult::summary() const {
  std::ostringstream ss;
  ss << "Adaptivity Result Summary\n";
  ss << "  Success: " << (success ? "true" : "false") << "\n";
  ss << "  Cells: " << initial_cell_count << " -> " << final_cell_count << "\n";
  ss << "  Vertices: " << initial_vertex_count << " -> " << final_vertex_count << "\n";
  ss << "  Refined: " << num_refined << "\n";
  ss << "  Coarsened: " << num_coarsened << "\n";
  ss << "  Min quality: " << min_quality << "\n";
  ss << "  Avg quality: " << avg_quality << "\n";
  return ss.str();
}

AdaptivityManager::AdaptivityManager(const AdaptivityOptions& options)
    : options_(options) {
  if (!error_estimator_) {
    error_estimator_ = ErrorEstimatorFactory::create(options_);
  }
  if (!marker_) {
    marker_ = MarkerFactory::create(options_);
  }
  if (!field_transfer_) {
    field_transfer_ = FieldTransferFactory::create(options_);
  }
  if (!quality_checker_) {
    quality_checker_ = QualityCheckerFactory::create(options_);
  }
  if (!conformity_enforcer_) {
    conformity_enforcer_ = ConformityEnforcerFactory::create(options_);
  }
}

AdaptivityManager::~AdaptivityManager() = default;

AdaptivityResult AdaptivityManager::adapt(MeshBase& mesh, MeshFields* fields) {
  AdaptivityResult result;
  result.initial_cell_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();

  if (mesh.n_cells() == 0) {
    result.final_cell_count = 0;
    result.final_vertex_count = mesh.n_vertices();
    result.success = true;
    return result;
  }

  // Compute error indicators + marks.
  last_indicators_ = estimate_error(mesh, fields);
  last_marks_ = mark_elements(last_indicators_, mesh);

  // Execute a single refinement/coarsening pass.
  auto [refine_flags, coarsen_flags] = MarkerUtils::marks_to_flags(last_marks_);
  if (options_.enable_refinement) {
    auto r = refine(mesh, refine_flags, fields);
    result.warning_messages.insert(result.warning_messages.end(),
                                  r.warning_messages.begin(), r.warning_messages.end());
    result.error_messages.insert(result.error_messages.end(),
                                r.error_messages.begin(), r.error_messages.end());
    result.refinement_delta = std::move(r.refinement_delta);
    result.num_refined = r.num_refined;
  }
  if (options_.enable_coarsening) {
    auto r = coarsen(mesh, coarsen_flags, fields);
    result.warning_messages.insert(result.warning_messages.end(),
                                  r.warning_messages.begin(), r.warning_messages.end());
    result.error_messages.insert(result.error_messages.end(),
                                r.error_messages.begin(), r.error_messages.end());
    result.num_coarsened = r.num_coarsened;
  }

  result.final_cell_count = mesh.n_cells();
  result.final_vertex_count = mesh.n_vertices();
  result.success = result.error_messages.empty();
  return result;
}

AdaptivityResult AdaptivityManager::refine(MeshBase& mesh,
                                          const std::vector<bool>& marks,
                                          MeshFields* fields) {
  AdaptivityResult result;
  result.initial_cell_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();

  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) {
    result.final_cell_count = 0;
    result.final_vertex_count = mesh.n_vertices();
    result.success = true;
    return result;
  }

  // Normalize marks length.
  std::vector<MarkType> mark_types(n_cells, MarkType::NONE);
  if (!marks.empty() && marks.size() != n_cells) {
    result.error_messages.push_back("Refine(): marks size mismatch");
    result.final_cell_count = mesh.n_cells();
    result.final_vertex_count = mesh.n_vertices();
    result.success = false;
    return result;
  }

  // Apply max_refinement_level pre-filter.
  for (size_t c = 0; c < n_cells; ++c) {
    if (!marks.empty() && !marks[c]) continue;
    if (!options_.enable_refinement) continue;
    const size_t lvl = mesh.refinement_level(static_cast<index_t>(c));
    if (lvl >= options_.max_refinement_level) continue;
    mark_types[c] = MarkType::REFINE;
  }

  // Enforce conformity if requested.
  std::map<size_t, RefinementSpec> refinement_specs;
  if (conformity_enforcer_ &&
      options_.conformity_mode == AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING) {
    (void)conformity_enforcer_->enforce_conformity(mesh, mark_types, options_);
    if (auto* closure = dynamic_cast<ClosureConformityEnforcer*>(conformity_enforcer_.get())) {
      refinement_specs = closure->get_cell_refinement_specs();
    }
  }

  last_marks_ = mark_types;

  // Count refined parents (excluding those blocked by max level).
  size_t num_parents_refined = 0;
  for (size_t c = 0; c < n_cells; ++c) {
    if (mark_types[c] != MarkType::REFINE) continue;
    if (mesh.refinement_level(static_cast<index_t>(c)) >= options_.max_refinement_level) continue;
    num_parents_refined++;
  }
  result.num_refined = num_parents_refined;

  if (num_parents_refined == 0) {
    result.final_cell_count = mesh.n_cells();
    result.final_vertex_count = mesh.n_vertices();
    result.success = true;
    return result;
  }

  // Build boundary label lookup from the pre-refinement mesh.
  std::unordered_map<std::vector<gid_t>, label_t, GidVecHash> boundary_label_by_root;
  if (mesh.n_faces() > 0) {
    for (index_t f : mesh.boundary_faces()) {
      const label_t lbl = mesh.boundary_label(f);
      if (lbl == INVALID_LABEL) continue;
      std::vector<gid_t> gids;
      for (index_t v : mesh.face_vertices(f)) {
        gids.push_back(mesh.vertex_gids().at(static_cast<size_t>(v)));
      }
      boundary_label_by_root[sorted_unique(std::move(gids))] = lbl;
    }
  }

  const int dim = (mesh.dim() > 0) ? mesh.dim() : 3;
  MeshBase new_mesh(dim);

  // Copy existing vertices (preserve order).
  std::vector<real_t> new_Xref = mesh.X_ref();
  std::vector<gid_t> new_vgids = mesh.vertex_gids();
  gid_t next_v_gid = max_gid_or(new_vgids, 0) + 1;

  // Cell builders.
  std::vector<offset_t> new_offsets;
  new_offsets.reserve(n_cells * 8 + 1);
  new_offsets.push_back(0);
  std::vector<index_t> new_conn;
  std::vector<CellShape> new_shapes;
  std::vector<gid_t> new_cgids;
  std::vector<size_t> new_levels;
  std::vector<label_t> new_regions;

  new_shapes.reserve(n_cells * 8);
  new_cgids.reserve(n_cells * 8);
  new_levels.reserve(n_cells * 8);
  new_regions.reserve(n_cells * 8);

  gid_t next_c_gid = max_gid_or(mesh.cell_gids(), 0) + 1;

  // Vertex dedup map for this refinement pass.
  std::unordered_map<WeightedGidKey, index_t, WeightedGidKeyHash> new_vertex_by_key;

  ParentChildMap parent_child;
  parent_child.child_to_parent.reserve(n_cells * 8);

  auto delta = std::make_unique<RefinementDelta>();

  const auto& old_vgids = mesh.vertex_gids();

  auto ensure_new_vertex = [&](const std::vector<std::pair<size_t, double>>& parent_local_weights,
                               const std::vector<index_t>& parent_cell_vertices,
                               const std::array<real_t, 3>& xyz) -> index_t {
    std::vector<std::pair<gid_t, double>> weights_gid;
    weights_gid.reserve(parent_local_weights.size());
    std::vector<std::pair<size_t, double>> weights_old_index;
    weights_old_index.reserve(parent_local_weights.size());

    for (const auto& [li, w] : parent_local_weights) {
      if (li >= parent_cell_vertices.size()) continue;
      const index_t ov = parent_cell_vertices[li];
      if (ov < 0 || static_cast<size_t>(ov) >= old_vgids.size()) continue;
      weights_gid.push_back({old_vgids[static_cast<size_t>(ov)], w});
      weights_old_index.push_back({static_cast<size_t>(ov), w});
    }

    const auto key = make_key_from_gid_weights(weights_gid);
    const auto it = new_vertex_by_key.find(key);
    if (it != new_vertex_by_key.end()) {
      // Ensure transfer weights exist for reused vertices too.
      if (!weights_old_index.empty()) {
        parent_child.child_vertex_weights[static_cast<size_t>(it->second)] = weights_old_index;
      }
      return it->second;
    }

    const index_t new_vidx = static_cast<index_t>(new_vgids.size());
    for (int d = 0; d < dim; ++d) {
      new_Xref.push_back(xyz[static_cast<size_t>(d)]);
    }
    const gid_t new_gid = next_v_gid++;
    new_vgids.push_back(new_gid);
    new_vertex_by_key.emplace(key, new_vidx);

    MeshLabels::record_vertex_provenance_gid(mesh, new_gid, weights_gid);
    delta->new_vertices.push_back(VertexProvenanceRecord{new_gid, weights_gid});
    if (!weights_old_index.empty()) {
      parent_child.child_vertex_weights[static_cast<size_t>(new_vidx)] = weights_old_index;
    }
    return new_vidx;
  };

  // Build refined + unrefined cells.
  for (size_t c = 0; c < n_cells; ++c) {
    const index_t cell = static_cast<index_t>(c);
    const CellShape& parent_shape = mesh.cell_shape(cell);
    const std::vector<index_t> parent_verts = mesh.cell_vertices(cell);
    const gid_t parent_gid = mesh.cell_gids().at(c);
    const label_t parent_region = mesh.region_label(cell);
    const size_t parent_level = mesh.refinement_level(cell);

    if (mark_types[c] != MarkType::REFINE || parent_level >= options_.max_refinement_level) {
      // Copy cell as-is.
      for (index_t v : parent_verts) new_conn.push_back(v);
      new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
      new_shapes.push_back(parent_shape);
      new_cgids.push_back(parent_gid);
      new_levels.push_back(parent_level);
      new_regions.push_back(parent_region);
      parent_child.child_to_parent.push_back(c);
      continue;
    }

    // Determine refinement spec (closure may have provided one).
    RefinementSpec spec{RefinementPattern::RED, 0u};
    const auto it_spec = refinement_specs.find(c);
    if (it_spec != refinement_specs.end()) {
      spec = it_spec->second;
    }
    if (options_.use_bisection &&
        (parent_shape.family == CellFamily::Triangle || parent_shape.family == CellFamily::Tetra)) {
      spec.pattern = RefinementPattern::BISECTION;
      spec.selector = 0u;
    }

    const int n_corners = (parent_shape.num_corners > 0) ? parent_shape.num_corners : static_cast<int>(parent_verts.size());
    std::vector<gid_t> child_cell_gids;
    const size_t child_level = parent_level + 1;

    if (parent_shape.order <= 1) {
      // Parent corner coordinates (for the rule interface).
      std::vector<std::array<real_t, 3>> corner_xyz;
      corner_xyz.reserve(static_cast<size_t>(n_corners));
      for (int i = 0; i < n_corners; ++i) {
        corner_xyz.push_back(mesh.get_vertex_coords(parent_verts[static_cast<size_t>(i)]));
      }

      // Apply refinement rule.
      RefinedElement refined =
          RefinementRulesManager::instance().refine(corner_xyz, parent_shape.family, spec, parent_level);

      if (options_.enforce_quality_after_refinement &&
          options_.min_refined_quality > 0.0 &&
          spec.pattern != RefinementPattern::RED) {
        const double min_q = min_child_quality_from_refined_element(dim, corner_xyz, refined);
        if (min_q < options_.min_refined_quality) {
          // Upgrade to RED to preserve conformity.
          spec = RefinementSpec{RefinementPattern::RED, 0u};
          refined =
              RefinementRulesManager::instance().refine(corner_xyz, parent_shape.family, spec, parent_level);
        }
      }

      // Local to global vertex mapping for this parent.
      std::vector<index_t> local_to_global;
      local_to_global.reserve(static_cast<size_t>(n_corners) + refined.new_vertices.size());
      for (int i = 0; i < n_corners; ++i) {
        local_to_global.push_back(parent_verts[static_cast<size_t>(i)]);
      }
      for (size_t nv = 0; nv < refined.new_vertices.size(); ++nv) {
        const auto gid_weights = refined.new_vertex_weights[nv];
        const index_t gv = ensure_new_vertex(gid_weights, parent_verts, refined.new_vertices[nv]);
        local_to_global.push_back(gv);
      }

      // Create children.
      child_cell_gids.reserve(refined.child_connectivity.size());
      for (size_t ci = 0; ci < refined.child_connectivity.size(); ++ci) {
        const size_t new_cell_index = new_shapes.size();
        const gid_t child_gid = next_c_gid++;
        child_cell_gids.push_back(child_gid);

        for (size_t li : refined.child_connectivity[ci]) {
          if (li >= local_to_global.size()) {
            throw std::runtime_error("Refine(): invalid child connectivity index");
          }
          new_conn.push_back(local_to_global[li]);
        }
        new_offsets.push_back(static_cast<offset_t>(new_conn.size()));

        CellShape child_shape;
        child_shape.family = refined.child_families[ci];
        child_shape.order = 1;
        child_shape.num_corners = static_cast<int>(refined.child_connectivity[ci].size());
        new_shapes.push_back(child_shape);
        new_cgids.push_back(child_gid);
        new_levels.push_back(refined.child_level);
        new_regions.push_back(parent_region);

        parent_child.child_to_parent.push_back(c);
        parent_child.parent_to_children[c].push_back(new_cell_index);
        parent_child.parent_patterns[c] = options_.refinement_pattern;
      }
    } else if (parent_shape.order == 2) {
      const size_t parent_n_nodes = parent_verts.size();
      const auto corner_ref = reference_corners(parent_shape.family);
      RefinedElement refined_ref =
          RefinementRulesManager::instance().refine(corner_ref, parent_shape.family, spec, parent_level);

      std::vector<ParametricPoint> local_parent_xi;
      local_parent_xi.reserve(static_cast<size_t>(n_corners) + refined_ref.new_vertices.size());
      for (int i = 0; i < n_corners; ++i) {
        const auto& p = corner_ref[static_cast<size_t>(i)];
        local_parent_xi.push_back({p[0], p[1], p[2]});
      }
      for (const auto& p : refined_ref.new_vertices) {
        local_parent_xi.push_back({p[0], p[1], p[2]});
      }

      if (!refined_ref.child_families.empty() && refined_ref.child_families.size() != refined_ref.child_connectivity.size()) {
        throw std::runtime_error("Refine(): child_families size mismatch for quadratic refinement");
      }

      const auto quadratic_lagrange_node_count = [](CellFamily family) -> size_t {
        switch (family) {
          case CellFamily::Line: return 3u;
          case CellFamily::Triangle: return 6u;
          case CellFamily::Quad: return 9u;
          case CellFamily::Tetra: return 10u;
          case CellFamily::Hex: return 27u;
          case CellFamily::Wedge: return 18u;
          case CellFamily::Pyramid: return 14u;
          default: break;
        }
        throw std::runtime_error("Refine(): unsupported quadratic Lagrange family");
      };

      child_cell_gids.reserve(refined_ref.child_connectivity.size());
      for (size_t ci = 0; ci < refined_ref.child_connectivity.size(); ++ci) {
        const CellFamily child_family =
            refined_ref.child_families.empty() ? parent_shape.family : refined_ref.child_families[ci];
        const int child_n_corners = default_num_corners(child_family);
        if (child_n_corners <= 0) {
          throw std::runtime_error("Refine(): unsupported child family for quadratic refinement");
        }

        // Child corner positions in parent reference space.
        const auto& child_corner_local = refined_ref.child_connectivity[ci];
        if (child_corner_local.size() != static_cast<size_t>(child_n_corners)) {
          throw std::runtime_error("Refine(): unexpected child corner count for quadratic refinement");
        }

        std::vector<ParametricPoint> child_corner_parent;
        child_corner_parent.reserve(static_cast<size_t>(child_n_corners));
        for (size_t li : child_corner_local) {
          if (li >= local_parent_xi.size()) {
            throw std::runtime_error("Refine(): invalid child corner index for quadratic refinement");
          }
          child_corner_parent.push_back(local_parent_xi[li]);
        }

        const size_t child_n_nodes = (child_family == parent_shape.family) ? parent_n_nodes
                                                                           : quadratic_lagrange_node_count(child_family);
        const auto child_nodes_xi = quadratic_reference_nodes(child_family, child_n_nodes);

        auto map_to_parent = [&](const ParametricPoint& xi_child) -> ParametricPoint {
          switch (child_family) {
            case CellFamily::Line: {
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              ParametricPoint A0{0.5 * (p1[0] - p0[0]), 0.0, 0.0};
              ParametricPoint b{p0[0] + A0[0], 0.0, 0.0};
              return {b[0] + xi_child[0] * A0[0], 0.0, 0.0};
            }
            case CellFamily::Quad: {
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p3 = child_corner_parent[3];
              const ParametricPoint A0{0.5 * (p1[0] - p0[0]), 0.5 * (p1[1] - p0[1]), 0.0};
              const ParametricPoint A1{0.5 * (p3[0] - p0[0]), 0.5 * (p3[1] - p0[1]), 0.0};
              const ParametricPoint b{p0[0] + A0[0] + A1[0], p0[1] + A0[1] + A1[1], 0.0};
              return {b[0] + xi_child[0] * A0[0] + xi_child[1] * A1[0],
                      b[1] + xi_child[0] * A0[1] + xi_child[1] * A1[1],
                      0.0};
            }
            case CellFamily::Hex: {
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p3 = child_corner_parent[3];
              const auto& p4 = child_corner_parent[4];
              const ParametricPoint A0{0.5 * (p1[0] - p0[0]), 0.5 * (p1[1] - p0[1]), 0.5 * (p1[2] - p0[2])};
              const ParametricPoint A1{0.5 * (p3[0] - p0[0]), 0.5 * (p3[1] - p0[1]), 0.5 * (p3[2] - p0[2])};
              const ParametricPoint A2{0.5 * (p4[0] - p0[0]), 0.5 * (p4[1] - p0[1]), 0.5 * (p4[2] - p0[2])};
              const ParametricPoint b{p0[0] + A0[0] + A1[0] + A2[0],
                                      p0[1] + A0[1] + A1[1] + A2[1],
                                      p0[2] + A0[2] + A1[2] + A2[2]};
              return {b[0] + xi_child[0] * A0[0] + xi_child[1] * A1[0] + xi_child[2] * A2[0],
                      b[1] + xi_child[0] * A0[1] + xi_child[1] * A1[1] + xi_child[2] * A2[1],
                      b[2] + xi_child[0] * A0[2] + xi_child[1] * A1[2] + xi_child[2] * A2[2]};
            }
            case CellFamily::Triangle: {
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p2 = child_corner_parent[2];
              const ParametricPoint e0{p1[0] - p0[0], p1[1] - p0[1], 0.0};
              const ParametricPoint e1{p2[0] - p0[0], p2[1] - p0[1], 0.0};
              return {p0[0] + xi_child[0] * e0[0] + xi_child[1] * e1[0],
                      p0[1] + xi_child[0] * e0[1] + xi_child[1] * e1[1],
                      0.0};
            }
            case CellFamily::Tetra: {
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p2 = child_corner_parent[2];
              const auto& p3 = child_corner_parent[3];
              const ParametricPoint e0{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
              const ParametricPoint e1{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
              const ParametricPoint e2{p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
              return {p0[0] + xi_child[0] * e0[0] + xi_child[1] * e1[0] + xi_child[2] * e2[0],
                      p0[1] + xi_child[0] * e0[1] + xi_child[1] * e1[1] + xi_child[2] * e2[1],
                      p0[2] + xi_child[0] * e0[2] + xi_child[1] * e1[2] + xi_child[2] * e2[2]};
            }
            case CellFamily::Wedge: {
              // Reference wedge coords: (x,y) barycentric on triangle, z in [-1,1].
              // Corners: 0:(0,0,-1) 1:(1,0,-1) 2:(0,1,-1) 3:(0,0,1) ...
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p2 = child_corner_parent[2];
              const auto& p3 = child_corner_parent[3];
              const ParametricPoint e0{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
              const ParametricPoint e1{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
              const ParametricPoint e2{p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
              const real_t t = static_cast<real_t>(0.5) * (xi_child[2] + static_cast<real_t>(1.0));
              return {p0[0] + xi_child[0] * e0[0] + xi_child[1] * e1[0] + t * e2[0],
                      p0[1] + xi_child[0] * e0[1] + xi_child[1] * e1[1] + t * e2[1],
                      p0[2] + xi_child[0] * e0[2] + xi_child[1] * e1[2] + t * e2[2]};
            }
            case CellFamily::Pyramid: {
              // Reference pyramid coords: base quad in z=0 with (x,y) in [-1,1]^2, apex at (0,0,1).
              // For this quadratic-refinement path, pyramid child bases are parallelograms so the mapping is affine.
              const auto& p0 = child_corner_parent[0];
              const auto& p1 = child_corner_parent[1];
              const auto& p3 = child_corner_parent[3];
              const auto& p4 = child_corner_parent[4];
              const ParametricPoint e0{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
              const ParametricPoint e1{p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
              const ParametricPoint e2{p4[0] - p0[0], p4[1] - p0[1], p4[2] - p0[2]};
              const real_t z = xi_child[2];
              const real_t alpha = static_cast<real_t>(0.5) * (xi_child[0] + static_cast<real_t>(1.0) - z);
              const real_t beta = static_cast<real_t>(0.5) * (xi_child[1] + static_cast<real_t>(1.0) - z);
              return {p0[0] + alpha * e0[0] + beta * e1[0] + z * e2[0],
                      p0[1] + alpha * e0[1] + beta * e1[1] + z * e2[1],
                      p0[2] + alpha * e0[2] + beta * e1[2] + z * e2[2]};
            }
            default:
              break;
          }
          throw std::runtime_error("Refine(): unsupported family for quadratic child mapping");
        };

        std::vector<index_t> child_conn;
        child_conn.reserve(child_nodes_xi.size());
        for (const auto& xi_child : child_nodes_xi) {
          const auto xi_parent = map_to_parent(xi_child);
          const auto sf = CurvilinearEvaluator::evaluate_shape_functions(parent_shape, parent_n_nodes, xi_parent);

          std::vector<std::pair<size_t, double>> w;
          w.reserve(sf.N.size());
          for (size_t j = 0; j < sf.N.size(); ++j) {
            const double Nj = static_cast<double>(sf.N[j]);
            if (std::abs(Nj) < 1e-14) continue;
            w.push_back({j, Nj});
          }

          size_t inj_local = 0;
          index_t gv = INVALID_INDEX;
          if (is_single_injection(w, inj_local) && inj_local < parent_verts.size()) {
            gv = parent_verts[inj_local];
          } else {
            std::array<real_t, 3> xyz{0.0, 0.0, 0.0};
            for (const auto& kv : w) {
              const size_t j = kv.first;
              if (j >= parent_verts.size()) continue;
              const double Nj = kv.second;
              const auto pj = mesh.get_vertex_coords(parent_verts[static_cast<size_t>(j)]);
              xyz[0] += static_cast<real_t>(Nj) * pj[0];
              xyz[1] += static_cast<real_t>(Nj) * pj[1];
              xyz[2] += static_cast<real_t>(Nj) * pj[2];
            }
            gv = ensure_new_vertex(w, parent_verts, xyz);
          }
          child_conn.push_back(gv);
        }

        const size_t new_cell_index = new_shapes.size();
        const gid_t child_gid = next_c_gid++;
        child_cell_gids.push_back(child_gid);

        for (index_t v : child_conn) {
          new_conn.push_back(v);
        }
        new_offsets.push_back(static_cast<offset_t>(new_conn.size()));

        CellShape child_shape;
        child_shape.family = child_family;
        child_shape.order = parent_shape.order;
        child_shape.num_corners = child_n_corners;
        new_shapes.push_back(child_shape);
        new_cgids.push_back(child_gid);
        new_levels.push_back(refined_ref.child_level);
        new_regions.push_back(parent_region);

        parent_child.child_to_parent.push_back(c);
        parent_child.parent_to_children[c].push_back(new_cell_index);
        parent_child.parent_patterns[c] = options_.refinement_pattern;
      }
    } else {
      if (!fe_interface_) {
        result.error_messages.push_back("Refine(): p>2 refinement requires an FE interface for high-order embedding");
        result.success = false;
        return result;
      }

      HighOrderEmbeddingKey key;
      key.parent_family = parent_shape.family;
      key.parent_order = static_cast<int>(parent_shape.order);
      key.parent_num_nodes = static_cast<int>(parent_verts.size());
      key.spec = spec;

      const auto& embedding = high_order_embedding_cache_.get_or_request(key, fe_interface_.get());

      child_cell_gids.reserve(embedding.children.size());
      for (size_t ci = 0; ci < embedding.children.size(); ++ci) {
        const auto& child = embedding.children[ci];
        const size_t child_n = static_cast<size_t>(child.child_num_nodes);

        std::vector<index_t> child_conn;
        child_conn.reserve(child_n);
        for (size_t i = 0; i < child_n; ++i) {
          const auto row0 = static_cast<size_t>(child.child_node_parent_weights.row_offsets[i]);
          const auto row1 = static_cast<size_t>(child.child_node_parent_weights.row_offsets[i + 1]);

          std::vector<std::pair<size_t, double>> w;
          w.reserve(row1 - row0);
          for (size_t k = row0; k < row1; ++k) {
            const size_t j = static_cast<size_t>(child.child_node_parent_weights.col_indices[k]);
            const double Nj = child.child_node_parent_weights.values[k];
            if (std::abs(Nj) < 1e-14) continue;
            w.push_back({j, Nj});
          }

          size_t inj_local = 0;
          index_t gv = INVALID_INDEX;
          if (is_single_injection(w, inj_local) && inj_local < parent_verts.size()) {
            gv = parent_verts[inj_local];
          } else {
            std::array<real_t, 3> xyz{0.0, 0.0, 0.0};
            for (const auto& kv : w) {
              const size_t j = kv.first;
              if (j >= parent_verts.size()) continue;
              const double Nj = kv.second;
              const auto pj = mesh.get_vertex_coords(parent_verts[static_cast<size_t>(j)]);
              xyz[0] += static_cast<real_t>(Nj) * pj[0];
              xyz[1] += static_cast<real_t>(Nj) * pj[1];
              xyz[2] += static_cast<real_t>(Nj) * pj[2];
            }
            gv = ensure_new_vertex(w, parent_verts, xyz);
          }
          child_conn.push_back(gv);
        }

        const size_t new_cell_index = new_shapes.size();
        const gid_t child_gid = next_c_gid++;
        child_cell_gids.push_back(child_gid);

        for (index_t v : child_conn) new_conn.push_back(v);
        new_offsets.push_back(static_cast<offset_t>(new_conn.size()));

        CellShape child_shape;
        child_shape.family = child.child_family;
        child_shape.order = child.child_order;
        child_shape.num_corners = default_num_corners(child.child_family);
        if (child_shape.num_corners <= 0) {
          throw std::runtime_error("Refine(): cannot infer child num_corners for FE-embedded family");
        }
        new_shapes.push_back(child_shape);
        new_cgids.push_back(child_gid);
        new_levels.push_back(child_level);
        new_regions.push_back(parent_region);

        parent_child.child_to_parent.push_back(c);
        parent_child.parent_to_children[c].push_back(new_cell_index);
        parent_child.parent_patterns[c] = options_.refinement_pattern;
      }
    }

    // Record provenance for coarsening / tree queries.
    std::vector<gid_t> parent_node_gids;
    parent_node_gids.reserve(parent_verts.size());
    for (index_t v : parent_verts) {
      parent_node_gids.push_back(old_vgids.at(static_cast<size_t>(v)));
    }
    MeshLabels::record_cell_refinement_gid(mesh,
                                          parent_gid,
                                          parent_shape,
                                          parent_node_gids,
                                          parent_region,
                                          parent_level,
                                          child_cell_gids,
                                          static_cast<int>(options_.refinement_pattern));

    delta->refined_cells.push_back(CellRefinementRecord{parent_gid, parent_shape.family, spec, child_cell_gids});
  }

  // Build new mesh and finalize topology.
  new_mesh.build_from_arrays(dim, new_Xref, new_offsets, new_conn, new_shapes);
  new_mesh.set_vertex_gids(std::move(new_vgids));
  new_mesh.set_cell_gids(std::move(new_cgids));
  new_mesh.finalize();

  // Restore per-cell metadata.
  new_mesh.set_cell_refinement_levels(std::move(new_levels));
  for (index_t c = 0; c < static_cast<index_t>(new_regions.size()); ++c) {
    new_mesh.set_region_label(c, new_regions[static_cast<size_t>(c)]);
  }

  // Propagate boundary labels using root-corner provenance.
  if (!boundary_label_by_root.empty() && new_mesh.n_faces() > 0) {
    for (index_t f : new_mesh.boundary_faces()) {
      std::vector<gid_t> root;
      for (index_t v : new_mesh.face_vertices(f)) {
        const gid_t g = new_mesh.vertex_gids().at(static_cast<size_t>(v));
        for (const auto& [rg, w] : MeshLabels::flatten_vertex_provenance_gid(mesh, g)) {
          (void)w;
          root.push_back(rg);
        }
      }
      root = sorted_unique(std::move(root));
      const auto it = boundary_label_by_root.find(root);
      if (it != boundary_label_by_root.end()) {
        new_mesh.set_boundary_label(f, it->second);
      }
    }
  }

  // Transfer fields if requested.
  if (fields && field_transfer_) {
    MeshFields old_fields;
    MeshFields new_fields;
    (void)field_transfer_->transfer(mesh, new_mesh, old_fields, new_fields, parent_child, options_);
  }

  // FE callback with refinement delta (GID-based).
  if (fe_interface_) {
    fe_interface_->on_mesh_adapted(mesh, new_mesh, *delta, options_);
  }

  mesh = std::move(new_mesh);

  result.refinement_delta = std::move(delta);
  result.final_cell_count = mesh.n_cells();
  result.final_vertex_count = mesh.n_vertices();
  result.success = true;
  return result;
}

AdaptivityResult AdaptivityManager::coarsen(MeshBase& mesh,
                                           const std::vector<bool>& marks,
                                           MeshFields* fields) {
  AdaptivityResult result;
  result.initial_cell_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();

  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) {
    result.final_cell_count = 0;
    result.final_vertex_count = mesh.n_vertices();
    result.success = true;
    return result;
  }

  if (!marks.empty() && marks.size() != n_cells) {
    result.error_messages.push_back("Coarsen(): marks size mismatch");
    result.final_cell_count = mesh.n_cells();
    result.final_vertex_count = mesh.n_vertices();
    result.success = false;
    return result;
  }

  if (!MeshLabels::has_refinement_provenance(mesh)) {
    result.warning_messages.push_back("No refinement provenance available; coarsening is a no-op.");
    result.final_cell_count = mesh.n_cells();
    result.final_vertex_count = mesh.n_vertices();
    result.num_coarsened = 0;
    result.success = true;
    return result;
  }

  // Candidate sibling groups keyed by parent GID (stored as index_t for legacy API).
  const auto sibling_groups = MeshLabels::group_siblings_by_parent(mesh);
  std::unordered_map<gid_t, std::vector<index_t>> accepted;

  for (const auto& [parent_gid_as_i, siblings] : sibling_groups) {
    const gid_t parent_gid = static_cast<gid_t>(parent_gid_as_i);
    if (!MeshLabels::is_sibling_group_complete(mesh, static_cast<index_t>(parent_gid_as_i), marks)) {
      continue;
    }

    // Snapshot must exist to restore.
    CellShape parent_shape;
    std::vector<gid_t> parent_node_gids;
    label_t parent_region = 0;
    size_t parent_level = 0;
    if (!MeshLabels::get_parent_cell_snapshot_gid(mesh, parent_gid, parent_shape, parent_node_gids, parent_region, parent_level)) {
      result.warning_messages.push_back("Missing parent snapshot for coarsening parent GID " + std::to_string(parent_gid));
      continue;
    }

    if (options_.check_coarsening_quality) {
      // Evaluate parent quality in current geometry.
      MeshBase tmp(mesh.dim());
      std::unordered_map<gid_t, index_t> local;
      for (gid_t g : parent_node_gids) {
        const index_t ov = mesh.global_to_local_vertex(g);
        if (ov == INVALID_INDEX) continue;
        const index_t nv = static_cast<index_t>(local.size());
        local[g] = nv;
        tmp.add_vertex(nv, mesh.get_vertex_coords(ov));
      }
      std::vector<index_t> conn;
      conn.reserve(parent_node_gids.size());
      for (gid_t g : parent_node_gids) {
        auto it = local.find(g);
        if (it == local.end()) {
          conn.clear();
          break;
        }
        conn.push_back(it->second);
      }
      if (!conn.empty()) {
        tmp.add_cell(0, parent_shape.family, conn);
        tmp.finalize();
        GeometricQualityChecker checker;
        const auto q = checker.compute_element_quality(tmp, 0);
        if (q.overall_quality() < options_.min_coarsening_quality) {
          continue; // reject this parent
        }
      }
    }

    accepted[parent_gid] = siblings;
  }

  result.num_coarsened = accepted.size();
  if (accepted.empty()) {
    result.final_cell_count = mesh.n_cells();
    result.final_vertex_count = mesh.n_vertices();
    result.success = true;
    return result;
  }

  // Build old boundary label lookup (root corners).
  std::unordered_map<std::vector<gid_t>, label_t, GidVecHash> boundary_label_by_root;
  if (mesh.n_faces() > 0) {
    for (index_t f : mesh.boundary_faces()) {
      const label_t lbl = mesh.boundary_label(f);
      if (lbl == INVALID_LABEL) continue;
      std::vector<gid_t> gids;
      for (index_t v : mesh.face_vertices(f)) {
        gids.push_back(mesh.vertex_gids().at(static_cast<size_t>(v)));
      }
      boundary_label_by_root[sorted_unique(std::move(gids))] = lbl;
    }
  }

  // Mark cells to remove (accepted children).
  std::vector<bool> remove_cell(n_cells, false);
  for (const auto& [pg, sib] : accepted) {
    for (index_t c : sib) {
      if (c >= 0 && static_cast<size_t>(c) < remove_cell.size()) {
        remove_cell[static_cast<size_t>(c)] = true;
      }
    }
  }

  // Collect restored parents and kept cells.
  struct RestoredParent {
    gid_t gid = INVALID_GID;
    CellShape shape{};
    std::vector<gid_t> node_gids;
    label_t region = 0;
    size_t level = 0;
    std::vector<index_t> old_children;
  };

  std::vector<RestoredParent> parents;
  parents.reserve(accepted.size());
  for (const auto& [pg, children] : accepted) {
    RestoredParent rp;
    rp.gid = pg;
    rp.old_children = children;
    if (!MeshLabels::get_parent_cell_snapshot_gid(mesh, pg, rp.shape, rp.node_gids, rp.region, rp.level)) {
      continue;
    }
    parents.push_back(std::move(rp));
  }

  // Determine required vertices (by GID).
  std::unordered_map<gid_t, index_t> new_vidx_by_gid;
  std::vector<real_t> new_Xref;
  std::vector<gid_t> new_vgids;

  const int dim = (mesh.dim() > 0) ? mesh.dim() : 3;

  auto require_gid = [&](gid_t g) {
    if (g == INVALID_GID) return;
    if (new_vidx_by_gid.find(g) != new_vidx_by_gid.end()) return;
    const index_t ov = mesh.global_to_local_vertex(g);
    if (ov == INVALID_INDEX) return;
    const index_t nv = static_cast<index_t>(new_vgids.size());
    new_vidx_by_gid[g] = nv;
    new_vgids.push_back(g);
    const auto xyz = mesh.get_vertex_coords(ov);
    for (int d = 0; d < dim; ++d) new_Xref.push_back(xyz[static_cast<size_t>(d)]);
  };

  // Preserve vertex order by scanning old vertex GIDs.
  std::unordered_map<gid_t, bool> needed;
  for (size_t c = 0; c < n_cells; ++c) {
    if (remove_cell[c]) continue;
    for (index_t v : mesh.cell_vertices(static_cast<index_t>(c))) {
      needed[mesh.vertex_gids().at(static_cast<size_t>(v))] = true;
    }
  }
  for (const auto& rp : parents) {
    for (gid_t g : rp.node_gids) needed[g] = true;
  }
  for (gid_t g : mesh.vertex_gids()) {
    if (needed.find(g) != needed.end()) require_gid(g);
  }

  // Build new cells.
  std::vector<offset_t> new_offsets;
  new_offsets.push_back(0);
  std::vector<index_t> new_conn;
  std::vector<CellShape> new_shapes;
  std::vector<gid_t> new_cgids;
  std::vector<size_t> new_levels;
  std::vector<label_t> new_regions;

  ParentChildMap parent_child;
  parent_child.child_to_parent.reserve(n_cells);

  auto map_old_vertex_index = [&](index_t ov) -> index_t {
    const gid_t g = mesh.vertex_gids().at(static_cast<size_t>(ov));
    const auto it = new_vidx_by_gid.find(g);
    if (it == new_vidx_by_gid.end()) return INVALID_INDEX;
    return it->second;
  };

  // Keep existing (non-removed) cells.
  for (size_t c = 0; c < n_cells; ++c) {
    if (remove_cell[c]) continue;
    const index_t cell = static_cast<index_t>(c);
    const auto conn_old = mesh.cell_vertices(cell);
    for (index_t ov : conn_old) {
      const index_t nv = map_old_vertex_index(ov);
      if (nv == INVALID_INDEX) throw std::runtime_error("Coarsen(): missing vertex mapping");
      new_conn.push_back(nv);
    }
    new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
    new_shapes.push_back(mesh.cell_shape(cell));
    new_cgids.push_back(mesh.cell_gids().at(c));
    new_levels.push_back(mesh.refinement_level(cell));
    new_regions.push_back(mesh.region_label(cell));
    parent_child.child_to_parent.push_back(c);
  }

  // Add restored parents.
  std::unordered_map<gid_t, size_t> new_cell_index_by_parent_gid;
  for (const auto& rp : parents) {
    const size_t new_cell_index = new_shapes.size();
    new_cell_index_by_parent_gid[rp.gid] = new_cell_index;

    for (gid_t g : rp.node_gids) {
      const auto it = new_vidx_by_gid.find(g);
      if (it == new_vidx_by_gid.end()) throw std::runtime_error("Coarsen(): missing parent vertex gid");
      new_conn.push_back(it->second);
    }
    new_offsets.push_back(static_cast<offset_t>(new_conn.size()));
    new_shapes.push_back(rp.shape);
    new_cgids.push_back(rp.gid);
    new_levels.push_back(rp.level);
    new_regions.push_back(rp.region);
    parent_child.child_to_parent.push_back(static_cast<size_t>(rp.gid));
  }

  MeshBase new_mesh(dim);
  new_mesh.build_from_arrays(dim, new_Xref, new_offsets, new_conn, new_shapes);
  new_mesh.set_vertex_gids(std::move(new_vgids));
  new_mesh.set_cell_gids(std::move(new_cgids));
  new_mesh.finalize();
  new_mesh.set_cell_refinement_levels(std::move(new_levels));
  for (index_t c = 0; c < static_cast<index_t>(new_regions.size()); ++c) {
    new_mesh.set_region_label(c, new_regions[static_cast<size_t>(c)]);
  }

  // Boundary label propagation for the coarsened mesh.
  if (!boundary_label_by_root.empty() && new_mesh.n_faces() > 0) {
    for (index_t f : new_mesh.boundary_faces()) {
      std::vector<gid_t> root;
      for (index_t v : new_mesh.face_vertices(f)) {
        const gid_t g = new_mesh.vertex_gids().at(static_cast<size_t>(v));
        for (const auto& [rg, w] : MeshLabels::flatten_vertex_provenance_gid(mesh, g)) {
          (void)w;
          root.push_back(rg);
        }
      }
      root = sorted_unique(std::move(root));
      const auto it = boundary_label_by_root.find(root);
      if (it != boundary_label_by_root.end()) {
        new_mesh.set_boundary_label(f, it->second);
      }
    }
  }

  // Build parent->children maps for field restriction (best-effort).
  for (const auto& rp : parents) {
    const auto itc = new_cell_index_by_parent_gid.find(rp.gid);
    if (itc == new_cell_index_by_parent_gid.end()) continue;
    parent_child.parent_to_children[itc->second] = {};
    for (index_t oc : rp.old_children) {
      if (oc >= 0) parent_child.parent_to_children[itc->second].push_back(static_cast<size_t>(oc));
    }
  }
  // Vertex restriction map: inject matching GIDs.
  for (size_t v = 0; v < new_mesh.n_vertices(); ++v) {
    const gid_t g = new_mesh.vertex_gids().at(v);
    const index_t ov = mesh.global_to_local_vertex(g);
    if (ov != INVALID_INDEX) {
      parent_child.parent_vertex_to_children[v].push_back(static_cast<size_t>(ov));
    }
  }

  if (fields && field_transfer_) {
    MeshFields old_fields;
    MeshFields new_fields;
    (void)field_transfer_->transfer(mesh, new_mesh, old_fields, new_fields, parent_child, options_);
  }

  mesh = std::move(new_mesh);

  result.final_cell_count = mesh.n_cells();
  result.final_vertex_count = mesh.n_vertices();
  result.success = true;
  return result;
}

void AdaptivityManager::set_options(const AdaptivityOptions& options) {
  options_ = options;
}

void AdaptivityManager::set_error_estimator(std::unique_ptr<ErrorEstimator> estimator) {
  error_estimator_ = std::move(estimator);
}

void AdaptivityManager::set_marker(std::unique_ptr<Marker> marker) {
  marker_ = std::move(marker);
}

void AdaptivityManager::set_field_transfer(std::unique_ptr<FieldTransfer> transfer) {
  field_transfer_ = std::move(transfer);
}

void AdaptivityManager::set_quality_checker(std::unique_ptr<QualityChecker> checker) {
  quality_checker_ = std::move(checker);
}

void AdaptivityManager::set_conformity_enforcer(std::unique_ptr<ConformityEnforcer> enforcer) {
  conformity_enforcer_ = std::move(enforcer);
}

void AdaptivityManager::set_fe_interface(std::shared_ptr<AdaptivityFEInterface> fe) {
  fe_interface_ = std::move(fe);
  high_order_embedding_cache_.clear();
}

bool AdaptivityManager::needs_adaptation(const MeshBase& mesh, const MeshFields* fields) const {
  (void)fields;
  if (!error_estimator_) {
    return false;
  }
  const auto indicators = error_estimator_->estimate(mesh, fields, options_);
  if (indicators.empty()) {
    return false;
  }
  const double max_val = *std::max_element(indicators.begin(), indicators.end());
  if (options_.enable_refinement && max_val >= options_.refine_threshold) {
    return true;
  }
  return false;
}

AdaptivityResult AdaptivityManager::estimate_adaptation(const MeshBase& mesh, const MeshFields* fields) const {
  AdaptivityResult result;
  result.initial_cell_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();
  if (error_estimator_) {
    const auto indicators = error_estimator_->estimate(mesh, fields, options_);
    result.initial_error = ErrorEstimatorUtils::compute_global_error(indicators, options_.error_norm_power);
    result.final_error = result.initial_error;
  }
  result.final_cell_count = result.initial_cell_count;
  result.final_vertex_count = result.initial_vertex_count;
  result.success = true;
  result.warning_messages.push_back("Estimate-only: refinement/coarsening not yet implemented.");
  return result;
}

std::vector<double> AdaptivityManager::estimate_error(const MeshBase& mesh, const MeshFields* fields) {
  if (!error_estimator_) {
    return std::vector<double>(mesh.n_cells(), 0.0);
  }
  return error_estimator_->estimate(mesh, fields, options_);
}

std::vector<MarkType> AdaptivityManager::mark_elements(const std::vector<double>& indicators,
                                                       const MeshBase& mesh) {
  if (!marker_) {
    return std::vector<MarkType>(indicators.size(), MarkType::NONE);
  }
  auto marks = marker_->mark(indicators, mesh, options_);
  MarkerUtils::apply_constraints(marks, mesh, options_);
  return marks;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_refinement(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {
  (void)mesh;
  (void)marks;
  (void)result;
  return nullptr;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_coarsening(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {
  (void)mesh;
  (void)marks;
  (void)result;
  return nullptr;
}

void AdaptivityManager::enforce_conformity(MeshBase& mesh,
                                          AdaptivityResult& result) {
  (void)mesh;
  (void)result;
}

void AdaptivityManager::transfer_fields(const MeshBase& old_mesh,
                                       MeshBase& new_mesh,
                                       MeshFields* old_fields,
                                       MeshFields* new_fields,
                                       AdaptivityResult& result) {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)result;
}

bool AdaptivityManager::check_quality(const MeshBase& mesh, AdaptivityResult& result) {
  (void)mesh;
  (void)result;
  return true;
}

void AdaptivityManager::finalize_mesh(MeshBase& mesh, AdaptivityResult& result) {
  (void)mesh;
  (void)result;
}

void AdaptivityManager::emit_events(MeshBase& mesh) const {
  (void)mesh;
}

void AdaptivityManager::write_intermediate_mesh(const MeshBase& mesh, const std::string& stage_name) const {
  (void)mesh;
  (void)stage_name;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_options(const AdaptivityOptions& options) {
  options_ = options;
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_error_estimator(std::unique_ptr<ErrorEstimator> estimator) {
  error_estimator_ = std::move(estimator);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_marker(std::unique_ptr<Marker> marker) {
  marker_ = std::move(marker);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_field_transfer(std::unique_ptr<FieldTransfer> transfer) {
  field_transfer_ = std::move(transfer);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_quality_checker(std::unique_ptr<QualityChecker> checker) {
  quality_checker_ = std::move(checker);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_conformity_enforcer(std::unique_ptr<ConformityEnforcer> enforcer) {
  conformity_enforcer_ = std::move(enforcer);
  return *this;
}

std::unique_ptr<AdaptivityManager> AdaptivityManagerBuilder::build() {
  auto manager = std::make_unique<AdaptivityManager>(options_);
  if (error_estimator_) manager->set_error_estimator(std::move(error_estimator_));
  if (marker_) manager->set_marker(std::move(marker_));
  if (field_transfer_) manager->set_field_transfer(std::move(field_transfer_));
  if (quality_checker_) manager->set_quality_checker(std::move(quality_checker_));
  if (conformity_enforcer_) manager->set_conformity_enforcer(std::move(conformity_enforcer_));
  return manager;
}

AdaptivityResult AdaptivityUtils::uniform_refinement(MeshBase& mesh, size_t num_levels, MeshFields* fields) {
  AdaptivityOptions opts;
  opts.enable_refinement = true;
  opts.enable_coarsening = false;
  opts.check_quality = false;
  opts.verbosity = 0;
  AdaptivityManager manager(opts);

  AdaptivityResult last;
  for (size_t level = 0; level < num_levels; ++level) {
    std::vector<bool> marks(mesh.n_cells(), true);
    last = manager.refine(mesh, marks, fields);
    if (!last.success) return last;
    if (last.num_refined == 0) break;
  }
  return last;
}

AdaptivityResult AdaptivityUtils::uniform_coarsening(MeshBase& mesh, size_t num_levels, MeshFields* fields) {
  AdaptivityOptions opts;
  opts.enable_refinement = false;
  opts.enable_coarsening = true;
  opts.check_quality = false;
  opts.verbosity = 0;
  AdaptivityManager manager(opts);

  AdaptivityResult last;
  for (size_t level = 0; level < num_levels; ++level) {
    std::vector<bool> marks(mesh.n_cells(), true);
    last = manager.coarsen(mesh, marks, fields);
    if (!last.success) return last;
    if (last.num_coarsened == 0) break;
  }
  return last;
}

AdaptivityResult AdaptivityUtils::local_refinement(
    MeshBase& mesh,
    const std::function<bool(const std::array<double, 3>&)>& region_predicate,
    size_t num_levels,
    MeshFields* fields) {
  AdaptivityOptions opts;
  opts.enable_refinement = true;
  opts.enable_coarsening = false;
  opts.check_quality = false;
  opts.verbosity = 0;
  AdaptivityManager manager(opts);

  AdaptivityResult last;
  for (size_t level = 0; level < num_levels; ++level) {
    const size_t n_cells = mesh.n_cells();
    std::vector<bool> marks(n_cells, false);
    for (size_t c = 0; c < n_cells; ++c) {
      const auto x = mesh.cell_centroid(static_cast<index_t>(c), Configuration::Reference);
      const std::array<double, 3> xd = {static_cast<double>(x[0]),
                                        static_cast<double>(x[1]),
                                        static_cast<double>(x[2])};
      if (region_predicate(xd)) {
        marks[c] = true;
      }
    }

    last = manager.refine(mesh, marks, fields);
    if (!last.success) return last;
    if (last.num_refined == 0) break;
  }
  return last;
}

bool AdaptivityUtils::is_adapted(const MeshBase& mesh) {
  if (MeshLabels::has_refinement_provenance(mesh)) {
    return true;
  }

  const size_t n_cells = mesh.n_cells();
  for (size_t c = 0; c < n_cells; ++c) {
    if (mesh.refinement_level(static_cast<index_t>(c)) > 0) {
      return true;
    }
  }
  return false;
}

AdaptivityUtils::LevelStats AdaptivityUtils::get_level_stats(const MeshBase& mesh) {
  LevelStats stats;
  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) {
    stats.min_level = 0;
    stats.max_level = 0;
    stats.avg_level = 0.0;
    stats.cell_count_per_level = {0};
    stats.element_count_per_level = stats.cell_count_per_level;
    return stats;
  }

  size_t min_level = std::numeric_limits<size_t>::max();
  size_t max_level = 0;
  double sum = 0.0;

  std::vector<size_t> levels;
  levels.reserve(n_cells);
  for (size_t c = 0; c < n_cells; ++c) {
    const size_t lvl = mesh.refinement_level(static_cast<index_t>(c));
    levels.push_back(lvl);
    min_level = std::min(min_level, lvl);
    max_level = std::max(max_level, lvl);
    sum += static_cast<double>(lvl);
  }

  stats.min_level = min_level;
  stats.max_level = max_level;
  stats.avg_level = sum / static_cast<double>(n_cells);

  stats.cell_count_per_level.assign(max_level + 1, 0);
  for (const auto lvl : levels) {
    stats.cell_count_per_level[lvl]++;
  }

  stats.element_count_per_level = stats.cell_count_per_level;
  return stats;
}

void AdaptivityUtils::write_metrics_to_fields(
    MeshBase& mesh,
    const std::vector<double>& error_indicators,
    const std::vector<MarkType>& marks) {
  ErrorEstimatorUtils::write_to_field(mesh, "error_indicator", error_indicators);
  MarkerUtils::write_marks_to_field(mesh, "refinement_marks", marks);
}

} // namespace svmp
