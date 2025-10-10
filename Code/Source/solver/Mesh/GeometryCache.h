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

#ifndef SVMP_GEOMETRY_CACHE_H
#define SVMP_GEOMETRY_CACHE_H

#include "Mesh.h"
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {

// ====================
// P0 #4: Quadrature & Geometry Cache Service
// ====================
// Reusable cache of quadrature rules and precomputed geometry
// (centers, measures, reference→physical Jacobians at quadrature points).
// Keyed by (cell_shape, geom_order, quadrature_key, configuration).
//
// Why: All physics integrate; the same per-cell data gets recomputed many times otherwise.
// This cache avoids redundant computation and provides consistent quadrature across physics.

// ====================
// Quadrature Rule
// ====================
struct QuadratureRule {
  std::vector<std::array<real_t,3>> points; // ξ coordinates in reference element
  std::vector<real_t> weights;               // integration weights

  size_t n_points() const { return points.size(); }

  // Factory methods for standard rules
  static QuadratureRule gauss_legendre_1d(int order);
  static QuadratureRule gauss_legendre_2d_quad(int order);
  static QuadratureRule gauss_legendre_3d_hex(int order);
  static QuadratureRule triangle(int order);
  static QuadratureRule tetrahedron(int order);
};

// ====================
// Cell Geometry Data
// ====================
// Precomputed geometric quantities at quadrature points for a single cell
struct CellGeometryData {
  Configuration config = Configuration::Reference;  // which configuration this was built for
  size_t n_qpts = 0;                                // number of quadrature points

  std::vector<std::array<real_t,3>> x_qpts;        // physical coordinates at quadrature points
  std::vector<std::array<std::array<real_t,3>,3>> J_qpts;  // Jacobians dx/dξ at quadrature points
  std::vector<real_t> detJ_qpts;                    // Jacobian determinants at quadrature points
  std::vector<std::array<std::array<real_t,3>,3>> invJ_qpts; // Inverse Jacobians at quadrature points

  real_t cell_measure = 0.0;                        // ∫ detJ dξ (volume/area/length)
  std::array<real_t,3> cell_center = {{0,0,0}};    // centroid

  // For face integration
  struct FaceGeometry {
    std::vector<std::array<real_t,3>> x_qpts;      // physical coordinates on face
    std::vector<std::array<real_t,3>> normals;     // outward normals at quadrature points
    std::vector<real_t> detJ_qpts;                 // surface Jacobian determinants
    real_t area = 0.0;                              // ∫ detJ_face dξ
  };
  std::vector<FaceGeometry> face_geom;              // one per face
};

// ====================
// Geometry Cache
// ====================
// Owns and manages precomputed geometry data for all cells in a mesh.
// Invalidated when mesh geometry changes (via MeshObserver pattern).

class GeometryCache {
public:
  explicit GeometryCache(const MeshBase& mesh) : mesh_(mesh) {}

  // ---- Quadrature rule access
  // Get quadrature rule for a given cell shape and integration order
  const QuadratureRule& quadrature(CellFamily family, int order) const;

  // ---- Cell geometry access
  // Get precomputed geometry for a cell at specified quadrature order
  const CellGeometryData& cell_geometry(
      index_t cell_id,
      int quad_order,
      Configuration cfg = Configuration::Reference);

  // Check if geometry is cached
  bool has_cell_geometry(index_t cell_id, int quad_order, Configuration cfg) const;

  // ---- Cache management
  // Precompute geometry for all cells at given quadrature order
  void build_all(int quad_order, Configuration cfg = Configuration::Reference);

  // Invalidate cache (call when mesh geometry changes)
  void invalidate();

  // Invalidate only current-configuration data (call when coordinates update but topology unchanged)
  void invalidate_current();

  // ---- Memory management
  size_t memory_usage_bytes() const;
  void clear();

private:
  const MeshBase& mesh_;

  // Quadrature rule registry
  mutable std::unordered_map<std::string, QuadratureRule> quad_rules_;

  // Geometry cache: key = (cell_id, quad_order, config)
  struct CacheKey {
    index_t cell_id;
    int quad_order;
    Configuration config;

    bool operator==(const CacheKey& other) const {
      return cell_id == other.cell_id && quad_order == other.quad_order && config == other.config;
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const {
      return std::hash<index_t>()(k.cell_id) ^
             (std::hash<int>()(k.quad_order) << 1) ^
             (std::hash<int>()(static_cast<int>(k.config)) << 2);
    }
  };

  mutable std::unordered_map<CacheKey, CellGeometryData, CacheKeyHash> cell_geom_cache_;

  // Helpers
  std::string quad_key(CellFamily family, int order) const;
  CellGeometryData compute_cell_geometry(index_t cell_id, const QuadratureRule& rule, Configuration cfg) const;
};

// ====================
// QuadratureRule factory implementations (minimal)
// ====================
inline QuadratureRule QuadratureRule::gauss_legendre_1d(int order) {
  QuadratureRule rule;
  // Simplified: only implement order 1 and 2 for demonstration
  if (order == 1) {
    rule.points = {{{0.0, 0.0, 0.0}}};
    rule.weights = {2.0};
  } else if (order == 2) {
    const real_t a = 1.0 / std::sqrt(3.0);
    rule.points = {{{-a, 0.0, 0.0}}, {{a, 0.0, 0.0}}};
    rule.weights = {1.0, 1.0};
  } else {
    throw std::runtime_error("QuadratureRule: Gauss-Legendre 1D order " + std::to_string(order) + " not implemented");
  }
  return rule;
}

inline QuadratureRule QuadratureRule::gauss_legendre_2d_quad(int order) {
  QuadratureRule rule;
  // Tensor product of 1D rules
  auto rule_1d = gauss_legendre_1d(order);
  for (size_t i = 0; i < rule_1d.n_points(); ++i) {
    for (size_t j = 0; j < rule_1d.n_points(); ++j) {
      rule.points.push_back({{rule_1d.points[i][0], rule_1d.points[j][0], 0.0}});
      rule.weights.push_back(rule_1d.weights[i] * rule_1d.weights[j]);
    }
  }
  return rule;
}

inline QuadratureRule QuadratureRule::gauss_legendre_3d_hex(int order) {
  QuadratureRule rule;
  // Tensor product of 1D rules
  auto rule_1d = gauss_legendre_1d(order);
  for (size_t i = 0; i < rule_1d.n_points(); ++i) {
    for (size_t j = 0; j < rule_1d.n_points(); ++j) {
      for (size_t k = 0; k < rule_1d.n_points(); ++k) {
        rule.points.push_back({{rule_1d.points[i][0], rule_1d.points[j][0], rule_1d.points[k][0]}});
        rule.weights.push_back(rule_1d.weights[i] * rule_1d.weights[j] * rule_1d.weights[k]);
      }
    }
  }
  return rule;
}

inline QuadratureRule QuadratureRule::triangle(int order) {
  QuadratureRule rule;
  // Simplified: order 1 (centroid)
  if (order == 1) {
    rule.points = {{{1.0/3.0, 1.0/3.0, 0.0}}};
    rule.weights = {0.5}; // area of reference triangle
  } else if (order == 2) {
    // 3-point rule (vertices)
    rule.points = {{{0.5, 0.5, 0.0}}, {{0.0, 0.5, 0.0}}, {{0.5, 0.0, 0.0}}};
    rule.weights = {1.0/6.0, 1.0/6.0, 1.0/6.0};
  } else {
    throw std::runtime_error("QuadratureRule: Triangle order " + std::to_string(order) + " not implemented");
  }
  return rule;
}

inline QuadratureRule QuadratureRule::tetrahedron(int order) {
  QuadratureRule rule;
  // Simplified: order 1 (centroid)
  if (order == 1) {
    rule.points = {{{0.25, 0.25, 0.25}}};
    rule.weights = {1.0/6.0}; // volume of reference tet
  } else {
    throw std::runtime_error("QuadratureRule: Tetrahedron order " + std::to_string(order) + " not implemented");
  }
  return rule;
}

// ====================
// GeometryCache implementation
// ====================
inline std::string GeometryCache::quad_key(CellFamily family, int order) const {
  return std::to_string(static_cast<int>(family)) + "_" + std::to_string(order);
}

inline const QuadratureRule& GeometryCache::quadrature(CellFamily family, int order) const {
  std::string key = quad_key(family, order);
  auto it = quad_rules_.find(key);
  if (it != quad_rules_.end()) {
    return it->second;
  }

  // Build rule on demand
  QuadratureRule rule;
  switch (family) {
    case CellFamily::Line:
      rule = QuadratureRule::gauss_legendre_1d(order);
      break;
    case CellFamily::Triangle:
      rule = QuadratureRule::triangle(order);
      break;
    case CellFamily::Quad:
      rule = QuadratureRule::gauss_legendre_2d_quad(order);
      break;
    case CellFamily::Tetra:
      rule = QuadratureRule::tetrahedron(order);
      break;
    case CellFamily::Hex:
      rule = QuadratureRule::gauss_legendre_3d_hex(order);
      break;
    default:
      throw std::runtime_error("GeometryCache: quadrature not implemented for cell family");
  }

  quad_rules_[key] = rule;
  return quad_rules_[key];
}

inline const CellGeometryData& GeometryCache::cell_geometry(
    index_t cell_id,
    int quad_order,
    Configuration cfg)
{
  CacheKey key{cell_id, quad_order, cfg};
  auto it = cell_geom_cache_.find(key);
  if (it != cell_geom_cache_.end()) {
    return it->second;
  }

  // Compute geometry on demand
  const auto& cell_shape = mesh_.cell_shape(cell_id);
  const auto& rule = quadrature(cell_shape.family, quad_order);
  auto geom = compute_cell_geometry(cell_id, rule, cfg);
  cell_geom_cache_[key] = geom;
  return cell_geom_cache_[key];
}

inline bool GeometryCache::has_cell_geometry(index_t cell_id, int quad_order, Configuration cfg) const {
  CacheKey key{cell_id, quad_order, cfg};
  return cell_geom_cache_.find(key) != cell_geom_cache_.end();
}

inline void GeometryCache::build_all(int quad_order, Configuration cfg) {
  for (size_t c = 0; c < mesh_.n_cells(); ++c) {
    cell_geometry(static_cast<index_t>(c), quad_order, cfg);
  }
}

inline void GeometryCache::invalidate() {
  cell_geom_cache_.clear();
}

inline void GeometryCache::invalidate_current() {
  // Remove only Current configuration entries
  for (auto it = cell_geom_cache_.begin(); it != cell_geom_cache_.end(); ) {
    if (it->first.config == Configuration::Current) {
      it = cell_geom_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

inline size_t GeometryCache::memory_usage_bytes() const {
  size_t total = 0;
  for (const auto& [key, geom] : cell_geom_cache_) {
    total += sizeof(CellGeometryData);
    total += geom.x_qpts.size() * sizeof(std::array<real_t,3>);
    total += geom.J_qpts.size() * sizeof(std::array<std::array<real_t,3>,3>);
    total += geom.detJ_qpts.size() * sizeof(real_t);
    total += geom.invJ_qpts.size() * sizeof(std::array<std::array<real_t,3>,3>);
  }
  return total;
}

inline void GeometryCache::clear() {
  cell_geom_cache_.clear();
  quad_rules_.clear();
}

inline CellGeometryData GeometryCache::compute_cell_geometry(
    index_t cell_id,
    const QuadratureRule& rule,
    Configuration cfg) const
{
  CellGeometryData geom;
  geom.config = cfg;
  geom.n_qpts = rule.n_points();
  geom.x_qpts.resize(geom.n_qpts);
  geom.J_qpts.resize(geom.n_qpts);
  geom.detJ_qpts.resize(geom.n_qpts);
  geom.invJ_qpts.resize(geom.n_qpts);

  // Compute geometry at each quadrature point
  real_t total_weight = 0.0;
  for (size_t q = 0; q < geom.n_qpts; ++q) {
    const auto& xi = rule.points[q];
    const real_t w = rule.weights[q];

    // Evaluate mapping and Jacobian
    geom.x_qpts[q] = mesh_.evaluate_map(cell_id, xi, cfg);
    geom.J_qpts[q] = mesh_.jacobian(cell_id, xi, cfg);
    geom.detJ_qpts[q] = mesh_.detJ(cell_id, xi, cfg);
    geom.invJ_qpts[q] = mesh_.invJ(cell_id, xi, cfg);

    // Accumulate measure and centroid
    geom.cell_measure += w * geom.detJ_qpts[q];
    for (int d = 0; d < 3; ++d) {
      geom.cell_center[d] += w * geom.detJ_qpts[q] * geom.x_qpts[q][d];
    }
    total_weight += w * geom.detJ_qpts[q];
  }

  // Normalize centroid
  if (total_weight > 1e-14) {
    for (int d = 0; d < 3; ++d) {
      geom.cell_center[d] /= total_weight;
    }
  }

  return geom;
}

} // namespace svmp

#endif // SVMP_GEOMETRY_CACHE_H
