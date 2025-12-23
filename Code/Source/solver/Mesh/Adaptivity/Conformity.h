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

#ifndef SVMP_CONFORMITY_H
#define SVMP_CONFORMITY_H

#include "Options.h"
#include "Marker.h"
#include "RefinementRules.h"
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;

/**
 * @brief Hanging node information
 */
struct HangingNode {
  /** Node index */
  size_t node_id;

  /** Parent edge or face that contains this node */
  std::pair<size_t, size_t> parent_entity;

  /** Is this on an edge (true) or face (false) */
  bool on_edge;

  /** Constraint coefficients (node_id -> weight) */
  std::map<size_t, double> constraints;

  /** Refinement level difference */
  size_t level_difference;
};

/**
 * @brief Non-conformity information for mesh
 */
struct NonConformity {
  /** List of hanging nodes */
  std::vector<HangingNode> hanging_nodes;

  /** Elements that need closure refinement */
  std::set<size_t> elements_needing_closure;

  /** Edges with hanging nodes */
  std::set<std::pair<size_t, size_t>> non_conforming_edges;

  /** Faces with hanging nodes */
  std::set<std::vector<size_t>> non_conforming_faces;

  /** Maximum level difference in mesh */
  size_t max_level_difference;

  /** Is mesh conforming */
  bool is_conforming() const {
    return hanging_nodes.empty() && elements_needing_closure.empty();
  }
};

/**
 * @brief Abstract base class for conformity enforcement strategies
 */
class ConformityEnforcer {
public:
  virtual ~ConformityEnforcer() = default;

  /**
   * @brief Check mesh conformity
   *
   * @param mesh The mesh to check
   * @param marks Current refinement marks
   * @return Non-conformity information
   */
  virtual NonConformity check_conformity(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks) const = 0;

  /**
   * @brief Enforce conformity by modifying marks
   *
   * @param mesh The mesh
   * @param marks Refinement marks (modified in place)
   * @param options Adaptivity options
   * @return Number of closure iterations performed
   */
  virtual size_t enforce_conformity(
      const MeshBase& mesh,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options) const = 0;

  /**
   * @brief Generate hanging node constraints
   *
   * @param mesh The mesh
   * @param non_conformity Non-conformity information
   * @return Map of constrained nodes to their constraint equations
   */
  virtual std::map<size_t, std::map<size_t, double>> generate_constraints(
      const MeshBase& mesh,
      const NonConformity& non_conformity) const = 0;

  /**
   * @brief Get enforcer name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Standard conformity enforcer using closure refinement
 *
 * Ensures mesh conformity by refining additional elements to eliminate
 * hanging nodes.
 */
class ClosureConformityEnforcer : public ConformityEnforcer {
public:
  /**
   * @brief Configuration for closure enforcement
   */
  struct Config {
    /** Maximum closure iterations */
    size_t max_iterations = 10;

    /** Maximum allowed level difference */
    size_t max_level_difference = 1;

    /** Use green refinement for closure when possible */
    bool use_green_closure = true;

    /** Propagate closure to neighbors */
    bool propagate_closure = true;

    /** Check face conformity (3D) */
    bool check_face_conformity = true;

    /** Check edge conformity */
    bool check_edge_conformity = true;
  };

  ClosureConformityEnforcer() : ClosureConformityEnforcer(Config{}) {}
  explicit ClosureConformityEnforcer(const Config& config);

  NonConformity check_conformity(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks) const override;

  size_t enforce_conformity(
      const MeshBase& mesh,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options) const override;

  std::map<size_t, std::map<size_t, double>> generate_constraints(
      const MeshBase& mesh,
      const NonConformity& non_conformity) const override;

  std::string name() const override { return "ClosureConformity"; }

private:
  Config config_;

  /** Check if edge is conforming */
  bool is_edge_conforming(
      const MeshBase& mesh,
      size_t v1, size_t v2,
      const std::vector<MarkType>& marks) const;

  /** Check if face is conforming */
  bool is_face_conforming(
      const MeshBase& mesh,
      const std::vector<size_t>& face_vertices,
      const std::vector<MarkType>& marks) const;

  /** Find elements that share an edge */
  std::vector<size_t> find_edge_elements(
      const MeshBase& mesh,
      size_t v1, size_t v2) const;

  /** Find elements that share a face */
  std::vector<size_t> find_face_elements(
      const MeshBase& mesh,
      const std::vector<size_t>& face_vertices) const;

  /** Mark element for closure refinement */
  void mark_for_closure(
      std::vector<MarkType>& marks,
      size_t elem_id) const;
};

/**
 * @brief Hanging node conformity enforcer
 *
 * Allows hanging nodes but generates constraint equations to maintain
 * solution continuity.
 */
class HangingNodeConformityEnforcer : public ConformityEnforcer {
public:
  /**
   * @brief Configuration for hanging node enforcement
   */
  struct Config {
    /** Maximum allowed hanging level difference */
    size_t max_hanging_level = 1;

    /** Generate constraints for vertex nodes */
    bool constrain_vertices = true;

    /** Generate constraints for edge nodes (high-order) */
    bool constrain_edges = false;

    /** Generate constraints for face nodes (high-order) */
    bool constrain_faces = false;

    /** Constraint tolerance */
    double constraint_tolerance = 1e-10;
  };

  HangingNodeConformityEnforcer() : HangingNodeConformityEnforcer(Config{}) {}
  explicit HangingNodeConformityEnforcer(const Config& config);

  NonConformity check_conformity(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks) const override;

  size_t enforce_conformity(
      const MeshBase& mesh,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options) const override;

  std::map<size_t, std::map<size_t, double>> generate_constraints(
      const MeshBase& mesh,
      const NonConformity& non_conformity) const override;

  std::string name() const override { return "HangingNode"; }

private:
  Config config_;

  /** Find hanging nodes on edge */
  std::vector<HangingNode> find_edge_hanging_nodes(
      const MeshBase& mesh,
      size_t v1, size_t v2,
      const std::vector<MarkType>& marks) const;

  /** Find hanging nodes on face */
  std::vector<HangingNode> find_face_hanging_nodes(
      const MeshBase& mesh,
      const std::vector<size_t>& face_vertices,
      const std::vector<MarkType>& marks) const;

  /** Generate constraint equation for hanging node */
  std::map<size_t, double> generate_node_constraint(
      const MeshBase& mesh,
      const HangingNode& node) const;
};

/**
 * @brief Minimal closure conformity enforcer
 *
 * Uses minimal refinement to achieve conformity, preferring green
 * refinement patterns where possible.
 */
class MinimalClosureEnforcer : public ConformityEnforcer {
public:
  /**
   * @brief Configuration for minimal closure
   */
  struct Config {
    /** Prefer green refinement */
    bool prefer_green = true;

    /** Prefer blue refinement */
    bool prefer_blue = false;

    /** Allow anisotropic refinement for closure */
    bool allow_anisotropic = false;

    /** Maximum closure depth */
    size_t max_closure_depth = 3;

    /** Cost weight for additional refinement */
    double refinement_cost = 1.0;

    /** Cost weight for pattern complexity */
    double pattern_cost = 0.5;
  };

  MinimalClosureEnforcer() : MinimalClosureEnforcer(Config{}) {}
  explicit MinimalClosureEnforcer(const Config& config);

  NonConformity check_conformity(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks) const override;

  size_t enforce_conformity(
      const MeshBase& mesh,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options) const override;

  std::map<size_t, std::map<size_t, double>> generate_constraints(
      const MeshBase& mesh,
      const NonConformity& non_conformity) const override;

  std::string name() const override { return "MinimalClosure"; }

private:
  Config config_;

  /** Compute optimal closure pattern */
  std::vector<std::pair<size_t, RefinementPattern>> compute_minimal_closure(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks,
      const NonConformity& non_conformity) const;

  /** Compute closure cost */
  double compute_closure_cost(
      const std::vector<std::pair<size_t, RefinementPattern>>& closure) const;
};

/**
 * @brief Factory for creating conformity enforcers
 */
class ConformityEnforcerFactory {
public:
  /**
   * @brief Create conformity enforcer based on options
   */
  static std::unique_ptr<ConformityEnforcer> create(const AdaptivityOptions& options);

  /**
   * @brief Create closure conformity enforcer
   */
  static std::unique_ptr<ConformityEnforcer> create_closure(
      const ClosureConformityEnforcer::Config& config = ClosureConformityEnforcer::Config{});

  /**
   * @brief Create hanging node conformity enforcer
   */
  static std::unique_ptr<ConformityEnforcer> create_hanging_node(
      const HangingNodeConformityEnforcer::Config& config = HangingNodeConformityEnforcer::Config{});

  /**
   * @brief Create minimal closure enforcer
   */
  static std::unique_ptr<ConformityEnforcer> create_minimal_closure(
      const MinimalClosureEnforcer::Config& config = MinimalClosureEnforcer::Config{});
};

/**
 * @brief Utility functions for conformity
 */
class ConformityUtils {
public:
  /**
   * @brief Check if mesh is globally conforming
   */
  static bool is_mesh_conforming(const MeshBase& mesh);

  /**
   * @brief Find all hanging nodes in mesh
   */
  static std::vector<HangingNode> find_hanging_nodes(const MeshBase& mesh);

  /**
   * @brief Check level difference between adjacent elements
   */
  static size_t check_level_difference(
      const MeshBase& mesh,
      size_t elem1,
      size_t elem2);

  /**
   * @brief Apply constraint equations to solution vector
   */
  static void apply_constraints(
      std::vector<double>& solution,
      const std::map<size_t, std::map<size_t, double>>& constraints);

  /**
   * @brief Remove constraints from system matrix
   */
  static void eliminate_constraints(
      std::vector<std::vector<double>>& matrix,
      std::vector<double>& rhs,
      const std::map<size_t, std::map<size_t, double>>& constraints);

  /**
   * @brief Visualize non-conformity
   */
  static void write_nonconformity_to_field(
      MeshFields& fields,
      const MeshBase& mesh,
      const NonConformity& non_conformity);
};

} // namespace svmp

#endif // SVMP_CONFORMITY_H
