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

#ifndef SVMP_REFINEMENT_RULES_H
#define SVMP_REFINEMENT_RULES_H

#include "../Topology/CellTopology.h"
#include <array>
#include <memory>
#include <vector>

namespace svmp {

/**
 * @brief Refinement pattern types
 */
enum class RefinementPattern {
  RED,          // Regular refinement (all edges split)
  GREEN,        // Compatibility refinement
  BLUE,         // Alternative compatibility
  BISECTION,    // Edge bisection
  ISOTROPIC,    // Uniform in all directions
  ANISOTROPIC   // Directional refinement
};

/**
 * @brief Information about a refined element
 */
struct RefinedElement {
  /** Child element connectivity (vertex indices) */
  std::vector<std::vector<index_t>> child_connectivity;

  /** New vertex positions (for vertices created by this element) */
  std::vector<std::array<real_t, 3>> new_vertices;

  /** Parent-child face mapping */
  std::vector<std::pair<index_t, index_t>> face_inheritance;

  /** Refinement level of children */
  size_t child_level;

  /** Pattern used for refinement */
  RefinementPattern pattern;
};

/**
 * @brief Abstract base class for element refinement rules
 */
class RefinementRule {
public:
  virtual ~RefinementRule() = default;

  /**
   * @brief Check if element can be refined
   */
  virtual bool can_refine(CellFamily family, size_t level) const = 0;

  /**
   * @brief Get number of children produced by refinement
   */
  virtual size_t num_children(CellFamily family, RefinementPattern pattern) const = 0;

  /**
   * @brief Refine an element
   *
   * @param vertices Element vertex coordinates
   * @param type Element type
   * @param pattern Refinement pattern to use
   * @param level Current refinement level
   * @return Refined element information
   */
  virtual RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const = 0;

  /**
   * @brief Get compatible refinement patterns for element type
   */
  virtual std::vector<RefinementPattern> compatible_patterns(CellFamily family) const = 0;

  /**
   * @brief Get default refinement pattern for element type
   */
  virtual RefinementPattern default_pattern(CellFamily family) const = 0;
};

/**
 * @brief Refinement rules for line elements
 */
class LineRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;
};

/**
 * @brief Refinement rules for triangle elements
 */
class TriangleRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;

private:
  /** Red refinement (4 children) */
  RefinedElement red_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t level) const;

  /** Green refinement (2 children) */
  RefinedElement green_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t edge_to_split,
      size_t level) const;

  /** Bisection refinement */
  RefinedElement bisect(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t level) const;
};

/**
 * @brief Refinement rules for quadrilateral elements
 */
class QuadRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;

private:
  /** Regular refinement (4 children) */
  RefinedElement regular_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t level) const;

  /** Anisotropic refinement in one direction (2 children) */
  RefinedElement anisotropic_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t direction,
      size_t level) const;
};

/**
 * @brief Refinement rules for tetrahedral elements
 */
class TetrahedronRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;

private:
  /** Red refinement (8 children) */
  RefinedElement red_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t level) const;

  /** Bisection refinement (2 children) */
  RefinedElement bisect(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t longest_edge,
      size_t level) const;
};

/**
 * @brief Refinement rules for hexahedral elements
 */
class HexahedronRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;

private:
  /** Regular refinement (8 children) */
  RefinedElement regular_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t level) const;

  /** Anisotropic refinement in one direction (4 children) */
  RefinedElement anisotropic_refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      size_t direction,
      size_t level) const;
};

/**
 * @brief Refinement rules for wedge/prism elements
 */
class WedgeRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;
};

/**
 * @brief Refinement rules for pyramid elements
 */
class PyramidRefinementRule : public RefinementRule {
public:
  bool can_refine(CellFamily family, size_t level) const override;
  size_t num_children(CellFamily family, RefinementPattern pattern) const override;
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const override;
  std::vector<RefinementPattern> compatible_patterns(CellFamily family) const override;
  RefinementPattern default_pattern(CellFamily family) const override;
};

/**
 * @brief Manager for all refinement rules
 */
class RefinementRulesManager {
public:
  /** Singleton instance */
  static RefinementRulesManager& instance();

  /**
   * @brief Get refinement rule for element type
   */
  RefinementRule* get_rule(CellFamily family) const;

  /**
   * @brief Check if element type can be refined
   */
  bool can_refine(CellFamily family, size_t level) const;

  /**
   * @brief Refine an element
   */
  RefinedElement refine(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family,
      RefinementPattern pattern,
      size_t level) const;

  /**
   * @brief Get number of children for refinement
   */
  size_t num_children(CellFamily family, RefinementPattern pattern) const;

  /**
   * @brief Register a custom refinement rule
   */
  void register_rule(CellFamily family, std::unique_ptr<RefinementRule> rule);

private:
  RefinementRulesManager();
  static constexpr size_t kCellFamilyCount = static_cast<size_t>(CellFamily::Polyhedron) + 1;
  std::array<std::unique_ptr<RefinementRule>, kCellFamilyCount> rules_{};
  std::array<RefinementRule*, kCellFamilyCount> rule_map_{};
};

/**
 * @brief Utility functions for refinement
 */
class RefinementUtils {
public:
  /**
   * @brief Compute edge midpoint
   */
  static std::array<real_t, 3> edge_midpoint(
      const std::array<real_t, 3>& v1,
      const std::array<real_t, 3>& v2);

  /**
   * @brief Compute face center
   */
  static std::array<real_t, 3> face_center(
      const std::vector<std::array<real_t, 3>>& face_vertices);

  /**
   * @brief Compute cell center
   */
  static std::array<real_t, 3> cell_center(
      const std::vector<std::array<real_t, 3>>& cell_vertices);

  /**
   * @brief Find longest edge of element
   */
  static size_t find_longest_edge(
      const std::vector<std::array<real_t, 3>>& vertices,
      CellFamily family);

  /**
   * @brief Check if refinement preserves quality
   */
  static bool check_refinement_quality(
      const RefinedElement& refined,
      real_t min_quality = 0.1);

  /**
   * @brief Generate edge-to-vertex connectivity for refined element
   */
  static std::vector<std::pair<size_t, size_t>> generate_edge_connectivity(
      const RefinedElement& refined);
};

} // namespace svmp

#endif // SVMP_REFINEMENT_RULES_H
