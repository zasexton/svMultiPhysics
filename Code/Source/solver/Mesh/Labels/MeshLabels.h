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

#ifndef SVMP_MESH_LABELS_H
#define SVMP_MESH_LABELS_H

#include "../Core/MeshTypes.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Label and set management for mesh entities
 *
 * This class manages:
 * - Region labels (material/subdomain IDs for cells)
 * - Boundary labels (boundary condition IDs for faces)
 * - Edge and vertex labels (feature/marker tags)
 * - Named sets of entities
 * - Label-name bidirectional mapping
 */
class MeshLabels {
public:
  // ---- Region labels (cells) ----

  /**
   * @brief Set region label for a cell
   * @param mesh The mesh
   * @param cell Cell index
   * @param label Region/material label
   */
  static void set_region_label(MeshBase& mesh, index_t cell, label_t label);

  /**
   * @brief Get region label for a cell
   * @param mesh The mesh
   * @param cell Cell index
   * @return Region label (0 by default, INVALID_LABEL for invalid index)
   */
  static label_t region_label(const MeshBase& mesh, index_t cell);

  /**
   * @brief Find all cells with given region label
   * @param mesh The mesh
   * @param label Region label
   * @return Vector of cell indices
   */
  static std::vector<index_t> cells_with_region(const MeshBase& mesh, label_t label);

  /**
   * @brief Set region labels for multiple cells
   * @param mesh The mesh
   * @param cells Cell indices
   * @param label Region label
   */
  static void set_region_labels(MeshBase& mesh,
                               const std::vector<index_t>& cells,
                               label_t label);

  /**
   * @brief Get all unique region labels
   * @param mesh The mesh
   * @return Set of region labels
   */
  static std::unordered_set<label_t> unique_region_labels(const MeshBase& mesh);

  /**
   * @brief Count cells by region label
   * @param mesh The mesh
   * @return Map from label to cell count
   */
  static std::unordered_map<label_t, size_t> count_by_region(const MeshBase& mesh);

  // ---- Boundary labels (faces) ----

  /**
   * @brief Set boundary label for a face
   * @param mesh The mesh
   * @param face Face index
   * @param label Boundary condition label
   */
  static void set_boundary_label(MeshBase& mesh, index_t face, label_t label);

  /**
   * @brief Get boundary label for a face
   * @param mesh The mesh
   * @param face Face index
   * @return Boundary label (INVALID_LABEL if not set or invalid index)
   */
  static label_t boundary_label(const MeshBase& mesh, index_t face);

  /**
   * @brief Find all faces with given boundary label
   * @param mesh The mesh
   * @param label Boundary label
   * @return Vector of face indices
   */
  static std::vector<index_t> faces_with_boundary(const MeshBase& mesh, label_t label);

  /**
   * @brief Set boundary labels for multiple faces
   * @param mesh The mesh
   * @param faces Face indices
   * @param label Boundary label
   */
  static void set_boundary_labels(MeshBase& mesh,
                                 const std::vector<index_t>& faces,
                                 label_t label);

  /**
   * @brief Get all unique boundary labels
   * @param mesh The mesh
   * @return Set of boundary labels
   */
  static std::unordered_set<label_t> unique_boundary_labels(const MeshBase& mesh);

  /**
   * @brief Count faces by boundary label
   * @param mesh The mesh
   * @return Map from label to face count
   */
  static std::unordered_map<label_t, size_t> count_by_boundary(const MeshBase& mesh);

  // ---- Edge labels ----

  static void set_edge_label(MeshBase& mesh, index_t edge, label_t label);
  static label_t edge_label(const MeshBase& mesh, index_t edge);
  static std::vector<index_t> edges_with_label(const MeshBase& mesh, label_t label);
  static void set_edge_labels(MeshBase& mesh,
                              const std::vector<index_t>& edges,
                              label_t label);
  static std::unordered_set<label_t> unique_edge_labels(const MeshBase& mesh);
  static std::unordered_map<label_t, size_t> count_by_edge(const MeshBase& mesh);

  // ---- Vertex labels ----

  static void set_vertex_label(MeshBase& mesh, index_t vertex, label_t label);
  static label_t vertex_label(const MeshBase& mesh, index_t vertex);
  static std::vector<index_t> vertices_with_label(const MeshBase& mesh, label_t label);
  static void set_vertex_labels(MeshBase& mesh,
                                const std::vector<index_t>& vertices,
                                label_t label);
  static std::unordered_set<label_t> unique_vertex_labels(const MeshBase& mesh);
  static std::unordered_map<label_t, size_t> count_by_vertex(const MeshBase& mesh);

  // ---- Named sets ----

  /**
   * @brief Add entity to named set
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   * @param entity_id Entity index
   */
  static void add_to_set(MeshBase& mesh,
                        EntityKind kind,
                        const std::string& set_name,
                        index_t entity_id);

  /**
   * @brief Add multiple entities to named set
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   * @param entity_ids Entity indices
   */
  static void add_to_set(MeshBase& mesh,
                        EntityKind kind,
                        const std::string& set_name,
                        const std::vector<index_t>& entity_ids);

  /**
   * @brief Remove entity from named set
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   * @param entity_id Entity index
   */
  static void remove_from_set(MeshBase& mesh,
                             EntityKind kind,
                             const std::string& set_name,
                             index_t entity_id);

  /**
   * @brief Get entities in named set
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   * @return Vector of entity indices
   */
  static std::vector<index_t> get_set(const MeshBase& mesh,
                                     EntityKind kind,
                                     const std::string& set_name);

  /**
   * @brief Check if set exists
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   * @return True if set exists
   */
  static bool has_set(const MeshBase& mesh,
                     EntityKind kind,
                     const std::string& set_name);

  /**
   * @brief Remove a named set
   * @param mesh The mesh
   * @param kind Entity type
   * @param set_name Set name
   */
  static void remove_set(MeshBase& mesh,
                        EntityKind kind,
                        const std::string& set_name);

  /**
   * @brief List all set names for an entity type
   * @param mesh The mesh
   * @param kind Entity type
   * @return Vector of set names
   */
  static std::vector<std::string> list_sets(const MeshBase& mesh,
                                           EntityKind kind);

  /**
   * @brief Create set from entities with given label
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @param set_name Set name to create
   * @param label Label value
   */
  static void create_set_from_label(MeshBase& mesh,
                                   EntityKind kind,
                                   const std::string& set_name,
                                   label_t label);

  // ---- Label-name registry ----

  /**
   * @brief Register a name for a label
   * @param mesh The mesh
   * @param name Label name
   * @param label Label value
   */
  static void register_label(MeshBase& mesh,
                            const std::string& name,
                            label_t label);

  /**
   * @brief Get name for a label
   * @param mesh The mesh
   * @param label Label value
   * @return Label name (empty if not registered)
   */
  static std::string label_name(const MeshBase& mesh, label_t label);

  /**
   * @brief Get label for a name
   * @param mesh The mesh
   * @param name Label name
   * @return Label value (-1 if not registered)
   */
  static label_t label_from_name(const MeshBase& mesh, const std::string& name);

  /**
   * @brief List all registered label names
   * @param mesh The mesh
   * @return Map from label to name
   */
  static std::unordered_map<label_t, std::string> list_label_names(const MeshBase& mesh);

  /**
   * @brief Clear label registry
   * @param mesh The mesh
   */
  static void clear_label_registry(MeshBase& mesh);

  // ---- Operations ----

  /**
   * @brief Renumber labels to be contiguous starting from 0
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @return Map from old to new labels
   */
  static std::unordered_map<label_t, label_t> renumber_labels(MeshBase& mesh,
                                                              EntityKind kind);

  /**
   * @brief Merge multiple labels into one
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @param source_labels Labels to merge
   * @param target_label Target label
   */
  static void merge_labels(MeshBase& mesh,
                         EntityKind kind,
                         const std::vector<label_t>& source_labels,
                         label_t target_label);

  /**
   * @brief Split entities with one label into multiple labels based on connectivity
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @param label Label to split
   * @return Map from entity to new component label
   */
  static std::unordered_map<index_t, label_t> split_by_connectivity(MeshBase& mesh,
                                                                    EntityKind kind,
                                                                    label_t label);

  /**
   * @brief Copy labels from one mesh to another
   * @param source Source mesh
   * @param target Target mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   */
  static void copy_labels(const MeshBase& source,
                         MeshBase& target,
                         EntityKind kind);

  /**
   * @brief Export labels to array
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @return Vector of labels indexed by entity
   */
  static std::vector<label_t> export_labels(const MeshBase& mesh,
                                           EntityKind kind);

  /**
   * @brief Import labels from array
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, or Volume)
   * @param labels Vector of labels indexed by entity
   */
  static void import_labels(MeshBase& mesh,
                          EntityKind kind,
                          const std::vector<label_t>& labels);
};

} // namespace svmp

#endif // SVMP_MESH_LABELS_H
