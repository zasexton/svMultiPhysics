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

#ifndef SVMP_MESH_FIELDS_H
#define SVMP_MESH_FIELDS_H

#include "../Core/MeshTypes.h"
#include "../MeshFieldDescriptor.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Field attachment and management for mesh entities
 *
 * This class manages fields (scalar, vector, tensor data) attached to mesh entities:
 * - Type-erased storage for arbitrary data types
 * - Field metadata and descriptors
 * - Efficient field access and iteration
 * - Field interpolation and restriction
 */
class MeshFields {
public:
  /**
   * @brief Field storage information
   */
  struct FieldInfo {
    FieldScalarType type;              // Scalar type
    size_t components;                  // Number of components per entity
    size_t bytes_per_component;         // Bytes per component (for Custom type)
    std::vector<uint8_t> data;          // Raw data storage
    FieldDescriptor descriptor;         // Optional metadata
  };

  /**
   * @brief Field collection for an entity type
   */
  struct FieldCollection {
    std::unordered_map<std::string, FieldInfo> fields;
    size_t entity_count = 0;
  };

  // ---- Field attachment ----

  /**
   * @brief Attach a new field to entities
   * @param mesh The mesh
   * @param kind Entity type (Vertex, Edge, Face, Cell)
   * @param name Field name
   * @param type Scalar data type
   * @param components Number of components per entity
   * @param custom_bytes_per_component Bytes per component for Custom type
   * @return Field handle for accessing the field
   */
  static FieldHandle attach_field(MeshBase& mesh,
                                 EntityKind kind,
                                 const std::string& name,
                                 FieldScalarType type,
                                 size_t components,
                                 size_t custom_bytes_per_component = 0);

  /**
   * @brief Attach a field with descriptor
   * @param mesh The mesh
   * @param kind Entity type
   * @param name Field name
   * @param type Scalar data type
   * @param descriptor Field descriptor with metadata
   * @return Field handle
   */
  static FieldHandle attach_field_with_descriptor(MeshBase& mesh,
                                                 EntityKind kind,
                                                 const std::string& name,
                                                 FieldScalarType type,
                                                 const FieldDescriptor& descriptor);

  /**
   * @brief Remove a field
   * @param mesh The mesh
   * @param handle Field handle
   */
  static void remove_field(MeshBase& mesh, const FieldHandle& handle);

  /**
   * @brief Check if a field exists
   * @param mesh The mesh
   * @param kind Entity type
   * @param name Field name
   * @return True if field exists
   */
  static bool has_field(const MeshBase& mesh,
                       EntityKind kind,
                       const std::string& name);

  // ---- Field access ----

  /**
   * @brief Get raw field data pointer
   * @param mesh The mesh
   * @param handle Field handle
   * @return Pointer to field data
   */
  static void* field_data(MeshBase& mesh, const FieldHandle& handle);
  static const void* field_data(const MeshBase& mesh, const FieldHandle& handle);

  /**
   * @brief Get typed field data pointer
   * @tparam T Data type
   * @param mesh The mesh
   * @param handle Field handle
   * @return Typed pointer to field data
   */
  template <typename T>
  static T* field_data_as(MeshBase& mesh, const FieldHandle& handle) {
    return reinterpret_cast<T*>(field_data(mesh, handle));
  }

  template <typename T>
  static const T* field_data_as(const MeshBase& mesh, const FieldHandle& handle) {
    return reinterpret_cast<const T*>(field_data(mesh, handle));
  }

  /**
   * @brief Get field properties
   */
  static size_t field_components(const MeshBase& mesh, const FieldHandle& handle);
  static FieldScalarType field_type(const MeshBase& mesh, const FieldHandle& handle);
  static size_t field_entity_count(const MeshBase& mesh, const FieldHandle& handle);
  static size_t field_bytes_per_entity(const MeshBase& mesh, const FieldHandle& handle);

  /**
   * @brief Get field descriptor if available
   * @param mesh The mesh
   * @param handle Field handle
   * @return Pointer to descriptor or nullptr
   */
  static const FieldDescriptor* field_descriptor(const MeshBase& mesh,
                                                const FieldHandle& handle);

  // ---- Field queries ----

  /**
   * @brief List all fields attached to an entity type
   * @param mesh The mesh
   * @param kind Entity type
   * @return Vector of field names
   */
  static std::vector<std::string> list_fields(const MeshBase& mesh,
                                             EntityKind kind);

  /**
   * @brief Get field handle by name
   * @param mesh The mesh
   * @param kind Entity type
   * @param name Field name
   * @return Field handle (id = 0 if not found)
   */
  static FieldHandle get_field_handle(const MeshBase& mesh,
                                     EntityKind kind,
                                     const std::string& name);

  /**
   * @brief Count total number of fields
   * @param mesh The mesh
   * @return Total field count
   */
  static size_t total_field_count(const MeshBase& mesh);

  /**
   * @brief Get memory usage of all fields
   * @param mesh The mesh
   * @return Total bytes used by fields
   */
  static size_t field_memory_usage(const MeshBase& mesh);

  // ---- Field operations ----

  /**
   * @brief Copy field data from one field to another
   * @param mesh The mesh
   * @param source Source field handle
   * @param target Target field handle
   */
  static void copy_field(MeshBase& mesh,
                        const FieldHandle& source,
                        const FieldHandle& target);

  /**
   * @brief Fill field with constant value
   * @tparam T Data type
   * @param mesh The mesh
   * @param handle Field handle
   * @param value Value to fill
   */
  template <typename T>
  static void fill_field(MeshBase& mesh,
                        const FieldHandle& handle,
                        const T& value) {
    T* data = field_data_as<T>(mesh, handle);
    size_t count = field_entity_count(mesh, handle) * field_components(mesh, handle);
    std::fill(data, data + count, value);
  }

  /**
   * @brief Resize field for new entity count
   * @param mesh The mesh
   * @param kind Entity type
   * @param new_count New entity count
   */
  static void resize_fields(MeshBase& mesh,
                          EntityKind kind,
                          size_t new_count);

  // ---- Field interpolation ----

  /**
   * @brief Interpolate field from cells to nodes
   * @param mesh The mesh
   * @param cell_field Cell field handle
   * @param node_field Node field handle (must exist)
   */
  static void interpolate_cell_to_node(const MeshBase& mesh,
                                      const FieldHandle& cell_field,
                                      const FieldHandle& node_field);

  /**
   * @brief Interpolate field from nodes to cells
   * @param mesh The mesh
   * @param node_field Node field handle
   * @param cell_field Cell field handle (must exist)
   */
  static void interpolate_node_to_cell(const MeshBase& mesh,
                                      const FieldHandle& node_field,
                                      const FieldHandle& cell_field);

  /**
   * @brief Restrict field from fine to coarse mesh
   * @param fine_mesh Fine mesh
   * @param coarse_mesh Coarse mesh
   * @param fine_field Field on fine mesh
   * @param coarse_field Field on coarse mesh
   */
  static void restrict_field(const MeshBase& fine_mesh,
                            const MeshBase& coarse_mesh,
                            const FieldHandle& fine_field,
                            const FieldHandle& coarse_field);

  /**
   * @brief Prolongate field from coarse to fine mesh
   * @param coarse_mesh Coarse mesh
   * @param fine_mesh Fine mesh
   * @param coarse_field Field on coarse mesh
   * @param fine_field Field on fine mesh
   */
  static void prolongate_field(const MeshBase& coarse_mesh,
                              const MeshBase& fine_mesh,
                              const FieldHandle& coarse_field,
                              const FieldHandle& fine_field);

  // ---- Field statistics ----

  /**
   * @brief Compute field statistics
   */
  struct FieldStats {
    real_t min = 0;
    real_t max = 0;
    real_t mean = 0;
    real_t std_dev = 0;
    real_t sum = 0;
  };

  /**
   * @brief Compute statistics for a scalar field
   * @param mesh The mesh
   * @param handle Field handle
   * @param component Component index for multi-component fields
   * @return Field statistics
   */
  static FieldStats compute_stats(const MeshBase& mesh,
                                 const FieldHandle& handle,
                                 size_t component = 0);

  /**
   * @brief Compute L2 norm of field
   * @param mesh The mesh
   * @param handle Field handle
   * @return L2 norm
   */
  static real_t compute_l2_norm(const MeshBase& mesh,
                               const FieldHandle& handle);

  /**
   * @brief Compute infinity norm of field
   * @param mesh The mesh
   * @param handle Field handle
   * @return Infinity norm
   */
  static real_t compute_inf_norm(const MeshBase& mesh,
                                const FieldHandle& handle);

private:
  // Helper to get entity count
  static size_t entity_count(const MeshBase& mesh, EntityKind kind);

  // Helper to access field storage
  static FieldCollection& get_collection(MeshBase& mesh, EntityKind kind);
  static const FieldCollection& get_collection(const MeshBase& mesh, EntityKind kind);
};

} // namespace svmp

#endif // SVMP_MESH_FIELDS_H