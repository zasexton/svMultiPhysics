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

#ifndef SVMP_TEST_MESH_FIELDS_FIXTURE_H
#define SVMP_TEST_MESH_FIELDS_FIXTURE_H

#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace test {

/**
 * @brief Fixture combining mesh and fields for error estimator testing
 *
 * Provides convenient mesh + field creation for tests that require
 * realistic field data. Reuses existing mesh builder patterns and follows
 * MeshFields static API conventions.
 *
 * Design Philosophy:
 * - NO DUPLICATION: Reuses mesh building patterns from existing tests
 * - FOLLOWS CONVENTIONS: Uses MeshFields static API (fields stored in MeshBase)
 * - MINIMAL: Adds only what's missing from current test infrastructure
 *
 * Usage Example:
 * @code
 * auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);
 * auto indicators = estimator->estimate(fixture.mesh, nullptr, options);
 * @endcode
 */
class MeshWithFieldsFixture {
public:
  MeshBase mesh;
  std::vector<FieldHandle> field_handles;

  // ---- Factory Methods ----

  /**
   * @brief Create 2D uniform triangle mesh with linear field
   *
   * Creates a mesh with field: f(x,y) = slope_x * x + slope_y * y + offset
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param slope_x Slope in x direction
   * @param slope_y Slope in y direction
   * @param offset Constant offset
   * @param field_name Name of the field (default: "solution")
   * @return Fixture with mesh and attached field
   */
  static MeshWithFieldsFixture create_2d_uniform_with_linear_field(
      size_t nx, size_t ny,
      double slope_x = 1.0,
      double slope_y = 1.0,
      double offset = 0.0,
      const std::string& field_name = "solution");

  /**
   * @brief Create 2D mesh with quadratic field
   *
   * Creates a mesh with field: f(x,y) = x^2 + y^2
   * Useful for testing gradient recovery on smooth nonlinear fields.
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param field_name Name of the field (default: "solution")
   * @return Fixture with mesh and attached field
   */
  static MeshWithFieldsFixture create_2d_uniform_with_quadratic_field(
      size_t nx, size_t ny,
      const std::string& field_name = "solution");

  /**
   * @brief Create 2D mesh with discontinuous field
   *
   * Creates a mesh with step function: f(x) = 0 for x < midpoint, 1 otherwise
   * Useful for testing jump indicators and shock capturing.
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param field_name Name of the field (default: "solution")
   * @return Fixture with mesh and attached field
   */
  static MeshWithFieldsFixture create_2d_uniform_with_discontinuous_field(
      size_t nx, size_t ny,
      const std::string& field_name = "solution");

  /**
   * @brief Create 2D mesh with constant field
   *
   * Creates a mesh with constant field value throughout.
   * Useful for testing estimators on zero-gradient solutions.
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param value Constant field value
   * @param field_name Name of the field (default: "solution")
   * @return Fixture with mesh and attached field
   */
  static MeshWithFieldsFixture create_2d_uniform_with_constant_field(
      size_t nx, size_t ny,
      double value = 1.0,
      const std::string& field_name = "solution");

  /**
   * @brief Create 2D mesh with multiple fields
   *
   * Creates a mesh with multiple named fields, all initialized to zero.
   * Useful for multi-physics testing and field composition.
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param field_names Vector of field names to create
   * @return Fixture with mesh and attached fields
   */
  static MeshWithFieldsFixture create_2d_uniform_with_multiple_fields(
      size_t nx, size_t ny,
      const std::vector<std::string>& field_names);

  /**
   * @brief Create anisotropic (stretched) mesh with linear field
   *
   * Creates a mesh with anisotropic elements (aspect ratio ~10:1)
   * and a linear field. Useful for testing estimators on badly-shaped elements.
   *
   * @param nx Number of cells in x direction
   * @param ny Number of cells in y direction
   * @param slope_x Slope in x direction
   * @param slope_y Slope in y direction
   * @param offset Constant offset
   * @param field_name Name of the field (default: "solution")
   * @return Fixture with anisotropic mesh and attached field
   */
  static MeshWithFieldsFixture create_2d_anisotropic_with_linear_field(
      size_t nx, size_t ny,
      double slope_x = 1.0,
      double slope_y = 0.0,
      double offset = 0.0,
      const std::string& field_name = "solution");

  // ---- Field Access ----

  /**
   * @brief Get field handle by name
   *
   * @param name Field name
   * @return Field handle
   * @throws std::runtime_error if field not found
   */
  FieldHandle get_field(const std::string& name) const;

  /**
   * @brief Get mutable field data pointer
   *
   * @tparam T Data type (default: double)
   * @param name Field name
   * @return Pointer to field data
   */
  template <typename T = double>
  T* field_data(const std::string& name) {
    auto handle = get_field(name);
    return MeshFields::field_data_as<T>(mesh, handle);
  }

  /**
   * @brief Get const field data pointer
   *
   * @tparam T Data type (default: double)
   * @param name Field name
   * @return Const pointer to field data
   */
  template <typename T = double>
  const T* field_data(const std::string& name) const {
    auto handle = get_field(name);
    return MeshFields::field_data_as<T>(mesh, handle);
  }

  /**
   * @brief Check if field exists
   *
   * @param name Field name
   * @return True if field exists
   */
  bool has_field(const std::string& name) const {
    return field_map_.find(name) != field_map_.end();
  }

private:
  std::unordered_map<std::string, FieldHandle> field_map_;

  /**
   * @brief Register a field in the internal map
   *
   * @param name Field name
   * @param handle Field handle
   */
  void register_field(const std::string& name, FieldHandle handle) {
    field_handles.push_back(handle);
    field_map_[name] = handle;
  }
};

}} // namespace svmp::test

#endif // SVMP_TEST_MESH_FIELDS_FIXTURE_H
