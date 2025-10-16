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

#ifndef SVMP_MESH_H
#define SVMP_MESH_H

/**
 * @file Mesh.h
 * @brief Main entry point for the refactored mesh infrastructure
 *
 * This header includes all the essential mesh components:
 * - Core types and data structures
 * - MeshBase class for runtime-dimensional meshes
 * - Mesh<Dim> template for compile-time dimensional meshes
 * - Component classes for specialized functionality
 */

// Core components
#include "Core/MeshTypes.h"
#include "Core/MeshBase.h"

// Topology components
#include "Topology/CellShape.h"

// Geometry components
#include "Geometry/MeshGeometry.h"
#include "Geometry/MeshQuality.h"

// Observer pattern
#include "Observer/MeshObserver.h"
#include "Fields/MeshFieldDescriptor.h"

// For backward compatibility, provide svmp namespace exports
namespace svmp {

// The main mesh class is MeshBase (defined in Core/MeshBase.h)
// Additional functionality is provided by component classes:
// - MeshGeometry: Geometric computations
// - MeshQuality: Quality metrics
// - MeshTopology: Adjacency operations (to be implemented)
// - MeshFields: Field management (to be implemented)
// - MeshLabels: Label operations (to be implemented)
// - MeshSearch: Point location (to be implemented)

/**
 * @brief Compile-time typed mesh view
 *
 * Provides a typed interface to MeshBase with dimension known at compile time.
 * This allows for more efficient code generation and type safety.
 *
 * @tparam Dim Spatial dimension (1, 2, or 3)
 */
template <int Dim>
class Mesh {
public:
  explicit Mesh(std::shared_ptr<MeshBase> base)
    : base_(std::move(base))
  {
    if (!base_) throw std::invalid_argument("Mesh<Dim>: null base");
    if (base_->dim() != 0 && base_->dim() != Dim) {
      throw std::invalid_argument("Mesh<Dim>: dimension mismatch");
    }
  }

  // Dimension is known at compile time
  static constexpr int dimension() noexcept { return Dim; }
  int dim() const noexcept { return Dim; }

  // Access to underlying MeshBase
  const MeshBase& base() const noexcept { return *base_; }
  MeshBase& base() noexcept { return *base_; }

  // Fast path aliases for common operations
  size_t n_vertices() const noexcept { return base_->n_vertices(); }
  size_t n_cells() const noexcept { return base_->n_cells(); }
  size_t n_faces() const noexcept { return base_->n_faces(); }
  size_t n_edges() const noexcept { return base_->n_edges(); }

  const std::vector<real_t>& X_ref() const noexcept { return base_->X_ref(); }
  const std::vector<real_t>& X_cur() const noexcept { return base_->X_cur(); }
  bool has_current_coords() const noexcept { return base_->has_current_coords(); }

  const std::vector<CellShape>& cell_shapes() const noexcept { return base_->cell_shapes(); }
  const CellShape& cell_shape(index_t c) const { return base_->cell_shape(c); }

  // Delegate geometry operations
  std::array<real_t,3> cell_center(index_t c, Configuration cfg = Configuration::Reference) const {
    return base_->cell_center(c, cfg);
  }

  real_t cell_measure(index_t c, Configuration cfg = Configuration::Reference) const {
    return base_->cell_measure(c, cfg);
  }

  // Delegate quality operations
  real_t compute_quality(index_t cell, const std::string& metric = "aspect_ratio") const {
    return base_->compute_quality(cell, metric);
  }

  // Type-safe coordinate access (when dimension is known)
  template <int D = Dim, typename std::enable_if<D == 2, int>::type = 0>
  std::array<real_t,2> vertex_coords_2d(index_t n, Configuration cfg = Configuration::Reference) const {
    const std::vector<real_t>& coords = (cfg == Configuration::Current && base_->has_current_coords())
                                       ? base_->X_cur() : base_->X_ref();
    return {{coords[n*2], coords[n*2+1]}};
  }

  template <int D = Dim, typename std::enable_if<D == 3, int>::type = 0>
  std::array<real_t,3> vertex_coords_3d(index_t n, Configuration cfg = Configuration::Reference) const {
    const std::vector<real_t>& coords = (cfg == Configuration::Current && base_->has_current_coords())
                                       ? base_->X_cur() : base_->X_ref();
    return {{coords[n*3], coords[n*3+1], coords[n*3+2]}};
  }

private:
  std::shared_ptr<MeshBase> base_;
};

// Convenience aliases
using Mesh1D = Mesh<1>;
using Mesh2D = Mesh<2>;
using Mesh3D = Mesh<3>;

} // namespace svmp

#endif // SVMP_MESH_H
