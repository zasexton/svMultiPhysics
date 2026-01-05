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

#include <algorithm>
#include <cctype>
#include <stdexcept>

/**
 * @file Mesh.h
 * @brief Main entry point for the refactored mesh infrastructure
 *
 * This header includes all the essential mesh components:
 * - Core types and data structures
 * - MeshBase class for runtime-dimensional meshes
 * - MeshView<Dim> template for compile-time dimensional local meshes
 * - Component classes for specialized functionality
 */

// Core components
#include "Core/MeshTypes.h"
#include "Core/MeshComm.h"
#include "Core/MeshBase.h"
#include "Core/DistributedMesh.h"

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

// The main public mesh class is Mesh (a DistributedMesh that works in serial + MPI).
// MeshBase remains the underlying local mesh implementation.
// Additional functionality is provided by component classes:
// - MeshGeometry: Geometric computations
// - MeshQuality: Quality metrics
// - MeshTopology: Adjacency and topology operations
// - MeshFields: Field management
// - MeshLabels: Label and set operations
// - MeshSearch: Point location and spatial queries

/**
 * @brief Compile-time typed local mesh view
 *
 * Provides a typed interface to MeshBase with dimension known at compile time.
 * This allows for more efficient code generation and type safety.
 *
 * @tparam Dim Spatial dimension (1, 2, or 3)
 */
template <int Dim>
class MeshView {
public:
  explicit MeshView(std::shared_ptr<MeshBase> base)
    : base_(std::move(base))
  {
    if (!base_) throw std::invalid_argument("MeshView<Dim>: null base");
    if (base_->dim() != 0 && base_->dim() != Dim) {
      throw std::invalid_argument("MeshView<Dim>: dimension mismatch");
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

  std::array<real_t,3> cell_centroid(index_t c, Configuration cfg = Configuration::Reference) const {
    return base_->cell_centroid(c, cfg);
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

// Main public mesh type (distributed when MPI is enabled, serial otherwise).
using Mesh = DistributedMesh;

// Convenience aliases for compile-time dimensional local views
using Mesh1D = MeshView<1>;
using Mesh2D = MeshView<2>;
using Mesh3D = MeshView<3>;

// Convenience aliases for compile-time dimensional distributed meshes
template <int Dim>
using Mesh_t = DistributedMesh_t<Dim>;

// ============================================================================
// DistributedMesh as Default Mesh Type
// ============================================================================
//
// DistributedMesh is the recommended mesh type for all applications:
// - In serial mode (no MPI): operates as a simple wrapper around MeshBase
// - In parallel mode (with MPI): provides full distributed mesh functionality
//
// This allows the same code to work in both serial and parallel contexts
// without modification.
// ============================================================================

/**
 * @brief Create an empty distributed mesh
 * @return Shared pointer to a new DistributedMesh
 */
inline std::shared_ptr<Mesh> create_mesh() {
  return std::make_shared<Mesh>(MeshComm::world());
}

/**
 * @brief Create a distributed mesh wrapping an existing MeshBase
 * @param base Shared pointer to an existing MeshBase
 * @return Shared pointer to a new DistributedMesh wrapping the base
 */
inline std::shared_ptr<Mesh> create_mesh(std::shared_ptr<MeshBase> base) {
  return std::make_shared<Mesh>(std::move(base));
}

/**
 * @brief Create a mesh with a specific communicator
 * @param comm Mesh communicator (serial-safe)
 * @return Shared pointer to a new DistributedMesh
 */
inline std::shared_ptr<Mesh> create_mesh(MeshComm comm) {
  return std::make_shared<Mesh>(comm);
}

/**
 * @brief Create a mesh wrapping an existing MeshBase with a communicator
 * @param base Shared pointer to an existing MeshBase
 * @param comm Mesh communicator (serial-safe)
 * @return Shared pointer to a new DistributedMesh
 */
inline std::shared_ptr<Mesh> create_mesh(std::shared_ptr<MeshBase> base, MeshComm comm) {
  return std::make_shared<Mesh>(std::move(base), comm);
}

#ifdef MESH_HAS_MPI
/**
 * @brief Create a mesh with a specific MPI communicator
 * @param comm MPI communicator
 * @return Shared pointer to a new Mesh
 */
[[deprecated("Use create_mesh(MeshComm) instead of MPI_Comm overloads.")]]
inline std::shared_ptr<Mesh> create_mesh(MPI_Comm comm) {
  return create_mesh(MeshComm(comm));
}

/**
 * @brief Create a mesh wrapping an existing MeshBase with MPI communicator
 * @param base Shared pointer to an existing MeshBase
 * @param comm MPI communicator
 * @return Shared pointer to a new Mesh
 */
[[deprecated("Use create_mesh(std::shared_ptr<MeshBase>, MeshComm) instead of MPI_Comm overloads.")]]
inline std::shared_ptr<Mesh> create_mesh(std::shared_ptr<MeshBase> base, MPI_Comm comm) {
  return create_mesh(std::move(base), MeshComm(comm));
}
#endif

/**
 * @brief Load a mesh file and wrap it in a DistributedMesh
 *
 * In MPI builds, when `comm.is_parallel()` this dispatches to
 * `DistributedMesh::load_parallel(...)` (rank-0 load + distribute or
 * format-specific parallel I/O). In serial builds this loads locally.
 *
 * @param options I/O options for loading
 * @param comm Mesh communicator (serial-safe)
 * @return Shared pointer to a DistributedMesh containing the loaded mesh
 */
inline std::shared_ptr<Mesh> load_mesh(const MeshIOOptions& options, MeshComm comm) {
  if (comm.is_parallel()) {
    auto dmesh = Mesh::load_parallel(options, comm);
    return std::make_shared<Mesh>(std::move(dmesh));
  }
  auto base = std::make_shared<MeshBase>(MeshBase::load(options));
  return std::make_shared<Mesh>(std::move(base), comm);
}

/**
 * @brief Load a mesh file using the default communicator policy.
 *
 * The default is `MeshComm::world()` so MPI programs get distributed load by default.
 */
inline std::shared_ptr<Mesh> load_mesh(const MeshIOOptions& options) {
  return load_mesh(options, MeshComm::world());
}

/**
 * @brief Load a mesh file and wrap it in a DistributedMesh
 *
 * @param filename Path to mesh file
 * @param comm Mesh communicator (serial-safe)
 * @return Shared pointer to a DistributedMesh containing the loaded mesh
 */
inline std::shared_ptr<Mesh> load_mesh(const std::string& filename, MeshComm comm) {
  MeshIOOptions opts;
  opts.path = filename;
  // Best-effort format inference from file extension.
  const auto dot = filename.find_last_of('.');
  if (dot != std::string::npos && dot + 1 < filename.size()) {
    std::string ext = filename.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    opts.format = ext;
  }
  return load_mesh(opts, comm);
}

/**
 * @brief Load a mesh file using the default communicator policy.
 *
 * The default is `MeshComm::world()` so MPI programs get distributed load by default.
 */
inline std::shared_ptr<Mesh> load_mesh(const std::string& filename) {
  return load_mesh(filename, MeshComm::world());
}

/**
 * @brief Save a mesh using distributed I/O policy (or local save in serial).
 */
inline void save_mesh(const Mesh& mesh, const MeshIOOptions& options) {
  mesh.save_parallel(options);
}

/**
 * @brief Save a mesh file with best-effort format inference from file extension.
 */
inline void save_mesh(const Mesh& mesh, const std::string& filename) {
  MeshIOOptions opts;
  opts.path = filename;
  const auto dot = filename.find_last_of('.');
  if (dot != std::string::npos && dot + 1 < filename.size()) {
    std::string ext = filename.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    opts.format = ext;
  }
  save_mesh(mesh, opts);
}

} // namespace svmp

#endif // SVMP_MESH_H
