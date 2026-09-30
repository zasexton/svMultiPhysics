/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

// PDE-based extension of the physical velocity into the dry part of an
// unfitted level-set mesh.  The result is the advection velocity w of the
// level-set transport: w = u on every "known" vertex (the wet vertices and all
// vertices of the retained interface cells, so that w_h = u_h on the
// interface), and on the remaining (dry) vertices w solves a parameter-free
// linear problem with that Dirichlet data:
//
//   Harmonic:            find w with  (grad w, grad v)_D = 0          for all v,
//   LeastSquaresNormal:  find w with  ((n.grad) w, (n.grad) v)_D = 0  for all v,
//                        n = grad(phi_h)/|grad(phi_h)| per cell, i.e. the
//                        minimizer of ||(n.grad) w||^2 (w constant along the
//                        level-set normals).
//
// D is the union of the cells that contain a dry vertex.  The outer boundary
// of D carries the natural (zero-flux) condition.  Velocity components that a
// strong homogeneous wall Dirichlet condition constrains are set to zero on
// dry wall vertices (no penetration); the other components keep the natural
// condition.  Components are independent because every supported wall mask is
// axis-aligned.
//
// The system is small (dry vertices only).  Each rank contributes the element
// matrices of its owned cells; every rank assembles the same global system in
// a canonical order (sorted global IDs) and solves it with the FE Eigen
// backend's direct solver, so the result does not depend on the partition.

#include "Application/Core/LevelSetVelocityExtensionMap.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Mesh.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace application::core {

enum class PdeVelocityExtensionOperator : std::uint8_t {
  LeastSquaresNormal,
  Harmonic,
};

[[nodiscard]] std::string_view pdeVelocityExtensionOperatorName(
    PdeVelocityExtensionOperator op) noexcept;

// Maps an input method token (case, '_' and '-' insensitive) to an operator:
// pde_normal / pde_least_squares_normal -> LeastSquaresNormal, pde_harmonic
// -> Harmonic.  Any other token returns nullopt.
[[nodiscard]] std::optional<PdeVelocityExtensionOperator>
pdeVelocityExtensionOperatorFromToken(std::string_view token);

struct PdeVelocityExtensionOptions {
  PdeVelocityExtensionOperator op{PdeVelocityExtensionOperator::Harmonic};
  // <= 0: the extension domain is every dry vertex of the mesh (default).
  // > 0: diagnostic truncation to this many vertex rings beyond the known
  // set; vertices outside receive zero.
  int band_layers{0};
  bool enforce_wall_impermeability{true};
};

struct PdeVelocityExtensionReport {
  std::size_t known_vertices{0u};
  std::size_t extension_vertices{0u};
  std::size_t outside_vertices{0u};
  std::size_t extension_cells{0u};
  std::array<std::size_t, 3> unknowns{0u, 0u, 0u};
  std::array<std::size_t, 3> wall_fixed{0u, 0u, 0u};
  double max_known_speed{0.0};
  double max_extended_speed{0.0};
  double max_relative_residual{0.0};
  double max_wall_normal_velocity{0.0};
};

// level_set: one value per local vertex (only its gradient is used).
// source: source_components values per local vertex.
// known: communicator-consistent mask of the known vertices.
// extended: resized to n_vertices * target_components on return.
PdeVelocityExtensionReport extendVelocityByPde(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> level_set,
    std::span<const double> source,
    std::size_t source_components,
    std::span<const std::uint8_t> known,
    std::size_t target_components,
    std::span<const WallVelocityExtensionConstraint> walls,
    const PdeVelocityExtensionOptions& options,
    std::vector<double>& extended);

} // namespace application::core
