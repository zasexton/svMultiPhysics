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
// With a factorization cache on three or more ranks, component c is
// factorized and solved on rank c only and its solution is broadcast bit for
// bit (the components are independent systems); the result is the same.

#include "Application/Core/LevelSetVelocityExtensionMap.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Mesh.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
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

// Sparse direct factorization of the dry-region systems.
//   LuColamd (default): Eigen::SparseLU with COLAMD ordering, the FE Eigen
//     backend's direct solve.
//   LdltAmd (opt-in): Eigen::SimplicialLDLT with AMD ordering.  Valid for the
//     symmetric positive definite harmonic operator only; much less fill in
//     3D.  Its results differ from LuColamd at round-off level (they are still
//     independent of the partition and the rank count).
//   Mumps (opt-in, FE_ENABLE_MUMPS builds): MUMPS LDL^T (METIS ordering)
//     distributed over every rank of the communicator; positive definite
//     mode for the harmonic operator, symmetric indefinite otherwise.  The
//     components are factorized and solved one after another on all ranks
//     (no per-rank component solves); every rank receives the solution.  Its
//     results differ from LuColamd at round-off level and may depend on the
//     rank count at round-off level.
enum class PdeVelocityExtensionFactorization : std::uint8_t {
  LuColamd,
  LdltAmd,
  Mumps,
};

[[nodiscard]] std::string_view pdeVelocityExtensionFactorizationName(
    PdeVelocityExtensionFactorization factorization) noexcept;

// Maps a token (case, '_' and '-' insensitive) to a factorization:
// lu / lu_colamd / default -> LuColamd, ldlt / ldlt_amd -> LdltAmd,
// mumps / mumps_distributed -> Mumps.  Any other token returns nullopt.
[[nodiscard]] std::optional<PdeVelocityExtensionFactorization>
pdeVelocityExtensionFactorizationFromToken(std::string_view token);

struct PdeVelocityExtensionOptions {
  PdeVelocityExtensionOperator op{PdeVelocityExtensionOperator::Harmonic};
  // <= 0: the extension domain is every dry vertex of the mesh (default).
  // > 0: diagnostic truncation to this many vertex rings beyond the known
  // set; vertices outside receive zero.
  int band_layers{0};
  bool enforce_wall_impermeability{true};
  PdeVelocityExtensionFactorization factorization{
      PdeVelocityExtensionFactorization::LuColamd};
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
  // True when the call reused the cached dry-region factorization (and rows).
  bool reused_factorization{false};
  // True when the components were factorized and solved on one rank each.
  bool distributed_solves{false};
  // True when SVMP_PDE_EXTENSION_SELF_CHECK compared the result with the
  // uncached replicated solve and found it bitwise identical.
  bool self_checked{false};
  // Entries held by the cache after the call.
  std::size_t cache_entries{0u};
  // Content hash of the dry-region solution in canonical (global vertex ID)
  // order; the same on every rank.
  std::uint64_t solution_hash{0u};
  // Wall time of the call on this rank and of this rank's component solves
  // (factorization and solve).
  double elapsed_seconds{0.0};
  double component_solve_seconds{0.0};
};

// Mesh revisions that, together with the exact system content, key a cached
// dry-region factorization (see PdeVelocityExtensionCache).
struct PdeVelocityExtensionMeshRevisions {
  std::uint64_t geometry{0u};
  std::uint64_t topology{0u};
  std::uint64_t ownership{0u};
  std::uint64_t numbering{0u};
};

// Reuse of the dry-region systems between calls of extendVelocityByPde.
//
// The dry-region matrix does not depend on the velocity.  For the harmonic
// operator it depends only on the mesh geometry, the known set, the dry cells
// and the wall masks; for the least-squares normal operator it also depends on
// the level-set normal of every dry cell.  The cache keeps the gathered
// system, the sparse LU factorization of every velocity component and the
// owner-local algebraic rows.  Its key is
//   - the mesh geometry, topology, ownership and numbering revisions, and
//   - the exact content of each rank's contribution: local vertex IDs,
//     vertex ownership, known set, extension domain, wall masks and the
//     element matrices of the owned dry cells, compared bitwise (the element
//     matrices carry the level-set normals of the least-squares operator),
// combined over the communicator.  The ranks agree on reuse collectively: a
// change on any rank refactors on every rank.  A reused call applies the same
// factorization to the new right-hand sides, so its result is bitwise
// identical to a fresh solve, and it skips the gather of the element
// matrices.  Any failure leaves the cache empty.
//
// The cache keeps several entries (most recently used first): the known set
// of successive outer passes can return to an earlier state, whose entry is
// then reused.  An entry is reused only if it matches on every rank; entries
// are added and evicted by collective decisions, so every rank holds the
// same entries.  With distributed component solves an entry keeps the matrix
// and factorization of a component only on the rank that solves it, so
// kDefaultDistributedEntries entries cost those ranks what one replicated
// entry costs every rank.  The capacity is SVMP_PDE_EXTENSION_CACHE_ENTRIES
// (1 to kMaxEntries) when set, otherwise kDefaultEntries for replicated and
// kDefaultDistributedEntries for distributed solves.
class PdeVelocityExtensionCache {
public:
  static constexpr std::size_t kMaxEntries = 8u;
  static constexpr std::size_t kDefaultEntries = 1u;
  static constexpr std::size_t kDefaultDistributedEntries = 3u;

  struct Statistics {
    std::uint64_t hits{0u};
    std::uint64_t misses{0u};
    // Footprint of the cached entries (estimate) and its peak.
    std::size_t bytes{0u};
    std::size_t peak_bytes{0u};
  };
  struct Entry;

  // Capacity from SVMP_PDE_EXTENSION_CACHE_ENTRIES, else the defaults.
  PdeVelocityExtensionCache();
  // Fixed capacity, clamped to [1, kMaxEntries].
  explicit PdeVelocityExtensionCache(std::size_t capacity);
  ~PdeVelocityExtensionCache();
  PdeVelocityExtensionCache(PdeVelocityExtensionCache&&) noexcept;
  PdeVelocityExtensionCache& operator=(PdeVelocityExtensionCache&&) noexcept;
  PdeVelocityExtensionCache(const PdeVelocityExtensionCache&) = delete;
  PdeVelocityExtensionCache& operator=(const PdeVelocityExtensionCache&) =
      delete;

  // Drops the cached entries on this rank.  The next call then refactors on
  // every rank of its communicator.
  void clear() noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  // Entries kept for replicated or distributed component solves.
  [[nodiscard]] std::size_t capacity(bool distributed_solves) const noexcept;
  [[nodiscard]] const Statistics& statistics() const noexcept;

private:
  friend struct PdeVelocityExtensionCacheAccess;
  std::vector<std::unique_ptr<Entry>> entries_;
  // 0: the defaults.
  std::size_t capacity_{0u};
  Statistics statistics_{};
};

// Diagnostics: SVMP_PDE_EXTENSION_DUMP=<prefix> makes rank 0 write the
// canonical dry-region solution of the first SVMP_PDE_EXTENSION_DUMP_CALLS
// (default 2) calls of the process to <prefix>.<call>.bin (uint64 vertex
// count, then per vertex its int64 global ID and three doubles).
//
// level_set: one value per local vertex (only its gradient is used).
// source: source_components values per local vertex.
// known: mask of the known vertices (made communicator-consistent inside).
// extended: resized to n_vertices * target_components on return; it holds
// the solution of the dry-region system.
// rows (optional): the same discrete problem as owner-local algebraic rows
// for the monolithic coupling, one per owned vertex and component:
//   known vertex:          w_c = u_c                       (source dependency),
//   dry vertex:            w_{i,c} = sum_j (-A_ij / A_ii) w_{j,c}
//                          over the vertices j sharing a dry-region cell
//                          with i (extension dependencies),
//   wall-constrained or outside the band:  w_c = 0.
// Together with the physical velocity these rows determine exactly the
// extension returned in `extended`.
// cache (optional): reuse of the dry-region factorization and rows between
// calls, keyed on `revisions` and the exact system content.  It must be
// either non-null on every rank of the communicator or null on every rank.
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
    std::vector<double>& extended,
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>* rows =
        nullptr,
    PdeVelocityExtensionCache* cache = nullptr,
    const PdeVelocityExtensionMeshRevisions& revisions = {});

// Combines one rank-local revision key per rank (in rank order) into a key
// that is identical on every rank of the communicator.
[[nodiscard]] std::uint64_t communicatorCombinedRevision(
    std::uint64_t local_key, const svmp::MeshComm& comm);

} // namespace application::core
