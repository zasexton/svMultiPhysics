/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetVelocityExtensionRefresh.h
 * @ingroup fe_level_set
 * @brief Refresh of a level-set advection velocity by the PDE extension of a
 *        physical velocity field into the dry part of an unfitted mesh.
 *
 * For every request, refreshLevelSetVelocityExtensions():
 *  - checks that the level set, the physical (source) velocity and the
 *    advection (target) velocity are fixed-order nodal Lagrange fields whose
 *    nodes are the mesh vertices (the geometry nodes, high-order nodes
 *    included): P1 on first-order cells, P2 on Triangle6, Quad9, Tetra10 or
 *    Hex27 cells, ...;
 *  - evaluates the level set and the source velocity at the mesh vertices;
 *  - forms the known set: the retained side plus every vertex of the
 *    retained interface cells, taken from the generated cut context when it
 *    is authoritative and from nodal crossings otherwise
 *    (LevelSetVelocityExtensionSeed.h);
 *  - resolves the strong wall masks of the request (homogeneous walls, and
 *    walls whose prescribed velocity the source carries);
 *  - solves the PDE extension (LevelSetPdeVelocityExtension.h) and checks
 *    that it equals the source on every known interface vertex and, for the
 *    monolithic coupling, that it does not amplify the wet speeds beyond the
 *    fixed guard;
 *  - writes the prescribed advection field, or, for the monolithic coupling
 *    (an algebraic extension unknown with a registered constraint kernel),
 *    installs the frozen rows and reprojects the extension state.
 *
 * Every step is collective over the active communicator of the FE system.
 *
 * LevelSetVelocityExtensionParticipant is the generated-state participant
 * (TimeStepping/GeneratedState.h) of the extension: it refreshes the
 * extension at every synchronization point of a solve and drops the frozen
 * rows of the monolithic requests when the generated state is invalidated.
 *
 * Moved from Application/Core/ApplicationDriver.cpp with the bodies, messages
 * and log lines unchanged.
 */

#include "Core/Types.h"
#include "LevelSet/Extension/LevelSetPdeVelocityExtension.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Mesh/Mesh.h"
#include "TimeStepping/GeneratedState.h"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
struct SystemStateView;
} // namespace systems

namespace level_set {

/// One advection-velocity extension of a level-set equation.
struct LevelSetVelocityExtensionRequest {
  struct WallConstraint {
    std::string face_name{};
    // Preserve the raw Navier--Stokes Effective_direction metadata until the
    // mesh dimension is known.  Empty (and all-zero/all-active) directions
    // have the same all-component semantics as the production NS translator
    // for a homogeneous Dirichlet condition.
    std::vector<int> effective_direction{};
    // The strong Dirichlet condition prescribes a velocity of any value
    // (time- or space-varying, from a file): the extension takes the source
    // velocity on the constrained components of the dry wall vertices
    // instead of zero (WallVelocityExtensionConstraint).
    bool prescribed_velocity{false};
  };

  std::string level_set_field_name{"level_set"};
  std::string domain_id{"free_surface"};
  std::string source_velocity_field_name{"Velocity"};
  std::string target_velocity_field_name{"LevelSetAdvectionVelocity"};
  std::string operator_tag{"level_set"};
  // pde_harmonic or pde_normal (Advection_velocity_extension_method).
  std::string extension_method{};
  bool enforce_wall_impermeability{true};
  std::vector<std::string> wall_face_names{};
  std::vector<WallConstraint> wall_constraints{};
  double isovalue{0.0};
  int requested_interface_marker{-1};
  std::optional<std::size_t> active_cut_request_index{};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
};

/// The dry-region factorization caches of the PDE extension, one per target
/// field (see PdeVelocityExtensionCache).
using PdeVelocityExtensionCaches =
    std::map<std::string, PdeVelocityExtensionCache>;

struct LevelSetVelocityExtensionRefreshOptions {
  /// False keeps the frozen rows of the monolithic requests (inside one
  /// nonlinear solve); their extension depends on nonsmooth choices.
  bool refresh_frozen_rows{true};
  PdeVelocityExtensionFactorization factorization{
      PdeVelocityExtensionFactorization::LuColamd};
  /// Write the per-request trace lines.
  bool trace{false};
};

/// The velocity components a strong Dirichlet condition with this
/// Effective_direction constrains (all of them unless the direction selects
/// a nonempty proper subset, as the Navier--Stokes translator does), for a
/// homogeneous wall and for a wall with a prescribed velocity alike.
[[nodiscard]] std::array<bool, 3> strongZeroVelocityComponentMask(
    std::span<const int> effective_direction,
    int dimension);

/// phi - isovalue, negated when the retained side is the positive one, so
/// that the retained side is nonpositive.
[[nodiscard]] std::vector<double> orientedLevelSetForVelocityExtension(
    std::span<const double> level_set_values,
    double isovalue,
    LevelSetActiveSide retained_side);

/// The generated-interface marker of @p request: the requested marker, the
/// lifecycle's stable marker when the request has keyed active-cut
/// provenance, or the marker of its active-cut request.
[[nodiscard]] std::optional<int> configuredInterfaceVelocityMarker(
    const systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request);

/// The configured marker when the cut context holds its volume or interface
/// rules.
[[nodiscard]] std::optional<int> interfaceVelocitySampleMarker(
    const systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request);

/// True when the generated interface of @p request is declared on this rank
/// (registered or expected), even with no local rules: its candidate cells
/// are then authoritative and are not replaced by nodal crossings.
[[nodiscard]] bool hasAuthoritativeInterfaceVelocityContext(
    const systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request);

/// The retained cut cells and interface parent cells of @p request, sorted.
/// Rank-local.
[[nodiscard]] std::vector<MeshIndex> interfaceVelocitySampleCandidateCells(
    const systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request);

/// Refreshes the advection velocity of every request from @p state; returns
/// true when a request was refreshed.  @p caches holds the dry-region
/// factorizations between calls.  Collective.
bool refreshLevelSetVelocityExtensions(
    systems::FESystem& system,
    const svmp::Mesh& mesh,
    const systems::SystemStateView& state,
    std::span<const LevelSetVelocityExtensionRequest> requests,
    const LevelSetVelocityExtensionRefreshOptions& options,
    PdeVelocityExtensionCaches& caches);

/**
 * @brief The generated-state participant of the velocity extension
 *        ("level_set_advection_velocity", role Extension).
 *
 * synchronize() refreshes every request.  The frozen rows of the monolithic
 * requests are refreshed, in a steady solve, only with the outer fixed
 * point, and in a transient solve where
 * refreshesFrozenLevelSetExtensionAtStateSync() says so.  invalidate() drops
 * those frozen rows (a rejected cut-topology change).  The FE system, the
 * mesh and the caches must outlive the participant.
 */
class LevelSetVelocityExtensionParticipant final
    : public timestepping::IGeneratedStateParticipant {
public:
  struct Options {
    timestepping::GeneratedStateSolve solve{
        timestepping::GeneratedStateSolve::Transient};
    /// The solve regenerates its state in an outer fixed point.
    bool outer_fixed_point{false};
    PdeVelocityExtensionFactorization factorization{
        PdeVelocityExtensionFactorization::LuColamd};
    /// Write the per-request trace lines.
    bool trace{false};
  };

  LevelSetVelocityExtensionParticipant(
      systems::FESystem& system,
      const svmp::Mesh& mesh,
      std::vector<LevelSetVelocityExtensionRequest> requests,
      Options options,
      PdeVelocityExtensionCaches& caches);

  [[nodiscard]] std::string_view name() const noexcept override
  {
    return "level_set_advection_velocity";
  }
  [[nodiscard]] timestepping::GeneratedStateRole role() const noexcept override
  {
    return timestepping::GeneratedStateRole::Extension;
  }

  void synchronize(const systems::SystemStateView& state,
                   timestepping::GeneratedStateSync& sync) override;
  void invalidate() override;

private:
  systems::FESystem* system_{nullptr};
  const svmp::Mesh* mesh_{nullptr};
  std::vector<LevelSetVelocityExtensionRequest> requests_{};
  Options options_{};
  PdeVelocityExtensionCaches* caches_{nullptr};
};

} // namespace level_set
} // namespace FE
} // namespace svmp
