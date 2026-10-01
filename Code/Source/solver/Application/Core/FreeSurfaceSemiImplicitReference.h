/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

// Reference velocity u_ref of the semi-implicit (lagged normal-increment)
// capillary term of an unfitted free surface
// (Surface_tension_semi_implicit=NormalIncrement; Physics/Docs/
// NavierStokesFreeSurface.md and the design note
// Documentation/free_surface_semi_implicit_surface_tension_design.md).
//
// The Navier--Stokes module registers u_ref as prescribed data in the
// velocity space.  The application copies the current velocity unknown into
// it at every generated-state refresh (the projected outer fixed-point,
// endpoint and restored synchronization points, and before each physical
// solve), so the term is identically zero in every freshly refreshed
// residual.  The copy is a coefficient copy between two fields that share
// one DOF map; no interpolation is involved.
//
// u_ref is the velocity of the iterate that generated the frozen interface
// only when the level set is advected by that velocity on the interface:
// the coupled fluid velocity, or a PDE extension of it (which equals the
// fluid velocity on every vertex of the retained interface cells).  Other
// transports fail closed.

#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "Mesh/Core/MeshComm.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
struct SystemStateView;
} // namespace svmp::FE::systems

namespace application::core {

// How one level_set equation advects its level set.
struct FreeSurfaceSemiImplicitTransport {
  std::string level_set_field_name{};
  svmp::FE::level_set::LevelSetVelocitySource velocity_source{
      svmp::FE::level_set::LevelSetVelocitySource::CoupledField};
  // Field that advects the level set (the fluid velocity for coupled_field,
  // the extension target for prescribed_data).
  std::string advection_velocity_field_name{};
  // prescribed_data only: the method that fills the advection velocity from
  // the fluid velocity (empty: none) and its source field.
  std::string extension_method{};
  std::string extension_source_velocity_field_name{};
};

struct FreeSurfaceSemiImplicitReferenceBinding {
  svmp::FE::FieldId reference_field{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::FieldId velocity_field{svmp::FE::INVALID_FIELD_ID};
  std::string velocity_field_name{};
  svmp::FE::GlobalIndex reference_coefficient_count{0};
  // Field-local DOFs referenced by the rank-local cells, and the matching
  // indices of the full FE state vector.
  std::vector<svmp::FE::GlobalIndex> field_dofs{};
  std::vector<svmp::FE::GlobalIndex> state_dofs{};
};

[[nodiscard]] bool hasFreeSurfaceSemiImplicitReference(
    const svmp::FE::systems::FESystem& system);

// Returns nullopt when no free surface requested the semi-implicit term.
// Otherwise validates, on every rank of `comm`, that the solve is transient
// with the generated-state outer fixed point, that the reference field
// shares the DOF map of the single declared free-surface velocity, and that
// every declared free-surface level set is transported by that velocity or
// by a PDE extension of it.  Throws std::runtime_error otherwise.
[[nodiscard]] std::optional<FreeSurfaceSemiImplicitReferenceBinding>
bindFreeSurfaceSemiImplicitReference(
    const svmp::FE::systems::FESystem& system,
    std::span<const FreeSurfaceSemiImplicitTransport> transports,
    bool transient_solve,
    bool outer_fixed_point,
    const svmp::MeshComm& comm);

// u_ref := u (velocity block of `state`).  Rank-local; no communication.
void refreshFreeSurfaceSemiImplicitReference(
    svmp::FE::systems::FESystem& system,
    const FreeSurfaceSemiImplicitReferenceBinding& binding,
    const svmp::FE::systems::SystemStateView& state);

} // namespace application::core
