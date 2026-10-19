/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSETGEOMETRY_H
#define SVMP_FE_LEVELSET_LEVELSETGEOMETRY_H

/**
 * @file LevelSetGeometry.h
 * @brief The cut-volume request of a level-set geometry
 *
 * The cut-volume request of the Application input (ActiveCutVolumeRequest)
 * and its pure functions, moved from Application/Core/LevelSetCutConfiguration;
 * the XML parsing stays there.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetVolume.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
class FESystem;
}

namespace level_set {

// ---------------------------------------------------------------------------
// Cut-volume request (resolved Application input)
// ---------------------------------------------------------------------------

enum class LevelSetActiveSide {
  Negative,
  Positive,
};

enum class ActiveCutVolumeRequestOrigin {
  Equation,
  FreeSurfaceBoundary,
  MaterialInterface,
};

enum class ActiveCutVolumeRetention {
  ActiveOnly,
  ActiveAndInactive,
};

struct ActiveCutVolumeRequest {
  std::string level_set_field_name{"level_set"};
  std::string domain_id{"free_surface"};
  std::string equation_type{};
  ActiveCutVolumeRequestOrigin origin{ActiveCutVolumeRequestOrigin::Equation};
  int requested_interface_marker{-1};
  double isovalue{0.0};
  std::optional<int> quadrature_order{};
  std::optional<int> interface_quadrature_order{};
  std::optional<int> volume_quadrature_order{};
  GeneratedInterfaceGeometryMode geometry_mode{
      GeneratedInterfaceGeometryMode::LinearCorner};
  ImplicitCutQuadratureBackend implicit_cut_backend{
      ImplicitCutQuadratureBackend::LinearCorner};
  ImplicitCutFallbackPolicy implicit_cut_fallback_policy{
      ImplicitCutFallbackPolicy::Fail};
  GeometryTangentPolicy geometry_tangent_policy{
      GeometryTangentPolicy::RefreshedFrozenQuadrature};
  // Parsing provenance, used only to resolve or reject two declarations of
  // the same generated domain. It is deliberately not part of the generated
  // cut policy key: equal resolved policies have equal numerical semantics.
  bool geometry_tangent_policy_explicit{false};
  double implicit_cut_root_tolerance{1.0e-10};
  double implicit_cut_root_coordinate_tolerance{1.0e-12};
  int implicit_cut_root_max_iterations{48};
  int implicit_cut_max_subdivision_depth{16};
  int affected_cell_neighborhood_layers{0};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
  ActiveCutVolumeRetention volume_retention{
      ActiveCutVolumeRetention::ActiveOnly};
  bool allow_corner_linearized_geometry{false};
  bool require_production_qualified_implicit_cut_backend{false};
};

[[nodiscard]] LevelSetGeneratedInterfaceOptions
generatedInterfaceOptionsForActiveCut(
    const ActiveCutVolumeRequest& request, int mesh_dimension);

[[nodiscard]] LevelSetVolumeOptions
volumeOptionsForCutMaintenance(
    const std::optional<ActiveCutVolumeRequest>& request,
    const std::string& maintenance_field_name,
    double maintenance_isovalue);

[[nodiscard]] bool hasHighOrderGeneratedInterfaceGeometry(
    const std::vector<ActiveCutVolumeRequest>& requests) noexcept;

[[nodiscard]] std::uint64_t activeCutVolumeRequestPolicyKey(
    const std::vector<ActiveCutVolumeRequest>& requests) noexcept;

[[nodiscard]] std::optional<int> resolvedActiveCutVolumeInterfaceMarker(
    const systems::FESystem& system,
    const ActiveCutVolumeRequest& request);

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSETGEOMETRY_H
