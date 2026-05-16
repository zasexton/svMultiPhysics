#pragma once

#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"

#include <optional>
#include <string>
#include <vector>

class Parameters;

namespace application {
namespace core {

enum class LevelSetActiveSide {
  Negative,
  Positive,
};

struct ActiveCutVolumeRequest {
  std::string level_set_field_name{"level_set"};
  std::string domain_id{"free_surface"};
  int requested_interface_marker{-1};
  double isovalue{0.0};
  std::optional<int> quadrature_order{};
  std::optional<int> interface_quadrature_order{};
  std::optional<int> volume_quadrature_order{};
  svmp::FE::level_set::GeneratedInterfaceGeometryMode geometry_mode{
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner};
  svmp::FE::level_set::ImplicitCutQuadratureBackend implicit_cut_backend{
      svmp::FE::level_set::ImplicitCutQuadratureBackend::LinearCorner};
  svmp::FE::level_set::ImplicitCutFallbackPolicy implicit_cut_fallback_policy{
      svmp::FE::level_set::ImplicitCutFallbackPolicy::Fail};
  svmp::FE::level_set::GeometryTangentPolicy geometry_tangent_policy{
      svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature};
  double implicit_cut_root_tolerance{1.0e-10};
  int implicit_cut_max_subdivision_depth{16};
  LevelSetActiveSide active_side{LevelSetActiveSide::Negative};
  bool allow_corner_linearized_geometry{false};
};

[[nodiscard]] svmp::FE::level_set::GeneratedInterfaceGeometryMode
parseGeneratedInterfaceGeometryMode(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::ImplicitCutQuadratureBackend
parseImplicitCutQuadratureBackend(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::ImplicitCutFallbackPolicy
parseImplicitCutFallbackPolicy(const std::string& raw);

[[nodiscard]] svmp::FE::level_set::GeometryTangentPolicy
parseGeometryTangentPolicy(const std::string& raw);

[[nodiscard]] std::vector<ActiveCutVolumeRequest>
activeCutVolumeRequests(const Parameters& params);

[[nodiscard]] bool hasHighOrderGeneratedInterfaceGeometry(
    const std::vector<ActiveCutVolumeRequest>& requests) noexcept;

} // namespace core
} // namespace application
