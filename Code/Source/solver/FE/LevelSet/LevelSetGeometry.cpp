/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetGeometry.h"

#include "Systems/FESystem.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

namespace svmp {
namespace FE {
namespace level_set {
namespace {

// The cut-volume request functions below were moved verbatim from
// Application/Core/LevelSetCutConfiguration.cpp (free-surface refactor, C2a).

svmp::FE::geometry::CutIntegrationSide cutIntegrationSide(
    LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

void mixRequestPolicyHash(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= 1099511628211ull;
}

void mixRequestPolicyHash(std::uint64_t& h, const std::string& value) noexcept
{
  for (const char c : value) {
    mixRequestPolicyHash(h, static_cast<unsigned char>(c));
  }
  mixRequestPolicyHash(h, 0xffu);
}

void mixRequestPolicyHash(std::uint64_t& h, double value) noexcept
{
  std::uint64_t bits = 0u;
  static_assert(sizeof(value) <= sizeof(bits));
  std::memcpy(&bits, &value, sizeof(value));
  mixRequestPolicyHash(h, bits);
}

template <typename Options>
void projectCommonCutOptions(
    Options& options, const ActiveCutVolumeRequest& request)
{
  static_assert(
      std::is_same_v<Options,
                     svmp::FE::level_set::LevelSetGeneratedInterfaceOptions> ||
      std::is_same_v<Options,
                     svmp::FE::level_set::LevelSetVolumeOptions>);
  options.geometry_mode = request.geometry_mode;
  options.implicit_cut_quadrature_backend = request.implicit_cut_backend;
  options.implicit_cut_fallback_policy =
      request.implicit_cut_fallback_policy;
  options.geometry_tangent_policy = request.geometry_tangent_policy;
  options.implicit_cut_root_tolerance =
      static_cast<svmp::FE::Real>(request.implicit_cut_root_tolerance);
  options.implicit_cut_root_coordinate_tolerance =
      static_cast<svmp::FE::Real>(
          request.implicit_cut_root_coordinate_tolerance);
  options.implicit_cut_root_max_iterations =
      request.implicit_cut_root_max_iterations;
  options.implicit_cut_max_subdivision_depth =
      request.implicit_cut_max_subdivision_depth;
  options.affected_cell_neighborhood_layers =
      request.affected_cell_neighborhood_layers;
  options.allow_corner_linearized_geometry =
      request.allow_corner_linearized_geometry;
  options.require_production_qualified_implicit_cut_backend =
      request.require_production_qualified_implicit_cut_backend;
}

} // namespace

svmp::FE::level_set::LevelSetGeneratedInterfaceOptions
generatedInterfaceOptionsForActiveCut(
    const ActiveCutVolumeRequest& request, int mesh_dimension)
{
  svmp::FE::level_set::LevelSetGeneratedInterfaceOptions options{};
  options.level_set_field_name = request.level_set_field_name;
  options.domain_id = request.domain_id;
  options.requested_interface_marker = request.requested_interface_marker;
  options.isovalue = static_cast<svmp::FE::Real>(request.isovalue);
  if (request.quadrature_order.has_value()) {
    options.quadrature_order = *request.quadrature_order;
  }
  if (request.interface_quadrature_order.has_value()) {
    options.interface_quadrature_order =
        *request.interface_quadrature_order;
  }
  if (request.volume_quadrature_order.has_value()) {
    options.volume_quadrature_order = *request.volume_quadrature_order;
  }
  if (!request.interface_quadrature_order.has_value() &&
      mesh_dimension == 2 &&
      options.interface_quadrature_order < 0) {
    options.interface_quadrature_order = options.volume_quadrature_order;
  }
  projectCommonCutOptions(options, request);
  options.aligned_zero_interface_parent_side =
      cutIntegrationSide(request.active_side);
  return options;
}

svmp::FE::level_set::LevelSetVolumeOptions
volumeOptionsForCutMaintenance(
    const std::optional<ActiveCutVolumeRequest>& request,
    const std::string& maintenance_field_name,
    double maintenance_isovalue)
{
  svmp::FE::level_set::LevelSetVolumeOptions options{};
  options.isovalue = static_cast<svmp::FE::Real>(maintenance_isovalue);
  if (!request.has_value()) {
    return options;
  }
  if (request->geometry_mode !=
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::
          HighOrderImplicit) {
    return options;
  }

  options.use_generated_interface_quadrature = true;
  options.level_set_field_name = maintenance_field_name;
  options.generated_domain_id =
      request->domain_id.empty()
          ? std::string{"volume_correction"}
          : request->domain_id + "_volume_correction";
  options.requested_interface_marker = request->requested_interface_marker;
  options.quadrature_order = request->quadrature_order;
  options.interface_quadrature_order = request->interface_quadrature_order;
  options.volume_quadrature_order = request->volume_quadrature_order;
  projectCommonCutOptions(options, *request);
  return options;
}

bool hasHighOrderGeneratedInterfaceGeometry(
    const std::vector<ActiveCutVolumeRequest>& requests) noexcept
{
  using Mode = svmp::FE::level_set::GeneratedInterfaceGeometryMode;
  return std::any_of(
      requests.begin(),
      requests.end(),
      [](const ActiveCutVolumeRequest& request) {
        return request.geometry_mode == Mode::HighOrderImplicit;
      });
}

std::uint64_t activeCutVolumeRequestPolicyKey(
    const std::vector<ActiveCutVolumeRequest>& requests) noexcept
{
  std::uint64_t h = 1469598103934665603ull;
  mixRequestPolicyHash(h, static_cast<std::uint64_t>(requests.size()));
  for (const auto& request : requests) {
    mixRequestPolicyHash(h, request.level_set_field_name);
    mixRequestPolicyHash(h, request.domain_id);
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.requested_interface_marker));
    mixRequestPolicyHash(h, request.isovalue);
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.quadrature_order.has_value()));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.quadrature_order.value_or(-1)));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.interface_quadrature_order.has_value()));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.interface_quadrature_order.value_or(-1)));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.volume_quadrature_order.has_value()));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.volume_quadrature_order.value_or(-1)));
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.geometry_mode));
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.implicit_cut_backend));
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.implicit_cut_fallback_policy));
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.geometry_tangent_policy));
    mixRequestPolicyHash(h, request.implicit_cut_root_tolerance);
    mixRequestPolicyHash(h, request.implicit_cut_root_coordinate_tolerance);
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.implicit_cut_root_max_iterations));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.implicit_cut_max_subdivision_depth));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.affected_cell_neighborhood_layers));
    mixRequestPolicyHash(h, static_cast<std::uint64_t>(request.active_side));
    mixRequestPolicyHash(
        h, static_cast<std::uint64_t>(request.volume_retention));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.allow_corner_linearized_geometry));
    mixRequestPolicyHash(
        h,
        static_cast<std::uint64_t>(
            request.require_production_qualified_implicit_cut_backend));
  }
  return h;
}

std::optional<int> resolvedActiveCutVolumeInterfaceMarker(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request)
{
  if (request.requested_interface_marker >= 0) {
    return request.requested_interface_marker;
  }

  const auto field_id = system.findFieldByName(request.level_set_field_name);
  if (field_id == svmp::FE::INVALID_FIELD_ID) {
    return std::nullopt;
  }

  svmp::FE::interfaces::GeneratedInterfaceMarkerKey key{};
  key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(field_id);
  key.domain_id = request.domain_id;
  key.isovalue = static_cast<svmp::FE::Real>(request.isovalue);
  key.requested_marker = request.requested_interface_marker;
  return svmp::FE::interfaces::stableGeneratedInterfaceMarker(key);
}

} // namespace level_set
} // namespace FE
} // namespace svmp
