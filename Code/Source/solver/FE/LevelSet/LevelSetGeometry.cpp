/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetGeometry.h"

#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace level_set {

// The cut-volume request functions below were moved verbatim from
// Application/Core/LevelSetCutConfiguration.cpp (free-surface refactor, C2a).

svmp::FE::geometry::CutIntegrationSide cutIntegrationSide(
    LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

namespace {

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

[[nodiscard]] geometry::CutIntegrationSide cutIntegrationSideOf(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Negative ? geometry::CutIntegrationSide::Negative
                                                  : geometry::CutIntegrationSide::Positive;
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

// Moved from Application/Core/ApplicationDriver.cpp (lines 16390-16400 at
// 7d9ddc8a); the driver passed its maintenance request, whose cut-volume
// request is now the argument.
std::optional<int> generatedCutContextMarkerForMaintenance(
    const svmp::FE::systems::FESystem& system,
    const std::optional<ActiveCutVolumeRequest>& volume_cut_request)
{
  if (!volume_cut_request.has_value()) {
    return std::nullopt;
  }
  const auto& cut_request = *volume_cut_request;
  return resolvedActiveCutVolumeInterfaceMarker(
      system, cut_request);
}

// ---------------------------------------------------------------------------
// Typed domains of a level-set geometry
// ---------------------------------------------------------------------------

forms::FormExpr outwardInterfaceNormal(forms::CutVolumeSide side)
{
    auto n = forms::FormExpr::normal();
    if (side == forms::CutVolumeSide::Positive) {
        n = -n;
    }
    return n;
}

LevelSetGeometry LevelSetGeometry::bind(const systems::FESystem& system,
                                        LevelSetGeometryDeclaration declaration)
{
    FieldId field = declaration.level_set_field;
    if (field == INVALID_FIELD_ID) {
        field = system.findFieldByName(declaration.level_set_field_name);
        if (field == INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "LevelSetGeometry::bind: unknown level-set field '" +
                declaration.level_set_field_name + "'");
        }
    }
    const auto& record = system.fieldRecord(field);
    if (record.components != 1 || !record.space || record.space->value_dimension() != 1) {
        throw std::invalid_argument(
            "LevelSetGeometry::bind: level-set field '" + record.name + "' must be scalar");
    }

    int marker = declaration.interface_marker;
    if (marker < 0) {
        if (declaration.domain_id.empty()) {
            throw std::invalid_argument(
                "LevelSetGeometry::bind: a generated interface marker for level-set field '" +
                record.name + "' requires a non-empty domain_id");
        }
        interfaces::GeneratedInterfaceMarkerKey key{};
        key.source = interfaces::LevelSetInterfaceSource::fromField(field);
        key.domain_id = declaration.domain_id;
        key.isovalue = declaration.isovalue;
        key.requested_marker = declaration.interface_marker;
        marker = interfaces::stableGeneratedInterfaceMarker(key);
    }
    return LevelSetGeometry(field, std::move(declaration.domain_id), declaration.isovalue, marker);
}

forms::InterfaceDomain LevelSetGeometry::interface() const
{
    return forms::InterfaceDomain(interface_marker_);
}

CutSide LevelSetGeometry::side(forms::CutVolumeSide side) const
{
    return CutSide(*this, side);
}

interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey LevelSetGeometry::contactLineKey(
    int wall_boundary_marker,
    int requested_marker) const
{
    interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = interfaces::LevelSetInterfaceSource::fromField(level_set_field_);
    key.domain_id = domain_id_;
    key.isovalue = isovalue_;
    key.interface_marker = interface_marker_;
    key.boundary_marker = wall_boundary_marker;
    key.requested_marker = requested_marker;
    return key;
}

forms::ContactLineDomain LevelSetGeometry::contactLine(int wall_boundary_marker,
                                                       int requested_marker) const
{
    return forms::ContactLineDomain(
        interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
            contactLineKey(wall_boundary_marker, requested_marker)),
        wall_boundary_marker);
}

forms::VolumeDomain CutSide::volume() const
{
    return forms::VolumeDomain::cut(geometry_.interfaceMarker(), side_);
}

forms::FormExpr CutSide::outwardNormal() const
{
    return outwardInterfaceNormal(side_);
}

interfaces::GeneratedActiveBoundaryMarkerKey CutSide::boundaryKey(int physical_boundary_marker) const
{
    interfaces::GeneratedActiveBoundaryMarkerKey key;
    key.source = interfaces::LevelSetInterfaceSource::fromField(geometry_.levelSetField());
    key.domain_id = geometry_.domainId();
    key.isovalue = geometry_.isovalue();
    key.interface_marker = geometry_.interfaceMarker();
    key.boundary_marker = physical_boundary_marker;
    key.side = cutIntegrationSideOf(side_);
    return key;
}

forms::BoundaryDomain CutSide::boundary(int physical_boundary_marker) const
{
    return forms::BoundaryDomain::generatedActiveSubset(
        physical_boundary_marker,
        interfaces::stableGeneratedActiveBoundaryMarker(boundaryKey(physical_boundary_marker)));
}

forms::FacetDomain CutSide::cutFacets() const
{
    return forms::FacetDomain::cutAdjacent(geometry_.interfaceMarker(), side_);
}

FieldId resolveScalarLevelSetField(
    const systems::FESystem& system,
    const std::string& name,
    const std::string& unknown_field_prefix,
    const std::string& owner)
{
    const auto phi_id = system.findFieldByName(name);
    if (phi_id == INVALID_FIELD_ID) {
        throw std::invalid_argument(unknown_field_prefix + name + "'");
    }

    const auto& rec = system.fieldRecord(phi_id);
    if (rec.components != 1 || !rec.space || rec.space->value_dimension() != 1) {
        throw std::invalid_argument(
            owner + ": level-set field '" + name + "' must be scalar");
    }
    return phi_id;
}

std::optional<int> resolveGeneratedActiveBoundaryMarker(
    int physical_boundary_marker,
    std::size_t owner_count,
    const std::function<std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>(
        std::size_t owner)>& key_of,
    const std::string& owner)
{
    std::optional<int> marker;
    std::string owner_key;
    for (std::size_t index = 0u; index < owner_count; ++index) {
        const auto key = key_of(index);
        if (!key.has_value()) {
            continue;
        }
        const int candidate =
            interfaces::stableGeneratedActiveBoundaryMarker(*key);
        if (!marker.has_value()) {
            marker = candidate;
            owner_key = key->stableKey();
            continue;
        }
        if (*marker != candidate || owner_key != key->stableKey()) {
            throw std::invalid_argument(
                owner + ": physical boundary marker " +
                std::to_string(physical_boundary_marker) +
                " is covered by multiple ambiguous unfitted active domains; select one explicit active-domain owner");
        }
    }
    return marker;
}

std::function<std::optional<int>(int)> wettedBoundaryResolver(
    systems::FESystem& system,
    std::function<std::optional<int>(int)> resolve)
{
    return [&system, resolve = std::move(resolve)](int marker) -> std::optional<int> {
        const auto generated = resolve(marker);
        if (generated.has_value()) {
            system.registerGeneratedEmbeddedInterfaceMarker(*generated);
        }
        return generated;
    };
}

} // namespace level_set
} // namespace FE
} // namespace svmp
