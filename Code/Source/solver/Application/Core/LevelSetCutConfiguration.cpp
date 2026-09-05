#include "Application/Core/LevelSetCutConfiguration.h"

#include "Application/Core/OopMpiLog.h"
#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace application {
namespace core {
namespace {

std::string trimCopy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string lowerCopy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool parseBoolRelaxed(const std::string& raw)
{
  const auto v = lowerCopy(trimCopy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

std::string normalizedToken(std::string value)
{
  value = lowerCopy(trimCopy(std::move(value)));
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](unsigned char c) {
                               return c == '_' || c == '-' || std::isspace(c);
                             }),
              value.end());
  return value;
}

const char* activeSideName(LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? "LevelSetNegative"
             : "LevelSetPositive";
}

svmp::FE::geometry::CutIntegrationSide cutIntegrationSide(
    LevelSetActiveSide side) noexcept
{
  return side == LevelSetActiveSide::Negative
             ? svmp::FE::geometry::CutIntegrationSide::Negative
             : svmp::FE::geometry::CutIntegrationSide::Positive;
}

const char* fieldSourceKindName(
    svmp::FE::systems::FieldSourceKind kind) noexcept
{
  switch (kind) {
    case svmp::FE::systems::FieldSourceKind::Unknown:
      return "Unknown";
    case svmp::FE::systems::FieldSourceKind::PrescribedData:
      return "PrescribedData";
    case svmp::FE::systems::FieldSourceKind::DerivedFromUnknown:
      return "DerivedFromUnknown";
  }
  return "Unknown";
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

std::optional<std::string> firstDefinedParameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  for (const char* key : keys) {
    const auto it = params.find(key);
    if (it != params.end() && !trimCopy(it->second).empty()) {
      return it->second;
    }
  }
  return std::nullopt;
}

std::optional<double> firstDefinedDoubleParameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = firstDefinedParameter(params, keys)) {
    return std::stod(*value);
  }
  return std::nullopt;
}

std::optional<int> firstDefinedIntParameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = firstDefinedParameter(params, keys)) {
    return std::stoi(*value);
  }
  return std::nullopt;
}

std::optional<bool> firstDefinedBoolParameter(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys)
{
  if (const auto value = firstDefinedParameter(params, keys)) {
    return parseBoolRelaxed(*value);
  }
  return std::nullopt;
}

bool activeCutDomainIsExplicitlyEnabled(
    const std::map<std::string, std::string>& params)
{
  const auto enabled =
      firstDefinedBoolParameter(
          params,
          {"Enable_level_set_cut_domain",
           "EnableLevelSetCutDomain",
           "Level_set_cut_domain",
           "LevelSetCutDomain",
           "Generated_level_set_cut_domain",
           "GeneratedLevelSetCutDomain",
           "Cut_domain",
           "CutDomain"});
  if (enabled.has_value()) {
    return *enabled;
  }

  const auto kind =
      firstDefinedParameter(
          params,
          {"Cut_domain_type",
           "CutDomainType",
           "Generated_cut_domain_type",
           "GeneratedCutDomainType"});
  if (!kind.has_value()) {
    return false;
  }
  const auto token = normalizedToken(*kind);
  return token == "levelset" || token == "generatedlevelset" ||
         token == "implicitcut" || token == "generatedinterface";
}

bool requestMatchesExceptGeometryTangentPolicy(
    const ActiveCutVolumeRequest& a,
    const ActiveCutVolumeRequest& b) noexcept
{
  return a.level_set_field_name == b.level_set_field_name &&
         a.domain_id == b.domain_id &&
         a.requested_interface_marker == b.requested_interface_marker &&
         a.isovalue == b.isovalue &&
         a.quadrature_order == b.quadrature_order &&
         a.interface_quadrature_order == b.interface_quadrature_order &&
         a.volume_quadrature_order == b.volume_quadrature_order &&
         a.geometry_mode == b.geometry_mode &&
         a.implicit_cut_backend == b.implicit_cut_backend &&
         a.implicit_cut_fallback_policy == b.implicit_cut_fallback_policy &&
         a.implicit_cut_root_tolerance == b.implicit_cut_root_tolerance &&
         a.implicit_cut_root_coordinate_tolerance ==
             b.implicit_cut_root_coordinate_tolerance &&
         a.implicit_cut_root_max_iterations ==
             b.implicit_cut_root_max_iterations &&
         a.implicit_cut_max_subdivision_depth ==
             b.implicit_cut_max_subdivision_depth &&
         a.affected_cell_neighborhood_layers ==
             b.affected_cell_neighborhood_layers &&
         a.active_side == b.active_side &&
         a.allow_corner_linearized_geometry ==
             b.allow_corner_linearized_geometry &&
         a.require_production_qualified_implicit_cut_backend ==
             b.require_production_qualified_implicit_cut_backend;
}

bool requestMatches(const ActiveCutVolumeRequest& a,
                    const ActiveCutVolumeRequest& b) noexcept
{
  return requestMatchesExceptGeometryTangentPolicy(a, b) &&
         a.geometry_tangent_policy == b.geometry_tangent_policy;
}

void mergeRequestRetention(ActiveCutVolumeRequest& target,
                           const ActiveCutVolumeRequest& source) noexcept
{
  if (source.volume_retention ==
      ActiveCutVolumeRetention::ActiveAndInactive) {
    target.volume_retention =
        ActiveCutVolumeRetention::ActiveAndInactive;
  }
  target.geometry_tangent_policy_explicit =
      target.geometry_tangent_policy_explicit ||
      source.geometry_tangent_policy_explicit;
}

void appendUniqueRequest(std::vector<ActiveCutVolumeRequest>& requests,
                         ActiveCutVolumeRequest request)
{
  const auto existing =
      std::find_if(
          requests.begin(),
          requests.end(),
          [&](const ActiveCutVolumeRequest& candidate) {
            return requestMatches(candidate, request);
          });
  if (existing != requests.end()) {
    mergeRequestRetention(*existing, request);
    return;
  }

  // A fluid equation-level cut declaration and its unfitted free-surface BC
  // commonly name the same generated domain. The generic equation default is
  // differentiated LinearCorner geometry, whereas the SurfaceStress default
  // must be refreshed-frozen because complete shape tangents are unsupported.
  // Resolve that one default/default (or default/explicit-frozen) pairing to
  // the free-surface policy. Never silently override an explicit equation
  // policy or two genuinely conflicting declarations.
  const auto same_domain_different_tangent =
      std::find_if(
          requests.begin(),
          requests.end(),
          [&](const ActiveCutVolumeRequest& candidate) {
            return requestMatchesExceptGeometryTangentPolicy(
                       candidate, request) &&
                   candidate.geometry_tangent_policy !=
                       request.geometry_tangent_policy;
          });
  if (same_domain_different_tangent != requests.end()) {
    using Policy = svmp::FE::level_set::GeometryTangentPolicy;
    const auto is_implicit_equation_differentiated =
        [](const ActiveCutVolumeRequest& candidate) {
          return candidate.origin ==
                     ActiveCutVolumeRequestOrigin::Equation &&
                 !candidate.geometry_tangent_policy_explicit &&
                 candidate.geometry_tangent_policy ==
                     Policy::DifferentiatedQuadrature;
        };
    const auto is_free_surface_frozen =
        [](const ActiveCutVolumeRequest& candidate) {
          return candidate.origin ==
                     ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary &&
                 candidate.geometry_tangent_policy ==
                     Policy::RefreshedFrozenQuadrature;
        };
    const bool existing_equation_then_free_surface =
        is_implicit_equation_differentiated(
            *same_domain_different_tangent) &&
        is_free_surface_frozen(request);
    const bool existing_free_surface_then_equation =
        is_free_surface_frozen(*same_domain_different_tangent) &&
        is_implicit_equation_differentiated(request);
    if (existing_equation_then_free_surface ||
        existing_free_surface_then_equation) {
      if (existing_equation_then_free_surface) {
        same_domain_different_tangent->geometry_tangent_policy =
            request.geometry_tangent_policy;
        same_domain_different_tangent->geometry_tangent_policy_explicit =
            request.geometry_tangent_policy_explicit;
      }
      mergeRequestRetention(*same_domain_different_tangent, request);
      return;
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conflicting Geometry_tangent_policy "
        "declarations target the same generated level-set domain. Use one "
        "resolved policy for the equation cut domain and free-surface BC.");
  }
  requests.push_back(std::move(request));
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

std::optional<ActiveCutVolumeRequest> activeCutVolumeRequestFromParameters(
    const std::map<std::string, std::string>& cut_params,
    bool require_active_domain,
    bool default_linear_corner_to_differentiated)
{
  const auto active_domain =
      firstDefinedParameter(cut_params,
                            {"Active_domain",
                             "ActiveDomain",
                             "Free_surface_active_domain",
                             "FreeSurfaceActiveDomain"});
  if (!active_domain.has_value()) {
    if (require_active_domain) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set cut-domain request requires Active_domain.");
    }
    return std::nullopt;
  }
  const auto active_token = normalizedToken(*active_domain);
  if (active_token.empty() || active_token == "none" ||
      active_token == "off" ||
      active_token == "disabled" || active_token == "inactive") {
    return std::nullopt;
  }

  const auto method =
      firstDefinedParameter(cut_params,
                            {"Active_domain_method",
                             "ActiveDomainMethod",
                             "Free_surface_active_domain_method",
                             "FreeSurfaceActiveDomainMethod"});
  if (method.has_value()) {
    const auto method_token = normalizedToken(*method);
    if (method_token == "smoothedindicator" ||
        method_token == "smoothindicator" ||
        method_token == "indicator") {
      return std::nullopt;
    }
    if (!method_token.empty() && method_token != "cutvolume" &&
        method_token != "cutcellvolume" &&
        method_token != "exactcutvolume") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Unknown Active_domain_method '" +
          trimCopy(*method) +
          "'. Supported values: CutVolume or SmoothedIndicator.");
    }
  }

  ActiveCutVolumeRequest request{};
  if (active_token == "levelsetnegative" ||
      active_token == "negative" ||
      active_token == "negativelevelset" ||
      active_token == "phinegative") {
    request.active_side = LevelSetActiveSide::Negative;
  } else if (active_token == "levelsetpositive" ||
             active_token == "positive" ||
             active_token == "positivelevelset" ||
             active_token == "phipositive") {
    request.active_side = LevelSetActiveSide::Positive;
  } else {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Unknown Active_domain '" +
        trimCopy(*active_domain) +
        "'. Supported values: None, LevelSetNegative, or LevelSetPositive.");
  }
  if (const auto field =
          firstDefinedParameter(cut_params, {"Level_set_field_name",
                                            "Level_set_field",
                                            "LevelSetFieldName",
                                            "LevelSetField"})) {
    request.level_set_field_name = trimCopy(*field);
  }
  if (const auto domain =
          firstDefinedParameter(cut_params, {"Generated_interface_domain_id",
                                            "GeneratedInterfaceDomainId",
                                            "Interface_domain_id",
                                            "InterfaceDomainId"})) {
    request.domain_id = trimCopy(*domain);
  }
  if (const auto marker =
          firstDefinedIntParameter(cut_params, {"Interface_marker",
                                               "InterfaceMarker"})) {
    request.requested_interface_marker = *marker;
  }
  if (const auto isovalue =
          firstDefinedDoubleParameter(cut_params, {"Level_set_isovalue",
                                                  "LevelSetIsovalue",
                                                  "Interface_isovalue",
                                                  "InterfaceIsovalue"})) {
    request.isovalue = *isovalue;
  }
  if (const auto quadrature_order =
          firstDefinedIntParameter(
              cut_params,
              {"Generated_interface_quadrature_order",
               "GeneratedInterfaceQuadratureOrder",
               "Cut_quadrature_order",
               "CutQuadratureOrder",
               "Level_set_cut_quadrature_order",
               "LevelSetCutQuadratureOrder"})) {
    request.quadrature_order = *quadrature_order;
  }
  if (const auto interface_quadrature_order =
          firstDefinedIntParameter(
              cut_params,
              {"Interface_quadrature_order",
               "InterfaceQuadratureOrder",
               "Generated_interface_surface_quadrature_order",
               "GeneratedInterfaceSurfaceQuadratureOrder",
               "Cut_interface_quadrature_order",
               "CutInterfaceQuadratureOrder"})) {
    request.interface_quadrature_order = *interface_quadrature_order;
  }
  if (const auto volume_quadrature_order =
          firstDefinedIntParameter(
              cut_params,
              {"Volume_quadrature_order",
               "VolumeQuadratureOrder",
               "Generated_cut_volume_quadrature_order",
               "GeneratedCutVolumeQuadratureOrder",
               "Cut_volume_quadrature_order",
               "CutVolumeQuadratureOrder"})) {
    request.volume_quadrature_order = *volume_quadrature_order;
  }
  if (const auto geometry_mode =
          firstDefinedParameter(
              cut_params,
              {"Generated_interface_geometry",
               "GeneratedInterfaceGeometry",
               "Implicit_geometry_mode",
               "ImplicitGeometryMode",
               "Generated_interface_geometry_mode",
               "GeneratedInterfaceGeometryMode"})) {
    request.geometry_mode =
        parseGeneratedInterfaceGeometryMode(*geometry_mode);
  }
  if (const auto backend =
          firstDefinedParameter(
              cut_params,
              {"Implicit_cut_quadrature_backend",
               "ImplicitCutQuadratureBackend",
               "Generated_interface_quadrature_backend",
               "GeneratedInterfaceQuadratureBackend"})) {
    request.implicit_cut_backend =
        parseImplicitCutQuadratureBackend(*backend);
  }
  if (const auto fallback_policy =
          firstDefinedParameter(
              cut_params,
              {"Implicit_cut_fallback_policy",
               "ImplicitCutFallbackPolicy",
               "Implicit_cut_quadrature_fallback",
               "ImplicitCutQuadratureFallback"})) {
    request.implicit_cut_fallback_policy =
        parseImplicitCutFallbackPolicy(*fallback_policy);
  }
  const auto tangent_policy =
      firstDefinedParameter(
          cut_params,
          {"Geometry_tangent_policy",
           "GeometryTangentPolicy",
           "Generated_interface_geometry_tangent_policy",
           "GeneratedInterfaceGeometryTangentPolicy",
           "Implicit_geometry_tangent_policy",
           "ImplicitGeometryTangentPolicy"});
  if (tangent_policy.has_value()) {
    request.geometry_tangent_policy =
        parseGeometryTangentPolicy(*tangent_policy);
    request.geometry_tangent_policy_explicit = true;
  }
  if (const auto root_tolerance =
          firstDefinedDoubleParameter(
              cut_params,
              {"Implicit_cut_root_tolerance",
               "ImplicitCutRootTolerance",
               "Implicit_geometry_root_tolerance",
               "ImplicitGeometryRootTolerance"})) {
    request.implicit_cut_root_tolerance = *root_tolerance;
  }
  if (const auto coordinate_tolerance =
          firstDefinedDoubleParameter(
              cut_params,
              {"Implicit_cut_root_coordinate_tolerance",
               "ImplicitCutRootCoordinateTolerance",
               "Implicit_geometry_root_coordinate_tolerance",
               "ImplicitGeometryRootCoordinateTolerance"})) {
    request.implicit_cut_root_coordinate_tolerance = *coordinate_tolerance;
  }
  if (const auto max_root_iterations =
          firstDefinedIntParameter(
              cut_params,
              {"Implicit_cut_root_max_iterations",
               "ImplicitCutRootMaxIterations",
               "Implicit_geometry_root_max_iterations",
               "ImplicitGeometryRootMaxIterations"})) {
    request.implicit_cut_root_max_iterations = *max_root_iterations;
  }
  if (const auto max_subdivision_depth =
          firstDefinedIntParameter(
              cut_params,
              {"Implicit_cut_max_subdivision_depth",
               "ImplicitCutMaxSubdivisionDepth",
               "Implicit_cut_subdivision_depth",
               "ImplicitCutSubdivisionDepth"})) {
    request.implicit_cut_max_subdivision_depth = *max_subdivision_depth;
  }
  if (const auto affected_neighborhood_layers =
          firstDefinedIntParameter(
              cut_params,
              {"Affected_cell_neighborhood_layers",
               "AffectedCellNeighborhoodLayers",
               "Generated_interface_affected_cell_neighborhood_layers",
               "GeneratedInterfaceAffectedCellNeighborhoodLayers",
               "Generated_cut_refresh_neighborhood_layers",
               "GeneratedCutRefreshNeighborhoodLayers"})) {
    if (*affected_neighborhood_layers < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Generated interface affected-cell "
          "neighborhood layers must be nonnegative.");
    }
    request.affected_cell_neighborhood_layers =
        *affected_neighborhood_layers;
  }
  if (const auto allow_corner_linearized =
          firstDefinedBoolParameter(
              cut_params,
              {"Allow_corner_linearized_cut_geometry",
               "AllowCornerLinearizedCutGeometry",
               "Allow_corner_linearized_geometry",
               "AllowCornerLinearizedGeometry"})) {
    request.allow_corner_linearized_geometry = *allow_corner_linearized;
  }
  // Keep the aliases and diffusivity-implies-enable rule synchronized with
  // NavierStokesRegister.cpp. The Application must retain the inactive-side
  // cut rules whenever Physics will construct the extension problem.
  const auto velocity_extension =
      firstDefinedBoolParameter(
          cut_params,
          {"Enable_velocity_extension",
           "EnableVelocityExtension",
           "Velocity_extension",
           "VelocityExtension",
           "Extend_velocity_to_inactive_domain",
           "ExtendVelocityToInactiveDomain",
           "Free_surface_velocity_extension",
           "FreeSurfaceVelocityExtension"});
  const bool velocity_extension_explicitly_disabled =
      velocity_extension.has_value() && !*velocity_extension;
  bool velocity_extension_enabled =
      velocity_extension.value_or(false);
  if (firstDefinedDoubleParameter(
          cut_params,
          {"Velocity_extension_diffusivity",
           "VelocityExtensionDiffusivity",
           "Inactive_velocity_extension_diffusivity",
           "InactiveVelocityExtensionDiffusivity"})
          .has_value() &&
      !velocity_extension_explicitly_disabled) {
    velocity_extension_enabled = true;
  }
  if (velocity_extension_enabled) {
    request.volume_retention =
        ActiveCutVolumeRetention::ActiveAndInactive;
  }
  // Retention A/B experiment knob: request the retained generated-volume
  // sides regardless of the velocity-extension-driven default. Inactive-side
  // rules are metadata consumers' concern (aggregation classification,
  // diagnostics). The free-surface post-processing below rejects the
  // active-only request when small-cut aggregation is enabled, because that
  // consumer requires a complete two-sided cell classification.
  if (const char* force = std::getenv("SVMP_CUT_RETENTION_FORCE")) {
    const std::string_view mode(force);
    if (mode == "active_and_inactive") {
      request.volume_retention = ActiveCutVolumeRetention::ActiveAndInactive;
    } else if (mode == "active_only") {
      request.volume_retention = ActiveCutVolumeRetention::ActiveOnly;
    }
  }
  if (const auto required_qualification =
          firstDefinedParameter(
              cut_params,
              {"Required_implicit_cut_backend_qualification",
               "RequiredImplicitCutBackendQualification",
               "Require_implicit_cut_backend_qualification",
               "RequireImplicitCutBackendQualification",
               "Require_production_qualified_implicit_cut_backend",
               "RequireProductionQualifiedImplicitCutBackend"})) {
    const auto token = normalizedToken(*required_qualification);
    if (token == "productionqualified" || token == "production" ||
        token == "required" || token == "true" || token == "yes" ||
        token == "on" || token == "1") {
      request.require_production_qualified_implicit_cut_backend = true;
    } else if (token == "none" || token == "off" || token == "false" ||
               token == "no" || token == "0" || token == "experimental" ||
               token == "any") {
      request.require_production_qualified_implicit_cut_backend = false;
    } else {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Unknown implicit cut backend qualification requirement '" +
          *required_qualification + "'. Supported values: ProductionQualified or none.");
    }
  }
  if (default_linear_corner_to_differentiated &&
      !tangent_policy.has_value() &&
      request.geometry_mode ==
          svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner &&
      request.implicit_cut_backend ==
          svmp::FE::level_set::ImplicitCutQuadratureBackend::LinearCorner) {
    request.geometry_tangent_policy =
        svmp::FE::level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
  }
  if (request.geometry_mode ==
          svmp::FE::level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit &&
      request.geometry_tangent_policy ==
          svmp::FE::level_set::GeometryTangentPolicy::DifferentiatedQuadrature) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] HighOrderImplicit generated interface "
        "geometry does not support Geometry_tangent_policy=DifferentiatedQuadrature. "
        "Use Geometry_tangent_policy=RefreshedFrozenQuadrature, or use "
        "LinearCorner geometry for differentiated quadrature.");
  }
  return request;
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

svmp::FE::level_set::GeneratedInterfaceGeometryMode
parseGeneratedInterfaceGeometryMode(const std::string& raw)
{
  const auto value = normalizedToken(raw);
  using Mode = svmp::FE::level_set::GeneratedInterfaceGeometryMode;
  if (value == "linearcorner" || value == "cornerlinear" ||
      value == "linear" || value == "legacy") {
    return Mode::LinearCorner;
  }
  if (value == "highorderimplicit" || value == "highorder" ||
      value == "curvedimplicit" || value == "implicitcurved") {
    return Mode::HighOrderImplicit;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Unknown generated interface geometry mode '" +
      raw + "'.");
}

svmp::FE::level_set::ImplicitCutQuadratureBackend
parseImplicitCutQuadratureBackend(const std::string& raw)
{
  const auto value = normalizedToken(raw);
  using Backend = svmp::FE::level_set::ImplicitCutQuadratureBackend;
  if (value == "linearcorner" || value == "cornerlinear" ||
      value == "linear" || value == "legacy") {
    return Backend::LinearCorner;
  }
  if (value == "saye" || value == "sayehyperrectangle" ||
      value == "hyperrectangle") {
    return Backend::SayeHyperrectangle;
  }
  if (value == "highordersubcell" || value == "subcell" ||
      value == "subtriangulation" || value == "subtetrahedra") {
    return Backend::HighOrderSubcell;
  }
  if (value == "momentfit" || value == "momentfitting" ||
      value == "momentfitted") {
    return Backend::MomentFit;
  }
  if (value == "auto" || value == "automatic" ||
      value == "mixed" || value == "percell") {
    return Backend::Auto;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Unknown implicit cut quadrature backend '" +
      raw + "'.");
}

svmp::FE::level_set::ImplicitCutFallbackPolicy
parseImplicitCutFallbackPolicy(const std::string& raw)
{
  const auto value = normalizedToken(raw);
  using Policy = svmp::FE::level_set::ImplicitCutFallbackPolicy;
  if (value == "fail" || value == "none" || value == "error") {
    return Policy::Fail;
  }
  if (value == "linearcorner" || value == "cornerlinear" ||
      value == "linear" || value == "legacy") {
    return Policy::LinearCorner;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Unknown implicit cut fallback policy '" +
      raw + "'.");
}

svmp::FE::level_set::GeometryTangentPolicy
parseGeometryTangentPolicy(const std::string& raw)
{
  const auto value = normalizedToken(raw);
  using Policy = svmp::FE::level_set::GeometryTangentPolicy;
  if (value == "refreshedfrozenquadrature" ||
      value == "refreshedfrozen" ||
      value == "frozenquadrature" ||
      value == "quasinewton" ||
      value == "quasinewtongeometry") {
    return Policy::RefreshedFrozenQuadrature;
  }
  if (value == "differentiatedquadrature" ||
      value == "differentiated" ||
      value == "exactgeometrytangent" ||
      value == "exactsensitivities" ||
      value == "shapederivative") {
    return Policy::DifferentiatedQuadrature;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Unknown geometry tangent policy '" +
      raw + "'.");
}

std::vector<ActiveCutVolumeRequest>
activeCutVolumeRequests(const EquationParameters& equation)
{
  std::vector<ActiveCutVolumeRequest> requests;

  auto& mutable_equation = const_cast<EquationParameters&>(equation);
  const auto eq_params = mutable_equation.get_parameter_list();
  if (activeCutDomainIsExplicitlyEnabled(eq_params)) {
    const auto request =
        activeCutVolumeRequestFromParameters(
            eq_params,
            true,
            /*default_linear_corner_to_differentiated=*/true);
    if (request.has_value()) {
      auto explicit_request = *request;
      explicit_request.origin = ActiveCutVolumeRequestOrigin::Equation;
      explicit_request.equation_type =
          equation.type.defined() ? trimCopy(equation.type.value()) : std::string{};
      appendUniqueRequest(requests, std::move(explicit_request));
    }
  }

  const auto equation_type =
      equation.type.defined() ? normalizedToken(equation.type.value())
                              : std::string{};
  const bool supports_two_fluid =
      equation_type == "fluid" || equation_type == "stokes";
  const auto physical_model = firstDefinedParameter(
      eq_params,
      {"Free_surface_physical_model", "FreeSurfacePhysicalModel"});
  if (supports_two_fluid && physical_model.has_value() &&
      trimCopy(*physical_model) == "IncompressibleTwoFluid") {
    const auto level_set_field = firstDefinedParameter(
        eq_params,
        {"Level_set_field_name", "LevelSetFieldName"});
    const auto generated_domain = firstDefinedParameter(
        eq_params,
        {"Generated_interface_domain_id", "GeneratedInterfaceDomainId"});
    const auto interface_marker = firstDefinedIntParameter(
        eq_params,
        {"Material_interface_marker", "MaterialInterfaceMarker"});
    if (!level_set_field.has_value() || !generated_domain.has_value() ||
        !interface_marker.has_value() || *interface_marker < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Incompressible two-fluid generated "
          "cut construction requires Level_set_field_name, "
          "Generated_interface_domain_id, and a nonnegative "
          "Material_interface_marker.");
    }

    auto material_interface_params = eq_params;
    material_interface_params["Level_set_field_name"] =
        trimCopy(*level_set_field);
    material_interface_params["Generated_interface_domain_id"] =
        trimCopy(*generated_domain);
    material_interface_params["Interface_marker"] =
        std::to_string(*interface_marker);
    material_interface_params["Active_domain"] = "LevelSetNegative";
    material_interface_params["Active_domain_method"] = "CutVolume";
    auto request = activeCutVolumeRequestFromParameters(
        material_interface_params,
        true,
        /*default_linear_corner_to_differentiated=*/false);
    if (!request.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Incompressible two-fluid generated "
          "cut construction did not produce a cut-volume request.");
    }
    if (const char* force = std::getenv("SVMP_CUT_RETENTION_FORCE");
        force != nullptr && std::string_view(force) == "active_only") {
      throw std::runtime_error(
          "[svMultiPhysics::Application] active-only cut retention is "
          "incompatible with incompressible two-fluid; both generated "
          "cut-volume sides are required.");
    }
    request->origin = ActiveCutVolumeRequestOrigin::MaterialInterface;
    request->equation_type = trimCopy(equation.type.value());
    request->active_side = LevelSetActiveSide::Negative;
    request->volume_retention =
        ActiveCutVolumeRetention::ActiveAndInactive;
    appendUniqueRequest(requests, std::move(*request));
  }

  if (equation_type != "fluid") {
    return requests;
  }
  for (auto* bc : equation.boundary_conditions) {
    if (bc == nullptr) {
      continue;
    }
    auto bc_params = bc->get_parameter_list();
    const auto type = firstDefinedParameter(bc_params, {"Type"});
    if (!type || normalizedToken(*type) != "freesurface") {
      continue;
    }
    const auto implementation =
        firstDefinedParameter(bc_params, {"Implementation",
                                         "Free_surface_implementation",
                                         "FreeSurfaceImplementation"});
    const auto implementation_token =
        implementation.has_value() ? normalizedToken(*implementation)
                                   : std::string{};
    // Keep the accepted unfitted aliases synchronized with
    // NavierStokesRegister.cpp::parse_free_surface_implementation. Otherwise
    // Physics can install an active-domain/aggregation consumer for a BC that
    // Application silently omits from generated cut-context construction.
    if (implementation_token != "unfitted" &&
        implementation_token != "unfittedlevelset" &&
        implementation_token != "levelset" &&
        implementation_token != "embeddedlevelset") {
      continue;
    }
    const auto request =
        activeCutVolumeRequestFromParameters(
            bc_params,
            false,
            // SurfaceStress is the unfitted Navier--Stokes default and its
            // complete differentiated geometry tangent is not implemented.
            // Keep the shared Physics/Application default refreshed-frozen;
            // an explicit differentiated request is preserved and rejected by
            // Physics when a SurfaceStress residual is active.
            /*default_linear_corner_to_differentiated=*/false);
    if (request.has_value()) {
      auto free_surface_request = *request;
      // IncompressibleNavierStokesVMSModule enables small-cut aggregation by
      // default for unfitted free surfaces.  Aggregation must classify the
      // complete cell graph (full active, cut, and full inactive cells) in
      // order to prove that every small-cut candidate reaches a valid
      // full-active root.  Retaining only the active cut-volume side makes
      // that proof impossible and can hide roots across an inactive/cut band.
      // Keep these aliases and the default-true behavior synchronized with
      // NavierStokesRegister.cpp and FreeSurfaceOptions.
      const bool small_cut_aggregation =
          firstDefinedBoolParameter(
              bc_params,
              {"Small_cut_aggregation", "SmallCutAggregation",
               "Enable_small_cut_aggregation", "EnableSmallCutAggregation"})
              .value_or(true);
      if (small_cut_aggregation) {
        if (const char* force = std::getenv("SVMP_CUT_RETENTION_FORCE");
            force != nullptr &&
            std::string_view(force) == "active_only") {
          throw std::runtime_error(
              "[svMultiPhysics::Application] "
              "SVMP_CUT_RETENTION_FORCE=active_only is incompatible with "
              "enabled small-cut aggregation. Disable small-cut aggregation "
              "for the retention A/B experiment, or retain both generated "
              "cut-volume sides.");
        }
        free_surface_request.volume_retention =
            ActiveCutVolumeRetention::ActiveAndInactive;
      }
      free_surface_request.origin =
          ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary;
      free_surface_request.equation_type = trimCopy(equation.type.value());
      appendUniqueRequest(requests, std::move(free_surface_request));
    }
  }

  return requests;
}

std::vector<ActiveCutVolumeRequest>
activeCutVolumeRequests(const Parameters& params)
{
  std::vector<ActiveCutVolumeRequest> requests;
  for (auto* eq : params.equation_parameters) {
    if (eq == nullptr) {
      continue;
    }
    for (auto request : activeCutVolumeRequests(*eq)) {
      appendUniqueRequest(requests, std::move(request));
    }
  }
  return requests;
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

int requireResolvedActiveCutVolumeInterfaceMarker(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request)
{
  const auto marker = resolvedActiveCutVolumeInterfaceMarker(system, request);
  if (marker.has_value()) {
    return *marker;
  }

  const std::string equation_type =
      request.equation_type.empty() ? std::string{"<unknown>"}
                                    : request.equation_type;
  throw std::runtime_error(
      "[svMultiPhysics::Application] Equation-level level-set cut domain for equation '" +
      equation_type + "' references level-set field '" +
      request.level_set_field_name +
      "' before that field is registered. Declare the level_set equation before "
      "the cut-domain consumer, or set Interface_marker explicitly.");
}

void validateEquationLevelCutVolumeConsumer(
    const svmp::FE::systems::FESystem& system,
    const ActiveCutVolumeRequest& request,
    int resolved_marker)
{
  if (request.origin != ActiveCutVolumeRequestOrigin::Equation) {
    return;
  }
  const auto side = cutIntegrationSide(request.active_side);
  if (system.cutVolumeKernelCount(resolved_marker, side) > 0u) {
    const auto phi_field = system.findFieldByName(request.level_set_field_name);
    if (phi_field != svmp::FE::INVALID_FIELD_ID) {
      const auto& phi_record = system.fieldRecord(phi_field);
      if (phi_record.source_kind !=
          svmp::FE::systems::FieldSourceKind::PrescribedData) {
        static std::set<std::string> warned;
        const std::string warning_key =
            request.equation_type + "|" + request.domain_id + "|" +
            request.level_set_field_name + "|" +
            std::to_string(resolved_marker) + "|" +
            activeSideName(request.active_side);
        if (warned.insert(warning_key).second) {
          oopCout()
              << "[svMultiPhysics::Application] WARNING equation-level "
              << "level-set cut-domain uses a moving level-set field with "
              << "only a first-order Hadamard cut-volume shape tangent; "
              << "full differentiated cut quadrature remains unavailable"
              << " equation_type='" << request.equation_type << "'"
              << " field='" << request.level_set_field_name << "'"
              << " field_source="
              << fieldSourceKindName(phi_record.source_kind)
              << " domain_id='" << request.domain_id << "'"
              << " marker=" << resolved_marker
              << " active_side=" << activeSideName(request.active_side)
              << " geometry_tangent_policy="
              << svmp::FE::level_set::geometryTangentPolicyName(
                     request.geometry_tangent_policy)
              << " diagnostic=equation_level_cut_domain_hadamard_shape_tangent"
              << std::endl;
        }
      }
    }
    return;
  }

  throw std::runtime_error(
      "[svMultiPhysics::Application] Equation-level level-set cut-domain "
      "request for equation_type='" + request.equation_type +
      "' field='" + request.level_set_field_name + "' domain_id='" +
      request.domain_id + "' resolved marker=" +
      std::to_string(resolved_marker) + " active_side=" +
      activeSideName(request.active_side) +
      " has no matching dCutVolume(...) form consumer. "
      "Add volume terms restricted with dCutVolume(marker, side), remove the "
      "equation-level cut-domain request, or keep this as an unfitted "
      "free-surface boundary request owned by Navier-Stokes.");
}

} // namespace core
} // namespace application
