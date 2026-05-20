#include "Application/Core/LevelSetCutConfiguration.h"

#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
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

bool requestMatches(const ActiveCutVolumeRequest& a,
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
         a.geometry_tangent_policy == b.geometry_tangent_policy &&
         a.implicit_cut_root_tolerance == b.implicit_cut_root_tolerance &&
         a.implicit_cut_root_coordinate_tolerance ==
             b.implicit_cut_root_coordinate_tolerance &&
         a.implicit_cut_root_max_iterations ==
             b.implicit_cut_root_max_iterations &&
         a.implicit_cut_max_subdivision_depth ==
             b.implicit_cut_max_subdivision_depth &&
         a.active_side == b.active_side &&
         a.allow_corner_linearized_geometry ==
             b.allow_corner_linearized_geometry &&
         a.require_production_qualified_implicit_cut_backend ==
             b.require_production_qualified_implicit_cut_backend;
}

void appendUniqueRequest(std::vector<ActiveCutVolumeRequest>& requests,
                         ActiveCutVolumeRequest request)
{
  const auto duplicate =
      std::any_of(
          requests.begin(),
          requests.end(),
          [&](const ActiveCutVolumeRequest& existing) {
            return requestMatches(existing, request);
          });
  if (!duplicate) {
    requests.push_back(std::move(request));
  }
}

std::optional<ActiveCutVolumeRequest> activeCutVolumeRequestFromParameters(
    const std::map<std::string, std::string>& cut_params,
    bool require_active_domain)
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
  if (active_token == "none" || active_token == "off" ||
      active_token == "inactive") {
    return std::nullopt;
  }

  const auto method =
      firstDefinedParameter(cut_params,
                            {"Active_domain_method",
                             "ActiveDomainMethod",
                             "Free_surface_active_domain_method",
                             "FreeSurfaceActiveDomainMethod"});
  if (method && normalizedToken(*method) == "smoothedindicator") {
    return std::nullopt;
  }

  ActiveCutVolumeRequest request{};
  if (active_token == "levelsetpositive" ||
      active_token == "positive" ||
      active_token == "phipositive") {
    request.active_side = LevelSetActiveSide::Positive;
  } else {
    request.active_side = LevelSetActiveSide::Negative;
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
  if (const auto tangent_policy =
          firstDefinedParameter(
              cut_params,
              {"Geometry_tangent_policy",
               "GeometryTangentPolicy",
               "Generated_interface_geometry_tangent_policy",
               "GeneratedInterfaceGeometryTangentPolicy",
               "Implicit_geometry_tangent_policy",
               "ImplicitGeometryTangentPolicy"})) {
    request.geometry_tangent_policy =
        parseGeometryTangentPolicy(*tangent_policy);
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
  if (const auto allow_corner_linearized =
          firstDefinedBoolParameter(
              cut_params,
              {"Allow_corner_linearized_cut_geometry",
               "AllowCornerLinearizedCutGeometry",
               "Allow_corner_linearized_geometry",
               "AllowCornerLinearizedGeometry"})) {
    request.allow_corner_linearized_geometry = *allow_corner_linearized;
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
  return request;
}

} // namespace

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
        activeCutVolumeRequestFromParameters(eq_params, true);
    if (request.has_value()) {
      auto explicit_request = *request;
      explicit_request.origin = ActiveCutVolumeRequestOrigin::Equation;
      explicit_request.equation_type =
          equation.type.defined() ? trimCopy(equation.type.value()) : std::string{};
      appendUniqueRequest(requests, std::move(explicit_request));
    }
  }

  if (!equation.type.defined() ||
      normalizedToken(equation.type.value()) != "fluid") {
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
    if (!implementation ||
        normalizedToken(*implementation) != "unfittedlevelset") {
      continue;
    }
    const auto request =
        activeCutVolumeRequestFromParameters(bc_params, false);
    if (request.has_value()) {
      auto free_surface_request = *request;
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
    mixRequestPolicyHash(h, static_cast<std::uint64_t>(request.active_side));
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

} // namespace core
} // namespace application
