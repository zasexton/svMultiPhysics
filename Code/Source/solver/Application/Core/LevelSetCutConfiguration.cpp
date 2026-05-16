#include "Application/Core/LevelSetCutConfiguration.h"

#include "Parameters.h"

#include <algorithm>
#include <cctype>
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
activeCutVolumeRequests(const Parameters& params)
{
  std::vector<ActiveCutVolumeRequest> requests;
  for (auto* eq : params.equation_parameters) {
    if (eq == nullptr || !eq->type.defined() ||
        normalizedToken(eq->type.value()) != "fluid") {
      continue;
    }
    for (auto* bc : eq->boundary_conditions) {
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
      const auto active_domain =
          firstDefinedParameter(bc_params, {"Active_domain",
                                           "ActiveDomain",
                                           "Free_surface_active_domain",
                                           "FreeSurfaceActiveDomain"});
      if (!active_domain) {
        continue;
      }
      const auto active_token = normalizedToken(*active_domain);
      if (active_token == "none" || active_token == "off" ||
          active_token == "inactive") {
        continue;
      }

      const auto method =
          firstDefinedParameter(bc_params, {"Active_domain_method",
                                           "ActiveDomainMethod",
                                           "Free_surface_active_domain_method",
                                           "FreeSurfaceActiveDomainMethod"});
      if (method && normalizedToken(*method) == "smoothedindicator") {
        continue;
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
              firstDefinedParameter(bc_params, {"Level_set_field_name",
                                               "Level_set_field",
                                               "LevelSetFieldName",
                                               "LevelSetField"})) {
        request.level_set_field_name = trimCopy(*field);
      }
      if (const auto domain =
              firstDefinedParameter(bc_params, {"Generated_interface_domain_id",
                                               "GeneratedInterfaceDomainId",
                                               "Interface_domain_id",
                                               "InterfaceDomainId"})) {
        request.domain_id = trimCopy(*domain);
      }
      if (const auto marker =
              firstDefinedIntParameter(bc_params, {"Interface_marker",
                                                   "InterfaceMarker"})) {
        request.requested_interface_marker = *marker;
      }
      if (const auto isovalue =
              firstDefinedDoubleParameter(bc_params, {"Level_set_isovalue",
                                                      "LevelSetIsovalue",
                                                      "Interface_isovalue",
                                                      "InterfaceIsovalue"})) {
        request.isovalue = *isovalue;
      }
      if (const auto quadrature_order =
              firstDefinedIntParameter(
                  bc_params,
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
                  bc_params,
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
                  bc_params,
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
                  bc_params,
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
                  bc_params,
                  {"Implicit_cut_quadrature_backend",
                   "ImplicitCutQuadratureBackend",
                   "Generated_interface_quadrature_backend",
                   "GeneratedInterfaceQuadratureBackend"})) {
        request.implicit_cut_backend =
            parseImplicitCutQuadratureBackend(*backend);
      }
      if (const auto fallback_policy =
              firstDefinedParameter(
                  bc_params,
                  {"Implicit_cut_fallback_policy",
                   "ImplicitCutFallbackPolicy",
                   "Implicit_cut_quadrature_fallback",
                   "ImplicitCutQuadratureFallback"})) {
        request.implicit_cut_fallback_policy =
            parseImplicitCutFallbackPolicy(*fallback_policy);
      }
      if (const auto tangent_policy =
              firstDefinedParameter(
                  bc_params,
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
                  bc_params,
                  {"Implicit_cut_root_tolerance",
                   "ImplicitCutRootTolerance",
                   "Implicit_geometry_root_tolerance",
                   "ImplicitGeometryRootTolerance"})) {
        request.implicit_cut_root_tolerance = *root_tolerance;
      }
      if (const auto max_subdivision_depth =
              firstDefinedIntParameter(
                  bc_params,
                  {"Implicit_cut_max_subdivision_depth",
                   "ImplicitCutMaxSubdivisionDepth",
                   "Implicit_cut_subdivision_depth",
                   "ImplicitCutSubdivisionDepth"})) {
        request.implicit_cut_max_subdivision_depth = *max_subdivision_depth;
      }
      if (const auto allow_corner_linearized =
              firstDefinedBoolParameter(
                  bc_params,
                  {"Allow_corner_linearized_cut_geometry",
                   "AllowCornerLinearizedCutGeometry",
                   "Allow_corner_linearized_geometry",
                   "AllowCornerLinearizedGeometry"})) {
        request.allow_corner_linearized_geometry = *allow_corner_linearized;
      }
      requests.push_back(std::move(request));
    }
  }
  return requests;
}

} // namespace core
} // namespace application
