#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Parameters.h"
#include "Spaces/H1Space.h"
#include "tinyxml2.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace level_set = svmp::FE::level_set;

using ActiveCutRequest = application::core::ActiveCutVolumeRequest;
using ActiveSide = application::core::LevelSetActiveSide;
using CutSide = svmp::FE::geometry::CutIntegrationSide;
using GeneratedOptions = level_set::LevelSetGeneratedInterfaceOptions;
using VolumeOptions = level_set::LevelSetVolumeOptions;

void expectGeneratedInterfaceOptionsEqual(
    const GeneratedOptions& actual, const GeneratedOptions& expected)
{
  EXPECT_EQ(actual.level_set_field_name, expected.level_set_field_name);
  EXPECT_EQ(actual.domain_id, expected.domain_id);
  EXPECT_EQ(actual.requested_interface_marker, expected.requested_interface_marker);
  EXPECT_EQ(actual.isovalue, expected.isovalue);
  EXPECT_EQ(actual.tolerance, expected.tolerance);
  EXPECT_EQ(actual.quadrature_order, expected.quadrature_order);
  EXPECT_EQ(actual.interface_quadrature_order, expected.interface_quadrature_order);
  EXPECT_EQ(actual.volume_quadrature_order, expected.volume_quadrature_order);
  EXPECT_EQ(actual.geometry_mode, expected.geometry_mode);
  EXPECT_EQ(actual.implicit_cut_quadrature_backend,
            expected.implicit_cut_quadrature_backend);
  EXPECT_EQ(actual.implicit_cut_fallback_policy, expected.implicit_cut_fallback_policy);
  EXPECT_EQ(actual.implicit_cut_root_tolerance, expected.implicit_cut_root_tolerance);
  EXPECT_EQ(actual.implicit_cut_root_coordinate_tolerance,
            expected.implicit_cut_root_coordinate_tolerance);
  EXPECT_EQ(actual.implicit_cut_root_max_iterations,
            expected.implicit_cut_root_max_iterations);
  EXPECT_EQ(actual.implicit_cut_max_subdivision_depth,
            expected.implicit_cut_max_subdivision_depth);
  EXPECT_EQ(actual.keep_degenerate_fragments, expected.keep_degenerate_fragments);
  EXPECT_EQ(actual.aligned_zero_interface_parent_side,
            expected.aligned_zero_interface_parent_side);
  EXPECT_EQ(actual.allow_corner_linearized_geometry,
            expected.allow_corner_linearized_geometry);
  EXPECT_EQ(actual.require_production_qualified_implicit_cut_backend,
            expected.require_production_qualified_implicit_cut_backend);
  EXPECT_EQ(actual.affected_cell_neighborhood_layers,
            expected.affected_cell_neighborhood_layers);
  EXPECT_EQ(actual.geometry_tangent_policy, expected.geometry_tangent_policy);
}

void expectVolumeOptionsEqual(
    const VolumeOptions& actual, const VolumeOptions& expected)
{
  EXPECT_EQ(actual.isovalue, expected.isovalue);
  EXPECT_EQ(actual.tolerance, expected.tolerance);
  EXPECT_EQ(actual.use_generated_interface_quadrature,
            expected.use_generated_interface_quadrature);
  EXPECT_EQ(actual.level_set_field_name, expected.level_set_field_name);
  EXPECT_EQ(actual.generated_domain_id, expected.generated_domain_id);
  EXPECT_EQ(actual.requested_interface_marker, expected.requested_interface_marker);
  EXPECT_EQ(actual.quadrature_order, expected.quadrature_order);
  EXPECT_EQ(actual.interface_quadrature_order, expected.interface_quadrature_order);
  EXPECT_EQ(actual.volume_quadrature_order, expected.volume_quadrature_order);
  EXPECT_EQ(actual.geometry_mode, expected.geometry_mode);
  EXPECT_EQ(actual.implicit_cut_quadrature_backend,
            expected.implicit_cut_quadrature_backend);
  EXPECT_EQ(actual.implicit_cut_fallback_policy, expected.implicit_cut_fallback_policy);
  EXPECT_EQ(actual.geometry_tangent_policy, expected.geometry_tangent_policy);
  EXPECT_EQ(actual.implicit_cut_root_tolerance, expected.implicit_cut_root_tolerance);
  EXPECT_EQ(actual.implicit_cut_root_coordinate_tolerance,
            expected.implicit_cut_root_coordinate_tolerance);
  EXPECT_EQ(actual.implicit_cut_root_max_iterations,
            expected.implicit_cut_root_max_iterations);
  EXPECT_EQ(actual.implicit_cut_max_subdivision_depth,
            expected.implicit_cut_max_subdivision_depth);
  EXPECT_EQ(actual.affected_cell_neighborhood_layers,
            expected.affected_cell_neighborhood_layers);
  EXPECT_EQ(actual.allow_corner_linearized_geometry,
            expected.allow_corner_linearized_geometry);
  EXPECT_EQ(actual.require_production_qualified_implicit_cut_backend,
            expected.require_production_qualified_implicit_cut_backend);
}

ActiveCutRequest makeCutOptionRequest(
    std::string field_name, std::string domain_id, ActiveSide active_side)
{
  ActiveCutRequest request{};
  request.level_set_field_name = std::move(field_name);
  request.domain_id = std::move(domain_id);
  request.equation_type = "fluid";
  request.origin =
      application::core::ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary;
  request.active_side = active_side;
  request.volume_retention =
      application::core::ActiveCutVolumeRetention::ActiveOnly;
  return request;
}

ActiveCutRequest makeCompleteCutOptionRequest(
    std::string field_name,
    std::string domain_id,
    level_set::GeneratedInterfaceGeometryMode geometry_mode,
    ActiveSide active_side)
{
  auto request = makeCutOptionRequest(
      std::move(field_name), std::move(domain_id), active_side);
  request.requested_interface_marker = 77;
  request.isovalue = 0.25;
  request.quadrature_order = 5;
  request.interface_quadrature_order = 4;
  request.volume_quadrature_order = 6;
  request.geometry_mode = geometry_mode;
  request.implicit_cut_backend =
      geometry_mode == level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit
          ? level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle
          : level_set::ImplicitCutQuadratureBackend::LinearCorner;
  request.implicit_cut_fallback_policy =
      level_set::ImplicitCutFallbackPolicy::LinearCorner;
  request.geometry_tangent_policy =
      geometry_mode == level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit
          ? level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature
          : level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
  request.geometry_tangent_policy_explicit = true;
  request.implicit_cut_root_tolerance = 2.5e-11;
  request.implicit_cut_root_coordinate_tolerance = 4.0e-12;
  request.implicit_cut_root_max_iterations = 31;
  request.implicit_cut_max_subdivision_depth = 12;
  request.affected_cell_neighborhood_layers = 1;
  request.allow_corner_linearized_geometry = true;
  request.require_production_qualified_implicit_cut_backend = true;
  return request;
}

GeneratedOptions expectedDefaultGeneratedOptions(
    std::string domain_id,
    int interface_order,
    CutSide aligned_side,
    int requested_marker = -1,
    bool allow_corner_geometry = false,
    level_set::GeometryTangentPolicy tangent_policy =
        level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature)
{
  return GeneratedOptions{
      .level_set_field_name = "phi_cut_reference", .domain_id = std::move(domain_id),
      .requested_interface_marker = requested_marker, .isovalue = 0.0,
      .tolerance = 1.0e-12, .quadrature_order = 1,
      .interface_quadrature_order = interface_order, .volume_quadrature_order = 2,
      .geometry_mode = level_set::GeneratedInterfaceGeometryMode::LinearCorner,
      .implicit_cut_quadrature_backend =
          level_set::ImplicitCutQuadratureBackend::LinearCorner,
      .implicit_cut_fallback_policy = level_set::ImplicitCutFallbackPolicy::Fail,
      .implicit_cut_root_tolerance = 1.0e-10,
      .implicit_cut_root_coordinate_tolerance = 1.0e-12,
      .implicit_cut_root_max_iterations = 48,
      .implicit_cut_max_subdivision_depth = 16,
      .keep_degenerate_fragments = false,
      .aligned_zero_interface_parent_side = aligned_side,
      .allow_corner_linearized_geometry = allow_corner_geometry,
      .require_production_qualified_implicit_cut_backend = false,
      .affected_cell_neighborhood_layers = 0,
      .geometry_tangent_policy = tangent_policy,
  };
}

GeneratedOptions expectedCompleteGeneratedOptions(CutSide aligned_side)
{
  return GeneratedOptions{
      .level_set_field_name = "phi_cut_reference", .domain_id = "water_air",
      .requested_interface_marker = 77, .isovalue = 0.25,
      .tolerance = 1.0e-12, .quadrature_order = 5,
      .interface_quadrature_order = 4, .volume_quadrature_order = 6,
      .geometry_mode =
          level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit,
      .implicit_cut_quadrature_backend =
          level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
      .implicit_cut_fallback_policy =
          level_set::ImplicitCutFallbackPolicy::LinearCorner,
      .implicit_cut_root_tolerance = 2.5e-11,
      .implicit_cut_root_coordinate_tolerance = 4.0e-12,
      .implicit_cut_root_max_iterations = 31,
      .implicit_cut_max_subdivision_depth = 12,
      .keep_degenerate_fragments = false,
      .aligned_zero_interface_parent_side = aligned_side,
      .allow_corner_linearized_geometry = true,
      .require_production_qualified_implicit_cut_backend = true,
      .affected_cell_neighborhood_layers = 1,
      .geometry_tangent_policy =
          level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature,
  };
}

VolumeOptions expectedDefaultVolumeOptions()
{
  return VolumeOptions{
      .isovalue = 0.375, .tolerance = 1.0e-12,
      .use_generated_interface_quadrature = false, .level_set_field_name = "",
      .generated_domain_id = "volume_correction", .requested_interface_marker = -1,
      .quadrature_order = std::nullopt,
      .interface_quadrature_order = std::nullopt,
      .volume_quadrature_order = std::nullopt,
      .geometry_mode = level_set::GeneratedInterfaceGeometryMode::LinearCorner,
      .implicit_cut_quadrature_backend =
          level_set::ImplicitCutQuadratureBackend::LinearCorner,
      .implicit_cut_fallback_policy = level_set::ImplicitCutFallbackPolicy::Fail,
      .geometry_tangent_policy =
          level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature,
      .implicit_cut_root_tolerance = 1.0e-10,
      .implicit_cut_root_coordinate_tolerance = 1.0e-12,
      .implicit_cut_root_max_iterations = 48,
      .implicit_cut_max_subdivision_depth = 16,
      .affected_cell_neighborhood_layers = 0,
      .allow_corner_linearized_geometry = false,
      .require_production_qualified_implicit_cut_backend = false,
  };
}

VolumeOptions expectedCompleteVolumeOptions(std::string generated_domain_id)
{
  return VolumeOptions{
      .isovalue = 0.375, .tolerance = 1.0e-12,
      .use_generated_interface_quadrature = true,
      .level_set_field_name = "phi_maintenance",
      .generated_domain_id = std::move(generated_domain_id),
      .requested_interface_marker = 77,
      .quadrature_order = 5, .interface_quadrature_order = 4,
      .volume_quadrature_order = 6,
      .geometry_mode =
          level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit,
      .implicit_cut_quadrature_backend =
          level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
      .implicit_cut_fallback_policy =
          level_set::ImplicitCutFallbackPolicy::LinearCorner,
      .geometry_tangent_policy =
          level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature,
      .implicit_cut_root_tolerance = 2.5e-11,
      .implicit_cut_root_coordinate_tolerance = 4.0e-12,
      .implicit_cut_root_max_iterations = 31,
      .implicit_cut_max_subdivision_depth = 12,
      .affected_cell_neighborhood_layers = 1,
      .allow_corner_linearized_geometry = true,
      .require_production_qualified_implicit_cut_backend = true,
  };
}

class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(const char* name, const char* value)
      : name_(name)
  {
    if (const char* previous = std::getenv(name); previous != nullptr) {
      previous_ = std::string(previous);
    }
    if (::setenv(name, value, 1) != 0) {
      throw std::runtime_error("failed to set test environment variable");
    }
  }

  ~ScopedEnvironmentVariable()
  {
    if (previous_.has_value()) {
      (void)::setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      (void)::unsetenv(name_.c_str());
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
  std::string name_;
  std::optional<std::string> previous_;
};

std::unique_ptr<Parameters> parseParametersXml(const char* xml)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.Parse(xml);
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(doc.ErrorStr());
  }
  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (root == nullptr) {
    throw std::runtime_error("missing root solver element");
  }
  auto params = std::make_unique<Parameters>();
  params->set_equation_values(root);
  return params;
}

fs::path repositoryRoot()
{
  fs::path path = fs::current_path();
  for (int depth = 0; depth < 12; ++depth) {
    if (fs::exists(path / "tests/cases/fluid/open_vessel_free_surface")) {
      return path;
    }
    if (!path.has_parent_path() || path == path.parent_path()) {
      break;
    }
    path = path.parent_path();
  }
  throw std::runtime_error("could not locate repository root from current path");
}

void loadEquationParameters(const fs::path& path, Parameters& params)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.LoadFile(path.string().c_str());
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("failed to load " + path.string() + ": " + doc.ErrorStr());
  }
  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (root == nullptr) {
    throw std::runtime_error("missing root solver element in " + path.string());
  }
  params.set_equation_values(root);
}

std::shared_ptr<svmp::Mesh> buildSingleQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(LevelSetCutConfiguration,
     GeneratedInterfaceOptionsMatchCapturedProjectionValues)
{
  struct Case {
    const char* label;
    ActiveCutRequest request;
    int dimension;
    GeneratedOptions expected;
  };

  std::vector<Case> cases;
  cases.push_back({
      "2d_default_negative",
      makeCutOptionRequest(
          "phi_cut_reference", "cut_option_2d_default_negative",
          ActiveSide::Negative),
      2,
      expectedDefaultGeneratedOptions(
          "cut_option_2d_default_negative", 2, CutSide::Negative)});
  cases.push_back({
      "2d_default_positive",
      makeCutOptionRequest(
          "phi_cut_reference", "cut_option_2d_default_positive",
          ActiveSide::Positive),
      2,
      expectedDefaultGeneratedOptions(
          "cut_option_2d_default_positive", 2, CutSide::Positive)});
  cases.push_back({
      "3d_default_negative",
      makeCutOptionRequest(
          "phi_cut_reference", "cut_option_3d_default_negative",
          ActiveSide::Negative),
      3,
      expectedDefaultGeneratedOptions(
          "cut_option_3d_default_negative", -1, CutSide::Negative)});
  cases.push_back({
      "3d_default_positive",
      makeCutOptionRequest(
          "phi_cut_reference", "cut_option_3d_default_positive",
          ActiveSide::Positive),
      3,
      expectedDefaultGeneratedOptions(
          "cut_option_3d_default_positive", -1, CutSide::Positive)});

  const auto complete_negative = makeCompleteCutOptionRequest(
      "phi_cut_reference", "water_air",
      level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit,
      ActiveSide::Negative);
  const auto complete_positive = makeCompleteCutOptionRequest(
      "phi_cut_reference", "water_air",
      level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit,
      ActiveSide::Positive);
  cases.push_back({"2d_nondefault_negative", complete_negative, 2,
                   expectedCompleteGeneratedOptions(CutSide::Negative)});
  cases.push_back({"2d_nondefault_positive", complete_positive, 2,
                   expectedCompleteGeneratedOptions(CutSide::Positive)});
  cases.push_back({"3d_nondefault_negative", complete_negative, 3,
                   expectedCompleteGeneratedOptions(CutSide::Negative)});
  cases.push_back({"3d_nondefault_positive", complete_positive, 3,
                   expectedCompleteGeneratedOptions(CutSide::Positive)});

  auto differentiated = makeCutOptionRequest(
      "phi_cut_reference", "cut_option_2d_linear_differentiated",
      ActiveSide::Negative);
  differentiated.requested_interface_marker = 78;
  differentiated.geometry_tangent_policy =
      level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
  differentiated.geometry_tangent_policy_explicit = true;
  differentiated.allow_corner_linearized_geometry = true;
  cases.push_back({
      "2d_linear_differentiated",
      differentiated,
      2,
      expectedDefaultGeneratedOptions(
          "cut_option_2d_linear_differentiated",
          2,
          CutSide::Negative,
          78,
          true,
          level_set::GeometryTangentPolicy::DifferentiatedQuadrature)});

  ASSERT_EQ(cases.size(), 9u);
  for (const auto& current_case : cases) {
    SCOPED_TRACE(current_case.label);
    const auto actual =
        application::core::generatedInterfaceOptionsForActiveCut(
            current_case.request, current_case.dimension);
    expectGeneratedInterfaceOptionsEqual(actual, current_case.expected);
  }
}

TEST(LevelSetCutConfiguration, VolumeOptionsMatchCapturedMaintenanceValues)
{
  struct Case {
    const char* label;
    std::optional<ActiveCutRequest> request;
    VolumeOptions expected;
  };

  std::vector<Case> cases;
  cases.push_back({"no_association", std::nullopt,
                   expectedDefaultVolumeOptions()});
  for (const auto side : {ActiveSide::Negative, ActiveSide::Positive}) {
    const bool negative = side == ActiveSide::Negative;
    cases.push_back({negative ? "linear_negative" : "linear_positive",
                     makeCompleteCutOptionRequest(
                         "phi_request", "water_air",
                         level_set::GeneratedInterfaceGeometryMode::LinearCorner,
                         side),
                     expectedDefaultVolumeOptions()});
    cases.push_back({negative ? "high_order_empty_domain_negative"
                              : "high_order_empty_domain_positive",
                     makeCompleteCutOptionRequest(
                         "phi_request", "",
                         level_set::GeneratedInterfaceGeometryMode::
                             HighOrderImplicit,
                         side),
                     expectedCompleteVolumeOptions("volume_correction")});
    cases.push_back({negative ? "high_order_water_air_negative"
                              : "high_order_water_air_positive",
                     makeCompleteCutOptionRequest(
                         "phi_request", "water_air",
                         level_set::GeneratedInterfaceGeometryMode::
                             HighOrderImplicit,
                         side),
                     expectedCompleteVolumeOptions(
                         "water_air_volume_correction")});
  }

  ASSERT_EQ(cases.size(), 7u);
  for (const auto& current_case : cases) {
    SCOPED_TRACE(current_case.label);
    const auto actual = application::core::volumeOptionsForCutMaintenance(
        current_case.request, "phi_maintenance", 0.375);
    expectVolumeOptionsEqual(actual, current_case.expected);
  }
}

TEST(LevelSetCutConfiguration,
     CutOptionConversionsPreserveExplicitAndOmittedOrders)
{
  struct Case {
    const char* label;
    std::optional<int> generic;
    std::optional<int> interface;
    std::optional<int> volume;
    std::array<int, 3> expected_2d;
    std::array<int, 3> expected_3d;
  };
  const std::array<Case, 7> cases{{
      {"omitted_omitted_omitted", std::nullopt, std::nullopt,
       std::nullopt, {1, 2, 2}, {1, -1, 2}},
      {"generic_5", 5, std::nullopt, std::nullopt,
       {5, 2, 2}, {5, -1, 2}},
      {"volume_6", std::nullopt, std::nullopt, 6,
       {1, 6, 6}, {1, -1, 6}},
      {"interface_negative_1", std::nullopt, -1, std::nullopt,
       {1, -1, 2}, {1, -1, 2}},
      {"interface_0", std::nullopt, 0, std::nullopt,
       {1, 0, 2}, {1, 0, 2}},
      {"generic_5_volume_negative_1", 5, std::nullopt, -1,
       {5, -1, -1}, {5, -1, -1}},
      {"generic_negative_1_specialized_2", -1, 2, 2,
       {-1, 2, 2}, {-1, 2, 2}},
  }};

  for (const auto& current_case : cases) {
    SCOPED_TRACE(current_case.label);
    auto active = makeCutOptionRequest(
        "phi_cut_reference", "cut_option_order_contract",
        ActiveSide::Negative);
    active.quadrature_order = current_case.generic;
    active.interface_quadrature_order = current_case.interface;
    active.volume_quadrature_order = current_case.volume;

    const auto active_2d =
        application::core::generatedInterfaceOptionsForActiveCut(active, 2);
    const auto active_3d =
        application::core::generatedInterfaceOptionsForActiveCut(active, 3);
    EXPECT_EQ((std::array<int, 3>{active_2d.quadrature_order,
                                  active_2d.interface_quadrature_order,
                                  active_2d.volume_quadrature_order}),
              current_case.expected_2d);
    EXPECT_EQ((std::array<int, 3>{active_3d.quadrature_order,
                                  active_3d.interface_quadrature_order,
                                  active_3d.volume_quadrature_order}),
              current_case.expected_3d);

    auto maintenance_request = makeCompleteCutOptionRequest(
        "phi_request", "water_air",
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit,
        ActiveSide::Negative);
    maintenance_request.quadrature_order = current_case.generic;
    maintenance_request.interface_quadrature_order = current_case.interface;
    maintenance_request.volume_quadrature_order = current_case.volume;
    const auto maintenance =
        application::core::volumeOptionsForCutMaintenance(
            maintenance_request, "phi_maintenance", 0.375);
    EXPECT_EQ(maintenance.quadrature_order, current_case.generic);
    EXPECT_EQ(maintenance.interface_quadrature_order,
              current_case.interface);
    EXPECT_EQ(maintenance.volume_quadrature_order, current_case.volume);
  }
}

TEST(LevelSetCutConfiguration,
     CutOptionConversionsDeferInvalidControlValidation)
{
  struct Case {
    const char* label;
    int generic_order;
  };
  const std::array<Case, 2> cases{{
      {"invalid_generic_and_root", -1},
      {"valid_generic_invalid_root", 1},
  }};

  for (const auto& current_case : cases) {
    SCOPED_TRACE(current_case.label);
    auto request = makeCutOptionRequest(
        "phi_cut_reference", "cut_option_invalid_control",
        ActiveSide::Negative);
    request.quadrature_order = current_case.generic_order;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 2;
    request.implicit_cut_root_coordinate_tolerance = 0.0;

    std::optional<GeneratedOptions> actual;
    EXPECT_NO_THROW(
        actual = application::core::generatedInterfaceOptionsForActiveCut(
            request, 2));
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(actual->quadrature_order, current_case.generic_order);
    EXPECT_EQ(actual->interface_quadrature_order, 2);
    EXPECT_EQ(actual->volume_quadrature_order, 2);
    EXPECT_EQ(actual->implicit_cut_root_coordinate_tolerance, 0.0);
  }
}

TEST(LevelSetCutConfiguration, DefaultsUseLinearCornerPath)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_EQ(request.level_set_field_name, "phi");
  EXPECT_EQ(request.domain_id, "water_air");
  EXPECT_EQ(request.origin,
            application::core::ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary);
  EXPECT_EQ(request.equation_type, "fluid");
  EXPECT_EQ(request.geometry_mode,
            level_set::GeneratedInterfaceGeometryMode::LinearCorner);
  EXPECT_EQ(request.implicit_cut_backend,
            level_set::ImplicitCutQuadratureBackend::LinearCorner);
  EXPECT_EQ(request.implicit_cut_fallback_policy,
            level_set::ImplicitCutFallbackPolicy::Fail);
  EXPECT_EQ(request.geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  EXPECT_FALSE(request.quadrature_order.has_value());
  EXPECT_FALSE(request.interface_quadrature_order.has_value());
  EXPECT_FALSE(request.volume_quadrature_order.has_value());
  EXPECT_EQ(request.implicit_cut_root_tolerance, 1.0e-10);
  EXPECT_EQ(request.implicit_cut_root_coordinate_tolerance, 1.0e-12);
  EXPECT_EQ(request.implicit_cut_root_max_iterations, 48);
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 16);
  EXPECT_EQ(request.affected_cell_neighborhood_layers, 0);
  EXPECT_FALSE(request.allow_corner_linearized_geometry);
  EXPECT_FALSE(request.require_production_qualified_implicit_cut_backend);
  EXPECT_EQ(request.volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);
  EXPECT_FALSE(application::core::hasHighOrderGeneratedInterfaceGeometry(
      requests));
}

TEST(LevelSetCutConfiguration,
     EveryPhysicsUnfittedImplementationAliasBuildsTwoSidedCutRequest)
{
  for (const std::string_view implementation :
       {"Unfitted", "UnfittedLevelSet", "LevelSet", "EmbeddedLevelSet"}) {
    const auto xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>)xml") +
                     std::string(implementation) + R"xml(</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
    auto params = parseParametersXml(xml.c_str());
    const auto requests = application::core::activeCutVolumeRequests(*params);
    ASSERT_EQ(requests.size(), 1u) << implementation;
    EXPECT_EQ(requests.front().origin,
              application::core::ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary)
        << implementation;
    EXPECT_EQ(requests.front().volume_retention,
              application::core::ActiveCutVolumeRetention::ActiveAndInactive)
        << implementation;
  }
}

TEST(LevelSetCutConfiguration,
     TwoFluidMaterialInterfaceBuildsTwoSidedCutRequestForFluidAndStokes)
{
  struct Case {
    const char* equation_type;
    const char* xml;
  };
  const std::vector<Case> cases{
      {"fluid", R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
    <Material_interface_marker>71</Material_interface_marker>
  </Add_equation>
</svMultiPhysicsFile>
)xml"},
      {"stokes", R"xml(
<svMultiPhysicsFile>
  <Add_equation type="stokes">
    <FreeSurfacePhysicalModel>IncompressibleTwoFluid</FreeSurfacePhysicalModel>
    <LevelSetFieldName>phi</LevelSetFieldName>
    <GeneratedInterfaceDomainId>material_interface</GeneratedInterfaceDomainId>
    <MaterialInterfaceMarker>71</MaterialInterfaceMarker>
  </Add_equation>
</svMultiPhysicsFile>
)xml"},
  };

  for (const auto& item : cases) {
    SCOPED_TRACE(item.equation_type);
    auto params = parseParametersXml(item.xml);
    const auto requests = application::core::activeCutVolumeRequests(*params);
    ASSERT_EQ(requests.size(), 1u);
    const auto& request = requests.front();
    EXPECT_EQ(request.origin,
              application::core::ActiveCutVolumeRequestOrigin::MaterialInterface);
    EXPECT_EQ(request.equation_type, item.equation_type);
    EXPECT_EQ(request.level_set_field_name, "phi");
    EXPECT_EQ(request.domain_id, "material_interface");
    EXPECT_EQ(request.requested_interface_marker, 71);
    EXPECT_EQ(request.isovalue, 0.0);
    EXPECT_EQ(request.active_side,
              application::core::LevelSetActiveSide::Negative);
    EXPECT_EQ(request.volume_retention,
              application::core::ActiveCutVolumeRetention::ActiveAndInactive);
    EXPECT_EQ(request.geometry_mode,
              level_set::GeneratedInterfaceGeometryMode::LinearCorner);
    EXPECT_EQ(request.implicit_cut_backend,
              level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_EQ(request.geometry_tangent_policy,
              level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  }
}

TEST(LevelSetCutConfiguration,
     TwoFluidMaterialInterfaceRejectsActiveOnlyRetentionOverride)
{
  const ScopedEnvironmentVariable force_retention(
      "SVMP_CUT_RETENTION_FORCE", "active_only");
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
    <Material_interface_marker>71</Material_interface_marker>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  try {
    (void)application::core::activeCutVolumeRequests(*params);
    FAIL() << "Expected two-fluid active-only retention diagnostic";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(
                  "active-only cut retention is incompatible with incompressible two-fluid"),
              std::string::npos);
  }
}

TEST(LevelSetCutConfiguration,
     EverySmallCutAggregationDisableAliasPermitsActiveOnlyRetention)
{
  for (const std::string_view parameter :
       {"<Small_cut_aggregation>false</Small_cut_aggregation>",
        "<SmallCutAggregation>false</SmallCutAggregation>",
        "<Enable_small_cut_aggregation>false</Enable_small_cut_aggregation>",
        "<EnableSmallCutAggregation>false</EnableSmallCutAggregation>"}) {
    const auto xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
)xml") + std::string(parameter) + R"xml(
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";

    auto params = parseParametersXml(xml.c_str());
    const auto requests = application::core::activeCutVolumeRequests(*params);
    ASSERT_EQ(requests.size(), 1u) << parameter;
    EXPECT_EQ(requests.front().volume_retention,
              application::core::ActiveCutVolumeRetention::ActiveOnly)
        << parameter;
  }
}

TEST(LevelSetCutConfiguration,
     ActiveOnlyRetentionOverrideRejectsEnabledSmallCutAggregation)
{
  const ScopedEnvironmentVariable force_retention(
      "SVMP_CUT_RETENTION_FORCE", "active_only");
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  try {
    (void)application::core::activeCutVolumeRequests(*params);
    FAIL() << "Expected incompatible retention/aggregation diagnostic";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(
                  "SVMP_CUT_RETENTION_FORCE=active_only is incompatible"),
              std::string::npos);
  }
}

TEST(LevelSetCutConfiguration, VelocityExtensionRetainsInactiveCutVolumeSide)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
      <Enable_velocity_extension>true</Enable_velocity_extension>
      <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);
  EXPECT_EQ(requests.front().geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
}

TEST(LevelSetCutConfiguration,
     VelocityExtensionRetentionMatchesPhysicsAliasesAndImplicitEnableRule)
{
  struct Case {
    std::string_view parameters;
    application::core::ActiveCutVolumeRetention expected;
  };
  const std::vector<Case> cases{
      {"<Enable_velocity_extension>true</Enable_velocity_extension>",
       application::core::ActiveCutVolumeRetention::ActiveAndInactive},
      {"<Extend_velocity_to_inactive_domain>true</Extend_velocity_to_inactive_domain>",
       application::core::ActiveCutVolumeRetention::ActiveAndInactive},
      {"<Free_surface_velocity_extension>true</Free_surface_velocity_extension>",
       application::core::ActiveCutVolumeRetention::ActiveAndInactive},
      {"<Inactive_velocity_extension_diffusivity>2.0</Inactive_velocity_extension_diffusivity>",
       application::core::ActiveCutVolumeRetention::ActiveAndInactive},
      {"<Enable_velocity_extension>false</Enable_velocity_extension>"
       "<Velocity_extension_diffusivity>2.0</Velocity_extension_diffusivity>",
       application::core::ActiveCutVolumeRetention::ActiveOnly},
  };

  for (const auto& item : cases) {
    const auto xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Small_cut_aggregation>false</Small_cut_aggregation>
)xml") + std::string(item.parameters) + R"xml(
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
    auto params = parseParametersXml(xml.c_str());
    const auto requests = application::core::activeCutVolumeRequests(*params);
    ASSERT_EQ(requests.size(), 1u) << item.parameters;
    EXPECT_EQ(requests.front().volume_retention, item.expected)
        << item.parameters;
  }
}

TEST(LevelSetCutConfiguration, ParsesEquationLevelNonFluidCutDomain)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="heatS">
    <EnableLevelSetCutDomain>true</EnableLevelSetCutDomain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>heated_region</Generated_interface_domain_id>
    <Interface_marker>123</Interface_marker>
    <Active_domain>LevelSetPositive</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
    <Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  ASSERT_EQ(params->equation_parameters.size(), 1u);
  ASSERT_NE(params->equation_parameters.front(), nullptr);
  const auto requests =
      application::core::activeCutVolumeRequests(*params->equation_parameters.front());
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_EQ(request.origin, application::core::ActiveCutVolumeRequestOrigin::Equation);
  EXPECT_EQ(request.equation_type, "heatS");
  EXPECT_EQ(request.level_set_field_name, "phi");
  EXPECT_EQ(request.domain_id, "heated_region");
  EXPECT_EQ(request.requested_interface_marker, 123);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
  EXPECT_EQ(request.geometry_mode,
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit);
  EXPECT_EQ(request.implicit_cut_backend,
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
  EXPECT_EQ(request.geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  EXPECT_EQ(request.volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveOnly);
}

TEST(LevelSetCutConfiguration, EquationLevelCutDomainValidationRequiresConsumer)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = buildSingleQuadMesh();
  auto space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);

  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = space,
          .components = 1,
          .source_kind =
              svmp::FE::systems::FieldSourceKind::PrescribedData});

  application::core::ActiveCutVolumeRequest request{};
  request.origin = application::core::ActiveCutVolumeRequestOrigin::Equation;
  request.equation_type = "heatS";
  request.level_set_field_name = "phi";
  request.domain_id = "heated_region";
  request.requested_interface_marker = 123;
  request.active_side = application::core::LevelSetActiveSide::Positive;

  try {
    application::core::validateEquationLevelCutVolumeConsumer(
        system,
        request,
        request.requested_interface_marker);
    FAIL() << "Expected missing cut-volume consumer diagnostic";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("has no matching dCutVolume(...) form consumer"),
              std::string::npos);
    EXPECT_NE(message.find("equation_type='heatS'"), std::string::npos);
    EXPECT_NE(message.find("active_side=LevelSetPositive"),
              std::string::npos);
  }

  system.addCutVolumeKernel(
      "equations",
      request.requested_interface_marker,
      svmp::FE::geometry::CutIntegrationSide::Positive,
      phi,
      nullptr);
  EXPECT_NO_THROW(application::core::validateEquationLevelCutVolumeConsumer(
      system,
      request,
      request.requested_interface_marker));
#endif
}

TEST(LevelSetCutConfiguration, ResolvesGeneratedInterfaceMarkersThroughSharedHelper)
{
  auto mesh = buildSingleQuadMesh();
  auto space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{.name = "phi", .space = space, .components = 1});

  application::core::ActiveCutVolumeRequest explicit_request{};
  explicit_request.level_set_field_name = "phi";
  explicit_request.domain_id = "shared_marker_domain";
  explicit_request.requested_interface_marker = 456;
  EXPECT_EQ(application::core::resolvedActiveCutVolumeInterfaceMarker(
                system, explicit_request),
            456);
  EXPECT_EQ(application::core::requireResolvedActiveCutVolumeInterfaceMarker(
                system, explicit_request),
            456);

  application::core::ActiveCutVolumeRequest generated_request{};
  generated_request.level_set_field_name = "phi";
  generated_request.domain_id = "shared_marker_domain";
  generated_request.requested_interface_marker = -1;
  generated_request.isovalue = 0.125;
  svmp::FE::interfaces::GeneratedInterfaceMarkerKey key{};
  key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = generated_request.domain_id;
  key.isovalue = static_cast<svmp::FE::Real>(generated_request.isovalue);
  key.requested_marker = generated_request.requested_interface_marker;
  EXPECT_EQ(application::core::resolvedActiveCutVolumeInterfaceMarker(
                system, generated_request),
            svmp::FE::interfaces::stableGeneratedInterfaceMarker(key));

  generated_request.level_set_field_name = "missing_phi";
  EXPECT_FALSE(application::core::resolvedActiveCutVolumeInterfaceMarker(
                   system, generated_request)
                   .has_value());
  EXPECT_THROW(
      (void)application::core::requireResolvedActiveCutVolumeInterfaceMarker(
          system, generated_request),
      std::runtime_error);
}

TEST(LevelSetCutConfiguration, ParsesCanonicalImplicitCutOptions)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Interface_marker>77</Interface_marker>
      <Level_set_isovalue>0.25</Level_set_isovalue>
      <Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
      <Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
      <Implicit_cut_fallback_policy>LinearCorner</Implicit_cut_fallback_policy>
      <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
      <Implicit_cut_root_tolerance>2.5e-11</Implicit_cut_root_tolerance>
      <Implicit_cut_root_coordinate_tolerance>4.0e-12</Implicit_cut_root_coordinate_tolerance>
      <Implicit_cut_root_max_iterations>31</Implicit_cut_root_max_iterations>
      <Implicit_cut_max_subdivision_depth>12</Implicit_cut_max_subdivision_depth>
      <Affected_cell_neighborhood_layers>1</Affected_cell_neighborhood_layers>
      <Generated_interface_quadrature_order>5</Generated_interface_quadrature_order>
      <Interface_quadrature_order>4</Interface_quadrature_order>
      <Volume_quadrature_order>6</Volume_quadrature_order>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Required_implicit_cut_backend_qualification>ProductionQualified</Required_implicit_cut_backend_qualification>
      <Active_domain>LevelSetPositive</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_EQ(request.requested_interface_marker, 77);
  EXPECT_DOUBLE_EQ(request.isovalue, 0.25);
  EXPECT_EQ(request.geometry_mode,
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit);
  EXPECT_EQ(request.implicit_cut_backend,
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
  EXPECT_EQ(request.implicit_cut_fallback_policy,
            level_set::ImplicitCutFallbackPolicy::LinearCorner);
  EXPECT_EQ(request.geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  ASSERT_TRUE(request.quadrature_order.has_value());
  ASSERT_TRUE(request.interface_quadrature_order.has_value());
  ASSERT_TRUE(request.volume_quadrature_order.has_value());
  EXPECT_EQ(*request.quadrature_order, 5);
  EXPECT_EQ(*request.interface_quadrature_order, 4);
  EXPECT_EQ(*request.volume_quadrature_order, 6);
  EXPECT_DOUBLE_EQ(request.implicit_cut_root_tolerance, 2.5e-11);
  EXPECT_DOUBLE_EQ(request.implicit_cut_root_coordinate_tolerance, 4.0e-12);
  EXPECT_EQ(request.implicit_cut_root_max_iterations, 31);
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 12);
  EXPECT_EQ(request.affected_cell_neighborhood_layers, 1);
  EXPECT_TRUE(request.allow_corner_linearized_geometry);
  EXPECT_TRUE(request.require_production_qualified_implicit_cut_backend);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
  EXPECT_TRUE(application::core::hasHighOrderGeneratedInterfaceGeometry(
      requests));
}

TEST(LevelSetCutConfiguration, ParsesImplicitCutOptionSynonyms)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <LevelSetField>phi</LevelSetField>
      <InterfaceDomainId>water_air</InterfaceDomainId>
      <InterfaceMarker>81</InterfaceMarker>
      <InterfaceIsovalue>-0.125</InterfaceIsovalue>
      <GeneratedInterfaceGeometryMode>curved implicit</GeneratedInterfaceGeometryMode>
      <GeneratedInterfaceQuadratureBackend>hyperrectangle</GeneratedInterfaceQuadratureBackend>
      <ImplicitCutQuadratureFallback>legacy</ImplicitCutQuadratureFallback>
      <GeneratedInterfaceGeometryTangentPolicy>quasi Newton geometry</GeneratedInterfaceGeometryTangentPolicy>
      <ImplicitGeometryRootTolerance>3.0e-9</ImplicitGeometryRootTolerance>
      <ImplicitGeometryRootCoordinateTolerance>7.0e-10</ImplicitGeometryRootCoordinateTolerance>
      <ImplicitGeometryRootMaxIterations>29</ImplicitGeometryRootMaxIterations>
      <ImplicitCutSubdivisionDepth>9</ImplicitCutSubdivisionDepth>
      <GeneratedCutRefreshNeighborhoodLayers>2</GeneratedCutRefreshNeighborhoodLayers>
      <CutQuadratureOrder>7</CutQuadratureOrder>
      <CutInterfaceQuadratureOrder>3</CutInterfaceQuadratureOrder>
      <CutVolumeQuadratureOrder>8</CutVolumeQuadratureOrder>
      <AllowCornerLinearizedGeometry>yes</AllowCornerLinearizedGeometry>
      <RequireProductionQualifiedImplicitCutBackend>yes</RequireProductionQualifiedImplicitCutBackend>
      <ActiveDomain>phiPositive</ActiveDomain>
      <ActiveDomainMethod>CutVolume</ActiveDomainMethod>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_EQ(request.level_set_field_name, "phi");
  EXPECT_EQ(request.domain_id, "water_air");
  EXPECT_EQ(request.requested_interface_marker, 81);
  EXPECT_DOUBLE_EQ(request.isovalue, -0.125);
  EXPECT_EQ(request.geometry_mode,
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit);
  EXPECT_EQ(request.implicit_cut_backend,
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
  EXPECT_EQ(request.implicit_cut_fallback_policy,
            level_set::ImplicitCutFallbackPolicy::LinearCorner);
  EXPECT_EQ(request.geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  ASSERT_TRUE(request.quadrature_order.has_value());
  ASSERT_TRUE(request.interface_quadrature_order.has_value());
  ASSERT_TRUE(request.volume_quadrature_order.has_value());
  EXPECT_EQ(*request.quadrature_order, 7);
  EXPECT_EQ(*request.interface_quadrature_order, 3);
  EXPECT_EQ(*request.volume_quadrature_order, 8);
  EXPECT_DOUBLE_EQ(request.implicit_cut_root_tolerance, 3.0e-9);
  EXPECT_DOUBLE_EQ(request.implicit_cut_root_coordinate_tolerance, 7.0e-10);
  EXPECT_EQ(request.implicit_cut_root_max_iterations, 29);
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 9);
  EXPECT_EQ(request.affected_cell_neighborhood_layers, 2);
  EXPECT_TRUE(request.allow_corner_linearized_geometry);
  EXPECT_TRUE(request.require_production_qualified_implicit_cut_backend);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
}

TEST(LevelSetCutConfiguration, ParsesActiveDomainAliasesAndOffTokens)
{
  struct Case {
    std::string token;
    bool expects_request;
    application::core::LevelSetActiveSide expected_side;
  };
  const std::vector<Case> cases = {
      {"LevelSetNegative", true, application::core::LevelSetActiveSide::Negative},
      {"negative", true, application::core::LevelSetActiveSide::Negative},
      {"negativeLevelSet", true, application::core::LevelSetActiveSide::Negative},
      {"phiNegative", true, application::core::LevelSetActiveSide::Negative},
      {"LevelSetPositive", true, application::core::LevelSetActiveSide::Positive},
      {"positive", true, application::core::LevelSetActiveSide::Positive},
      {"positiveLevelSet", true, application::core::LevelSetActiveSide::Positive},
      {"phiPositive", true, application::core::LevelSetActiveSide::Positive},
      {"None", false, application::core::LevelSetActiveSide::Negative},
      {"off", false, application::core::LevelSetActiveSide::Negative},
      {"disabled", false, application::core::LevelSetActiveSide::Negative},
      {"inactive", false, application::core::LevelSetActiveSide::Negative},
  };

  for (const auto& item : cases) {
    const auto xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>)xml") + item.token + R"xml(</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";
    auto params = parseParametersXml(xml.c_str());
    const auto requests = application::core::activeCutVolumeRequests(*params);
    if (!item.expects_request) {
      EXPECT_TRUE(requests.empty()) << item.token;
      continue;
    }
    ASSERT_EQ(requests.size(), 1u) << item.token;
    EXPECT_EQ(requests.front().active_side, item.expected_side) << item.token;
  }
}

TEST(LevelSetCutConfiguration,
     RejectsHighOrderDifferentiatedQuadratureTangentPolicy)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
      <Implicit_cut_quadrature_backend>SayeHyperrectangle</Implicit_cut_quadrature_backend>
      <Geometry_tangent_policy>DifferentiatedQuadrature</Geometry_tangent_policy>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  try {
    (void)application::core::activeCutVolumeRequests(*params);
    FAIL() << "Expected high-order differentiated tangent policy to throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("HighOrderImplicit"), std::string::npos);
    EXPECT_NE(message.find("DifferentiatedQuadrature"), std::string::npos);
    EXPECT_NE(message.find("RefreshedFrozenQuadrature"), std::string::npos);
    EXPECT_NE(message.find("LinearCorner"), std::string::npos);
  }
}

TEST(LevelSetCutConfiguration, ParsesAutoImplicitCutBackend)
{
  EXPECT_EQ(application::core::parseImplicitCutQuadratureBackend("Auto"),
            level_set::ImplicitCutQuadratureBackend::Auto);
  EXPECT_EQ(application::core::parseImplicitCutQuadratureBackend("automatic"),
            level_set::ImplicitCutQuadratureBackend::Auto);
  EXPECT_EQ(application::core::parseImplicitCutQuadratureBackend("per cell"),
            level_set::ImplicitCutQuadratureBackend::Auto);
}

TEST(LevelSetCutConfiguration, ParsesEquationLevelCutDomainForNonFluidEquation)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="darcy">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi_heat</Level_set_field_name>
    <Generated_interface_domain_id>heated_liquid</Generated_interface_domain_id>
    <Interface_marker>123</Interface_marker>
    <Level_set_isovalue>0.125</Level_set_isovalue>
    <Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>
    <Implicit_cut_quadrature_backend>Auto</Implicit_cut_quadrature_backend>
    <Implicit_cut_fallback_policy>Fail</Implicit_cut_fallback_policy>
    <Generated_interface_quadrature_order>6</Generated_interface_quadrature_order>
    <Interface_quadrature_order>5</Interface_quadrature_order>
    <Volume_quadrature_order>7</Volume_quadrature_order>
    <Active_domain>LevelSetPositive</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  const auto& request = requests.front();
  EXPECT_EQ(request.level_set_field_name, "phi_heat");
  EXPECT_EQ(request.domain_id, "heated_liquid");
  EXPECT_EQ(request.origin,
            application::core::ActiveCutVolumeRequestOrigin::Equation);
  EXPECT_EQ(request.equation_type, "darcy");
  EXPECT_EQ(request.requested_interface_marker, 123);
  EXPECT_DOUBLE_EQ(request.isovalue, 0.125);
  EXPECT_EQ(request.geometry_mode,
            level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit);
  EXPECT_EQ(request.implicit_cut_backend,
            level_set::ImplicitCutQuadratureBackend::Auto);
  EXPECT_EQ(request.implicit_cut_fallback_policy,
            level_set::ImplicitCutFallbackPolicy::Fail);
  EXPECT_EQ(request.geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
  ASSERT_TRUE(request.quadrature_order.has_value());
  ASSERT_TRUE(request.interface_quadrature_order.has_value());
  ASSERT_TRUE(request.volume_quadrature_order.has_value());
  EXPECT_EQ(*request.quadrature_order, 6);
  EXPECT_EQ(*request.interface_quadrature_order, 5);
  EXPECT_EQ(*request.volume_quadrature_order, 7);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
}

TEST(LevelSetCutConfiguration, EquationLevelCutDomainRequiresActiveDomain)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="darcy">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, DeduplicatesEquationAndFreeSurfaceCutRequests)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().origin,
            application::core::ActiveCutVolumeRequestOrigin::Equation);
  EXPECT_EQ(requests.front().equation_type, "fluid");
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);
  EXPECT_EQ(requests.front().geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
}

TEST(LevelSetCutConfiguration,
     DeduplicatesFreeSurfaceAndLaterEquationCutRequestsIndependentlyOfOrder)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
  <Add_equation type="level_set">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().origin,
            application::core::ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);
  EXPECT_EQ(requests.front().geometry_tangent_policy,
            level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature);
}

TEST(LevelSetCutConfiguration,
     RejectsExplicitEquationAndFreeSurfaceTangentPolicyConflict)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Geometry_tangent_policy>DifferentiatedQuadrature</Geometry_tangent_policy>
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, DeduplicatedRequestMergesVelocityExtensionRetention)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Enable_velocity_extension>true</Enable_velocity_extension>
      <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().origin,
            application::core::ActiveCutVolumeRequestOrigin::Equation);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);
}

TEST(LevelSetCutConfiguration, RejectsUnknownImplicitCutBackend)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Implicit_cut_quadrature_backend>unknown_backend</Implicit_cut_quadrature_backend>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, RejectsUnknownActiveDomainToken)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="darcy">
    <Enable_level_set_cut_domain>true</Enable_level_set_cut_domain>
    <Level_set_field_name>phi</Level_set_field_name>
    <Active_domain>LevelSetNegativ</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  try {
    (void)application::core::activeCutVolumeRequests(*params);
    FAIL() << "Expected unknown Active_domain token to throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Unknown Active_domain"), std::string::npos);
    EXPECT_NE(message.find("LevelSetNegativ"), std::string::npos);
  }
}

TEST(LevelSetCutConfiguration, RejectsUnknownGeometryTangentPolicy)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Geometry_tangent_policy>unknown_policy</Geometry_tangent_policy>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, RejectsUnknownBackendQualificationRequirement)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Required_implicit_cut_backend_qualification>ResearchOnly</Required_implicit_cut_backend_qualification>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, ActiveCutRequestPolicyKeyTracksBackendOptions)
{
  application::core::ActiveCutVolumeRequest base;
  base.level_set_field_name = "phi";
  base.domain_id = "water_air";
  base.geometry_mode =
      level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
  base.implicit_cut_backend =
      level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
  base.geometry_tangent_policy =
      level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;

  const auto base_key =
      application::core::activeCutVolumeRequestPolicyKey({base});
  ASSERT_NE(base_key, 0u);

  auto changed_backend = base;
  changed_backend.implicit_cut_backend =
      level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_backend}));

  auto changed_tangent = base;
  changed_tangent.geometry_tangent_policy =
      level_set::GeometryTangentPolicy::DifferentiatedQuadrature;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_tangent}));

  auto changed_order = base;
  changed_order.interface_quadrature_order = 5;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_order}));

  auto changed_root_coordinate = base;
  changed_root_coordinate.implicit_cut_root_coordinate_tolerance = 5.0e-11;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_root_coordinate}));

  auto changed_root_iterations = base;
  changed_root_iterations.implicit_cut_root_max_iterations = 24;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_root_iterations}));

  auto changed_required_qualification = base;
  changed_required_qualification.require_production_qualified_implicit_cut_backend =
      true;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_required_qualification}));

  auto changed_neighborhood = base;
  changed_neighborhood.affected_cell_neighborhood_layers = 1;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_neighborhood}));

  auto changed_retention = base;
  changed_retention.volume_retention =
      application::core::ActiveCutVolumeRetention::ActiveAndInactive;
  EXPECT_NE(base_key,
            application::core::activeCutVolumeRequestPolicyKey(
                {changed_retention}));
}

TEST(LevelSetCutConfiguration, IgnoresSmoothedIndicatorRequests)
{
  for (const std::string_view method :
       {"SmoothedIndicator", "SmoothIndicator", "Indicator"}) {
    const auto xml = std::string(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>)xml") + std::string(method) + R"xml(</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml";

    auto params = parseParametersXml(xml.c_str());
    EXPECT_TRUE(application::core::activeCutVolumeRequests(*params).empty())
        << method;
  }
}

TEST(LevelSetCutConfiguration, RejectsUnknownActiveDomainMethod)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>not-a-method</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW((void)application::core::activeCutVolumeRequests(*params),
               std::runtime_error);
}

TEST(LevelSetCutConfiguration, TrackedProductionFixturesStayOnLinearPath)
{
  const auto root = repositoryRoot();
  const std::vector<fs::path> solver_files = {
      root / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/solver.xml",
      root / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml",
      root / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml",
      root / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml",
      root / "tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml",
  };

  for (const auto& solver_file : solver_files) {
    Parameters params;
    ASSERT_NO_THROW(loadEquationParameters(solver_file, params))
        << solver_file.string();
    const auto requests = application::core::activeCutVolumeRequests(params);
    ASSERT_FALSE(requests.empty()) << solver_file.string();
    for (const auto& request : requests) {
      EXPECT_EQ(request.geometry_mode,
                level_set::GeneratedInterfaceGeometryMode::LinearCorner)
          << solver_file.string();
      EXPECT_EQ(request.implicit_cut_backend,
                level_set::ImplicitCutQuadratureBackend::LinearCorner)
          << solver_file.string();
      EXPECT_EQ(request.implicit_cut_fallback_policy,
                level_set::ImplicitCutFallbackPolicy::Fail)
          << solver_file.string();
    }
  }
}
