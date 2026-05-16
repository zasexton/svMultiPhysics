#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace level_set = svmp::FE::level_set;

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

} // namespace

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
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 16);
  EXPECT_FALSE(request.allow_corner_linearized_geometry);
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
      <Implicit_cut_max_subdivision_depth>12</Implicit_cut_max_subdivision_depth>
      <Generated_interface_quadrature_order>5</Generated_interface_quadrature_order>
      <Interface_quadrature_order>4</Interface_quadrature_order>
      <Volume_quadrature_order>6</Volume_quadrature_order>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
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
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 12);
  EXPECT_TRUE(request.allow_corner_linearized_geometry);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
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
      <GeneratedInterfaceGeometryTangentPolicy>exact sensitivities</GeneratedInterfaceGeometryTangentPolicy>
      <ImplicitGeometryRootTolerance>3.0e-9</ImplicitGeometryRootTolerance>
      <ImplicitCutSubdivisionDepth>9</ImplicitCutSubdivisionDepth>
      <CutQuadratureOrder>7</CutQuadratureOrder>
      <CutInterfaceQuadratureOrder>3</CutInterfaceQuadratureOrder>
      <CutVolumeQuadratureOrder>8</CutVolumeQuadratureOrder>
      <AllowCornerLinearizedGeometry>yes</AllowCornerLinearizedGeometry>
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
            level_set::GeometryTangentPolicy::DifferentiatedQuadrature);
  ASSERT_TRUE(request.quadrature_order.has_value());
  ASSERT_TRUE(request.interface_quadrature_order.has_value());
  ASSERT_TRUE(request.volume_quadrature_order.has_value());
  EXPECT_EQ(*request.quadrature_order, 7);
  EXPECT_EQ(*request.interface_quadrature_order, 3);
  EXPECT_EQ(*request.volume_quadrature_order, 8);
  EXPECT_DOUBLE_EQ(request.implicit_cut_root_tolerance, 3.0e-9);
  EXPECT_EQ(request.implicit_cut_max_subdivision_depth, 9);
  EXPECT_TRUE(request.allow_corner_linearized_geometry);
  EXPECT_EQ(request.active_side, application::core::LevelSetActiveSide::Positive);
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

TEST(LevelSetCutConfiguration, IgnoresSmoothedIndicatorRequests)
{
  auto params = parseParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>SmoothedIndicator</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_TRUE(application::core::activeCutVolumeRequests(*params).empty());
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
