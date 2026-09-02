/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidInterface.h"
#include "Physics/Formulations/NavierStokes/IncompressibleTwoFluidModule.h"

#include "FE/Forms/FormCompiler.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace ns = svmp::Physics::formulations::navier_stokes;

struct InterfaceExpressions {
  std::shared_ptr<FE::spaces::H1Space> scalar_space;
  std::shared_ptr<FE::spaces::ProductSpace> vector_space;
  FE::forms::FormExpr u_negative;
  FE::forms::FormExpr p_negative;
  FE::forms::FormExpr v_negative;
  FE::forms::FormExpr q_negative;
  FE::forms::FormExpr u_positive;
  FE::forms::FormExpr p_positive;
  FE::forms::FormExpr v_positive;
  FE::forms::FormExpr q_positive;
};

class RevisionTrackedTriangleMeshAccess final
    : public FE::assembly::IMeshAccess {
 public:
  [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
  [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
  [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
  [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
  [[nodiscard]] int dimension() const override { return 2; }
  [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
  [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
  [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override {
    return FE::ElementType::Triangle3;
  }
  void getCellNodes(FE::GlobalIndex,
                    std::vector<FE::GlobalIndex>& nodes) const override {
    nodes = {0, 1, 2};
  }
  [[nodiscard]] std::array<FE::Real, 3>
  getNodeCoordinates(FE::GlobalIndex node) const override {
    return coordinates_.at(static_cast<std::size_t>(node));
  }
  void getCellCoordinates(
      FE::GlobalIndex,
      std::vector<std::array<FE::Real, 3>>& coordinates) const override {
    coordinates.assign(coordinates_.begin(), coordinates_.end());
  }
  [[nodiscard]] FE::LocalIndex
  getLocalFaceIndex(FE::GlobalIndex, FE::GlobalIndex) const override {
    return 0;
  }
  [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex) const override {
    return -1;
  }
  [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
  getInteriorFaceCells(FE::GlobalIndex) const override {
    return {0, 0};
  }
  void forEachCell(
      std::function<void(FE::GlobalIndex)> callback) const override {
    callback(0);
  }
  void forEachOwnedCell(
      std::function<void(FE::GlobalIndex)> callback) const override {
    callback(0);
  }
  void forEachBoundaryFace(
      int,
      std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override {}
  void forEachInteriorFace(
      std::function<void(
          FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>) const override {}

 private:
  const std::array<std::array<FE::Real, 3>, 3> coordinates_{
      {{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}}};
};

InterfaceExpressions makeExpressions() {
  auto scalar_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
  auto vector_space =
      std::make_shared<FE::spaces::ProductSpace>(scalar_space, 3);
  return InterfaceExpressions{
      .scalar_space = scalar_space,
      .vector_space = vector_space,
      .u_negative = FE::forms::TrialFunction(*vector_space, "u_negative"),
      .p_negative = FE::forms::TrialFunction(*scalar_space, "p_negative"),
      .v_negative = FE::forms::TestFunction(*vector_space, "v_negative"),
      .q_negative = FE::forms::TestFunction(*scalar_space, "q_negative"),
      .u_positive = FE::forms::TrialFunction(*vector_space, "u_positive"),
      .p_positive = FE::forms::TrialFunction(*scalar_space, "p_positive"),
      .v_positive = FE::forms::TestFunction(*vector_space, "v_positive"),
      .q_positive = FE::forms::TestFunction(*scalar_space, "q_positive"),
  };
}

ns::IncompressibleTwoFluidInterfaceParameters makeParameters() {
  return ns::IncompressibleTwoFluidInterfaceParameters{
      .dimension = 3,
      .interface_marker = 71,
      .negative_density = FE::Real{1000.0},
      .positive_density = FE::Real{1.0},
      .negative_viscosity = FE::Real{0.01},
      .positive_viscosity = FE::Real{0.001},
      .nitsche_gamma = FE::Real{24.0},
      .surface_tension = FE::Real{0.072},
      .include_transient_penalty = false,
  };
}

ns::IncompressibleTwoFluidInterfaceForms
build(const ns::IncompressibleTwoFluidInterfaceParameters &parameters) {
  const auto expressions = makeExpressions();
  return ns::buildIncompressibleTwoFluidInterfaceForms(
      expressions.u_negative, expressions.p_negative, expressions.v_negative,
      expressions.q_negative, expressions.u_positive, expressions.p_positive,
      expressions.v_positive, expressions.q_positive, parameters);
}

std::shared_ptr<const FE::interfaces::FreeSurfaceGeometrySnapshot>
makeTwoFluidDiagnosticSnapshot(
    const FE::assembly::IMeshAccess& mesh,
    int marker) {
  FE::interfaces::CutInterfaceDomainRequest request;
  request.source = FE::interfaces::LevelSetInterfaceSource::fromField(
      FE::FieldId{0}, 0u, 7u);
  request.generated_domain_id = "two_fluid_interface";
  request.interface_marker = marker;
  request.quadrature_order = 1;
  request.interface_quadrature_order = 1;
  request.volume_quadrature_order = 1;
  request.implicit_geometry_mode = "LinearCorner";
  request.implicit_quadrature_backend = "LinearCorner";
  request.implicit_fallback_status = "None";

  FE::interfaces::LevelSetInterfaceDomain domain(request);
  FE::interfaces::CutInterfaceFragment fragment;
  fragment.interface_marker = marker;
  fragment.parent_cell = 0;
  fragment.local_fragment_index = 0;
  fragment.stable_id = 1u;
  fragment.kind = FE::interfaces::CutInterfaceFragmentKind::Segment;
  fragment.measure = FE::Real{0.5};
  fragment.normal = {{1.0, 0.0, 0.0}};
  fragment.min_gradient_norm = FE::Real{1.0};
  fragment.vertices = {
      FE::interfaces::CutInterfaceVertex{
          .point = {{0.5, 0.0, 0.0}},
          .parent_coordinate = {{0.5, 0.0, 0.0}}},
      FE::interfaces::CutInterfaceVertex{
          .point = {{0.5, 0.5, 0.0}},
          .parent_coordinate = {{0.5, 0.5, 0.0}}},
  };
  fragment.quadrature_points = {
      FE::interfaces::CutInterfaceQuadraturePoint{
          .point = {{0.5, 0.25, 0.0}},
          .parent_coordinate = {{0.5, 0.25, 0.0}},
          .normal = fragment.normal,
          .weight = FE::Real{0.5},
          .reference_measure_factor = FE::Real{0.5},
          .gradient_norm = FE::Real{1.0}},
  };
  domain.addFragment(std::move(fragment));

  const auto add_volume = [&](FE::geometry::CutIntegrationSide side,
                              FE::LocalIndex local_index,
                              FE::GlobalIndex stable_id,
                              FE::Real measure,
                              std::array<FE::Real, 3> centroid) {
    FE::interfaces::CutInterfaceVolumeRegion region;
    region.interface_marker = marker;
    region.parent_cell = 0;
    region.local_region_index = local_index;
    region.stable_id = stable_id;
    region.side = side;
    region.measure = measure;
    region.parent_measure = FE::Real{0.5};
    region.volume_fraction = measure / region.parent_measure;
    region.centroid = centroid;
    region.normal = side == FE::geometry::CutIntegrationSide::Negative
                        ? std::array<FE::Real, 3>{{1.0, 0.0, 0.0}}
                        : std::array<FE::Real, 3>{{-1.0, 0.0, 0.0}};
    region.quadrature_points = {
        FE::geometry::CutQuadraturePoint{
            .point = centroid,
            .normal = region.normal,
            .weight = measure,
            .parent_coordinate = centroid,
            .reference_measure_factor = measure},
    };
    domain.addVolumeRegion(std::move(region));
  };
  add_volume(FE::geometry::CutIntegrationSide::Negative,
             0u,
             2,
             FE::Real{0.375},
             {{FE::Real{2.0} / FE::Real{9.0},
               FE::Real{7.0} / FE::Real{18.0},
               0.0}});
  add_volume(FE::geometry::CutIntegrationSide::Positive,
             1u,
             3,
             FE::Real{0.125},
             {{FE::Real{2.0} / FE::Real{3.0},
               FE::Real{1.0} / FE::Real{6.0},
               0.0}});

  FE::interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
  policy.require_complete_exterior_boundary_partition = false;
  policy.minimum_achieved_quadrature_order = 1;
  FE::interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
  scalar.value = [](FE::GlobalIndex,
                    const std::array<FE::Real, 3>& point,
                    const FE::geometry::CutQuadratureProvenance&) {
    return point[0] - FE::Real{0.5};
  };
  scalar.reference_gradient =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{1.0, 0.0, 0.0}};
      };
  return FE::interfaces::buildFreeSurfaceGeometrySnapshot(
      std::move(domain),
      {},
      {},
      mesh,
      policy,
      std::move(scalar),
      "two-fluid-diagnostic-test");
}

bool expressionContains(const FE::forms::FormExpr &expression,
                        FE::forms::FormExprType target) {
  if (!expression.isValid()) {
    return false;
  }
  std::vector<const FE::forms::FormExprNode *> pending{expression.node()};
  while (!pending.empty()) {
    const auto *node = pending.back();
    pending.pop_back();
    if (node->type() == target) {
      return true;
    }
    for (const auto *child : node->children()) {
      if (child != nullptr) {
        pending.push_back(child);
      }
    }
  }
  return false;
}

struct TwoFluidRegistrationFixture {
  std::shared_ptr<RevisionTrackedTriangleMeshAccess> mesh{
      std::make_shared<RevisionTrackedTriangleMeshAccess>()};
  std::shared_ptr<FE::spaces::H1Space> scalar_space{
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1)};
  std::shared_ptr<FE::spaces::ProductSpace> velocity_space{
      std::make_shared<FE::spaces::ProductSpace>(scalar_space, 2)};
  FE::systems::FESystem system{mesh};

  TwoFluidRegistrationFixture() {
    (void)system.addField(FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
  }

  [[nodiscard]] ns::IncompressibleTwoFluidModule makeModule(
      ns::IncompressibleTwoFluidOptions options = {}) const {
    return ns::IncompressibleTwoFluidModule(
        velocity_space, scalar_space, velocity_space, scalar_space,
        std::move(options));
  }
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh>
makeTwoTriangleTwoFluidBoundaryMesh(int boundary_marker) {
  auto base = std::make_shared<svmp::MeshBase>();
  const std::vector<svmp::real_t> coordinates{
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell_offsets{0, 3, 6};
  const std::vector<svmp::index_t> cell_vertices{
      0, 1, 2,
      1, 3, 2,
  };
  const svmp::CellShape triangle{
      svmp::CellFamily::Triangle, 3, 1};
  base->build_from_arrays(
      /*spatial_dim=*/2,
      coordinates,
      cell_offsets,
      cell_vertices,
      std::vector<svmp::CellShape>(2u, triangle));
  base->finalize();

  base->register_label(
      "two_fluid_wall", static_cast<svmp::label_t>(boundary_marker));
  const auto& face_cells = base->face2cell();
  for (svmp::index_t face = 0;
       face < static_cast<svmp::index_t>(face_cells.size());
       ++face) {
    const auto& cells = face_cells[static_cast<std::size_t>(face)];
    const bool first_valid = cells[0] != svmp::INVALID_INDEX;
    const bool second_valid = cells[1] != svmp::INVALID_INDEX;
    if (first_valid == second_valid) {
      continue;
    }
    base->set_boundary_label(
        face, static_cast<svmp::label_t>(boundary_marker));
  }

  const auto phi_handle = svmp::MeshFields::attach_field(
      *base,
      svmp::EntityKind::Vertex,
      "level_set",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi = svmp::MeshFields::field_data_as<svmp::real_t>(
      *base, phi_handle);
  if (phi == nullptr) {
    throw std::runtime_error(
        "two-fluid boundary test could not allocate its level-set field");
  }
  phi[0] = -1.0;
  phi[1] = 0.0;
  phi[2] = 0.0;
  phi[3] = 1.0;
  return svmp::create_mesh(std::move(base));
}
#endif

TEST(IncompressibleTwoFluidModule,
     RejectsUnsupportedConfigurationSemanticsBeforeMutation)
{
  const auto expect_rejected = [](
      const std::function<void(ns::IncompressibleTwoFluidOptions&)>& configure,
      std::string_view expected_diagnostic) {
    TwoFluidRegistrationFixture fixture;
    ns::IncompressibleTwoFluidOptions options;
    configure(options);
    auto module = fixture.makeModule(std::move(options));

    try {
      module.registerOn(fixture.system);
      FAIL() << "Expected unsupported two-fluid configuration semantics to be rejected";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(
          std::string_view(error.what()).find(expected_diagnostic),
          std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(fixture.system.registeredFieldCount(), 1u);
    EXPECT_EQ(fixture.system.findFieldByName("u_negative"),
              FE::INVALID_FIELD_ID);
    EXPECT_EQ(fixture.system.findFieldByName("u_positive"),
              FE::INVALID_FIELD_ID);
    EXPECT_FALSE(fixture.system.hasOperator("equations"));
    EXPECT_TRUE(
        fixture.system.materialInterfaceTransportVelocityDeclarations()
            .empty());
    EXPECT_TRUE(
        fixture.system.twoFluidAcceptedStageDiagnosticDeclarations()
            .empty());
  };

  expect_rejected(
      [](auto& options) {
        options.body_force[0] =
            std::numeric_limits<FE::Real>::infinity();
      },
      "unsupported_two_fluid_nonfinite_body_force");
  expect_rejected(
      [](auto& options) { options.body_force[2] = FE::Real{1.0}; },
      "unsupported_two_fluid_out_of_plane_body_force");
  expect_rejected(
      [](auto& options) {
        options.prescribed_pressure_jump =
            std::numeric_limits<FE::Real>::infinity();
      },
      "unsupported_two_fluid_nonfinite_prescribed_pressure_jump");
  expect_rejected(
      [](auto& options) {
        options.prescribed_viscous_traction_jump =
            std::array<FE::Real, 3>{{
                FE::Real{1.0},
                std::numeric_limits<FE::Real>::infinity(),
                FE::Real{0.0}}};
      },
      "unsupported_two_fluid_nonfinite_prescribed_viscous_traction_jump");
  expect_rejected(
      [](auto& options) {
        options.prescribed_viscous_traction_jump =
            std::array<FE::Real, 3>{{
                FE::Real{1.0}, FE::Real{2.0}, FE::Real{3.0}}};
      },
      "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            boundary;
        boundary.boundary_marker = 7;
        boundary.value[0] =
            ns::IncompressibleNavierStokesVMSOptions::ScalarValue{1.0};
        options.negative_phase.velocity_dirichlet.push_back(
            std::move(boundary));
      },
      "unsupported_two_fluid_nonhomogeneous_velocity_boundary");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            boundary;
        boundary.boundary_marker = 7;
        boundary.value[0] =
            ns::IncompressibleNavierStokesVMSOptions::ScalarValue{
                std::numeric_limits<FE::Real>::infinity()};
        options.shared_velocity_dirichlet.push_back(std::move(boundary));
      },
      "unsupported_two_fluid_nonfinite_shared_velocity_boundary");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            boundary;
        boundary.boundary_marker = 7;
        boundary.value[0] = FE::forms::FormExpr::constant(FE::Real{1.0});
        options.shared_velocity_dirichlet.push_back(std::move(boundary));
      },
      "unsupported_two_fluid_shared_velocity_form_expression");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            boundary;
        boundary.boundary_marker = 7;
        boundary.value[0] = FE::forms::ScalarCoefficient{};
        options.shared_velocity_dirichlet.push_back(std::move(boundary));
      },
      "unsupported_two_fluid_empty_shared_velocity_coefficient");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            shared;
        shared.boundary_marker = 7;
        options.shared_velocity_dirichlet.push_back(std::move(shared));
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            phase_local;
        phase_local.boundary_marker = 7;
        options.negative_phase.velocity_dirichlet.push_back(
            std::move(phase_local));
      },
      "unsupported_two_fluid_overlapping_shared_velocity_boundary_marker");
  expect_rejected(
      [](auto& options) {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC
            boundary;
        boundary.boundary_marker = 7;
        options.positive_phase.velocity_dirichlet = {boundary, boundary};
      },
      "unsupported_two_fluid_duplicate_velocity_boundary_marker");
  expect_rejected(
      [](auto& options) {
        options.shared_pressure_gauge =
            ns::IncompressibleTwoFluidSharedPressureGauge{
                .vertex_gid = -1,
                .pressure = FE::Real{0.0},
            };
      },
      "unsupported_two_fluid_negative_shared_pressure_gauge_vertex_gid");
  expect_rejected(
      [](auto& options) {
        options.shared_pressure_gauge =
            ns::IncompressibleTwoFluidSharedPressureGauge{
                .vertex_gid = 1,
                .pressure =
                    std::numeric_limits<FE::Real>::infinity(),
            };
      },
      "unsupported_two_fluid_nonfinite_shared_pressure_gauge_value");
}

TEST(IncompressibleTwoFluidModule,
     RegistersSharedNonhomogeneousVelocityBoundaryOnBothPhases) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC boundary;
  boundary.boundary_marker = 7;
  boundary.active_components = {{true, true, false}};
  boundary.value[0] =
      ns::IncompressibleNavierStokesVMSOptions::ScalarValue{1.0};
  boundary.value[1] = FE::forms::TimeScalarCoefficient{
      [](FE::Real x, FE::Real, FE::Real, FE::Real time) {
        return x + time;
      }};
  options.shared_velocity_dirichlet.push_back(std::move(boundary));
  auto module = fixture.makeModule(std::move(options));

  ASSERT_NO_THROW(module.registerOn(fixture.system));
  const auto u_negative = fixture.system.findFieldByName("u_negative");
  const auto u_positive = fixture.system.findFieldByName("u_positive");
  ASSERT_NE(u_negative, FE::INVALID_FIELD_ID);
  ASSERT_NE(u_positive, FE::INVALID_FIELD_ID);
  const auto shared_descriptors = static_cast<std::size_t>(std::count_if(
      fixture.system.boundaryConditionDescriptors().begin(),
      fixture.system.boundaryConditionDescriptors().end(),
      [&](const auto& descriptor) {
        return descriptor.boundary_marker == 7 &&
               (descriptor.primary_variable.field_id == u_negative ||
                descriptor.primary_variable.field_id == u_positive);
      }));
  EXPECT_EQ(shared_descriptors, 4u);
  const auto artifact = module.effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->json.find("\"shared_velocity_dirichlet_count\":1"),
            std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"shared_velocity_dirichlet_policy\":\"identical_external_data_on_both_phase_restrictions\""),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"shared_velocity_dirichlet\":[{\"marker\":7,\"active_components\":[true,true],\"values\":[{\"kind\":\"literal\",\"value\":1},{\"kind\":\"time_coefficient\"}]}]"),
      std::string::npos);
}

TEST(IncompressibleTwoFluidModule,
     EffectiveArtifactDistinguishesStokesAndConvectiveMomentum) {
  TwoFluidRegistrationFixture stokes_fixture;
  ns::IncompressibleTwoFluidOptions stokes_options;
  stokes_options.enable_convection = false;
  auto stokes_module = stokes_fixture.makeModule(std::move(stokes_options));
  stokes_module.registerOn(stokes_fixture.system);
  const auto stokes_artifact =
      stokes_module.effectiveConfigurationArtifact();
  ASSERT_TRUE(stokes_artifact.has_value());
  EXPECT_NE(stokes_artifact->json.find("\"momentum_operator\":\"stokes\""),
            std::string::npos);

  TwoFluidRegistrationFixture convective_fixture;
  auto convective_module = convective_fixture.makeModule({});
  convective_module.registerOn(convective_fixture.system);
  const auto convective_artifact =
      convective_module.effectiveConfigurationArtifact();
  ASSERT_TRUE(convective_artifact.has_value());
  EXPECT_NE(
      convective_artifact->json.find(
          "\"momentum_operator\":\"navier_stokes\""),
      std::string::npos);
}

TEST(IncompressibleTwoFluidModule,
     SharedNonhomogeneousVelocityBoundaryOwnsInactiveExteriorTraceDofs) {
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires native mesh support.";
#else
  constexpr int wall_marker = 73;
  auto mesh = makeTwoTriangleTwoFluidBoundaryMesh(wall_marker);
  auto pressure_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
  auto velocity_space =
      std::make_shared<FE::spaces::ProductSpace>(pressure_space, 2);
  FE::systems::FESystem system(mesh);
  (void)system.addField(FE::systems::FieldSpec{
      .name = "level_set",
      .space = pressure_space,
      .components = 1,
      .source_kind = FE::systems::FieldSourceKind::PrescribedData,
  });

  ns::IncompressibleTwoFluidOptions options;
  options.interface_marker = 7301;
  options.enable_convection = false;
  ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC wall;
  wall.boundary_marker = wall_marker;
  wall.active_components = {{true, true, false}};
  wall.value[0] =
      ns::IncompressibleNavierStokesVMSOptions::ScalarValue{1.25};
  wall.value[1] =
      ns::IncompressibleNavierStokesVMSOptions::ScalarValue{-0.75};
  options.shared_velocity_dirichlet.push_back(std::move(wall));

  ns::IncompressibleTwoFluidModule module(
      velocity_space,
      pressure_space,
      velocity_space,
      pressure_space,
      std::move(options));
  ASSERT_NO_THROW(module.registerOn(system));
  ASSERT_NO_THROW(system.setup({}));

  constexpr std::array<FE::Real, 2> expected{{1.25, -0.75}};
  const auto expect_physical_trace = [&] {
    for (const std::string_view field : {"u_negative", "u_positive"}) {
      for (int component = 0; component < 2; ++component) {
        const auto dofs = system.fieldMap()
                              .getComponentDofs(
                                  std::string(field),
                                  static_cast<FE::LocalIndex>(component))
                              .toVector();
        ASSERT_EQ(dofs.size(), 4u);
        for (const auto dof : dofs) {
          const auto constraint = system.constraints().getConstraint(dof);
          ASSERT_TRUE(constraint.has_value());
          EXPECT_TRUE(constraint->isDirichlet());
          EXPECT_NEAR(
              constraint->inhomogeneity,
              expected[static_cast<std::size_t>(component)],
              FE::Real{1.0e-14});
        }
      }
    }
  };
  expect_physical_trace();
#endif
}

TEST(IncompressibleTwoFluidModule,
     ExplicitSharedPressureGaugePinsRequestedGlobalVertex) {
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires native mesh support.";
#else
  constexpr int wall_marker = 73;
  auto mesh = makeTwoTriangleTwoFluidBoundaryMesh(wall_marker);
  mesh->base().set_vertex_gids({101, 102, 103, 104});
  auto pressure_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
  auto velocity_space =
      std::make_shared<FE::spaces::ProductSpace>(pressure_space, 2);
  FE::systems::FESystem system(mesh);
  (void)system.addField(FE::systems::FieldSpec{
      .name = "level_set",
      .space = pressure_space,
      .components = 1,
      .source_kind = FE::systems::FieldSourceKind::PrescribedData,
  });

  ns::IncompressibleTwoFluidOptions options;
  options.interface_marker = 7301;
  options.enable_convection = false;
  options.shared_pressure_gauge =
      ns::IncompressibleTwoFluidSharedPressureGauge{
          .vertex_gid = 101,
          .pressure = FE::Real{4.25},
      };
  ns::IncompressibleTwoFluidModule module(
      velocity_space,
      pressure_space,
      velocity_space,
      pressure_space,
      std::move(options));
  ASSERT_NO_THROW(module.registerOn(system));

  const auto artifact = module.effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(
      artifact->json.find(
          "\"shared_gauge_policy\":\"explicit_global_vertex_gid\""),
      std::string::npos);
  EXPECT_NE(artifact->json.find("\"shared_gauge_vertex_gid\":101"),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"shared_gauge_value\":4.25"),
            std::string::npos);

  ASSERT_NO_THROW(system.setup({}));
  const auto pressure = system.findFieldByName("p_negative");
  ASSERT_NE(pressure, FE::INVALID_FIELD_ID);
  const auto* entity_map =
      system.fieldDofHandler(pressure).getEntityDofMap();
  ASSERT_NE(entity_map, nullptr);
  const auto vertex_dofs = entity_map->getVertexDofs(0);
  ASSERT_EQ(vertex_dofs.size(), 1u);
  const auto constraint = system.constraints().getConstraint(
      system.fieldDofOffset(pressure) + vertex_dofs.front());
  ASSERT_TRUE(constraint.has_value());
  EXPECT_TRUE(constraint->isDirichlet());
  EXPECT_DOUBLE_EQ(constraint->inhomogeneity, FE::Real{4.25});
#endif
}

TEST(IncompressibleTwoFluidInterface,
     ViscosityWeightsAreComplementaryAndSideReversalInvariant) {
  const auto parameters = makeParameters();
  const auto weights = ns::incompressibleTwoFluidInterfaceWeights(parameters);
  EXPECT_NEAR(weights.negative_traction + weights.positive_traction,
              FE::Real{1.0},
              FE::Real{4.0} * std::numeric_limits<FE::Real>::epsilon());
  EXPECT_DOUBLE_EQ(weights.negative_complement, weights.positive_traction);
  EXPECT_DOUBLE_EQ(weights.positive_complement, weights.negative_traction);

  auto reversed = parameters;
  std::swap(reversed.negative_density, reversed.positive_density);
  std::swap(reversed.negative_viscosity, reversed.positive_viscosity);
  const auto reversed_weights =
      ns::incompressibleTwoFluidInterfaceWeights(reversed);
  EXPECT_DOUBLE_EQ(reversed_weights.negative_traction,
                   weights.positive_traction);
  EXPECT_DOUBLE_EQ(reversed_weights.positive_traction,
                   weights.negative_traction);
  EXPECT_DOUBLE_EQ(reversed_weights.harmonic_viscosity,
                   weights.harmonic_viscosity);
  EXPECT_DOUBLE_EQ(reversed_weights.harmonic_density, weights.harmonic_density);
}

TEST(IncompressibleTwoFluidInterface,
     WeightsRemainFiniteWithoutOverflowAtExtremeMaterialScales) {
  auto parameters = makeParameters();
  parameters.negative_density = std::numeric_limits<FE::Real>::max();
  parameters.positive_density =
      std::numeric_limits<FE::Real>::max() / FE::Real{2.0};
  parameters.negative_viscosity =
      std::numeric_limits<FE::Real>::max() / FE::Real{4.0};
  parameters.positive_viscosity =
      std::numeric_limits<FE::Real>::max() / FE::Real{2.0};
  const auto weights = ns::incompressibleTwoFluidInterfaceWeights(parameters);
  EXPECT_TRUE(std::isfinite(weights.negative_traction));
  EXPECT_TRUE(std::isfinite(weights.positive_traction));
  EXPECT_TRUE(std::isfinite(weights.harmonic_viscosity));
  EXPECT_TRUE(std::isfinite(weights.harmonic_density));
  EXPECT_NEAR(weights.negative_traction + weights.positive_traction,
              FE::Real{1.0},
              FE::Real{4.0} * std::numeric_limits<FE::Real>::epsilon());
}

TEST(IncompressibleTwoFluidInterface,
     ResidualCompilesAsFourFieldGeneratedInterfaceCoupling) {
  const auto forms = build(makeParameters());
  EXPECT_TRUE(forms.consistency.isValid());
  EXPECT_TRUE(forms.adjoint.isValid());
  EXPECT_TRUE(forms.penalty.isValid());
  EXPECT_TRUE(forms.surface_energy.isValid());
  EXPECT_TRUE(forms.residual.isValid());

  FE::forms::FormCompiler compiler;
  const auto mixed =
      compiler.compileMixed(forms.residual, FE::forms::FormKind::Residual);
  EXPECT_EQ(mixed.numTestFields(), 4u);
  EXPECT_EQ(mixed.numTrialFields(), 4u);
  EXPECT_TRUE(mixed.domainSummary().has_interface_face_terms);
  ASSERT_EQ(mixed.domainSummary().interface_markers.size(), 1u);
  EXPECT_EQ(mixed.domainSummary().interface_markers.front(), 71);
  EXPECT_EQ(mixed.numActiveBlocks(), 12u);
}

TEST(IncompressibleTwoFluidInterface,
     LiteralZeroSurfaceTensionOmitsSurfaceEnergyTree) {
  auto parameters = makeParameters();
  parameters.surface_tension = FE::Real{0.0};
  const auto forms = build(parameters);
  EXPECT_FALSE(forms.surface_energy.isValid());
  EXPECT_TRUE(forms.residual.isValid());
}

TEST(IncompressibleTwoFluidInterface,
     PrescribedPlanarPressureJumpUsesBothComplementaryTestTraces) {
  auto parameters = makeParameters();
  parameters.surface_tension = FE::Real{0.0};
  parameters.prescribed_pressure_jump = FE::Real{3.0};
  const auto forms = build(parameters);
  ASSERT_TRUE(forms.prescribed_pressure_jump.isValid());
  EXPECT_FALSE(forms.surface_energy.isValid());
  EXPECT_TRUE(forms.residual.isValid());
  const auto expression = forms.prescribed_pressure_jump.toString();
  EXPECT_NE(expression.find("inner(n, v_negative)"), std::string::npos);
  EXPECT_NE(expression.find("inner(n, v_positive)"), std::string::npos);
  EXPECT_NE(expression.find("3"), std::string::npos);
}

TEST(IncompressibleTwoFluidInterface,
     PrescribedViscousTractionJumpUsesGlobalVectorAndComposesWithPressure) {
  auto parameters = makeParameters();
  parameters.surface_tension = FE::Real{0.0};
  parameters.prescribed_pressure_jump = FE::Real{3.0};
  parameters.prescribed_viscous_traction_jump =
      std::array<FE::Real, 3>{{
          FE::Real{2.0}, FE::Real{-4.0}, FE::Real{1.0}}};
  const auto forms = build(parameters);

  ASSERT_TRUE(forms.prescribed_pressure_jump.isValid());
  ASSERT_TRUE(forms.prescribed_viscous_traction_jump.isValid());
  EXPECT_TRUE(forms.residual.isValid());
  const auto expression = forms.prescribed_viscous_traction_jump.toString();
  EXPECT_NE(expression.find("inner(as_vector(2, -4, 1), v_negative)"),
            std::string::npos)
      << expression;
  EXPECT_NE(expression.find("inner(as_vector(2, -4, 1), v_positive)"),
            std::string::npos)
      << expression;
}

TEST(IncompressibleTwoFluidInterface,
     TransientPenaltyRetainsEffectiveStepTerminal) {
  auto parameters = makeParameters();
  parameters.include_transient_penalty = true;
  parameters.surface_tension = FE::Real{0.0};
  const auto forms = build(parameters);
  EXPECT_TRUE(expressionContains(forms.penalty,
                                 FE::forms::FormExprType::EffectiveTimeStep));
}

TEST(IncompressibleTwoFluidInterface, RejectsInvalidParameters) {
  auto parameters = makeParameters();
  parameters.dimension = 1;
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.interface_marker = -1;
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.negative_density = FE::Real{0.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.positive_viscosity = std::numeric_limits<FE::Real>::quiet_NaN();
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.nitsche_gamma = FE::Real{-1.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.surface_tension = FE::Real{-1.0};
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.prescribed_pressure_jump =
      std::numeric_limits<FE::Real>::quiet_NaN();
  EXPECT_THROW(build(parameters), std::invalid_argument);

  parameters = makeParameters();
  parameters.prescribed_viscous_traction_jump =
      std::array<FE::Real, 3>{{
          FE::Real{0.0},
          std::numeric_limits<FE::Real>::quiet_NaN(),
          FE::Real{0.0}}};
  EXPECT_THROW(build(parameters), std::invalid_argument);
}

TEST(IncompressibleTwoFluidInterface, RejectsMissingExpression) {
  const auto expressions = makeExpressions();
  EXPECT_THROW((void)ns::buildIncompressibleTwoFluidInterfaceForms(
                   FE::forms::FormExpr{}, expressions.p_negative,
                   expressions.v_negative, expressions.q_negative,
                   expressions.u_positive, expressions.p_positive,
                   expressions.v_positive, expressions.q_positive,
                   makeParameters()),
               std::invalid_argument);
}

TEST(IncompressibleTwoFluidDiagnostics,
     EvaluatesRawInterfaceAndPhaseMeasuresOnOneSnapshot) {
  RevisionTrackedTriangleMeshAccess mesh;
  const auto snapshot = makeTwoFluidDiagnosticSnapshot(mesh, 71);

  FE::interfaces::IncompressibleTwoFluidPhaseEvaluators negative;
  negative.velocity.value =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{1.0, 2.0, 0.0}};
      };
  negative.velocity.physical_gradient =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return FE::interfaces::
            FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                {{1.0, 2.0, 0.0}},
                {{3.0, 4.0, 0.0}},
                {{0.0, 0.0, 0.0}},
            }};
      };
  negative.pressure.value =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{5.0};
      };
  negative.pressure.reference_gradient =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{2.0, -1.0, 0.0}};
      };

  FE::interfaces::IncompressibleTwoFluidPhaseEvaluators positive;
  positive.velocity.value =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{0.0, 1.0, 0.0}};
      };
  positive.velocity.physical_gradient =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return FE::interfaces::
            FreeSurfaceDiscreteFunctionalPhysicalGradient{{
                {{-1.0, 1.0, 0.0}},
                {{2.0, 0.0, 0.0}},
                {{0.0, 0.0, 0.0}},
            }};
      };
  positive.pressure.value =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{2.0};
      };
  positive.pressure.reference_gradient =
      [](FE::GlobalIndex,
         const std::array<FE::Real, 3>&,
         const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{3.0, -1.0, 0.0}};
      };

  FE::interfaces::IncompressibleTwoFluidDiagnosticParameters parameters{
      .dimension = 2,
      .interface_marker = 71,
      .negative_density = FE::Real{4.0},
      .positive_density = FE::Real{2.0},
      .negative_viscosity = FE::Real{2.0},
      .positive_viscosity = FE::Real{1.0},
      .nitsche_gamma = FE::Real{6.0},
      .surface_tension = FE::Real{0.5},
      .include_transient_penalty = true,
      .prescribed_pressure_jump = FE::Real{3.0},
      .body_force = {{FE::Real{0.5}, FE::Real{-0.25}, FE::Real{0.0}}},
  };
  FE::interfaces::IncompressibleTwoFluidCellMeasureEvaluator cell_measure;
  cell_measure.physical_cell_measure =
      [](FE::GlobalIndex) { return FE::Real{0.5}; };

  const auto local =
      FE::interfaces::evaluateLocalIncompressibleTwoFluidDiagnostics(
          *snapshot,
          parameters,
          negative,
          positive,
          cell_measure,
          FE::Real{0.25});
  const auto state =
      FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
          local, parameters);
  constexpr FE::Real tolerance = FE::Real{1.0e-13};

  EXPECT_EQ(state.snapshot_revision_key,
            snapshot->revision().snapshot_revision_key);
  ASSERT_TRUE(state.transient_penalty_effective_dt.has_value());
  EXPECT_NEAR(*state.transient_penalty_effective_dt,
              FE::Real{0.25},
              tolerance);
  EXPECT_EQ(state.interface_quadrature_point_count, 1u);
  EXPECT_NEAR(state.interface_measure, FE::Real{0.5}, tolerance);
  EXPECT_NEAR(state.velocity_jump_squared, FE::Real{1.0}, tolerance);
  EXPECT_NEAR(state.normal_velocity_jump_squared, FE::Real{0.5}, tolerance);
  EXPECT_NEAR(state.tangential_velocity_jump_squared,
              FE::Real{0.5},
              tolerance);
  EXPECT_NEAR(state.negative_normal_flux, FE::Real{0.5}, tolerance);
  EXPECT_NEAR(state.positive_normal_flux, FE::Real{0.0}, tolerance);
  EXPECT_NEAR(state.normal_flux_jump, FE::Real{0.5}, tolerance);
  EXPECT_NEAR(state.negative_mass_flux, FE::Real{2.0}, tolerance);
  EXPECT_NEAR(state.positive_mass_flux, FE::Real{0.0}, tolerance);
  EXPECT_NEAR(state.negative_traction_integral[0],
              FE::Real{-0.5},
              tolerance);
  EXPECT_NEAR(state.negative_traction_integral[1],
              FE::Real{5.0},
              tolerance);
  EXPECT_NEAR(state.positive_traction_integral[0],
              FE::Real{-2.0},
              tolerance);
  EXPECT_NEAR(state.positive_traction_integral[1],
              FE::Real{1.5},
              tolerance);
  EXPECT_NEAR(state.traction_jump_integral[0],
              FE::Real{1.5},
              tolerance);
  EXPECT_NEAR(state.traction_jump_integral[1],
              FE::Real{3.5},
              tolerance);
  EXPECT_NEAR(state.traction_jump_squared, FE::Real{29.0}, tolerance);
  EXPECT_NEAR(state.traction_jump_normal_integral,
              FE::Real{1.5},
              tolerance);
  EXPECT_NEAR(state.pressure_jump_integral, FE::Real{1.5}, tolerance);
  EXPECT_NEAR(state.mean_pressure_jump, FE::Real{3.0}, tolerance);
  EXPECT_NEAR(state.pressure_jump_squared, FE::Real{4.5}, tolerance);
  ASSERT_TRUE(state.prescribed_pressure_jump_error_squared.has_value());
  EXPECT_NEAR(*state.prescribed_pressure_jump_error_squared,
              FE::Real{0.0},
              tolerance);
  ASSERT_TRUE(state.prescribed_stress_jump_residual_squared.has_value());
  EXPECT_NEAR(*state.prescribed_stress_jump_residual_squared,
              FE::Real{42.5},
              tolerance);
  EXPECT_NEAR(state.surface_energy_work,
              FE::Real{2.0} / FE::Real{3.0},
              tolerance);
  EXPECT_NEAR(state.nitsche_consistency_work,
              FE::Real{-7.0} / FE::Real{6.0},
              tolerance);
  EXPECT_NEAR(state.nitsche_adjoint_work,
              state.nitsche_consistency_work,
              tolerance);
  EXPECT_NEAR(state.nitsche_penalty_work, FE::Real{132.0}, tolerance);

  EXPECT_EQ(state.negative_phase.quadrature_point_count, 1u);
  EXPECT_NEAR(state.negative_phase.volume, FE::Real{0.375}, tolerance);
  EXPECT_NEAR(state.negative_phase.mass, FE::Real{1.5}, tolerance);
  EXPECT_NEAR(state.negative_phase.momentum[0], FE::Real{1.5}, tolerance);
  EXPECT_NEAR(state.negative_phase.momentum[1], FE::Real{3.0}, tolerance);
  EXPECT_NEAR(state.negative_phase.kinetic_energy,
              FE::Real{3.75},
              tolerance);
  EXPECT_NEAR(state.negative_phase.pressure_integral,
              FE::Real{1.875},
              tolerance);
  EXPECT_NEAR(state.negative_phase.mean_pressure,
              FE::Real{5.0},
              tolerance);
  EXPECT_NEAR(state.negative_phase.pressure_squared_integral,
              FE::Real{9.375},
              tolerance);
  EXPECT_EQ(state.negative_phase.pressure_gradient_integral,
            (std::array<FE::Real, 3>{{0.75, -0.375, 0.0}}));
  EXPECT_EQ(state.negative_phase.body_force_density_integral,
            (std::array<FE::Real, 3>{{0.75, -0.375, 0.0}}));
  EXPECT_EQ(state.negative_phase.hydrostatic_residual_integral,
            (std::array<FE::Real, 3>{{0.0, 0.0, 0.0}}));
  EXPECT_NEAR(state.negative_phase.hydrostatic_residual_squared,
              FE::Real{0.0},
              tolerance);
  EXPECT_EQ(state.positive_phase.quadrature_point_count, 1u);
  EXPECT_NEAR(state.positive_phase.volume, FE::Real{0.125}, tolerance);
  EXPECT_NEAR(state.positive_phase.mass, FE::Real{0.25}, tolerance);
  EXPECT_NEAR(state.positive_phase.momentum[0], FE::Real{0.0}, tolerance);
  EXPECT_NEAR(state.positive_phase.momentum[1], FE::Real{0.25}, tolerance);
  EXPECT_NEAR(state.positive_phase.kinetic_energy,
              FE::Real{0.125},
              tolerance);
  EXPECT_NEAR(state.positive_phase.pressure_integral,
              FE::Real{0.25},
              tolerance);
  EXPECT_NEAR(state.positive_phase.mean_pressure,
              FE::Real{2.0},
              tolerance);
  EXPECT_NEAR(state.positive_phase.pressure_squared_integral,
              FE::Real{0.5},
              tolerance);
  EXPECT_EQ(state.positive_phase.pressure_gradient_integral,
            (std::array<FE::Real, 3>{{0.375, -0.125, 0.0}}));
  EXPECT_EQ(state.positive_phase.body_force_density_integral,
            (std::array<FE::Real, 3>{{0.125, -0.0625, 0.0}}));
  EXPECT_EQ(state.positive_phase.hydrostatic_residual_integral,
            (std::array<FE::Real, 3>{{0.25, -0.0625, 0.0}}));
  EXPECT_NEAR(state.positive_phase.hydrostatic_residual_squared,
              FE::Real{0.53125},
              tolerance);

  auto composed_parameters = parameters;
  composed_parameters.prescribed_viscous_traction_jump =
      std::array<FE::Real, 3>{{
          FE::Real{6.0}, FE::Real{7.0}, FE::Real{0.0}}};
  const auto composed_local =
      FE::interfaces::evaluateLocalIncompressibleTwoFluidDiagnostics(
          *snapshot,
          composed_parameters,
          negative,
          positive,
          cell_measure,
          FE::Real{0.25});
  const auto composed_state =
      FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
          composed_local, composed_parameters);
  EXPECT_EQ(composed_state.interface_normal_integral,
            (std::array<FE::Real, 3>{{0.5, 0.0, 0.0}}));
  EXPECT_EQ(composed_state.viscous_traction_jump_integral,
            (std::array<FE::Real, 3>{{3.0, 3.5, 0.0}}));
  EXPECT_NEAR(composed_state.viscous_traction_jump_squared,
              FE::Real{42.5},
              tolerance);
  ASSERT_TRUE(composed_state
                  .prescribed_viscous_traction_jump_error_squared
                  .has_value());
  EXPECT_NEAR(
      *composed_state.prescribed_viscous_traction_jump_error_squared,
      FE::Real{0.0},
      tolerance);
  ASSERT_TRUE(
      composed_state.prescribed_stress_jump_residual_squared.has_value());
  EXPECT_NEAR(*composed_state.prescribed_stress_jump_residual_squared,
              FE::Real{0.0},
              tolerance);

  auto missing_pressure_gradient = negative;
  missing_pressure_gradient.pressure.reference_gradient = {};
  EXPECT_THROW(
      (void)FE::interfaces::evaluateLocalIncompressibleTwoFluidDiagnostics(
          *snapshot,
          parameters,
          missing_pressure_gradient,
          positive,
          cell_measure,
          FE::Real{0.25}),
      std::invalid_argument);

  auto nonfinite_body_force = parameters;
  nonfinite_body_force.body_force[0] =
      std::numeric_limits<FE::Real>::quiet_NaN();
  EXPECT_THROW(
      (void)FE::interfaces::evaluateLocalIncompressibleTwoFluidDiagnostics(
          *snapshot,
          nonfinite_body_force,
          negative,
          positive,
          cell_measure,
          FE::Real{0.25}),
      std::invalid_argument);

  auto out_of_plane_body_force = parameters;
  out_of_plane_body_force.body_force[2] = FE::Real{1.0};
  EXPECT_THROW(
      (void)FE::interfaces::evaluateLocalIncompressibleTwoFluidDiagnostics(
          *snapshot,
          out_of_plane_body_force,
          negative,
          positive,
          cell_measure,
          FE::Real{0.25}),
      std::invalid_argument);
}

TEST(IncompressibleTwoFluidDiagnostics,
     SeparatesViscousTargetErrorFromComposedStressTargetResidual) {
  FE::interfaces::IncompressibleTwoFluidDiagnosticParameters parameters{
      .dimension = 2,
      .interface_marker = 71,
      .negative_density = FE::Real{4.0},
      .positive_density = FE::Real{2.0},
      .negative_viscosity = FE::Real{2.0},
      .positive_viscosity = FE::Real{1.0},
      .nitsche_gamma = FE::Real{6.0},
      .surface_tension = FE::Real{0.0},
      .include_transient_penalty = false,
      .prescribed_pressure_jump = FE::Real{3.0},
      .prescribed_viscous_traction_jump =
          std::array<FE::Real, 3>{{
              FE::Real{6.0}, FE::Real{7.0}, FE::Real{0.0}}},
  };
  FE::interfaces::IncompressibleTwoFluidDiagnosticAccumulator accumulator;
  accumulator.snapshot_revision_key = 91u;
  accumulator.owned_interface_quadrature_point_count = 1u;
  accumulator.interface_measure = FE::Real{0.5};
  accumulator.interface_normal_integral =
      {{FE::Real{0.5}, FE::Real{0.0}, FE::Real{0.0}}};
  accumulator.negative_traction_integral =
      {{FE::Real{-0.5}, FE::Real{5.0}, FE::Real{0.0}}};
  accumulator.positive_traction_integral =
      {{FE::Real{-2.0}, FE::Real{1.5}, FE::Real{0.0}}};
  accumulator.traction_jump_integral =
      {{FE::Real{1.5}, FE::Real{3.5}, FE::Real{0.0}}};
  accumulator.traction_jump_normal_integral = FE::Real{1.5};
  accumulator.traction_jump_squared = FE::Real{29.0};
  accumulator.negative_viscous_traction_integral =
      {{FE::Real{2.0}, FE::Real{5.0}, FE::Real{0.0}}};
  accumulator.positive_viscous_traction_integral =
      {{FE::Real{-1.0}, FE::Real{1.5}, FE::Real{0.0}}};
  accumulator.viscous_traction_jump_integral =
      {{FE::Real{3.0}, FE::Real{3.5}, FE::Real{0.0}}};
  accumulator.viscous_traction_jump_squared = FE::Real{42.5};
  accumulator.pressure_jump_integral = FE::Real{1.5};
  accumulator.pressure_jump_squared = FE::Real{4.5};
  accumulator.prescribed_pressure_jump_error_squared = FE::Real{0.0};
  accumulator.prescribed_viscous_traction_jump_error_squared = FE::Real{0.0};
  accumulator.prescribed_stress_jump_residual_squared = FE::Real{0.0};
  accumulator.negative_phase.owned_quadrature_point_count = 1u;
  accumulator.negative_phase.volume = FE::Real{0.25};
  accumulator.positive_phase.owned_quadrature_point_count = 1u;
  accumulator.positive_phase.volume = FE::Real{0.25};

  const auto state =
      FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
          accumulator, parameters);
  EXPECT_EQ(state.negative_viscous_traction_integral,
            accumulator.negative_viscous_traction_integral);
  EXPECT_EQ(state.positive_viscous_traction_integral,
            accumulator.positive_viscous_traction_integral);
  EXPECT_EQ(state.viscous_traction_jump_integral,
            accumulator.viscous_traction_jump_integral);
  EXPECT_DOUBLE_EQ(state.viscous_traction_jump_squared, FE::Real{42.5});
  ASSERT_TRUE(
      state.prescribed_viscous_traction_jump_error_squared.has_value());
  EXPECT_DOUBLE_EQ(
      *state.prescribed_viscous_traction_jump_error_squared, FE::Real{0.0});
  ASSERT_TRUE(state.prescribed_stress_jump_residual_squared.has_value());
  EXPECT_DOUBLE_EQ(
      *state.prescribed_stress_jump_residual_squared, FE::Real{0.0});
}

TEST(IncompressibleTwoFluidDiagnostics,
     AcceptsHighContrastHydrostaticIdentityWithinOperandScaledRoundoff) {
  FE::interfaces::IncompressibleTwoFluidDiagnosticParameters parameters{
      .dimension = 2,
      .interface_marker = 71,
      .negative_density = FE::Real{10000.0},
      .positive_density = FE::Real{1.0},
      .negative_viscosity = FE::Real{0.01},
      .positive_viscosity = FE::Real{0.001},
      .nitsche_gamma = FE::Real{24.0},
      .surface_tension = FE::Real{0.0},
      .include_transient_penalty = false,
      .body_force = {{FE::Real{-9.81}, FE::Real{0.0}, FE::Real{0.0}}},
  };
  FE::interfaces::IncompressibleTwoFluidDiagnosticAccumulator accumulator;
  accumulator.snapshot_revision_key = 1u;
  accumulator.owned_interface_quadrature_point_count = 1u;
  accumulator.interface_measure = FE::Real{1.0};
  accumulator.negative_phase.owned_quadrature_point_count = 1u;
  accumulator.negative_phase.volume = FE::Real{0.37};
  accumulator.positive_phase.owned_quadrature_point_count = 1u;
  accumulator.positive_phase.volume = FE::Real{0.63};

  const FE::Real negative_body_force_moment =
      parameters.negative_density * parameters.body_force[0] *
      accumulator.negative_phase.volume;
  FE::Real rounded_pressure_gradient_moment = negative_body_force_moment;
  for (int offset = 0; offset < 8; ++offset) {
    rounded_pressure_gradient_moment = std::nextafter(
        rounded_pressure_gradient_moment,
        std::numeric_limits<FE::Real>::infinity());
  }
  accumulator.negative_phase.pressure_gradient_integral[0] =
      rounded_pressure_gradient_moment;

  const FE::Real positive_body_force_moment =
      parameters.positive_density * parameters.body_force[0] *
      accumulator.positive_phase.volume;
  accumulator.positive_phase.pressure_gradient_integral[0] =
      positive_body_force_moment;

  EXPECT_NO_THROW(
      (void)FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
          accumulator, parameters));
}

TEST(IncompressibleTwoFluidDiagnostics,
     AcceptsTractionDifferenceWithinPhaseOperandScaledRoundoff) {
  FE::interfaces::IncompressibleTwoFluidDiagnosticParameters parameters{
      .dimension = 2,
      .interface_marker = 71,
      .negative_density = FE::Real{10000.0},
      .positive_density = FE::Real{1.0},
      .negative_viscosity = FE::Real{0.01},
      .positive_viscosity = FE::Real{0.001},
      .nitsche_gamma = FE::Real{24.0},
      .surface_tension = FE::Real{0.0},
      .include_transient_penalty = false,
  };
  FE::interfaces::IncompressibleTwoFluidDiagnosticAccumulator accumulator;
  accumulator.snapshot_revision_key = 1u;
  accumulator.owned_interface_quadrature_point_count = 1u;
  accumulator.interface_measure = FE::Real{1.0};
  accumulator.negative_phase.owned_quadrature_point_count = 1u;
  accumulator.negative_phase.volume = FE::Real{0.37};
  accumulator.positive_phase.owned_quadrature_point_count = 1u;
  accumulator.positive_phase.volume = FE::Real{0.63};

  constexpr FE::Real hydrostatic_traction = FE::Real{36297.0};
  accumulator.negative_traction_integral[0] = hydrostatic_traction;
  accumulator.positive_traction_integral[0] = std::nextafter(
      hydrostatic_traction,
      -std::numeric_limits<FE::Real>::infinity());

  EXPECT_NO_THROW(
      (void)FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
          accumulator, parameters));
}

TEST(IncompressibleTwoFluidDiagnostics,
     MomentumReconciliationAcceptsMeasurePreservingConstantVelocity) {
  FE::interfaces::IncompressibleTwoFluidDiagnosticState raw;
  raw.snapshot_revision_key = 101u;
  raw.negative_phase.side = FE::geometry::CutIntegrationSide::Negative;
  raw.negative_phase.density = FE::Real{1000.0};
  raw.negative_phase.volume = FE::Real{0.2};
  raw.negative_phase.mass = FE::Real{200.0};
  raw.negative_phase.momentum = {{100.0, -50.0, 0.0}};
  raw.negative_phase.kinetic_energy = FE::Real{31.25};
  raw.positive_phase.side = FE::geometry::CutIntegrationSide::Positive;
  raw.positive_phase.density = FE::Real{1.0};
  raw.positive_phase.volume = FE::Real{0.8};
  raw.positive_phase.mass = FE::Real{0.8};
  raw.positive_phase.momentum = {{0.4, -0.2, 0.0}};
  raw.positive_phase.kinetic_energy = FE::Real{0.125};
  auto corrected = raw;
  corrected.snapshot_revision_key = 102u;

  FE::interfaces::FreeSurfaceGeometryRevision raw_geometry;
  raw_geometry.source_id = "field:0";
  raw_geometry.domain_id = "material_interface";
  raw_geometry.interface_marker = 71;
  raw_geometry.source_value_revision = 11u;
  raw_geometry.snapshot_revision_key = 101u;
  auto corrected_geometry = raw_geometry;
  corrected_geometry.source_value_revision = 12u;
  corrected_geometry.snapshot_revision_key = 102u;

  const auto record = FE::interfaces::
      buildIncompressibleTwoFluidMomentumReconciliation(
          71,
          raw_geometry,
          raw,
          /*raw_algebraic_revision=*/201u,
          corrected_geometry,
          corrected,
          /*corrected_algebraic_revision=*/202u,
          FE::Real{1.0e-12});
  EXPECT_TRUE(record.satisfied);
  EXPECT_FALSE(record.velocity_update_applied);
  EXPECT_EQ(record.raw_algebraic_revision, 201u);
  EXPECT_EQ(record.corrected_algebraic_revision, 202u);
  EXPECT_EQ(record.raw_geometry_revision.snapshot_revision_key, 101u);
  EXPECT_EQ(record.corrected_geometry_revision.snapshot_revision_key, 102u);
  EXPECT_DOUBLE_EQ(record.negative_phase.momentum_delta_norm, 0.0);
  EXPECT_DOUBLE_EQ(record.positive_phase.momentum_delta_norm, 0.0);
  EXPECT_TRUE(record.negative_phase.satisfied);
  EXPECT_TRUE(record.positive_phase.satisfied);
  EXPECT_NO_THROW(
      FE::interfaces::validateIncompressibleTwoFluidMomentumReconciliation(
          record));
}

TEST(IncompressibleTwoFluidDiagnostics,
     MomentumReconciliationRejectsUntrackedPhaseMomentumChange) {
  FE::interfaces::IncompressibleTwoFluidDiagnosticState raw;
  raw.snapshot_revision_key = 301u;
  raw.negative_phase.side = FE::geometry::CutIntegrationSide::Negative;
  raw.negative_phase.density = FE::Real{1000.0};
  raw.negative_phase.volume = FE::Real{0.2};
  raw.negative_phase.mass = FE::Real{200.0};
  raw.negative_phase.momentum = {{100.0, 0.0, 0.0}};
  raw.negative_phase.kinetic_energy = FE::Real{25.0};
  raw.positive_phase.side = FE::geometry::CutIntegrationSide::Positive;
  raw.positive_phase.density = FE::Real{1.0};
  raw.positive_phase.volume = FE::Real{0.8};
  raw.positive_phase.mass = FE::Real{0.8};
  raw.positive_phase.momentum = {{0.4, 0.0, 0.0}};
  raw.positive_phase.kinetic_energy = FE::Real{0.1};
  auto corrected = raw;
  corrected.snapshot_revision_key = 302u;
  corrected.negative_phase.momentum[0] += FE::Real{0.1};
  corrected.negative_phase.kinetic_energy = FE::Real{25.1};

  FE::interfaces::FreeSurfaceGeometryRevision raw_geometry;
  raw_geometry.source_id = "field:0";
  raw_geometry.domain_id = "material_interface";
  raw_geometry.interface_marker = 71;
  raw_geometry.source_value_revision = 21u;
  raw_geometry.snapshot_revision_key = 301u;
  auto corrected_geometry = raw_geometry;
  corrected_geometry.source_value_revision = 22u;
  corrected_geometry.snapshot_revision_key = 302u;

  const auto record = FE::interfaces::
      buildIncompressibleTwoFluidMomentumReconciliation(
          71,
          raw_geometry,
          raw,
          /*raw_algebraic_revision=*/401u,
          corrected_geometry,
          corrected,
          /*corrected_algebraic_revision=*/402u,
          FE::Real{1.0e-6});
  EXPECT_FALSE(record.satisfied);
  EXPECT_FALSE(record.negative_phase.satisfied);
  EXPECT_TRUE(record.positive_phase.satisfied);
  EXPECT_GT(record.negative_phase.momentum_delta_norm,
            record.negative_phase.allowed_momentum_delta);
  EXPECT_NO_THROW(
      FE::interfaces::validateIncompressibleTwoFluidMomentumReconciliation(
          record));

  auto malformed = record;
  malformed.negative_phase.momentum_delta[0] = FE::Real{0.0};
  EXPECT_THROW(
      FE::interfaces::validateIncompressibleTwoFluidMomentumReconciliation(
          malformed),
      std::invalid_argument);
  EXPECT_THROW(
      FE::interfaces::buildIncompressibleTwoFluidMomentumReconciliation(
          71,
          raw_geometry,
          raw,
          401u,
          corrected_geometry,
          corrected,
          402u,
          FE::Real{0.0}),
      std::invalid_argument);
}

TEST(IncompressibleTwoFluidInterface,
     InternalInterfacePenaltyAssemblesWithTheCutInterfaceMeasure) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.interface_marker = 71;
  options.surface_tension = FE::Real{0.0};
  options.include_transient_interface_penalty = false;
  options.enable_convection = false;
  auto module = fixture.makeModule(std::move(options));
  module.registerOn(fixture.system);

  auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
  context->addFreeSurfaceGeometrySnapshot(
      makeTwoFluidDiagnosticSnapshot(*fixture.mesh, 71));
  FE::assembly::CutFacetSetHandle facet_set;
  facet_set.marker = 71;
  facet_set.name = "generated-cut-adjacent-facets";
  context->addFacetSetHandle(std::move(facet_set));
  fixture.system.setCutIntegrationContext(context);
  fixture.system.setup({});

  const auto dofs = fixture.system.dofHandler().getNumDofs();
  std::vector<FE::Real> solution(
      static_cast<std::size_t>(dofs), FE::Real{0.0});
  const auto previous = solution;
  FE::systems::SystemStateView state;
  state.dt = FE::Real{0.1};
  state.effective_dt = state.dt;
  state.u = std::span<const FE::Real>(solution);
  state.u_prev = std::span<const FE::Real>(previous);
  const FE::systems::BackwardDifferenceIntegrator integrator;
  const auto time_context =
      integrator.buildContext(/*max_time_derivative_order=*/1, state);
  state.time_integration = &time_context;

  FE::assembly::DenseMatrixView matrix(dofs);
  matrix.zero();
  FE::systems::AssemblyRequest request;
  request.op = "equations";
  request.want_matrix = true;
  const auto result =
      fixture.system.assemble(request, state, &matrix, nullptr);
  ASSERT_TRUE(result.success) << result.error_message;
  FE::Real maximum_entry{0.0};
  for (const auto value : matrix.data()) {
    ASSERT_TRUE(std::isfinite(value));
    maximum_entry = std::max(maximum_entry, std::abs(value));
  }
  EXPECT_GT(maximum_entry, FE::Real{0.0});
}

TEST(IncompressibleTwoFluidModule,
     RegistersFourPhaseFieldsOneInterfaceBlockAndOneSharedGauge) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.surface_tension = FE::Real{0.072};
  options.body_force =
      {{FE::Real{0.5}, FE::Real{-0.25}, FE::Real{0.0}}};
  auto module = fixture.makeModule(std::move(options));
  module.registerOn(fixture.system);

  const auto u_negative = fixture.system.findFieldByName("u_negative");
  const auto p_negative = fixture.system.findFieldByName("p_negative");
  const auto u_positive = fixture.system.findFieldByName("u_positive");
  const auto p_positive = fixture.system.findFieldByName("p_positive");
  EXPECT_NE(u_negative, FE::INVALID_FIELD_ID);
  EXPECT_NE(p_negative, FE::INVALID_FIELD_ID);
  EXPECT_NE(u_positive, FE::INVALID_FIELD_ID);
  EXPECT_NE(p_positive, FE::INVALID_FIELD_ID);
  EXPECT_TRUE(fixture.system.hasOperator("equations"));

  const auto level_set = fixture.system.findFieldByName("level_set");
  FE::interfaces::GeneratedInterfaceMarkerKey marker_key{};
  marker_key.source =
      FE::interfaces::LevelSetInterfaceSource::fromField(level_set);
  marker_key.domain_id = "two_fluid_interface";
  marker_key.isovalue = FE::Real{0.0};
  const auto interface_marker =
      FE::interfaces::stableGeneratedInterfaceMarker(marker_key);
  const auto negative_cut_terms = fixture.system.cutVolumeKernelCount(
      interface_marker, FE::geometry::CutIntegrationSide::Negative);
  const auto positive_cut_terms = fixture.system.cutVolumeKernelCount(
      interface_marker, FE::geometry::CutIntegrationSide::Positive);
  EXPECT_GT(negative_cut_terms, 0u);
  EXPECT_EQ(negative_cut_terms, positive_cut_terms);

  const auto interface_records = static_cast<std::size_t>(std::count_if(
      fixture.system.formulationRecords().begin(),
      fixture.system.formulationRecords().end(),
      [](const auto &record) {
        return record.operator_tag == "equations" &&
               record.active_fields.size() == 4u &&
               std::find(record.active_domains.begin(),
                         record.active_domains.end(),
                         FE::analysis::DomainKind::InterfaceFace) !=
                   record.active_domains.end();
      }));
  EXPECT_EQ(interface_records, 1u);
  EXPECT_TRUE(fixture.system.freeSurfaceResidualWorkDeclarations().empty());
  EXPECT_FALSE(
      fixture.system.freeSurfaceExternalBoundaryEnergyDeclaration().has_value());
  EXPECT_TRUE(
      fixture.system.freeSurfaceDiscreteFunctionalDeclarations().empty());
  const auto diagnostic_declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(diagnostic_declarations.size(), 1u);
  EXPECT_EQ(diagnostic_declarations.front().interface_marker,
            interface_marker);
  EXPECT_EQ(diagnostic_declarations.front().negative_velocity_field,
            u_negative);
  EXPECT_EQ(diagnostic_declarations.front().positive_velocity_field,
            u_positive);
  EXPECT_DOUBLE_EQ(
      diagnostic_declarations.front().parameters.negative_density,
      FE::Real{1000.0});
  EXPECT_DOUBLE_EQ(
      diagnostic_declarations.front().parameters.positive_density,
      FE::Real{1.0});
  EXPECT_EQ(diagnostic_declarations.front().parameters.body_force,
            (std::array<FE::Real, 3>{{0.5, -0.25, 0.0}}));
  EXPECT_FALSE(diagnostic_declarations.front()
                   .parameters.prescribed_pressure_jump.has_value());
  const auto transport_velocity_declarations =
      fixture.system.materialInterfaceTransportVelocityDeclarations();
  ASSERT_EQ(transport_velocity_declarations.size(), 1u);
  const auto& transport_velocity =
      transport_velocity_declarations.front();
  EXPECT_EQ(transport_velocity.interface_marker, interface_marker);
  EXPECT_EQ(transport_velocity.level_set_field, level_set);
  EXPECT_EQ(transport_velocity.negative_velocity_field, u_negative);
  EXPECT_EQ(transport_velocity.positive_velocity_field, u_positive);
  EXPECT_NEAR(
      transport_velocity.negative_trace_weight,
      FE::Real{0.001} / FE::Real{0.00101},
      FE::Real{1.0e-15});
  EXPECT_NEAR(
      transport_velocity.positive_trace_weight,
      FE::Real{0.00001} / FE::Real{0.00101},
      FE::Real{1.0e-15});

  const auto shared_gauge_evidence = static_cast<std::size_t>(std::count_if(
      fixture.system.gaugeRegistry().anchoring().begin(),
      fixture.system.gaugeRegistry().anchoring().end(),
      [](const auto &evidence) {
        return evidence.source ==
               "Coupled two-fluid shared pressure gauge constraint";
      }));
  EXPECT_EQ(shared_gauge_evidence, 2u);
  EXPECT_EQ(fixture.system.gaugeRegistry().anchoring().size(), 2u);

  const auto artifact_before_setup =
      module.effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact_before_setup.has_value());

  fixture.system.setup({});
  std::size_t pressure_pins = 0u;
  for (const auto field : {p_negative, p_positive}) {
    const auto offset = fixture.system.fieldDofOffset(field);
    const auto count = fixture.system.fieldDofHandler(field).getNumDofs();
    for (FE::GlobalIndex local = 0; local < count; ++local) {
      const auto line = fixture.system.constraints().getConstraint(
          offset + local);
      if (line.has_value() && line->entries.empty()) {
        ++pressure_pins;
      }
    }
  }
  EXPECT_EQ(pressure_pins, 1u);

  const auto artifact = module.effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->json, artifact_before_setup->json);
  EXPECT_EQ(artifact->component, "incompressible_two_fluid");
  EXPECT_NE(artifact->json.find("\"artifact_schema_version\":3"),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"shared_gauge_count\":1"),
            std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"shared_gauge_policy\":\"automatic_first_unconstrained_pressure_dof\""),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"conservative_graph_velocity\":\"complementary_weighted_every_node\""),
      std::string::npos);
  EXPECT_NE(artifact->json.find("\"compressible_gas\""),
            std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"prescribed_pressure_jump_applicable\":false"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"prescribed_pressure_jump_mode\":\"manufactured_normal_traction_and_diagnostic_target\""),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"generic_fallback_allowed\":false"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"accepted_stage_evidence\":{\"required\":true"),
      std::string::npos);
  EXPECT_NE(artifact->json.find("\"body_force\":[0.5,-0.25]"),
            std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"hydrostatic_balance_diagnostic\":\"phasewise_integrated_pressure_gradient_minus_density_body_force\""),
      std::string::npos);

  TwoFluidRegistrationFixture repeated_fixture;
  ns::IncompressibleTwoFluidOptions repeated_options;
  repeated_options.surface_tension = FE::Real{0.072};
  repeated_options.body_force =
      {{FE::Real{0.5}, FE::Real{-0.25}, FE::Real{0.0}}};
  auto repeated_module =
      repeated_fixture.makeModule(std::move(repeated_options));
  repeated_module.registerOn(repeated_fixture.system);
  const auto repeated_artifact =
      repeated_module.effectiveConfigurationArtifact();
  ASSERT_TRUE(repeated_artifact.has_value());
  EXPECT_EQ(repeated_artifact->component, artifact->component);
  EXPECT_EQ(repeated_artifact->json, artifact->json);
}

TEST(IncompressibleTwoFluidModule,
     PrescribedPlanarPressureJumpChangesInstalledInterfaceResidual) {
  const auto installed_interface_residual = [](const auto& options) {
    TwoFluidRegistrationFixture fixture;
    auto module = fixture.makeModule(options);
    module.registerOn(fixture.system);
    const auto& records = fixture.system.formulationRecords();
    const auto interface_record = std::find_if(
        records.begin(), records.end(), [](const auto& record) {
          return record.active_fields.size() == 4u;
        });
    EXPECT_NE(interface_record, records.end());
    if (interface_record == records.end() ||
        interface_record->residual_expr == nullptr) {
      return std::string{};
    }
    return interface_record->residual_expr->toString();
  };

  ns::IncompressibleTwoFluidOptions baseline;
  baseline.surface_tension = FE::Real{0.0};
  baseline.include_transient_interface_penalty = false;
  auto prescribed = baseline;
  prescribed.prescribed_pressure_jump = FE::Real{3.0};

  const auto baseline_residual = installed_interface_residual(baseline);
  const auto prescribed_residual = installed_interface_residual(prescribed);
  ASSERT_FALSE(baseline_residual.empty());
  ASSERT_FALSE(prescribed_residual.empty());
  EXPECT_NE(prescribed_residual, baseline_residual);
}

FE::systems::AcceptedTwoFluidStageDiagnosticState
makeAcceptedTwoFluidDiagnosticState(
    const FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration& declaration) {
  const auto& parameters = declaration.parameters;
  FE::interfaces::IncompressibleTwoFluidDiagnosticAccumulator accumulator;
  accumulator.snapshot_revision_key = 91u;
  if (parameters.include_transient_penalty) {
    accumulator.transient_penalty_effective_dt = FE::Real{0.1};
  }
  accumulator.owned_interface_quadrature_point_count = 2u;
  accumulator.interface_measure = FE::Real{0.5};
  accumulator.interface_normal_integral = {{0.5, 0.0, 0.0}};
  accumulator.negative_normal_flux = FE::Real{0.25};
  accumulator.positive_normal_flux = FE::Real{0.25};
  accumulator.pressure_jump_integral = FE::Real{1.5};
  accumulator.pressure_jump_squared = FE::Real{4.5};
  const auto viscous_target =
      parameters.prescribed_viscous_traction_jump.value_or(
          std::array<FE::Real, 3>{});
  const std::array<FE::Real, 3> traction_point{{
      viscous_target[0] - FE::Real{3.0},
      viscous_target[1],
      viscous_target[2],
  }};
  accumulator.negative_traction_integral = {{
      FE::Real{0.5} * traction_point[0],
      FE::Real{0.5} * traction_point[1],
      FE::Real{0.5} * traction_point[2],
  }};
  accumulator.positive_traction_integral = {{0.0, 0.0, 0.0}};
  accumulator.traction_jump_integral =
      accumulator.negative_traction_integral;
  accumulator.traction_jump_normal_integral =
      FE::Real{0.5} * traction_point[0];
  accumulator.traction_jump_squared = FE::Real{0.5} * (
      traction_point[0] * traction_point[0] +
      traction_point[1] * traction_point[1] +
      traction_point[2] * traction_point[2]);
  accumulator.negative_viscous_traction_integral = {{
      FE::Real{0.5} * viscous_target[0],
      FE::Real{0.5} * viscous_target[1],
      FE::Real{0.5} * viscous_target[2],
  }};
  accumulator.viscous_traction_jump_integral =
      accumulator.negative_viscous_traction_integral;
  accumulator.viscous_traction_jump_squared = FE::Real{0.5} * (
      viscous_target[0] * viscous_target[0] +
      viscous_target[1] * viscous_target[1] +
      viscous_target[2] * viscous_target[2]);
  accumulator.negative_phase.owned_quadrature_point_count = 3u;
  accumulator.negative_phase.volume = FE::Real{0.25};
  accumulator.positive_phase.owned_quadrature_point_count = 3u;
  accumulator.positive_phase.volume = FE::Real{0.25};
  if (parameters.prescribed_pressure_jump.has_value()) {
    accumulator.prescribed_pressure_jump_error_squared = FE::Real{0.0};
  }
  if (parameters.prescribed_viscous_traction_jump.has_value()) {
    accumulator.prescribed_viscous_traction_jump_error_squared =
        FE::Real{0.0};
  }
  if (parameters.prescribed_pressure_jump.has_value() ||
      parameters.prescribed_viscous_traction_jump.has_value()) {
    accumulator.prescribed_stress_jump_residual_squared = FE::Real{0.0};
  }
  return FE::systems::AcceptedTwoFluidStageDiagnosticState{
      .interface_marker = declaration.interface_marker,
      .geometry_revision =
          FE::interfaces::FreeSurfaceGeometryRevision{
              .source_id =
                  FE::interfaces::LevelSetInterfaceSource::fromField(
                      declaration.level_set_field)
                      .identifier(),
              .domain_id = declaration.geometry_domain_id,
              .interface_marker = declaration.interface_marker,
              .isovalue = declaration.level_set_isovalue,
              .source_value_revision = 7u,
              .snapshot_revision_key = 91u,
          },
      .diagnostics =
          FE::interfaces::finalizeIncompressibleTwoFluidDiagnostics(
              accumulator, parameters),
  };
}

FE::systems::OperatorStageMeasurementMetadata
makeTwoFluidStageMetadata(std::uint64_t attempt = 1u) {
  return FE::systems::OperatorStageMeasurementMetadata{
      .scheme_name = "BackwardEuler",
      .temporal_order = 1,
      .prospective_accepted_step = 1u,
      .prospective_attempt = attempt,
      .step_start_time = FE::Real{0.0},
      .step_end_time = FE::Real{0.1},
      .state_time = FE::Real{0.1},
      .rate_time = FE::Real{0.1},
      .dt = FE::Real{0.1},
      .expected_stage_geometry =
          FE::systems::OperatorStageGeometryMetadata{},
      .state_revision = 41u,
      .rate_revision = 42u,
  };
}

void bindAcceptedTwoFluidStageNumerics(
    FE::systems::FESystem& system,
    const FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration& declaration,
    const FE::systems::AcceptedTwoFluidStageDiagnosticState& state);

TEST(IncompressibleTwoFluidModule,
     AcceptedStageAllowsTractionDifferenceWithinPhaseOperandRoundoff) {
  TwoFluidRegistrationFixture fixture;
  auto module = fixture.makeModule({});
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  constexpr FE::Real hydrostatic_traction = FE::Real{36297.0};
  state.diagnostics.negative_traction_integral[0] =
      hydrostatic_traction;
  state.diagnostics.positive_traction_integral[0] = std::nextafter(
      hydrostatic_traction,
      -std::numeric_limits<FE::Real>::infinity());
  state.diagnostics.traction_jump_integral[0] = FE::Real{0.0};
  const std::array states{state};

  EXPECT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageRejectsMalformedPhasePressureAndHydrostaticMoments) {
  const auto expect_rejected = [](const auto& mutate) {
    TwoFluidRegistrationFixture fixture;
    auto module = fixture.makeModule({});
    module.registerOn(fixture.system);
    fixture.system.setup({});

    const auto declarations =
        fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
    ASSERT_EQ(declarations.size(), 1u);
    auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
    mutate(state.diagnostics.negative_phase);
    const std::array states{state};

    EXPECT_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
                     makeTwoFluidStageMetadata(), states),
                 FE::InvalidArgumentException);
    EXPECT_TRUE(
        fixture.system.pendingTwoFluidAcceptedStageDiagnostics().empty());
  };

  expect_rejected([](auto& phase) {
    phase.pressure_integral =
        std::numeric_limits<FE::Real>::quiet_NaN();
  });
  expect_rejected([](auto& phase) {
    phase.pressure_integral = FE::Real{0.25};
    phase.pressure_squared_integral = FE::Real{0.25};
    phase.mean_pressure = FE::Real{0.0};
  });
  expect_rejected([](auto& phase) {
    phase.pressure_gradient_integral[0] = FE::Real{1.0};
    phase.hydrostatic_residual_integral[0] = FE::Real{0.0};
    phase.hydrostatic_residual_squared = FE::Real{4.0};
  });
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageHistoryDiscardsRejectsAndReplaysExactly) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.prescribed_pressure_jump = FE::Real{3.0};
  options.prescribed_viscous_traction_jump =
      std::array<FE::Real, 3>{{0.0, 2.0, 0.0}};
  auto module = fixture.makeModule(std::move(options));
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};

  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
  EXPECT_EQ(fixture.system.pendingTwoFluidAcceptedStageDiagnostics().size(),
            1u);
  EXPECT_TRUE(fixture.system.twoFluidAcceptedStageDiagnosticHistory().empty());
  fixture.system.discardPendingTwoFluidAcceptedStageDiagnostics();
  EXPECT_TRUE(
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics().empty());

  auto wrong_source = state;
  wrong_source.geometry_revision.source_id = "field:999";
  const std::array wrong_source_states{wrong_source};
  EXPECT_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
                   makeTwoFluidStageMetadata(), wrong_source_states),
               FE::InvalidArgumentException);
  EXPECT_TRUE(
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics().empty());

  auto stale_local_mesh = state;
  stale_local_mesh.local_mesh_revision.mesh_geometry_revision = 1u;
  const std::array stale_local_mesh_states{stale_local_mesh};
  EXPECT_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
                   makeTwoFluidStageMetadata(), stale_local_mesh_states),
               FE::InvalidArgumentException);
  EXPECT_TRUE(
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics().empty());

  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
  ASSERT_NO_THROW(bindAcceptedTwoFluidStageNumerics(
      fixture.system, declarations.front(), state));
  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));
  ASSERT_EQ(fixture.system.twoFluidAcceptedStageDiagnosticHistory().size(),
            1u);
  EXPECT_DOUBLE_EQ(
      fixture.system.twoFluidAcceptedStageDiagnosticHistory()
          .front()
          .state.diagnostics.mean_pressure_jump,
      FE::Real{3.0});
  EXPECT_TRUE(fixture.system.twoFluidAcceptedStageDiagnosticHistory()
                  .front()
                  .state.diagnostics.prescribed_pressure_jump_error_squared
                  .has_value());
  EXPECT_TRUE(
      fixture.system.twoFluidAcceptedStageDiagnosticHistory()
          .front()
          .state.diagnostics
          .prescribed_viscous_traction_jump_error_squared.has_value());

  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
  ASSERT_NO_THROW(bindAcceptedTwoFluidStageNumerics(
      fixture.system, declarations.front(), state));
  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));
  EXPECT_EQ(fixture.system.twoFluidAcceptedStageDiagnosticHistory().size(),
            1u);

  auto changed = state;
  changed.diagnostics.surface_energy_work = FE::Real{1.0};
  const std::array changed_states{changed};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), changed_states));
  ASSERT_NO_THROW(bindAcceptedTwoFluidStageNumerics(
      fixture.system, declarations.front(), changed));
  EXPECT_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
                   1u, FE::Real{0.1}, FE::Real{0.1}),
               FE::InvalidArgumentException);
  fixture.system.discardPendingTwoFluidAcceptedStageDiagnostics();
  EXPECT_EQ(fixture.system.twoFluidAcceptedStageDiagnosticHistory().size(),
            1u);
}

TEST(IncompressibleTwoFluidModule,
     RequiredMomentumReconciliationBindsBeforeAcceptedHistoryCommit) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.require_conservative_phase_momentum_reconciliation = true;
  auto module = fixture.makeModule(std::move(options));
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  EXPECT_TRUE(declarations.front()
                  .require_conservative_phase_momentum_reconciliation);
  const auto raw = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array raw_states{raw};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), raw_states));
  ASSERT_NO_THROW(bindAcceptedTwoFluidStageNumerics(
      fixture.system, declarations.front(), raw));
  EXPECT_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
                   1u, FE::Real{0.1}, FE::Real{0.1}),
               FE::InvalidArgumentException);

  auto corrected = raw;
  corrected.geometry_revision.source_value_revision = 8u;
  corrected.geometry_revision.snapshot_revision_key = 92u;
  corrected.diagnostics.snapshot_revision_key = 92u;
  const auto valid = FE::interfaces::
      buildIncompressibleTwoFluidMomentumReconciliation(
          declarations.front().interface_marker,
          raw.geometry_revision,
          raw.diagnostics,
          /*raw_algebraic_revision=*/41u,
          corrected.geometry_revision,
          corrected.diagnostics,
          /*corrected_algebraic_revision=*/43u,
          FE::Real{1.0e-10});

  auto wrong_revision = valid;
  wrong_revision.raw_algebraic_revision = 99u;
  const std::array wrong_revision_records{wrong_revision};
  EXPECT_THROW(
      fixture.system.bindPendingTwoFluidMomentumReconciliations(
          wrong_revision_records),
      FE::InvalidArgumentException);
  ASSERT_EQ(
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics().size(), 1u);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .momentum_reconciliation.has_value());

  auto changed = corrected;
  changed.diagnostics.negative_phase.momentum[0] = FE::Real{1.0};
  changed.diagnostics.negative_phase.kinetic_energy = FE::Real{1.0};
  const auto failed = FE::interfaces::
      buildIncompressibleTwoFluidMomentumReconciliation(
          declarations.front().interface_marker,
          raw.geometry_revision,
          raw.diagnostics,
          41u,
          changed.geometry_revision,
          changed.diagnostics,
          43u,
          FE::Real{1.0e-10});
  ASSERT_FALSE(failed.satisfied);
  const std::array failed_records{failed};
  EXPECT_THROW(
      fixture.system.bindPendingTwoFluidMomentumReconciliations(
          failed_records),
      FE::InvalidArgumentException);

  const std::array valid_records{valid};
  ASSERT_NO_THROW(
      fixture.system.bindPendingTwoFluidMomentumReconciliations(
          valid_records));
  ASSERT_TRUE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                  .front()
                  .momentum_reconciliation.has_value());
  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));
  const auto history =
      fixture.system.twoFluidAcceptedStageDiagnosticHistory();
  ASSERT_EQ(history.size(), 1u);
  ASSERT_TRUE(history.front().momentum_reconciliation.has_value());
  const auto& accepted = *history.front().momentum_reconciliation;
  EXPECT_EQ(accepted.interface_marker, valid.interface_marker);
  EXPECT_EQ(accepted.raw_algebraic_revision,
            valid.raw_algebraic_revision);
  EXPECT_EQ(accepted.corrected_algebraic_revision,
            valid.corrected_algebraic_revision);
  EXPECT_EQ(accepted.relative_tolerance, valid.relative_tolerance);
  EXPECT_EQ(accepted.negative_phase, valid.negative_phase);
  EXPECT_EQ(accepted.positive_phase, valid.positive_phase);
}

std::array<FE::constraints::SmallCutAggregationRefreshReport, 4>
makeTwoFluidAcceptedStageAggregationReports(
    const FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration& declaration,
    const FE::systems::AcceptedTwoFluidStageDiagnosticState& state) {
  const auto make_report = [&](FE::FieldId field,
                               FE::geometry::CutIntegrationSide side,
                               std::size_t candidate_vertices) {
    FE::constraints::SmallCutAggregationRefreshReport report;
    report.field = field;
    report.active_side = side;
    report.interface_marker = declaration.interface_marker;
    report.geometry_identity.kind = FE::constraints::
        SmallCutAggregationGeometryIdentityKind::
            AuthoritativeFreeSurfaceSnapshot;
    report.geometry_identity.available = true;
    report.geometry_identity.communicator_fingerprint_consensus_validated =
        true;
    report.geometry_identity.source_id = state.geometry_revision.source_id;
    report.geometry_identity.domain_id = state.geometry_revision.domain_id;
    report.geometry_identity.interface_marker =
        state.geometry_revision.interface_marker;
    report.geometry_identity.isovalue = state.geometry_revision.isovalue;
    report.geometry_identity.source_layout_revision =
        state.geometry_revision.source_layout_revision;
    report.geometry_identity.source_value_revision =
        state.geometry_revision.source_value_revision;
    report.geometry_identity.quadrature_policy_key =
        state.geometry_revision.quadrature_policy_key;
    report.geometry_identity.snapshot_revision_key =
        state.geometry_revision.snapshot_revision_key;
    report.geometry_identity.distributed_mesh_geometry_revision =
        state.geometry_revision.mesh_geometry_revision;
    report.geometry_identity.distributed_mesh_topology_revision =
        state.geometry_revision.mesh_topology_revision;
    report.geometry_identity.distributed_ownership_revision =
        state.geometry_revision.ownership_revision;
    report.geometry_identity.distributed_numbering_revision =
        state.geometry_revision.numbering_revision;
    report.geometry_identity.canonical_fingerprint = 700u;
    report.canonical_feature_class_fingerprint = 800u + field;
    report.canonical_slave_set_fingerprint = 900u + field;
    report.maximum_root_path_length = 4u;
    report.maximum_observed_root_path = 2u;
    report.maximum_reference_extrapolation_distance = FE::Real{1.5};
    report.maximum_observed_reference_extrapolation = FE::Real{0.25};
    report.maximum_absolute_coefficient = FE::Real{8.0};
    report.maximum_observed_absolute_coefficient = FE::Real{1.25};
    report.maximum_row_l1_norm = FE::Real{12.0};
    report.maximum_observed_row_l1_norm = FE::Real{2.0};
    report.canonical_candidate_vertices = candidate_vertices;
    report.canonical_rooted_candidate_vertices = candidate_vertices;
    report.canonical_owned_aggregate_dofs = candidate_vertices + 1u;
    report.canonical_active_feature_count = 1u;
    report.canonical_rooted_active_feature_count = 1u;
    return report;
  };

  return {
      make_report(declaration.negative_velocity_field,
                  FE::geometry::CutIntegrationSide::Negative,
                  2u),
      make_report(declaration.negative_pressure_field,
                  FE::geometry::CutIntegrationSide::Negative,
                  3u),
      make_report(declaration.positive_velocity_field,
                  FE::geometry::CutIntegrationSide::Positive,
                  5u),
      make_report(declaration.positive_pressure_field,
                  FE::geometry::CutIntegrationSide::Positive,
                  7u),
  };
}

FE::systems::TwoFluidAcceptedStageSolveTelemetry
makeTwoFluidAcceptedStageSolveTelemetry() {
  return FE::systems::TwoFluidAcceptedStageSolveTelemetry{
      .nonlinear =
          FE::systems::TwoFluidAcceptedStageNonlinearTelemetry{
              .converged = true,
              .iterations = 3,
              .outer_iterations = 1,
              .inner_iterations_total = 3,
              .initial_residual_norm = FE::Real{2.0},
              .final_residual_norm = FE::Real{1.0e-9},
              .reason = "converged",
          },
      .linear =
          FE::systems::TwoFluidAcceptedStageLinearTelemetry{
              .converged = true,
              .numerical_breakdown = false,
              .iterations = 17,
              .initial_residual_norm = FE::Real{1.0},
              .final_residual_norm = FE::Real{2.0e-10},
              .relative_residual = FE::Real{2.0e-10},
              .blockschur_outer_iterations = 6,
              .blockschur_momentum_solve_calls = 6,
              .blockschur_momentum_iterations = 12,
              .blockschur_schur_solve_calls = 6,
              .blockschur_schur_iterations = 9,
              .reason = "relative residual target reached",
          },
  };
}

void bindAcceptedTwoFluidStageNumerics(
    FE::systems::FESystem& system,
    const FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration& declaration,
    const FE::systems::AcceptedTwoFluidStageDiagnosticState& state) {
  const auto reports =
      makeTwoFluidAcceptedStageAggregationReports(declaration, state);
  system.bindPendingTwoFluidAcceptedStageNumerics(
      makeTwoFluidAcceptedStageSolveTelemetry(), reports);
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageNumericsRequireCanonicalPhaseCoverageBeforeCommit) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.pressure_gradient_penalty = FE::Real{2.5};
  options.use_cut_metadata_scale = true;
  options.cut_metadata_scale_cap = FE::Real{16.0};
  auto module = fixture.makeModule(std::move(options));
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  EXPECT_TRUE(declarations.front().require_accepted_stage_numerics);
  EXPECT_EQ(declarations.front().pressure_stabilization_coefficient,
            FE::Real{2.5});
  EXPECT_TRUE(
      declarations.front().pressure_stabilization_use_cut_metadata_scale);
  ASSERT_TRUE(declarations.front()
                  .pressure_stabilization_cut_metadata_scale_cap
                  .has_value());
  EXPECT_EQ(*declarations.front()
                 .pressure_stabilization_cut_metadata_scale_cap,
            FE::Real{16.0});
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));

  EXPECT_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
                   1u, FE::Real{0.1}, FE::Real{0.1}),
               FE::InvalidArgumentException);
  ASSERT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  const auto solve = makeTwoFluidAcceptedStageSolveTelemetry();
  auto reports = makeTwoFluidAcceptedStageAggregationReports(
      declarations.front(), state);
  auto mismatched = reports;
  mismatched.back().geometry_identity.snapshot_revision_key += 1u;
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, mismatched),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  ASSERT_NO_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
      solve, reports));
  const auto pending =
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics();
  ASSERT_EQ(pending.size(), 1u);
  ASSERT_TRUE(pending.front().numerics.has_value());
  const auto& numerics = *pending.front().numerics;
  EXPECT_EQ(numerics.solve, solve);
  EXPECT_EQ(numerics.negative_phase.side,
            FE::geometry::CutIntegrationSide::Negative);
  EXPECT_EQ(numerics.positive_phase.side,
            FE::geometry::CutIntegrationSide::Positive);
  EXPECT_EQ(numerics.negative_phase.velocity_aggregation
                .canonical_candidate_vertices,
            2u);
  EXPECT_EQ(numerics.positive_phase.pressure_aggregation
                .canonical_candidate_vertices,
            7u);
  EXPECT_EQ(numerics.negative_phase.pressure_stabilization.coefficient,
            FE::Real{2.5});
  EXPECT_TRUE(numerics.negative_phase.pressure_stabilization
                  .use_cut_metadata_scale);
  ASSERT_TRUE(numerics.negative_phase.pressure_stabilization
                  .cut_metadata_scale_cap.has_value());
  EXPECT_EQ(*numerics.negative_phase.pressure_stabilization
                 .cut_metadata_scale_cap,
            FE::Real{16.0});
  EXPECT_EQ(numerics.negative_phase.linear_iteration_scope,
            FE::systems::TwoFluidAcceptedStageLinearIterationScope::
                SharedCoupledSolve);
  EXPECT_FALSE(numerics.negative_phase.linear_iterations.has_value());
  EXPECT_EQ(numerics.negative_phase.linear_reason,
            "not_individually_resolved_by_shared_coupled_backend");
  EXPECT_EQ(numerics.negative_phase.nonlinear_iteration_scope,
            FE::systems::TwoFluidAcceptedStageNonlinearIterationScope::
                SharedCoupledSolve);
  EXPECT_FALSE(numerics.negative_phase.nonlinear_iterations.has_value());
  EXPECT_EQ(numerics.negative_phase.nonlinear_reason,
            "not_individually_resolved_by_shared_coupled_backend");
  EXPECT_FALSE(numerics.negative_phase.pressure_stabilization
                   .accepted_stage_work.has_value());
  EXPECT_EQ(numerics.negative_phase.pressure_stabilization.work_reason,
            "not_separately_resolved_by_coupled_operator");

  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));
  const auto history =
      fixture.system.twoFluidAcceptedStageDiagnosticHistory();
  ASSERT_EQ(history.size(), 1u);
  ASSERT_TRUE(history.front().numerics.has_value());
  EXPECT_EQ(history.front().numerics->solve.linear.iterations, 17);
  EXPECT_EQ(history.front().numerics->positive_phase.pressure_aggregation
                .canonical_owned_aggregate_dofs,
            8u);
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageNumericsRejectInvalidOrDuplicateEvidenceTransactionally) {
  TwoFluidRegistrationFixture fixture;
  auto module = fixture.makeModule();
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));

  const auto solve = makeTwoFluidAcceptedStageSolveTelemetry();
  const auto reports = makeTwoFluidAcceptedStageAggregationReports(
      declarations.front(), state);
  const std::array missing{reports[0], reports[1], reports[2]};
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, missing),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  auto invalid_solve = solve;
  invalid_solve.linear.final_residual_norm =
      std::numeric_limits<FE::Real>::infinity();
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   invalid_solve, reports),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  auto unattempted_solve = solve;
  unattempted_solve.linear.attempted = false;
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   unattempted_solve, reports),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  auto wrong_side = reports;
  wrong_side.front().active_side =
      FE::geometry::CutIntegrationSide::Positive;
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, wrong_side),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  auto violated_guard = reports;
  violated_guard.front().maximum_observed_root_path =
      violated_guard.front().maximum_root_path_length + 1u;
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, violated_guard),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  auto stale_transition = reports;
  auto& transition =
      stale_transition.front().canonical_topology_transition.emplace();
  transition.geometry_identity_before =
      stale_transition.front().geometry_identity;
  transition.geometry_identity_after =
      stale_transition.front().geometry_identity;
  transition.geometry_identity_after.snapshot_revision_key += 1u;
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, stale_transition),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  std::vector<FE::constraints::SmallCutAggregationRefreshReport> duplicate(
      reports.begin(), reports.end());
  duplicate.push_back(reports.front());
  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   solve, duplicate),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());

  std::vector<FE::constraints::SmallCutAggregationRefreshReport> complete(
      reports.begin(), reports.end());
  auto unrelated = reports.front();
  unrelated.interface_marker += 100;
  complete.push_back(std::move(unrelated));
  ASSERT_NO_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
      solve, complete));
  EXPECT_TRUE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                  .front()
                  .numerics.has_value());
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageNumericsRejectSameInterfaceUnexpectedFieldTransactionally) {
  TwoFluidRegistrationFixture fixture;
  auto module = fixture.makeModule();
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));

  const auto baseline_reports = makeTwoFluidAcceptedStageAggregationReports(
      declarations.front(), state);
  std::vector<FE::constraints::SmallCutAggregationRefreshReport> reports(
      baseline_reports.begin(), baseline_reports.end());
  auto unexpected = reports.front();
  unexpected.field = declarations.front().level_set_field;
  reports.push_back(std::move(unexpected));

  EXPECT_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
                   makeTwoFluidAcceptedStageSolveTelemetry(), reports),
               FE::InvalidArgumentException);
  EXPECT_FALSE(fixture.system.pendingTwoFluidAcceptedStageDiagnostics()
                   .front()
                   .numerics.has_value());
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageNumericsAdmitExactEntryConvergenceWithoutLinearSolve) {
  TwoFluidRegistrationFixture fixture;
  auto module = fixture.makeModule();
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));

  const auto reports = makeTwoFluidAcceptedStageAggregationReports(
      declarations.front(), state);
  const FE::systems::TwoFluidAcceptedStageSolveTelemetry solve{
      .nonlinear =
          FE::systems::TwoFluidAcceptedStageNonlinearTelemetry{
              .converged = true,
              .outer_iterations = 1,
              .reason = "entry state satisfies nonlinear tolerances",
          },
      .linear =
          FE::systems::TwoFluidAcceptedStageLinearTelemetry{
              .attempted = false,
              .converged = true,
              .reason =
                  "not required: entry state satisfies nonlinear tolerances",
          },
  };

  ASSERT_NO_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
      solve, reports));
  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));

  const auto history =
      fixture.system.twoFluidAcceptedStageDiagnosticHistory();
  ASSERT_EQ(history.size(), 1u);
  ASSERT_TRUE(history.front().numerics.has_value());
  EXPECT_FALSE(history.front().numerics->solve.linear.attempted);
  EXPECT_EQ(history.front().numerics->solve.linear.iterations, 0);
}

TEST(IncompressibleTwoFluidModule,
     AcceptedStageNumericsAdmitCanonicalGeneratedPublicationIdentity) {
  TwoFluidRegistrationFixture fixture;
  auto module = fixture.makeModule();
  module.registerOn(fixture.system);
  fixture.system.setup({});

  const auto declarations =
      fixture.system.twoFluidAcceptedStageDiagnosticDeclarations();
  ASSERT_EQ(declarations.size(), 1u);
  const auto state = makeAcceptedTwoFluidDiagnosticState(declarations.front());
  const std::array states{state};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));

  auto reports = makeTwoFluidAcceptedStageAggregationReports(
      declarations.front(), state);
  std::uint64_t local_ordinal = 1u;
  for (auto& report : reports) {
    report.geometry_identity.kind = FE::constraints::
        SmallCutAggregationGeometryIdentityKind::
            GeneratedPublicationSource;
    report.geometry_identity.snapshot_revision_key = 0u;
    report.geometry_identity.distributed_mesh_geometry_revision = 0u;
    report.geometry_identity.distributed_mesh_topology_revision = 0u;
    report.geometry_identity.distributed_ownership_revision = 0u;
    report.geometry_identity.distributed_numbering_revision = 0u;
    report.local_lineage.successful_publication_ordinal = local_ordinal++;
    report.local_lineage.mesh_geometry_revision = 100u + local_ordinal;
  }

  ASSERT_NO_THROW(fixture.system.bindPendingTwoFluidAcceptedStageNumerics(
      makeTwoFluidAcceptedStageSolveTelemetry(), reports));
  const auto pending =
      fixture.system.pendingTwoFluidAcceptedStageDiagnostics();
  ASSERT_TRUE(pending.front().numerics.has_value());
  EXPECT_EQ(pending.front()
                .numerics->negative_phase.velocity_aggregation
                .geometry_identity.kind,
            FE::constraints::SmallCutAggregationGeometryIdentityKind::
                GeneratedPublicationSource);
  EXPECT_EQ(pending.front()
                .numerics->positive_phase.pressure_aggregation
                .geometry_identity.snapshot_revision_key,
            0u);
}

TEST(IncompressibleTwoFluidModule,
     MissingLevelSetFailsBeforePhaseFieldsOrOperatorAreAdded) {
  auto mesh =
      std::make_shared<FE::forms::test::SingleTriangleMeshAccess>();
  auto scalar_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
  auto velocity_space =
      std::make_shared<FE::spaces::ProductSpace>(scalar_space, 2);
  FE::systems::FESystem system(mesh);
  ns::IncompressibleTwoFluidModule module(
      velocity_space, scalar_space, velocity_space, scalar_space);

  EXPECT_THROW(module.registerOn(system), std::invalid_argument);
  EXPECT_EQ(system.findFieldByName("u_negative"), FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_negative"), FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"), FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_positive"), FE::INVALID_FIELD_ID);
  EXPECT_FALSE(system.hasOperator("equations"));
  EXPECT_FALSE(module.effectiveConfigurationArtifact().has_value());
}

TEST(IncompressibleTwoFluidModule,
     RejectsSpacesOutsideAffineSimplexEnvelopeBeforeMutation) {
  auto mesh =
      std::make_shared<FE::forms::test::SingleTriangleMeshAccess>();
  auto scalar_space =
      std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, 1);
  auto velocity_space =
      std::make_shared<FE::spaces::ProductSpace>(scalar_space, 2);
  FE::systems::FESystem system(mesh);
  (void)system.addField(FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
      .source_kind = FE::systems::FieldSourceKind::PrescribedData,
  });
  ns::IncompressibleTwoFluidModule module(
      velocity_space, scalar_space, velocity_space, scalar_space);

  EXPECT_THROW(module.registerOn(system), std::invalid_argument);
  EXPECT_EQ(system.findFieldByName("u_negative"), FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"), FE::INVALID_FIELD_ID);
  EXPECT_FALSE(system.hasOperator("equations"));
}

TEST(IncompressibleTwoFluidModule,
     InternalPhaseRoleRejectsExteriorTractionBeforeMutation) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleNavierStokesVMSOptions options;
  options.velocity_field_name = "phase_velocity";
  options.pressure_field_name = "phase_pressure";
  ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary interface;
  interface.role =
      ns::FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
  interface.implementation =
      ns::FreeSurfaceImplementation::UnfittedLevelSet;
  interface.interface_marker = 481;
  interface.level_set_field_name = "level_set";
  interface.generated_interface_domain_id = "invalid_internal_phase";
  interface.active_domain =
      ns::FreeSurfaceActiveDomain::LevelSetNegative;
  interface.active_domain_method =
      ns::FreeSurfaceActiveDomainMethod::CutVolume;
  interface.external_pressure = FE::Real{1.0};
  interface.surface_tension = FE::Real{0.0};
  interface.use_level_set_curvature = false;
  interface.normal_kinematic_policy =
      ns::FreeSurfaceNormalKinematicPolicy::None;
  interface.tangential_mesh_policy =
      ns::FreeSurfaceTangentialMeshPolicy::Free;
  options.free_surface.push_back(std::move(interface));
  ns::IncompressibleNavierStokesVMSModule module(
      fixture.velocity_space, fixture.scalar_space, std::move(options));

  EXPECT_THROW(module.registerOn(fixture.system), std::invalid_argument);
  EXPECT_EQ(fixture.system.findFieldByName("phase_velocity"),
            FE::INVALID_FIELD_ID);
  EXPECT_EQ(fixture.system.findFieldByName("phase_pressure"),
            FE::INVALID_FIELD_ID);
  EXPECT_FALSE(fixture.system.hasOperator("equations"));
}

TEST(IncompressibleTwoFluidModule,
     InternalPhaseRoleSerializesVolumeOnlyOwnership) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleNavierStokesVMSOptions options;
  options.velocity_field_name = "phase_velocity";
  options.pressure_field_name = "phase_pressure";
  options.enable_convection = false;
  ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary interface;
  interface.role =
      ns::FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
  interface.implementation =
      ns::FreeSurfaceImplementation::UnfittedLevelSet;
  interface.interface_marker = 482;
  interface.level_set_field_name = "level_set";
  interface.generated_interface_domain_id = "valid_internal_phase";
  interface.active_domain =
      ns::FreeSurfaceActiveDomain::LevelSetNegative;
  interface.active_domain_method =
      ns::FreeSurfaceActiveDomainMethod::CutVolume;
  interface.external_pressure = FE::Real{0.0};
  interface.surface_tension = FE::Real{0.0};
  interface.use_level_set_curvature = false;
  interface.normal_kinematic_policy =
      ns::FreeSurfaceNormalKinematicPolicy::None;
  interface.tangential_mesh_policy =
      ns::FreeSurfaceTangentialMeshPolicy::Free;
  interface.cut_cell_stabilization.enabled = true;
  interface.small_cut_aggregation = true;
  options.free_surface.push_back(std::move(interface));
  ns::IncompressibleNavierStokesVMSModule module(
      fixture.velocity_space, fixture.scalar_space, std::move(options));

  ASSERT_NO_THROW(module.registerOn(fixture.system));
  const auto artifact = module.effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(
      artifact->json.find(
          "\"capability_label\":\"internal_material_interface_phase_volume\""),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"role\":\"InternalMaterialInterfaceVolume\""),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"surface_tension_form_effective\":\"NotApplicableInternalVolume\""),
      std::string::npos);
  EXPECT_FALSE(
      fixture.system.freeSurfaceExternalBoundaryEnergyDeclaration().has_value());
  EXPECT_TRUE(fixture.system.freeSurfaceResidualWorkDeclarations().empty());
  EXPECT_TRUE(
      fixture.system.freeSurfaceDiscreteFunctionalDeclarations().empty());
}

} // namespace
