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
#include "FE/Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"

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
  request.quadrature_order = 0;
  request.interface_quadrature_order = 0;
  request.volume_quadrature_order = 0;
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
  policy.minimum_achieved_quadrature_order = 0;
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
  EXPECT_EQ(state.positive_phase.quadrature_point_count, 1u);
  EXPECT_NEAR(state.positive_phase.volume, FE::Real{0.125}, tolerance);
  EXPECT_NEAR(state.positive_phase.mass, FE::Real{0.25}, tolerance);
  EXPECT_NEAR(state.positive_phase.momentum[0], FE::Real{0.0}, tolerance);
  EXPECT_NEAR(state.positive_phase.momentum[1], FE::Real{0.25}, tolerance);
  EXPECT_NEAR(state.positive_phase.kinetic_energy,
              FE::Real{0.125},
              tolerance);
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
  EXPECT_FALSE(diagnostic_declarations.front()
                   .parameters.prescribed_pressure_jump.has_value());

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
  EXPECT_NE(artifact->json.find("\"shared_gauge_count\":1"),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"compressible_gas\""),
            std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"prescribed_pressure_jump_applicable\":false"),
      std::string::npos);

  TwoFluidRegistrationFixture repeated_fixture;
  ns::IncompressibleTwoFluidOptions repeated_options;
  repeated_options.surface_tension = FE::Real{0.072};
  auto repeated_module =
      repeated_fixture.makeModule(std::move(repeated_options));
  repeated_module.registerOn(repeated_fixture.system);
  const auto repeated_artifact =
      repeated_module.effectiveConfigurationArtifact();
  ASSERT_TRUE(repeated_artifact.has_value());
  EXPECT_EQ(repeated_artifact->component, artifact->component);
  EXPECT_EQ(repeated_artifact->json, artifact->json);
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
  accumulator.negative_normal_flux = FE::Real{0.25};
  accumulator.positive_normal_flux = FE::Real{0.25};
  accumulator.pressure_jump_integral = FE::Real{1.5};
  accumulator.pressure_jump_squared = FE::Real{4.5};
  accumulator.negative_traction_integral = {{-1.5, 0.0, 0.0}};
  accumulator.positive_traction_integral = {{0.0, 0.0, 0.0}};
  accumulator.traction_jump_integral = {{-1.5, 0.0, 0.0}};
  accumulator.traction_jump_normal_integral = FE::Real{-1.5};
  accumulator.traction_jump_squared = FE::Real{4.5};
  accumulator.negative_phase.owned_quadrature_point_count = 3u;
  accumulator.negative_phase.volume = FE::Real{0.25};
  accumulator.positive_phase.owned_quadrature_point_count = 3u;
  accumulator.positive_phase.volume = FE::Real{0.25};
  if (parameters.prescribed_pressure_jump.has_value()) {
    accumulator.prescribed_pressure_jump_error_squared = FE::Real{0.0};
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

TEST(IncompressibleTwoFluidModule,
     AcceptedStageHistoryDiscardsRejectsAndReplaysExactly) {
  TwoFluidRegistrationFixture fixture;
  ns::IncompressibleTwoFluidOptions options;
  options.prescribed_pressure_jump = FE::Real{3.0};
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

  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
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

  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), states));
  ASSERT_NO_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
      1u, FE::Real{0.1}, FE::Real{0.1}));
  EXPECT_EQ(fixture.system.twoFluidAcceptedStageDiagnosticHistory().size(),
            1u);

  auto changed = state;
  changed.diagnostics.surface_energy_work = FE::Real{1.0};
  const std::array changed_states{changed};
  ASSERT_NO_THROW(fixture.system.stageTwoFluidAcceptedStageDiagnostics(
      makeTwoFluidStageMetadata(), changed_states));
  EXPECT_THROW(fixture.system.commitPendingTwoFluidAcceptedStageDiagnostics(
                   1u, FE::Real{0.1}, FE::Real{0.1}),
               FE::InvalidArgumentException);
  fixture.system.discardPendingTwoFluidAcceptedStageDiagnostics();
  EXPECT_EQ(fixture.system.twoFluidAcceptedStageDiagnosticHistory().size(),
            1u);
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
