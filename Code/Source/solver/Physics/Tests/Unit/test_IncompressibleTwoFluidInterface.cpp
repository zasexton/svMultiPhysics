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
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
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
  std::shared_ptr<FE::forms::test::SingleTriangleMeshAccess> mesh{
      std::make_shared<FE::forms::test::SingleTriangleMeshAccess>()};
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
