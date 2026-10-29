/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Analysis/GeneratedBoundaryAggregateTraceCertificate.h"
#include "Interfaces/ContactLineGeometryCertificate.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/MeshBoundaryKinematics.h"
#include "Systems/MeshNormalConstraintStageMeasurements.h"
#include "Systems/FormsInstaller.h"
#include "Forms/Vocabulary.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace systems = svmp::FE::systems;
using svmp::FE::ElementType;
using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;

std::string invalidArgumentMessage(const std::function<void()>& call)
{
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}

struct MovingBoundaryDeclarations {
    std::shared_ptr<ProductSpace> space = std::make_shared<ProductSpace>(
        std::make_shared<H1Space>(ElementType::Tetra4, 1), 3);
    systems::FESystem system{
        std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>()};
    svmp::FE::FieldId displacement = system.addMeshDisplacementUnknown("d", space);
    svmp::FE::FieldId velocity = system.addField({.name = "u", .space = space, .components = 3});
    svmp::FE::forms::FormExpr target = svmp::FE::forms::normalTrace(
        svmp::FE::forms::StateField(velocity, *space, "u"), svmp::FE::forms::currentNormal());
    std::vector<systems::MeshNormalKinematicDeclaration> declarations;

    MovingBoundaryDeclarations()
    {
        using namespace svmp::FE::forms;
        system.addOperator("moving");
        const auto d = StateField(displacement, *space, "d");
        const auto u = StateField(velocity, *space, "u");
        const auto w = TestField(displacement, *space, "w");
        systems::FormInstallOptions install;
        install.extra_trial_fields = {velocity};
        (void)systems::installFormulation(system, "moving", {displacement},
            inner(dt(d) - u, w).ds(7) + inner(dt(d) - u, w).ds(3), install);
        for (const int marker : {7, 3}) {
            const auto suffix = std::to_string(marker);
            const auto policy = marker == 7 ? systems::MeshTangentialBoundaryPolicy::Prescribed
                                            : systems::MeshTangentialBoundaryPolicy::Free;
            system.declareMeshTangentialBoundaryPolicy({
                .mesh_displacement_field = displacement, .boundary_marker = marker,
                .policy = policy, .owner_component = "moving boundary"});
            systems::declareMeshBoundaryRowDescriptors(system, "moving", {
                .boundary_marker = marker, .mesh_displacement_field = displacement,
                .velocity_field = velocity, .normal_relation = true,
                .tangential_prescribed = marker == 7, .tangential_policy = policy,
                .mesh_normal_source = "mesh " + suffix,
                .fluid_normal_source = "fluid " + suffix,
                .tangential_source = "tangent " + suffix}, true);
            declarations.push_back({
                .constraint = {
                    .mesh_displacement_field = displacement, .boundary_marker = marker,
                    .quantity = systems::MeshNormalBoundaryQuantity::MeshVelocityTrace,
                    .target_kind = systems::MeshNormalBoundaryTargetKind::FluidNormalVelocity,
                    .target_expression = target, .related_velocity_field = velocity,
                    .owner_component = "moving boundary",
                    .requires_mesh_flux_consistency = marker == 7,
                    .mesh_flux_consistency_nitsche_gamma = marker == 7 ? 8.0 : 0.0},
                .binding = {.mesh_row_source = "mesh " + suffix,
                            .fluid_row_source = "fluid " + suffix,
                            .prescribed_tangential_mesh_velocity = marker == 7
                                ? std::optional<std::array<svmp::FE::Real, 3>>{{1, 2, 3}}
                                : std::nullopt}});
        }
    }
};

} // namespace

TEST(MeshBoundaryKinematics, DisplacementRecordThrowsTheGivenMessages)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    systems::FESystem sys(mesh);
    const systems::ALEBinding uncoupled{};
    EXPECT_EQ(invalidArgumentMessage([&] {
                  (void)systems::requireMeshDisplacementRecord(
                      sys, uncoupled, "Owner: no unknown", "Owner: no space");
              }),
              "Owner: no unknown");

    auto scalar_space = std::make_shared<H1Space>(ElementType::Tetra4, /*order=*/1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/3);
    systems::ALEBinding coupled{};
    coupled.mesh_displacement_field =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    const auto& record =
        systems::requireMeshDisplacementRecord(
            sys, coupled, "Owner: no unknown", "Owner: no space");
    EXPECT_EQ(record.name, "mesh_displacement");
}

TEST(MeshBoundaryKinematics, NormalKinematicsResolveEachBindingBeforeDeclaring)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    systems::FESystem sys(mesh);

    int calls = 0;
    systems::declareMeshNormalKinematics(
        sys, "op", {}, [&](int) {
            ++calls;
            return systems::MeshNormalKinematicBinding{};
        });
    EXPECT_EQ(calls, 0);

    std::vector<systems::MeshNormalBoundaryConstraintDeclaration> declarations(1);
    declarations.front().boundary_marker = 7;
    std::vector<int> asked;
    EXPECT_THROW(
        systems::declareMeshNormalKinematics(
            sys,
            "op",
            std::move(declarations),
            [&](int marker) -> systems::MeshNormalKinematicBinding {
                asked.push_back(marker);
                throw std::logic_error("lost its owner");
            }),
        std::logic_error);
    EXPECT_EQ(asked, std::vector<int>{7});
    EXPECT_TRUE(sys.meshNormalBoundaryConstraints().empty());
}

TEST(MeshBoundaryKinematics, PreparedAndCallbackDeclarationsKeepRelationsConsumersAndMeasurements)
{
    using namespace svmp::FE;
    for (const bool prepared : {false, true}) {
        MovingBoundaryDeclarations fixture;
        if (prepared) {
            systems::declareMeshNormalKinematics(fixture.system, "moving", fixture.declarations);
        } else {
            std::vector<systems::MeshNormalBoundaryConstraintDeclaration> constraints;
            for (const auto& declaration : fixture.declarations) constraints.push_back(declaration.constraint);
            std::vector<int> requested;
            systems::declareMeshNormalKinematics(fixture.system, "moving", std::move(constraints),
                [&](int marker) {
                    // Each earlier item is completely installed before the next callback.
                    EXPECT_EQ(fixture.system.meshNormalBoundaryConstraints().size(), requested.size());
                    EXPECT_EQ(systems::meshNormalConstraintStageMeasurements(fixture.system)
                        .fittedALENormalOperatorStageMeasurementDeclarations().size(), requested.size());
                    requested.push_back(marker);
                    return fixture.declarations[marker == 7 ? 0 : 1].binding;
                });
            EXPECT_EQ(requested, (std::vector<int>{7, 3}));
        }
        const auto constraints = fixture.system.meshNormalBoundaryConstraints();
        const auto measurements = systems::meshNormalConstraintStageMeasurements(fixture.system)
            .fittedALENormalOperatorStageMeasurementDeclarations();
        ASSERT_EQ(constraints.size(), 2u);
        ASSERT_EQ(measurements.size(), 2u);
        // The FE registries sort their published records by field and marker.
        const std::array markers{3, 7};
        const std::array mesh_sources{"mesh 3", "mesh 7"};
        const std::array fluid_sources{"fluid 3", "fluid 7"};
        for (std::size_t i = 0; i < 2; ++i) {
            const auto& constraint = constraints[i];
            EXPECT_EQ(constraint.boundary_marker, markers[i]);
            EXPECT_EQ(constraint.mesh_displacement_field, fixture.displacement);
            EXPECT_EQ(constraint.related_velocity_field, fixture.velocity);
            EXPECT_EQ(constraint.quantity, systems::MeshNormalBoundaryQuantity::MeshVelocityTrace);
            EXPECT_EQ(constraint.target_kind, systems::MeshNormalBoundaryTargetKind::FluidNormalVelocity);
            EXPECT_EQ(constraint.target_expression.node(), fixture.target.node());
            EXPECT_EQ(constraint.owner_component, "moving boundary");
            EXPECT_EQ(constraint.requires_mesh_flux_consistency, i == 1);
            EXPECT_EQ(constraint.mesh_flux_consistency_nitsche_gamma, i == 1 ? 8.0 : 0.0);
            ASSERT_TRUE(constraint.consumer_binding);
            EXPECT_EQ(constraint.consumer_binding->operator_tag, "moving");
            EXPECT_EQ(constraint.consumer_binding->mesh_descriptor_source, mesh_sources[i]);
            ASSERT_TRUE(constraint.consumer_binding->related_fluid);
            EXPECT_EQ(constraint.consumer_binding->related_fluid->descriptor_source, fluid_sources[i]);
            EXPECT_EQ(constraint.consumer_binding->related_fluid->enforcement_kind,
                      analysis::EnforcementKind::WeakConsistent);
            const auto& measurement = measurements[i];
            EXPECT_EQ(measurement.key.boundary_marker, markers[i]);
            EXPECT_EQ(measurement.key.mesh_displacement_field, fixture.displacement);
            EXPECT_EQ(measurement.related_velocity_field, fixture.velocity);
            EXPECT_EQ(measurement.normal_constraint.target_expression.node(), fixture.target.node());
            EXPECT_EQ(measurement.normal_constraint.consumer_binding, constraint.consumer_binding);
        }
        EXPECT_EQ(measurements[0].tangential_policy, systems::MeshTangentialBoundaryPolicy::Free);
        EXPECT_FALSE(measurements[0].prescribed_tangential_mesh_velocity);
        EXPECT_TRUE(measurements[0].tangential_target_gap_squared_integral_functional.empty());
        EXPECT_EQ(measurements[1].tangential_policy, systems::MeshTangentialBoundaryPolicy::Prescribed);
        EXPECT_EQ(measurements[1].prescribed_tangential_mesh_velocity,
                  (std::optional<std::array<Real, 3>>{{1, 2, 3}}));
        EXPECT_FALSE(measurements[1].tangential_target_gap_squared_integral_functional.empty());
    }
}

TEST(MeshBoundaryKinematics, PreparedDeclarationsKeepEarlierItemsWhenALaterItemFails)
{
    MovingBoundaryDeclarations fixture;
    fixture.declarations[1].constraint.owner_component.clear();
    EXPECT_ANY_THROW(systems::declareMeshNormalKinematics(
        fixture.system, "moving", std::move(fixture.declarations)));
    const auto constraints = fixture.system.meshNormalBoundaryConstraints();
    const auto measurements = systems::meshNormalConstraintStageMeasurements(fixture.system)
        .fittedALENormalOperatorStageMeasurementDeclarations();
    ASSERT_EQ(constraints.size(), 1u);
    EXPECT_EQ(constraints[0].boundary_marker, 7);
    ASSERT_EQ(measurements.size(), 1u);
    EXPECT_EQ(measurements[0].key.boundary_marker, 7);
}

TEST(MeshBoundaryKinematics, NormalRowWithoutVelocityThrowsTheGivenMessage)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    systems::FESystem sys(mesh);
    systems::MeshBoundaryRows rows{};
    rows.boundary_marker = 3;
    rows.normal_relation = true;
    rows.missing_velocity_message = "Owner: rows need a velocity";
    EXPECT_EQ(invalidArgumentMessage([&] {
                  systems::declareMeshBoundaryRowDescriptors(
                      sys, "op", rows, /*rows_installed=*/true);
              }),
              "Owner: rows need a velocity");
    EXPECT_TRUE(sys.boundaryConditionDescriptors().empty());
}

TEST(ContactLineGeometryCertificate, RegisteredCertificateCertifiesWithoutAContext)
{
    namespace interfaces = svmp::FE::interfaces;
    constexpr int wall = 4;
    auto mesh =
        std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(wall);
    systems::FESystem sys(mesh);
    const interfaces::ContactLineGeometryCheck check{
        .contact_marker = 41, .wall_marker = wall, .wall_normal = {0.0, 0.0, -1.0}};
    EXPECT_NO_THROW(interfaces::registerContactLineCertificate(
        sys,
        check,
        "unit_every_context:41",
        "unit_every_context",
        "unit_every_context_initial",
        interfaces::ContactLineCertificateRefresh::EveryContext));
    EXPECT_NO_THROW(interfaces::registerContactLineCertificate(
        sys,
        check,
        "unit_set_up:41",
        "unit_set_up",
        "unit_set_up_initial",
        interfaces::ContactLineCertificateRefresh::SetUpSystemContext));
}

TEST(GeneratedBoundaryAggregateTraceCertificate, CertifiedTraceSpacesAreLinearSimplexProducts)
{
    using svmp::FE::analysis::isCertifiedGeneratedBoundaryTraceSpace;
    const auto tet1 = std::make_shared<H1Space>(ElementType::Tetra4, /*order=*/1);
    const auto tri1 = std::make_shared<H1Space>(ElementType::Triangle3, /*order=*/1);
    const auto tet2 = std::make_shared<H1Space>(ElementType::Tetra4, /*order=*/2);
    EXPECT_TRUE(isCertifiedGeneratedBoundaryTraceSpace(ProductSpace(tet1, 3), 3));
    EXPECT_TRUE(isCertifiedGeneratedBoundaryTraceSpace(ProductSpace(tri1, 2), 2));
    EXPECT_FALSE(isCertifiedGeneratedBoundaryTraceSpace(ProductSpace(tet1, 3), 2));
    EXPECT_FALSE(isCertifiedGeneratedBoundaryTraceSpace(ProductSpace(tri1, 2), 3));
    EXPECT_FALSE(isCertifiedGeneratedBoundaryTraceSpace(ProductSpace(tet2, 3), 3));
    EXPECT_FALSE(isCertifiedGeneratedBoundaryTraceSpace(*tet1, 1));
}
