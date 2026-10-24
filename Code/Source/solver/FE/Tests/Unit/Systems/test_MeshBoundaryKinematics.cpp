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
