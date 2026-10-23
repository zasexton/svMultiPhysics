/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/MeshNormalConstraintStageMeasurements.h"
#include "TimeStepping/MeshNormalStageParticipant.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::MeshNormalConstraintStageMeasurementStore;
using svmp::FE::systems::meshNormalConstraintStageMeasurements;
using svmp::FE::timestepping::MeshNormalStageContext;
using svmp::FE::timestepping::MeshNormalStageParticipant;
using svmp::FE::timestepping::SchemeKind;
using svmp::FE::timestepping::StepHook;

std::shared_ptr<svmp::FE::forms::test::SingleTetraMeshAccess> tetraMesh()
{
    return std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
}

// An FE system with a mesh-displacement and a velocity field and no
// mesh-normal constraint.
struct TwoVectorFields {
    std::shared_ptr<svmp::FE::spaces::H1Space> scalar =
        std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, /*order=*/1);
    std::shared_ptr<svmp::FE::spaces::ProductSpace> vector =
        std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, /*components=*/3);
    FESystem system{tetraMesh()};
    FieldId displacement{};
    FieldId velocity{};

    TwoVectorFields()
    {
        displacement = system.addField(
            FieldSpec{.name = "mesh_displacement", .space = vector, .components = 3});
        velocity = system.addField(
            FieldSpec{.name = "u", .space = vector, .components = 3});
        system.setup();
    }
};

std::string invalidArgumentMessage(const std::function<void()>& call)
{
    try {
        call();
    } catch (const svmp::FE::InvalidArgumentException& error) {
        return error.what();
    }
    return {};
}

} // namespace

TEST(MeshNormalConstraintStageMeasurements, ReadOnlyAccessDoesNotCreateRecords)
{
    FESystem system(tetraMesh());
    const FESystem& const_system = system;
    const auto records = meshNormalConstraintStageMeasurements(const_system);
    EXPECT_TRUE(records.fittedALENormalOperatorStageMeasurementDeclarations().empty());
    EXPECT_TRUE(records.pendingFittedALENormalOperatorStageMeasurements().empty());
    EXPECT_TRUE(records.fittedALENormalOperatorStageMeasurementHistory().empty());
    EXPECT_EQ(system.findAttachment<MeshNormalConstraintStageMeasurementStore>(), nullptr);
}

TEST(MeshNormalConstraintStageMeasurements, RecordsAreOneAttachmentOfTheSystem)
{
    FESystem system(tetraMesh());
    auto records = meshNormalConstraintStageMeasurements(system);
    const auto* store = system.findAttachment<MeshNormalConstraintStageMeasurementStore>();
    ASSERT_NE(store, nullptr);
    EXPECT_TRUE(records.fittedALENormalOperatorStageMeasurementDeclarations().empty());
    records.discardPendingFittedALENormalOperatorStageMeasurements();
    EXPECT_TRUE(records.pendingFittedALENormalOperatorStageMeasurements().empty());
    (void)meshNormalConstraintStageMeasurements(system);
    EXPECT_EQ(system.findAttachment<MeshNormalConstraintStageMeasurementStore>(), store);
    EXPECT_FALSE(store->declarations_frozen);
    EXPECT_FALSE(store->transaction_active);
}

TEST(MeshNormalConstraintStageMeasurements, RegistrationRequiresAFluidNormalConstraint)
{
    TwoVectorFields f;
    const auto message = invalidArgumentMessage([&] {
        meshNormalConstraintStageMeasurements(f.system)
            .registerFittedALENormalOperatorStageMeasurement(f.displacement, /*marker=*/3);
    });
    EXPECT_NE(message.find("no matching mesh normal boundary constraint exists for the "
                           "fitted-ALE operator-stage measurement"),
              std::string::npos)
        << message;
    EXPECT_TRUE(meshNormalConstraintStageMeasurements(std::as_const(f.system))
                    .fittedALENormalOperatorStageMeasurementDeclarations()
                    .empty());
}

TEST(MeshNormalConstraintStageMeasurements, SchemeCoverageRequiresASupportedScheme)
{
    using svmp::FE::timestepping::requireMeshNormalStageSchemeCoverage;
    EXPECT_NO_THROW(requireMeshNormalStageSchemeCoverage(0u, false, "BDF2", 2));
    EXPECT_NO_THROW(requireMeshNormalStageSchemeCoverage(3u, true, "BackwardEuler", 1));
    try {
        requireMeshNormalStageSchemeCoverage(2u, false, "BDF2", 2);
        ADD_FAILURE() << "an unsupported scheme was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()),
                  "[svMultiPhysics::Application] Registered fitted-ALE operator-stage "
                  "normal measurements require a supported temporal-order-one scheme "
                  "before the transient TimeLoop starts; scheme='BDF2' temporal_order=2 "
                  "declaration_count=2. Supported schemes are BackwardEuler, DG0, and "
                  "first-order GeneralizedAlpha.");
    }
}

TEST(MeshNormalConstraintStageMeasurements, ParticipantWithoutMeasurementsHasNoHooks)
{
    TwoVectorFields f;
    MeshNormalStageContext context;
    context.system = &f.system;
    context.scheme = SchemeKind::BackwardEuler;
    context.scheme_name = "BackwardEuler";
    int fingerprints = 0;
    context.state_fingerprint = [&](const svmp::FE::backends::GenericVector&) {
        ++fingerprints;
        return std::uint64_t{0};
    };
    MeshNormalStageParticipant participant(context);
    EXPECT_FALSE(participant.evaluate_fitted_ale_operator_stage);
    EXPECT_EQ(participant.fitted_ale_operator_stage_state_size,
              f.system.dofHandler().getNumDofs());
    EXPECT_EQ(participant.hooks(), StepHook::None);
    participant.discardCandidate();
    EXPECT_EQ(fingerprints, 0);

    // Without measurements an unsupported scheme is not an error.
    context.scheme = SchemeKind::BDF2;
    context.scheme_name = "BDF2";
    MeshNormalStageParticipant bdf2(context);
    EXPECT_EQ(bdf2.hooks(), StepHook::None);
}
