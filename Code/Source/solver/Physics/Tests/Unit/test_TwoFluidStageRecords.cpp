/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// The two-fluid accepted-stage records are one accepted-stage record store
// of the FE system: created on first use, shared by every handle, empty in
// the read-only view before that, and a discarded attempt clears the
// pending group and ends the transaction.

#include <gtest/gtest.h>

#include "FE/Systems/AcceptedStageRecords.h"
#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/NavierStokes/TwoFluid/TwoFluidStageRecords.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::two_fluid::test {
using FE::systems::FESystem;
using FE::systems::acceptedStageRecords;
namespace {

TEST(TwoFluidStageRecords, AreOneAcceptedStageRecordStoreOfTheSystem)
{
    FESystem system(std::make_shared<FE::forms::test::SingleTetraMeshAccess>());
    const FESystem& view = system;
    EXPECT_EQ(view.findAttachment<TwoFluidStageRecordStore>(), nullptr);
    EXPECT_TRUE(twoFluidStageRecords(view).twoFluidAcceptedStageDiagnosticDeclarations().empty());
    EXPECT_TRUE(twoFluidStageRecords(view).twoFluidAcceptedStageDiagnosticHistory().empty());

    auto records = twoFluidStageRecords(system);   // creates the store
    EXPECT_NE(view.findAttachment<TwoFluidStageRecordStore>(), nullptr);
    EXPECT_TRUE(records.pendingTwoFluidAcceptedStageDiagnostics().empty());

    auto store = acceptedStageRecords<TwoFluidStageRecordStore>(system);
    store.beginTransaction();
    store.stagePending({TwoFluidAcceptedStageDiagnosticHistoryRecord{}});
    EXPECT_EQ(records.pendingTwoFluidAcceptedStageDiagnostics().size(), 1u);
    EXPECT_EQ(twoFluidStageRecords(view).pendingTwoFluidAcceptedStageDiagnostics().size(), 1u);

    records.discardPendingTwoFluidAcceptedStageDiagnostics();
    EXPECT_TRUE(twoFluidStageRecords(view).pendingTwoFluidAcceptedStageDiagnostics().empty());
    EXPECT_FALSE(store.transactionActive());
    EXPECT_TRUE(twoFluidStageRecords(view).twoFluidAcceptedStageDiagnosticHistory().empty());
}

} // namespace
} // namespace svmp::FE::systems::test
