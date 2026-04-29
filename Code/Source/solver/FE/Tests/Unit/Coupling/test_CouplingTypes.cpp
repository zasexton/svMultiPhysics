#include "Coupling/CouplingTypes.h"
#include "Coupling/TransferPlan.h"

#include <gtest/gtest.h>

using namespace svmp::FE::coupling;

TEST(CouplingTypes, PortIdentityUsesContractInstanceAndPortName)
{
    const CouplingPortId a{"contract_a", "out"};
    const CouplingPortId b{"contract_a", "in"};
    const CouplingPortId c{"contract_b", "out"};

    EXPECT_TRUE(a.valid());
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(validateCouplingPortId(a).ok());

    const CouplingPortId invalid{"", "out"};
    EXPECT_FALSE(validateCouplingPortId(invalid).ok());
}

TEST(CouplingTypes, ValueDescriptorValidatesGeneralTensorShape)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::GeneralTensor;
    value.components = 6;
    value.tensor_extents = {2, 3};
    value.tensor_packing = "row_major";

    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
    EXPECT_EQ(couplingTensorExtentProduct(value.tensor_extents), 6u);

    value.components = 5;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, CoordinateConfigurationMapsExplicitlyToMeshConfiguration)
{
    EXPECT_EQ(toMeshConfiguration(CouplingCoordinateConfiguration::Reference),
              svmp::Configuration::Reference);
    EXPECT_EQ(toMeshConfiguration(CouplingCoordinateConfiguration::Current),
              svmp::Configuration::Current);
    EXPECT_FALSE(toMeshConfiguration(
                     static_cast<CouplingCoordinateConfiguration>(255))
                     .has_value());
}

TEST(CouplingTypes, ValueDescriptorRequiresMixedBlockLayout)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::MixedBlock;
    value.components = 2;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"primary", "secondary"};
    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, ValueDescriptorRejectsAmbiguousComponentLayout)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::Vector;
    value.components = 2;
    value.component_layout = {"normal", ""};
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"normal", "normal"};
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"normal", "tangent"};
    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, TemporalSlotValidationUsesLogicalHistoryAndStageRules)
{
    CouplingTemporalSlotDescriptor history;
    history.slot = CouplingTemporalSlot::History;
    history.history_index = 1;
    EXPECT_TRUE(validateCouplingTemporalSlot(history).ok());

    history.history_index = 0;
    EXPECT_FALSE(validateCouplingTemporalSlot(history).ok());

    CouplingTemporalSlotDescriptor current;
    current.slot = CouplingTemporalSlot::Current;
    current.history_index = 1;
    EXPECT_FALSE(validateCouplingTemporalSlot(current).ok());

    CouplingTemporalSlotDescriptor stage;
    stage.slot = CouplingTemporalSlot::Stage;
    stage.stage_index = 0;
    EXPECT_TRUE(validateCouplingTemporalSlot(stage).ok());
}

TEST(CouplingTypes, EndpointValidationRequiresParticipantScopeForFEBackedKinds)
{
    CouplingEndpointRef endpoint;
    endpoint.kind = CouplingEndpointKind::Field;
    endpoint.endpoint_name = "primary";
    endpoint.participant_name = "participant";
    EXPECT_TRUE(validateCouplingEndpointRef(endpoint).ok());

    endpoint.participant_name.reset();
    EXPECT_FALSE(validateCouplingEndpointRef(endpoint).ok());

    endpoint.kind = CouplingEndpointKind::ExternalBuffer;
    EXPECT_TRUE(validateCouplingEndpointRef(endpoint).ok());
}

TEST(CouplingTypes, TransferKindIdentifiesInterfaceTransfers)
{
    EXPECT_FALSE(isInterfaceTransferKind(CouplingTransferKind::Identity));
    EXPECT_TRUE(isInterfaceTransferKind(CouplingTransferKind::InterfacePointwiseInterpolation));
    EXPECT_STREQ(toString(CouplingTransferKind::DriverOwned), "driver_owned");
}
