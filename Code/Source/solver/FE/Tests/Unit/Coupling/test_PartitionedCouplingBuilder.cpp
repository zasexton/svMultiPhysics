#include "Coupling/PartitionedCouplingBuilder.h"
#include "Core/FEException.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

CouplingValueDescriptor vectorDescriptor(int components)
{
    return CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = components,
    };
}

} // namespace

TEST(PartitionedCouplingBuilder, BuildsFieldExchangeDeclarations)
{
    PartitionedCouplingBuilder builder("fsi_wall");

    CouplingTransferDeclaration transfer;
    transfer.kind = CouplingTransferKind::Identity;

    builder.exchange(
               "solid_displacement",
               CouplingFieldUse{
                   .participant_name = "solid",
                   .field_name = "displacement",
               },
               CouplingFieldUse{
                   .participant_name = "fluid",
                   .field_name = "velocity",
               })
        .sharedInterface("wall")
        .value(vectorDescriptor(3))
        .transfer(transfer)
        .producerTemporal(CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Accepted,
        })
        .consumerTemporal(CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Current,
        });

    const auto& declarations = builder.declarations();
    ASSERT_EQ(declarations.size(), 1u);
    const auto& exchange = declarations.front();

    EXPECT_EQ(exchange.producer_port.contract_instance_name, "fsi_wall");
    EXPECT_EQ(exchange.producer_port.port_name,
              "solid_displacement.producer");
    EXPECT_EQ(exchange.consumer_port.contract_instance_name, "fsi_wall");
    EXPECT_EQ(exchange.consumer_port.port_name,
              "solid_displacement.consumer");
    ASSERT_TRUE(exchange.producer.has_value());
    EXPECT_EQ(exchange.producer->kind, CouplingEndpointKind::Field);
    ASSERT_TRUE(exchange.producer->participant_name.has_value());
    EXPECT_EQ(*exchange.producer->participant_name, "solid");
    EXPECT_EQ(exchange.producer->endpoint_name, "displacement");
    EXPECT_EQ(exchange.producer->temporal.slot, CouplingTemporalSlot::Accepted);
    ASSERT_TRUE(exchange.consumer.has_value());
    EXPECT_EQ(exchange.consumer->kind, CouplingEndpointKind::Field);
    ASSERT_TRUE(exchange.consumer->participant_name.has_value());
    EXPECT_EQ(*exchange.consumer->participant_name, "fluid");
    EXPECT_EQ(exchange.consumer->endpoint_name, "velocity");
    EXPECT_EQ(exchange.consumer->temporal.slot, CouplingTemporalSlot::Current);
    ASSERT_TRUE(exchange.shared_region_name.has_value());
    EXPECT_EQ(*exchange.shared_region_name, "wall");
    EXPECT_EQ(exchange.value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(exchange.value.components, 3);
    EXPECT_EQ(exchange.transfer.kind, CouplingTransferKind::Identity);
}

TEST(PartitionedCouplingBuilder, BuildsGenericEndpointExchangeDeclarations)
{
    PartitionedCouplingBuilder builder("junction");

    auto handle =
        builder.exchange(
                   "branch_pressure",
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::AuxiliaryOutput,
                       .participant_name = "junction_state",
                       .endpoint_name = "pressure",
                   },
                   CouplingEndpointRef{
                       .kind = CouplingEndpointKind::RegionData,
                       .participant_name = "branch",
                       .endpoint_name = "outlet_pressure",
                   })
            .producerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "junction_state",
                .region_name = "node",
            })
            .consumerRegion(CouplingRegionEndpointDeclaration{
                .participant_name = "branch",
                .region_name = "outlet",
            });

    const auto& exchange = handle.declaration();
    ASSERT_TRUE(exchange.producer.has_value());
    EXPECT_EQ(exchange.producer->kind, CouplingEndpointKind::AuxiliaryOutput);
    ASSERT_TRUE(exchange.consumer.has_value());
    EXPECT_EQ(exchange.consumer->kind, CouplingEndpointKind::RegionData);
    ASSERT_TRUE(exchange.producer_region.has_value());
    EXPECT_EQ(exchange.producer_region->region_name, "node");
    ASSERT_TRUE(exchange.consumer_region.has_value());
    EXPECT_EQ(exchange.consumer_region->region_name, "outlet");

    auto declarations = builder.takeDeclarations();
    ASSERT_EQ(declarations.size(), 1u);
    EXPECT_TRUE(builder.declarations().empty());
    EXPECT_EQ(declarations.front().producer_port.port_name,
              "branch_pressure.producer");
}

TEST(PartitionedCouplingBuilder, RejectsMissingContractOrExchangeNames)
{
    EXPECT_THROW(static_cast<void>(PartitionedCouplingBuilder("")),
                 InvalidArgumentException);

    PartitionedCouplingBuilder builder("contract");
    EXPECT_THROW(static_cast<void>(
                     builder.exchange(
                         "",
                         CouplingFieldUse{
                             .participant_name = "left",
                             .field_name = "primary",
                         },
                         CouplingFieldUse{
                             .participant_name = "right",
                             .field_name = "primary",
                         })),
                 InvalidArgumentException);
}
