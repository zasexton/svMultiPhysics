#include "Coupling/CouplingDeclaration.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* partitionedSystemToken(int index)
{
    return reinterpret_cast<const systems::FESystem*>(
        static_cast<std::uintptr_t>(index));
}

CouplingContextBuilder partitionedContextBuilder(int components)
{
    const auto left_system = partitionedSystemToken(1);
    const auto right_system = partitionedSystemToken(2);
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = components,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .field_name = "primary",
        .field_id = 2,
        .space = space,
        .components = components,
    });
    return builder;
}

CouplingContext partitionedContextWithComponents(int components)
{
    return partitionedContextBuilder(components).build();
}

CouplingContext partitionedContext()
{
    return partitionedContextWithComponents(1);
}

CouplingRegionRef partitionedRegion(std::string participant,
                                    std::string system_name,
                                    const systems::FESystem* system,
                                    CouplingInterfaceSide side,
                                    int marker)
{
    return CouplingRegionRef{
        .participant_name = std::move(participant),
        .system_name = std::move(system_name),
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = side,
    };
}

CouplingContext partitionedContextWithSharedRegion()
{
    auto builder = partitionedContextBuilder(1);
    const auto left = partitionedRegion(
        "left", "left_system", partitionedSystemToken(1),
        CouplingInterfaceSide::Minus, 10);
    const auto right = partitionedRegion(
        "right", "right_system", partitionedSystemToken(2),
        CouplingInterfaceSide::Plus, 11);
    builder.addRegion(left)
        .addRegion(right)
        .addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .participant_regions = {left, right},
        });
    return builder.build();
}

CouplingEndpointRef fieldEndpoint(std::string participant,
                                  CouplingTemporalSlotDescriptor temporal)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = std::move(participant),
        .endpoint_name = "primary",
        .temporal = temporal,
    };
}

CouplingEndpointRef fieldEndpoint(std::string participant)
{
    return fieldEndpoint(
        std::move(participant),
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current});
}

CouplingEndpointRef externalBufferEndpoint(std::string key)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::ExternalBuffer,
        .endpoint_name = std::move(key),
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::External,
        },
    };
}

CouplingPortId port(std::string name)
{
    return CouplingPortId{
        .contract_instance_name = "generic_instance",
        .port_name = std::move(name),
    };
}

CouplingExchangeDeclaration identityExchange()
{
    CouplingExchangeDeclaration exchange;
    exchange.producer_port = port("left_out");
    exchange.consumer_port = port("right_in");
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Scalar,
        .components = 1,
    };
    exchange.producer = fieldEndpoint("left");
    exchange.consumer = fieldEndpoint("right");
    exchange.transfer.kind = CouplingTransferKind::Identity;
    return exchange;
}

CouplingExternalBufferDescriptor externalBufferDescriptor(
    std::string name,
    CouplingValueDescriptor value,
    CouplingExternalBufferAccess access,
    std::vector<CouplingTemporalSlotDescriptor> supported_temporal_slots)
{
    return CouplingExternalBufferDescriptor{
        .buffer_name = std::move(name),
        .value = std::move(value),
        .access = access,
        .extents = {1},
        .strides = {1},
        .packing = "contiguous",
        .supported_temporal_slots = std::move(supported_temporal_slots),
        .layout_revision_key = 3,
        .data_revision_key = 5,
    };
}

CouplingDriverOwnedTransferDescriptor driverOwnedTransferDescriptor(
    std::string name,
    std::vector<CouplingValueRank> supported_ranks)
{
    return CouplingDriverOwnedTransferDescriptor{
        .transfer_name = std::move(name),
        .supported_ranks = std::move(supported_ranks),
        .supported_source_temporal_slots = {
            CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}},
        .supported_target_temporal_slots = {
            CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}},
        .registry_revision_key = 11,
    };
}

CouplingExchangeDeclaration interfaceExchange(CouplingValueDescriptor value,
                                              CouplingInterfaceFramePolicy frame_policy)
{
    auto exchange = identityExchange();
    exchange.value = std::move(value);
    exchange.transfer.kind = CouplingTransferKind::InterfacePointwiseInterpolation;
    exchange.transfer.interface_declaration = CouplingInterfaceTransferDeclaration{
        .frame_policy = frame_policy,
    };
    return exchange;
}

CouplingContractDeclaration partitionedDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.partitioned_exchange_declarations.push_back(identityExchange());
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "sync_group",
        .participant_names = {"left", "right"},
    });
    return declaration;
}

} // namespace

TEST(PartitionedCouplingPlanGenerator, GeneratesFieldIdentityExchange)
{
    const PartitionedCouplingPlanGenerator generator;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{identityExchange()};

    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    EXPECT_EQ(plan.exchanges[0].producer.field_id, 1);
    EXPECT_EQ(plan.exchanges[0].producer.system_name, "left_system");
    EXPECT_EQ(plan.exchanges[0].consumer.field_id, 2);
    EXPECT_EQ(plan.exchanges[0].transfer.kind, CouplingTransferKind::Identity);
    EXPECT_TRUE(plan.cycles.empty());
}

TEST(PartitionedCouplingPlanGenerator, GeneratesVectorFieldIdentityExchange)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContextWithComponents(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    EXPECT_EQ(plan.exchanges[0].value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(plan.exchanges[0].value.components, 2);
}

TEST(PartitionedCouplingPlanGenerator, RejectsFieldEndpointComponentMismatch)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("component count does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsGeneralTensorWithoutDriverOwnedTransfer)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::GeneralTensor,
        .components = 4,
        .tensor_extents = {2, 2},
        .tensor_packing = "row_major",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(4),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("general tensor partitioned values"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, AcceptsScalarInterfaceTransferMetadata)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(PartitionedCouplingPlanGenerator, AcceptsVectorInterfaceFrameTransform)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 3,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(3),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceFramePayloadMismatch)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("vector frame transforms require vector"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsSymmetricTensorInterfaceTransfer)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::SymmetricTensor,
            .components = 6,
        },
        CouplingInterfaceFramePolicy::None);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(6),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("symmetric tensor interface transfers"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsTrue2DVectorInterfaceTransformWithoutAdapter)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 2,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    exchange.transfer.interface_declaration->source_embedding_policy =
        CouplingFrameSourceEmbeddingPolicy::Embed2DInXY;
    exchange.transfer.interface_declaration->target_restriction_policy =
        CouplingFrameTargetRestrictionPolicy::RestrictToXY;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("true 2D vector interface transforms"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsVectorFramePassThroughWithoutLayout)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 4,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(4),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("pass-through components require component layout"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, ResolvesFieldHistoryTemporalSlot)
{
    auto exchange = identityExchange();
    exchange.producer = fieldEndpoint(
        "left",
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::History,
            .history_index = 2,
        });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::SystemStateHistory);
    ASSERT_TRUE(plan.exchanges[0].producer.temporal.storage_index.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.temporal.storage_index, 1);
    ASSERT_TRUE(plan.exchanges[0].producer.temporal.request.history_index.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.temporal.request.history_index, 2);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnsupportedFieldTemporalSlots)
{
    const std::vector<CouplingTemporalSlotDescriptor> unsupported_temporal_slots{
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted},
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted},
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Stage,
            .stage_index = 0,
        },
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External},
    };

    const PartitionedCouplingPlanGenerator generator;
    for (const auto& temporal : unsupported_temporal_slots) {
        SCOPED_TRACE(toString(temporal.slot));
        auto exchange = identityExchange();
        exchange.producer = fieldEndpoint("left", temporal);
        const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

        const auto validation = generator.validate(
            partitionedContext(),
            std::span<const CouplingExchangeDeclaration>(exchanges));

        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("field endpoint temporal slot"),
                  std::string::npos);
    }
}

TEST(PartitionedCouplingPlanGenerator, RejectsEndpointKindsWithoutResolver)
{
    auto exchange = identityExchange();
    exchange.producer = CouplingEndpointRef{
        .kind = CouplingEndpointKind::Parameter,
        .participant_name = "left",
        .endpoint_name = "coefficient",
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Current,
        },
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registry resolver"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferWithoutDescriptor)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registered descriptor"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesExternalBufferEndpointWithDescriptor)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::ReadOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}}),
    });
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].producer.external_buffer.has_value());
    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::ExternalBuffer);
    EXPECT_EQ(plan.exchanges[0].producer.layout_revision_key, 3u);
    EXPECT_EQ(plan.exchanges[0].producer.registry_revision_key, 5u);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferUnsupportedTemporalSlot)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::ReadOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}}),
    });

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("does not support the requested temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferProducerWithoutReadAccess)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::WriteOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}}),
    });

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires read access"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesDriverOwnedTransferDescriptor)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("copy", {CouplingValueRank::Scalar}));
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].transfer.driver_owned_descriptor.has_value());
    EXPECT_EQ(plan.exchanges[0].transfer.driver_owned_descriptor->registry_revision_key,
              11u);
}

TEST(PartitionedCouplingPlanGenerator, RejectsDriverOwnedTransferWithoutDescriptor)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registered descriptor"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsDriverOwnedTransferUnsupportedRank)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(2);
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("copy", {CouplingValueRank::Scalar}));

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("does not support the exchange value rank"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, InheritsExchangeSharedRegionForEndpointRegions)
{
    auto exchange = identityExchange();
    exchange.shared_region_name = "interface";
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "surface",
    };
    exchange.consumer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "right",
        .region_name = "surface",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto context = partitionedContextWithSharedRegion();
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].producer_region.has_value());
    ASSERT_TRUE(plan.exchanges[0].consumer_region.has_value());
    EXPECT_EQ(plan.exchanges[0].producer_region->marker, 10);
    EXPECT_EQ(plan.exchanges[0].producer_region->side, CouplingInterfaceSide::Minus);
    EXPECT_EQ(plan.exchanges[0].consumer_region->marker, 11);
    EXPECT_EQ(plan.exchanges[0].consumer_region->side, CouplingInterfaceSide::Plus);
}

TEST(PartitionedCouplingPlanGenerator, RejectsConflictingEndpointSharedRegion)
{
    auto exchange = identityExchange();
    exchange.shared_region_name = "interface";
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "surface",
        .shared_region_name = "other_interface",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("conflicts with the exchange shared region"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsMissingParticipantRegionEndpoint)
{
    auto exchange = identityExchange();
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "missing",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("region endpoint is missing"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnspecifiedTransfer)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::Unspecified;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("explicit transfer"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnknownFieldEndpoint)
{
    auto exchange = identityExchange();
    exchange.consumer = fieldEndpoint("missing");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("consumer field endpoint is missing"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RecordsDirectedExchangeCycles)
{
    auto forward = identityExchange();
    auto backward = identityExchange();
    backward.producer_port = port("right_out");
    backward.consumer_port = port("left_in");
    backward.producer = fieldEndpoint("right");
    backward.consumer = fieldEndpoint("left");

    forward.consumer_port = backward.producer_port;
    backward.consumer_port = forward.producer_port;

    const std::array<CouplingExchangeDeclaration, 2> exchanges{forward, backward};
    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_FALSE(plan.cycles.empty());
    EXPECT_GE(plan.cycles[0].ports.size(), 3u);
    EXPECT_EQ(plan.cycles[0].ports.front(), plan.cycles[0].ports.back());
}

TEST(PartitionedCouplingPlanGenerator, GeneratesFromContractDeclarations)
{
    const std::array<CouplingContractDeclaration, 1> declarations{
        partitionedDeclaration()};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    ASSERT_EQ(plan.group_hints.size(), 1u);
    EXPECT_EQ(plan.group_hints[0].name, "sync_group");
    EXPECT_EQ(plan.group_hints[0].participant_names.size(), 2u);
}

TEST(PartitionedCouplingPlanGenerator, MergesDeclarationAndTemplateExchanges)
{
    auto extra_exchange = identityExchange();
    extra_exchange.producer_port = port("right_in");
    extra_exchange.consumer_port = port("left_out");
    extra_exchange.producer = fieldEndpoint("right");
    extra_exchange.consumer = fieldEndpoint("left");

    const std::array<CouplingContractDeclaration, 1> declarations{
        partitionedDeclaration()};
    const std::array<CouplingExchangeDeclaration, 1> templates{extra_exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingExchangeDeclaration>(templates));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingExchangeDeclaration>(templates));

    EXPECT_EQ(plan.exchanges.size(), 2u);
    EXPECT_EQ(plan.group_hints.size(), 1u);
    EXPECT_FALSE(plan.cycles.empty());
}

TEST(PartitionedCouplingPlanGenerator, RejectsGroupHintWithUnknownParticipant)
{
    auto declaration = partitionedDeclaration();
    declaration.group_hints[0].participant_names.push_back("missing");
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("group hint references an unknown participant"),
              std::string::npos);
}
