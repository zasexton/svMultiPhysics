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

CouplingContext partitionedContextWithComponents(int components)
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
    return builder.build();
}

CouplingContext partitionedContext()
{
    return partitionedContextWithComponents(1);
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
    exchange.producer = CouplingEndpointRef{
        .kind = CouplingEndpointKind::ExternalBuffer,
        .endpoint_name = "driver_value",
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::External,
        },
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registered descriptor"),
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
