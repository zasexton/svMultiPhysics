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

CouplingContext partitionedContext()
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
        .components = 1,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .field_name = "primary",
        .field_id = 2,
        .space = space,
        .components = 1,
    });
    return builder.build();
}

CouplingEndpointRef fieldEndpoint(std::string participant)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = std::move(participant),
        .endpoint_name = "primary",
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Current,
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
