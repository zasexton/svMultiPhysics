#include "Coupling/CouplingContext.h"

#include "Core/FEException.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* systemToken(std::uintptr_t value)
{
    return reinterpret_cast<const systems::FESystem*>(value);
}

std::shared_ptr<const spaces::FunctionSpace> scalarSpace()
{
    return std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
}

CouplingParticipantRef participant(std::string name,
                                   std::string system_name,
                                   const systems::FESystem* system)
{
    return CouplingParticipantRef{
        .participant_name = std::move(name),
        .system_name = std::move(system_name),
        .system = system,
    };
}

CouplingFieldRef field(std::string participant_name,
                       std::string system_name,
                       const systems::FESystem* system,
                       std::string field_name,
                       FieldId field_id)
{
    return CouplingFieldRef{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .field_name = std::move(field_name),
        .field_id = field_id,
        .space = scalarSpace(),
        .components = 1,
    };
}

CouplingRegionRef region(std::string participant_name,
                         std::string system_name,
                         const systems::FESystem* system,
                         std::string region_name,
                         CouplingRegionKind kind,
                         int marker)
{
    return CouplingRegionRef{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .region_name = std::move(region_name),
        .kind = kind,
        .marker = marker,
    };
}

CouplingTemporalSlotDescriptor currentSlot()
{
    return CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current};
}

CouplingExternalBufferDescriptor externalBuffer(std::string name,
                                                std::uint64_t data_revision)
{
    return CouplingExternalBufferDescriptor{
        .buffer_name = std::move(name),
        .value = CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        .extents = {1},
        .strides = {1},
        .packing = "contiguous",
        .supported_temporal_slots = {currentSlot()},
        .data_revision_key = data_revision,
    };
}

CouplingDriverOwnedTransferDescriptor driverOwnedTransfer(std::string name)
{
    return CouplingDriverOwnedTransferDescriptor{
        .transfer_name = std::move(name),
        .supported_ranks = {CouplingValueRank::Scalar},
        .supported_source_temporal_slots = {currentSlot()},
        .supported_target_temporal_slots = {currentSlot()},
        .registry_revision_key = 7,
    };
}

} // namespace

TEST(CouplingContext, ResolvesParticipantsFieldsAndRegions)
{
    const auto* system = systemToken(1);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addField(field("left", "shared_system", system, "primary", 0))
        .addRegion(region("left", "shared_system", system, "surface",
                          CouplingRegionKind::Boundary, 12));

    const auto context = builder.build();

    EXPECT_TRUE(context.hasParticipant("left"));
    EXPECT_TRUE(context.hasField("left", "primary"));
    EXPECT_TRUE(context.hasRegion("left", "surface"));

    EXPECT_EQ(context.participant("left").system_name, "shared_system");
    EXPECT_EQ(context.field("left", "primary").field_id, 0);
    EXPECT_EQ(context.region("left", "surface").marker, 12);
}

TEST(CouplingContext, RejectsDuplicateFieldMappings)
{
    const auto* system = systemToken(1);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addField(field("left", "shared_system", system, "primary", 0))
        .addField(field("left", "shared_system", system, "primary", 1));

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("duplicate coupling field mapping"),
              std::string::npos);
}

TEST(CouplingContext, RejectsFieldsWithoutOwningParticipant)
{
    const auto* system = systemToken(1);

    CouplingContextBuilder builder;
    builder.addField(field("left", "shared_system", system, "primary", 0));

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("unknown participant"), std::string::npos);
}

TEST(CouplingContext, RejectsInvalidFieldMetadata)
{
    const auto* system = systemToken(1);

    auto bad_field = field("left", "shared_system", system, "primary", 0);
    bad_field.space.reset();

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addField(bad_field);

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("field id, space"), std::string::npos);
}

TEST(CouplingContext, SharedRegionLookupReturnsParticipantMapping)
{
    const auto* system = systemToken(1);
    const auto surface = region("left", "shared_system", system, "surface",
                                CouplingRegionKind::Boundary, 12);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addRegion(surface)
        .addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::Boundary,
            .participant_regions = {surface},
        });

    const auto context = builder.build();
    EXPECT_TRUE(context.hasSharedRegion("interface"));
    EXPECT_EQ(context.sharedRegion("interface", "left").marker, 12);
    EXPECT_EQ(context.sharedRegionGroup("interface").participant_regions.size(), 1u);
}

TEST(CouplingContext, ResolvesExternalBufferDescriptorsByScope)
{
    const auto* system = systemToken(1);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addExternalBuffer(CouplingExternalBufferRegistration{
            .descriptor = externalBuffer("driver_value", 1),
        })
        .addExternalBuffer(CouplingExternalBufferRegistration{
            .participant_name = "left",
            .descriptor = externalBuffer("driver_value", 2),
        });

    const auto context = builder.build();
    const auto* global =
        context.externalBufferDescriptor(std::nullopt, "driver_value");
    ASSERT_NE(global, nullptr);
    EXPECT_EQ(global->data_revision_key, 1u);

    const auto* scoped = context.externalBufferDescriptor(
        std::optional<std::string_view>{"left"},
        "driver_value");
    ASSERT_NE(scoped, nullptr);
    EXPECT_EQ(scoped->data_revision_key, 2u);

    EXPECT_EQ(context.externalBufferDescriptor(
                  std::optional<std::string_view>{"right"},
                  "driver_value"),
              nullptr);
}

TEST(CouplingContext, ResolvesDriverOwnedTransferDescriptors)
{
    CouplingContextBuilder builder;
    builder.addDriverOwnedTransfer(driverOwnedTransfer("copy"));

    const auto context = builder.build();
    const auto* descriptor = context.driverOwnedTransfer("copy");
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->registry_revision_key, 7u);
    ASSERT_EQ(descriptor->supported_ranks.size(), 1u);
    EXPECT_EQ(descriptor->supported_ranks[0], CouplingValueRank::Scalar);
}

TEST(CouplingContext, RejectsDuplicateExternalBufferDescriptorsInOneScope)
{
    CouplingContextBuilder builder;
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBuffer("driver_value", 1),
    });
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBuffer("driver_value", 2),
    });

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("duplicate external buffer descriptor"),
              std::string::npos);
}

TEST(CouplingContext, RejectsInvalidDriverOwnedTransferDescriptors)
{
    auto descriptor = driverOwnedTransfer("");
    descriptor.supported_ranks.push_back(CouplingValueRank::Scalar);

    CouplingContextBuilder builder;
    builder.addDriverOwnedTransfer(descriptor);

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a name"), std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("duplicate ranks"), std::string::npos);
}

TEST(CouplingContext, MissingLookupsThrow)
{
    CouplingContext context;
    EXPECT_THROW(static_cast<void>(context.participant("missing")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.field("missing", "primary")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.region("missing", "surface")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.sharedRegionGroup("missing")), InvalidArgumentException);
}
