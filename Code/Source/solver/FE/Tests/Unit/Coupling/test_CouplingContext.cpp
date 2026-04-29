#include "Coupling/CouplingContext.h"

#include "Core/FEException.h"
#include "Spaces/H1Space.h"
#include "Systems/InterfaceOperators.h"

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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::search::LogicalInterfaceRegionId logicalRegion(std::string persistent_id,
                                                     int label)
{
    return svmp::search::LogicalInterfaceRegionId{
        .persistent_id = std::move(persistent_id),
        .physical_label = label,
    };
}

svmp::search::InterfaceMap interfaceMap(std::string name)
{
    svmp::search::InterfaceMap map;
    map.name = std::move(name);
    map.source.boundary_label = 12;
    map.source.logical_region = logicalRegion("left-surface", 12);
    map.target.boundary_label = 7;
    map.target.logical_region = logicalRegion("right-surface", 7);
    map.state = svmp::search::InterfaceMapState::Committed;
    return map;
}

CouplingInterfaceMapProvenance interfaceMapProvenance(std::string map_name)
{
    return CouplingInterfaceMapProvenance{
        .interface_map_name = std::move(map_name),
        .interface_entry_name = "surface_pair",
        .interface_search_registry_name = "searches",
        .source_system_name = "left_system",
        .target_system_name = "right_system",
        .source_interface_marker = 12,
        .target_interface_marker = 7,
        .source_logical_region = logicalRegion("left-surface", 12),
        .target_logical_region = logicalRegion("right-surface", 7),
        .map_state = svmp::search::InterfaceMapState::Committed,
        .operator_state = systems::InterfaceOperatorState::AcceptedTimeStep,
    };
}
#endif

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

TEST(CouplingContext, PreservesInterfaceFieldScopeAndMarker)
{
    const auto* system = systemToken(1);
    auto interface_field = field("left", "shared_system", system, "trace", 0);
    interface_field.scope = systems::FieldScope::InterfaceFace;
    interface_field.interface_marker = 9;
    auto interface_region = region("left", "shared_system", system, "surface",
                                   CouplingRegionKind::InterfaceFace, 9);
    interface_region.side = CouplingInterfaceSide::Minus;

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addField(interface_field)
        .addRegion(interface_region);

    const auto context = builder.build();
    const auto resolved = context.field("left", "trace");
    EXPECT_EQ(resolved.scope, systems::FieldScope::InterfaceFace);
    EXPECT_EQ(resolved.interface_marker, 9);
}

TEST(CouplingContext, RejectsInconsistentInterfaceFieldAndRegionMetadata)
{
    const auto* system = systemToken(1);
    auto missing_marker = field("left", "shared_system", system, "trace", 0);
    missing_marker.scope = systems::FieldScope::InterfaceFace;

    auto missing_region = field("left", "shared_system", system, "trace_with_marker", 1);
    missing_region.scope = systems::FieldScope::InterfaceFace;
    missing_region.interface_marker = 9;

    auto volume_marker = field("left", "shared_system", system, "volume", 2);
    volume_marker.interface_marker = 9;

    auto invalid_region = region("left", "shared_system", system, "surface",
                                 CouplingRegionKind::InterfaceFace, -1);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "shared_system", system))
        .addField(missing_marker)
        .addField(missing_region)
        .addField(volume_marker)
        .addRegion(invalid_region);

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    const auto diagnostics = formatDiagnostics(validation);
    EXPECT_NE(diagnostics.find("interface coupling field requires an interface marker"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("interface coupling field requires a matching interface region"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("non-interface coupling field cannot specify an interface marker"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("interface coupling region requires an interface marker"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("interface coupling region requires a minus or plus side"),
              std::string::npos);
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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(CouplingContext, ResolvesCommittedInterfaceMapRuntimeHandles)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);
    svmp::search::InterfaceSearchRegistry registry;
    registry.commit_map(interfaceMap("surface_map"));

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "left_system", left_system))
        .addParticipant(participant("right", "right_system", right_system))
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        });

    const auto context = builder.build();
    const auto handles = context.interfaceMapHandles(
        interfaceMapProvenance("surface_map"));

    EXPECT_EQ(handles.source_system, left_system);
    EXPECT_EQ(handles.target_system, right_system);
    EXPECT_EQ(handles.search_registry, &registry);
    EXPECT_EQ(handles.sliding_map, nullptr);
    ASSERT_NE(handles.interface_map, nullptr);
    EXPECT_EQ(handles.interface_map->name, "surface_map");
}

TEST(CouplingContext, ResolvesSlidingInterfaceMapRuntimeHandles)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);
    svmp::search::InterfaceSearchRegistry registry;
    systems::SlidingInterfaceMap sliding;
    sliding.name = "surface_map";
    sliding.interface_map = interfaceMap("surface_map");
    sliding.state = systems::InterfaceOperatorState::AcceptedTimeStep;

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "left_system", left_system))
        .addParticipant(participant("right", "right_system", right_system))
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        })
        .addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{
            .interface_map_name = "surface_map",
            .sliding_map = &sliding,
        });

    const auto context = builder.build();
    const auto handles = context.interfaceMapHandles(
        interfaceMapProvenance("surface_map"));

    EXPECT_EQ(context.interfaceSearchRegistry("searches"), &registry);
    EXPECT_EQ(context.slidingInterfaceMap("surface_map"), &sliding);
    EXPECT_EQ(handles.sliding_map, &sliding);
    EXPECT_EQ(handles.interface_map, &sliding.interface_map);
}

TEST(CouplingContext, RejectsInvalidInterfaceRuntimeRegistrations)
{
    svmp::search::InterfaceSearchRegistry registry;
    systems::SlidingInterfaceMap sliding;

    CouplingContextBuilder builder;
    builder.addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{})
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        })
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        })
        .addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{})
        .addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{
            .interface_map_name = "surface_map",
            .sliding_map = &sliding,
        })
        .addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{
            .interface_map_name = "surface_map",
            .sliding_map = &sliding,
        });

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    const auto diagnostics = formatDiagnostics(validation);
    EXPECT_NE(diagnostics.find("interface search registry registration requires"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("duplicate interface search registry registration"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("sliding interface map registration requires"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("duplicate sliding interface map registration"),
              std::string::npos);
}

TEST(CouplingContext, MissingInterfaceRuntimeHandleLookupsThrow)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "left_system", left_system))
        .addParticipant(participant("right", "right_system", right_system));

    const auto context = builder.build();
    EXPECT_EQ(context.interfaceSearchRegistry("missing"), nullptr);
    EXPECT_EQ(context.slidingInterfaceMap("missing"), nullptr);
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(
                     interfaceMapProvenance("surface_map"))),
                 InvalidArgumentException);
}

TEST(CouplingContext, RejectsMismatchedInterfaceMapRuntimeState)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);
    svmp::search::InterfaceSearchRegistry registry;
    const auto map = interfaceMap("surface_map");
    registry.commit_map(map);

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "left_system", left_system))
        .addParticipant(participant("right", "right_system", right_system))
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        });
    const auto context = builder.build();

    auto provenance = interfaceMapProvenance("surface_map");
    provenance.map_state = svmp::search::InterfaceMapState::Trial;
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(provenance)),
                 InvalidArgumentException);

    provenance = interfaceMapProvenance("surface_map");
    provenance.map_revision_key = map.revision_key() + 1u;
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(provenance)),
                 InvalidArgumentException);

    provenance = interfaceMapProvenance("surface_map");
    provenance.source_logical_region = logicalRegion("other-surface", 12);
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(provenance)),
                 InvalidArgumentException);
}

TEST(CouplingContext, RejectsMismatchedSlidingInterfaceMapRuntimeState)
{
    const auto* left_system = systemToken(1);
    const auto* right_system = systemToken(2);
    svmp::search::InterfaceSearchRegistry registry;
    systems::SlidingInterfaceMap sliding;
    sliding.name = "surface_map";
    sliding.map_kind = systems::SlidingInterfaceMapKind::Sliding;
    sliding.interface_map = interfaceMap("surface_map");
    sliding.state = systems::InterfaceOperatorState::Trial;
    sliding.trial_revision_key = 5;

    CouplingContextBuilder builder;
    builder.addParticipant(participant("left", "left_system", left_system))
        .addParticipant(participant("right", "right_system", right_system))
        .addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
            .registry_name = "searches",
            .registry = &registry,
        })
        .addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{
            .interface_map_name = "surface_map",
            .sliding_map = &sliding,
        });
    const auto context = builder.build();

    auto provenance = interfaceMapProvenance("surface_map");
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(provenance)),
                 InvalidArgumentException);

    provenance.operator_state = systems::InterfaceOperatorState::Trial;
    provenance.trial_revision_key = 6;
    EXPECT_THROW(static_cast<void>(context.interfaceMapHandles(provenance)),
                 InvalidArgumentException);
}
#endif

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

TEST(CouplingContext, RejectsUnsupportedExternalBufferScalarType)
{
    auto descriptor = externalBuffer("driver_value", 1);
    descriptor.scalar_type = "Float";

    CouplingContextBuilder builder;
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = descriptor,
    });

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("scalar type must be Real"),
              std::string::npos);
}

TEST(CouplingContext, RejectsDuplicateExternalBufferTemporalSlots)
{
    auto descriptor = externalBuffer("driver_value", 1);
    descriptor.supported_temporal_slots.push_back(currentSlot());

    CouplingContextBuilder builder;
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = descriptor,
    });

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "external buffer descriptor has duplicate temporal slots"),
              std::string::npos);
}

TEST(CouplingContext, RejectsInvalidDriverOwnedTransferDescriptors)
{
    auto descriptor = driverOwnedTransfer("");
    descriptor.supported_ranks.push_back(CouplingValueRank::Scalar);
    descriptor.supported_source_temporal_slots.push_back(currentSlot());
    descriptor.supported_target_temporal_slots.push_back(currentSlot());

    CouplingContextBuilder builder;
    builder.addDriverOwnedTransfer(descriptor);

    const auto validation = builder.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a name"), std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("duplicate ranks"), std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("source has duplicate temporal slots"),
              std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("target has duplicate temporal slots"),
              std::string::npos);
}

TEST(CouplingContext, MissingLookupsThrow)
{
    CouplingContext context;
    EXPECT_THROW(static_cast<void>(context.participant("missing")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.field("missing", "primary")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.region("missing", "surface")), InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(context.sharedRegionGroup("missing")), InvalidArgumentException);
}
