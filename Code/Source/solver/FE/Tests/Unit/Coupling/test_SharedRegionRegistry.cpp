#include "Coupling/SharedRegionRegistry.h"

#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* registrySystemToken(std::uintptr_t value)
{
    return reinterpret_cast<const systems::FESystem*>(value);
}

CouplingRegionRef registryRegion(std::string participant_name,
                                 std::string region_name,
                                 CouplingRegionKind kind,
                                 int marker)
{
    return CouplingRegionRef{
        .participant_name = std::move(participant_name),
        .system_name = "system",
        .system = registrySystemToken(1),
        .region_name = std::move(region_name),
        .kind = kind,
        .marker = marker,
    };
}

} // namespace

TEST(SharedRegionRegistry, RegistersAndFindsParticipantRegions)
{
    SharedRegionRegistry registry;
    registry.add(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {
            registryRegion("left", "surface", CouplingRegionKind::InterfaceFace, 3),
            registryRegion("right", "surface", CouplingRegionKind::InterfaceFace, 3),
        },
    });

    EXPECT_TRUE(registry.validate().ok());
    ASSERT_NE(registry.find("interface"), nullptr);
    ASSERT_NE(registry.findParticipantRegion("interface", "left"), nullptr);
    EXPECT_EQ(registry.findParticipantRegion("interface", "right")->marker, 3);
}

TEST(SharedRegionRegistry, RegistersAndFindsNParticipantRegions)
{
    SharedRegionRegistry registry;
    registry.add(SharedRegionRef{
        .name = "triple_interface",
        .required_region_kind = CouplingRegionKind::Boundary,
        .required_participant_names = {"left", "middle", "right"},
        .participant_regions = {
            registryRegion("left", "surface", CouplingRegionKind::Boundary, 3),
            registryRegion("middle", "surface", CouplingRegionKind::Boundary, 4),
            registryRegion("right", "surface", CouplingRegionKind::Boundary, 5),
        },
    });

    EXPECT_TRUE(registry.validate().ok()) << formatDiagnostics(registry.validate());
    ASSERT_NE(registry.find("triple_interface"), nullptr);
    EXPECT_EQ(registry.find("triple_interface")->participant_regions.size(), 3u);
    ASSERT_NE(registry.findParticipantRegion("triple_interface", "middle"), nullptr);
    EXPECT_EQ(registry.findParticipantRegion("triple_interface", "middle")->marker, 4);
}

TEST(SharedRegionRegistry, RejectsMissingRequiredParticipantMappings)
{
    SharedRegionRegistry registry;
    registry.add(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
        .required_participant_names = {"left", "right"},
        .participant_regions = {
            registryRegion("left", "surface", CouplingRegionKind::Boundary, 3),
        },
    });

    const auto validation = registry.validate();
    EXPECT_FALSE(validation.ok());
    const auto diagnostics = formatDiagnostics(validation);
    EXPECT_NE(diagnostics.find("shared region is missing required participant mapping"),
              std::string::npos);
    EXPECT_NE(diagnostics.find("participant='right'"), std::string::npos);
    EXPECT_NE(diagnostics.find("region='interface'"), std::string::npos);
}

TEST(SharedRegionRegistry, RejectsDuplicateSharedRegionNames)
{
    SharedRegionRegistry registry;
    registry.add(SharedRegionRef{.name = "interface"});
    registry.add(SharedRegionRef{.name = "interface"});

    EXPECT_FALSE(registry.validate().ok());
}

TEST(SharedRegionRegistry, EnforcesRequiredRegionKindAsCompatibilityConstraint)
{
    SharedRegionRegistry registry;
    registry.add(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {
            registryRegion("left", "surface", CouplingRegionKind::Boundary, 3),
        },
    });

    const auto validation = registry.validate();
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("required region kind"), std::string::npos);
}
