/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "LevelSet/AcceptedFunctionalRecorder.h"

#include <cstdint>
#include <set>

namespace {

using svmp::FE::level_set::acceptedFunctionalDeclarationContentKey;
using svmp::FE::systems::FreeSurfaceCapillaryBalanceMethod;
using svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration;

FreeSurfaceDiscreteFunctionalDeclaration declaration()
{
    FreeSurfaceDiscreteFunctionalDeclaration result{};
    result.interface_marker = 4;
    result.level_set_field = 2;
    result.velocity_field = 0;
    result.geometry_domain_id = "liquid";
    result.parameters.surface_tension = 0.07;
    result.parameters.volume_multiplier = 1.0;
    result.owner_component = "test";
    return result;
}

} // namespace

TEST(AcceptedFunctionalRecorder, ContentKeyIsDeterministicAndSeesTheDeclaredContent)
{
    const auto base = declaration();
    const auto key = acceptedFunctionalDeclarationContentKey(base);
    EXPECT_EQ(acceptedFunctionalDeclarationContentKey(declaration()), key);

    std::set<std::uint64_t> keys{key};
    auto marker = base;
    marker.interface_marker = 5;
    keys.insert(acceptedFunctionalDeclarationContentKey(marker));
    auto field = base;
    field.level_set_field = 7;
    keys.insert(acceptedFunctionalDeclarationContentKey(field));
    auto domain = base;
    domain.geometry_domain_id = "gas";
    keys.insert(acceptedFunctionalDeclarationContentKey(domain));
    auto tension = base;
    tension.parameters.surface_tension = 0.0725;
    keys.insert(acceptedFunctionalDeclarationContentKey(tension));
    auto wall = base;
    wall.parameters.young_wall_coefficients.push_back({});
    wall.parameters.young_wall_coefficients.back().boundary_marker = 2;
    keys.insert(acceptedFunctionalDeclarationContentKey(wall));
    EXPECT_EQ(keys.size(), 6u);
}

TEST(AcceptedFunctionalRecorder, ContentKeyTreatsSignedZeroAsZero)
{
    auto positive = declaration();
    positive.parameters.surface_tension = 0.0;
    auto negative = declaration();
    negative.parameters.surface_tension = -0.0;
    EXPECT_EQ(acceptedFunctionalDeclarationContentKey(positive),
              acceptedFunctionalDeclarationContentKey(negative));
}

TEST(AcceptedFunctionalRecorder, ContentKeySeesTheOwnerAndTheCapillaryMethod)
{
    const auto key = acceptedFunctionalDeclarationContentKey(declaration());
    auto owner = declaration();
    owner.owner_component = "another owner";
    EXPECT_NE(acceptedFunctionalDeclarationContentKey(owner), key);
    auto method = declaration();
    method.capillary_balance_method =
        FreeSurfaceCapillaryBalanceMethod::DiscreteEnergyVolumeStationarity;
    EXPECT_NE(acceptedFunctionalDeclarationContentKey(method), key);
    EXPECT_NE(key, 0u);
}
