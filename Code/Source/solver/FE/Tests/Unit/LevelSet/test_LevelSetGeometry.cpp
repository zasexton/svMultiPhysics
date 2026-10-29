/**
 * @file test_LevelSetGeometry.cpp
 * @brief Typed domains of a bound level-set geometry use the stable generated markers
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/Vocabulary.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

namespace svmp::FE::level_set::test {
namespace {

using forms::CutVolumeSide;
using forms::FormExpr;

struct GeometrySystem {
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh{
        std::make_shared<forms::test::SingleTetraMeshAccess>()};
    std::shared_ptr<spaces::H1Space> scalar{
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1)};
    std::shared_ptr<spaces::ProductSpace> vector{
        std::make_shared<spaces::ProductSpace>(scalar, 3)};
    std::unique_ptr<systems::FESystem> system{};
    FieldId phi{INVALID_FIELD_ID};
    FieldId u{INVALID_FIELD_ID};

    GeometrySystem()
    {
        system = std::make_unique<systems::FESystem>(mesh);
        u = system->addField(systems::FieldSpec{.name = "u", .space = vector, .components = 3});
        phi = system->addField(systems::FieldSpec{.name = "phi", .space = scalar, .components = 1});
    }
};

TEST(LevelSetGeometry, BindResolvesTheStableGeneratedInterfaceMarker)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(
        *s.system, {.level_set_field_name = "phi", .domain_id = "drop", .isovalue = 0.25});
    interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
    key.domain_id = "drop";
    key.isovalue = 0.25;
    key.requested_marker = -1;
    EXPECT_EQ(G.levelSetField(), s.phi);
    EXPECT_EQ(G.interfaceMarker(), interfaces::stableGeneratedInterfaceMarker(key));
    EXPECT_EQ(G.interface().marker(), G.interfaceMarker());

    // By field id, and with an explicit marker.
    const auto by_id = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .isovalue = 0.25});
    EXPECT_EQ(by_id.interfaceMarker(), G.interfaceMarker());
    const auto explicit_marker =
        LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .interface_marker = 12});
    EXPECT_EQ(explicit_marker.interfaceMarker(), 12);
}

TEST(LevelSetGeometry, BindRejectsUnknownOrVectorFieldsAndMissingDomainIds)
{
    GeometrySystem s;
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field_name = "psi", .domain_id = "drop"}),
                 std::invalid_argument);
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field = s.u, .domain_id = "drop"}),
                 std::invalid_argument);
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi}), std::invalid_argument);
}

TEST(LevelSetGeometry, SideDomainsUseTheInterfaceMarkerAndSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    for (const auto side : {CutVolumeSide::Negative, CutVolumeSide::Positive}) {
        const auto L = G.side(side);
        EXPECT_EQ(L.side(), side);
        EXPECT_EQ(L.volume(), forms::VolumeDomain::cut(G.interfaceMarker(), side));
        EXPECT_EQ(L.interface(), forms::InterfaceDomain(G.interfaceMarker()));
        EXPECT_EQ(L.cutFacets(), forms::FacetDomain::cutAdjacent(G.interfaceMarker(), side));
    }
}

TEST(LevelSetGeometry, OutwardNormalIsTheRuleNormalFlippedOnThePositiveSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    const auto n = FormExpr::normal();
    EXPECT_EQ(systems::formExprTreeHash(G.side(CutVolumeSide::Negative).outwardNormal()),
              systems::formExprTreeHash(n));
    EXPECT_EQ(systems::formExprTreeHash(G.side(CutVolumeSide::Positive).outwardNormal()),
              systems::formExprTreeHash(-n));
    EXPECT_EQ(G.side(CutVolumeSide::Positive).outwardNormal().toString(), (-n).toString());
}

TEST(LevelSetGeometry, ContactLineUsesTheStableIntersectionMarker)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .isovalue = 0.0});
    interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
    key.domain_id = "drop";
    key.isovalue = 0.0;
    key.interface_marker = G.interfaceMarker();
    key.boundary_marker = 4;
    key.requested_marker = -1;
    EXPECT_EQ(G.contactLineKey(4).stableKey(), key.stableKey());
    const auto CL = G.contactLine(4);
    EXPECT_EQ(CL.marker(), interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key));
    EXPECT_EQ(CL.wallBoundaryMarker(), 4);
    EXPECT_EQ(G.side(CutVolumeSide::Negative).contactLine(4), CL);
    key.requested_marker = 77;
    EXPECT_EQ(G.contactLine(4, 77).marker(),
              interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key));
}

TEST(LevelSetGeometry, WettedBoundaryIsTheGeneratedActiveSubsetOfTheSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    for (const auto side : {CutVolumeSide::Negative, CutVolumeSide::Positive}) {
        const auto L = G.side(side);
        interfaces::GeneratedActiveBoundaryMarkerKey key;
        key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
        key.domain_id = "drop";
        key.isovalue = 0.0;
        key.interface_marker = G.interfaceMarker();
        key.boundary_marker = 2;
        key.side = side == CutVolumeSide::Negative ? geometry::CutIntegrationSide::Negative
                                                   : geometry::CutIntegrationSide::Positive;
        EXPECT_EQ(L.boundaryKey(2).stableKey(), key.stableKey());
        const auto expected = forms::BoundaryDomain::generatedActiveSubset(
            2, interfaces::stableGeneratedActiveBoundaryMarker(key));
        EXPECT_EQ(L.boundary(2), expected);
        EXPECT_EQ(L.wetted(2), expected);
    }
    EXPECT_NE(G.side(CutVolumeSide::Negative).wetted(2), G.side(CutVolumeSide::Positive).wetted(2));
}

TEST(LevelSetGeometry, DomainsGiveTheSameFormsAsMarkersWrittenByHand)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    const auto L = G.side(CutVolumeSide::Positive);
    const auto v = forms::TestField(s.u, *s.vector, "v");
    const auto u = forms::StateField(s.u, *s.vector, "u");
    const auto integrand = forms::inner(L.outwardNormal(), v);
    const int m = G.interfaceMarker();
    EXPECT_EQ(systems::formExprTreeHash(integrand.dI(L.interface())),
              systems::formExprTreeHash(integrand.dI(m)));
    EXPECT_EQ(systems::formExprTreeHash(forms::inner(u, v).dx(L.volume())),
              systems::formExprTreeHash(forms::inner(u, v).dCutVolume(m, CutVolumeSide::Positive)));
    const auto CL = G.contactLine(4);
    EXPECT_EQ(systems::formExprTreeHash(integrand.dL(CL)),
              systems::formExprTreeHash(integrand.dI(CL.marker())));
    const auto W = L.wetted(4);
    EXPECT_EQ(systems::formExprTreeHash(forms::inner(u, v).ds(W)),
              systems::formExprTreeHash(forms::inner(u, v).dExteriorBoundary(
                  forms::ExteriorBoundaryMeasure::generatedActiveSubset(4, W.generatedActiveBoundaryMarker()))));
}

TEST(LevelSetGeometry, CutVolumeRequestPolicyKeyIgnoresProvenance)
{
    ActiveCutVolumeRequest a;
    a.level_set_field_name = "phi";
    a.domain_id = "drop";
    ActiveCutVolumeRequest b = a;
    b.origin = ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary;
    b.equation_type = "fluid";
    b.geometry_tangent_policy_explicit = true;
    EXPECT_EQ(activeCutVolumeRequestPolicyKey({a}), activeCutVolumeRequestPolicyKey({b}));
    b.active_side = LevelSetActiveSide::Positive;
    EXPECT_NE(activeCutVolumeRequestPolicyKey({a}), activeCutVolumeRequestPolicyKey({b}));
    EXPECT_FALSE(hasHighOrderGeneratedInterfaceGeometry({a}));
}

TEST(LevelSetGeometry, ScalarLevelSetFieldIsResolvedWithTheOwnersMessages)
{
    GeometrySystem s;
    EXPECT_EQ(resolveScalarLevelSetField(*s.system, "phi", "Owner: unknown field '", "Owner"), s.phi);
    try {
        (void)resolveScalarLevelSetField(*s.system, "psi", "Owner: unknown field '", "Owner");
        FAIL() << "unknown field accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "Owner: unknown field 'psi'");
    }
    try {
        (void)resolveScalarLevelSetField(*s.system, "u", "Owner: unknown field '", "Owner");
        FAIL() << "vector field accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "Owner: level-set field 'u' must be scalar");
    }
}

TEST(LevelSetGeometry, GeneratedActiveBoundaryMarkerMustAgreeOverItsOwners)
{
    interfaces::GeneratedActiveBoundaryMarkerKey first{};
    first.source = interfaces::LevelSetInterfaceSource::fromField(FieldId{1});
    first.domain_id = "drop";
    first.interface_marker = 5;
    first.boundary_marker = 2;
    auto second = first;
    second.side = geometry::CutIntegrationSide::Positive;
    const std::vector<std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>> agree{
        std::nullopt, first, first};
    std::vector<std::size_t> asked;
    const auto marker = resolveGeneratedActiveBoundaryMarker(
        2, agree.size(), [&](std::size_t i) { asked.push_back(i); return agree[i]; }, "Owner");
    ASSERT_TRUE(marker.has_value());
    EXPECT_EQ(*marker, interfaces::stableGeneratedActiveBoundaryMarker(first));
    EXPECT_EQ(asked, (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_FALSE(resolveGeneratedActiveBoundaryMarker(
                     2, 1, [](std::size_t) { return std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>{}; },
                     "Owner")
                     .has_value());

    const std::vector<std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>> ambiguous{first, second};
    try {
        (void)resolveGeneratedActiveBoundaryMarker(
            2, ambiguous.size(), [&](std::size_t i) { return ambiguous[i]; }, "Owner");
        FAIL() << "ambiguous owners accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "Owner: physical boundary marker 2 is covered by multiple ambiguous unfitted active "
                  "domains; select one explicit active-domain owner");
    }
}

TEST(LevelSetGeometry, WettedBoundaryResolverRegistersTheGeneratedMarkers)
{
    GeometrySystem s;
    const auto resolve = wettedBoundaryResolver(
        *s.system, [](int marker) -> std::optional<int> {
            return marker == 1 ? std::optional<int>(3000007) : std::nullopt;
        });
    EXPECT_FALSE(resolve(2).has_value());
    EXPECT_FALSE(s.system->isGeneratedEmbeddedInterfaceMarkerRegistered(3000007));
    const auto generated = resolve(1);
    ASSERT_TRUE(generated.has_value());
    EXPECT_EQ(*generated, 3000007);
    EXPECT_TRUE(s.system->isGeneratedEmbeddedInterfaceMarkerRegistered(3000007));
}

TEST(LevelSetGeometry, ActiveBoundaryResolverOwnsItsInputsUntilExplicitRegistration)
{
    std::function<std::optional<int>(int)> resolve;
    int expected_marker = -1;
    {
        GeometrySystem source;
        const auto geometry = LevelSetGeometry::bind(*source.system,
            {.level_set_field = source.phi, .domain_id = "owned", .interface_marker = 17});
        std::vector<CutSide> owners{geometry.side(CutVolumeSide::Negative)};
        expected_marker = owners.front().boundary(4).generatedActiveBoundaryMarker();
        resolve = generatedActiveBoundaryResolver(std::move(owners), "Owner", "negative marker");
    }
    auto moved = std::move(resolve);
    EXPECT_EQ(moved(4), expected_marker);
    try {
        (void)moved(-1);
        FAIL() << "negative marker was accepted with an owner";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "negative marker");
    }

    GeometrySystem consumer;
    const auto wetted = wettedBoundaryResolver(*consumer.system, std::move(moved));
    EXPECT_FALSE(consumer.system->isGeneratedEmbeddedInterfaceMarkerRegistered(expected_marker));
    EXPECT_EQ(wetted(4), expected_marker);
    EXPECT_TRUE(consumer.system->isGeneratedEmbeddedInterfaceMarkerRegistered(expected_marker));
}

TEST(LevelSetGeometry, ActiveBoundaryResolverAllowsEmptyAndRepeatedOwnersButRejectsAmbiguity)
{
    const auto empty = generatedActiveBoundaryResolver({}, "Owner", "negative marker");
    EXPECT_FALSE(empty(-1));
    EXPECT_FALSE(empty(4));

    GeometrySystem source;
    const auto geometry = LevelSetGeometry::bind(*source.system,
        {.level_set_field = source.phi, .domain_id = "shared", .interface_marker = 17});
    const auto negative = geometry.side(CutVolumeSide::Negative);
    const auto repeated = generatedActiveBoundaryResolver({negative, negative}, "Owner", "negative marker");
    EXPECT_EQ(repeated(4), negative.boundary(4).generatedActiveBoundaryMarker());
    const auto ambiguous = generatedActiveBoundaryResolver(
        {negative, geometry.side(CutVolumeSide::Positive)}, "Owner", "negative marker");
    try {
        (void)ambiguous(4);
        FAIL() << "ambiguous owners were accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
            "Owner: physical boundary marker 4 is covered by multiple ambiguous unfitted active "
            "domains; select one explicit active-domain owner");
    }
}

} // namespace
} // namespace svmp::FE::level_set::test
